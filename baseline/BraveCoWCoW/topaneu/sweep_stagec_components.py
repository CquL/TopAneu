#!/usr/bin/env python3
"""P2: regenerate components from the strict-OOF probability cache.

No neural network inference is performed. Candidate masks are written under
one directory per threshold/minimum-size rule so the official evaluator can be
run on each rule independently.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch
from scipy import ndimage

from topaneu.stagec.train_stagec import StageCClassifier, extract_component_features


DEFAULT_ROOT = Path("/data/cyf/shared_data/TopAneu/BraveCoWCoW_predroi_160")
DATASET = "Dataset505_TopAneuBraveCoWCoWPredROI"
GT_STRUCTURE = ndimage.generate_binary_structure(3, 2)


def read(path, dtype=None):
    a = sitk.GetArrayFromImage(sitk.ReadImage(str(path)))
    return a.astype(dtype, copy=False) if dtype is not None else a


def physical_scale(meta, shape):
    tr = meta["roi_transform"]
    crop = np.asarray(tr["crop_upper_zyx_exclusive"], float) - np.asarray(tr["crop_lower_zyx"], float)
    source_spacing_zyx = np.asarray(tr["source_spacing_xyz"], float)[::-1]
    return source_spacing_zyx * crop / np.asarray(shape, float)


def load_stagec(root, fold, device):
    state = torch.load(root / "stagec_results" / f"fold_{fold}" / "checkpoint_best_stagec.pth",
                       map_location=device, weights_only=False)
    model = StageCClassifier().to(device)
    model.load_state_dict(state["model"], strict=True)
    model.eval()
    return model, state["mean"].astype(np.float32), np.maximum(state["std"].astype(np.float32), 1e-6), torch.as_tensor(state["seen_location_mask"], dtype=torch.bool, device=device)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--cache", type=Path, default=None)
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--thresholds", nargs="+", type=float, default=[.05, .10, .15, .20, .30, .40, .50])
    ap.add_argument("--min-voxels", nargs="+", type=int, default=[1, 3, 5, 10])
    ap.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--limit", type=int, default=0, help="cases per fold for smoke testing")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    root = args.data_root.resolve()
    cache = (args.cache or root / "stagec_pred_oof_v2").resolve()
    output = (args.output or root / "stagec_pred_oof_v2" / "evaluations").resolve()
    raw = root / "nnXNet_raw" / DATASET
    metadata = cache / "metadata"
    probs = cache / "probabilities"
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")

    splits = json.loads((metadata / "splits_final.json").read_text())
    for fold in args.folds:
        model, mean, std, seen = load_stagec(root, fold, device)
        rows_by_rule = {(float(t), int(m)): [] for t in args.thresholds for m in args.min_voxels}
        cases = list(splits[fold]["val"])
        if args.limit:
            cases = cases[:args.limit]
        for index, case in enumerate(cases, 1):
            with np.load(probs / f"fold_{fold}" / f"{case}.npz") as z:
                aneurysm = z["aneurysm_probability"].astype(np.float32)
                vessel = z["vessel_probability"].argmax(axis=0).astype(np.uint8)
            image = read(raw / "imagesTr" / f"{case}_0000.nii.gz", np.float32)
            meta = json.loads((metadata / f"fold_{fold}" / f"{case}.json").read_text())
            scale = physical_scale(meta, aneurysm.shape)
            spacing_volume = float(np.prod(scale))
            distance_map = ndimage.distance_transform_edt(~np.isin(vessel, np.arange(1, 37)), sampling=scale)
            for threshold in args.thresholds:
                cc, n = ndimage.label(aneurysm >= threshold, GT_STRUCTURE)
                for minimum in args.min_voxels:
                    rule = (float(threshold), int(minimum))
                    result = np.zeros(aneurysm.shape, dtype=np.uint8)
                    for component_id in range(1, n + 1):
                        component = cc == component_id
                        voxels = int(component.sum())
                        if voxels < minimum:
                            continue
                        feature = extract_component_features(image, component, vessel, case)
                        x = torch.as_tensor(((feature - mean) / std)[None], dtype=torch.float32, device=device)
                        with torch.inference_mode():
                            logits = model(x)["location"]
                            logits[:, ~seen] = -1e9
                            p = torch.softmax(logits, dim=1)[0]
                        order = torch.argsort(p, descending=True)
                        top1 = int(order[0]) + 1
                        top2 = int(order[1]) + 1
                        result[component] = top1
                        near = ndimage.binary_dilation(component, iterations=3)
                        near_vessel = np.isin(vessel[near], np.arange(1, 37))
                        rows_by_rule[rule].append({
                            "case": case, "fold": fold, "component_id": component_id,
                            "voxel_count": voxels, "volume_mm3": voxels * spacing_volume,
                            "equivalent_diameter_mm": (6.0 * voxels * spacing_volume / np.pi) ** (1.0 / 3.0),
                            "centroid_z_mm": float(np.argwhere(component).mean(axis=0)[0] * scale[0]),
                            "centroid_y_mm": float(np.argwhere(component).mean(axis=0)[1] * scale[1]),
                            "centroid_x_mm": float(np.argwhere(component).mean(axis=0)[2] * scale[2]),
                            "nearest_vessel_distance_mm": float(distance_map[component].min()) if np.any(component) else None,
                            "vessel_contact_voxels": int(near_vessel.sum()),
                            "stagec_top1": top1, "stagec_top2": top2,
                            "stagec_top1_probability": float(p[order[0]]),
                            "stagec_top2_probability": float(p[order[1]]),
                            "stagec_margin": float(p[order[0]] - p[order[1]]),
                            "stagec_entropy": float(-(p[p > 0] * p[p > 0].log()).sum()),
                        })
                    rule_dir = output / f"threshold_{threshold:g}_minvox_{minimum}" / f"fold_{fold}" / "validation_location_masks_component"
                    rule_dir.mkdir(parents=True, exist_ok=True)
                    ref = sitk.ReadImage(str(raw / "imagesTr" / f"{case}_0000.nii.gz"))
                    out_img = sitk.GetImageFromArray(result); out_img.CopyInformation(ref)
                    sitk.WriteImage(out_img, str(rule_dir / f"{case}.nii.gz"), True)
            print(f"fold={fold} case={case} saved {index}/{len(cases)}", flush=True)
        for rule, rows in rows_by_rule.items():
            rule_dir = output / f"threshold_{rule[0]:g}_minvox_{rule[1]}"
            with (rule_dir / "components.csv").open("w", newline="") as handle:
                fields = list(rows[0]) if rows else ["case", "fold"]
                writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
        del model
        if device.type == "cuda": torch.cuda.empty_cache()
    manifest = {"cache": str(cache), "output": str(output), "folds": args.folds, "limit": args.limit,
                "thresholds": args.thresholds, "min_voxels": args.min_voxels,
                "network_inference": False, "physical_scale": "source spacing × crop extent / 160",
                "stagec": "best checkpoint per fold; top-1 location assigned per component"}
    (output / "p2_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
