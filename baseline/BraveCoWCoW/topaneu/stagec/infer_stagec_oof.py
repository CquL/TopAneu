#!/usr/bin/env python3
"""Assign 52 TopAneu locations to OOF aneurysm components with Stage-C."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch
from scipy import ndimage

from topaneu.stagec.train_stagec import StageCClassifier, extract_component_features

DATA_ROOT = Path("/data/cyf/shared_data/TopAneu/BraveCoWCoW_predroi_160")
DATASET = "Dataset505_TopAneuBraveCoWCoWPredROI"
OLD_EXPERIMENT = "nnXNetTrainer_TopAneuAneurysmOnly__TopAneuBraveCoWCoWPredROIPlansV2_160__3d_fullres"
V2_EXPERIMENT = "nnXNetTrainer_TopAneuAneurysmV2__TopAneuBraveCoWCoWPredROIPlansV2_160__3d_fullres"
VESSEL_EXPERIMENT = "nnXNetTrainer_TopAneuVesselOnly__TopAneuBraveCoWCoWPredROIPlansV2_160__3d_fullres"


def read(path, dtype):
    return sitk.GetArrayFromImage(sitk.ReadImage(str(path))).astype(dtype, copy=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--stagec-results", type=Path, default=DATA_ROOT / "stagec_results")
    parser.add_argument("--output", type=Path, default=DATA_ROOT / "stagec_pred_oof")
    parser.add_argument("--min-component-voxels", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    raw = args.data_root / "nnXNet_raw" / DATASET
    splits = json.loads((args.data_root / "nnXNet_preprocessed" / DATASET / "splits_final.json").read_text())
    results = args.data_root / "nnXNet_results" / DATASET
    old, v2, vessel = results / OLD_EXPERIMENT, results / V2_EXPERIMENT, results / VESSEL_EXPERIMENT
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    if args.min_component_voxels < 1:
        raise ValueError("--min-component-voxels must be at least 1")

    all_cases = []
    for fold, split in enumerate(splits):
        checkpoint = args.stagec_results / f"fold_{fold}" / "checkpoint_best_stagec.pth"
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        state = torch.load(checkpoint, map_location=device, weights_only=False)
        model = StageCClassifier().to(device)
        model.load_state_dict(state["model"], strict=True)
        model.eval()
        mean, std = state["mean"].astype(np.float32), np.maximum(state["std"].astype(np.float32), 1e-6)
        seen = torch.as_tensor(state["seen_location_mask"], dtype=torch.bool, device=device)
        aneurysm_dir = (v2 if fold == 0 else old) / f"fold_{fold}" / "validation_aneurysm_masks"
        vessel_dir = vessel / f"fold_{fold}" / "validation_vessel_masks"
        out_dir = args.output / f"fold_{fold}" / "validation_location_masks_stagec_pred"
        out_dir.mkdir(parents=True, exist_ok=True)
        rows = []
        for case in split["val"]:
            image_path = raw / "imagesTr" / f"{case}_0000.nii.gz"
            aneurysm_path, vessel_path = aneurysm_dir / f"{case}.nii.gz", vessel_dir / f"{case}.nii.gz"
            if not aneurysm_path.is_file() or not vessel_path.is_file():
                raise FileNotFoundError(f"{case}: missing OOF mask")
            image, aneurysm, vessel_mask = read(image_path, np.float32), read(aneurysm_path, np.uint8) > 0, read(vessel_path, np.uint8)
            if not (image.shape == aneurysm.shape == vessel_mask.shape):
                raise ValueError(f"{case}: image/mask shape mismatch")
            cc, n = ndimage.label(aneurysm, ndimage.generate_binary_structure(3, 2))
            prediction = np.zeros(aneurysm.shape, dtype=np.uint8)
            for component_id in range(1, n + 1):
                component = cc == component_id
                size = int(component.sum())
                if size < args.min_component_voxels:
                    continue
                features = extract_component_features(image, component, vessel_mask, case)
                x = torch.as_tensor(((features - mean) / std)[None], device=device)
                with torch.inference_mode():
                    logits = model(x)["location"]
                    logits[:, ~seen] = -1e9
                    probabilities = torch.softmax(logits, dim=1)[0]
                    label, confidence = int(probabilities.argmax().item()) + 1, float(probabilities.max().item())
                prediction[component] = label
                rows.append([case, component_id, size, label, confidence])
            reference = sitk.ReadImage(str(image_path))
            output = sitk.GetImageFromArray(prediction)
            output.CopyInformation(reference)
            sitk.WriteImage(output, str(out_dir / f"{case}.nii.gz"), True)
            all_cases.append(case)
            print(f"fold={fold} case={case} components={n}", flush=True)
        with (args.output / f"fold_{fold}" / "components.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["case", "component_id", "size_voxels", "location_id", "confidence"])
            writer.writerows(rows)
    if len(all_cases) != 416 or len(set(all_cases)) != 416:
        raise RuntimeError(f"Expected 416 unique OOF cases, got {len(all_cases)}/{len(set(all_cases))}")
    (args.output / "manifest.json").write_text(json.dumps({
        "cases": len(all_cases), "fold0_aneurysm": "AneurysmV2 checkpoint_best epoch 50 output",
        "fold1to4_aneurysm": "AneurysmOnly checkpoint_final output", "vessel": "VesselOnly OOF output",
        "stagec": "checkpoint_best_stagec per fold", "min_component_voxels": args.min_component_voxels,
        "note": "No false-positive rejection head; every predicted component receives a location."}, indent=2))
    print(f"saved {len(all_cases)} OOF 52-class masks to {args.output}")


if __name__ == "__main__":
    main()
