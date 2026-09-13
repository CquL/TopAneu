#!/usr/bin/env python3
"""Build a strict five-fold OOF dense probability cache.

Each case is inferred only with the model belonging to its validation fold.
The cache is deliberately independent from ``stagec_pred_oof`` and from the
older threshold scan.  Arrays are stored in the preprocessed 160^3 ROI
coordinate system as float16, while the per-case JSON records the transform
needed to restore them to the source image space.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F
from torch._dynamo import OptimizedModule

DATASET = "Dataset505_TopAneuBraveCoWCoWPredROI"
PLAN = "TopAneuBraveCoWCoWPredROIPlansV2_160"
CONFIGURATION = "3d_fullres"
VESSEL_EXPERIMENT = (
    "nnXNetTrainer_TopAneuVesselOnly__"
    f"{PLAN}__{CONFIGURATION}"
)
ANEURYSM_EXPERIMENT = (
    "nnXNetTrainer_TopAneuAneurysmOnly__"
    f"{PLAN}__{CONFIGURATION}"
)
ANEURYSM_V2_EXPERIMENT = (
    "nnXNetTrainer_TopAneuAneurysmV2__"
    f"{PLAN}__{CONFIGURATION}"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit(repo: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def load_transform(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def pad_to_patch(data: np.ndarray, patch: np.ndarray) -> tuple[torch.Tensor, np.ndarray, np.ndarray]:
    spatial = np.asarray(data.shape[1:], dtype=int)
    if np.any(spatial > patch):
        raise ValueError(f"preprocessed ROI shape {spatial.tolist()} exceeds patch {patch.tolist()}")
    before = (patch - spatial) // 2
    after = patch - spatial - before
    padding: list[int] = []
    # F.pad consumes dimensions in reverse order (x, y, z).
    for lower, upper in zip(before[::-1], after[::-1]):
        padding.extend([int(lower), int(upper)])
    # nnXNet may return a read-only view from a compressed NumPy cache. Make a
    # writable contiguous copy before handing it to torch.
    tensor = F.pad(torch.from_numpy(np.ascontiguousarray(data[None])), padding)
    return tensor, before, after


def unpad(logits: torch.Tensor, before: np.ndarray, spatial: np.ndarray) -> np.ndarray:
    crop = tuple(slice(int(v), int(v + n)) for v, n in zip(before, spatial))
    return logits[..., crop[0], crop[1], crop[2]]


def instantiate(trainer_class, plans: dict, dataset_json: dict, fold: int, device: torch.device, runtime: Path):
    # Trainer construction normally writes logs below nnXNet_results. Point
    # those transient files into the experiment cache instead of touching old
    # training result directories.
    import nnxnet.training.nnXNetTrainer.nnXNetTrainer as base_module

    previous = base_module.nnXNet_results
    base_module.nnXNet_results = str(runtime)
    try:
        trainer = trainer_class(plans, CONFIGURATION, fold, dataset_json, device=device)
    finally:
        base_module.nnXNet_results = previous
    trainer.initialize()
    trainer.set_deep_supervision_enabled(False)
    trainer.network.eval()
    return trainer


def load_weights(trainer, checkpoint: Path) -> dict[str, Any]:
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    network = trainer.network
    target = network._orig_mod if isinstance(network, OptimizedModule) else network
    weights = state.get("network_weights") or state.get("state_dict") or state
    target.load_state_dict(weights, strict=True)
    trainer.network.eval()
    return state


def infer_case(aneurysm_trainer, vessel_trainer, dataset_val, case: str, patch: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    data, _, properties = dataset_val.load_case(case)
    spatial = np.asarray(data.shape[1:], dtype=int)
    tensor, before, _ = pad_to_patch(data, patch)
    tensor = tensor.to(aneurysm_trainer.device, non_blocking=True)
    with torch.inference_mode():
        context = torch.autocast("cuda", enabled=True) if aneurysm_trainer.device.type == "cuda" else torch.autocast("cpu", enabled=False)
        with context:
            aneurysm_logits = aneurysm_trainer.network(tensor, aneurysm_only=True)
        aneurysm = torch.softmax(aneurysm_logits.float(), dim=1)[0, 1]
        aneurysm = unpad(aneurysm[None], before, spatial)[0].cpu().numpy().astype(np.float16, copy=False)
        with torch.autocast("cuda", enabled=vessel_trainer.device.type == "cuda"):
            vessel_logits = vessel_trainer.network(tensor, vessel_only=True)
        vessel = torch.softmax(vessel_logits.float(), dim=1)[0]
        vessel = unpad(vessel, before, spatial).cpu().numpy().astype(np.float16, copy=False)
    props = jsonable(properties)
    del tensor, vessel_logits, aneurysm_logits
    return aneurysm, vessel, props


def checkpoint_spec(data_root: Path, fold: int, args) -> dict[str, Any]:
    results = data_root / "nnXNet_results" / DATASET
    vessel = results / VESSEL_EXPERIMENT / f"fold_{fold}" / "checkpoint_best.pth"
    if fold == 0 and args.aneurysm_fold0 == "v2":
        aneurysm = results / ANEURYSM_V2_EXPERIMENT / "fold_0" / "checkpoint_best.pth"
        aneurysm_kind = "v2_best"
    else:
        aneurysm = results / ANEURYSM_EXPERIMENT / f"fold_{fold}" / f"checkpoint_{args.aneurysm_checkpoint}.pth"
        aneurysm_kind = args.aneurysm_checkpoint
    return {
        "fold": fold,
        "vessel": str(vessel),
        "vessel_sha256": sha256_file(vessel),
        "vessel_bytes": vessel.stat().st_size,
        "aneurysm": str(aneurysm),
        "aneurysm_sha256": sha256_file(aneurysm),
        "aneurysm_bytes": aneurysm.stat().st_size,
        "aneurysm_kind": aneurysm_kind,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path(os.environ.get("TOPANEU_BRAVECOWCOW_DATA", "/data/cyf/shared_data/TopAneu/BraveCoWCoW_predroi_160")))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--folds", nargs="+", type=int, default=list(range(5)))
    parser.add_argument("--limit", type=int, default=0, help="cases per fold for smoke testing")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--aneurysm-fold0", choices=["v2", "old"], default="v2")
    parser.add_argument("--aneurysm-checkpoint", choices=["best", "final"], default="final")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    data_root = args.data_root.resolve()
    output = (args.output or data_root / "stagec_pred_oof_v2").resolve()
    output.mkdir(parents=True, exist_ok=True)
    probabilities = output / "probabilities"
    metadata = output / "metadata"
    runtime = output / "_runtime"
    probabilities.mkdir(exist_ok=True)
    metadata.mkdir(exist_ok=True)
    runtime.mkdir(exist_ok=True)

    pre = data_root / "nnXNet_preprocessed" / DATASET
    raw = data_root / "nnXNet_raw" / DATASET
    transform_dir = data_root / "metadata" / "roi_transforms"
    plans = json.loads((pre / f"{PLAN}.json").read_text())
    dataset_json = json.loads((pre / "dataset.json").read_text())
    splits = json.loads((pre / "splits_final.json").read_text())
    patch = np.asarray(plans["configurations"][CONFIGURATION]["patch_size"], dtype=int)

    repo = Path(__file__).resolve().parents[1]
    manifest = {
        "experiment": "TopAneu P1 strict OOF dense probability cache",
        "created_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        "repo": str(repo),
        "git_commit": git_commit(repo),
        "dataset": DATASET,
        "plans": PLAN,
        "configuration": CONFIGURATION,
        "network_patch_zyx": patch.tolist(),
        "storage": {"aneurysm": "float16 probability channel 1", "vessel": "float16 softmax channels 0..36"},
        "folds": args.folds,
        "limit_per_fold": args.limit,
        "aneurysm_fold0_selection": args.aneurysm_fold0,
        "aneurysm_checkpoint_selection": args.aneurysm_checkpoint,
        "strict_oof": True,
        "thresholds_not_applied": True,
        "checkpoint_manifest": [],
    }
    for fold in args.folds:
        manifest["checkpoint_manifest"].append(checkpoint_spec(data_root, fold, args))
    existing_manifest = metadata / "experiment_manifest.json"
    if existing_manifest.exists() and not args.overwrite:
        old = json.loads(existing_manifest.read_text())
        # The timestamp is deliberately allowed to differ on a resumed run.
        old_compare = dict(old)
        new_compare = dict(manifest)
        old_compare.pop("created_utc", None)
        new_compare.pop("created_utc", None)
        if old_compare != new_compare:
            raise RuntimeError(f"Existing cache provenance differs: {existing_manifest}; choose another output or --overwrite")
        manifest["created_utc"] = old.get("created_utc", manifest["created_utc"])
    existing_manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    (metadata / "splits_final.json").write_text(json.dumps(splits, indent=2) + "\n")
    (metadata / "plans.json").write_text(json.dumps(plans, indent=2) + "\n")
    (metadata / "fold_assignment.json").write_text(
        json.dumps({str(fold): list(splits[fold]["val"]) for fold in args.folds}, indent=2) + "\n"
    )
    (metadata / "checkpoint_manifest.json").write_text(
        json.dumps(manifest["checkpoint_manifest"], indent=2) + "\n"
    )

    from nnxnet.training.nnXNetTrainer.nnXNetTrainer_TopAneuVesselOnly import nnXNetTrainer_TopAneuVesselOnly
    from nnxnet.training.nnXNetTrainer.nnXNetTrainer_TopAneuAneurysmOnly import nnXNetTrainer_TopAneuAneurysmOnly
    from nnxnet.training.nnXNetTrainer.nnXNetTrainer_TopAneuAneurysmV2 import nnXNetTrainer_TopAneuAneurysmV2

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    case_rows: list[dict[str, Any]] = []
    for fold in args.folds:
        spec = manifest["checkpoint_manifest"][args.folds.index(fold)]
        aneurysm_class = nnXNetTrainer_TopAneuAneurysmV2 if spec["aneurysm_kind"] == "v2_best" else nnXNetTrainer_TopAneuAneurysmOnly
        vessel_trainer = instantiate(nnXNetTrainer_TopAneuVesselOnly, plans, dataset_json, fold, device, runtime / f"fold_{fold}_vessel")
        aneurysm_trainer = instantiate(aneurysm_class, plans, dataset_json, fold, device, runtime / f"fold_{fold}_aneurysm")
        vessel_state = load_weights(vessel_trainer, Path(spec["vessel"]))
        aneurysm_state = load_weights(aneurysm_trainer, Path(spec["aneurysm"]))
        dataset_val = vessel_trainer.get_tr_and_val_datasets()[1]
        keys = list(splits[fold]["val"])
        if args.limit:
            keys = keys[:args.limit]
        fold_prob = probabilities / f"fold_{fold}"
        fold_meta = metadata / f"fold_{fold}"
        fold_prob.mkdir(exist_ok=True)
        fold_meta.mkdir(exist_ok=True)
        for index, case in enumerate(keys, 1):
            npz_path = fold_prob / f"{case}.npz"
            json_path = fold_meta / f"{case}.json"
            if npz_path.exists() and json_path.exists() and not args.overwrite:
                print(f"fold={fold} case={case} cached {index}/{len(keys)}", flush=True)
                case_rows.append(json.loads(json_path.read_text()))
                continue
            aneurysm, vessel, properties = infer_case(aneurysm_trainer, vessel_trainer, dataset_val, case, patch)
            # Atomic replacement prevents an interrupted GPU job from leaving
            # a truncated .npz that a resumed run would mistake for a cache.
            temporary_npz = npz_path.with_suffix(".partial.npz")
            np.savez_compressed(
                temporary_npz,
                aneurysm_probability=aneurysm,
                vessel_probability=vessel,
            )
            temporary_npz.replace(npz_path)
            transform_path = transform_dir / f"{case}.json"
            transform = load_transform(transform_path)
            case_meta = {
                "case": case, "fold": fold, "checkpoint": {"vessel": spec["vessel"], "aneurysm": spec["aneurysm"]},
                "checkpoint_sha256": {"vessel": spec["vessel_sha256"], "aneurysm": spec["aneurysm_sha256"]},
                "epoch": {"vessel": vessel_state.get("current_epoch"), "aneurysm": aneurysm_state.get("current_epoch")},
                "probability_file": str(npz_path.relative_to(output)),
                "array_shape_zyx": list(aneurysm.shape), "array_dtype": "float16",
                "network_spacing_zyx": [1.0, 1.0, 1.0], "preprocessed_properties": properties,
                "roi_transform": transform,
                "source_spacing_xyz": transform.get("source_spacing_xyz"),
                "source_origin_xyz": transform.get("source_origin_xyz"),
                "source_direction": transform.get("source_direction"),
            }
            temporary_json = json_path.with_suffix(".partial.json")
            temporary_json.write_text(json.dumps(case_meta, indent=2) + "\n")
            temporary_json.replace(json_path)
            case_rows.append(case_meta)
            print(f"fold={fold} case={case} saved {index}/{len(keys)}", flush=True)
        del vessel_trainer, aneurysm_trainer
        torch.cuda.empty_cache()

    (metadata / "cases.jsonl").write_text("".join(json.dumps(row) + "\n" for row in case_rows))
    # This map is convenient for P2 without opening one JSON per case.
    (metadata / "spatial_transforms.json").write_text(
        json.dumps({row["case"]: row["roi_transform"] for row in case_rows}, indent=2) + "\n"
    )
    summary = {"cases_written": len(case_rows), "requested_folds": args.folds, "limit": args.limit, "output": str(output), "complete_oof": len(case_rows) == sum(len(splits[f]["val"]) for f in args.folds)}
    (metadata / "cache_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
