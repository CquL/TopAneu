#!/usr/bin/env python3
"""Create a reproducible, locked manifest for a TopAneu experiment.

The manifest is deliberately independent of model execution.  It records the
exact evaluator source, fold assignment, preprocessing/restoration protocol,
and checkpoint selection used by an experiment.  Existing manifests are not
overwritten unless ``--force`` is supplied.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_DEFAULT = Path(__file__).resolve().parents[3]
DATA_DEFAULT = Path(os.environ.get(
    "TOPANEU_BRAVECOWCOW_DATA",
    "/data/cyf/shared_data/TopAneu/BraveCoWCoW_predroi_160",
))
DATASET_DEFAULT = "Dataset505_TopAneuBraveCoWCoWPredROI"
EVALUATOR_DEFAULT = REPO_DEFAULT / "TopAneu-26-main/eval/task2/evaluate.py"


def _json_default(value: Any):
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(root: Path, *args: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), *args],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def git_info(root: Path) -> dict[str, Any]:
    resolved = root.resolve()
    return {
        "root": str(resolved),
        "sha": git(resolved, "rev-parse", "HEAD"),
        "branch": git(resolved, "branch", "--show-current"),
        "dirty": bool(git(resolved, "status", "--porcelain")),
        "remote_origin": git(resolved, "remote", "get-url", "origin"),
    }


def file_info(path: Path, hash_file: bool = False) -> dict[str, Any]:
    path = path.expanduser().resolve()
    info: dict[str, Any] = {"path": str(path), "exists": path.is_file()}
    if path.is_file():
        stat = path.stat()
        info.update({
            "size_bytes": stat.st_size,
            "mtime_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
        })
        if hash_file:
            info["sha256"] = sha256_file(path)
    return info


def checkpoint_info(path: Path, hash_checkpoints: bool) -> dict[str, Any]:
    return file_info(path, hash_file=hash_checkpoints)


def locate(path: Path, *, required: bool, label: str) -> Path | None:
    path = path.expanduser().resolve()
    if required and not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    return path if path.is_file() else None


def read_splits(path: Path) -> tuple[list[dict[str, Any]], dict[str, int]]:
    data = json.loads(path.read_text())
    if not isinstance(data, list) or len(data) != 5:
        raise ValueError(f"Expected a list of 5 folds in {path}")
    folds: list[dict[str, Any]] = []
    case_to_fold: dict[str, int] = {}
    for fold, split in enumerate(data):
        train = list(split.get("train", []))
        val = list(split.get("val", []))
        if set(train) & set(val):
            raise ValueError(f"fold {fold} has train/val overlap")
        for case in val:
            if case in case_to_fold:
                raise ValueError(f"case appears in multiple validation folds: {case}")
            case_to_fold[case] = fold
        folds.append({
            "fold": fold,
            "train_count": len(train),
            "validation_count": len(val),
            "train_cases": train,
            "validation_cases": val,
        })
    if sum(x["validation_count"] for x in folds) != len(case_to_fold):
        raise ValueError("duplicate validation cases in split file")
    return folds, case_to_fold


def make_stage_checkpoints(data_root: Path, dataset: str, stage1: Path | None, hash_checkpoints: bool) -> dict[str, Any]:
    results = data_root / "nnXNet_results" / dataset
    vessel_exp = results / (
        "nnXNetTrainer_TopAneuVesselOnly__"
        "TopAneuBraveCoWCoWPredROIPlansV2_160__3d_fullres"
    )
    aneurysm_old = results / (
        "nnXNetTrainer_TopAneuAneurysmOnly__"
        "TopAneuBraveCoWCoWPredROIPlansV2_160__3d_fullres"
    )
    aneurysm_v2 = results / (
        "nnXNetTrainer_TopAneuAneurysmV2__"
        "TopAneuBraveCoWCoWPredROIPlansV2_160__3d_fullres"
    )
    stagec = data_root / "stagec_results"

    def per_fold(exp: Path, names: list[str]) -> dict[str, Any]:
        return {
            str(fold): {
                name: checkpoint_info(exp / f"fold_{fold}" / filename, hash_checkpoints)
                for name, filename in names
            }
            for fold in range(5)
        }

    # OOF selection used by stagec/infer_stagec_oof.py: V2 best for fold 0,
    # old AneurysmOnly final for folds 1-4, VesselOnly masks for every fold.
    aneurysm = per_fold(aneurysm_old, [("final", "checkpoint_final.pth")])
    aneurysm["0"] = {"v2_best": checkpoint_info(
        aneurysm_v2 / "fold_0/checkpoint_best.pth", hash_checkpoints,
    )}
    return {
        "stage1_roi": {
            "source": "publisher RSNA pretrained 2D vessel-box ROI model",
            "checkpoint": file_info(stage1, hash_file=hash_checkpoints) if stage1 else None,
            "selection": "fold_0/checkpoint_final.pth; fixed/non-OOF publisher weight",
        },
        "stage2_vessel": {
            "experiment": str(vessel_exp),
            "checkpoint_selection": "checkpoint_best.pth per fold",
            "folds": per_fold(vessel_exp, [("best", "checkpoint_best.pth")]),
        },
        "stage2_aneurysm": {
            "experiment_old": str(aneurysm_old),
            "experiment_v2": str(aneurysm_v2),
            "checkpoint_selection": "fold0 V2 best; folds1-4 AneurysmOnly final",
            "folds": aneurysm,
        },
        "stagec": {
            "experiment": str(stagec),
            "checkpoint_selection": "checkpoint_best_stagec.pth per fold",
            "folds": {
                str(fold): {"best": checkpoint_info(
                    stagec / f"fold_{fold}/checkpoint_best_stagec.pth", hash_checkpoints,
                )}
                for fold in range(5)
            },
        },
    }


def build_manifest(args: argparse.Namespace) -> dict[str, Any]:
    repo = args.repo_root.resolve()
    data_root = args.data_root.resolve()
    dataset_dir = data_root / "nnXNet_preprocessed" / args.dataset
    split_path = locate(dataset_dir / "splits_final.json", required=True, label="split file")
    plans_path = locate(dataset_dir / args.plans_name, required=True, label="plans file")
    raw_dataset = data_root / "nnXNet_raw" / args.dataset
    raw_dataset_json = locate(raw_dataset / "dataset.json", required=True, label="raw dataset.json")
    preprocessed_dataset_json = locate(dataset_dir / "dataset.json", required=True, label="preprocessed dataset.json")
    evaluator = locate(args.evaluator.resolve(), required=True, label="official evaluator")
    requirements = evaluator.parent / "requirements.txt"
    folds, case_to_fold = read_splits(split_path)

    stage1 = args.stage1_checkpoint.expanduser().resolve() if args.stage1_checkpoint else None
    stage1_info = file_info(stage1, hash_file=args.hash_checkpoints) if stage1 else None
    evaluator_info = file_info(evaluator, hash_file=True)
    evaluator_git_root = Path(git(evaluator.parent, "rev-parse", "--show-toplevel") or evaluator.parent)

    out = args.output.expanduser().resolve()
    if out.exists() and not args.force:
        raise FileExistsError(f"Manifest already exists; use --force to replace: {out}")

    manifest: dict[str, Any] = {
        "manifest_schema": "topaneu.experiment_manifest.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_name": args.experiment_name,
        "purpose": args.purpose,
        "repository": git_info(repo),
        "official_evaluator": {
            "name": "TopAneu-26 Task 2",
            "source_file": evaluator_info,
            "requirements_file": file_info(requirements, hash_file=True) if requirements.is_file() else None,
            "git": git_info(evaluator_git_root),
            "implementation": "evaluation_function + evaluation_aggregation + evaluation_average",
            "metrics": ["PRECISION", "RECALL", "MCC", "DICE", "HD95", "VOLSIM"],
            "notes": {
                "class_count": 52,
                "empty_class_values": "DICE/HD95/VOLSIM are zero when both prediction and GT are empty; only non-TN classes contribute to these averages",
                "hd95": "normalized by image-array diagonal; lower is better",
            },
        },
        "data": {
            "root": str(data_root),
            "dataset": args.dataset,
            "raw_dataset": str(raw_dataset),
            "preprocessed_dataset": str(dataset_dir),
            "files": {
                "raw_dataset_json": file_info(raw_dataset_json, hash_file=True),
                "preprocessed_dataset_json": file_info(preprocessed_dataset_json, hash_file=True),
                "plans": file_info(plans_path, hash_file=True),
                "splits_final": file_info(split_path, hash_file=True),
            },
            "case_count": len(case_to_fold),
            "folds": folds,
            "case_to_validation_fold": case_to_fold,
        },
        "preprocessing": {
            "target_shape_zyx": [160, 160, 160],
            "target_spacing_xyz_mm": [1.0, 0.55, 0.5],
            "normalization": "per-ROI z-score in submission inference; nnXNet plans use ZScoreNormalization",
            "roi": {
                "source": "Stage 1 publisher 2D vessel-box model",
                "bbox_margin_fraction": 0.05,
                "bbox_min_margin_voxels": 4,
                "roi_resize_order": 3,
            },
            "restoration": {
                "array_order": "zyx (SimpleITK array order)",
                "label_resize_order": 0,
                "write_copy_information_from_input": True,
                "crop_bounds": "crop_lower_zyx/crop_upper_zyx_exclusive from ROI transform",
            },
        },
        "checkpoints": make_stage_checkpoints(data_root, args.dataset, stage1, args.hash_checkpoints),
        "inference": {
            "folds": [0, 1, 2, 3, 4],
            "aggregation": "mean fold logits, then softmax/argmax",
            "aneurysm_probability_threshold": args.aneurysm_threshold,
            "min_component_voxels": args.min_component_voxels,
            "component_connectivity": 2,
            "stagec_location_rule": "mean fold location logits; mask unseen locations; top-1 label assigned to whole component",
            "vessel_distance_filter": None,
            "objectness_rejection": None,
        },
        "oof": {
            "strict": True,
            "validation_case_count": len(case_to_fold),
            "stage1_is_oof": False,
            "stage2_and_stagec_are_fold_specific": True,
            "fold_assignment_source": str(split_path),
        },
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "nnXNet_compile": os.environ.get("nnXNet_compile", "unset"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "unset"),
        },
    }
    # Results are optional because a fresh P0 manifest is normally created
    # before OOF evaluation. When supplied, their content hashes make the
    # reported metrics traceable to the exact artifact used.
    if args.evaluation_result:
        manifest["evaluation_result"] = file_info(
            args.evaluation_result, hash_file=True,
        )
    if args.leaderboard_result:
        manifest["leaderboard_result"] = file_info(
            args.leaderboard_result, hash_file=True,
        )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DATA_DEFAULT / "stagec_pred_oof_v2/experiment_manifest.json")
    parser.add_argument("--repo-root", type=Path, default=REPO_DEFAULT)
    parser.add_argument("--data-root", type=Path, default=DATA_DEFAULT)
    parser.add_argument("--dataset", default=DATASET_DEFAULT)
    parser.add_argument("--evaluator", type=Path, default=EVALUATOR_DEFAULT)
    parser.add_argument("--plans-name", default="TopAneuBraveCoWCoWPredROIPlansV2_160.json")
    parser.add_argument("--stage1-checkpoint", type=Path, default=None)
    parser.add_argument("--experiment-name", default="stagec_pred_oof_v2")
    parser.add_argument("--purpose", default="P0 locked strict-OOF baseline and calibration provenance")
    parser.add_argument("--aneurysm-threshold", type=float, default=0.10)
    parser.add_argument("--min-component-voxels", type=int, default=1)
    parser.add_argument("--hash-checkpoints", action="store_true", help="Hash large checkpoint files; can take several minutes")
    parser.add_argument("--evaluation-result", type=Path, default=None, help="Optional local official-evaluator JSON/CSV to record")
    parser.add_argument("--leaderboard-result", type=Path, default=None, help="Optional challenge leaderboard result artifact to record")
    parser.add_argument("--force", action="store_true", help="Replace an existing manifest")
    args = parser.parse_args()
    if not 0 <= args.aneurysm_threshold <= 1:
        parser.error("--aneurysm-threshold must be between 0 and 1")
    if args.min_component_voxels < 0:
        parser.error("--min-component-voxels must be >= 0")
    manifest = build_manifest(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(manifest, ensure_ascii=False, indent=2, default=_json_default) + "\n"
    args.output.write_text(serialized)
    # Sidecar hash lets a copied manifest be verified without mutating the
    # JSON to include a self-referential digest.
    sidecar = args.output.with_name(args.output.name + ".sha256")
    sidecar.write_text(f"{sha256_file(args.output)}  {args.output.name}\n")
    print(json.dumps({
        "saved": str(args.output),
        "sha256_sidecar": str(sidecar),
        "manifest_schema": manifest["manifest_schema"],
        "repository_sha": manifest["repository"]["sha"],
        "evaluator_sha256": manifest["official_evaluator"]["source_file"].get("sha256"),
        "cases": manifest["data"]["case_count"],
        "folds": len(manifest["data"]["folds"]),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
