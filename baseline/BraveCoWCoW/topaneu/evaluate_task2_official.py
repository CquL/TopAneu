#!/usr/bin/env python3
"""Evaluate five-fold TopAneu OOF masks with the official Task-2 code.

The network predictions are saved in the 160^3 vessel ROI. This script first
maps them back to the original image array using roi_transforms/*.json, then
calls eval/task2/evaluate.py without changing its metric implementation.
"""

import argparse
import importlib.util
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy.ndimage import zoom


def load_official(path: Path):
    spec = importlib.util.spec_from_file_location("topaneu_official_task2_evaluate", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def resize_nearest(array: np.ndarray, target_shape) -> np.ndarray:
    target_shape = np.asarray(target_shape, dtype=int)
    factors = target_shape.astype(float) / np.asarray(array.shape, dtype=float)
    resized = zoom(array, factors, order=0, mode="nearest", prefilter=False)
    fixed = np.zeros(tuple(target_shape), dtype=array.dtype)
    common = tuple(slice(0, min(a, b)) for a, b in zip(resized.shape, target_shape))
    fixed[common] = resized[common]
    return fixed


def restore_prediction(prediction_path: Path, transform_path: Path) -> np.ndarray:
    pred = sitk.GetArrayFromImage(sitk.ReadImage(str(prediction_path))).astype(np.uint8)
    meta = json.loads(transform_path.read_text())
    lower = np.asarray(meta["crop_lower_zyx"], dtype=int)
    upper = np.asarray(meta["crop_upper_zyx_exclusive"], dtype=int)
    crop_shape = upper - lower
    restored_crop = resize_nearest(pred, crop_shape)
    if "source_size_xyz" in meta:
        source_shape = tuple(reversed(meta["source_size_xyz"]))
    elif "source_shape_zyx" in meta:
        source_shape = tuple(meta["source_shape_zyx"])
    else:
        raise KeyError(f"ROI transform lacks source shape: {transform_path}")
    restored = np.zeros(source_shape, dtype=np.uint8)
    restored[tuple(slice(int(a), int(b)) for a, b in zip(lower, upper))] = restored_crop
    return restored


def evaluate_one(args):
    prediction_path, transform_dir, ground_truth_dir, official_path = args
    prediction_path = Path(prediction_path)
    case = prediction_path.name.removesuffix(".nii.gz")
    prediction = restore_prediction(prediction_path, Path(transform_dir) / f"{case}.json")
    gt_path = Path(ground_truth_dir) / f"{case}.nii.gz"
    gt_image = sitk.ReadImage(str(gt_path))
    if prediction.shape != sitk.GetArrayFromImage(gt_image).shape:
        raise ValueError(f"{case}: prediction={prediction.shape}, gt={sitk.GetArrayFromImage(gt_image).shape}")
    official = load_official(Path(official_path))
    pred_image = sitk.GetImageFromArray(prediction)
    pred_image.CopyInformation(gt_image)
    return case, official.evaluation_function(pred_image, gt_path, execute_in_docker=False)


def native(value):
    if isinstance(value, dict):
        return {key: native(item) for key, item in value.items()}
    if isinstance(value, list):
        return [native(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--official-evaluate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prediction-subdir", default="validation_location_masks_component")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    predictions = []
    fold_by_case = {}
    for fold in range(5):
        folder = args.results / f"fold_{fold}" / args.prediction_subdir
        for path in sorted(folder.glob("*.nii.gz")):
            if path.name in fold_by_case:
                raise RuntimeError(f"Duplicate OOF prediction: {path.name}")
            fold_by_case[path.name] = fold
            predictions.append(path)
    gt_names = {path.name for path in args.ground_truth.glob("*.nii.gz")}
    prediction_names = {path.name for path in predictions}
    if prediction_names != gt_names:
        raise RuntimeError(
            f"OOF/GT mismatch: predictions={len(prediction_names)}, gt={len(gt_names)}, "
            f"missing={sorted(gt_names-prediction_names)[:5]}, extra={sorted(prediction_names-gt_names)[:5]}"
        )

    jobs = [
        (str(path), str(args.metadata), str(args.ground_truth), str(args.official_evaluate))
        for path in predictions
    ]
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for index, result in enumerate(pool.map(evaluate_one, jobs), 1):
            results.append(result)
            if index % 20 == 0 or index == len(jobs):
                print(f"Official Task-2 evaluation: {index}/{len(jobs)}", flush=True)

    official = load_official(args.official_evaluate)
    metrics_per_case = [metrics for _, metrics in results]
    aggregates = official.evaluation_aggregation(metrics_per_case)
    averages = official.evaluation_average(aggregates)
    payload = native({
        "evaluation": "TopAneu-26 official eval/task2/evaluate.py",
        "num_oof_cases": len(results),
        "prediction_space": "fixed-size ROI restored to original image array",
        "prediction_subdir": args.prediction_subdir,
        "aggregates_avg": averages,
        "official_metrics": ["PRECISION", "RECALL", "F1", "MCC", "DICE", "HD95", "VOLSIM"],
        "aggregates_per_locations": aggregates,
        "case_to_fold": {case: fold_by_case[f"{case}.nii.gz"] for case, _ in results},
    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload["aggregates_avg"], indent=2))
    print("saved:", args.output)


if __name__ == "__main__":
    main()
