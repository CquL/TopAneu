#!/usr/bin/env python3
"""Compute aggregate 36-class vessel metrics for the exported OOF masks."""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import SimpleITK as sitk


def confusion(job):
    prediction_path, label_path = job
    pred = sitk.GetArrayFromImage(sitk.ReadImage(str(prediction_path))).astype(np.int16)
    combined = sitk.GetArrayFromImage(sitk.ReadImage(str(label_path))).astype(np.int16)
    gt = np.where((combined >= 1) & (combined <= 36), combined, 0)
    return np.bincount((gt.ravel() * 37 + pred.ravel()), minlength=37 * 37).reshape(37, 37)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    predictions = sorted(args.results.glob("fold_*/validation_vessel_masks/*.nii.gz"))
    jobs = [(p, args.labels / p.name) for p in predictions]
    matrix = np.zeros((37, 37), dtype=np.int64)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for index, item in enumerate(pool.map(confusion, jobs), 1):
            matrix += item
            if index % 32 == 0 or index == len(jobs):
                print(f"Vessel OOF: {index}/{len(jobs)}", flush=True)
    per_class = {}
    dice_values = []
    for class_id in range(1, 37):
        tp = int(matrix[class_id, class_id])
        fp = int(matrix[:, class_id].sum() - tp)
        fn = int(matrix[class_id, :].sum() - tp)
        score = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None
        if score is not None:
            dice_values.append(score)
        per_class[str(class_id)] = {"tp": tp, "fp": fp, "fn": fn, "dice": score}
    foreground_tp = sum(matrix[i, i] for i in range(1, 37))
    pred_foreground = matrix[:, 1:].sum()
    gt_foreground = matrix[1:, :].sum()
    payload = {
        "num_cases": len(predictions),
        "macro_dice_36_classes": float(np.mean(dice_values)),
        "median_dice_36_classes": float(np.median(dice_values)),
        "micro_exact_class_dice": float(2 * foreground_tp / (pred_foreground + gt_foreground)),
        "per_class": per_class,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    print(json.dumps({k: v for k, v in payload.items() if k != "per_class"}, indent=2))


if __name__ == "__main__":
    main()
