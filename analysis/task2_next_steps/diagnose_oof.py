#!/usr/bin/env python3
"""Decompose TopAneu Task-2 OOF errors in the 160^3 ROI space."""

import argparse
import csv
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy import ndimage


SIDE = {i: "R" for i in [1, 3, 5, 9, 11, 13, 15, 18, 20, 22, 24, 26, 28, 30, 32, 34, 37, 39, 41, 43, 45, 47, 49, 51]}
SIDE.update({i: "L" for i in [2, 4, 6, 10, 12, 14, 16, 19, 21, 23, 25, 27, 29, 31, 33, 35, 38, 40, 42, 44, 46, 48, 50, 52]})


def territory(label_id):
    if 1 <= label_id <= 17:
        return "VA/BA"
    if 18 <= label_id <= 21:
        return "PCA"
    if 22 <= label_id <= 35:
        return "ICA"
    if 36 <= label_id <= 44:
        return "ACA"
    if 45 <= label_id <= 52:
        return "MCA"
    return "background"


def read_array(path):
    return sitk.GetArrayFromImage(sitk.ReadImage(str(path)))


def dice(a, b):
    denom = int(a.sum()) + int(b.sum())
    return (2.0 * int(np.count_nonzero(a & b)) / denom) if denom else 1.0


def analyze_case(job):
    case, pred_path, gt_path, gt_presence = job
    pred = read_array(pred_path).astype(np.uint8)
    combined = read_array(gt_path)
    gt = np.where(combined > 36, combined - 36, 0).astype(np.uint8)
    pred_bin = pred > 0
    gt_bin = gt > 0

    pred_cc, pred_n = ndimage.label(pred_bin)
    oracle = pred.copy()
    pred_overlap = pred_exact = pred_territory = pred_side_n = pred_side_ok = 0
    pred_fp_components = 0
    for component_id in range(1, pred_n + 1):
        component = pred_cc == component_id
        labels = gt[component]
        labels = labels[labels > 0]
        if labels.size == 0:
            pred_fp_components += 1
            continue
        pred_overlap += 1
        true_label = int(np.bincount(labels, minlength=53).argmax())
        predicted_values = pred[component]
        predicted_values = predicted_values[predicted_values > 0]
        predicted_label = int(np.bincount(predicted_values, minlength=53).argmax())
        oracle[component] = true_label
        pred_exact += int(predicted_label == true_label)
        pred_territory += int(territory(predicted_label) == territory(true_label))
        if true_label in SIDE and predicted_label in SIDE:
            pred_side_n += 1
            pred_side_ok += int(SIDE[predicted_label] == SIDE[true_label])

    gt_components = gt_detected = gt_exact = gt_territory = gt_side_n = gt_side_ok = 0
    for true_label in range(1, 53):
        class_cc, class_n = ndimage.label(gt == true_label)
        gt_components += class_n
        for component_id in range(1, class_n + 1):
            predicted_values = pred[class_cc == component_id]
            predicted_values = predicted_values[predicted_values > 0]
            if predicted_values.size == 0:
                continue
            gt_detected += 1
            predicted_label = int(np.bincount(predicted_values, minlength=53).argmax())
            gt_exact += int(predicted_label == true_label)
            gt_territory += int(territory(predicted_label) == territory(true_label))
            if true_label in SIDE and predicted_label in SIDE:
                gt_side_n += 1
                gt_side_ok += int(SIDE[predicted_label] == SIDE[true_label])

    tp_binary = int(np.count_nonzero(pred_bin & gt_bin))
    pred_voxels = int(pred_bin.sum())
    gt_voxels = int(gt_bin.sum())
    tp_location = int(np.count_nonzero((pred == gt) & gt_bin))
    tp_oracle = int(np.count_nonzero((oracle == gt) & gt_bin))
    return {
        "case": case,
        "gt_presence": int(gt_presence),
        "pred_presence": int(pred_n > 0),
        "pred_components": int(pred_n),
        "gt_components": int(gt_components),
        "pred_overlap_components": int(pred_overlap),
        "pred_fp_components": int(pred_fp_components),
        "gt_detected_components": int(gt_detected),
        "pred_exact_components": int(pred_exact),
        "pred_territory_components": int(pred_territory),
        "pred_side_denominator": int(pred_side_n),
        "pred_side_correct": int(pred_side_ok),
        "gt_exact_detected": int(gt_exact),
        "gt_territory_detected": int(gt_territory),
        "gt_side_denominator": int(gt_side_n),
        "gt_side_correct": int(gt_side_ok),
        "pred_voxels": pred_voxels,
        "gt_voxels": gt_voxels,
        "tp_binary_voxels": tp_binary,
        "tp_location_voxels": tp_location,
        "tp_oracle_location_voxels": tp_oracle,
        "binary_dice": dice(pred_bin, gt_bin),
    }


def safe_ratio(a, b):
    return float(a / b) if b else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    metadata = {row["SeriesInstanceUID"]: int(row["Aneurysm Present"])
                for row in csv.DictReader(args.metadata.open())}
    predictions = sorted(args.results.glob("fold_*/validation_location_masks_component/*.nii.gz"))
    jobs = []
    for prediction in predictions:
        case = prediction.name.removesuffix(".nii.gz")
        jobs.append((case, prediction, args.labels / f"{case}.nii.gz", metadata[case]))
    if len(jobs) != len(metadata):
        raise RuntimeError(f"Expected {len(metadata)} OOF cases, found {len(jobs)}")

    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for index, row in enumerate(pool.map(analyze_case, jobs), 1):
            rows.append(row)
            if index % 32 == 0 or index == len(jobs):
                print(f"OOF decomposition: {index}/{len(jobs)}", flush=True)

    sums = {key: sum(row[key] for row in rows) for key in [
        "pred_components", "gt_components", "pred_overlap_components", "pred_fp_components",
        "gt_detected_components", "pred_exact_components", "pred_territory_components",
        "pred_side_denominator", "pred_side_correct", "gt_exact_detected",
        "gt_territory_detected", "gt_side_denominator", "gt_side_correct", "pred_voxels",
        "gt_voxels", "tp_binary_voxels", "tp_location_voxels", "tp_oracle_location_voxels",
    ]}
    tp_presence = sum(r["gt_presence"] and r["pred_presence"] for r in rows)
    fp_presence = sum((not r["gt_presence"]) and r["pred_presence"] for r in rows)
    fn_presence = sum(r["gt_presence"] and (not r["pred_presence"]) for r in rows)
    tn_presence = sum((not r["gt_presence"]) and (not r["pred_presence"]) for r in rows)
    denom_voxels = sums["pred_voxels"] + sums["gt_voxels"]
    positive_case_dice = [r["binary_dice"] for r in rows if r["gt_presence"]]

    summary = {
        "num_cases": len(rows),
        "num_positive_cases": sum(r["gt_presence"] for r in rows),
        "num_negative_cases": sum(not r["gt_presence"] for r in rows),
        "presence": {
            "tp": tp_presence, "fp": fp_presence, "fn": fn_presence, "tn": tn_presence,
            "precision": safe_ratio(tp_presence, tp_presence + fp_presence),
            "recall": safe_ratio(tp_presence, tp_presence + fn_presence),
            "specificity": safe_ratio(tn_presence, tn_presence + fp_presence),
        },
        "binary_segmentation": {
            "micro_dice": safe_ratio(2 * sums["tp_binary_voxels"], denom_voxels),
            "micro_precision": safe_ratio(sums["tp_binary_voxels"], sums["pred_voxels"]),
            "micro_recall": safe_ratio(sums["tp_binary_voxels"], sums["gt_voxels"]),
            "mean_dice_positive_cases": float(np.mean(positive_case_dice)),
            "median_dice_positive_cases": float(np.median(positive_case_dice)),
        },
        "component_detection": {
            **{key: sums[key] for key in ["pred_components", "gt_components", "pred_overlap_components", "pred_fp_components", "gt_detected_components"]},
            "pred_component_spatial_precision": safe_ratio(sums["pred_overlap_components"], sums["pred_components"]),
            "gt_lesion_detection_recall": safe_ratio(sums["gt_detected_components"], sums["gt_components"]),
        },
        "location_given_overlap": {
            "exact_accuracy_pred_component_grain": safe_ratio(sums["pred_exact_components"], sums["pred_overlap_components"]),
            "territory_accuracy_pred_component_grain": safe_ratio(sums["pred_territory_components"], sums["pred_overlap_components"]),
            "side_accuracy_when_both_lateralized": safe_ratio(sums["pred_side_correct"], sums["pred_side_denominator"]),
            "exact_accuracy_detected_gt_lesion_grain": safe_ratio(sums["gt_exact_detected"], sums["gt_detected_components"]),
            "territory_accuracy_detected_gt_lesion_grain": safe_ratio(sums["gt_territory_detected"], sums["gt_detected_components"]),
        },
        "voxel_location": {
            "current_micro_dice": safe_ratio(2 * sums["tp_location_voxels"], denom_voxels),
            "oracle_relabel_micro_dice": safe_ratio(2 * sums["tp_oracle_location_voxels"], denom_voxels),
            "oracle_definition": "Each predicted binary component that overlaps GT is relabeled to its dominant overlapping GT location; pure spatial false positives are retained.",
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "oof_case_decomposition.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "oof_error_decomposition.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
