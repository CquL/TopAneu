#!/usr/bin/env python3

import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy import ndimage


DATA_ROOT = Path(
    "/data/cyf/shared_data/TopAneu/BraveCoWCoW_predroi_160"
)

GT_ROOT = (
    DATA_ROOT
    / "nnXNet_raw"
    / "Dataset505_TopAneuBraveCoWCoWPredROI"
    / "labelsTr"
)

RESULT_ROOT = (
    DATA_ROOT
    / "nnXNet_results"
    / "Dataset505_TopAneuBraveCoWCoWPredROI"
    / "nnXNetTrainer_TopAneuAneurysmOnly__"
      "TopAneuBraveCoWCoWPredROIPlansV2_160__3d_fullres"
)

OUT_ROOT = Path(
    "/data/cyf/codes/TopAneu/analysis/stage2b_error_decomposition"
)

STRUCTURE = ndimage.generate_binary_structure(3, 2)


def get_center(case):
    m = re.match(r"topaneu_(center\d+)_", case)
    return m.group(1) if m else "unknown"


def get_modality(case):
    if "_ct_" in case:
        return "CTA"
    if "_mr_" in case:
        return "MRA"
    return "unknown"


def safe_div(a, b):
    return float(a / b) if b else None


def hd95(pred, gt, spacing_zyx):
    if not pred.any() or not gt.any():
        return None

    structure = ndimage.generate_binary_structure(3, 1)

    pred_surface = pred & ~ndimage.binary_erosion(
        pred, structure=structure, border_value=0
    )
    gt_surface = gt & ~ndimage.binary_erosion(
        gt, structure=structure, border_value=0
    )

    d_to_gt = ndimage.distance_transform_edt(
        ~gt_surface, sampling=spacing_zyx
    )
    d_to_pred = ndimage.distance_transform_edt(
        ~pred_surface, sampling=spacing_zyx
    )

    distances = np.concatenate(
        [d_to_gt[pred_surface], d_to_pred[gt_surface]]
    )

    if len(distances) == 0:
        return None

    return float(np.percentile(distances, 95))


def size_bin(volume_mm3):
    if volume_mm3 < 10:
        return "<10 mm3"
    if volume_mm3 < 50:
        return "10-49 mm3"
    if volume_mm3 < 200:
        return "50-199 mm3"
    return ">=200 mm3"


case_rows = []
lesion_rows = []

for fold in range(5):

    fold_root = RESULT_ROOT / f"fold_{fold}"
    pred_root = fold_root / "validation_aneurysm_masks"

    if not pred_root.is_dir():
        raise RuntimeError(f"Missing prediction directory: {pred_root}")

    pred_files = sorted(pred_root.glob("*.nii.gz"))

    print(f"fold {fold}: {len(pred_files)} validation cases")

    for pred_path in pred_files:

        case = pred_path.name.removesuffix(".nii.gz")
        gt_path = GT_ROOT / f"{case}.nii.gz"

        if not gt_path.is_file():
            raise FileNotFoundError(gt_path)

        pred_itk = sitk.ReadImage(str(pred_path))
        gt_itk = sitk.ReadImage(str(gt_path))

        pred = sitk.GetArrayFromImage(pred_itk) > 0

        # Dataset505:
        # 0       = background
        # 1..36   = vessel
        # 37..88  = aneurysm locations
        combined = sitk.GetArrayFromImage(gt_itk)
        gt = combined >= 37

        if pred.shape != gt.shape:
            raise RuntimeError(
                f"{case}: shape mismatch pred={pred.shape}, gt={gt.shape}"
            )

        spacing_xyz = gt_itk.GetSpacing()
        spacing_zyx = tuple(float(x) for x in spacing_xyz[::-1])
        voxel_volume = float(np.prod(spacing_xyz))

        tp = int(np.count_nonzero(pred & gt))
        fp = int(np.count_nonzero(pred & ~gt))
        fn = int(np.count_nonzero(~pred & gt))

        dice = safe_div(2 * tp, 2 * tp + fp + fn)
        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)

        gt_cc, gt_n = ndimage.label(gt, structure=STRUCTURE)
        pred_cc, pred_n = ndimage.label(pred, structure=STRUCTURE)

        detected_lesions = 0

        # -------------------------------------------------
        # GT lesion-level analysis
        # -------------------------------------------------
        for lesion_id in range(1, gt_n + 1):

            lesion = gt_cc == lesion_id
            size_vox = int(lesion.sum())
            volume_mm3 = size_vox * voxel_volume

            overlap = int(np.count_nonzero(pred & lesion))
            detected = overlap > 0

            if detected:
                detected_lesions += 1

            lesion_rows.append({
                "case": case,
                "fold": fold,
                "center": get_center(case),
                "modality": get_modality(case),

                "lesion_id": lesion_id,

                "size_voxels": size_vox,
                "volume_mm3": volume_mm3,
                "size_bin": size_bin(volume_mm3),

                "detected": int(detected),
                "overlap_voxels": overlap,

                # 在该 GT lesion 内，有多少比例被预测覆盖
                "lesion_voxel_recall": safe_div(overlap, size_vox),
            })

        # -------------------------------------------------
        # Predicted component FP analysis
        # -------------------------------------------------
        fp_components = 0

        for component_id in range(1, pred_n + 1):
            component = pred_cc == component_id

            if not np.any(gt & component):
                fp_components += 1

        case_rows.append({
            "case": case,
            "fold": fold,
            "center": get_center(case),
            "modality": get_modality(case),

            "positive": int(gt.any()),

            "gt_voxels": int(gt.sum()),
            "pred_voxels": int(pred.sum()),

            "tp": tp,
            "fp": fp,
            "fn": fn,

            "dice": dice,
            "precision": precision,
            "recall": recall,

            "hd95_mm": hd95(pred, gt, spacing_zyx),

            "gt_lesions": int(gt_n),
            "detected_lesions": int(detected_lesions),

            "pred_components": int(pred_n),
            "fp_components": int(fp_components),
        })


# =========================================================
# Save CSV
# =========================================================

case_csv = OUT_ROOT / "stage2b_case_metrics.csv"
lesion_csv = OUT_ROOT / "stage2b_lesion_metrics.csv"

with case_csv.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=case_rows[0].keys())
    writer.writeheader()
    writer.writerows(case_rows)

with lesion_csv.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=lesion_rows[0].keys())
    writer.writeheader()
    writer.writerows(lesion_rows)


# =========================================================
# Aggregate case metrics
# =========================================================

def summarize_cases(rows):

    tp = sum(r["tp"] for r in rows)
    fp = sum(r["fp"] for r in rows)
    fn = sum(r["fn"] for r in rows)

    gt_lesions = sum(r["gt_lesions"] for r in rows)
    detected = sum(r["detected_lesions"] for r in rows)

    pred_components = sum(r["pred_components"] for r in rows)
    fp_components = sum(r["fp_components"] for r in rows)

    positive_dice = [
        r["dice"]
        for r in rows
        if r["positive"] and r["dice"] is not None
    ]

    hd = [
        r["hd95_mm"]
        for r in rows
        if r["hd95_mm"] is not None
    ]

    return {
        "cases": len(rows),
        "positive_cases": sum(r["positive"] for r in rows),

        "micro_dice": safe_div(2 * tp, 2 * tp + fp + fn),
        "micro_precision": safe_div(tp, tp + fp),
        "micro_recall": safe_div(tp, tp + fn),

        "mean_positive_case_dice": (
            float(np.mean(positive_dice))
            if positive_dice else None
        ),

        "gt_lesions": gt_lesions,
        "detected_lesions": detected,
        "lesion_recall": safe_div(detected, gt_lesions),

        "pred_components": pred_components,
        "fp_components": fp_components,

        "mean_finite_hd95_mm": (
            float(np.mean(hd)) if hd else None
        ),
    }


def summarize_lesions(rows):

    n = len(rows)
    detected = sum(r["detected"] for r in rows)

    recalls = [
        r["lesion_voxel_recall"]
        for r in rows
        if r["lesion_voxel_recall"] is not None
    ]

    sizes = [r["volume_mm3"] for r in rows]

    return {
        "lesions": n,
        "detected": detected,
        "detection_recall": safe_div(detected, n),

        "mean_lesion_voxel_recall": (
            float(np.mean(recalls))
            if recalls else None
        ),

        "median_volume_mm3": (
            float(np.median(sizes))
            if sizes else None
        ),
    }


summary = {
    "overall": summarize_cases(case_rows),
    "by_fold": {},
    "by_modality": {},
    "by_center": {},
    "by_fold_modality": {},
    "by_size": {},
}


for fold in range(5):
    rows = [r for r in case_rows if r["fold"] == fold]
    summary["by_fold"][str(fold)] = summarize_cases(rows)


for modality in sorted(set(r["modality"] for r in case_rows)):
    rows = [r for r in case_rows if r["modality"] == modality]
    summary["by_modality"][modality] = summarize_cases(rows)


for center in sorted(set(r["center"] for r in case_rows)):
    rows = [r for r in case_rows if r["center"] == center]
    summary["by_center"][center] = summarize_cases(rows)


for fold in range(5):
    for modality in ("CTA", "MRA"):
        rows = [
            r for r in case_rows
            if r["fold"] == fold and r["modality"] == modality
        ]
        if rows:
            summary["by_fold_modality"][
                f"fold{fold}_{modality}"
            ] = summarize_cases(rows)


for b in ["<10 mm3", "10-49 mm3", "50-199 mm3", ">=200 mm3"]:
    rows = [r for r in lesion_rows if r["size_bin"] == b]
    summary["by_size"][b] = summarize_lesions(rows)


summary_path = OUT_ROOT / "stage2b_error_summary.json"
summary_path.write_text(json.dumps(summary, indent=2))


# =========================================================
# Print important summary
# =========================================================

print("\n====================================================")
print("OVERALL")
print("====================================================")
print(json.dumps(summary["overall"], indent=2))

print("\n====================================================")
print("BY FOLD")
print("====================================================")
for k, v in summary["by_fold"].items():
    print(f"\nfold {k}")
    print(json.dumps(v, indent=2))

print("\n====================================================")
print("BY MODALITY")
print("====================================================")
for k, v in summary["by_modality"].items():
    print(f"\n{k}")
    print(json.dumps(v, indent=2))

print("\n====================================================")
print("BY CENTER")
print("====================================================")
for k, v in summary["by_center"].items():
    print(f"\n{k}")
    print(json.dumps(v, indent=2))

print("\n====================================================")
print("BY LESION SIZE")
print("====================================================")
for k, v in summary["by_size"].items():
    print(f"\n{k}")
    print(json.dumps(v, indent=2))

print("\nSaved:")
print(case_csv)
print(lesion_csv)
print(summary_path)

