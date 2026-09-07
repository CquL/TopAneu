#!/usr/bin/env python3
"""Evaluate Stage-1 predicted crops against TopAneu vessel and aneurysm masks."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import ndimage
import SimpleITK as sitk


DEFAULT_RELEASE = Path("/data/cyf/shared_data/TopAneu/topaneu_release")
DEFAULT_STAGE1 = Path("/data/cyf/shared_data/TopAneu/BraveCoWCoW_stage1_roi")


def modality(case: str) -> str:
    return "MRA" if "_mr_" in case else "CTA"


def center(case: str) -> str:
    match = re.match(r"topaneu_(center\d+)_", case)
    return match.group(1) if match else "unknown"


def components_by_location(location: np.ndarray):
    structure = ndimage.generate_binary_structure(3, 2)
    for location_id in np.unique(location):
        if location_id == 0:
            continue
        labeled, count = ndimage.label(location == location_id, structure=structure)
        for component_id in range(1, count + 1):
            coords = np.argwhere(labeled == component_id)
            if coords.size:
                yield int(location_id), coords.min(0), coords.max(0) + 1, int(len(coords))


def summarize(rows: list[dict]) -> dict:
    positive = [row for row in rows if row["positive"]]
    lesion_total = sum(row["lesions"] for row in rows)
    lesion_full = sum(row["lesions_fully_covered"] for row in rows)
    lesion_any = sum(row["lesions_with_any_overlap"] for row in rows)
    location_voxels = sum(row["location_voxels"] for row in rows)
    covered_location_voxels = sum(row["covered_location_voxels"] for row in rows)
    vessel_voxels = sum(row["vessel_voxels"] for row in rows)
    covered_vessel_voxels = sum(row["covered_vessel_voxels"] for row in rows)
    crop_fractions = [row["crop_fraction"] for row in rows]
    return {
        "cases": len(rows),
        "positive_cases": len(positive),
        "positive_cases_fully_covered": sum(row["case_fully_covered"] for row in positive),
        "positive_case_full_coverage_rate": (
            sum(row["case_fully_covered"] for row in positive) / len(positive)
            if positive else None
        ),
        "lesions": lesion_total,
        "lesions_fully_covered": lesion_full,
        "lesion_full_coverage_rate": lesion_full / lesion_total if lesion_total else None,
        "lesions_with_any_overlap": lesion_any,
        "lesion_any_overlap_rate": lesion_any / lesion_total if lesion_total else None,
        "location_voxel_recall": (
            covered_location_voxels / location_voxels if location_voxels else None
        ),
        "vessel_voxel_recall": covered_vessel_voxels / vessel_voxels if vessel_voxels else None,
        "full_volume_fallbacks": sum(row["used_full_volume_fallback"] for row in rows),
        "crop_fraction_mean": float(np.mean(crop_fractions)) if crop_fractions else None,
        "crop_fraction_median": float(np.median(crop_fractions)) if crop_fractions else None,
        "crop_fraction_p95": float(np.percentile(crop_fractions, 95)) if crop_fractions else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release", type=Path, default=DEFAULT_RELEASE)
    parser.add_argument("--stage1-root", type=Path, default=DEFAULT_STAGE1)
    args = parser.parse_args()

    transforms = sorted((args.stage1_root / "roi_transforms").glob("*.json"))
    if not transforms:
        raise RuntimeError(f"No ROI transforms found under {args.stage1_root}")

    rows = []
    for transform_path in transforms:
        metadata = json.loads(transform_path.read_text())
        case = metadata["case"]
        location_itk = sitk.ReadImage(str(args.release / "location_masks" / f"{case}.nii.gz"))
        vessel_itk = sitk.ReadImage(str(args.release / "vessel_masks" / f"{case}.nii.gz"))
        location = sitk.GetArrayFromImage(location_itk)
        vessel = sitk.GetArrayFromImage(vessel_itk)
        expected_shape = tuple(metadata["source_shape_zyx"])
        if location.shape != expected_shape or vessel.shape != expected_shape:
            raise ValueError(f"Geometry mismatch for {case}")
        lower = np.asarray(metadata["crop_lower_zyx"], dtype=int)
        upper = np.asarray(metadata["crop_upper_zyx_exclusive"], dtype=int)
        slices = tuple(slice(int(a), int(b)) for a, b in zip(lower, upper))

        lesion_total = lesion_full = lesion_any = 0
        for _, component_lower, component_upper, _ in components_by_location(location):
            lesion_total += 1
            is_full = bool(np.all(component_lower >= lower) and np.all(component_upper <= upper))
            has_overlap = bool(np.all(component_upper > lower) and np.all(component_lower < upper))
            lesion_full += int(is_full)
            lesion_any += int(has_overlap)

        location_voxels = int(np.count_nonzero(location))
        covered_location_voxels = int(np.count_nonzero(location[slices]))
        vessel_voxels = int(np.count_nonzero(vessel))
        covered_vessel_voxels = int(np.count_nonzero(vessel[slices]))
        rows.append({
            "case": case,
            "center": center(case),
            "modality": modality(case),
            "positive": int(location_voxels > 0),
            "case_fully_covered": int(location_voxels == covered_location_voxels),
            "lesions": lesion_total,
            "lesions_fully_covered": lesion_full,
            "lesions_with_any_overlap": lesion_any,
            "location_voxels": location_voxels,
            "covered_location_voxels": covered_location_voxels,
            "vessel_voxels": vessel_voxels,
            "covered_vessel_voxels": covered_vessel_voxels,
            "crop_fraction": float(np.prod(upper - lower) / np.prod(location.shape)),
            "used_full_volume_fallback": int(metadata["used_full_volume_fallback"]),
        })

    by_group: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_group[f"modality::{row['modality']}"] .append(row)
        by_group[f"center::{row['center']}"] .append(row)
    report = {
        "overall": summarize(rows),
        "groups": {key: summarize(value) for key, value in sorted(by_group.items())},
    }
    args.stage1_root.mkdir(parents=True, exist_ok=True)
    output_json = args.stage1_root / "roi_coverage_summary.json"
    output_csv = args.stage1_root / "roi_coverage_per_case.csv"
    output_json.write_text(json.dumps(report, indent=2))
    with output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(report["overall"], indent=2))
    print(f"Summary: {output_json}")
    print(f"Per-case: {output_csv}")


if __name__ == "__main__":
    main()

