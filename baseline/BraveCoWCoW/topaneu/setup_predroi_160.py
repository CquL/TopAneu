#!/usr/bin/env python3
"""Build an isolated 160^3 Stage-2 dataset from published Stage-1 predictions."""

from __future__ import annotations

import argparse
import json
import shutil
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from topaneu import setup_topaneu as base


DATASET_NAME = "Dataset505_TopAneuBraveCoWCoWPredROI"
PLANS_NAME = "TopAneuBraveCoWCoWPredROIPlansV2_160"
DATA_IDENTIFIER = f"{PLANS_NAME}_3d_fullres"
TARGET_SHAPE = np.asarray((160, 160, 160), dtype=int)


def prepare_case(
    image_path: Path,
    release: Path,
    transforms_dir: Path,
    images_tr: Path,
    labels_tr: Path,
    output_transforms: Path,
    force: bool,
) -> str:
    case = image_path.name.removesuffix("_0000.nii.gz")
    transform_path = transforms_dir / f"{case}.json"
    if not transform_path.is_file():
        raise FileNotFoundError(f"Missing Stage-1 transform: {transform_path}")
    output_image = images_tr / f"{case}_0000.nii.gz"
    output_label = labels_tr / f"{case}.nii.gz"
    output_transform = output_transforms / f"{case}.json"
    if (
        not force
        and output_image.is_file()
        and output_label.is_file()
        and output_transform.is_file()
    ):
        return case

    transform = json.loads(transform_path.read_text())
    image_itk = sitk.ReadImage(str(image_path))
    vessel_itk = sitk.ReadImage(str(release / "vessel_masks" / f"{case}.nii.gz"))
    location_itk = sitk.ReadImage(str(release / "location_masks" / f"{case}.nii.gz"))
    if image_itk.GetSize() != vessel_itk.GetSize() or image_itk.GetSize() != location_itk.GetSize():
        raise ValueError(f"Geometry size mismatch for {case}")

    image = sitk.GetArrayFromImage(image_itk).astype(np.float32, copy=False)
    vessel = sitk.GetArrayFromImage(vessel_itk).astype(np.uint8, copy=False)
    location = sitk.GetArrayFromImage(location_itk).astype(np.uint8, copy=False)
    if tuple(transform["source_shape_zyx"]) != image.shape:
        raise ValueError(f"Stage-1 transform shape mismatch for {case}")
    lower = np.asarray(transform["crop_lower_zyx"], dtype=int)
    upper = np.asarray(transform["crop_upper_zyx_exclusive"], dtype=int)
    if np.any(lower < 0) or np.any(upper > image.shape) or np.any(upper <= lower):
        raise ValueError(f"Invalid Stage-1 crop for {case}: {lower}..{upper}")
    slices = tuple(slice(int(a), int(b)) for a, b in zip(lower, upper))

    image_roi = base.resize_array(image[slices], TARGET_SHAPE, order=3).astype(np.float32, copy=False)
    vessel_roi = base.resize_array(vessel[slices], TARGET_SHAPE, order=0).astype(np.uint8, copy=False)
    location_roi = base.resize_array(location[slices], TARGET_SHAPE, order=0).astype(np.uint8, copy=False)
    target = vessel_roi.copy()
    aneurysm = location_roi > 0
    target[aneurysm] = 36 + location_roi[aneurysm]

    base.write_nifti(image_roi, output_image, sitk.sitkFloat32)
    base.write_nifti(target, output_label, sitk.sitkUInt8)
    transform["network_shape_zyx"] = TARGET_SHAPE.tolist()
    transform["stage2_dataset"] = DATASET_NAME
    output_transform.write_text(json.dumps(transform, indent=2))
    return case


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--stage1-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--template-plans", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    release = args.release.resolve()
    root = args.data_root.resolve()
    transforms_dir = args.stage1_root.resolve() / "roi_transforms"
    raw_dataset = root / "nnXNet_raw" / DATASET_NAME
    preprocessed_dataset = root / "nnXNet_preprocessed" / DATASET_NAME
    metadata_dir = root / "metadata"
    images_tr = raw_dataset / "imagesTr"
    labels_tr = raw_dataset / "labelsTr"
    output_transforms = metadata_dir / "roi_transforms"
    for folder in (
        images_tr,
        labels_tr,
        preprocessed_dataset,
        metadata_dir,
        output_transforms,
        root / "nnXNet_results",
    ):
        folder.mkdir(parents=True, exist_ok=True)

    vessel_labels = base.load_labels(release / "vessel_mapping.json")
    location_labels = base.load_labels(release / "location_mapping.json")
    if sorted(vessel_labels.values()) != list(range(1, 37)):
        raise ValueError("Expected vessel labels 1..36")
    if sorted(location_labels.values()) != list(range(1, 53)):
        raise ValueError("Expected location labels 1..52")

    image_paths = sorted((release / "images").glob("*_0000.nii.gz"))
    if len(image_paths) != 416:
        raise RuntimeError(f"Expected 416 TopAneu images, found {len(image_paths)}")
    worker = partial(
        prepare_case,
        release=release,
        transforms_dir=transforms_dir,
        images_tr=images_tr,
        labels_tr=labels_tr,
        output_transforms=output_transforms,
        force=args.force,
    )
    cases = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for index, case in enumerate(executor.map(worker, image_paths), start=1):
            cases.append(case)
            if index % 20 == 0 or index == len(image_paths):
                print(f"Prepared predicted ROI {index}/{len(image_paths)}", flush=True)

    base.build_dataset_json(raw_dataset, cases, vessel_labels, location_labels)
    shutil.copy2(raw_dataset / "dataset.json", preprocessed_dataset / "dataset.json")
    metadata_path, counts, pos_weights = base.build_metadata(
        cases, release, location_labels, metadata_dir
    )
    # Reuse the proven Stage-2 plan structure but give this dataset independent
    # identifiers so it never reads or writes Dataset503.
    base.DATASET_NAME = DATASET_NAME
    base.PLANS_NAME = PLANS_NAME
    base.PREPROCESSED_DATA_IDENTIFIER = DATA_IDENTIFIER
    plans_path = base.build_plans(
        args.template_plans.resolve(), preprocessed_dataset, counts, pos_weights
    )
    # Do not inherit an older host's setup_topaneu.py batch-size default. The
    # predicted-ROI dataset is explicitly planned for 160^3, batch 4 on A40.
    plans = json.loads(plans_path.read_text())
    configuration = plans["configurations"]["3d_fullres"]
    configuration["batch_size"] = 4
    configuration["patch_size"] = TARGET_SHAPE.tolist()
    plans_path.write_text(json.dumps(plans, indent=2))
    shutil.copy2(args.splits.resolve(), preprocessed_dataset / "splits_final.json")

    summary = {
        "dataset": DATASET_NAME,
        "num_cases": len(cases),
        "roi_source": str(transforms_dir),
        "raw_dataset": str(raw_dataset),
        "plans": str(plans_path),
        "metadata": str(metadata_path),
    }
    (metadata_dir / "setup_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
