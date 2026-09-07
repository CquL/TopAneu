#!/usr/bin/env python3
"""Run the published BraveCoWCoW Stage-1 ROI model on TopAneu images.

The model is loaded once and reused for all cases. Outputs are lightweight JSON
transforms; optional NIfTI bbox masks are intended for visual QA only.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch

from nnxnet.inference.predict_from_raw_data_2D_orthogonal_planes_fast import (
    nnXNetPredictor as ROIPredictor,
)


DEFAULT_RELEASE = Path("/data/cyf/shared_data/TopAneu/topaneu_release")
DEFAULT_OUTPUT = Path("/data/cyf/shared_data/TopAneu/BraveCoWCoW_stage1_roi")
DEFAULT_MODEL = Path(
    "/data/cyf/codes/TopAneu/submission/BraveCoWCoW_Task2/model/stage1_roi/"
    "Dataset180_2D_vessel_box_seg_stable/nnUNetTrainer__nnUNetPlans__2d"
)
TARGET_SPACING = np.asarray((1.0, 0.55, 0.5), dtype=float)


def bbox_from_mask(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    coords = np.argwhere(mask > 0)
    if coords.size == 0:
        return None
    return coords.min(0), coords.max(0) + 1


def expand_bbox(
    lower: np.ndarray,
    upper: np.ndarray,
    shape: tuple[int, ...],
    margin_fraction: float,
    minimum_margin_voxels: int,
) -> tuple[np.ndarray, np.ndarray]:
    extent = np.maximum(upper - lower, 1)
    margin = np.maximum(
        minimum_margin_voxels, np.ceil(extent * margin_fraction)
    ).astype(int)
    return (
        np.maximum(0, lower - margin),
        np.minimum(np.asarray(shape, dtype=int), upper + margin),
    )


class Stage1Runner:
    def __init__(self, model_dir: Path, device: torch.device, batch_size: int):
        if not model_dir.is_dir():
            raise FileNotFoundError(f"Stage-1 model directory does not exist: {model_dir}")
        checkpoint = model_dir / "fold_0" / "checkpoint_final.pth"
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Stage-1 checkpoint does not exist: {checkpoint}")
        self.device = device
        self.batch_size = batch_size
        self.predictor = ROIPredictor(
            tile_step_size=0.5,
            use_mirroring=False,
            use_gaussian=True,
            perform_everything_on_device=device.type == "cuda",
            device=device,
            allow_tqdm=False,
        )
        self.predictor.initialize_from_trained_model_folder(
            str(model_dir), use_folds=(0,), checkpoint_name="checkpoint_final.pth"
        )
        self.predictor.initialize_network_and_gaussian()
        # The custom predictor already loaded fold 0 into the network. Releasing
        # this duplicate state saves host memory during a long batch run.
        self.predictor.list_of_parameters = []

    def predict_mask(self, image: np.ndarray, spacing_xyz: tuple[float, ...]) -> np.ndarray:
        with torch.inference_mode():
            return self.predictor.predict_from_multi_axial_slices(
                image[None],
                np.asarray(spacing_xyz, dtype=float),
                TARGET_SPACING,
                max_batch_size=self.batch_size,
                mask_return=True,
            ).astype(np.uint8, copy=False)

    def close(self) -> None:
        del self.predictor
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()


def case_id(path: Path) -> str:
    name = path.name
    suffix = "_0000.nii.gz"
    if not name.endswith(suffix):
        raise ValueError(f"Unexpected TopAneu image filename: {name}")
    return name[: -len(suffix)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release", type=Path, default=DEFAULT_RELEASE)
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path(os.environ.get("TOPANEU_STAGE1_MODEL_DIR", DEFAULT_MODEL)),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(os.environ.get("TOPANEU_STAGE1_DATA", DEFAULT_OUTPUT)),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--margin-fraction", type=float, default=0.05)
    parser.add_argument("--minimum-margin-voxels", type=int, default=4)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--save-mask", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.margin_fraction < 0 or args.minimum_margin_voxels < 0:
        raise ValueError("ROI margins must be non-negative")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    image_dir = args.release / "images"
    images = sorted(image_dir.glob("*_0000.nii.gz"))
    if args.case:
        requested = set(args.case)
        images = [path for path in images if case_id(path) in requested]
        missing = requested - {case_id(path) for path in images}
        if missing:
            raise FileNotFoundError(f"Requested cases not found: {sorted(missing)}")
    if args.limit is not None:
        images = images[: args.limit]
    if not images:
        raise RuntimeError(f"No TopAneu images found below {image_dir}")

    transforms_dir = args.output_root / "roi_transforms"
    masks_dir = args.output_root / "bbox_masks"
    transforms_dir.mkdir(parents=True, exist_ok=True)
    if args.save_mask:
        masks_dir.mkdir(parents=True, exist_ok=True)

    runner = Stage1Runner(args.model_dir, device, args.batch_size)
    try:
        for index, image_path in enumerate(images, start=1):
            case = case_id(image_path)
            output_json = transforms_dir / f"{case}.json"
            output_mask = masks_dir / f"{case}.nii.gz"
            if output_json.is_file() and not args.overwrite:
                print(f"[{index}/{len(images)}] skip {case}")
                continue

            image_itk = sitk.ReadImage(str(image_path))
            image = sitk.GetArrayFromImage(image_itk).astype(np.float32, copy=False)
            predicted_mask = runner.predict_mask(image, image_itk.GetSpacing())
            raw_bbox = bbox_from_mask(predicted_mask)
            used_full_volume_fallback = raw_bbox is None
            if raw_bbox is None:
                raw_lower = np.zeros(3, dtype=int)
                raw_upper = np.asarray(image.shape, dtype=int)
            else:
                raw_lower, raw_upper = raw_bbox
            lower, upper = expand_bbox(
                raw_lower,
                raw_upper,
                image.shape,
                args.margin_fraction,
                args.minimum_margin_voxels,
            )
            payload = {
                "case": case,
                "image_path": str(image_path),
                "model_dir": str(args.model_dir),
                "checkpoint": "fold_0/checkpoint_final.pth",
                "source_shape_zyx": list(map(int, image.shape)),
                "source_spacing_xyz": list(map(float, image_itk.GetSpacing())),
                "source_origin_xyz": list(map(float, image_itk.GetOrigin())),
                "source_direction": list(map(float, image_itk.GetDirection())),
                "raw_bbox_lower_zyx": raw_lower.tolist(),
                "raw_bbox_upper_zyx_exclusive": raw_upper.tolist(),
                "crop_lower_zyx": lower.tolist(),
                "crop_upper_zyx_exclusive": upper.tolist(),
                "margin_fraction": args.margin_fraction,
                "minimum_margin_voxels": args.minimum_margin_voxels,
                "used_full_volume_fallback": used_full_volume_fallback,
            }
            output_json.write_text(json.dumps(payload, indent=2))

            if args.save_mask:
                final_mask = np.zeros(image.shape, dtype=np.uint8)
                final_mask[tuple(slice(int(a), int(b)) for a, b in zip(lower, upper))] = 1
                mask_itk = sitk.GetImageFromArray(final_mask)
                mask_itk.CopyInformation(image_itk)
                sitk.WriteImage(mask_itk, str(output_mask), True)
            crop_fraction = float(np.prod(upper - lower) / np.prod(image.shape))
            print(
                f"[{index}/{len(images)}] {case}: "
                f"crop={lower.tolist()}..{upper.tolist()} "
                f"volume={crop_fraction:.3f} fallback={used_full_volume_fallback}"
            )
    finally:
        runner.close()


if __name__ == "__main__":
    main()

