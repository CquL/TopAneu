"""TopAneu Task-2 inference: Stage-1 ROI -> Stage-2 multi-task ensemble."""

from __future__ import annotations

import gc
import os
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.ndimage import zoom
import SimpleITK as sitk
import torch

from nnxnet.inference.predict_from_raw_data_2D_orthogonal_planes_fast import (
    nnXNetPredictor as ROIPredictor,
)
from nnxnet.inference.predict_from_raw_data_two_seg_with_cls_no_seg_return_no_filter import (
    nnXNetPredictor as Stage2Predictor,
)
from topaneu.component_location import assign_components


APP_DIR = Path(__file__).resolve().parent
MODEL_ROOT = Path(os.environ.get("TOPANEU_MODEL_DIR", "/opt/ml/model"))
if not MODEL_ROOT.is_dir():
    MODEL_ROOT = APP_DIR / "model"

STAGE1_DIR = (
    MODEL_ROOT
    / "stage1_roi"
    / "Dataset180_2D_vessel_box_seg_stable"
    / "nnUNetTrainer__nnUNetPlans__2d"
)
STAGE2_DIR = MODEL_ROOT / "stage2_topaneu"
TARGET_SHAPE = np.asarray((160, 160, 160), dtype=int)  # z, y, x
ROI_TARGET_SPACING = np.asarray((1.0, 0.55, 0.5), dtype=float)


def _folds() -> tuple[int, ...]:
    raw = os.environ.get("TOPANEU_FOLDS", "0,1,2,3,4")
    result = tuple(int(value.strip()) for value in raw.split(",") if value.strip())
    if not result or any(value not in range(5) for value in result):
        raise ValueError(f"Invalid TOPANEU_FOLDS={raw!r}; expected a subset of 0,1,2,3,4")
    return result


def _resize(array: np.ndarray, target_shape: Iterable[int], order: int) -> np.ndarray:
    target_shape = np.asarray(tuple(target_shape), dtype=int)
    factors = target_shape.astype(float) / np.asarray(array.shape, dtype=float)
    result = zoom(array, factors, order=order, mode="nearest", prefilter=order > 1)
    if tuple(result.shape) == tuple(target_shape):
        return result
    fixed = np.zeros(tuple(target_shape), dtype=result.dtype)
    common = tuple(slice(0, min(a, b)) for a, b in zip(result.shape, target_shape))
    fixed[common] = result[common]
    return fixed


def _expand_bbox(bbox: tuple[int, int, int, int, int, int], shape: tuple[int, ...]):
    lower = np.asarray((bbox[0], bbox[2], bbox[4]), dtype=int)
    upper = np.asarray((bbox[1], bbox[3], bbox[5]), dtype=int)
    extent = np.maximum(upper - lower, 1)
    margin = np.maximum(4, np.ceil(extent * 0.05)).astype(int)
    lower = np.maximum(0, lower - margin)
    upper = np.minimum(np.asarray(shape, dtype=int), upper + margin)
    if np.any(upper <= lower):
        return np.zeros(3, dtype=int), np.asarray(shape, dtype=int)
    return lower, upper


def _predict_bbox(image: np.ndarray, spacing_xyz: tuple[float, ...], device: torch.device):
    if not STAGE1_DIR.is_dir():
        raise FileNotFoundError(f"Stage-1 model not found: {STAGE1_DIR}")
    predictor = ROIPredictor(
        tile_step_size=0.5,
        use_mirroring=False,
        use_gaussian=True,
        perform_everything_on_device=True,
        device=device,
        allow_tqdm=False,
    )
    predictor.initialize_from_trained_model_folder(
        str(STAGE1_DIR), use_folds=(0,), checkpoint_name="checkpoint_final.pth"
    )
    predictor.initialize_network_and_gaussian()
    predictor.list_of_parameters = []
    with torch.inference_mode():
        bbox = predictor.predict_from_multi_axial_slices(
            image[None], np.asarray(spacing_xyz), ROI_TARGET_SPACING,
            max_batch_size=int(os.environ.get("TOPANEU_STAGE1_BATCH", "8")),
        )
    del predictor
    gc.collect()
    torch.cuda.empty_cache()
    return _expand_bbox(bbox, image.shape)


def _run_stage2(roi: np.ndarray, device: torch.device):
    if not STAGE2_DIR.is_dir():
        raise FileNotFoundError(f"Stage-2 model not found: {STAGE2_DIR}")

    roi = _resize(roi, TARGET_SHAPE, order=3).astype(np.float32, copy=False)
    mean = float(roi.mean())
    std = max(float(roi.std()), 1e-8)
    tensor = torch.from_numpy(((roi - mean) / std)[None, None]).to(device)

    vessel_sum = None
    aneurysm_sum = None
    location_sum = None
    folds = _folds()
    for fold in folds:
        predictor = Stage2Predictor(
            tile_step_size=1.0,
            use_gaussian=False,
            use_mirroring=False,
            perform_everything_on_device=True,
            device=device,
            verbose=False,
            allow_tqdm=False,
        )
        predictor.initialize_from_trained_model_folder(
            str(STAGE2_DIR), use_folds=(fold,), checkpoint_name="checkpoint_final.pth"
        )
        predictor.initialize_network_and_gaussian()
        predictor.list_of_parameters = []
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            vessel_logits, aneurysm_logits, cls_logits, _ = predictor.network(tensor)
        current_vessel = vessel_logits[0].float().cpu()
        current_aneurysm = aneurysm_logits[0].float().cpu()
        current_location = cls_logits[1][0].float().cpu()
        vessel_sum = current_vessel if vessel_sum is None else vessel_sum + current_vessel
        aneurysm_sum = current_aneurysm if aneurysm_sum is None else aneurysm_sum + current_aneurysm
        location_sum = current_location if location_sum is None else location_sum + current_location
        del predictor, vessel_logits, aneurysm_logits, cls_logits
        gc.collect()
        torch.cuda.empty_cache()

    pred_vessel = vessel_sum.argmax(0).numpy().astype(np.uint8)
    pred_aneurysm = aneurysm_sum.argmax(0).numpy().astype(np.uint8)
    location_probs = torch.sigmoid(location_sum / len(folds)).numpy()
    return assign_components(pred_aneurysm, pred_vessel, location_probs=location_probs)[0]


def _infer(image: sitk.Image) -> sitk.Image:
    if not torch.cuda.is_available():
        raise RuntimeError("TopAneu submission requires an NVIDIA GPU")
    device = torch.device("cuda:0")
    image_array = sitk.GetArrayFromImage(image).astype(np.float32, copy=False)
    if image_array.ndim != 3:
        raise ValueError(f"Expected a 3-D image, got shape {image_array.shape}")

    lower, upper = _predict_bbox(image_array, image.GetSpacing(), device)
    crop_slices = tuple(slice(int(a), int(b)) for a, b in zip(lower, upper))
    location_roi = _run_stage2(image_array[crop_slices], device)
    restored_roi = _resize(location_roi, upper - lower, order=0).astype(np.uint8, copy=False)
    output_array = np.zeros(image_array.shape, dtype=np.uint8)
    output_array[crop_slices] = restored_roi
    if output_array.min() < 0 or output_array.max() > 52:
        raise RuntimeError("Output labels must be in [0, 52]")

    output = sitk.GetImageFromArray(output_array)
    output.CopyInformation(image)
    return output


def infer_ct(img: sitk.Image) -> sitk.Image:
    return _infer(img)


def infer_mr(img: sitk.Image) -> sitk.Image:
    return _infer(img)
