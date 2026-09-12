"""TopAneu Task-2 multi-stage inference."""
from __future__ import annotations

import gc
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import SimpleITK as sitk
import torch
from scipy import ndimage
from scipy.ndimage import zoom

from nnxnet.inference.predict_from_raw_data_2D_orthogonal_planes_fast import nnXNetPredictor as ROIPredictor
from nnxnet.inference.predict_from_raw_data_two_seg_with_cls_no_seg_return_no_filter import nnXNetPredictor as Stage2Predictor
from topaneu.stagec.train_stagec import StageCClassifier, extract_component_features

APP_DIR = Path(__file__).resolve().parent
MODEL_ROOT = Path(os.environ.get("TOPANEU_MODEL_DIR", "/opt/ml/model"))
if not MODEL_ROOT.is_dir():
    MODEL_ROOT = APP_DIR / "model"
STAGE1_DIR = MODEL_ROOT / "stage1_roi" / "Dataset180_2D_vessel_box_seg_stable" / "nnUNetTrainer__nnUNetPlans__2d"
VESSEL_DIR = MODEL_ROOT / "stage2_vessel"
ANEURYSM_DIR = MODEL_ROOT / "stage2_aneurysm"
STAGEC_DIR = MODEL_ROOT / "stagec"
TARGET_SHAPE = np.asarray((160, 160, 160), dtype=int)
ROI_TARGET_SPACING = np.asarray((1.0, 0.55, 0.5), dtype=float)


def _folds():
    raw = os.environ.get("TOPANEU_FOLDS", "0,1,2,3,4")
    folds = tuple(int(x.strip()) for x in raw.split(",") if x.strip())
    if not folds or any(x not in range(5) for x in folds):
        raise ValueError(f"Invalid TOPANEU_FOLDS={raw!r}")
    return folds


def _resize(array: np.ndarray, target_shape: Iterable[int], order: int) -> np.ndarray:
    target = np.asarray(tuple(target_shape), dtype=int)
    result = zoom(array, target.astype(float) / np.asarray(array.shape), order=order, mode="nearest", prefilter=order > 1)
    if tuple(result.shape) == tuple(target):
        return result
    fixed = np.zeros(tuple(target), dtype=result.dtype)
    fixed[tuple(slice(0, min(a, b)) for a, b in zip(result.shape, target))] = result[tuple(slice(0, min(a, b)) for a, b in zip(result.shape, target))]
    return fixed


def _expand_bbox(bbox, shape):
    lower = np.asarray((bbox[0], bbox[2], bbox[4]), dtype=int)
    upper = np.asarray((bbox[1], bbox[3], bbox[5]), dtype=int)
    margin = np.maximum(4, np.ceil((upper - lower) * 0.05)).astype(int)
    return np.maximum(0, lower - margin), np.minimum(np.asarray(shape, dtype=int), upper + margin)


def _predict_bbox(image, spacing_xyz, device):
    predictor = ROIPredictor(tile_step_size=0.5, use_mirroring=False, use_gaussian=True, perform_everything_on_device=True, device=device, allow_tqdm=False)
    predictor.initialize_from_trained_model_folder(str(STAGE1_DIR), use_folds=(0,), checkpoint_name="checkpoint_final.pth")
    predictor.initialize_network_and_gaussian()
    with torch.inference_mode():
        bbox = predictor.predict_from_multi_axial_slices(image[None], np.asarray(spacing_xyz), ROI_TARGET_SPACING, max_batch_size=8)
    del predictor
    gc.collect(); torch.cuda.empty_cache()
    return _expand_bbox(bbox, image.shape)


def _load_stagec(fold, device):
    state = torch.load(STAGEC_DIR / f"fold_{fold}" / "checkpoint_best_stagec.pth", map_location=device, weights_only=False)
    model = StageCClassifier().to(device); model.load_state_dict(state["model"], strict=True); model.eval()
    return model, state["mean"].astype(np.float32), np.maximum(state["std"].astype(np.float32), 1e-6), torch.as_tensor(state["seen_location_mask"], dtype=torch.bool, device=device)


def _run_stage2(roi: np.ndarray, modality: str, device: torch.device) -> np.ndarray:
    roi = _resize(roi, TARGET_SHAPE, 3).astype(np.float32, copy=False)
    tensor = torch.from_numpy(((roi - roi.mean()) / max(float(roi.std()), 1e-8))[None, None]).to(device)
    vessel_sum = None; aneurysm_sum = None; folds = _folds()
    for fold in folds:
        predictor = Stage2Predictor(tile_step_size=1.0, use_gaussian=False, use_mirroring=False, perform_everything_on_device=True, device=device, verbose=False, allow_tqdm=False)
        predictor.initialize_from_trained_model_folder(str(VESSEL_DIR), use_folds=(fold,), checkpoint_name="checkpoint_best.pth")
        predictor.initialize_network_and_gaussian()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            vessel_logits = predictor.network(tensor, vessel_only=True)
        vessel_sum = vessel_logits.float() if vessel_sum is None else vessel_sum + vessel_logits.float()
        state = torch.load(ANEURYSM_DIR / f"fold_{fold}" / "checkpoint_best.pth", map_location=device, weights_only=False)
        predictor.network.load_state_dict(state["network_weights"], strict=True); del state
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            aneurysm_logits = predictor.network(tensor, aneurysm_only=True)
        aneurysm_sum = aneurysm_logits.float() if aneurysm_sum is None else aneurysm_sum + aneurysm_logits.float()
        del predictor, vessel_logits, aneurysm_logits; gc.collect(); torch.cuda.empty_cache()
    vessel = vessel_sum.div(len(folds)).argmax(1)[0].cpu().numpy().astype(np.uint8)
    probability = torch.softmax(aneurysm_sum.div(len(folds)), dim=1)[0, 1].cpu().numpy()
    threshold = float(os.environ.get("TOPANEU_ANEURYSM_THRESHOLD", "0.10"))
    binary = probability >= threshold
    cc, count = ndimage.label(binary, ndimage.generate_binary_structure(3, 2))
    result = np.zeros(TARGET_SHAPE, dtype=np.uint8)
    for component_id in range(1, count + 1):
        component = cc == component_id
        if int(component.sum()) < int(os.environ.get("TOPANEU_MIN_COMPONENT_VOXELS", "1")):
            continue
        case_name = "topaneu_center2_ct_dummy" if modality == "ct" else "topaneu_center1_mr_dummy"
        features = extract_component_features(roi, component, vessel, case_name)
        x = torch.as_tensor(features[None], device=device)
        predictions = []
        with torch.inference_mode():
            for fold in folds:
                model, mean, std, seen = _load_stagec(fold, device)
                logits = model((x - torch.as_tensor(mean, device=device)) / torch.as_tensor(std, device=device))["location"]
                logits[:, ~seen] = -1e9; predictions.append(logits)
                del model
            location = torch.stack(predictions).mean(0).argmax(1).item() + 1
        result[component] = np.uint8(location)
    return result


def _infer(image: sitk.Image, modality: str) -> sitk.Image:
    if not torch.cuda.is_available():
        raise RuntimeError("TopAneu Task 2 requires an NVIDIA GPU")
    device = torch.device("cuda:0")
    array = sitk.GetArrayFromImage(image).astype(np.float32, copy=False)
    lower, upper = _predict_bbox(array, image.GetSpacing(), device)
    crop = array[tuple(slice(int(a), int(b)) for a, b in zip(lower, upper))]
    prediction = _run_stage2(crop, modality, device)
    restored = _resize(prediction, upper - lower, 0).astype(np.uint8, copy=False)
    output_array = np.zeros(array.shape, dtype=np.uint8)
    output_array[tuple(slice(int(a), int(b)) for a, b in zip(lower, upper))] = restored
    output = sitk.GetImageFromArray(output_array); output.CopyInformation(image)
    return output


def infer_ct(img: sitk.Image) -> sitk.Image:
    return _infer(img, "ct")


def infer_mr(img: sitk.Image) -> sitk.Image:
    return _infer(img, "mr")
