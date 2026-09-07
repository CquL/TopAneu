"""Stage 2-B trainer: binary aneurysm decoder on frozen vessel features."""

from __future__ import annotations

import json
import os
from typing import List

import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F
from scipy import ndimage
from torch import autocast
from torch._dynamo import OptimizedModule

from nnxnet.training.loss.dice import get_tp_fp_fn_tn
from nnxnet.training.nnXNetTrainer.nnXNetTrainer import nnXNetTrainer
from nnxnet.training.nnXNetTrainer.nnXNetTrainer_TopAneu52 import nnXNetTrainer_TopAneu52
from nnxnet.utilities.helpers import dummy_context


class nnXNetTrainer_TopAneuAneurysmOnly(nnXNetTrainer_TopAneu52):
    """Train background-versus-aneurysm segmentation only.

    The encoder, shared decoder and vessel decoder are frozen in this first
    binary stage. This preserves the 36-class anatomy model while adapting the
    pre-existing second decoder to the merged aneurysm target.
    """

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        unpack_dataset: bool = True,
        device: torch.device = torch.device("cuda"),
    ):
        # Keep an explicit signature because nnXNet records these arguments.
        super().__init__(
            plans, configuration, fold, dataset_json, unpack_dataset, device
        )
        self.num_epochs = 250
        self.save_every = 10
        # This legacy-named column is computed from TopAneu aneurysm-location
        # prevalence and therefore intentionally upweights rare positive cases.
        self.use_sampling_weight = "vessel"

    def configure_rotation_dummyDA_mirroring_and_inital_patch_size(self):
        rotation, dummy_2d, initial_patch, _ = (
            super().configure_rotation_dummyDA_mirroring_and_inital_patch_size()
        )
        self.inference_allowed_mirroring_axes = None
        self.print_to_log_file(
            "Stage 2-B binary-aneurysm: spatial mirroring disabled"
        )
        return rotation, dummy_2d, initial_patch, None

    def initialize(self):
        super().initialize()
        module = (
            self.network._orig_mod
            if isinstance(self.network, OptimizedModule)
            else self.network
        )
        frozen_names = (
            "conv_encoder_blocks",
            "transpconvs",
            "decoder_blocks",
            "transpconvs_last_two_1",
            "decoder_blocks_last_two_1",
            "seg_layers_1",
            "cls_head_list",
            "cls_modality_head",
        )
        for name in frozen_names:
            for parameter in getattr(module, name).parameters():
                parameter.requires_grad_(False)
        trainable = sum(
            parameter.numel()
            for parameter in module.parameters()
            if parameter.requires_grad
        )
        frozen = sum(
            parameter.numel()
            for parameter in module.parameters()
            if not parameter.requires_grad
        )
        self.print_to_log_file(
            f"Stage 2-B parameters: trainable={trainable:,}, frozen={frozen:,}"
        )

    def train_step(self, batch: dict) -> dict:
        data = batch["data"].to(self.device, non_blocking=True)
        target = batch["target"]
        target = (
            [item.to(self.device, non_blocking=True) for item in target]
            if isinstance(target, list)
            else target.to(self.device, non_blocking=True)
        )
        self.optimizer.zero_grad(set_to_none=True)
        context = (
            autocast(self.device.type, enabled=True)
            if self.device.type == "cuda"
            else dummy_context()
        )
        with context:
            output = self.network(data, aneurysm_only=True)
            merged = (
                [self.merge_target_labels(item, "seg_index_2") for item in target]
                if isinstance(target, list)
                else self.merge_target_labels(target, "seg_index_2")
            )
            loss = self.seg_loss_2(output, merged)

        if self.grad_scaler is not None:
            self.grad_scaler.scale(loss).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()
        return {"loss": loss.detach().cpu().numpy()}

    def validation_step(self, batch: dict) -> dict:
        data = batch["data"].to(self.device, non_blocking=True)
        target = batch["target"]
        target = (
            [item.to(self.device, non_blocking=True) for item in target]
            if isinstance(target, list)
            else target.to(self.device, non_blocking=True)
        )
        context = (
            autocast(self.device.type, enabled=True)
            if self.device.type == "cuda"
            else dummy_context()
        )
        with context:
            output = self.network(data, aneurysm_only=True)
            merged = (
                [self.merge_target_labels(item, "seg_index_2") for item in target]
                if isinstance(target, list)
                else self.merge_target_labels(target, "seg_index_2")
            )
            loss = self.seg_loss_2(output, merged)

        output_main = output[0] if self.enable_deep_supervision else output
        target_main = merged[0] if self.enable_deep_supervision else merged
        axes = [0] + list(range(2, output_main.ndim))
        valid_mask = (target_main != -1).float()
        metric_target = target_main.clone()
        metric_target[metric_target == -1] = 0
        output_seg = output_main.argmax(1)[:, None]
        onehot = torch.zeros(
            output_main.shape, device=output_main.device, dtype=torch.float32
        )
        onehot.scatter_(1, output_seg, 1)
        tp, fp, fn, _ = get_tp_fp_fn_tn(
            onehot, metric_target, axes=axes, mask=valid_mask
        )
        return {
            "loss": loss.detach().cpu().numpy(),
            "tp_hard": tp.detach().cpu().numpy()[1:],
            "fp_hard": fp.detach().cpu().numpy()[1:],
            "fn_hard": fn.detach().cpu().numpy()[1:],
        }

    def on_train_epoch_end(self, outputs: List[dict]):
        return nnXNetTrainer.on_train_epoch_end(self, outputs)

    def on_validation_epoch_end(self, outputs: List[dict]):
        return nnXNetTrainer.on_validation_epoch_end(self, outputs)

    def on_epoch_end(self):
        return nnXNetTrainer.on_epoch_end(self)

    @staticmethod
    def _hd95(prediction: np.ndarray, target: np.ndarray, spacing_zyx):
        if not prediction.any() or not target.any():
            return None
        structure = ndimage.generate_binary_structure(3, 1)
        pred_surface = prediction & ~ndimage.binary_erosion(
            prediction, structure=structure, border_value=0
        )
        target_surface = target & ~ndimage.binary_erosion(
            target, structure=structure, border_value=0
        )
        distance_to_target = ndimage.distance_transform_edt(
            ~target_surface, sampling=spacing_zyx
        )
        distance_to_pred = ndimage.distance_transform_edt(
            ~pred_surface, sampling=spacing_zyx
        )
        distances = np.concatenate(
            (distance_to_target[pred_surface], distance_to_pred[target_surface])
        )
        return float(np.percentile(distances, 95))

    @torch.inference_mode()
    def perform_actual_validation(self, save_probabilities: bool = False):
        del save_probabilities
        self.set_deep_supervision_enabled(False)
        self.network.eval()
        _, val_keys = self.do_split()
        dataset_val = self.get_tr_and_val_datasets()[1]
        output_dir = os.path.join(self.output_folder, "validation_aneurysm_masks")
        os.makedirs(output_dir, exist_ok=True)
        raw_dataset = os.path.join(
            os.environ["nnXNet_raw"], self.plans_manager.dataset_name
        )
        tp = fp = fn = 0
        gt_lesions = detected_lesions = pred_components = fp_components = 0
        finite_hd95 = []
        empty_mismatch_cases = 0
        structure = ndimage.generate_binary_structure(3, 2)

        for case in val_keys:
            data, _, properties = dataset_val.load_case(case)
            spatial = np.asarray(data.shape[1:], dtype=int)
            patch = np.asarray(self.configuration_manager.patch_size, dtype=int)
            if np.any(spatial > patch):
                raise RuntimeError(
                    f"{case}: preprocessed ROI {spatial} exceeds patch {patch}"
                )
            before = ((patch - spatial) // 2).astype(int)
            after = (patch - spatial - before).astype(int)
            padding = []
            for lower, upper in zip(before[::-1], after[::-1]):
                padding.extend([int(lower), int(upper)])
            tensor = F.pad(
                torch.from_numpy(data[None]).to(self.device), padding
            )
            with autocast(
                self.device.type, enabled=self.device.type == "cuda"
            ):
                output = self.network(tensor, aneurysm_only=True)
            prediction = output.argmax(1)[0].cpu().numpy() == 1
            slicer = tuple(
                slice(int(lower), int(lower + size))
                for lower, size in zip(before, spatial)
            )
            prediction = prediction[slicer]
            full = np.zeros(
                tuple(properties["shape_before_cropping"]), dtype=bool
            )
            bbox = tuple(
                slice(int(a), int(b))
                for a, b in properties["bbox_used_for_cropping"]
            )
            full[bbox] = prediction

            reference = sitk.ReadImage(
                os.path.join(raw_dataset, "imagesTr", f"{case}_0000.nii.gz")
            )
            combined = sitk.GetArrayFromImage(
                sitk.ReadImage(
                    os.path.join(raw_dataset, "labelsTr", f"{case}.nii.gz")
                )
            )
            target = combined >= 37
            tp += int(np.count_nonzero(full & target))
            fp += int(np.count_nonzero(full & ~target))
            fn += int(np.count_nonzero(~full & target))

            gt_cc, gt_n = ndimage.label(target, structure=structure)
            pred_cc, pred_n = ndimage.label(full, structure=structure)
            gt_lesions += int(gt_n)
            pred_components += int(pred_n)
            for component_id in range(1, gt_n + 1):
                detected_lesions += int(np.any(full[gt_cc == component_id]))
            for component_id in range(1, pred_n + 1):
                fp_components += int(not np.any(target[pred_cc == component_id]))

            if full.any() and target.any():
                spacing_zyx = tuple(float(v) for v in reference.GetSpacing()[::-1])
                finite_hd95.append(self._hd95(full, target, spacing_zyx))
            elif full.any() != target.any():
                empty_mismatch_cases += 1

            image = sitk.GetImageFromArray(full.astype(np.uint8))
            image.CopyInformation(reference)
            sitk.WriteImage(
                image, os.path.join(output_dir, f"{case}.nii.gz"), True
            )

        dice_denominator = 2 * tp + fp + fn
        summary = {
            "validation_cases": len(val_keys),
            "tp_voxels": tp,
            "fp_voxels": fp,
            "fn_voxels": fn,
            "dice": float(2 * tp / dice_denominator)
            if dice_denominator
            else None,
            "precision": float(tp / (tp + fp)) if tp + fp else None,
            "recall": float(tp / (tp + fn)) if tp + fn else None,
            "gt_lesions": gt_lesions,
            "detected_lesions": detected_lesions,
            "lesion_recall": float(detected_lesions / gt_lesions)
            if gt_lesions
            else None,
            "predicted_components": pred_components,
            "false_positive_components": fp_components,
            "hd95_finite_case_mean_mm": float(np.mean(finite_hd95))
            if finite_hd95
            else None,
            "hd95_finite_cases": len(finite_hd95),
            "hd95_empty_mismatch_cases": empty_mismatch_cases,
        }
        with open(
            os.path.join(self.output_folder, "validation_aneurysm_summary.json"),
            "w",
        ) as handle:
            json.dump(summary, handle, indent=2)
        self.print_to_log_file(
            f"Binary-aneurysm validation complete: {summary}"
        )
        self.set_deep_supervision_enabled(True)
