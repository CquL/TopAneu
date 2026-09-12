"""Stage 2-A trainer: shared encoder plus 36-class vessel decoder only."""

from __future__ import annotations

import json
import os
from typing import List

import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F
from torch import autocast
from torch._dynamo import OptimizedModule

from nnxnet.training.loss.compound_losses import DC_and_CE_loss
from nnxnet.training.loss.deep_supervision import DeepSupervisionWrapper
from nnxnet.training.loss.dice import MemoryEfficientSoftDiceLoss, get_tp_fp_fn_tn
from nnxnet.training.nnXNetTrainer.nnXNetTrainer import nnXNetTrainer
from nnxnet.training.nnXNetTrainer.nnXNetTrainer_TopAneu52 import nnXNetTrainer_TopAneu52
from nnxnet.utilities.helpers import dummy_context


class nnXNetTrainer_TopAneuVesselOnly(nnXNetTrainer_TopAneu52):
    """Optimize only the shared encoder and 36-class vessel branch.

    Aneurysm voxels overwrite vessel IDs in the legacy combined target, so they
    are ignored rather than treated as vessel background in this stage.
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
        # nnXNetTrainer records constructor arguments through signature
        # introspection, so this must remain explicit (no *args/**kwargs).
        super().__init__(
            plans, configuration, fold, dataset_json, unpack_dataset, device
        )
        self.num_epochs = 250
        self.save_every = 10
        # Existing "vessel_weight" values are based on rare aneurysm-location
        # labels, not vessel anatomy. Use uniform case sampling in Stage 2-A.
        self.use_sampling_weight = None

    def configure_rotation_dummyDA_mirroring_and_inital_patch_size(self):
        rotation, dummy_2d, initial_patch, _ = super().configure_rotation_dummyDA_mirroring_and_inital_patch_size()
        self.inference_allowed_mirroring_axes = None
        self.print_to_log_file("Stage 2-A vessel-only: spatial mirroring disabled")
        return rotation, dummy_2d, initial_patch, None

    def _build_loss(self):
        weights_1 = torch.tensor(
            [1.0] + self.seg_ce_class_weights_1, dtype=torch.float32, device=self.device
        )
        weights_2 = torch.tensor(
            [1.0] + self.seg_ce_class_weights_2, dtype=torch.float32, device=self.device
        )

        def make_loss(weights):
            return DC_and_CE_loss(
                {
                    "batch_dice": self.configuration_manager.batch_dice,
                    "smooth": 1e-5,
                    "do_bg": False,
                    "ddp": self.is_ddp,
                },
                {"weight": weights},
                weight_ce=1,
                weight_dice=1,
                ignore_label=-1,
                dice_class=MemoryEfficientSoftDiceLoss,
            )

        loss_1, loss_2 = make_loss(weights_1), make_loss(weights_2)
        if self.enable_deep_supervision:
            scales = self._get_deep_supervision_scales()
            ds_weights = np.asarray([1 / (2**i) for i in range(len(scales))], dtype=float)
            ds_weights[-1] = 1e-6 if self.is_ddp and not self._do_i_compile() else 0
            ds_weights /= ds_weights.sum()
            loss_1 = DeepSupervisionWrapper(loss_1, ds_weights)
            loss_2 = DeepSupervisionWrapper(loss_2, ds_weights)
        return loss_1, loss_2

    def initialize(self):
        super().initialize()
        module = self.network._orig_mod if isinstance(self.network, OptimizedModule) else self.network
        frozen_names = (
            "transpconvs_last_two_2",
            "decoder_blocks_last_two_2",
            "seg_layers_2",
            "cls_head_list",
            "cls_modality_head",
        )
        for name in frozen_names:
            for parameter in getattr(module, name).parameters():
                parameter.requires_grad_(False)
        trainable = sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)
        frozen = sum(parameter.numel() for parameter in module.parameters() if not parameter.requires_grad)
        self.print_to_log_file(
            f"Stage 2-A parameters: trainable={trainable:,}, frozen={frozen:,}"
        )

    def merge_target_labels(self, target, head_name=None):
        if head_name != "seg_index_1":
            return super().merge_target_labels(target, head_name)
        target = target.long()
        merged = torch.where(
            (target >= 1) & (target <= 36), target, torch.zeros_like(target)
        )
        ignore = (target >= 37) | (target == -1)
        return torch.where(ignore, torch.full_like(merged, -1), merged)

    def train_step(self, batch: dict) -> dict:
        data = batch["data"].to(self.device, non_blocking=True)
        target = batch["target"]
        target = (
            [item.to(self.device, non_blocking=True) for item in target]
            if isinstance(target, list)
            else target.to(self.device, non_blocking=True)
        )
        self.optimizer.zero_grad(set_to_none=True)
        context = autocast(self.device.type, enabled=True) if self.device.type == "cuda" else dummy_context()
        with context:
            output_1 = self.network(data, vessel_only=True)
            merged = (
                [self.merge_target_labels(item, "seg_index_1") for item in target]
                if isinstance(target, list)
                else self.merge_target_labels(target, "seg_index_1")
            )
            loss = self.seg_loss_1(output_1, merged)

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
        context = autocast(self.device.type, enabled=True) if self.device.type == "cuda" else dummy_context()
        with context:
            output_1 = self.network(data, vessel_only=True)
            merged = (
                [self.merge_target_labels(item, "seg_index_1") for item in target]
                if isinstance(target, list)
                else self.merge_target_labels(target, "seg_index_1")
            )
            loss = self.seg_loss_1(output_1, merged)

        output = output_1[0] if self.enable_deep_supervision else output_1
        target_main = merged[0] if self.enable_deep_supervision else merged
        axes = [0] + list(range(2, output.ndim))
        valid_mask = (target_main != -1).float()
        target_for_metric = target_main.clone()
        target_for_metric[target_for_metric == -1] = 0
        output_seg = output.argmax(1)[:, None]
        onehot = torch.zeros(output.shape, device=output.device, dtype=torch.float32)
        onehot.scatter_(1, output_seg, 1)
        tp, fp, fn, _ = get_tp_fp_fn_tn(
            onehot, target_for_metric, axes=axes, mask=valid_mask
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

    @torch.inference_mode()
    def perform_actual_validation(self, save_probabilities: bool = False):
        del save_probabilities
        self.set_deep_supervision_enabled(False)
        self.network.eval()
        _, val_keys = self.do_split()
        dataset_val = self.get_tr_and_val_datasets()[1]
        output_dir = os.path.join(self.output_folder, "validation_vessel_masks")
        os.makedirs(output_dir, exist_ok=True)
        raw_dataset = os.path.join(os.environ["nnXNet_raw"], self.plans_manager.dataset_name)
        totals = np.zeros((36, 3), dtype=np.int64)

        for case in val_keys:
            data, _, properties = dataset_val.load_case(case)
            spatial = np.asarray(data.shape[1:], dtype=int)
            patch = np.asarray(self.configuration_manager.patch_size, dtype=int)
            before = ((patch - spatial) // 2).astype(int)
            after = patch - spatial - before
            padding = []
            for lower, upper in zip(before[::-1], after[::-1]):
                padding.extend([int(lower), int(upper)])
            tensor = F.pad(torch.from_numpy(data[None]).to(self.device), padding)
            with autocast(self.device.type, enabled=self.device.type == "cuda"):
                output_1 = self.network(tensor, vessel_only=True)
            prediction = output_1.argmax(1)[0].cpu().numpy().astype(np.uint8)
            slicer = tuple(
                slice(int(lower), int(lower + size)) for lower, size in zip(before, spatial)
            )
            prediction = prediction[slicer]
            full = np.zeros(tuple(properties["shape_before_cropping"]), dtype=np.uint8)
            bbox = tuple(slice(int(a), int(b)) for a, b in properties["bbox_used_for_cropping"])
            full[bbox] = prediction

            reference = sitk.ReadImage(
                os.path.join(raw_dataset, "imagesTr", f"{case}_0000.nii.gz")
            )
            combined = sitk.GetArrayFromImage(
                sitk.ReadImage(os.path.join(raw_dataset, "labelsTr", f"{case}.nii.gz"))
            )
            valid = combined <= 36
            gt = np.where((combined >= 1) & (combined <= 36), combined, 0)
            for class_id in range(1, 37):
                pred_mask = full == class_id
                gt_mask = gt == class_id
                totals[class_id - 1] += (
                    np.count_nonzero(pred_mask & gt_mask),
                    np.count_nonzero(pred_mask & ~gt_mask & valid),
                    np.count_nonzero(~pred_mask & gt_mask),
                )
            output = sitk.GetImageFromArray(full)
            output.CopyInformation(reference)
            sitk.WriteImage(output, os.path.join(output_dir, f"{case}.nii.gz"), True)

        per_class = {}
        dice_values = []
        for class_id, (tp, fp, fn) in enumerate(totals, start=1):
            denominator = 2 * tp + fp + fn
            dice = float(2 * tp / denominator) if denominator else None
            if dice is not None:
                dice_values.append(dice)
            per_class[str(class_id)] = {
                "tp_voxels": int(tp),
                "fp_voxels": int(fp),
                "fn_voxels": int(fn),
                "dice": dice,
            }
        summary = {
            "macro_dice_present_classes": float(np.mean(dice_values)) if dice_values else None,
            "zero_dice_classes": [
                int(class_id) for class_id, values in per_class.items() if values["dice"] == 0
            ],
            "per_class": per_class,
        }
        with open(os.path.join(self.output_folder, "validation_vessel_summary.json"), "w") as handle:
            json.dump(summary, handle, indent=2)
        self.print_to_log_file(f"Vessel-only validation complete: {summary['macro_dice_present_classes']}")
        self.set_deep_supervision_enabled(True)
