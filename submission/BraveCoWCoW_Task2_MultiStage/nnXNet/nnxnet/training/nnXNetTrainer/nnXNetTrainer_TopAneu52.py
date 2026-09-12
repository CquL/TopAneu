"""TopAneu-26 adaptation of the BraveCoWCoW multi-task trainer."""

import os
import json
from typing import Tuple

import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F

from nnxnet.training.dataloading.data_loader_3d_with_global_cls import (
    nnXNetDataLoader3DWithGlobalCls,
)
from nnxnet.training.nnXNetTrainer.nnXNetTrainer_ResEncoderUNet_two_seg_with_cls_modality import (
    nnXNetTrainer_ResEncoderUNet_two_seg_with_cls_modality,
)
from topaneu.component_location import assign_components


class nnXNetTrainer_TopAneu52(nnXNetTrainer_ResEncoderUNet_two_seg_with_cls_modality):
    """Two decoders + presence/location/modality heads for 52 TopAneu locations.

    All dataset-specific inputs are passed via environment variables so this
    trainer never depends on the author's original absolute RSNA paths.
    """

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 250
        self.task1_categories = ['Aneurysm Present']
        labels_by_value = sorted(
            ((value, name) for name, value in dataset_json['topaneu_location_labels'].items()),
            key=lambda item: item[0],
        )
        self.task2_categories = [name for _, name in labels_by_value]
        self.cls_modality_categories = ['CTA', 'MRA']
        self.use_sampling_weight = 'vessel'

        self.metadata_csv = os.environ.get('TOPANEU_BRAVECOWCOW_METADATA')
        self.sampling_weights_csv = os.environ.get('TOPANEU_BRAVECOWCOW_SAMPLING_WEIGHTS')
        if not self.metadata_csv or not os.path.isfile(self.metadata_csv):
            raise RuntimeError(
                'TOPANEU_BRAVECOWCOW_METADATA is missing or invalid. '
                'Source topaneu_bravecowcow_env.sh before training.'
            )
        if not self.sampling_weights_csv:
            self.sampling_weights_csv = os.path.join(
                os.path.dirname(self.metadata_csv), 'train_case_sampling_weight.csv')

    def merge_target_labels(self, target, head_name=None):
        """Convert the legacy combined target into dense vessel/binary targets."""
        target = target.long()
        if head_name == 'seg_index_1':
            merged = torch.where((target >= 1) & (target <= 36), target,
                                 torch.zeros_like(target))
        elif head_name == 'seg_index_2':
            merged = (target >= 37).long()
        else:
            return target.clone()
        # Preserve padded ignore voxels used by the batch loader.
        return torch.where(target == -1, torch.full_like(merged, -1), merged)

    def get_plain_dataloaders(self, initial_patch_size: Tuple[int, ...], dim: int):
        if dim != 3:
            raise NotImplementedError('TopAneu BraveCoWCoW supports 3D only')

        dataset_tr, dataset_val = self.get_tr_and_val_datasets()
        common = dict(
            label_manager=self.label_manager,
            oversample_foreground_percent=self.oversample_foreground_percent,
            sampling_probabilities=None,
            pad_sides=None,
            csv_path=self.metadata_csv,
            case_sampling_weight_path=self.sampling_weights_csv,
            label_columns=self.task2_categories,
            modality_mapping={'CTA': 0, 'MRA': 1},
        )
        dl_tr = nnXNetDataLoader3DWithGlobalCls(
            dataset_tr, self.batch_size, initial_patch_size,
            self.configuration_manager.patch_size,
            use_sampling_weight=self.use_sampling_weight, **common)
        dl_val = nnXNetDataLoader3DWithGlobalCls(
            dataset_val, self.batch_size, self.configuration_manager.patch_size,
            self.configuration_manager.patch_size,
            use_sampling_weight=None, **common)
        return dl_tr, dl_val

    @torch.inference_mode()
    def perform_actual_validation(self, save_probabilities: bool = False):
        """Export real 52-class ROI masks instead of using the single-head predictor.

        The upstream predictor assumes one segmentation decoder and cannot unpack
        this repository's four network outputs. TopAneu validation is one 224^3
        patch per case, so direct whole-ROI inference is both simpler and exact.
        """
        del save_probabilities  # probabilities are intentionally not persisted (very large)
        self.set_deep_supervision_enabled(False)
        self.network.eval()
        _, val_keys = self.do_split()
        dataset_val = self.get_tr_and_val_datasets()[1]
        output_dir = os.path.join(self.output_folder, 'validation_location_masks_component')
        vessel_dir = os.path.join(self.output_folder, 'validation_vessel_masks')
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(vessel_dir, exist_ok=True)

        raw_dataset = os.path.join(os.environ['nnXNet_raw'], self.plans_manager.dataset_name)
        totals = {
            'component': np.zeros((52, 3), dtype=np.int64),
        }  # tp, fp, fn

        for case in val_keys:
            data, _, properties = dataset_val.load_case(case)
            spatial = np.asarray(data.shape[1:], dtype=int)
            patch = np.asarray(self.configuration_manager.patch_size, dtype=int)
            if np.any(spatial > patch):
                raise RuntimeError(f'{case}: preprocessed ROI {spatial} exceeds patch {patch}')
            before = ((patch - spatial) // 2).astype(int)
            after = (patch - spatial - before).astype(int)
            padding = []
            for lo, hi in zip(before[::-1], after[::-1]):
                padding.extend([int(lo), int(hi)])
            tensor = torch.from_numpy(data[None]).to(self.device, non_blocking=True)
            tensor = F.pad(tensor, padding)
            with torch.autocast(self.device.type, enabled=self.device.type == 'cuda'):
                output_1, output_2, cls_pred_list, _ = self.network(tensor)
            pred_vessel = output_1.argmax(1)[0].cpu().numpy().astype(np.uint8)
            pred_aneurysm = (output_2.argmax(1)[0].cpu().numpy() == 1)
            location_probs = torch.sigmoid(cls_pred_list[1])[0].float().cpu().numpy()
            slicer = tuple(slice(int(lo), int(lo + size)) for lo, size in zip(before, spatial))
            pred_vessel = pred_vessel[slicer]
            pred_aneurysm = pred_aneurysm[slicer]
            pred_location, assignments = assign_components(
                pred_aneurysm, pred_vessel, location_probs=location_probs)

            shape_before = tuple(int(v) for v in properties['shape_before_cropping'])
            bbox = properties['bbox_used_for_cropping']
            bbox_slicer = tuple(slice(int(v[0]), int(v[1])) for v in bbox)
            pred_full = np.zeros(shape_before, dtype=np.uint8)
            vessel_full = np.zeros(shape_before, dtype=np.uint8)
            pred_full[bbox_slicer] = pred_location
            vessel_full[bbox_slicer] = pred_vessel

            reference = sitk.ReadImage(os.path.join(raw_dataset, 'imagesTr', f'{case}_0000.nii.gz'))
            gt_combined = sitk.GetArrayFromImage(sitk.ReadImage(
                os.path.join(raw_dataset, 'labelsTr', f'{case}.nii.gz')))
            gt_location = np.where(gt_combined > 36, gt_combined - 36, 0).astype(np.uint8)
            for name, prediction in (('component', pred_full),):
                for class_id in range(1, 53):
                    pred_mask = prediction == class_id
                    gt_mask = gt_location == class_id
                    totals[name][class_id - 1] += (
                        np.count_nonzero(pred_mask & gt_mask),
                        np.count_nonzero(pred_mask & ~gt_mask),
                        np.count_nonzero(~pred_mask & gt_mask),
                    )
            with open(os.path.join(output_dir, f'{case}.json'), 'w') as handle:
                json.dump(assignments, handle, indent=2)
            for folder, prediction in ((output_dir, pred_full), (vessel_dir, vessel_full)):
                out = sitk.GetImageFromArray(prediction)
                out.CopyInformation(reference)
                sitk.WriteImage(out, os.path.join(folder, f'{case}.nii.gz'), True)

        summary = {}
        for name, values in totals.items():
            per_class = {}
            valid_dice = []
            for class_id, (tp, fp, fn) in enumerate(values, 1):
                denominator = 2 * tp + fp + fn
                dice = float(2 * tp / denominator) if denominator else None
                precision = float(tp / (tp + fp)) if tp + fp else None
                recall = float(tp / (tp + fn)) if tp + fn else None
                if dice is not None:
                    valid_dice.append(dice)
                per_class[str(class_id)] = {
                    'tp_voxels': int(tp), 'fp_voxels': int(fp), 'fn_voxels': int(fn),
                    'dice': dice, 'precision': precision, 'recall': recall,
                }
            summary[name] = {
                'macro_dice_present_classes': float(np.mean(valid_dice)) if valid_dice else None,
                'per_class': per_class,
            }
        with open(os.path.join(self.output_folder, 'validation_location_summary.json'), 'w') as handle:
            json.dump(summary, handle, indent=2)
        self.print_to_log_file(f'TopAneu location-mask validation complete: {summary}')
        self.set_deep_supervision_enabled(True)
