#!/usr/bin/env python3
"""Build an isolated, paper-compatible TopAneu dataset for BraveCoWCoW.

Stage 2 in the original solution consumes a vessel ROI resized to 224^3. For
training we derive that ROI from the supplied TopAneu vessel mask, resize image
and masks together, and encode a single nnXNet target as:

  1..36  : vessel anatomy
  37..88 : aneurysm location (36 + original TopAneu location 1..52)
"""

import argparse
import csv
import json
import os
import shutil
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy.ndimage import zoom
from sklearn.model_selection import StratifiedGroupKFold


DATASET_NAME = 'Dataset504_TopAneuBraveCoWCoW128'
PLANS_NAME = 'TopAneuBraveCoWCoW128Plans'
TARGET_SHAPE = np.asarray((128, 128, 128), dtype=int)  # z, y, x


def load_labels(path: Path):
    labels = json.loads(path.read_text())['labels']
    return {name: int(value) for name, value in labels.items() if int(value) != 0}


def resize_array(array, target_shape, order):
    factors = np.asarray(target_shape, dtype=float) / np.asarray(array.shape, dtype=float)
    result = zoom(array, factors, order=order, mode='nearest', prefilter=order > 1)
    # scipy rounding can differ by one voxel on unusual shapes.
    fixed = np.zeros(tuple(target_shape), dtype=result.dtype)
    common = tuple(slice(0, min(a, b)) for a, b in zip(result.shape, target_shape))
    fixed[common] = result[common]
    return fixed


def write_nifti(array, path, pixel_id=None):
    image = sitk.GetImageFromArray(array)
    image.SetSpacing((1.0, 1.0, 1.0))
    image.SetOrigin((0.0, 0.0, 0.0))
    image.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
    if pixel_id is not None:
        image = sitk.Cast(image, pixel_id)
    sitk.WriteImage(image, str(path), True)


def crop_bounds(vessel, location, margin_fraction):
    foreground = (vessel > 0) | (location > 0)
    coordinates = np.argwhere(foreground)
    if coordinates.size == 0:
        return np.zeros(3, dtype=int), np.asarray(vessel.shape, dtype=int)
    lower = coordinates.min(0)
    upper = coordinates.max(0) + 1
    margin = np.maximum(4, np.ceil((upper - lower) * margin_fraction)).astype(int)
    return np.maximum(0, lower - margin), np.minimum(vessel.shape, upper + margin)


def prepare_case(image_path, release, images_tr, labels_tr, roi_meta, margin_fraction, force):
    case = image_path.name.removesuffix('_0000.nii.gz')
    output_image = images_tr / f'{case}_0000.nii.gz'
    output_label = labels_tr / f'{case}.nii.gz'
    output_meta = roi_meta / f'{case}.json'
    if not force and output_image.is_file() and output_label.is_file() and output_meta.is_file():
        return case

    image_itk = sitk.ReadImage(str(image_path))
    vessel_itk = sitk.ReadImage(str(release / 'vessel_masks' / f'{case}.nii.gz'))
    location_itk = sitk.ReadImage(str(release / 'location_masks' / f'{case}.nii.gz'))
    if image_itk.GetSize() != vessel_itk.GetSize() or image_itk.GetSize() != location_itk.GetSize():
        raise ValueError(f'Geometry size mismatch for {case}')

    image = sitk.GetArrayFromImage(image_itk).astype(np.float32, copy=False)
    vessel = sitk.GetArrayFromImage(vessel_itk).astype(np.uint8, copy=False)
    location = sitk.GetArrayFromImage(location_itk).astype(np.uint8, copy=False)
    lower, upper = crop_bounds(vessel, location, margin_fraction)
    slices = tuple(slice(int(a), int(b)) for a, b in zip(lower, upper))

    image_roi = resize_array(image[slices], TARGET_SHAPE, order=3).astype(np.float32, copy=False)
    vessel_roi = resize_array(vessel[slices], TARGET_SHAPE, order=0).astype(np.uint8, copy=False)
    location_roi = resize_array(location[slices], TARGET_SHAPE, order=0).astype(np.uint8, copy=False)
    target = vessel_roi
    aneurysm = location_roi > 0
    target[aneurysm] = 36 + location_roi[aneurysm]

    write_nifti(image_roi, output_image, sitk.sitkFloat32)
    write_nifti(target, output_label, sitk.sitkUInt8)
    output_meta.write_text(json.dumps({
        'case': case,
        'source_size_xyz': list(image_itk.GetSize()),
        'source_spacing_xyz': list(image_itk.GetSpacing()),
        'source_origin_xyz': list(image_itk.GetOrigin()),
        'source_direction': list(image_itk.GetDirection()),
        'crop_lower_zyx': lower.tolist(),
        'crop_upper_zyx_exclusive': upper.tolist(),
        'network_shape_zyx': TARGET_SHAPE.tolist(),
    }, indent=2))
    return case


def build_metadata(cases, release, location_labels, metadata_dir):
    columns = [name for name, _ in sorted(location_labels.items(), key=lambda item: item[1])]
    counts = np.zeros(len(columns), dtype=int)
    rows = []
    for case in cases:
        locations = json.loads((release / 'location_jsons' / f'{case}.json').read_text())['locations']
        location_set = {int(value) for value in locations}
        flags = [int(i in location_set) for i in range(1, len(columns) + 1)]
        counts += flags
        modality = 'MRA' if '_mr_' in case else 'CTA'
        rows.append([case, 0, 'Unknown', modality, int(bool(location_set)), *flags])

    metadata_path = metadata_dir / 'topaneu_multitask_labels.csv'
    with metadata_path.open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['SeriesInstanceUID', 'PatientAge', 'PatientSex', 'Modality',
                         'Aneurysm Present', *columns])
        writer.writerows(rows)

    n = len(rows)
    presence = sum(row[4] for row in rows)
    modality_stats = {}
    for row in rows:
        total, positive = modality_stats.get(row[3], (0, 0))
        modality_stats[row[3]] = (total + 1, positive + int(row[4]))
    weights_path = metadata_dir / 'train_case_sampling_weight.csv'
    with weights_path.open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['SeriesInstanceUID', 'modality_weight', 'vessel_weight'])
        for row in rows:
            total, positive = modality_stats[row[3]]
            modality_weight = total / positive if positive else 1.0
            positive_indices = [i for i, flag in enumerate(row[5:]) if flag]
            vessel_weight = 1.0
            if positive_indices:
                vessel_weight += np.mean([300.0 / counts[i] for i in positive_indices])
            writer.writerow([row[0], modality_weight, vessel_weight])

    pos_weights = [min(20.0, (n - int(c)) / int(c)) if c else 1.0 for c in counts]
    return metadata_path, counts, [min(20.0, (n - presence) / presence) if presence else 1.0, pos_weights]


def build_dataset_json(raw_dataset, cases, vessel_labels, location_labels):
    labels = {'background': 0}
    labels.update({f'vessel::{name}': value for name, value in vessel_labels.items()})
    labels.update({f'aneurysm::{name}': 36 + value for name, value in location_labels.items()})
    payload = {
        'channel_names': {'0': 'angiography'},
        'labels': labels,
        'numTraining': len(cases),
        'file_ending': '.nii.gz',
        'overwrite_image_reader_writer': 'SimpleITKIO',
        'topaneu_location_labels': location_labels,
        'topaneu_vessel_labels': vessel_labels,
    }
    (raw_dataset / 'dataset.json').write_text(json.dumps(payload, indent=2))


def build_plans(template_path, preprocessed_dataset, counts, pos_weights):
    plans = json.loads(template_path.read_text())
    config = plans['configurations']['3d_fullres']
    plans['dataset_name'] = DATASET_NAME
    plans['plans_name'] = PLANS_NAME
    plans['original_median_spacing_after_transp'] = [1.0, 1.0, 1.0]
    plans['original_median_shape_after_transp'] = TARGET_SHAPE.tolist()
    plans['configurations'] = {'3d_fullres': config}
    config['data_identifier'] = f'{PLANS_NAME}_3d_fullres'
    config['batch_size'] = 4
    config['patch_size'] = TARGET_SHAPE.tolist()
    config['median_image_size_in_voxels'] = TARGET_SHAPE.astype(float).tolist()
    config['spacing'] = [1.0, 1.0, 1.0]
    config['seg_index_1'] = [[i] for i in range(1, 37)] + [list(range(37, 89))]
    config['seg_index_2'] = [[i] for i in range(37, 89)]
    config['seg_ce_class_weights_1'] = [1.0] * 36 + [10.0]
    config['seg_ce_class_weights_2'] = [5.0 if count else 1.0 for count in counts.tolist()]
    config['cls_task_index'] = [[list(range(37, 89))], [[i] for i in range(37, 89)]]
    config['pos_weights_list'] = pos_weights
    arch = config['architecture']['arch_kwargs']
    arch['cls_head_num_classes_list'] = [1, 52]
    arch['cls_drop_out_list'] = [0.0, 0.0]
    arch['cls_query_num_list'] = [2, 52]
    arch['cls_modality_num_classes'] = 2
    arch['cls_modality_query_num'] = 2
    output = preprocessed_dataset / f'{PLANS_NAME}.json'
    output.write_text(json.dumps(plans, indent=2))
    return output


def build_splits(cases, release):
    groups, strata = [], []
    for case in cases:
        parts = case.split('_')
        # Center-4 longitudinal scans append a scan counter after patient ID.
        patient = '_'.join(parts[:4]) if len(parts) == 5 and parts[1] == 'center4' else case
        locations = json.loads((release / 'location_jsons' / f'{case}.json').read_text())['locations']
        modality = 'mra' if '_mr_' in case else 'cta'
        groups.append(patient)
        strata.append(f'{modality}_{int(bool(locations))}')
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=2026)
    case_array = np.asarray(cases)
    return [
        {'train': case_array[train].tolist(), 'val': case_array[val].tolist()}
        for train, val in splitter.split(case_array, strata, groups)
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--release', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--template-plans', type=Path, required=True)
    parser.add_argument('--splits', type=Path)
    parser.add_argument('--margin-fraction', type=float, default=0.05)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()

    release = args.release.resolve()
    root = args.data_root.resolve()
    raw_dataset = root / 'nnXNet_raw' / DATASET_NAME
    preprocessed_dataset = root / 'nnXNet_preprocessed' / DATASET_NAME
    metadata_dir = root / 'metadata_128'
    images_tr, labels_tr = raw_dataset / 'imagesTr', raw_dataset / 'labelsTr'
    roi_meta = metadata_dir / 'roi_transforms'
    for folder in (images_tr, labels_tr, preprocessed_dataset, metadata_dir, roi_meta,
                   root / 'nnXNet_results'):
        folder.mkdir(parents=True, exist_ok=True)

    vessel_labels = load_labels(release / 'vessel_mapping.json')
    location_labels = load_labels(release / 'location_mapping.json')
    if sorted(vessel_labels.values()) != list(range(1, 37)):
        raise ValueError('Expected TopAneu vessel labels 1..36')
    if sorted(location_labels.values()) != list(range(1, 53)):
        raise ValueError('Expected TopAneu location labels 1..52')

    image_paths = sorted((release / 'images').glob('*_0000.nii.gz'))
    if not image_paths:
        raise FileNotFoundError(f'No images found in {release / "images"}')
    worker = partial(prepare_case, release=release, images_tr=images_tr,
                     labels_tr=labels_tr, roi_meta=roi_meta,
                     margin_fraction=args.margin_fraction, force=args.force)
    cases = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for index, case in enumerate(executor.map(worker, image_paths), 1):
            cases.append(case)
            if index % 20 == 0 or index == len(image_paths):
                print(f'Prepared {index}/{len(image_paths)} cases', flush=True)

    build_dataset_json(raw_dataset, cases, vessel_labels, location_labels)
    shutil.copy2(raw_dataset / 'dataset.json', preprocessed_dataset / 'dataset.json')
    metadata_path, counts, pos_weights = build_metadata(cases, release, location_labels, metadata_dir)
    plans_path = build_plans(args.template_plans.resolve(), preprocessed_dataset, counts, pos_weights)
    if args.splits:
        shutil.copy2(args.splits.resolve(), preprocessed_dataset / 'splits_final.json')
    else:
        (preprocessed_dataset / 'splits_final.json').write_text(
            json.dumps(build_splits(cases, release), indent=2))

    summary = {
        'dataset': DATASET_NAME,
        'num_cases': len(cases),
        'raw_dataset': str(raw_dataset),
        'plans': str(plans_path),
        'metadata': str(metadata_path),
        'location_case_counts': {
            name: int(counts[value - 1]) for name, value in location_labels.items()
        },
        'absent_location_classes': [
            name for name, value in location_labels.items() if counts[value - 1] == 0
        ],
    }
    (metadata_dir / 'setup_summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
