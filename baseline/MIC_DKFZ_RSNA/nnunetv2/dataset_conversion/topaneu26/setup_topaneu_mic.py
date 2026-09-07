import argparse
import json
import os
import random
import shutil
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

import numpy as np


def load_json(path: Path):
    with path.open() as f:
        return json.load(f)


def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(obj, f, indent=2)
        f.write("\n")


def ensure_symlink(source: Path, destination: Path):
    source = source.resolve()
    if destination.is_symlink():
        if destination.resolve() != source:
            raise RuntimeError(f"Unexpected symlink target: {destination}")
        return
    if destination.exists():
        raise RuntimeError(f"Refusing to replace existing path: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.symlink_to(source, target_is_directory=source.is_dir())


def case_id_from_image(path: Path) -> str:
    suffix = "_0000.nii.gz"
    if not path.name.endswith(suffix):
        raise ValueError(f"Unexpected image filename: {path.name}")
    return path.name[: -len(suffix)]


def patient_id(case_id: str) -> str:
    # Only center-4 contains longitudinal IDs such as ..._008_1 and ..._008_2.
    parts = case_id.split("_")
    if len(parts) == 5 and parts[1] == "center4" and parts[2] == "ct":
        return "_".join(parts[:-1])
    return case_id


def make_grouped_multilabel_splits(
    case_ids: list[str], release: Path, n_splits: int = 5, seed: int = 12345
):
    labels_by_case = {
        case_id: load_json(release / "location_jsons" / f"{case_id}.json")[
            "locations"
        ]
        for case_id in case_ids
    }

    groups = defaultdict(list)
    for case_id in case_ids:
        groups[patient_id(case_id)].append(case_id)

    # Features: 52 location counts, CT/MR, four centers, positive/negative, size.
    n_features = 52 + 2 + 4 + 2 + 1
    group_vectors = {}
    centers = ["center1", "center2", "center4", "center5"]
    for group_id, cases in groups.items():
        vector = np.zeros(n_features, dtype=np.float64)
        for case_id in cases:
            locations = labels_by_case[case_id]
            for label in locations:
                vector[label - 1] += 1
            modality = case_id.split("_")[2]
            vector[52 + (0 if modality == "ct" else 1)] += 1
            center = case_id.split("_")[1]
            vector[54 + centers.index(center)] += 1
            vector[58 + (0 if locations else 1)] += 1
            vector[60] += 1
        group_vectors[group_id] = vector

    totals = np.sum(list(group_vectors.values()), axis=0)
    target = totals / n_splits
    location_weights = np.divide(
        1.0,
        np.maximum(totals[:52], 1.0),
        out=np.zeros(52, dtype=np.float64),
        where=totals[:52] > 0,
    )
    weights = np.concatenate(
        [location_weights, np.full(2, 0.10), np.full(4, 0.10), np.full(2, 0.15), [0.25]]
    )

    rng = random.Random(seed)
    group_ids = list(groups)
    rng.shuffle(group_ids)
    group_ids.sort(
        key=lambda group_id: (
            np.sum(
                group_vectors[group_id][:52]
                / np.maximum(totals[:52], 1.0)
            ),
            np.sum(group_vectors[group_id][:52]),
            group_vectors[group_id][60],
        ),
        reverse=True,
    )

    fold_vectors = np.zeros((n_splits, n_features), dtype=np.float64)
    fold_groups = [[] for _ in range(n_splits)]
    for group_id in group_ids:
        vector = group_vectors[group_id]
        costs = []
        for fold in range(n_splits):
            candidate = fold_vectors.copy()
            candidate[fold] += vector
            normalized_error = (candidate - target) / np.maximum(target, 1.0)
            costs.append(float(np.sum(weights * normalized_error**2)))
        best_fold = min(
            range(n_splits),
            key=lambda fold: (costs[fold], fold_vectors[fold, 60], fold),
        )
        fold_vectors[best_fold] += vector
        fold_groups[best_fold].append(group_id)

    all_cases = set(case_ids)
    splits = []
    for groups_in_fold in fold_groups:
        val = sorted(case for group in groups_in_fold for case in groups[group])
        train = sorted(all_cases - set(val))
        splits.append({"train": train, "val": val})

    return splits


def validate_splits(splits, case_ids):
    all_cases = set(case_ids)
    val_occurrences = defaultdict(int)
    patient_folds = defaultdict(set)
    for fold, split in enumerate(splits):
        train, val = set(split["train"]), set(split["val"])
        if train & val:
            raise RuntimeError(f"Fold {fold} has train/validation overlap")
        if train | val != all_cases:
            raise RuntimeError(f"Fold {fold} does not cover all cases")
        for case_id in val:
            val_occurrences[case_id] += 1
            patient_folds[patient_id(case_id)].add(fold)
    if any(count != 1 for count in val_occurrences.values()):
        raise RuntimeError("Each case must occur in exactly one validation fold")
    if any(len(folds) != 1 for folds in patient_folds.values()):
        raise RuntimeError("A longitudinal patient was split across folds")


def build_plans(existing_plans: Path, mic_plans: Path, dataset_name: str, batch_size: int):
    base = load_json(existing_plans)
    mic = load_json(mic_plans)
    configuration = deepcopy(base["configurations"]["3d_fullres"])

    # Keep the existing preprocessing contract so that the 39 GB of arrays are
    # reusable, but use the MIC-DKFZ ResEncUNetM architecture and patch shape.
    configuration["architecture"] = deepcopy(
        mic["configurations"]["3d_fullres"]["architecture"]
    )
    configuration["patch_size"] = deepcopy(
        mic["configurations"]["3d_fullres"]["patch_size"]
    )
    configuration["batch_size"] = batch_size
    configuration["data_identifier"] = "nnUNetPlans_3d_fullres"
    configuration["preprocessor_name"] = "DefaultPreprocessor"
    configuration["normalization_schemes"] = ["TopAneuMixedNormalization"]
    configuration["use_mask_for_norm"] = [False]

    base["dataset_name"] = dataset_name
    base["plans_name"] = "TopAneuMICPlans"
    base["configurations"] = {
        "3d_fullres": configuration,
        # One fold per GPU for parallel cross-validation. This reuses the same
        # preprocessed arrays and architecture, changing only the batch size.
        "3d_fullres_bs4": {
            "inherits_from": "3d_fullres",
            "batch_size": 4,
        },
    }
    return base


def main():
    parser = argparse.ArgumentParser(
        description="Create an isolated TopAneu MIC-DKFZ dataset configuration."
    )
    parser.add_argument(
        "--release",
        type=Path,
        default=Path("/data/cyf/shared_data/TopAneu/topaneu_release"),
    )
    parser.add_argument(
        "--source-preprocessed",
        type=Path,
        default=Path(
            "/data/cyf/shared_data/TopAneu/nnUNet_data/nnUNet_preprocessed/Dataset501_TopAneu"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/data/cyf/shared_data/TopAneu/MIC_DKFZ_RSNA_data"),
    )
    parser.add_argument("--dataset-id", type=int, default=502)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[3]
    dataset_name = f"Dataset{args.dataset_id:03d}_TopAneuMIC"
    raw_dataset = args.output_root / "nnUNet_raw" / dataset_name
    preprocessed_dataset = args.output_root / "nnUNet_preprocessed" / dataset_name
    (args.output_root / "nnUNet_results").mkdir(parents=True, exist_ok=True)

    images = sorted((args.release / "images").glob("*_0000.nii.gz"))
    case_ids = [case_id_from_image(path) for path in images]
    masks = {path.name.removesuffix(".nii.gz"): path for path in (args.release / "location_masks").glob("*.nii.gz")}
    json_ids = {path.name.removesuffix(".json") for path in (args.release / "location_jsons").glob("*.json")}
    if len(images) != 416 or set(case_ids) != set(masks) or set(case_ids) != json_ids:
        raise RuntimeError("TopAneu release image/mask/JSON sets are inconsistent")

    for image, case_id in zip(images, case_ids):
        ensure_symlink(image, raw_dataset / "imagesTr" / image.name)
        ensure_symlink(masks[case_id], raw_dataset / "labelsTr" / masks[case_id].name)

    mapping = load_json(args.release / "location_mapping.json")["labels"]
    ordered_labels = {
        name: value for name, value in sorted(mapping.items(), key=lambda item: item[1])
    }
    dataset_json = {
        "name": dataset_name,
        "channel_names": {"0": "topaneu_mixed"},
        "labels": ordered_labels,
        "numTraining": len(case_ids),
        "file_ending": ".nii.gz",
        "overwrite_image_reader_writer": "SimpleITKIO",
    }
    save_json(dataset_json, raw_dataset / "dataset.json")
    save_json(dataset_json, preprocessed_dataset / "dataset.json")

    ensure_symlink(
        args.source_preprocessed / "nnUNetPlans_3d_fullres",
        preprocessed_dataset / "nnUNetPlans_3d_fullres",
    )
    ensure_symlink(
        args.source_preprocessed / "gt_segmentations",
        preprocessed_dataset / "gt_segmentations",
    )
    shutil.copy2(
        args.source_preprocessed / "dataset_fingerprint.json",
        preprocessed_dataset / "dataset_fingerprint.json",
    )

    plans = build_plans(
        args.source_preprocessed / "nnUNetPlans.json",
        repo_root
        / "nnunetv2/dataset_conversion/kaggle_2025_rsna/plans/nnUNetResEncUNetMPlans.json",
        dataset_name,
        args.batch_size,
    )
    save_json(plans, preprocessed_dataset / "TopAneuMICPlans.json")

    splits = make_grouped_multilabel_splits(case_ids, args.release)
    validate_splits(splits, case_ids)
    save_json(splits, preprocessed_dataset / "splits_final.json")

    metadata = {
        "dataset_name": dataset_name,
        "release": str(args.release.resolve()),
        "reused_preprocessed_arrays": str(
            (args.source_preprocessed / "nnUNetPlans_3d_fullres").resolve()
        ),
        "plans": "TopAneuMICPlans",
        "trainer": "TopAneuMICTrainer",
        "epochs": 1000,
        "global_batch_size": args.batch_size,
        "cases": len(case_ids),
        "reuse_mode": "read-only directory symlink",
    }
    save_json(metadata, args.output_root / "setup_metadata.json")

    print(f"Configured {dataset_name} at {args.output_root}")
    print(f"Cases: {len(case_ids)}")
    print("Validation fold sizes:", [len(split["val"]) for split in splits])
    print("Preprocessed arrays are reused read-only through a directory symlink.")


if __name__ == "__main__":
    main()
