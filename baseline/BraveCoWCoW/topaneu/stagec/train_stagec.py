#!/usr/bin/env python3

import argparse
import json
import math
import os
import random
import re
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch
import torch.nn as nn
import torch.nn.functional as F

from scipy import ndimage
from sklearn.metrics import f1_score
from torch.utils.data import (
    Dataset,
    DataLoader,
    WeightedRandomSampler,
)

from topaneu.component_location import LOCATION_VESSEL_CANDIDATES


# ============================================================
# Constants
# ============================================================

N_LOCATION = 52
N_VESSEL = 36

SIDE_NAMES = [
    "left",
    "right",
    "midline_or_unknown",
]

TERRITORY_NAMES = [
    "ICA",
    "MCA",
    "ACA",
    "PCA",
    "PCom",
    "ACom",
    "VA",
    "BA",
    "PICA",
    "AICA",
    "SCA",
    "AChA",
    "OA",
    "OTHER",
]

GT_STRUCTURE = ndimage.generate_binary_structure(3, 2)

DEFAULT_DATA_ROOT = Path(
    "/data/cyf/shared_data/TopAneu/"
    "BraveCoWCoW_predroi_160"
)

DATASET_NAME = "Dataset505_TopAneuBraveCoWCoWPredROI"

DEFAULT_RAW = (
    DEFAULT_DATA_ROOT
    / "nnXNet_raw"
    / DATASET_NAME
)

DEFAULT_SPLITS = (
    DEFAULT_DATA_ROOT
    / "nnXNet_preprocessed"
    / DATASET_NAME
    / "splits_final.json"
)

DEFAULT_FEATURES = (
    DEFAULT_DATA_ROOT
    / "stagec"
    / "stagec_gt_gt_features.npz"
)

DEFAULT_OUTPUT = (
    DEFAULT_DATA_ROOT
    / "stagec_results"
)

DEFAULT_TRANSFORMS = (
    DEFAULT_DATA_ROOT
    / "metadata"
    / "roi_transforms"
)


# ============================================================
# Utilities
# ============================================================

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_nii(path, dtype=None):

    img = sitk.ReadImage(str(path))

    arr = sitk.GetArrayFromImage(img)

    if dtype is not None:
        arr = arr.astype(dtype, copy=False)

    return arr


def resize_nearest(array, target_shape):

    factors = (
        np.asarray(target_shape, dtype=float)
        / np.asarray(array.shape, dtype=float)
    )

    result = ndimage.zoom(
        array,
        factors,
        order=0,
        mode="nearest",
        prefilter=False,
    )

    fixed = np.zeros(
        tuple(target_shape),
        dtype=result.dtype,
    )

    common = tuple(
        slice(0, min(a, b))
        for a, b in zip(
            result.shape,
            target_shape,
        )
    )

    fixed[common] = result[common]

    return fixed


def normalize_name(name):

    return re.sub(
        r"[^A-Z0-9]+",
        "_",
        name.upper(),
    ).strip("_")


# ============================================================
# Hierarchical labels
# ============================================================

def infer_side(vessel_names):

    left = False
    right = False

    for name in vessel_names:

        n = normalize_name(name)

        tokens = set(n.split("_"))

        if (
            "LEFT" in tokens
            or "L" in tokens
            or n.startswith("LEFT_")
            or n.startswith("L_")
        ):
            left = True

        if (
            "RIGHT" in tokens
            or "R" in tokens
            or n.startswith("RIGHT_")
            or n.startswith("R_")
        ):
            right = True

    if left and not right:
        return 0

    if right and not left:
        return 1

    return 2


def infer_territory(vessel_names):

    names = [
        normalize_name(x)
        for x in vessel_names
    ]

    text = " ".join(names)

    ordered = [
        ("PICA", "PICA"),
        ("AICA", "AICA"),
        ("SCA", "SCA"),

        ("ANTERIOR_CHOROIDAL", "AChA"),
        ("ACHA", "AChA"),

        ("POSTERIOR_COMMUNICATING", "PCom"),
        ("PCOM", "PCom"),

        ("ANTERIOR_COMMUNICATING", "ACom"),
        ("ACOM", "ACom"),

        ("OPHTHALMIC", "OA"),

        ("MCA", "MCA"),
        ("ACA", "ACA"),
        ("PCA", "PCA"),
        ("ICA", "ICA"),

        ("VERTEBRAL", "VA"),
        ("BASILAR", "BA"),
    ]

    for token, territory in ordered:

        if token in text:

            return TERRITORY_NAMES.index(
                territory
            )

    return TERRITORY_NAMES.index(
        "OTHER"
    )


def get_hierarchy_labels(
    location_id,
    vessel_names_by_id,
):

    parent_ids = list(
        LOCATION_VESSEL_CANDIDATES[
            int(location_id)
        ]
    )

    parent_names = [
        vessel_names_by_id.get(
            vessel_id,
            f"vessel_{vessel_id}",
        )
        for vessel_id in parent_ids
    ]

    parent_target = np.zeros(
        N_VESSEL,
        dtype=np.float32,
    )

    for vessel_id in parent_ids:

        parent_target[
            vessel_id - 1
        ] = 1.0

    side = infer_side(
        parent_names
    )

    territory = infer_territory(
        parent_names
    )

    return (
        parent_target,
        side,
        territory,
    )


# ============================================================
# GT vessel source
# ============================================================

def load_gt_vessel(
    case,
    combined,
    release_root,
    transforms_root,
):

    # --------------------------------------------------------
    # fallback:
    #
    # Dataset505 itself contains:
    #
    # 1..36 vessel
    # 37..88 aneurysm
    #
    # Note that aneurysm voxels replace vessel voxels.
    # --------------------------------------------------------

    fallback = np.where(
        (combined >= 1)
        & (combined <= 36),
        combined,
        0,
    ).astype(
        np.uint8
    )

    if release_root is None:
        return fallback

    source = (
        release_root
        / "vessel_masks"
        / f"{case}.nii.gz"
    )

    if not source.is_file():

        print(
            "[WARN]",
            case,
            "silver vessel unavailable;"
            " using Dataset505 vessel target",
        )

        return fallback

    transform_file = (
        transforms_root
        / f"{case}.json"
    )

    if not transform_file.is_file():
        raise FileNotFoundError(
            transform_file
        )

    vessel = read_nii(
        source,
        np.uint8,
    )

    transform = json.loads(
        transform_file.read_text()
    )

    lower = np.asarray(
        transform[
            "crop_lower_zyx"
        ],
        dtype=int,
    )

    upper = np.asarray(
        transform[
            "crop_upper_zyx_exclusive"
        ],
        dtype=int,
    )

    slices = tuple(
        slice(
            int(a),
            int(b),
        )
        for a, b in zip(
            lower,
            upper,
        )
    )

    cropped = vessel[slices]

    resized = resize_nearest(
        cropped,
        combined.shape,
    )

    return resized.astype(
        np.uint8,
        copy=False,
    )


# ============================================================
# Feature extraction
# ============================================================

def intensity_stats(array):

    array = array[
        np.isfinite(array)
    ]

    if array.size == 0:

        return [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]

    p10, p50, p90 = np.percentile(
        array,
        [10, 50, 90],
    )

    return [
        float(array.mean()),
        float(array.std()),
        float(p10),
        float(p50),
        float(p90),
    ]


def modality_features(case):

    if "_ct_" in case:
        return [1.0, 0.0]

    if "_mr_" in case:
        return [0.0, 1.0]

    return [0.0, 0.0]


def extract_component_features(
    image,
    component,
    vessel,
    case,
    near_radius=3,
    distance_clip=20.0,
):

    coords = np.argwhere(
        component
    )

    if coords.size == 0:
        raise ValueError(
            "Empty component"
        )

    shape = np.asarray(
        component.shape,
        dtype=float,
    )

    centroid = coords.mean(
        axis=0
    )

    lower = coords.min(
        axis=0
    )

    upper = (
        coords.max(
            axis=0
        )
        + 1
    )

    extent = (
        upper - lower
    )

    n_voxels = float(
        component.sum()
    )

    bbox_volume = float(
        np.prod(
            np.maximum(
                extent,
                1,
            )
        )
    )

    # --------------------------------------------------------
    # Geometry = 8
    # --------------------------------------------------------

    normalized_centroid = (
        centroid
        / np.maximum(
            shape - 1.0,
            1.0,
        )
    )

    normalized_extent = (
        extent / shape
    )

    log_volume = (
        np.log1p(
            n_voxels
        )
        / np.log1p(
            np.prod(shape)
        )
    )

    occupancy = (
        n_voxels
        / max(
            bbox_volume,
            1.0,
        )
    )

    geometry = [
        *normalized_centroid.tolist(),
        *normalized_extent.tolist(),
        float(log_volume),
        float(occupancy),
    ]

    # --------------------------------------------------------
    # Image = 10
    # --------------------------------------------------------

    lesion_image_stats = (
        intensity_stats(
            image[component]
        )
    )

    local_region = (
        ndimage.binary_dilation(
            component,
            iterations=5,
        )
    )

    shell = (
        local_region
        & ~component
    )

    shell_image_stats = (
        intensity_stats(
            image[shell]
        )
    )

    # --------------------------------------------------------
    # Vessel topology = 108
    #
    # 36 distance
    # 36 contact
    # 36 near flags
    # --------------------------------------------------------

    component_distance = (
        ndimage.distance_transform_edt(
            ~component
        )
    )

    near_region = (
        ndimage.binary_dilation(
            component,
            iterations=near_radius,
        )
    )

    near_denominator = max(
        float(
            near_region.sum()
        ),
        1.0,
    )

    distances = []
    contacts = []
    near_flags = []

    for vessel_id in range(
        1,
        N_VESSEL + 1,
    ):

        vessel_mask = (
            vessel == vessel_id
        )

        if vessel_mask.any():

            distance = float(
                component_distance[
                    vessel_mask
                ].min()
            )

            distance = (
                min(
                    distance,
                    distance_clip,
                )
                / distance_clip
            )

        else:

            distance = 1.0

        contact = float(
            np.count_nonzero(
                vessel_mask
                & near_region
            )
            / near_denominator
        )

        near_flag = float(
            np.any(
                vessel_mask
                & near_region
            )
        )

        distances.append(
            distance
        )

        contacts.append(
            contact
        )

        near_flags.append(
            near_flag
        )

    # --------------------------------------------------------
    # Feature vector:
    #
    # geometry      8
    # modality      2
    # image         10
    # vessel        108
    #
    # total         128
    # --------------------------------------------------------

    features = np.asarray(
        geometry
        + modality_features(case)
        + lesion_image_stats
        + shell_image_stats
        + distances
        + contacts
        + near_flags,
        dtype=np.float32,
    )

    if features.shape != (128,):
        raise RuntimeError(
            f"unexpected feature shape "
            f"{features.shape}"
        )

    return features


# ============================================================
# Build Stage C dataset
# ============================================================

def build_feature_dataset(
    raw_root,
    splits_file,
    output_file,
    release_root=None,
    transforms_root=DEFAULT_TRANSFORMS,
):

    dataset_json = json.loads(
        (
            raw_root
            / "dataset.json"
        ).read_text()
    )

    vessel_labels = {
        name: int(value)
        for name, value
        in dataset_json[
            "topaneu_vessel_labels"
        ].items()
    }

    location_labels = {
        name: int(value)
        for name, value
        in dataset_json[
            "topaneu_location_labels"
        ].items()
    }

    vessel_names_by_id = {
        value: name
        for name, value
        in vessel_labels.items()
    }

    location_names_by_id = {
        value: name
        for name, value
        in location_labels.items()
    }

    if sorted(
        vessel_names_by_id
    ) != list(
        range(
            1,
            37,
        )
    ):
        raise RuntimeError(
            "Expected vessel IDs 1..36"
        )

    if sorted(
        location_names_by_id
    ) != list(
        range(
            1,
            53,
        )
    ):
        raise RuntimeError(
            "Expected location IDs 1..52"
        )

    splits = json.loads(
        splits_file.read_text()
    )

    case_set = set()

    for split in splits:

        case_set.update(
            split["train"]
        )

        case_set.update(
            split["val"]
        )

    cases = sorted(
        case_set
    )

    if len(cases) != 416:

        raise RuntimeError(
            f"Expected 416 cases, "
            f"got {len(cases)}"
        )

    X = []

    y_location = []
    y_parent = []
    y_side = []
    y_territory = []

    sample_case = []
    sample_size = []
    sample_name = []

    for case_index, case in enumerate(
        cases,
        start=1,
    ):

        image = read_nii(
            raw_root
            / "imagesTr"
            / f"{case}_0000.nii.gz",
            np.float32,
        )

        combined = read_nii(
            raw_root
            / "labelsTr"
            / f"{case}.nii.gz",
            np.uint8,
        )

        vessel = load_gt_vessel(
            case,
            combined,
            release_root,
            transforms_root,
        )

        # ----------------------------------------------------
        # IMPORTANT
        #
        # Components are computed separately for each true
        # location class.
        #
        # This avoids the known 396 vs 393 discrepancy caused
        # when two adjacent different-location lesions merge
        # after binary collapse.
        # ----------------------------------------------------

        for location_id in range(
            1,
            N_LOCATION + 1,
        ):

            location_mask = (
                combined
                == (
                    36
                    + location_id
                )
            )

            if not location_mask.any():
                continue

            components, n_components = (
                ndimage.label(
                    location_mask,
                    structure=GT_STRUCTURE,
                )
            )

            for component_id in range(
                1,
                n_components + 1,
            ):

                component = (
                    components
                    == component_id
                )

                if not component.any():
                    continue

                features = (
                    extract_component_features(
                        image,
                        component,
                        vessel,
                        case,
                    )
                )

                (
                    parent,
                    side,
                    territory,
                ) = get_hierarchy_labels(
                    location_id,
                    vessel_names_by_id,
                )

                X.append(
                    features
                )

                y_location.append(
                    location_id - 1
                )

                y_parent.append(
                    parent
                )

                y_side.append(
                    side
                )

                y_territory.append(
                    territory
                )

                sample_case.append(
                    case
                )

                sample_size.append(
                    int(
                        component.sum()
                    )
                )

                sample_name.append(
                    location_names_by_id[
                        location_id
                    ]
                )

        if (
            case_index % 25 == 0
            or case_index == len(cases)
        ):

            print(
                f"processed "
                f"{case_index}/"
                f"{len(cases)} "
                f"cases; "
                f"components="
                f"{len(X)}",
                flush=True,
            )

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    X = np.stack(
        X
    ).astype(
        np.float32
    )

    y_location = np.asarray(
        y_location,
        dtype=np.int64,
    )

    y_parent = np.stack(
        y_parent
    ).astype(
        np.float32
    )

    y_side = np.asarray(
        y_side,
        dtype=np.int64,
    )

    y_territory = np.asarray(
        y_territory,
        dtype=np.int64,
    )

    np.savez_compressed(
        output_file,

        X=X,

        y_location=y_location,

        y_parent=y_parent,

        y_side=y_side,

        y_territory=y_territory,

        case=np.asarray(
            sample_case
        ),

        size_voxels=np.asarray(
            sample_size,
            dtype=np.int64,
        ),

        location_name=np.asarray(
            sample_name
        ),
    )

    counts = np.bincount(
        y_location,
        minlength=N_LOCATION,
    )

    summary = {
        "cases": len(cases),

        "components": int(
            len(X)
        ),

        "feature_dim": int(
            X.shape[1]
        ),

        "vessel_source": (
            "release_silver"
            if release_root is not None
            else "Dataset505_combined_gt"
        ),

        "location_counts": {
            str(i + 1): int(v)
            for i, v
            in enumerate(counts)
        },
    }

    summary_file = (
        output_file.with_suffix(
            ".summary.json"
        )
    )

    summary_file.write_text(
        json.dumps(
            summary,
            indent=2,
        )
    )

    print(
        json.dumps(
            summary,
            indent=2,
        )
    )


# ============================================================
# Dataset
# ============================================================

class ComponentDataset(
    Dataset
):

    def __init__(
        self,
        X,
        y_location,
        y_parent,
        y_side,
        y_territory,
    ):

        self.X = torch.from_numpy(
            X
        ).float()

        self.y_location = (
            torch.from_numpy(
                y_location
            ).long()
        )

        self.y_parent = (
            torch.from_numpy(
                y_parent
            ).float()
        )

        self.y_side = (
            torch.from_numpy(
                y_side
            ).long()
        )

        self.y_territory = (
            torch.from_numpy(
                y_territory
            ).long()
        )

    def __len__(self):

        return len(
            self.X
        )

    def __getitem__(self, i):

        return (
            self.X[i],

            self.y_location[i],

            self.y_parent[i],

            self.y_side[i],

            self.y_territory[i],
        )


# ============================================================
# Model
# ============================================================

class StageCClassifier(
    nn.Module
):

    def __init__(
        self,
        input_dim=128,
        dropout=0.25,
    ):

        super().__init__()

        self.features = nn.Sequential(

            nn.Linear(
                input_dim,
                512,
            ),

            nn.LayerNorm(
                512
            ),

            nn.GELU(),

            nn.Dropout(
                dropout
            ),

            nn.Linear(
                512,
                256,
            ),

            nn.LayerNorm(
                256
            ),

            nn.GELU(),

            nn.Dropout(
                dropout
            ),
        )

        self.location_head = (
            nn.Linear(
                256,
                52,
            )
        )

        self.parent_head = (
            nn.Linear(
                256,
                36,
            )
        )

        self.side_head = (
            nn.Linear(
                256,
                3,
            )
        )

        self.territory_head = (
            nn.Linear(
                256,
                len(
                    TERRITORY_NAMES
                ),
            )
        )

    def forward(
        self,
        x,
    ):

        x = self.features(
            x
        )

        return {
            "location":
                self.location_head(x),

            "parent":
                self.parent_head(x),

            "side":
                self.side_head(x),

            "territory":
                self.territory_head(x),
        }


# ============================================================
# Sampling
# ============================================================

def build_sampler(
    locations
):

    counts = np.bincount(
        locations,
        minlength=N_LOCATION,
    ).astype(
        np.float64
    )

    inverse = np.ones(
        N_LOCATION,
        dtype=np.float64,
    )

    present = counts > 0

    inverse[present] = (
        1.0
        / np.sqrt(
            counts[present]
        )
    )

    class_weight = (
        inverse[
            locations
        ]
    )

    class_weight /= (
        class_weight.mean()
    )

    # 50% uniform
    # +
    # 50% class-balanced

    sample_weight = (
        0.5
        + 0.5
        * class_weight
    )

    return WeightedRandomSampler(

        weights=torch.as_tensor(
            sample_weight,
            dtype=torch.double,
        ),

        num_samples=len(
            locations
        ),

        replacement=True,
    )


# ============================================================
# Metrics
# ============================================================

@torch.no_grad()
def validate(
    model,
    loader,
    device,
    seen_location_mask,
):

    model.eval()

    gt_location = []
    pred_location = []

    gt_parent = []
    pred_parent = []

    gt_side = []
    pred_side = []

    gt_territory = []
    pred_territory = []

    for batch in loader:

        (
            X,
            y_location,
            y_parent,
            y_side,
            y_territory,
        ) = batch

        X = X.to(
            device
        )

        outputs = model(
            X
        )

        location_logits = (
            outputs[
                "location"
            ].clone()
        )

        # Do not allow zero-sample training-fold
        # locations to compete.

        location_logits[
            :,
            ~seen_location_mask,
        ] = -1e9

        location_pred = (
            location_logits
            .argmax(
                dim=1
            )
            .cpu()
            .numpy()
        )

        parent_pred = (
            torch.sigmoid(
                outputs[
                    "parent"
                ]
            )
            >= 0.5
        ).cpu().numpy()

        side_pred = (
            outputs[
                "side"
            ]
            .argmax(
                dim=1
            )
            .cpu()
            .numpy()
        )

        territory_pred = (
            outputs[
                "territory"
            ]
            .argmax(
                dim=1
            )
            .cpu()
            .numpy()
        )

        gt_location.append(
            y_location.numpy()
        )

        pred_location.append(
            location_pred
        )

        gt_parent.append(
            y_parent.numpy()
        )

        pred_parent.append(
            parent_pred
        )

        gt_side.append(
            y_side.numpy()
        )

        pred_side.append(
            side_pred
        )

        gt_territory.append(
            y_territory.numpy()
        )

        pred_territory.append(
            territory_pred
        )

    y = np.concatenate(
        gt_location
    )

    p = np.concatenate(
        pred_location
    )

    y_parent = np.concatenate(
        gt_parent
    )

    p_parent = np.concatenate(
        pred_parent
    )

    y_side = np.concatenate(
        gt_side
    )

    p_side = np.concatenate(
        pred_side
    )

    y_territory = np.concatenate(
        gt_territory
    )

    p_territory = np.concatenate(
        pred_territory
    )

    seen_np = (
        seen_location_mask
        .cpu()
        .numpy()
    )

    seen_samples = (
        seen_np[y]
    )

    exact_accuracy = float(
        np.mean(
            p == y
        )
    )

    if seen_samples.any():

        seen_accuracy = float(
            np.mean(
                p[seen_samples]
                == y[seen_samples]
            )
        )

        labels = sorted(
            set(
                y[
                    seen_samples
                ].tolist()
            )
        )

        macro_f1 = float(
            f1_score(
                y[seen_samples],
                p[seen_samples],
                labels=labels,
                average="macro",
                zero_division=0,
            )
        )

    else:

        seen_accuracy = 0.0
        macro_f1 = 0.0

    parent_f1 = float(
        f1_score(
            y_parent.reshape(-1),
            p_parent.reshape(-1),
            average="binary",
            zero_division=0,
        )
    )

    side_accuracy = float(
        np.mean(
            p_side
            == y_side
        )
    )

    territory_accuracy = float(
        np.mean(
            p_territory
            == y_territory
        )
    )

    return {
        "exact_location_accuracy":
            exact_accuracy,

        "seen_location_accuracy":
            seen_accuracy,

        "seen_location_macro_f1":
            macro_f1,

        "parent_vessel_micro_f1":
            parent_f1,

        "side_accuracy":
            side_accuracy,

        "territory_accuracy":
            territory_accuracy,
    }


# ============================================================
# Train fold
# ============================================================

def train_fold(
    fold,
    features_file,
    splits_file,
    output_root,
    epochs=150,
    batch_size=16,
    lr=3e-4,
    weight_decay=1e-4,
    seed=2026,
):

    seed_all(
        seed + fold
    )

    data = np.load(
        features_file,
        allow_pickle=False,
    )

    X = data[
        "X"
    ].astype(
        np.float32
    )

    y_location = data[
        "y_location"
    ].astype(
        np.int64
    )

    y_parent = data[
        "y_parent"
    ].astype(
        np.float32
    )

    y_side = data[
        "y_side"
    ].astype(
        np.int64
    )

    y_territory = data[
        "y_territory"
    ].astype(
        np.int64
    )

    cases = data[
        "case"
    ].astype(
        str
    )

    splits = json.loads(
        splits_file.read_text()
    )

    split = splits[
        fold
    ]

    train_cases = set(
        split[
            "train"
        ]
    )

    val_cases = set(
        split[
            "val"
        ]
    )

    train_mask = np.asarray([
        case in train_cases
        for case in cases
    ])

    val_mask = np.asarray([
        case in val_cases
        for case in cases
    ])

    if np.any(
        train_mask
        & val_mask
    ):

        raise RuntimeError(
            "train/val case leakage"
        )

    # --------------------------------------------------------
    # Feature normalization using train split ONLY
    # --------------------------------------------------------

    mean = X[
        train_mask
    ].mean(
        axis=0
    )

    std = X[
        train_mask
    ].std(
        axis=0
    )

    std[
        std < 1e-6
    ] = 1.0

    X = (
        X - mean
    ) / std

    train_dataset = (
        ComponentDataset(

            X[
                train_mask
            ],

            y_location[
                train_mask
            ],

            y_parent[
                train_mask
            ],

            y_side[
                train_mask
            ],

            y_territory[
                train_mask
            ],
        )
    )

    val_dataset = (
        ComponentDataset(

            X[
                val_mask
            ],

            y_location[
                val_mask
            ],

            y_parent[
                val_mask
            ],

            y_side[
                val_mask
            ],

            y_territory[
                val_mask
            ],
        )
    )

    sampler = build_sampler(
        y_location[
            train_mask
        ]
    )

    train_loader = DataLoader(

        train_dataset,

        batch_size=batch_size,

        sampler=sampler,

        num_workers=0,

        pin_memory=True,
    )

    val_loader = DataLoader(

        val_dataset,

        batch_size=128,

        shuffle=False,

        num_workers=0,

        pin_memory=True,
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model = StageCClassifier(
        input_dim=X.shape[1],
    ).to(
        device
    )

    optimizer = torch.optim.AdamW(

        model.parameters(),

        lr=lr,

        weight_decay=weight_decay,
    )

    # --------------------------------------------------------
    # Location class weights
    # --------------------------------------------------------

    location_counts = np.bincount(

        y_location[
            train_mask
        ],

        minlength=N_LOCATION,

    ).astype(
        np.float64
    )

    seen_location = (
        location_counts > 0
    )

    location_weight = np.ones(
        N_LOCATION,
        dtype=np.float32,
    )

    if seen_location.any():

        values = (
            1.0
            / np.sqrt(
                location_counts[
                    seen_location
                ]
            )
        )

        values /= (
            values.min()
        )

        location_weight[
            seen_location
        ] = np.clip(
            values,
            1.0,
            10.0,
        )

    location_weight = (
        torch.from_numpy(
            location_weight
        )
        .to(
            device
        )
    )

    seen_location_mask = (
        torch.from_numpy(
            seen_location
        )
        .bool()
        .to(
            device
        )
    )

    # --------------------------------------------------------
    # Parent vessel BCE weights
    # --------------------------------------------------------

    train_parent = (
        y_parent[
            train_mask
        ]
    )

    positives = (
        train_parent.sum(
            axis=0
        )
    )

    negatives = (
        len(
            train_parent
        )
        - positives
    )

    parent_pos_weight = (
        np.ones(
            N_VESSEL,
            dtype=np.float32,
        )
    )

    valid_parent = (
        positives > 0
    )

    parent_pos_weight[
        valid_parent
    ] = np.clip(

        negatives[
            valid_parent
        ]
        / np.maximum(
            positives[
                valid_parent
            ],
            1.0,
        ),

        1.0,
        20.0,
    )

    parent_pos_weight = (
        torch.from_numpy(
            parent_pos_weight
        )
        .to(
            device
        )
    )

    fold_output = (
        output_root
        / f"fold_{fold}"
    )

    fold_output.mkdir(
        parents=True,
        exist_ok=True,
    )

    best_score = -1.0

    history = []

    warmup_epochs = 5

    for epoch in range(
        epochs
    ):

        model.train()

        # ----------------------------------------------------
        # 5 epoch warmup + cosine decay
        # ----------------------------------------------------

        if epoch < warmup_epochs:

            lr_scale = (
                epoch + 1
            ) / warmup_epochs

        else:

            progress = (
                epoch
                - warmup_epochs
            ) / max(
                epochs
                - warmup_epochs
                - 1,
                1,
            )

            lr_scale = (
                0.5
                * (
                    1.0
                    + math.cos(
                        math.pi
                        * progress
                    )
                )
            )

        current_lr = (
            lr
            * lr_scale
        )

        for group in (
            optimizer.param_groups
        ):

            group[
                "lr"
            ] = current_lr

        train_losses = []

        for batch in train_loader:

            (
                features,
                target_location,
                target_parent,
                target_side,
                target_territory,
            ) = batch

            features = features.to(
                device,
                non_blocking=True,
            )

            target_location = (
                target_location.to(
                    device
                )
            )

            target_parent = (
                target_parent.to(
                    device
                )
            )

            target_side = (
                target_side.to(
                    device
                )
            )

            target_territory = (
                target_territory.to(
                    device
                )
            )

            outputs = model(
                features
            )

            location_logits = (
                outputs[
                    "location"
                ].clone()
            )

            # Training-fold zero sample classes
            # are excluded from 52-way competition.

            location_logits[
                :,
                ~seen_location_mask,
            ] = -1e9

            loss_location = (
                F.cross_entropy(

                    location_logits,

                    target_location,

                    weight=location_weight,
                )
            )

            loss_parent = (
                F.binary_cross_entropy_with_logits(

                    outputs[
                        "parent"
                    ],

                    target_parent,

                    pos_weight=
                        parent_pos_weight,
                )
            )

            loss_side = (
                F.cross_entropy(

                    outputs[
                        "side"
                    ],

                    target_side,
                )
            )

            loss_territory = (
                F.cross_entropy(

                    outputs[
                        "territory"
                    ],

                    target_territory,
                )
            )

            # ------------------------------------------------
            # Hierarchical objective
            # ------------------------------------------------

            loss = (

                0.50
                * loss_location

                + 1.00
                * loss_parent

                + 0.50
                * loss_side

                + 0.75
                * loss_territory
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=5.0,
            )

            optimizer.step()

            train_losses.append(
                float(
                    loss
                    .detach()
                    .cpu()
                )
            )

        # ----------------------------------------------------
        # FULL component-level validation EVERY epoch.
        #
        # No pseudo Dice.
        # No EMA selection.
        # ----------------------------------------------------

        metrics = validate(

            model,

            val_loader,

            device,

            seen_location_mask,
        )

        selection_score = (

            0.70
            * metrics[
                "seen_location_macro_f1"
            ]

            + 0.20
            * metrics[
                "seen_location_accuracy"
            ]

            + 0.10
            * metrics[
                "parent_vessel_micro_f1"
            ]
        )

        record = {

            "epoch":
                epoch,

            "lr":
                current_lr,

            "train_loss":
                float(
                    np.mean(
                        train_losses
                    )
                ),

            "selection_score":
                float(
                    selection_score
                ),

            **metrics,
        }

        history.append(
            record
        )

        if (
            selection_score
            > best_score
        ):

            best_score = (
                selection_score
            )

            torch.save(
                {
                    "model":
                        model.state_dict(),

                    "mean":
                        mean.astype(
                            np.float32
                        ),

                    "std":
                        std.astype(
                            np.float32
                        ),

                    "seen_location_mask":
                        seen_location,

                    "fold":
                        fold,

                    "epoch":
                        epoch,

                    "metrics":
                        metrics,
                },

                fold_output
                / "checkpoint_best_stagec.pth",
            )

        if (
            epoch % 10 == 0
            or epoch == epochs - 1
        ):

            print(
                json.dumps(
                    record
                ),
                flush=True,
            )

    final_metrics = validate(

        model,

        val_loader,

        device,

        seen_location_mask,
    )

    torch.save(
        {
            "model":
                model.state_dict(),

            "mean":
                mean.astype(
                    np.float32
                ),

            "std":
                std.astype(
                    np.float32
                ),

            "seen_location_mask":
                seen_location,

            "fold":
                fold,

            "epoch":
                epochs - 1,

            "metrics":
                final_metrics,
        },

        fold_output
        / "checkpoint_final_stagec.pth",
    )

    (
        fold_output
        / "history.json"
    ).write_text(
        json.dumps(
            history,
            indent=2,
        )
    )

    summary = {

        "fold":
            fold,

        "train_components":
            int(
                train_mask.sum()
            ),

        "val_components":
            int(
                val_mask.sum()
            ),

        "best_selection_score":
            float(
                best_score
            ),

        "final":
            final_metrics,
    }

    (
        fold_output
        / "summary.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
        )
    )

    print(
        json.dumps(
            summary,
            indent=2,
        )
    )


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prepare",
        action="store_true",
    )

    parser.add_argument(
        "--fold",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--raw-root",
        type=Path,
        default=DEFAULT_RAW,
    )

    parser.add_argument(
        "--splits",
        type=Path,
        default=DEFAULT_SPLITS,
    )

    parser.add_argument(
        "--features",
        type=Path,
        default=DEFAULT_FEATURES,
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT,
    )

    parser.add_argument(
        "--release-root",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--transforms-root",
        type=Path,
        default=DEFAULT_TRANSFORMS,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=150,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
    )

    args = parser.parse_args()

    if args.prepare:

        build_feature_dataset(

            raw_root=
                args.raw_root,

            splits_file=
                args.splits,

            output_file=
                args.features,

            release_root=
                args.release_root,

            transforms_root=
                args.transforms_root,
        )

        return

    if args.fold is None:

        raise RuntimeError(
            "--fold is required "
            "unless --prepare is used"
        )

    if not (
        args.features.is_file()
    ):

        raise FileNotFoundError(
            f"{args.features}\n"
            "Run --prepare first."
        )

    train_fold(

        fold=args.fold,

        features_file=
            args.features,

        splits_file=
            args.splits,

        output_root=
            args.output_root,

        epochs=
            args.epochs,

        batch_size=
            args.batch_size,

        lr=
            args.lr,
    )


if __name__ == "__main__":
    main()
