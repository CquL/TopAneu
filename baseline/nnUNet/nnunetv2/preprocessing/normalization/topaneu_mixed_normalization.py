import numpy as np

from nnunetv2.preprocessing.normalization.default_normalization_schemes import (
    ImageNormalization,
    CTNormalization,
    ZScoreNormalization,
)


class TopAneuMixedNormalization(ImageNormalization):
    """
    TopAneu mixed CTA/MRA normalization.

    CTA:
        CT-style dataset-level normalization.
        Statistics were computed from CTA GT foreground voxels
        (109 CTA cases, 508080 foreground voxels).

    MRA:
        Per-case Z-score normalization.

    Modality identification:
        CTA retains HU-like negative intensities.
        MRA is non-negative in the current TopAneu training set.
    """

    leaves_pixels_outside_mask_at_zero_if_use_mask_for_norm_is_true = False

    CTA_PROPERTIES = {
        "percentile_00_5": 64.96642303466797,
        "percentile_99_5": 584.0269165039062,
        "mean": 308.4395751953125,
        "std": 109.69413757324219,
    }

    # Current TopAneu:
    # CTA p0.5 is strongly negative (roughly -1000 HU or lower)
    # MRA p0.5 is >= 0
    CTA_P005_THRESHOLD = -100.0

    def run(self, image: np.ndarray, seg: np.ndarray = None) -> np.ndarray:
        image = image.astype(self.target_dtype, copy=False)

        p005 = float(np.percentile(image, 0.5))

        if p005 < self.CTA_P005_THRESHOLD:
            # CTA
            normalizer = CTNormalization(
                use_mask_for_norm=False,
                intensityproperties=self.CTA_PROPERTIES,
                target_dtype=self.target_dtype,
            )
        else:
            # MRA
            normalizer = ZScoreNormalization(
                use_mask_for_norm=False,
                intensityproperties=self.intensityproperties,
                target_dtype=self.target_dtype,
            )

        return normalizer.run(image, seg)
