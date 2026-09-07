import numpy as np

from nnunetv2.preprocessing.normalization.default_normalization_schemes import (
    CTNormalization,
    ImageNormalization,
    ZScoreNormalization,
)


class TopAneuMixedNormalization(ImageNormalization):
    """Modality-aware normalization for the mixed CTA/MRA TopAneu dataset."""

    leaves_pixels_outside_mask_at_zero_if_use_mask_for_norm_is_true = False

    # Computed from the 109 CTA training cases and their aneurysm foreground.
    CTA_PROPERTIES = {
        "percentile_00_5": 64.96642303466797,
        "percentile_99_5": 584.0269165039062,
        "mean": 308.4395751953125,
        "std": 109.69413757324219,
    }

    # The released CTA scans retain HU-like negative values; MRA is non-negative.
    CTA_P005_THRESHOLD = -100.0

    def run(self, image: np.ndarray, seg: np.ndarray = None) -> np.ndarray:
        image = image.astype(self.target_dtype, copy=False)
        p005 = float(np.percentile(image, 0.5))

        if p005 < self.CTA_P005_THRESHOLD:
            normalizer = CTNormalization(
                use_mask_for_norm=False,
                intensityproperties=self.CTA_PROPERTIES,
                target_dtype=self.target_dtype,
            )
        else:
            normalizer = ZScoreNormalization(
                use_mask_for_norm=False,
                intensityproperties=self.intensityproperties,
                target_dtype=self.target_dtype,
            )

        return normalizer.run(image, seg)
