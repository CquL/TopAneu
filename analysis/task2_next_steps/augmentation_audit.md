# TopAneu V2 augmentation audit

- All 416 ROI transform records preserve a consistent physical orientation: image index x maps to physical x with positive sign.
- The network tensor spatial axes are z, y, x, so augmentation axis 2 is the patient left-right axis.
- Every final fold checkpoint records `inference_allowed_mirroring_axes = (0, 1, 2)`.
- The base trainer applies `MirrorTransform((0, 1, 2))` to image and voxel target.
- No TopAneu-specific transform swaps the R/L vessel IDs or the paired R/L location labels.
- The global 52-label target is loaded independently from the CSV after augmentation and is never swapped.
- Therefore an x-mirrored input can show right anatomy on the left while retaining a right-side class target. This is a deterministic semantic conflict in training supervision.
- The current validation and Docker paths use direct network inference with mirroring disabled, so the defect is in training augmentation, not the current OOF/Docker inference pass.

Recommended controlled fix: disable all mirroring for one fresh fold-0 run. This is safer than only excluding x because superior/inferior and anterior/posterior flips can also corrupt the coordinate priors needed to distinguish trunk, junction and distal subtypes.
