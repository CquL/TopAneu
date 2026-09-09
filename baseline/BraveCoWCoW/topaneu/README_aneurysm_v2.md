# Aneurysm V2 fold-0 pilot

This experiment reuses Dataset505, its fixed splits, 160-cube preprocessing,
batch size 4, and the existing binary Dice+CE objective and case sampling.
It initializes from the matching **VesselOnly fold-0 best checkpoint** using
the repository's pretrained-weight loader; optimizer and epoch start fresh.

The first 20 epochs train the aneurysm branch only. Epochs 20 through 249
also train the last two encoder blocks and the last two available shared
decoder blocks (including transposed convolutions). Branch LR starts at
0.001 and feature LR at 0.0001, both with polynomial decay. The vessel and
classification branches stay frozen. Changing shared features can still
change their outputs: use the original VesselOnly checkpoint for downstream
vessel prediction until the V2 vessel outputs are separately evaluated.

Every 10 completed epochs and at the final epoch, deterministic, unaugmented
full-ROI validation selects `checkpoint_best.pth` using binary micro Dice
at argmax (equivalent to probability 0.5 except ties). Patch EMA is diagnostic
only. `best_full_validation.json` records the selected epoch and diagnostics;
`full_validation_history.jsonl` records all selection checks. The inherited
checkpoint field `_best_ema` stores full-ROI Dice in this trainer for resume
compatibility. Final validation uses the selected best checkpoint via
`--val_best`. `selection_validation/` contains the most recent periodic masks,
not necessarily the best masks; use root `validation_aneurysm_masks/` after
the launch script completes for the best-checkpoint predictions.

These metrics are measured on resized Dataset505 ROIs, not native-space
official Task2 masks. The inherited HD95 uses the ROI image spacing and
excludes empty mismatches; do not treat it as native millimetre performance.
Compare mean positive-case Dice, lesion recall and size-stratified failures
in addition to micro Dice. Selecting checkpoints on fold-0 validation is
development tuning, not an independent estimate of improvement.

Run on a dedicated GPU (memory can rise after unfreezing):

```bash
conda activate topaneu_bravecowcow
cd /data/cyf/codes/TopAneu/baseline/BraveCoWCoW
bash topaneu/train_aneurysm_v2_fold0.sh 7
```

The new trainer name creates a separate result directory. The script refuses
to restart into a nonempty V2 fold-0 directory. No threshold tuning, hard vessel
filter, changed loss, or new preprocessing is included in this pilot. The
CPU initialization/phase/LR/selection checks passed; full GPU training and
peak-memory measurement remain to be run.
