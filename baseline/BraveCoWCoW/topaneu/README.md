# TopAneu-26 configuration

This directory is an isolated adaptation of BraveCoWCoW for the TopAneu
release. It does not read or write the nnUNet Dataset501 or MIC-DKFZ Dataset502
directories.

## Design

- Input: vessel ROI resized to `160 x 160 x 160` for the lower-memory V2 experiment.
- Shared encoder, two segmentation decoders, and three classification heads.
- Decoder 1: background + 36 dense vessel-anatomy classes.
- Decoder 2: background + binary aneurysm segmentation.
- Classification heads: aneurysm presence (1), location (52), modality (CTA/MRA, 2).
- Location IDs are assigned per connected aneurysm component using the 36-class
  vessel topology and 52-way classifier; they are not trained as sparse
  voxel-wise segmentation classes.
- Training length: 250 epochs, batch size 4 for a 48 GB A40 with 160³ ROI.
- Dataset ID: `Dataset503_TopAneuBraveCoWCoW`.
- Five patient-grouped folds are generated locally (seed 2026); no Dataset501
  files are required at setup or training time.

The supplied TopAneu vessel masks are used only to construct the training ROI.
At challenge inference time, stage 1 must predict this ROI before stage 2; using
the ground-truth vessel mask at inference would leak labels.

## Published Stage-1 ROI model

The original two-stage architecture uses the published
`Dataset180_2D_vessel_box_seg_stable` checkpoint before this 3-D model. The
checkpoint already available in the local submission bundle is referenced via
`TOPANEU_STAGE1_MODEL_DIR`; it is not copied into this source tree.

Run a smoke test on one case, then evaluate its crop:

```bash
bash topaneu/run_stage1_roi.sh 3 --limit 1 --save-mask
python topaneu/evaluate_stage1_roi.py
```

Run all 416 cases (resumable because existing JSON transforms are skipped):

```bash
bash topaneu/run_stage1_roi.sh 3
python topaneu/evaluate_stage1_roi.py
```

Outputs are isolated below
`/data/cyf/shared_data/TopAneu/BraveCoWCoW_stage1_roi`. The default 5%/minimum
4-voxel bbox expansion matches the current submission inference implementation.

After ROI coverage is accepted, build the independent Stage-2 predicted-ROI
dataset locally on the training host:

```bash
source topaneu/topaneu_predroi_env.sh
bash topaneu/prepare_predroi_and_preprocess.sh
```

This creates `Dataset505_TopAneuBraveCoWCoWPredROI` below
`/data/cyf/shared_data/TopAneu/BraveCoWCoW_predroi_160`; Dataset503 remains
untouched.

## Environment

Create the dedicated environment once:

```bash
bash topaneu/create_env.sh
conda activate topaneu_bravecowcow
source topaneu/topaneu_bravecowcow_env.sh
```

## Data conversion and preprocessing

```bash
bash topaneu/prepare_and_preprocess.sh
```

The output root is
`/data/cyf/shared_data/TopAneu/BraveCoWCoW_data`. The conversion is resumable.

## Train one fold on one GPU

```bash
bash topaneu/train_fold.sh 0 0
```

Run folds 0 through 4 separately. Results are written below the isolated
`nnXNet_results` directory.

## Staged Stage-2 training

Stage 2-A trains the shared encoder and 36-class vessel decoder:

```bash
bash topaneu/train_vessel_fold.sh FOLD GPU PREVIOUS_TOPANEU_CHECKPOINT
```

Stage 2-B loads the matching Stage 2-A `checkpoint_best.pth`, freezes the
encoder/shared/vessel/classification modules, and optimizes only the merged
background-versus-aneurysm decoder:

```bash
bash topaneu/train_aneurysm_fold.sh FOLD GPU VESSEL_CHECKPOINT
```

Both stages use 250 epochs and write to separate trainer result directories.
