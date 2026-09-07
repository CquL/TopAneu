# TopAneu-26 Task 1 with the MIC-DKFZ RSNA heatmap model

This adaptation is intentionally isolated from the sibling nnUNet and nnXNet
source trees. It uses dataset ID 502 and its own raw, preprocessed and results
roots under `/data/cyf/shared_data/TopAneu/MIC_DKFZ_RSNA_data`.

The large preprocessed arrays are reused through a read-only directory symlink.
Plans, dataset metadata, cross-validation splits and training results are owned
by the MIC-DKFZ adaptation and do not modify Dataset501.

## Environment

Use the dedicated environment:

```bash
conda activate topaneu_mic
source /data/cyf/codes/TopAneu/baseline/MIC_DKFZ_RSNA/set_topaneu_mic_env.sh
```

## Paths

```bash
export nnUNet_raw=/data/cyf/shared_data/TopAneu/MIC_DKFZ_RSNA_data/nnUNet_raw
export nnUNet_preprocessed=/data/cyf/shared_data/TopAneu/MIC_DKFZ_RSNA_data/nnUNet_preprocessed
export nnUNet_results=/data/cyf/shared_data/TopAneu/MIC_DKFZ_RSNA_data/nnUNet_results
```

## Training

For one fold per GPU, use the single-GPU batch-4 configuration:

```bash
CUDA_VISIBLE_DEVICES=1 nnUNet_n_proc_DA=8 \
nnUNetv2_train 502 3d_fullres_bs4 0 \
  -tr TopAneuMICTrainer -p TopAneuMICPlans
```

Run all five folds (0 through 4) to produce out-of-fold predictions for
threshold calibration. Multiple folds may run concurrently on separate GPUs.

For four-GPU DDP training on one fold, use the global-batch-16 configuration:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 nnUNet_n_proc_DA=8 \
nnUNetv2_train 502 3d_fullres 0 \
  -tr TopAneuMICTrainer -p TopAneuMICPlans -num_gpus 4
```

After checkpoint and threshold selection, train on all cases:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 nnUNet_n_proc_DA=8 \
nnUNetv2_train 502 3d_fullres all \
  -tr TopAneuMICTrainer -p TopAneuMICPlans -num_gpus 4
```

The plan uses a global batch size of 16. With four GPUs this is four samples per
GPU. TopAneu has 53 output channels (52 locations plus the auxiliary any-
aneurysm channel), so the original RSNA global batch size of 32 is not used by
default.
