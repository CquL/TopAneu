This repository contains our solution to the [RSNA Intracranial Aneurysm Detection](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection) Kaggle challenge. We placed 7th out of 1147 teams.

Our writeup can be found [here](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/writeups/7th-place-solution)

The repository you see here is a fork of [nnU-Net](https://github.com/MIC-DKFZ/nnUNet). Please head over there to read more about it.

# Installation
We strongly recommend installing this in a dedicated virtual environment (for example conda).
We recommend using a Linux based operating system, for example Ubuntu. Windows should work as well but is not tested.

Some dependencies should be installed manually:
- Install pytorch according to the instructions on the [pytorch website](https://pytorch.org/get-started/locally/). We recommend at least version 2.7. Pick the correct CUDA version for your system. Higher is better.
- Install batchgeneratorsv2 via `pip install git+https://github.com/MIC-DKFZ/batchgeneratorsv2.git@07541d7eb5a4839aa4a5e494a123f3fe69ccfd4f`

Now you can just clone this repository and install it:

```commandline
git clone https://github.com/MIC-DKFZ/kaggle-rsna-intracranial-aneurysm-detection-2025-solution.git
cd kaggle-rsna-intracranial-aneurysm-detection-2025-solution
pip install -e .
```

# Inference
Download the [model checkpoint](https://www.kaggle.com/datasets/st3v3d/rsna-2025-7th-place-checkpoint/data) and extract it. 

Inference script is provided at [nnunetv2/inference/kaggle_2025_rsna/inference.py](nnunetv2/inference/kaggle_2025_rsna/inference.py).

Example usage:
```bash
  python inference.py \
     -i /path/to/rsna-intracranial-aneurysm-detection/series \
     -o output.csv \
     -m /path/to/downloaded-checkpoint/Dataset004_iarsna_crop/Kaggle2025RSNATrainer__nnUNetResEncUNetMPlans__3d_fullres_bs32 \
     -c checkpoint_epoch_1500.pth \
     --fold "('all',)"
     --disable_tta
```

# Training
Here is how to reproduce the model training.

## Path setup
nnU-Net requires environment variables pointing it to raw data, preprocessed data and results. Set them with

```
export nnUNet_results=/path/to/nnUNet_results
export nnUNet_preprocessed=/path/to/nnUNet_preprocessed
export nnUNet_raw=/path/to/nnUNet_raw
```
Make sure at least `$nnUNet_preprocessed` (but ideally all of them) are on a fast storage such as a local SSD or very good network drive! 

## Dataset download
1. Download the [dataset](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/data). 
2. Convert the data to the nnUNet_raw format (this includes a 200x160x160 mm RoI crop): 

```
python nnunetv2/dataset_conversion/kaggle_2025_rsna/official_data_to_nnunet.py \
  -i /path/to/rsna-intracranial-aneurysm-detection \
  -o $nnUNet_raw/Dataset004_iarsna_crop
```


## nnUNet experiment planning and preprocessing
Run the following commands (anywhere on your system)

```nnUNetv2_extract_fingerprint -d 004 -np 64```\
This will extract a 'dataset fingerprint' that nnU-Net uses for autoconfig. Set -np to a reasonable number of processes. Default here is 64. More is better but eats more RAM and I/O

```nnUNetv2_plan_experiment -d 004 -pl nnUNetPlannerResEncM```\
This will generate an automatically configured nnU-Net pipeline to use for your dataset. Usually we would just use it but for the competition we made some manual changes (larger batch and patch size and necessary adjustments to network topology).

Copy our manually adjusted nnU-Net plans from [nnunetv2/dataset_conversion/kaggle_2025_rsna/plans/nnUNetResEncUNetMPlans.json](nnunetv2/dataset_conversion/kaggle_2025_rsna/plans/nnUNetResEncUNetMPlans.json) to `$nnUNet_preprocessed/Dataset004_iarsna_crop`. This should overwrite a file with the same name.

Now you can perform preprocessing:\
```nnUNetv2_preprocess -d 004 -np 64 -c 3d_fullres_bs32 -p nnUNetResEncUNetMPlans```\
Again steer the number of processes used with `-np`. Sit back and grab a cup of coffee, this may take a while

## Training
Training our final model requires 4 GPUs with at least 40GB VRAM each. Maybe 32GB will work as well - no guarantee though!

```nnUNet_n_proc_DA=16 nnUNetv2_train 004 3d_fullres_bs32 all -tr Kaggle2025RSNATrainer -num_gpus 4 -p nnUNetResEncUNetMPlans```\
Note that nnUNet_n_proc_DA=16 steers the number of data augmentation workers per GPU. Adjust to your system.

On 4x Nvidia A100 (40GB) PCIe (SXM will be faster) the training should take ~130s per epoch for a total of 4.5 days. If your training is slower than that, check for CPU or I/O bottlenecks.

Done. Your final checkpoint will be located at `$nnUNet_results/Dataset004_iarsna_crop/Kaggle2025RSNATrainer__nnUNetResEncUNetMPlans__3d_fullres_bs32`

IMPORTANT: Trainings in nnU-Net are not seeded, so you are unlikely to get exactly the same result. The result will be comparable. Maybe slightly better, maybe slightly worse. 
