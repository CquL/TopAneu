#!/usr/bin/env bash
# Source this file for Dataset505 predicted-ROI preprocessing/training.

source /data/cyf/codes/TopAneu/baseline/BraveCoWCoW/topaneu/topaneu_bravecowcow_env.sh
export TOPANEU_BRAVECOWCOW_DATA="/data/cyf/shared_data/TopAneu/BraveCoWCoW_predroi_160"
export nnXNet_raw="${TOPANEU_BRAVECOWCOW_DATA}/nnXNet_raw"
export nnXNet_preprocessed="${TOPANEU_BRAVECOWCOW_DATA}/nnXNet_preprocessed"
export nnXNet_results="${TOPANEU_BRAVECOWCOW_DATA}/nnXNet_results"
export TOPANEU_BRAVECOWCOW_METADATA="${TOPANEU_BRAVECOWCOW_DATA}/metadata/topaneu_multitask_labels.csv"
export TOPANEU_BRAVECOWCOW_SAMPLING_WEIGHTS="${TOPANEU_BRAVECOWCOW_DATA}/metadata/train_case_sampling_weight.csv"

