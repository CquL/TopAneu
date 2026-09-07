#!/usr/bin/env bash
# Source this file: source topaneu/topaneu_bravecowcow_env.sh

export TOPANEU_BRAVECOWCOW_CODE="/data/cyf/codes/TopAneu/baseline/BraveCoWCoW"
export PYTHONPATH="${TOPANEU_BRAVECOWCOW_CODE}:${TOPANEU_BRAVECOWCOW_CODE}/nnXNet${PYTHONPATH:+:${PYTHONPATH}}"
export TOPANEU_BRAVECOWCOW_DATA="/data/cyf/shared_data/TopAneu/BraveCoWCoW_data_160"
export nnXNet_raw="${TOPANEU_BRAVECOWCOW_DATA}/nnXNet_raw"
export nnXNet_preprocessed="${TOPANEU_BRAVECOWCOW_DATA}/nnXNet_preprocessed"
export nnXNet_results="${TOPANEU_BRAVECOWCOW_DATA}/nnXNet_results"
export TOPANEU_BRAVECOWCOW_METADATA="${TOPANEU_BRAVECOWCOW_DATA}/metadata/topaneu_multitask_labels.csv"
export TOPANEU_BRAVECOWCOW_SAMPLING_WEIGHTS="${TOPANEU_BRAVECOWCOW_DATA}/metadata/train_case_sampling_weight.csv"
export TOPANEU_STAGE1_DATA="/data/cyf/shared_data/TopAneu/BraveCoWCoW_stage1_roi"
export TOPANEU_STAGE1_MODEL_DIR="/data/cyf/codes/TopAneu/submission/BraveCoWCoW_Task2/model/stage1_roi/Dataset180_2D_vessel_box_seg_stable/nnUNetTrainer__nnUNetPlans__2d"
