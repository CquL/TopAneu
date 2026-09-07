#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 FOLD GPU_ID" >&2
  exit 2
fi

fold="$1"
gpu="$2"
source /data/cyf/codes/TopAneu/baseline/BraveCoWCoW/topaneu/topaneu_bravecowcow_env.sh
export CUDA_VISIBLE_DEVICES="${gpu}"

nnXNet_train Dataset503_TopAneuBraveCoWCoW 3d_fullres "${fold}" \
  -tr nnXNetTrainer_TopAneu52 \
  -p TopAneuBraveCoWCoWPlansV2_160
