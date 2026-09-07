#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "Usage: $0 FOLD GPU_ID VESSEL_CHECKPOINT" >&2
  exit 2
fi

fold="$1"
gpu="$2"
pretrained="$3"

if [[ ! -f "${pretrained}" ]]; then
  echo "Vessel checkpoint does not exist: ${pretrained}" >&2
  exit 1
fi

source /data/cyf/codes/TopAneu/baseline/BraveCoWCoW/topaneu/topaneu_predroi_env.sh
export CUDA_VISIBLE_DEVICES="${gpu}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

nnXNet_train Dataset505_TopAneuBraveCoWCoWPredROI 3d_fullres "${fold}" \
  -tr nnXNetTrainer_TopAneuAneurysmOnly \
  -p TopAneuBraveCoWCoWPredROIPlansV2_160 \
  -pretrained_weights "${pretrained}"
