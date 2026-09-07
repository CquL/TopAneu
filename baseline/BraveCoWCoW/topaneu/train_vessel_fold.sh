#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "Usage: $0 FOLD GPU_ID [PRETRAINED_CHECKPOINT]" >&2
  exit 2
fi

fold="$1"
gpu="$2"
pretrained="${3:-}"
source /data/cyf/codes/TopAneu/baseline/BraveCoWCoW/topaneu/topaneu_predroi_env.sh
export CUDA_VISIBLE_DEVICES="${gpu}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

command=(
  nnXNet_train Dataset505_TopAneuBraveCoWCoWPredROI 3d_fullres "${fold}"
  -tr nnXNetTrainer_TopAneuVesselOnly
  -p TopAneuBraveCoWCoWPredROIPlansV2_160
)
if [[ -n "${pretrained}" ]]; then
  command+=( -pretrained_weights "${pretrained}" )
fi
"${command[@]}"
