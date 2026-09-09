#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 1 ]]; then
    echo "Usage: bash topaneu/train_aneurysm_v2_fold0.sh GPU_ID"
    exit 2
fi
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo"
source topaneu/topaneu_predroi_env.sh
export PYTHONPATH="$repo:$repo/nnXNet:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="$1"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Avoid graph recompilation across freezing and full-validation transitions.
export nnXNet_compile=false
vessel="$nnXNet_results/Dataset505_TopAneuBraveCoWCoWPredROI/nnXNetTrainer_TopAneuVesselOnly__TopAneuBraveCoWCoWPredROIPlansV2_160__3d_fullres/fold_0/checkpoint_best.pth"
out="$nnXNet_results/Dataset505_TopAneuBraveCoWCoWPredROI/nnXNetTrainer_TopAneuAneurysmV2__TopAneuBraveCoWCoWPredROIPlansV2_160__3d_fullres/fold_0"
[[ -f "$vessel" ]] || { echo "Missing vessel initialization: $vessel"; exit 1; }
if [[ -d "$out" ]] && [[ -n "$(ls -A "$out")" ]]; then
    echo "V2 output already exists: $out. Inspect before restarting or resuming."
    exit 1
fi
nnXNet_train Dataset505_TopAneuBraveCoWCoWPredROI 3d_fullres 0 \
    -tr nnXNetTrainer_TopAneuAneurysmV2 -p TopAneuBraveCoWCoWPredROIPlansV2_160 \
    -pretrained_weights "$vessel" --val_best
