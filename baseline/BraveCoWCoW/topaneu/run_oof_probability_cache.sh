#!/usr/bin/env bash
set -euo pipefail

# Run from the dedicated TopAneu environment.  The default output is a new
# stagec_pred_oof_v2 directory and therefore never overwrites older OOF runs.
source /data/miniconda/etc/profile.d/conda.sh
conda activate topaneu_bravecowcow
cd /data/cyf/codes/TopAneu/baseline/BraveCoWCoW
source topaneu/topaneu_predroi_env.sh
export PYTHONPATH="$PWD:$PWD/nnXNet:${PYTHONPATH:-}"
export nnXNet_compile=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

exec python -u topaneu/build_oof_probability_cache.py "$@"
