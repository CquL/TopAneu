#!/usr/bin/env bash
set -euo pipefail

repo_root="/data/cyf/codes/TopAneu/baseline/BraveCoWCoW"
source "${repo_root}/topaneu/topaneu_predroi_env.sh"

python "${repo_root}/topaneu/setup_predroi_160.py" \
  --release /data/cyf/shared_data/TopAneu/topaneu_release \
  --stage1-root /data/cyf/shared_data/TopAneu/BraveCoWCoW_stage1_roi \
  --data-root "${TOPANEU_BRAVECOWCOW_DATA}" \
  --template-plans "${repo_root}/nnXNetResEncUNetM_two_seg_with_cls_ps_224_224_224_Plans.json" \
  --splits /data/cyf/shared_data/TopAneu/BraveCoWCoW_data_160/nnXNet_preprocessed/Dataset503_TopAneuBraveCoWCoW/splits_final.json \
  --workers "${TOPANEU_PREPROCESS_WORKERS:-8}"

nnXNet_preprocess -d 505 \
  -plans_name TopAneuBraveCoWCoWPredROIPlansV2_160 \
  -c 3d_fullres -np "${TOPANEU_NNXNET_PREPROCESS_WORKERS:-4}"

