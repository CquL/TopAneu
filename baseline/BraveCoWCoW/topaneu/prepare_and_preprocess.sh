#!/usr/bin/env bash
set -euo pipefail

source /data/cyf/codes/TopAneu/baseline/BraveCoWCoW/topaneu/topaneu_bravecowcow_env.sh

python /data/cyf/codes/TopAneu/baseline/BraveCoWCoW/topaneu/setup_topaneu.py \
  --release /data/cyf/shared_data/TopAneu/topaneu_release \
  --data-root "${TOPANEU_BRAVECOWCOW_DATA}" \
  --template-plans /data/cyf/codes/TopAneu/baseline/BraveCoWCoW/nnXNetResEncUNetM_two_seg_with_cls_ps_224_224_224_Plans.json

nnXNet_preprocess -d 503 -plans_name TopAneuBraveCoWCoWPlansV2_160 -c 3d_fullres -np 4
