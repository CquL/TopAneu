#!/usr/bin/env bash
set -euo pipefail

repo=/data/cyf/codes/TopAneu/baseline/BraveCoWCoW

if ! conda env list | awk '{print $1}' | grep -qx topaneu_bravecowcow; then
  # Ignore stale machine-wide mirror entries (one configured msys2 URL is 404).
  conda create -y -n topaneu_bravecowcow --override-channels -c defaults python=3.11 pip
fi

conda run -n topaneu_bravecowcow python -m pip install \
  'torch==2.6.0' 'torchvision==0.21.0' 'torchaudio==2.6.0' \
  --index-url https://download.pytorch.org/whl/cu124
conda run -n topaneu_bravecowcow python -m pip install -e "${repo}/nnXNet"
