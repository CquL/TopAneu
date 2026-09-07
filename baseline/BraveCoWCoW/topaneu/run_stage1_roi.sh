#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

gpu="${1:-0}"
shift || true

export CUDA_VISIBLE_DEVICES="${gpu}"
python -u topaneu/stage1_roi.py --device cuda:0 "$@"
