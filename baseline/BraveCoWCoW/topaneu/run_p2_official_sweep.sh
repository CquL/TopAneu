#!/usr/bin/env bash
set -euo pipefail

ROOT="${TOPANEU_P2_ROOT:-/data/cyf/shared_data/TopAneu/BraveCoWCoW_predroi_160}"
EVAL_ROOT="$ROOT/stagec_pred_oof_v2/evaluations"
GT=/data/cyf/shared_data/TopAneu/topaneu_release/location_masks
TRANSFORMS="$ROOT/metadata/roi_transforms"
OFFICIAL=/data/cyf/codes/TopAneu/TopAneu-26-main/eval/task2/evaluate.py
SITE="${TOPANEU_EVAL_SITE:-/data/cyf/topaneu_task2_eval_site_311}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
export PYTHONPATH="$SITE:$REPO/baseline/BraveCoWCoW:$REPO/baseline/BraveCoWCoW/nnXNet:${PYTHONPATH:-}"
SUMMARY="$EVAL_ROOT/official_sweep_summary.csv"
printf 'rule,precision,recall,f1,mcc,dice,hd95,volsim\n' > "$SUMMARY"

for rule in "$EVAL_ROOT"/threshold_*_minvox_*; do
  [[ -d "$rule" ]] || continue
  name="$(basename "$rule")"
  output="$rule/official_task2_metrics.json"
  if [[ ! -s "$output" ]]; then
    python -u "$REPO/baseline/BraveCoWCoW/topaneu/evaluate_task2_official.py" \
      --results "$rule" --prediction-subdir validation_location_masks_component \
      --metadata "$TRANSFORMS" --ground-truth "$GT" --official-evaluate "$OFFICIAL" \
      --output "$output" --workers "${TOPANEU_EVAL_WORKERS:-8}"
  else
    echo "cached: $name"
  fi
  python -c 'import json,sys; x=json.load(open(sys.argv[2]))["aggregates_avg"]; print(",".join([sys.argv[1]]+[str(x.get(k,"nan")) for k in ["PRECISION","RECALL","F1","MCC","DICE","HD95","VOLSIM"]]))' "$name" "$output" >> "$SUMMARY"
done

python -c 'import csv,sys; rows=list(csv.DictReader(open(sys.argv[1]))); rows.sort(key=lambda r: float(r["dice"]), reverse=True); [print(r) for r in rows]' "$SUMMARY"
echo "saved: $SUMMARY"
