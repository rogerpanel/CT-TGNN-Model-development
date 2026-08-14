#!/usr/bin/env bash
# Full experiment sweep: every model x every seed.
# Nothing is reported until every run has written scores_test.npz.
set -euo pipefail

CONFIG="${CONFIG:-configs/default.yaml}"
SEEDS="${SEEDS:-0 1 2 3 4}"
MODELS="${MODELS:-edgebank mlp_flow cnn_lstm discrete_gnn temporal_attn ct_tgnn}"
OUT="${OUT:-runs}"

echo "config=$CONFIG  seeds=[$SEEDS]  models=[$MODELS]"
for m in $MODELS; do
  for s in $SEEDS; do
    if [ -f "$OUT/${m}_seed${s}/scores_test.npz" ]; then
      echo "skip ${m} seed ${s} (already done)"; continue
    fi
    echo "=== ${m} seed ${s} ==="
    python -m src.train --config "$CONFIG" --model "$m" --seed "$s" --out-root "$OUT"
  done
done

python -m src.report --runs "$OUT" --out paper
echo
echo "Done. Tables and figures in paper/ -- every value recomputed from score files."
