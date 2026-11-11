#!/bin/bash
set -e
OUT=${OUT:-runs}
mkdir -p "$OUT"
for R in ${CONTROLS:-BF16 INT8}; do
    python -m ptq_trust.generate --recipe "$R" --out_dir "$OUT" --env_path "${ENV_DIR:-/env}"
done
echo "Analyse (baseline = the released FP16 run):"
echo "  python -m ptq_trust.analysis --fp16 data/per_item/out_FP16.csv --nf4 data/per_item/out_NF4.csv --awq data/per_item/out_AWQ_W4.csv --control BF16=$OUT/out_BF16.csv --control INT8=$OUT/out_INT8.csv --out $OUT/analysis_results.txt"
