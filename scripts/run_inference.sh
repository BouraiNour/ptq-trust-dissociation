set -e
OUT=${OUT:-runs}
mkdir -p "$OUT"
for R in FP16 NF4 AWQ_W4; do
    python -m ptq_trust.generate --recipe "$R" --out_dir "$OUT" --env_path "${ENV_DIR:-/env}"
done
echo "Done. Analyse with:"
echo "  python -m ptq_trust.analysis --fp16 $OUT/out_FP16.csv --nf4 $OUT/out_NF4.csv --awq $OUT/out_AWQ_W4.csv --out $OUT/analysis_results.txt"
