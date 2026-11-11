#!/bin/bash
set -e
if [ "$1" == "--with-inference" ]; then
    bash scripts/install_env_kaggle.sh
    OUT=runs bash scripts/run_inference.sh
    D=runs
else
    D=data/per_item
fi
python -m pytest -q tests
python -m ptq_trust.analysis --fp16 $D/out_FP16.csv --nf4 $D/out_NF4.csv --awq $D/out_AWQ_W4.csv --out results/analysis_results.txt
python -m ptq_trust.make_figure --summary results/summary.json --out results/fig_dissociation_v2.png
echo "Done. See results/."
