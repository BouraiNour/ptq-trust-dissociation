#!/bin/bash
set -e

echo "=== Running Inference (takes ~3-4 hrs per recipe) ==="
bash scripts/run_inference.sh

echo "=== Running Analysis & Plotting ==="
python -m src.analysis --config configs/base.yaml

echo "=== Reproducibility Complete. Check results/ ==="
