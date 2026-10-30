#!/bin/bash
set -e

# Ensure results directory exists
mkdir -p results

# Run for each recipe
python -m src.evaluation --config configs/fp16.yaml
python -m src.evaluation --config configs/nf4.yaml
python -m src.evaluation --config configs/awq_w4.yaml
