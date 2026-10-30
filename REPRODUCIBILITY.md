# Reproducibility Guide

## Hardware Requirements
- One 24 GB GPU (e.g., A10, L4, RTX 3090) for AWQ and NF4.
- Two GPUs or a 40+ GB VRAM GPU for FP16.

## Environment Setup
```bash
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
pip install -r requirements.txt
```
# Running the Pipeline
To reproduce the inference and analysis from scratch:
```bash
bash scripts/reproduce_all.sh
```

#Determinism
The analysis script uses seed=42 for all random sampling and bootstrapping. While exact float reductions may vary slightly across different GPU architectures (which can shift CI bounds by the third decimal place), the point estimates and significance levels are stable and reproducible.
