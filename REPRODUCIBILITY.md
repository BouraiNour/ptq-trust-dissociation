# Reproducibility Guide

## Hardware Requirements

This pipeline was developed and validated on **2× NVIDIA T4 (16 GB each)**.

- **FP16:** Requires ~2× T4 or a single GPU with 40+ GB VRAM (A100/H100). The 2× T4 setup used `device_map="auto"` to shard the model across both GPUs.
- **NF4:** Runs on a single T4 (16 GB). We used `device_map="auto"`; no sharding was necessary.
- **AWQ_W4:** Runs on a single T4 (16 GB). We forced `device_map="cuda:0"` to avoid OOM.

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

# Determinism
The analysis script uses seed=42 for all random sampling and bootstrapping. While exact float reductions may vary slightly across different GPU architectures (which can shift CI bounds by the third decimal place), the point estimates and significance levels are stable and reproducible.
