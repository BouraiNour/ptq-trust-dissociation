# Reproducibility Guide

## Hardware Requirements
- One 24 GB GPU (e.g., A10, L4, RTX 3090) for AWQ and NF4.
- Two GPUs or a 40+ GB VRAM GPU for FP16.

## Environment Setup
```bash
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
pip install -r requirements.txt
