#!/bin/bash
set -e
ENV_DIR=${ENV_DIR:-/env}
pip install --target="$ENV_DIR" --upgrade --no-deps \
    "transformers==4.51.3" "tokenizers==0.21.0" "huggingface-hub==0.36.0" "datasets==3.2.0" \
    "numpy==2.2.6" "scipy==1.15.3" "bitsandbytes==0.50.2" "accelerate==1.15.0"
pip install --target="$ENV_DIR" --no-deps "autoawq==0.2.9"
echo "Installed into $ENV_DIR. Verify with:  python -m ptq_trust.generate --recipe FP16 --limit 5"
