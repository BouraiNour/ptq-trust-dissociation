# Quantization and the Trustworthiness of Medical LLMs: A Paired Audit

**Version:** 2.0.0 (Corrected Release)  
**Status:** Under review  
**Authors:** Nour El Houda Bourai, A. Djebbar, H. F. Merouani

## Abstract

Post-training quantization (PTQ) is widely used to deploy large language models in clinical settings, but standard checks of accuracy and recalibrated calibration may miss changes in how a model relies on its own reasoning. We audit Qwen2.5-7B-Instruct on 1000 four-option MedQA test questions under FP16, NF4, and AWQ, using paired bootstrap inference with Holm correction. 

Strict accuracy falls by 2.0 percentage points under NF4 and by 3.6 under AWQ. All three models are severely overconfident (raw ECE ~0.37); after temperature scaling, ECE is ~0.04 for all, but the confidence intervals are too wide to prove calibration is preserved. A leave-one-sentence-out faithfulness score decreases significantly under NF4 (−0.72 nats) and less under AWQ (−0.33 nats). 

These differences largely track a global shrinkage of logit margins. Net accuracy conceals massive answer churn: the written answer differs from FP16 on 23% (NF4) and 26% (AWQ) of questions, against a 10% bfloat16 numerical-noise floor and 15% under 8-bit weights. Only the 4-bit models shrink the final logit margin by ~15%. We conclude that accuracy and recalibrated ECE are insufficient to characterize quantized medical LLMs, but that faithfulness estimates require explicit scale and length controls before they can be interpreted.

## Results

**N = 1000 questions per model; faithfulness n = 985.**
*(All confidence intervals are 95% paired bootstrap intervals, 20,000 resamples, seed=42).*

| Metric | FP16 | NF4 | AWQ |
| :--- | :--- | :--- | :--- |
| **Strict accuracy** | 62.9% | 60.9% | 59.3% |
| *Change vs FP16 (pp)* | - | -2.0 [-4.6, +0.6] | -3.6 [-6.4, -0.8] |
| **Raw ECE** (no recalibration) | 0.366 | 0.378 | 0.392 |
| **Cross-fitted ECE** (after T*≈12) | 0.051 | 0.049 | 0.041 |
| *Change vs FP16* | - | -0.002 [-0.036, +0.035] | -0.011 [-0.043, +0.028] |
| **Faithfulness** (nats) | 4.16 | 3.44 | 3.84 |
| *Change vs FP16* | - | **-0.72** [-0.98, -0.48] | -0.33 [-0.58, -0.08] |
| **Answer differs from FP16** | - | 23.3% [20.7, 26.0] | 26.0% [23.3, 28.7] |
| **Final logit margin change** | - | -2.74 [-3.15, -2.33] | -2.91 [-3.34, -2.49] |

### Noise-Floor Controls

| Metric | BF16 (Compute Control) | INT8 (8-bit Weights) | NF4 | AWQ |
| :--- | :--- | :--- | :--- | :--- |
| **Answer differs from FP16** | 9.9% | 14.7% | 23.3% | 26.0% |
| **Excess flips over BF16** | - | - | +13.4 pp | +16.1 pp |
| **Δ Strict accuracy (pp)** | -0.4 | +0.3 | -2.0 | -3.6 |
| **Δ Final margin (nats)** | +0.03 | -0.15 | -2.74 | -2.91 |
| **Δ Faithfulness (nats)** | -0.04 | -0.02 | -0.72 | -0.33 |

## Findings

*   NF4 reduces a leave-one-sentence-out faithfulness score by about 0.72 nats, an effect robust to test choice, medians, and correctness strata.
*   All three models are severely overconfident (raw ECE 0.37–0.39). Temperature scaling hides this.
*   Net accuracy hides answer churn. Both 4-bit models change 13–16 points more answers than the BF16 numerical noise floor.
*   The shrinkage of final logit margins and the NF4 faithfulness decrease are specific to 4-bit weights; BF16 and INT8 leave both unchanged.

## Quick Start (No GPU Required)

You can verify all parsing logic and run the statistical analysis on the provided per-item data without a GPU.

```bash
git clone https://github.com/BouraiNour/ptq-trust-dissociation.git
cd ptq-trust-dissociation
pip install -r requirements.txt
```

make test        # Runs 51 tests (parsing, metrics, data consistency)
make analysis    # Runs 20,000-resample paired bootstrap (seed 42) -> results/
make figure      # Regenerates the plots


## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
