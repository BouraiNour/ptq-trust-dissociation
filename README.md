# Calibration-Faithfulness Dissociation Under 4-bit Quantization

**Paper:** Calibration-Faithfulness Dissociation Under 4-bit Quantization: Evidence from Qwen2.5-7B on MedQA  
**Status:** Under review (CAIA'26)  
**Authors:** Nour El Houda Bourai, A. Djebbar, H. F. Merouani

## Abstract

Post-training quantization (PTQ) is essential for deploying LLMs in clinical environments, but standard metrics — accuracy and calibration — fail to capture shifts in trust properties. In an audit of 743 paired samples from MedQA, we demonstrate a dissociation between calibration and explanation faithfulness under 4-bit quantization in Qwen2.5-7B. NF4 preserves both accuracy and calibration while causing a statistically significant degradation in rationale faithfulness (−0.62 nats, 95% CI [−0.93, −0.32]). AWQ significantly degrades accuracy (−5.10 pp, 95% CI [−8.08, −2.15]) and faithfulness (−0.33 nats, 95% CI [−0.64, −0.02]), while calibration is statistically indistinguishable from FP16. These results show that calibration metrics alone are insufficient for certifying quantized medical LLMs: models can remain calibrated while losing either accuracy or the causal fidelity of their reasoning. We release per-sample data and paired-bootstrap tooling for replication.

## Results

| Recipe | Δ Accuracy (pp) | Δ ECE | Δ Faithfulness (nats) |
| :--- | :--- | :--- | :--- |
| **NF4** | -1.75 [-4.98, +1.35] | -0.0074 [-0.0488, +0.0342] | **-0.62** [-0.93, -0.32] |
| **AWQ_W4** | **-5.10** [-8.08, -2.15] | -0.0089 [-0.0502, +0.0333] | **-0.33** [-0.64, -0.02] |

*All confidence intervals are 95% paired bootstrap intervals (20,000 resamples, seed=42).*

## Installation

```bash
git clone https://github.com/BouraiNour/ptq-trust-dissociation.git
cd ptq-trust-dissociation
pip install -r requirements.txt
```
# Reproducing

**Hardware:** This pipeline was developed and validated on 2× NVIDIA T4 (16 GB each).

- **FP16:** Requires ~2× T4 or a single GPU with 40+ GB VRAM. We used `device_map="auto"` to shard across both GPUs.
- **NF4:** Runs on a single T4. We used `device_map="auto"`.
- **AWQ_W4:** Runs on a single T4. We forced `device_map="cuda:0"` to avoid OOM.

To run the full inference and analysis pipeline from scratch:

```bash
bash scripts/reproduce_all.sh
```
> **Note:** The inference script checkpoints every 50 samples, making it safe to interrupt and resume.

## Repository Layout

```text
ptq-trust-dissociation/
├── configs/               # YAML configs for hardware/quantization settings
├── data/                  # Local directory for datasets (see data/README.md)
├── scripts/               # Execution scripts (reproduce_all.sh, run_inference.sh)
├── src/                   # Core source code (data, models, evaluation, analysis)
├── tests/                 # Determinism tests
├── results/               # Pre-computed data, bootstrap results, and figures
├── paper/                 # Link to the CAIA'26 submission
├── REPRODUCIBILITY.md     # Detailed environment and determinism guide
├── requirements.txt       # Python dependencies
├── Makefile               # Shortcuts (make install, make test)
└── README.md
```
## Citation

```bibtex
@inproceedings{bourai2026ptq,
  title={Calibration-Faithfulness Dissociation Under 4-bit Quantization: Evidence from Qwen2.5-7B on MedQA},
  author={Bourai, Nour El Houda and Djebbar, A. and Merouani, H. F.},
  booktitle={CAIA'26},
  year={2026}
}
```
## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---
