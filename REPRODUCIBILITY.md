# Reproducibility

## 1. Re-run the analysis on the released data (no GPU, ~10 min)
```bash
pip install -r requirements.txt
make test        # 51 tests, incl. re-deriving every parse-dependent column in the released CSVs from the stored generations
make analysis    # writes results/analysis_results.txt and results/summary.json  (20,000 bootstrap resamples, seed 42)
make figure      # writes results/fig_dissociation_v2.png
```
`make analysis-fast` runs the same code with fewer resamples as a smoke test.

## 2. Regenerate the per-item data (GPU, about 14 GPU-hours)
Hardware used: Kaggle, 2x NVIDIA T4 (15.6 GB each). FP16 and NF4 use `device_map="auto"` (sharded over both GPUs);
AWQ is forced onto `cuda:0`.

```bash
bash scripts/install_env_kaggle.sh                 # pinned environment, see requirements-gpu.txt
python -m ptq_trust.generate --recipe FP16 --limit 5   # smoke test; inspect gen_text, parse_method, finished
bash scripts/run_inference.sh                      # FP16, NF4, AWQ_W4 -> runs/out_*.csv (resumable)
```
Speed from 5-question smoke tests on 2x T4 (full-run timings were not logged): FP16 about 7.5 s/question, NF4 about 9 s, AWQ about 33 s, i.e. roughly 2 h, 2.5 h and 9 h per 1000 questions. Treat these as estimates.
`ptq_trust.generate` **aborts** if the loaded transformers is not 4.51.3 (override with `--allow_any_version`).

## 2b. Noise-floor controls and a determinism check
```bash
bash scripts/run_controls.sh                      # BF16 and INT8 by default (FP16_EAGER is invalid for Qwen2.5 in fp16)
# `make analysis` already includes the released controls (data/per_item/out_BF16.csv, out_INT8.csv)
```
Determinism check (cheap): re-run 20 FP16 questions and compare with the released rows. If `gen_text` is identical, the
pipeline is bitwise reproducible on your hardware and the FP16_EAGER control isolates kernel differences only.
```bash
python -m ptq_trust.generate --recipe FP16 --limit 20 --out_dir /tmp/rerun
python -c "import pandas as pd; a=pd.read_csv('data/per_item/out_FP16.csv').set_index('idx'); b=pd.read_csv('/tmp/rerun/out_FP16.csv').set_index('idx'); print((a.loc[b.index,'gen_text']==b.gen_text).mean())"
```

## 3. What is and is not bit-for-bit reproducible
* Decoding is greedy, so there is no sampling randomness. All bootstrap/permutation randomness is seeded (`--seed 42`).
* **Determinism check (10 questions, idx 0-9, 2x T4):** two independent FP16 reruns gave identical generations and
  identical logits (10/10 each), and the generations were also identical to the released FP16 rows (10/10) even though the
  rerun used torch 2.11.0+cu130 and the released run torch 2.11.0+cu128. This is a small sample (logits against the
  released rows were not compared), so it suggests, but does not prove, that the CUDA build has little effect.
* GPU floating-point reductions depend on hardware, driver and library versions. We saw the same item give a final
  margin of 12.59 under transformers 5.16.1 and 12.55 under 4.51.3 (different Kaggle sessions), so **re-generated logits will differ
  slightly**, and a few borderline answers may flip. Expect the same qualitative picture, not identical decimals.
  Compare against `data/per_item/` rather than expecting a byte-identical file.
* The released CSVs were produced with: transformers 4.51.3, torch 2.11.0+cu128, bitsandbytes 0.50.2, autoawq 0.2.9,
  accelerate 1.15.0, datasets 3.2.0 (see `data/per_item/run_info_*.json`). AutoAWQ is deprecated; its own notice lists
  torch 2.6.0 / transformers 4.51.3 as the last tested configuration, so this combination is outside its tested range.
* NF4 (on-the-fly bitsandbytes) and AWQ (a public pre-quantized checkpoint, `Qwen/Qwen2.5-7B-Instruct-AWQ`) differ in
  toolchain, calibration data and kernels, not only in the quantization algorithm.
* The analysis uses 15 equal-width ECE bins, a 2-fold cross-fit with folds assigned per item before resampling, and
  Holm correction over the six primary tests; see the docstring at the top of `ptq_trust/analysis.py`.
