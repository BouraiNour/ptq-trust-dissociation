.PHONY: install test analysis analysis-fast figure all clean

install:
	pip install -r requirements.txt

test:
	python -m pytest -q tests

# Re-runs the full paired analysis on the released per-item data (CPU only, ~10 min because of the
# 20,000-resample cross-fitted ECE bootstrap). Use `make analysis-fast` for a quick smoke run.
analysis:
	python -m ptq_trust.analysis --fp16 data/per_item/out_FP16.csv --nf4 data/per_item/out_NF4.csv \
	    --awq data/per_item/out_AWQ_W4.csv --control BF16=data/per_item/out_BF16.csv \
	    --control INT8=data/per_item/out_INT8.csv --out results/analysis_results.txt

analysis-fast:
	python -m ptq_trust.analysis --fp16 data/per_item/out_FP16.csv --nf4 data/per_item/out_NF4.csv \
	    --awq data/per_item/out_AWQ_W4.csv --n_boot 2000 --n_boot_ece 500 --out /tmp/analysis_fast.txt

figure:
	python -m ptq_trust.make_figure --summary results/summary.json --out results/fig_dissociation_v2.png

all: test analysis figure

clean:
	rm -f results/analysis_results.txt results/summary.json results/fig_dissociation_v2.png
