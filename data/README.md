# Data Directory

This directory is intended for local copies of large datasets (e.g., MedQA). 

The pre-computed outputs are stored in `results/dissociation_1000_final.csv`. 

To regenerate the data from scratch:
1. Ensure your HuggingFace environment is authenticated.
2. Run `bash scripts/run_inference.sh`.

The script automatically pulls `bigbio/med_qa` (config: `med_qa_en_source`, split: `test[:1000]`) via the HuggingFace `datasets` library.
