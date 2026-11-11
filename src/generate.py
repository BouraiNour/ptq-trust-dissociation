#!/usr/bin/env python3
"""
ptq_trust/generate.py  --  MedQA quantization audit, fixed generation + faithfulness pipeline.

Run ONE recipe per process (avoids memory build-up and OOM cascades):
    python -m ptq_trust.generate --recipe FP16   --limit 5      # smoke test first, read out_FP16.csv gen_text!
    python -m ptq_trust.generate --recipe FP16
    python -m ptq_trust.generate --recipe NF4
    python -m ptq_trust.generate --recipe AWQ_W4
Re-running a command RESUMES (skips idx already in the output file). Rows that errored are NOT written,
so they are retried on resume. Errors are logged to errors_<recipe>.log.

Fixes vs the first pipeline
  1. Saves the raw generated text, token count and a `finished` flag (EOS reached before the cap).
  2. Case-sensitive answer parser that takes the LAST 'Answer: X'; a flagged looser 'answer is X' fallback.
     Old regex matched 'answer depends' -> D and cut the rationale there.
  3. MAX_NEW raised 200 -> 400 so rationales are not truncated; truncation is flagged, not silently scored.
  4. Faithfulness baseline = the SAME re-joined text used for ablation (old code compared raw vs re-joined).
     Short sentence fragments are merged into the previous sentence, not dropped.
  5. No tokenizer truncation (old max_length=2048 could cut off the trailing 'Answer:').
  6. Two explicit accuracy columns (correct_strict, correct_argmax); no ambiguous `correct`.
  7. Dataset = MedQA 4-option config, so no items are lost to a gold='E' filter.
  8. Extra faithfulness variants and a null-rationale control: margin drops, probability drops,
     per-sentence drops (JSON), K, m_full, m_empty.
  9. Per-item try/except + immediate row-by-row writes (old code could die silently mid-run).
"""
import argparse
import json
import os
from ptq_trust.parsing import (LETTERS, parse_answer, split_sentences, format_question,
                                build_messages, margin)

# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recipe", required=True, choices=["FP16", "NF4", "AWQ_W4", "FP16_EAGER", "BF16", "INT8"],
                    help="FP16/NF4/AWQ_W4 = the audited recipes. FP16_EAGER (INVALID for Qwen2.5 in fp16: NaN logits, see data/failed_runs), BF16, INT8 = churn/faithfulness CONTROLS: "
                         "same weights and dtype but eager attention (pure numerical-kernel noise), bfloat16 compute "
                         "(precision-format noise), and LLM.int8 8-bit weights (mild compression).")
    ap.add_argument("--n_eval", type=int, default=1000)
    ap.add_argument("--max_new", type=int, default=400)
    ap.add_argument("--limit", type=int, default=None, help="process only first N items (smoke test)")
    ap.add_argument("--dataset_config", default="med_qa_en_4options_source")
    ap.add_argument("--out_dir", default="/kaggle/working")
    ap.add_argument("--expect_tf", default="4.51.3", help="abort unless transformers version starts with this")
    ap.add_argument("--allow_any_version", action="store_true", help="skip the version guard")
    ap.add_argument("--ds_cache", default=None,
                    help="HF datasets cache dir (default <out_dir>/hf_ds_cache). Separate from ~/.cache so metadata "
                         "written by a different `datasets` version can never break loading.")
    ap.add_argument("--env_path", default="/kaggle/working/env")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    import sys, traceback
    if args.env_path and os.path.isdir(args.env_path) and args.env_path not in sys.path:
        sys.path.insert(0, args.env_path)
    import numpy as np
    import pandas as pd
    import torch
    import transformers
    from datasets import load_dataset
    from tqdm.auto import tqdm
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    MODEL_BASE = "Qwen/Qwen2.5-7B-Instruct"
    model_id = "Qwen/Qwen2.5-7B-Instruct-AWQ" if args.recipe == "AWQ_W4" else MODEL_BASE
    out_csv = os.path.join(args.out_dir, f"out_{args.recipe}.csv")
    err_log = os.path.join(args.out_dir, f"errors_{args.recipe}.log")

    print("transformers", transformers.__version__, "| torch", torch.__version__)
    if not args.allow_any_version and not transformers.__version__.startswith(args.expect_tf):
        raise SystemExit(
            f"\nABORT: transformers {transformers.__version__} loaded, expected {args.expect_tf}. "
            f"The pinned env is not being used (is {args.env_path} present and populated? re-run the install cell). "
            f"Pass --allow_any_version to override.")
    import datasets as _datasets
    ds_cache = args.ds_cache or os.path.join(args.out_dir, f"hf_ds_cache_{_datasets.__version__}")  # per-version
    print("datasets", _datasets.__version__, "| dataset cache:", ds_cache)
    ds = load_dataset("bigbio/med_qa", args.dataset_config, split=f"test[:{args.n_eval}]", cache_dir=ds_cache)
    n_opts = {len(ex["options"]) for ex in ds}
    print(f"dataset {args.dataset_config}: n={len(ds)}  option counts present: {sorted(n_opts)}")
    assert n_opts == {4}, "Expected only 4-option questions; check --dataset_config"

    done = set()
    if os.path.exists(out_csv):
        done = set(pd.read_csv(out_csv, usecols=["idx"])["idx"].tolist())
    print(f"{args.recipe}: {len(done)} items already done, resuming.")

    tok = AutoTokenizer.from_pretrained(MODEL_BASE)
    dtype_key = "dtype" if int(transformers.__version__.split(".")[0]) >= 5 else "torch_dtype"
    if dtype_key == "dtype":
        print("WARNING: transformers>=5 detected. The paper/old runs used 4.51.3; for comparability prefer "
              "`pip install transformers==4.51.3` (AWQ/bitsandbytes backends are also tested on that version).")
    kw = {dtype_key: torch.bfloat16 if args.recipe == "BF16" else torch.float16, "low_cpu_mem_usage": True,
          "device_map": "cuda:0" if args.recipe == "AWQ_W4" else "auto"}
    if args.recipe == "FP16_EAGER":
        kw["attn_implementation"] = "eager"          # default recipes use the library default (sdpa)
    if args.recipe == "INT8":
        kw["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    if args.recipe == "BF16" and torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] < 8:
        print("NOTE: bfloat16 is emulated on pre-Ampere GPUs (e.g. T4) and may be much slower; "
              "if it is unworkably slow use FP16_EAGER as the noise-floor control instead.")
    if args.recipe == "NF4":
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)
    model = AutoModelForCausalLM.from_pretrained(model_id, **kw).eval()

    info = {"recipe": args.recipe, "model_id": model_id, "transformers": transformers.__version__,
            "torch": torch.__version__, "max_new": args.max_new, "dataset_config": args.dataset_config,
            "n_eval": args.n_eval, "greedy": True,
            "attn_implementation": kw.get("attn_implementation", "default"),
            "dtype": str(kw[dtype_key]).replace("torch.", "")}
    try:
        import importlib.metadata as md
        for pkg in ("bitsandbytes", "autoawq", "awq", "accelerate"):
            try: info[pkg] = md.version(pkg)
            except Exception: pass
    except Exception:
        pass
    with open(os.path.join(args.out_dir, f"run_info_{args.recipe}.json"), "w") as f:   # report these in the paper
        json.dump(info, f, indent=2)

    letter_ids = {}
    for L in LETTERS:
        ids = set()
        for s in (L, " " + L, L.lower(), " " + L.lower()):
            e = tok.encode(s, add_special_tokens=False)
            if len(e) == 1:
                ids.add(e[0])
        letter_ids[L] = sorted(ids)
    print("letter token ids:", letter_ids)
    assert all(letter_ids[L] for L in LETTERS)
    eos_ids = {tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|endoftext|>")}

    @torch.no_grad()
    def score(prompt_text, reasoning):
        full = prompt_text + reasoning + "\nAnswer:"
        ids = tok(full, return_tensors="pt").to(model.device)       # NO truncation
        lg = model(**ids).logits[0, -1].float()
        return [float(max(lg[i] for i in letter_ids[L])) for L in LETTERS]

    def softmax(v):
        v = np.array(v); e = np.exp(v - v.max()); return e / e.sum()

    header_written = os.path.exists(out_csv)
    n_err = 0; consec = 0
    items = list(enumerate(ds))
    if args.limit:
        items = items[:args.limit]
    for i, ex in tqdm(items, desc=args.recipe):
        if i in done:
            continue
        try:
            gold = ex["answer_idx"].strip().upper()[:1]
            assert gold in LETTERS, f"unexpected gold {ex['answer_idx']!r}"
            msgs = build_messages(format_question(ex))
            prompt_text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
            # Tokenise the already-templated text ourselves: apply_chat_template(return_tensors=...) returns a
            # BatchEncoding in transformers>=5 (bare AttributeError on .shape), a tensor in 4.x.
            enc = tok(prompt_text, return_tensors="pt", add_special_tokens=False).to(model.device)
            prompt_len = int(enc["input_ids"].shape[1])

            with torch.no_grad():
                out = model.generate(**enc, max_new_tokens=args.max_new, do_sample=False,
                                     pad_token_id=tok.eos_token_id, temperature=None, top_p=None, top_k=None)
            new_ids = out[0][prompt_len:]
            gen_text = tok.decode(new_ids, skip_special_tokens=True)
            n_new = int(new_ids.shape[0])
            finished = bool(n_new < args.max_new or int(new_ids[-1]) in eos_ids)

            gen_letter, gen_loose, method, rationale = parse_answer(gen_text)
            sents = split_sentences(rationale)
            base_text = " ".join(sents) if sents else rationale        # SAME text for baseline and ablations

            lg_full = score(prompt_text, base_text)
            t = int(np.argmax(lg_full)); scored_letter = LETTERS[t]
            m_full = margin(lg_full, t); p_full = float(softmax(lg_full)[t])

            lg_empty = score(prompt_text, "")                         # null-rationale control
            m_empty = margin(lg_empty, t); p_empty = float(softmax(lg_empty)[t])

            faith_m = faith_p = float("nan"); drops_m = []; drops_p = []
            if method != "none" and len(sents) >= 2:                   # only for rationales that ended in an answer
                for j in range(len(sents)):
                    lg_a = score(prompt_text, " ".join(sents[:j] + sents[j + 1:]))
                    drops_m.append(m_full - margin(lg_a, t))
                    drops_p.append(p_full - float(softmax(lg_a)[t]))
                faith_m = float(np.mean(drops_m)); faith_p = float(np.mean(drops_p))

            row = {
                "recipe": args.recipe, "idx": i, "gold": gold, "gold_idx": LETTERS.index(gold),
                "n_options": len(ex["options"]),
                "gen_text": gen_text, "n_new_tokens": n_new, "finished": int(finished),
                "parse_method": method, "gen_letter": gen_letter or "", "gen_letter_loose": gen_loose or "",
                "scored_letter": scored_letter,
                "correct_strict": int(gen_letter == gold),               # primary 'Answer: X' only; unparsed = wrong
                "correct_loose": int(gen_loose == gold),
                "correct_argmax": int(scored_letter == gold),
                "logits": ",".join(f"{v:.4f}" for v in lg_full),
                "K": len(sents), "m_full": m_full, "m_empty": m_empty, "p_full": p_full, "p_empty": p_empty,
                "faith_margin": faith_m, "faith_prob": faith_p,
                "drops_margin_json": json.dumps([round(x, 4) for x in drops_m]),
                "drops_prob_json": json.dumps([round(x, 5) for x in drops_p]),
            }
            pd.DataFrame([row]).to_csv(out_csv, mode="a", header=not header_written, index=False)
            header_written = True
            consec = 0
        except Exception as e:                                         # log, keep going, retry on resume
            n_err += 1; consec += 1
            with open(err_log, "a") as f:
                f.write(f"idx={i} {type(e).__name__}: {e}\n{traceback.format_exc()}\n")
            torch.cuda.empty_cache()
            if consec >= 5:
                raise SystemExit(f"5 consecutive failures -- systematic bug, stopping. See {err_log}")

    print(f"\nDone {args.recipe}. errors this run: {n_err} (see {err_log}). Output: {out_csv}")


if __name__ == "__main__":
    main()
