import argparse
import os
import torch
import numpy as np
import pandas as pd
import yaml
from tqdm.auto import tqdm
from src.data import load_medqa, format_question, build_messages, extract_generated_letter, strip_answer, split_sentences, normalize_letter
from src.models import load_model_and_tokenizer, LETTERS

def load_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    if "defaults" in config:
        base_path = os.path.join(os.path.dirname(config_path), config["defaults"])
        with open(base_path, "r") as f:
            base_config = yaml.safe_load(f)
            base_config.update(config)
            config = base_config
    return config

def score_answer(model, tok, letter_ids, msgs, reasoning_text):
    prompt_text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    full = prompt_text + reasoning_text + "\nAnswer:"
    ids = tok(full, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
    with torch.no_grad():
        logits = model(**ids).logits[0, -1].float()
    return torch.stack([torch.stack([logits[i] for i in letter_ids[L]]).max() for L in LETTERS])

def _save_checkpoint(path, rows):
    df_new = pd.DataFrame(rows)
    if os.path.exists(path):
        df_all = pd.concat([pd.read_csv(path), df_new], ignore_index=True)
        df_all.to_csv(path, index=False)
    else:
        df_new.to_csv(path, index=False)

def run_inference(config_path):
    config = load_config(config_path)
    recipe = config["recipe_name"]
    out_csv = os.path.join(config["output_dir"], "paired_medqa_quantization.csv")
    os.makedirs(config["output_dir"], exist_ok=True)
    
    print(f"\n{'='*70}\nRECIPE: {recipe}\n{'='*70}")
    
    done_indices = set()
    if os.path.exists(out_csv):
        df_existing = pd.read_csv(out_csv)
        done_indices = set(df_existing[df_existing["recipe"] == recipe]["idx"].values)
    
    if len(done_indices) >= config["n_eval"] * 0.7:
        print(f"  Skipping {recipe} — {len(done_indices)} rows already present.")
        return
        
    print(f"  Resuming {recipe}: {len(done_indices)} rows done.")
    
    ds = load_medqa(config["n_eval"], config["dataset"], config["dataset_config"], config["split"])
    model, tok, letter_ids = load_model_and_tokenizer(
        config["model_id"], config["quantization"], config["device_map"]
    )
    
    rows = []
    for i, ex in tqdm(enumerate(ds), total=len(ds), initial=len(done_indices), desc=recipe):
        if i in done_indices: continue
        gold = normalize_letter(ex["answer_idx"])
        if gold is None: continue

        msgs = build_messages(format_question(ex))
        prompt_ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            out = model.generate(prompt_ids, max_new_tokens=config["max_new_tokens"], do_sample=False, pad_token_id=tok.eos_token_id)
            
        gen_text = tok.decode(out[0][prompt_ids.shape[1]:], skip_special_tokens=True)
        gen_letter = extract_generated_letter(gen_text)
        reasoning = strip_answer(gen_text)

        letter_logits = score_answer(model, tok, letter_ids, msgs, reasoning)
        probs_np = torch.softmax(letter_logits, dim=-1).cpu().numpy()
        scored_idx = int(probs_np.argmax())
        scored_letter = LETTERS[scored_idx]
        
        sents = split_sentences(reasoning)
        faith = float("nan")
        if len(sents) >= 2:
            t = LETTERS.index(scored_letter)
            m_full = (letter_logits[t] - torch.cat([letter_logits[:t], letter_logits[t+1:]]).max()).item()
            drops = []
            for j in range(len(sents)):
                ablated = " ".join(sents[:j] + sents[j+1:])
                ll_abl = score_answer(model, tok, letter_ids, msgs, ablated)
                m_abl = (ll_abl[t] - torch.cat([ll_abl[:t], ll_abl[t+1:]]).max()).item()
                drops.append(m_full - m_abl)
            faith = float(np.mean(drops))

        rows.append({
            "recipe": recipe, "idx": i, "gold": gold, "gold_idx": LETTERS.index(gold),
            "gen_letter": gen_letter or "", "scored_letter": scored_letter,
            "correct": int(scored_letter == gold),
            "logits": ",".join(f"{v:.4f}" for v in letter_logits.cpu().numpy()),
            "faithfulness": faith,
        })
        
        if len(rows) >= 50:
            _save_checkpoint(out_csv, rows)
            rows = []
            
    if rows:
        _save_checkpoint(out_csv, rows)
    print(f"  ✅ {recipe} complete.")
    del model
    torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args()
    run_inference(args.config)
