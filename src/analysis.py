import os
import argparse
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import yaml

def fit_temperature(logits, labels_idx):
    L = np.asarray(logits, dtype=np.float32)
    y = np.asarray(labels_idx, dtype=np.int64)
    best_T, best_nll = 1.0, float("inf")
    for T in np.linspace(0.5, 25.0, 100):
        L_scaled = L / T
        max_L = L_scaled.max(axis=1, keepdims=True)
        exp_L = np.exp(L_scaled - max_L)
        log_probs = L_scaled - max_L - np.log(exp_L.sum(axis=1, keepdims=True))
        nll = -np.mean(log_probs[np.arange(len(y)), y])
        if nll < best_nll:
            best_nll, best_T = nll, T
    return float(best_T)

def ece(confidences, corrects, n_bins=10):
    confs, corrs = np.asarray(confidences, dtype=float), np.asarray(corrects, dtype=float)
    bins, total = np.linspace(0, 1, n_bins + 1), 0.0
    for i in range(n_bins):
        mask = (confs > bins[i]) & (confs <= bins[i + 1])
        if mask.sum() == 0: continue
        total += (mask.sum() / len(confs)) * abs(corrs[mask].mean() - confs[mask].mean())
    return float(total)

def ece_with_crossfit(logits, labels, n_splits=2):
    n = len(labels)
    idx = np.arange(n)
    np.random.shuffle(idx)
    half = n // 2
    folds = [idx[:half], idx[half:]]
    eces = []
    for i, eval_fold in enumerate(folds):
        fit_fold = folds[1 - i]
        T = fit_temperature(logits[fit_fold], labels[fit_fold])
        L = torch.tensor(logits[eval_fold], dtype=torch.float32) / T
        p = torch.softmax(L, dim=-1).numpy()
        eces.append(ece(p.max(1), (p.argmax(1) == labels[eval_fold])))
    return float(np.mean(eces))

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

def run_analysis(config_path):
    config = load_config(config_path)
    seed = config["seed"]
    n_boot = config["bootstrap_resamples"]
    out_dir = config["output_dir"]
    in_csv = os.path.join(out_dir, "paired_medqa_quantization.csv")
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    df = pd.read_csv(in_csv).dropna(subset=["faithfulness"])
    df["strict_correct"] = (df["gen_letter"] == df["gold"]).astype(int)

    recipes = ["FP16", "NF4", "AWQ_W4"]
    common_idx = set.intersection(*[set(df[df["recipe"] == r]["idx"]) for r in recipes])
    n_samples = len(common_idx)
    print(f"Total valid paired samples per recipe: {n_samples}")

    data_dict = {}
    for r in recipes:
        sub = df[(df["recipe"] == r) & (df["idx"].isin(common_idx))].sort_values("idx")
        logits = np.array([list(map(float, s.split(","))) for s in sub["logits"]])
        data_dict[r] = {
            "logits": logits, 
            "labels": sub["gold_idx"].values, 
            "faith": sub["faithfulness"].values,
            "acc": sub["strict_correct"].values
        }

    results = {r: {"faith": [], "ece": [], "acc": []} for r in ["NF4", "AWQ_W4"]}
    for _ in range(n_boot):
        sample_idx = np.random.choice(n_samples, n_samples, replace=True)
        f_fp16 = data_dict["FP16"]["faith"][sample_idx].mean()
        e_fp16 = ece_with_crossfit(data_dict["FP16"]["logits"][sample_idx], data_dict["FP16"]["labels"][sample_idx])
        a_fp16 = data_dict["FP16"]["acc"][sample_idx].mean()
        
        for r in ["NF4", "AWQ_W4"]:
            f_r = data_dict[r]["faith"][sample_idx].mean()
            e_r = ece_with_crossfit(data_dict[r]["logits"][sample_idx], data_dict[r]["labels"][sample_idx])
            a_r = data_dict[r]["acc"][sample_idx].mean()
            
            results[r]["faith"].append(f_r - f_fp16)
            results[r]["ece"].append(e_r - e_fp16)
            results[r]["acc"].append(a_r - a_fp16)

    lo_idx, hi_idx = int(0.025 * n_boot), int(0.975 * n_boot) - 1
    lines = ["="*80, f"PAIRED BOOTSTRAP RESULTS (N={n_samples}, {n_boot} resamples, seed={seed})", "="*80]
    final_metrics = {}

    for r in ["NF4", "AWQ_W4"]:
        f_deltas, e_deltas, a_deltas = np.sort(results[r]["faith"]), np.sort(results[r]["ece"]), np.sort(results[r]["acc"])
        f_mean, f_lo, f_hi = f_deltas.mean(), f_deltas[lo_idx], f_deltas[hi_idx]
        e_mean, e_lo, e_hi = e_deltas.mean(), e_deltas[lo_idx], e_deltas[hi_idx]
        a_mean, a_lo, a_hi = a_deltas.mean(), a_deltas[lo_idx], a_deltas[hi_idx]
        
        final_metrics[r] = {"e_mean": e_mean, "e_lo": e_lo, "e_hi": e_hi, "f_mean": f_mean, "f_lo": f_lo, "f_hi": f_hi}
        
        lines.append(f"\n[{r}]")
        lines.append(f"Δ Accuracy:     {a_mean*100:+.2f} pp  [95% CI: {a_lo*100:+.2f}, {a_hi*100:+.2f}]")
        lines.append(f"Δ ECE:          {e_mean:+.4f}     [95% CI: {e_lo:+.4f}, {e_hi:+.4f}]")
        lines.append(f"Δ Faithfulness: {f_mean:+.4f} nats [95% CI: {f_lo:+.4f}, {f_hi:+.4f}]")
    
    lines.append("="*80)
    report = "\n".join(lines)
    print("\n" + report)
    
    with open(os.path.join(out_dir, "bootstrap_results.txt"), "w", encoding="utf-8") as fh:
        fh.write(report + "\n")

    # Plotting
    plt.figure(figsize=(9, 7))
    plt.scatter(0, 0, color="black", marker="X", s=150, zorder=4, label="FP16")
    
    colors = {"NF4": "tab:blue", "AWQ_W4": "tab:orange"}
    for r in ["NF4", "AWQ_W4"]:
        m = final_metrics[r]
        xerr = [[m["e_mean"] - m["e_lo"]], [m["e_hi"] - m["e_mean"]]]
        yerr = [[m["f_mean"] - m["f_lo"]], [m["f_hi"] - m["f_mean"]]]
        
        plt.errorbar(m["e_mean"], m["f_mean"], xerr=xerr, yerr=yerr, fmt='o', color=colors[r], 
                     markersize=12, capsize=5, capthick=2, alpha=0.9, label=r, zorder=3)
        plt.annotate(r, (m["e_mean"], m["f_mean"]), xytext=(12, -5), textcoords='offset points', fontsize=12, fontweight='bold')

    plt.axhline(0, color='gray', linestyle='--', linewidth=1.5, zorder=1)
    plt.axvline(0, color='gray', linestyle='--', linewidth=1.5, zorder=1)
    plt.xlabel("Δ Cross-Fitted ECE (vs FP16)\n[Right = Calibration Degrades]", fontsize=12, fontweight='bold')
    plt.ylabel("Δ Faithfulness (nats)\n[Down = Faithfulness Degrades]", fontsize=12, fontweight='bold')
    plt.title(f"Calibration-Faithfulness Dissociation (N={n_samples}, 95% CIs)", fontsize=15, pad=15)
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    
    fig_path = os.path.join(out_dir, f"final_plot_N{n_samples}.png")
    plt.savefig(fig_path, dpi=300)
    print(f"\nSaved figure → {fig_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args()
    run_analysis(args.config)
