import argparse
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default="results/summary.json")
    ap.add_argument("--out", default="results/fig_dissociation_v2.png")
    ap.add_argument("--out_controls", default="results/fig_controls.png")
    a = ap.parse_args()
    S = json.load(open(a.summary)); P = S["primary"]
    names = {"NF4": "NF4", "AWQ_W4": "AWQ"}; col = {"NF4": "#1f77b4", "AWQ_W4": "#ff7f0e"}
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    panels = [("ece", "\u0394 cross-fitted ECE vs FP16  [right = worse calibration]", "(a) Calibration vs faithfulness"),
              ("strict", "\u0394 strict accuracy vs FP16 (pp)  [left = less accurate]", "(b) Accuracy vs faithfulness")]
    for ax, (kind, xl, title) in zip(axs, panels):
        ax.axhline(0, color="gray", ls="--", lw=1); ax.axvline(0, color="gray", ls="--", lw=1)
        ax.plot(0, 0, marker="X", color="k", ms=11, ls="", label="FP16 (reference)", zorder=5)
        for m in ("NF4", "AWQ_W4"):
            x, y = P[m][kind], P[m]["faith"]
            ax.errorbar(x["est"], y["est"], xerr=[[x["est"] - x["lo"]], [x["hi"] - x["est"]]],
                        yerr=[[y["est"] - y["lo"]], [y["hi"] - y["est"]]], fmt="o", color=col[m], ms=9, capsize=4,
                        lw=1.8, label=names[m], zorder=4)
        ax.set_xlabel(xl, fontsize=10); ax.set_title(title, fontsize=11); ax.grid(ls=":", alpha=.5)
    axs[0].set_ylabel("\u0394 faithfulness (nats, leave-one-sentence-out)\n[down = less faithful]", fontsize=10)
    axs[0].legend(loc="lower left", fontsize=9)
    fig.suptitle(f"Change relative to FP16, N={S['N']} items (faithfulness n={S['N_faith']}); 95% paired-bootstrap CIs "
                 f"(B={S['n_boot']:,})", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94]); fig.savefig(a.out, dpi=300); print("saved", a.out)
    if "controls" in S:
        controls_figure(S, a.out_controls)


def controls_figure(S, out):
    """Churn / margin / faithfulness for the non-4-bit controls (grey) next to NF4 and AWQ (colour)."""
    C = S["controls"]; order = [c for c in ("BF16", "INT8") if c in C] + ["NF4", "AWQ_W4"]
    col = {"BF16": "#7f7f7f", "INT8": "#bdbdbd", "NF4": "#1f77b4", "AWQ_W4": "#ff7f0e"}
    lab = {"BF16": "BF16 (control)", "INT8": "INT8 (control)", "NF4": "NF4", "AWQ_W4": "AWQ"}
    def flip(n):
        if n in C: return C[n]["flip_pct"]
        c = S["churn"][n]; return [100 * c["flip"], 100 * c["flip_lo"], 100 * c["flip_hi"]]
    def margin(n):
        return C[n]["mfull_delta"] if n in C else S["mfull_delta"][n]
    def faith(n):
        if n in C: return C[n]["faith_delta"]
        p = S["primary"][n]["faith"]; return [p["est"], p["lo"], p["hi"]]
    panels = [(flip, "Written answer differs from FP16 (%)", "(a) Answer churn"),
              (margin, "\u0394 mean final logit margin vs FP16 (nats)", "(b) Logit-margin shrinkage"),
              (faith, "\u0394 faithfulness vs FP16 (nats)", "(c) Leave-one-out faithfulness")]
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.9))
    for ax, (fn, xl, title) in zip(axs, panels):
        for i, n in enumerate(order):
            e, lo, hi = fn(n)
            ax.errorbar(e, i, xerr=[[e - lo], [hi - e]], fmt="o", color=col[n], ms=8, capsize=4, lw=1.8)
        ax.set_yticks(range(len(order))); ax.set_yticklabels([lab[n] for n in order]); ax.invert_yaxis()
        ax.set_xlabel(xl, fontsize=9.5); ax.set_title(title, fontsize=11); ax.grid(axis="x", ls=":", alpha=.5)
        if fn is not flip: ax.axvline(0, color="gray", ls="--", lw=1)
    axs[0].set_xlim(left=0)
    fig.suptitle("4-bit effects against numerical-noise (BF16) and 8-bit (INT8) controls; 95% paired-bootstrap CIs",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93]); fig.savefig(out, dpi=300); print("saved", out)


if __name__ == "__main__":
    main()
