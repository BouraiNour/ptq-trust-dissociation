#!/usr/bin/env python3
"""
ptq_trust/analysis.py -- paired analysis of the ptq_trust.generate outputs (FP16 / NF4 / AWQ_W4).

    python -m ptq_trust.analysis --fp16 data/per_item/out_FP16.csv --nf4 data/per_item/out_NF4.csv --awq data/per_item/out_AWQ_W4.csv

Design choices (state these in the paper):
  * Accuracy: ALL items present in all three recipes (no faithfulness-based filtering).
      strict = parsed 'Answer: X' == gold (unparsed = wrong);  argmax = argmax of the 4 answer-token logits == gold.
  * Calibration: confidence = max softmax over the 4 letter logits; correctness = argmax==gold.
      raw ECE (no recalibration) AND cross-fitted ECE (fold assigned per ITEM before resampling, so no leakage).
  * Faithfulness: items with a valid faithfulness value in ALL three recipes.
  * Multiplicity: Holm over the 6 primary tests (acc_strict, ECE_crossfit, faith_margin) x (NF4, AWQ) vs FP16;
      separate Holm over the 3 direct AWQ-vs-NF4 contrasts.
"""
import argparse
import json
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp, softmax

ap = argparse.ArgumentParser()
ap.add_argument("--fp16", required=True)
ap.add_argument("--nf4", required=True)
ap.add_argument("--awq", required=True)
ap.add_argument("--n_boot", type=int, default=20000)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--n_bins", type=int, default=15)
ap.add_argument("--n_boot_ece", type=int, default=20000, help="resamples for the (slow) cross-fitted ECE only")
ap.add_argument("--out", default="results/analysis_results.txt")
ap.add_argument("--only_k_matched", action="store_true", help="run only section 8 (fast)")
ap.add_argument("--control", action="append", default=[], metavar="NAME=CSV",
                help="control run to compare against (repeatable), e.g. --control FP16_EAGER=runs/out_FP16_EAGER.csv")
ap.add_argument("--only_controls", action="store_true", help="run only section 10 (fast); needs --control")
ap.add_argument("--m_acc", type=float, default=3.0, help="equivalence margin, pp (justify clinically!)")
ap.add_argument("--m_ece", type=float, default=0.02)
ap.add_argument("--m_faith", type=float, default=0.5, help="nats")
args = ap.parse_args()

_f = open(args.out, "w")
def out(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); _f.write(s + "\n"); _f.flush()
def hdr(t):
    out("\n" + "=" * 92); out(t); out("=" * 92)

B = args.n_boot
rng = np.random.default_rng(args.seed)
R = ["FP16", "NF4", "AWQ_W4"]; M = ["NF4", "AWQ_W4"]

# ----------------------------------------------------------------------------- load
raw = {"FP16": args.fp16, "NF4": args.nf4, "AWQ_W4": args.awq}
W = {r: pd.read_csv(p).drop_duplicates("idx").set_index("idx").sort_index() for r, p in raw.items()}
idx = sorted(set.intersection(*[set(W[r].index) for r in R]))
N = len(idx)
for r in R:
    W[r] = W[r].loc[idx]

gold = W["FP16"]["gold_idx"].to_numpy().astype(int)
for r in R:
    assert (W[r]["gold_idx"].to_numpy() == gold).all(), "gold mismatch across recipes"
LG = {r: np.array([[float(v) for v in s.split(",")] for s in W[r]["logits"]]) for r in R}
strict = {r: W[r]["correct_strict"].to_numpy().astype(float) for r in R}
argm = {r: W[r]["correct_argmax"].to_numpy().astype(float) for r in R}
fa = {r: W[r]["faith_margin"].to_numpy().astype(float) for r in R}
K = {r: W[r]["K"].to_numpy().astype(float) for r in R}
mf = {r: W[r]["m_full"].to_numpy().astype(float) for r in R}
me = {r: W[r]["m_empty"].to_numpy().astype(float) for r in R}

# ----------------------------------------------------------------------------- helpers
from ptq_trust import metrics as mt
pci, bp, holm = mt.percentile_ci, mt.bootstrap_p, mt.holm
ece_fn = lambda conf, corr: mt.ece(conf, corr, args.n_bins)
nll_T = lambda T, lg, y: mt.nll_at_T(lg, y, T)
fit_T, conf_corr = mt.fit_temperature, mt.conf_correct
crossfit_ece = lambda lg, y, f: mt.crossfit_ece(lg, y, f, args.n_bins)
raw_ece = lambda lg, y: mt.raw_ece(lg, y, args.n_bins)

folds = rng.permutation(N) % 2                  # per-ITEM fold, assigned before resampling
D = rng.integers(0, N, size=(B, N))              # shared bootstrap draws (paired across recipes)

# valid-faithfulness subset (all three recipes)
Fm = np.all([~np.isnan(fa[r]) for r in R], axis=0)
Fi = np.where(Fm)[0]; NF = len(Fi)
DF = rng.integers(0, NF, size=(B, NF))


def k_matched_section():
    """Section 8: faithfulness on items where all three recipes wrote the SAME number of sentences (length-controlled).
    Uses its own RNG (seed) so the result is identical whether or not the slow sections run."""
    hdr("8. LENGTH-CONTROLLED FAITHFULNESS: items with identical sentence count K in all three recipes")
    r2 = np.random.default_rng(args.seed)
    ok = np.all([~np.isnan(fa[r]) for r in R], axis=0) & (K["FP16"] == K["NF4"]) & (K["NF4"] == K["AWQ_W4"])
    ii = np.where(ok)[0]; Dk = r2.integers(0, len(ii), size=(B, len(ii)))
    out(f"n = {len(ii)} items (selection is on K, a post-treatment variable: treat as a sensitivity analysis)")
    for m in M:
        d = fa[m][ii] - fa["FP16"][ii]; lo, hi = pci(d[Dk].mean(1))
        rat = fa[m][ii].mean() / mf[m][ii].mean() - fa["FP16"][ii].mean() / mf["FP16"][ii].mean()
        out(f"  {m:7s} - FP16: {d.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]   margin-normalised delta {rat:+.4f}")
    d = fa["AWQ_W4"][ii] - fa["NF4"][ii]; lo, hi = pci(d[Dk].mean(1))
    out(f"  AWQ_W4 - NF4: {d.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]")


def controls_section(controls):
    """Section 10: churn and faithfulness change under perturbations that are NOT 4-bit quantization.
    Controls keep the FP16 weights (FP16_EAGER: eager attention = numerical-kernel noise only; BF16: bfloat16 compute;
    INT8: LLM.int8 weights = mild compression). Own RNG so results do not depend on which other sections run."""
    hdr("10. CONTROLS: how much churn / faithfulness change does a non-4-bit perturbation cause?")
    r2 = np.random.default_rng(args.seed + 2); D2 = r2.integers(0, N, size=(B, N))
    def mean_ci(v, mask=None):
        v = np.asarray(v, float)
        if mask is not None:
            v = v[mask]
        Dd = D2 if mask is None else r2.integers(0, len(v), size=(B, len(v)))
        lo, hi = pci(v[Dd].mean(1)); return float(v.mean()), float(lo), float(hi)
    def ratio_ci(fa_a, mf_a, fa_b, mf_b, mask):
        a, b, c, d = fa_a[mask], mf_a[mask], fa_b[mask], mf_b[mask]
        Dd = r2.integers(0, len(a), size=(B, len(a)))
        est = a.mean() / b.mean() - c.mean() / d.mean()
        lo, hi = pci(a[Dd].mean(1) / b[Dd].mean(1) - c[Dd].mean(1) / d[Dd].mean(1)); return float(est), float(lo), float(hi)
    def parsed(a):
        return np.array([isinstance(v, str) for v in a])
    def flips(ga, gb):
        ok = parsed(ga) & parsed(gb)
        return np.array([(x != y) if o else False for x, y, o in zip(ga, gb, ok)], float)
    g_fp = W["FP16"]["gen_letter"].to_numpy(dtype=object)
    flip_q = {m: flips(g_fp, W[m]["gen_letter"].to_numpy(dtype=object)) for m in M}
    res = {}
    for name, path in controls:
        d = pd.read_csv(path).drop_duplicates("idx").set_index("idx").sort_index()
        miss = sorted(set(idx) - set(d.index))
        if miss:
            raise SystemExit(f"control {name} ({path}) is missing {len(miss)} of {N} items (first: {miss[:5]}); "
                             f"finish the run (generate.py resumes).")
        d = d.loc[idx]
        assert (d["gold_idx"].to_numpy() == gold).all(), f"gold mismatch in control {name}"
        st = d["correct_strict"].to_numpy().astype(float); gc = d["gen_letter"].to_numpy(dtype=object)
        fac = d["faith_margin"].to_numpy(float); mfc = d["m_full"].to_numpy(float)
        flip_c = flips(g_fp, gc)
        c2w = ((strict["FP16"] == 1) & (st == 0)).astype(float); w2c = ((strict["FP16"] == 0) & (st == 1)).astype(float)
        out(f"\n-- control {name}: unparsed answers = {int((~parsed(gc)).sum())}, unfinished = {int((d.finished == 0).sum())}")
        r = {}
        e, lo, hi = mean_ci(100 * (st - strict["FP16"])); r["acc_delta_pp"] = [e, lo, hi]
        out(f"   strict accuracy {100*st.mean():.1f}%  change vs FP16 {e:+.2f} pp [{lo:+.2f}, {hi:+.2f}]")
        e, lo, hi = mean_ci(100 * flip_c); r["flip_pct"] = [e, lo, hi]
        out(f"   written answer differs from FP16: {e:.1f}% [{lo:.1f}, {hi:.1f}]  | correct->wrong {100*c2w.mean():.1f}% | "
            f"wrong->correct {100*w2c.mean():.1f}% | gross churn {100*(c2w.mean()+w2c.mean()):.1f}%")
        r["excess_flip_pp"] = {}
        for m in M:
            e, lo, hi = mean_ci(100 * (flip_q[m] - flip_c)); r["excess_flip_pp"][m] = [e, lo, hi]
            verdict = "above the control (CI excludes 0)" if lo > 0 else ("below the control" if hi < 0 else "NOT distinguishable from the control")
            out(f"   excess flips, {m} minus {name}: {e:+.1f} pp [{lo:+.1f}, {hi:+.1f}]  -> {verdict}")
        ok_c = ~np.isnan(fac) & ~np.isnan(fa["FP16"])
        e, lo, hi = mean_ci(fac - fa["FP16"], ok_c); r["faith_delta"] = [e, lo, hi]
        out(f"   faithfulness change vs FP16: {e:+.3f} [{lo:+.3f}, {hi:+.3f}] (n={int(ok_c.sum())})")
        e, lo, hi = ratio_ci(fac, mfc, fa["FP16"], mf["FP16"], ok_c); r["faith_fraction_delta"] = [e, lo, hi]
        out(f"   faithfulness as fraction of final margin, change vs FP16: {e:+.4f} [{lo:+.4f}, {hi:+.4f}]")
        e, lo, hi = mean_ci(mfc - mf["FP16"]); r["mfull_delta"] = [e, lo, hi]
        out(f"   final margin m_full change vs FP16: {e:+.2f} [{lo:+.2f}, {hi:+.2f}]")
        r["faith_m_minus_control"] = {}
        for m in M:
            okm = ok_c & ~np.isnan(fa[m])
            e, lo, hi = mean_ci(fa[m] - fac, okm); r["faith_m_minus_control"][m] = [e, lo, hi]
            out(f"   faithfulness {m} minus {name}: {e:+.3f} [{lo:+.3f}, {hi:+.3f}] (n={int(okm.sum())})")
        res[name] = r
    out("\nInterpretation guide: if a control's flip rate / faithfulness change is as large as the 4-bit models', the")
    out("corresponding effect cannot be attributed to 4-bit quantization; only the EXCESS over the control can.")
    return res

if args.only_controls:
    _c = [c.split("=", 1) for c in args.control]
    _r = controls_section(_c)
    import json as _j, os as _o
    with open(_o.path.join(_o.path.dirname(args.out) or ".", "controls_summary.json"), "w") as _f:
        _j.dump(_r, _f, indent=2)
    raise SystemExit

if args.only_k_matched:
    k_matched_section(); out("Done (k-matched only)."); raise SystemExit

# ----------------------------------------------------------------------------- 0. audit
hdr("0. DATA AUDIT")
out(f"items in all three recipes: N={N}   | items with valid faithfulness in all three: {NF}")
out(f"{'recipe':8s} {'unparsed':>8s} {'unfinished':>10s} {'faith NaN':>9s} {'mean K':>7s} {'mean new tok':>12s}")
for r in R:
    d = W[r]
    out(f"{r:8s} {int((d.parse_method=='none').sum()):8d} {int((d.finished==0).sum()):10d} "
        f"{int(np.isnan(fa[r]).sum()):9d} {K[r].mean():7.2f} {d.n_new_tokens.mean():12.1f}")

# ----------------------------------------------------------------------------- 1. descriptives
hdr("1. POINT ESTIMATES (all N items; faithfulness on the valid subset)")
out(f"{'recipe':8s} {'strict%':>8s} {'argmax%':>8s} {'meanConf':>9s} {'rawECE':>8s} {'T*':>6s} {'cfECE':>8s} "
    f"{'NLL':>7s} {'Brier':>7s} {'faith':>7s} {'faith med':>9s} {'m_full':>7s} {'m_empty':>8s}")
pt = {}
for r in R:
    c, o = conf_corr(LG[r], gold); T = fit_T(LG[r], gold)
    p = softmax(LG[r], axis=1); brier = ((p - np.eye(4)[gold]) ** 2).sum(1).mean()
    pt[("strict", r)] = 100 * strict[r].mean(); pt[("argm", r)] = 100 * argm[r].mean()
    pt[("ece", r)] = crossfit_ece(LG[r], gold, folds); pt[("rawece", r)] = raw_ece(LG[r], gold)
    pt[("faith", r)] = fa[r][Fi].mean()
    out(f"{r:8s} {pt[('strict', r)]:8.2f} {pt[('argm', r)]:8.2f} {c.mean():9.3f} {pt[('rawece', r)]:8.4f} {T:6.2f} "
        f"{pt[('ece', r)]:8.4f} {nll_T(1.0, LG[r], gold):7.3f} {brier:7.3f} {fa[r][Fi].mean():7.3f} "
        f"{np.median(fa[r][Fi]):9.3f} {mf[r].mean():7.2f} {me[r].mean():8.2f}")
out("conf = max softmax over the 4 answer-token logits; high meanConf at ~60% accuracy = strong overconfidence.")

# ----------------------------------------------------------------------------- 2. bootstrap
hdr("2. PAIRED BOOTSTRAP vs FP16 (B=%d, seed=%d)" % (B, args.seed))
boot = {}
for r in R:
    boot[("strict", r)] = 100 * strict[r][D].mean(1)
    boot[("argm", r)] = 100 * argm[r][D].mean(1)
    boot[("faith", r)] = fa[r][Fi][DF].mean(1)
Be = min(args.n_boot_ece, B)
out(f"computing cross-fitted ECE bootstrap with B_ece={Be} (all other metrics use B={B})...")
for r in R:
    boot[("ece", r)] = np.array([crossfit_ece(LG[r][d], gold[d], folds[d]) for d in D[:Be]])
    boot[("rawece", r)] = np.array([raw_ece(LG[r][d], gold[d]) for d in D])

def row(kind, m, base="FP16"):
    d = boot[(kind, m)] - boot[(kind, base)]; d = d[~np.isnan(d)]
    est = pt[(kind, m)] - pt[(kind, base)]; lo, hi = pci(d)
    return est, lo, hi, bp(d), d

primary = [("strict", "acc strict (pp)"), ("ece", "ECE cross-fit"), ("faith", "faith margin (nats)")]
extra = [("argm", "acc argmax (pp)"), ("rawece", "ECE raw (uncalibrated)")]
prim_rows = []
out(f"{'model':7s} {'metric':24s} {'delta':>9s} {'95% CI':>24s} {'boot p':>8s} {'Holm p':>8s}")
for m in M:
    for k, lab in primary:
        prim_rows.append((m, k, lab) + row(k, m))
adjs = holm([x[6] for x in prim_rows])
for x, a in zip(prim_rows, adjs):
    out(f"{x[0]:7s} {x[2]:24s} {x[3]:9.4f} [{x[4]:9.4f}, {x[5]:9.4f}] {x[6]:8.4f} {a:8.4f}")
summary = {"N": N, "N_faith": NF, "seed": args.seed, "n_boot": B, "n_boot_ece": Be, "primary": {}}
for x, a_ in zip(prim_rows, adjs):
    summary["primary"].setdefault(x[0], {})[x[1]] = dict(est=float(x[3]), lo=float(x[4]), hi=float(x[5]), p=float(x[6]), holm=float(a_))
out("-- secondary (not in the Holm family) --")
for m in M:
    for k, lab in extra:
        e, lo, hi, p, _ = row(k, m)
        out(f"{m:7s} {lab:24s} {e:9.4f} [{lo:9.4f}, {hi:9.4f}] {p:8.4f}")

# ----------------------------------------------------------------------------- 3. contrast
hdr("3. DIRECT PAIRED CONTRAST AWQ_W4 - NF4  (does it support 'distinct failure modes'?)")
cr = []
for k, lab in primary + extra:
    e, lo, hi, p, _ = row(k, "AWQ_W4", base="NF4"); cr.append((lab, e, lo, hi, p))
ca = holm([c[4] for c in cr[:3]])
for i, (lab, e, lo, hi, p) in enumerate(cr):
    out(f"{lab:24s} {e:9.4f} [{lo:9.4f}, {hi:9.4f}] boot p={p:.4f}" + (f"  Holm(3)={ca[i]:.4f}" if i < 3 else ""))
for a_, b_ in [("NF4", "FP16"), ("AWQ_W4", "FP16"), ("AWQ_W4", "NF4")]:
    x = int(((strict[a_] == 1) & (strict[b_] == 0)).sum()); y = int(((strict[a_] == 0) & (strict[b_] == 1)).sum())
    out(f"strict McNemar {a_} vs {b_}: {x} vs {y}  exact p={stats.binomtest(x, x + y, 0.5).pvalue:.4f}")

# ----------------------------------------------------------------------------- 4. equivalence
hdr("4. EQUIVALENCE (TOST via 90%% CI). margins: acc ±%.1f pp, ECE ±%.3f, faith ±%.2f nats (YOUR choice)" %
    (args.m_acc, args.m_ece, args.m_faith))
mg = {"strict": args.m_acc, "ece": args.m_ece, "faith": args.m_faith}
for x in prim_rows:
    d = x[7]; l, h = pci(d, 5, 95); g = mg[x[1]]
    out(f"{x[0]:7s} {x[2]:24s} 90% CI [{l:8.4f}, {h:8.4f}] vs ±{g}: "
        f"{'EQUIVALENT' if (l > -g and h < g) else 'not shown equivalent'}")

# ----------------------------------------------------------------------------- 5. faithfulness robustness
hdr("5. FAITHFULNESS ROBUSTNESS (n=%d valid items)" % NF)
for m in M:
    d = fa[m][Fi] - fa["FP16"][Fi]
    flips = rng.choice([-1, 1], size=(B, NF)); perm = (flips * d).mean(1)
    pperm = (np.sum(np.abs(perm) >= abs(d.mean())) + 1) / (B + 1)
    bmed = np.median(fa[m][Fi][DF], 1) - np.median(fa["FP16"][Fi][DF], 1); lo, hi = pci(bmed)
    out(f"{m:7s} mean {d.mean():+.3f} | median of paired diffs {np.median(d):+.3f} | 20%-trim {stats.trim_mean(d, .2):+.3f} | "
        f"Wilcoxon p={stats.wilcoxon(d).pvalue:.3g} | sign-flip p={pperm:.4g} | diff-of-medians CI [{lo:+.3f}, {hi:+.3f}]")

# ----------------------------------------------------------------------------- 6. scale / confidence / K confounds
hdr("6. IS THE FAITHFULNESS DROP A SCALE / CONFIDENCE / LENGTH ARTIFACT?")
out("(a) Faithfulness as a FRACTION of the final margin: mean(faith)/mean(m_full)")
for m in M:
    rr = lambda r, d: fa[r][Fi][d].mean(1) / mf[r][Fi][d].mean(1)
    est = fa[m][Fi].mean() / mf[m][Fi].mean() - fa["FP16"][Fi].mean() / mf["FP16"][Fi].mean()
    lo, hi = pci(rr(m, DF) - rr("FP16", DF))
    out(f"   {m:7s} FP16 {fa['FP16'][Fi].mean()/mf['FP16'][Fi].mean():.3f} -> {m} {fa[m][Fi].mean()/mf[m][Fi].mean():.3f}   "
        f"delta {est:+.4f} [{lo:+.4f}, {hi:+.4f}]")
    summary.setdefault("faith_fraction", {})[m] = [float(est), float(lo), float(hi)]
out("(b) Margin scale and the no-rationale baseline (all N items): does quantization shrink the overall margin?")
for m in M:
    for nm, A in (("m_full", mf), ("m_empty", me), ("m_full-m_empty (rationale contribution)", {r: mf[r] - me[r] for r in R})):
        d = A[m][D].mean(1) - A["FP16"][D].mean(1); lo, hi = pci(d)
        out(f"   {m:7s} {nm:42s} {A[m].mean()-A['FP16'].mean():+.3f} [{lo:+.3f}, {hi:+.3f}]")
        if nm == "m_full":
            summary.setdefault("mfull_delta", {})[m] = [float(A[m].mean() - A["FP16"].mean()), float(lo), float(hi)]
out("(c) Sentence-count confound: faith is a MEAN over K sentences. Paired OLS  d_faith = a + b1*dK + b2*d_mfull")
for m in M:
    y = fa[m][Fi] - fa["FP16"][Fi]
    X = np.column_stack([np.ones(NF), K[m][Fi] - K["FP16"][Fi], mf[m][Fi] - mf["FP16"][Fi]])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    bs = np.array([np.linalg.lstsq(X[d], y[d], rcond=None)[0] for d in DF])
    ci = [pci(bs[:, j]) for j in range(3)]
    out(f"   {m:7s} intercept {beta[0]:+.3f} [{ci[0][0]:+.3f}, {ci[0][1]:+.3f}] | b_dK {beta[1]:+.3f} [{ci[1][0]:+.3f}, {ci[1][1]:+.3f}] "
        f"| b_dMfull {beta[2]:+.3f} [{ci[2][0]:+.3f}, {ci[2][1]:+.3f}]")
out("   (intercept = faithfulness change NOT explained by changes in sentence count or final margin)")
out("(d) Per-sentence drops (pooled over all sentences, valid items)")
for r in R:
    dr = np.concatenate([json.loads(s) for s in W[r]["drops_margin_json"].iloc[Fi]])
    out(f"   {r:7s} sentences={len(dr)} mean drop={dr.mean():.3f} median={np.median(dr):.3f} share>0={np.mean(dr>0):.3f}")
out("(e) Spearman(faith, m_full) per recipe: " + ", ".join(
    f"{r} {stats.spearmanr(fa[r][Fi], mf[r][Fi])[0]:+.3f}" for r in R))
out("(f) Probability-based faithfulness is saturated: share of items with p_full>0.999: " + ", ".join(
    f"{r} {np.mean(W[r].p_full > 0.999):.3f}" for r in R))

# ----------------------------------------------------------------------------- 7. stratified
hdr("7. FAITHFULNESS DROP BY CORRECTNESS STRATUM (valid items)")
st = {"FP16 correct (strict)": strict["FP16"][Fi] == 1, "FP16 wrong (strict)": strict["FP16"][Fi] == 0,
      "correct in ALL three": np.all([strict[r][Fi] == 1 for r in R], axis=0)}
for name, mask in st.items():
    ii = np.where(mask)[0]; out(f"-- {name} (n={len(ii)})")
    for m in M:
        d = fa[m][Fi][ii] - fa["FP16"][Fi][ii]
        bs = rng.integers(0, len(ii), size=(B, len(ii))); lo, hi = pci(d[bs].mean(1))
        out(f"   {m:7s} d_faith {d.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]")

k_matched_section()

def churn_section():
    hdr("9. ANSWER CHURN vs FP16 (strict letters; flips counted where both answers parsed)")
    g = {r: W[r]["gen_letter"].to_numpy(dtype=object) for r in R}
    parsed = {r: np.array([isinstance(v, str) for v in g[r]]) for r in R}
    res = {}
    for m in M + ["AWQ_W4_vs_NF4"]:
        a_, b_ = ("FP16", m) if m != "AWQ_W4_vs_NF4" else ("NF4", "AWQ_W4")
        both = parsed[a_] & parsed[b_]
        flip = np.array([(x != y) if ok else False for x, y, ok in zip(g[a_], g[b_], both)], float)
        lo, hi = pci(flip[D].mean(1))
        line = f"  {m:14s} flip rate {100*flip.mean():5.1f}% [{100*lo:.1f}, {100*hi:.1f}]"
        res[m] = dict(flip=float(flip.mean()), flip_lo=float(lo), flip_hi=float(hi))
        if m != "AWQ_W4_vs_NF4":
            c2w = ((strict["FP16"] == 1) & (strict[m] == 0)).astype(float)
            w2c = ((strict["FP16"] == 0) & (strict[m] == 1)).astype(float)
            net = w2c.mean() - c2w.mean(); gross = c2w.mean() + w2c.mean()
            line += (f" | correct->wrong {100*c2w.mean():.1f}% | wrong->correct {100*w2c.mean():.1f}% | "
                     f"net {100*net:+.1f} pp | gross churn {100*gross:.1f}% ({gross/abs(net):.1f}x |net|)")
            res[m].update(c2w=float(c2w.mean()), w2c=float(w2c.mean()))
        out(line)
    out("  NOTE: no numerical-noise control (e.g. BF16 vs FP16, INT8 vs FP16) was run, so these flip rates are an upper")
    out("  bound on the part attributable to 4-bit quantization. Treat as descriptive until that control exists.")
    return res

summary["churn"] = churn_section()
if args.control:
    summary["controls"] = controls_section([c.split("=", 1) for c in args.control])
import json as _json, os as _os
_sp = _os.path.join(_os.path.dirname(args.out) or ".", "summary.json")
with open(_sp, "w") as _fh:
    _json.dump(summary, _fh, indent=2)
out("summary written ->", _sp)

out("\nDone ->", args.out)
