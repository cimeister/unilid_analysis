"""Metric decomposition of the saved full-test predictions (Exp 24, analysis only).

Recomputes, from the saved full-test prediction memmaps (Exp 16, job 2784115; floor-21
pass, job 2791722), two views of stratified performance:

1. within-stratum macro-F1: metrics restricted to examples whose TRUE label is in the
   stratum. This is what every stratum row in outputs/tables/full_test_eval.md and
   full_test_floor21.md reports, and what the selection guard bounds. A head-true line
   predicted as a tail language is excluded from the tail row by construction.
2. global per-language precision/recall/F1 over the full confusion matrix. The "overall"
   rows of the same tables already use this view (macro-F1 over all 1,940 languages).

Plus the false-positive flow into tail labels: source strata, concentration across
absorbing languages, and the residual (pred <- true) pairs under floor-21.

No new scoring happens here. Consistency gates (abort on failure):
- kept-line count must equal the recorded 45,377,279;
- recomputed within-stratum macro-F1 must match every recorded stratum value in
  full_test_eval.md / full_test_floor21.md to within rounding;
- recomputed global per-language F1 must match outputs/diagnostic/full_test_per_lang_f1.csv
  for the three Exp 16 configurations.

Outputs: outputs/tables/metric_decomposition.md and
outputs/diagnostic/full_test_per_lang_prf.csv (per-language precision/recall/F1/FP for
all four configurations, including floor21 which full_test_per_lang_f1.csv lacks).
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from analysis.metrics import compute_metrics

SCRATCH_DIR = "/capstor/scratch/cscs/cmeister747/unilid_analysis/full_test_eval"
F1_CSV = "outputs/diagnostic/full_test_per_lang_f1.csv"
DIAG_CSV = "outputs/diagnostic/lang_diagnostic.csv"
OUT_MD = "outputs/tables/metric_decomposition.md"
OUT_CSV = "outputs/diagnostic/full_test_per_lang_prf.csv"

CONFIGS = ["baseline", "freq_prior", "learned_bias", "floor21"]
TAIL_N = 1_000                     # same stratum thresholds as full_test_eval.lang_flags
HEAD_N = 18_000
EXPECTED_KEPT = 45_377_279         # recorded in full_test_eval.md ("Lines evaluated")
GATE_TOL = 6e-5                    # recorded tables round to 4 decimals

# Recorded within-stratum macro-F1 (full_test_eval.md, full_test_floor21.md). The gate
# proves the memmaps, language order, and stratum definitions here reproduce the tables
# of record exactly.
RECORDED = {
    "baseline":     {"overall": 0.9292, "tail": 0.9132, "magnets": 0.9138,
                     "twins": 0.9167, "head": 0.9602},
    "freq_prior":   {"overall": 0.9408, "tail": 0.8950, "magnets": 0.8965,
                     "twins": 0.9178, "head": 0.9613},
    "learned_bias": {"overall": 0.9421, "tail": 0.9114, "magnets": 0.9056,
                     "twins": 0.9283, "head": 0.9703},
    "floor21":      {"overall": 0.9421, "tail": 0.8928, "magnets": 0.8974,
                     "twins": 0.9166, "head": 0.9599},
}

TOP_ABSORBERS = 15                 # display counts for the report tables only
TOP_PAIRS = 12
TOP_SOURCES = 3


def _per_lang_stats(pred: np.ndarray, yk: np.ndarray, n_lang: int):
    """Global per-language TP/FP/FN over the full confusion. pred == -1 (empty after
    preprocess) counts as a false negative for the true language and no false positive."""
    valid = pred >= 0
    correct = (pred == yk) & valid
    tp = np.bincount(yk[correct], minlength=n_lang).astype(float)
    fp = np.bincount(pred[valid & ~correct], minlength=n_lang).astype(float)
    fn = np.bincount(yk[~correct], minlength=n_lang).astype(float)
    prec = np.divide(tp, tp + fp, out=np.zeros(n_lang), where=(tp + fp) > 0)
    rec = np.divide(tp, tp + fn, out=np.zeros(n_lang), where=(tp + fn) > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros(n_lang),
                   where=(prec + rec) > 0)
    return prec, rec, f1, tp, fp, fn


def run():
    for p in [F1_CSV, DIAG_CSV]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"required artifact missing: {p}")
    f1csv = pd.read_csv(F1_CSV)
    diag = pd.read_csv(DIAG_CSV)
    if list(f1csv.lang) != list(diag.lang):
        raise RuntimeError("language order differs between "
                           f"{F1_CSV} and {DIAG_CSV}; alignment assumption broken")
    langs = f1csv.lang.tolist()
    n_lang = len(langs)
    N = f1csv.N.values
    category = f1csv.category.values     # written by full_test_eval from lang_diagnostic
    script = diag.script.values

    y_path = os.path.join(SCRATCH_DIR, "y_true.npy")
    if not os.path.exists(y_path):
        raise FileNotFoundError(f"required artifact missing: {y_path}")
    y = np.asarray(np.load(y_path, mmap_mode="r"))
    kept = y >= 0
    yk = y[kept]
    if int(kept.sum()) != EXPECTED_KEPT:
        raise RuntimeError(f"kept lines {int(kept.sum()):,} != recorded {EXPECTED_KEPT:,}")

    strata = {
        "overall": np.ones(n_lang, bool),
        "tail": N < TAIL_N,
        "magnets": category == "flat_magnet",
        "twins": category == "twin",
        "head": N >= HEAD_N,
    }
    global_groups = {
        "tail (N<1k)": N < TAIL_N,
        "lowmid (1k<=N<18k)": (N >= TAIL_N) & (N < HEAD_N),
        "head (N>=18k)": N >= HEAD_N,
        "flat_magnet": category == "flat_magnet",
        "twin": category == "twin",
        "isolated_tail": category == "isolated_tail",
        "tight_lowres": category == "tight_lowres",
        "all 1,940": np.ones(n_lang, bool),
    }
    tail_idx = np.where(strata["tail"])[0]

    preds, stats = {}, {}
    for c in CONFIGS:
        p_path = os.path.join(SCRATCH_DIR, f"pred_{c}.npy")
        if not os.path.exists(p_path):
            raise FileNotFoundError(f"required artifact missing: {p_path}")
        preds[c] = np.asarray(np.load(p_path, mmap_mode="r"))[kept]
        stats[c] = _per_lang_stats(preds[c], yk, n_lang)

    # --- consistency gates ---
    within = {c: {} for c in CONFIGS}
    for c in CONFIGS:
        for st, flags in strata.items():
            m = flags[yk]
            got = compute_metrics(yk[m], preds[c][m])["macro_f1"]
            within[c][st] = got
            want = RECORDED[c][st]
            if abs(got - want) > GATE_TOL:
                raise RuntimeError(f"within-stratum gate failed: {c}/{st} recomputed "
                                   f"{got:.6f} vs recorded {want:.4f}")
    for c in ["baseline", "freq_prior", "learned_bias"]:
        diff = np.abs(stats[c][2] - f1csv[f"f1_{c}"].values).max()
        if diff > 1e-9:
            raise RuntimeError(f"global per-language F1 gate failed for {c}: "
                               f"max |diff| = {diff:.2e} vs {F1_CSV}")
    print("All consistency gates passed (within-stratum tables reproduced; "
          "per-language F1 matches the saved CSV).")

    # --- per-language CSV for all four configs ---
    out = {"lang": langs, "N": N.astype(int), "category": category,
           "test_support": f1csv.test_support.values}
    for c in CONFIGS:
        prec, rec, f1, tp, fp, fn = stats[c]
        out[f"prec_{c}"] = prec
        out[f"rec_{c}"] = rec
        out[f"f1_{c}"] = f1
        out[f"fp_{c}"] = fp.astype(int)
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    pd.DataFrame(out).to_csv(OUT_CSV, index=False)

    # --- FP flow into tail labels (baseline) ---
    b_prec, b_rec, b_f1, b_tp, b_fp, b_fn = stats["baseline"]
    pb = preds["baseline"]
    fp_mask = np.isin(pb, tail_idx) & (pb != yk) & (pb >= 0)
    src_N = N[yk[fp_mask]]
    fp_sorted = np.sort(b_fp[tail_idx])[::-1]
    fp_tot = fp_sorted.sum()
    miss_mask = np.isin(yk, tail_idx) & (pb != yk)
    dest = pb[miss_mask]
    dest = dest[dest >= 0]

    L = []
    L.append("# Metric decomposition of the full-test predictions (Exp 24)\n")
    L.append(f"Inputs: prediction memmaps under {SCRATCH_DIR} (Exp 16 job 2784115; "
             "floor-21 job 2791722), 45,377,279 kept lines. Analysis only, no new "
             "scoring. All within-stratum values below reproduce the recorded tables "
             "(gate tolerance 6e-5); global per-language F1 reproduces "
             "full_test_per_lang_f1.csv exactly.\n")
    L.append("Definitions. Within-stratum macro-F1 restricts both truth and predictions "
             "to examples whose true label is in the stratum, so false positives "
             "arriving from outside the stratum are excluded; this is the view every "
             "stratum row and the selection guard use. Global per-language F1 counts "
             "the full confusion row and column for each language; the overall rows "
             "already use it.\n")

    L.append("## Tail (96 languages), two views\n")
    L.append("| config | within-stratum F1 | global mean F1 | global mean precision | "
             "global mean recall | FPs into tail labels |")
    L.append("|---|---|---|---|---|---|")
    for c in CONFIGS:
        prec, rec, f1, tp, fp, fn = stats[c]
        L.append(f"| {c} | {within[c]['tail']:.4f} | {f1[tail_idx].mean():.4f} | "
                 f"{prec[tail_idx].mean():.4f} | {rec[tail_idx].mean():.4f} | "
                 f"{fp[tail_idx].sum():,.0f} |")
    rec_only = 2 * b_rec[tail_idx] / (1 + b_rec[tail_idx])
    L.append(f"\nBaseline counterfactual: with precision fixed at 1.0 the tail global "
             f"mean F1 would be {rec_only.mean():.4f}, against the within-stratum "
             f"0.9132; the two views differ almost entirely by the false positives. "
             f"Baseline tail languages with precision below 0.5: "
             f"{(b_prec[tail_idx] < 0.5).sum()}/96; with recall below 0.5: "
             f"{(b_rec[tail_idx] < 0.5).sum()}/96.\n")

    L.append("## Global per-language mean F1 by group\n")
    hdr = "| config | " + " | ".join(global_groups) + " |"
    L.append(hdr)
    L.append("|" + "---|" * (len(global_groups) + 1))
    for c in CONFIGS:
        f1 = stats[c][2]
        L.append(f"| {c} | " + " | ".join(f"{f1[m].mean():.4f}"
                                          for m in global_groups.values()) + " |")
    L.append("\nGroup sizes: " + ", ".join(f"{k} = {int(m.sum())}"
                                           for k, m in global_groups.items()) + ".\n")

    L.append("## Overall delta decomposition by category (vs baseline)\n")
    L.append("Overall global macro-F1 delta as the sum of per-language deltas divided "
             "by 1,940, split by diagnostic category.\n")
    cats = ["flat_magnet", "twin", "isolated_tail", "tight_lowres", "mid", "head"]
    L.append("| config | total | " + " | ".join(cats) + " |")
    L.append("|" + "---|" * (len(cats) + 2))
    for c in CONFIGS[1:]:
        d = stats[c][2] - b_f1
        L.append(f"| {c} | {d.mean():+.4f} | "
                 + " | ".join(f"{d[category == k].sum() / n_lang:+.4f}" for k in cats)
                 + " |")

    L.append("\n## False-positive flow into tail labels (baseline)\n")
    L.append(f"- Total FPs into tail labels: {int(fp_mask.sum()):,} against "
             f"{int((np.isin(yk, tail_idx)).sum()):,} true tail examples.")
    L.append(f"- Source-language resources: median N = {np.median(src_N):,.0f}; "
             f"{np.mean(src_N >= HEAD_N):.1%} of FPs come from head sources "
             f"(N>=18k), {np.mean(src_N < TAIL_N):.1%} from other tail languages.")
    L.append(f"- Concentration across receiving tail languages: top 5 hold "
             f"{fp_sorted[:5].sum() / fp_tot:.1%}, top 10 "
             f"{fp_sorted[:10].sum() / fp_tot:.1%}, top 20 "
             f"{fp_sorted[:20].sum() / fp_tot:.1%} of all FPs into tail.")
    L.append(f"- Tail recall misses: {int(miss_mask.sum()):,} of "
             f"{int(np.isin(yk, tail_idx).sum()):,} "
             f"({miss_mask.sum() / np.isin(yk, tail_idx).sum():.1%}); "
             f"{np.mean(N[dest] >= HEAD_N):.1%} of the misrouted lines go to head "
             f"languages, {np.mean(N[dest] < TAIL_N):.1%} to other tail languages.\n")

    L.append(f"## The {TOP_ABSORBERS} tail languages receiving the most false "
             "positives, baseline vs floor-21\n")
    L.append("| lang | N | prec base -> floor21 | rec base -> floor21 | "
             "F1 base -> floor21 | FP base -> floor21 | top baseline FP sources |")
    L.append("|---|---|---|---|---|---|---|")
    f_prec, f_rec, f_f1, f_tp, f_fp, f_fn = stats["floor21"]
    order = tail_idx[np.argsort(-b_fp[tail_idx])][:TOP_ABSORBERS]
    for i in order:
        src = yk[(pb == i) & (yk != i)]
        top = pd.Series(src).value_counts().head(TOP_SOURCES)
        tops = ", ".join(f"{langs[j]} ({int(cnt)})" for j, cnt in top.items())
        L.append(f"| {langs[i]} | {N[i]} | {b_prec[i]:.3f} -> {f_prec[i]:.3f} | "
                 f"{b_rec[i]:.3f} -> {f_rec[i]:.3f} | {b_f1[i]:.3f} -> {f_f1[i]:.3f} | "
                 f"{int(b_fp[i]):,} -> {int(f_fp[i]):,} | {tops} |")

    pf = preds["floor21"]
    r_mask = np.isin(pf, tail_idx) & (pf != yk) & (pf >= 0)
    pairs = pd.DataFrame({"pred": pf[r_mask], "true": yk[r_mask]})
    counts = pairs.value_counts()
    same = sum(cnt for (pi, ti), cnt in counts.items() if script[pi] == script[ti])
    L.append(f"\n## Residual FPs into tail labels under floor-21\n")
    L.append(f"{int(r_mask.sum()):,} residual FPs; {same / r_mask.sum():.1%} are "
             f"same-script. Top directed pairs (pred <- true):\n")
    for (pi, ti), cnt in counts.head(TOP_PAIRS).items():
        L.append(f"- {langs[pi]} <- {langs[ti]}: {int(cnt):,}")

    os.makedirs(os.path.dirname(OUT_MD), exist_ok=True)
    with open(OUT_MD, "w") as f:
        f.write("\n".join(L) + "\n")
    print(f"Wrote {OUT_MD} and {OUT_CSV}")


if __name__ == "__main__":
    run()
