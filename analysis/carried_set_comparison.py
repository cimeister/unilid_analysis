"""Per-language comparison of the carried-forward set under the primary quantity
(Exp 38; user decisions 2026-07-25).

Primary quantity: per-language F1 computed over natural-distribution test data with
all false positives counted, averaged unweighted over the 1,940 languages. Dataset
here: the held-out remainder (the 45,004,014 kept lines outside draws 101 and 201),
so no selection data enters the comparison.

Configurations: the baseline plus the six carried-forward configurations
(freq_prior, learned_bias, floor21, margin_q5, margin_q5_head, gt_margin_adaptive).

Outputs: outputs/diagnostic/carried_set_per_lang_f1.csv (per-language F1 for all
seven) and outputs/tables/carried_set_comparison.md (group means, the primary-
quantity ranking, and per-language leader counts).
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from analysis.config import TOTAL_LINES
from analysis.metric_decomposition import _per_lang_stats
from analysis.balanced_split import OUT_DIR as DRAW_DIR, SEEDS, TEST_SEED
from analysis.full_test_eval import SCRATCH_DIR

PRF_CSV = "outputs/diagnostic/full_test_per_lang_prf.csv"
CARRIED = ["baseline", "freq_prior", "learned_bias", "floor21", "margin_q5",
           "margin_q5_head", "gt_margin_adaptive"]
EXPECTED_REMAINDER = 45_004_014
OUT_CSV = "outputs/diagnostic/carried_set_per_lang_f1.csv"
OUT_MD = "outputs/tables/carried_set_comparison.md"


def run() -> str:
    prf = pd.read_csv(PRF_CSV)
    langs = prf.lang.tolist()
    n_lang = len(langs)
    N = prf.N.values
    category = prf.category.values
    y = np.asarray(np.lib.format.open_memmap(
        os.path.join(SCRATCH_DIR, "y_true.npy"), mode="r"))
    excl = np.zeros(TOTAL_LINES, bool)
    for s in [SEEDS[0], TEST_SEED]:
        excl[np.load(os.path.join(DRAW_DIR, f"val_lines_seed{s}.npy"))] = True
    mask = (y >= 0) & ~excl
    if int(mask.sum()) != EXPECTED_REMAINDER:
        raise RuntimeError(f"remainder {int(mask.sum()):,} != {EXPECTED_REMAINDER:,}")
    ym = y[mask]

    f1s = {}
    for c in CARRIED:
        p = os.path.join(SCRATCH_DIR, f"pred_{c}.npy")
        if not os.path.exists(p):
            raise FileNotFoundError(f"prediction memmap missing: {p}")
        pred = np.asarray(np.lib.format.open_memmap(p, mode="r"))[mask]
        f1s[c] = _per_lang_stats(pred, ym, n_lang)[2]

    out = {"lang": langs, "N": N.astype(int), "category": category,
           "support": np.bincount(ym, minlength=n_lang)}
    for c in CARRIED:
        out[f"f1_{c}"] = f1s[c]
    pd.DataFrame(out).to_csv(OUT_CSV, index=False)

    groups = {"all 1,940": np.ones(n_lang, bool), "tail (N<1k)": N < 1_000,
              "lowmid (1k-18k)": (N >= 1_000) & (N < 18_000),
              "head (N>=18k)": N >= 18_000, "flat_magnet": category == "flat_magnet",
              "twin": category == "twin"}
    L = ["# Carried-forward set under the primary quantity (Exp 38)\n",
         f"Per-language F1 on the held-out remainder ({EXPECTED_REMAINDER:,} lines, "
         "natural distribution, all false positives counted), averaged unweighted "
         "over languages.\n",
         "| config | " + " | ".join(groups) + " |",
         "|" + "---|" * (len(groups) + 1)]
    order = sorted(CARRIED, key=lambda c: -f1s[c].mean())
    for c in order:
        L.append(f"| {c} | " + " | ".join(f"{f1s[c][m].mean():.4f}"
                                          for m in groups.values()) + " |")

    stack = np.stack([f1s[c] for c in CARRIED])
    mx = stack.max(axis=0)
    is_max = stack == mx[None, :]
    n_tied = int((is_max.sum(axis=0) > 1).sum())
    L += [f"\nPer-language leaders. {n_tied} of {n_lang} languages have ties at "
          "the maximum; strict counts exclude them (a config is counted only where "
          "it is the UNIQUE best):",
          ", ".join(f"{CARRIED[i]}: {int((is_max[i] & (is_max.sum(axis=0) == 1)).sum())} strict"
                    for i in range(len(CARRIED)))]
    with open(OUT_MD, "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))
    return OUT_MD


if __name__ == "__main__":
    run()
