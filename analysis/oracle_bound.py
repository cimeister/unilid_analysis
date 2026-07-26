"""Oracle upper bound over the carried-forward set (Exp 40).

For each language, take the best F1 attained by any of the seven configurations in
outputs/diagnostic/carried_set_per_lang_f1.csv (baseline plus the six carried);
the group means of that per-language maximum bound what a perfect per-language
method chooser could achieve on the held-out remainder. This is an upper bound on
method COMBINATION, not an achievable configuration: the chooser is an oracle.

Gate: each configuration's recomputed group means must match the Exp 38 table
(same CSV, so exact equality is required).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from analysis.carried_set_comparison import CARRIED, OUT_CSV as CARRIED_CSV

OUT_MD = "outputs/tables/oracle_bound.md"


def run(out_md: str = OUT_MD) -> str:
    df = pd.read_csv(CARRIED_CSV)
    n_lang = len(df)
    if n_lang != 1_940:
        raise RuntimeError(f"{CARRIED_CSV} has {n_lang} rows, expected 1,940")
    N = df.N.values
    category = df.category.values
    f1 = np.stack([df[f"f1_{c}"].values for c in CARRIED])
    oracle = f1.max(axis=0)

    groups = {"all 1,940": np.ones(n_lang, bool), "tail (N<1k)": N < 1_000,
              "lowmid (1k-18k)": (N >= 1_000) & (N < 18_000),
              "head (N>=18k)": N >= 18_000, "flat_magnet": category == "flat_magnet",
              "twin": category == "twin"}
    best_single = {g: max(f1[i][m].mean() for i in range(len(CARRIED)))
                   for g, m in groups.items()}
    L = ["# Oracle upper bound over the carried set (Exp 40)\n",
         "Per-language maximum F1 over the seven configurations of Exp 38, on the "
         "held-out remainder. An upper bound for per-language method combination; "
         "not an achievable configuration by itself.\n",
         "| quantity | " + " | ".join(groups) + " |",
         "|" + "---|" * (len(groups) + 1),
         "| oracle (per-language best) | "
         + " | ".join(f"{oracle[m].mean():.4f}" for m in groups.values()) + " |",
         "| best single configuration | "
         + " | ".join(f"{best_single[g]:.4f}" for g in groups) + " |",
         "| oracle gain | "
         + " | ".join(f"{oracle[m].mean() - best_single[g]:+.4f}"
                      for g, m in groups.items()) + " |"]
    with open(out_md, "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))
    return out_md


if __name__ == "__main__":
    run()
