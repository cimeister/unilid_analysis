"""Degenerate-row scan across all packed models (Exp 35).

A row is flagged when fewer than MIN_ESTIMATED tokens sit off its floor plateau.
Purpose: (a) bound the EM-degeneracy concern raised 2026-07-25 (which prior results
rest on models with degenerate rows); (b) serve as the post-training gate for any
future retrain (run it on every new .unilid before evaluation).

Output: outputs/tables/degenerate_rows.md with the flagged set per model, script
composition, and per-language F1 where the evaluation exists.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from analysis.transfer_sweep import _load_model_data, UNILID_MODEL_PATH

MODELS = {
    "100k": UNILID_MODEL_PATH,
    "200k": "/capstor/scratch/cscs/cmeister747/unilid_analysis/glotlid_apertus200k.unilid",
    "131k": "/capstor/scratch/cscs/cmeister747/unilid_analysis/glotlid_apertus131k.unilid",
}
MIN_ESTIMATED = 100
OUT_MD = "outputs/tables/degenerate_rows.md"


def run(out_md: str = OUT_MD) -> str:
    diag = pd.read_csv("outputs/diagnostic/lang_diagnostic.csv")
    script = dict(zip(diag.lang, diag.script))
    f131 = pd.read_csv("outputs/diagnostic/full_test_131k_per_lang_prf.csv")
    f131 = dict(zip(f131.lang, f131.f1_131k))
    L = [f"# Degenerate-row scan (fewer than {MIN_ESTIMATED} estimated tokens)\n"]
    for name, path in MODELS.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"model missing: {path}")
        W, langs, _ = _load_model_data(path)
        vocab = W.shape[1]
        flagged = []
        for i in range(len(langs)):
            row = np.asarray(W[i], dtype=np.float64)
            est = int(vocab - (row == row.min()).sum())
            if est < MIN_ESTIMATED:
                flagged.append((langs[i], est))
        L.append(f"## {name} ({vocab:,} vocab): {len(flagged)} flagged rows\n")
        if flagged:
            L.append("| lang | estimated tokens | script | 131k F1 (if evaluated) |")
            L.append("|---|---|---|---|")
            for l, est in flagged:
                f1 = f"{f131[l]:.3f}" if (name == "131k" and l in f131) else ""
                L.append(f"| {l} | {est} | {script.get(l, '?')} | {f1} |")
        L.append("")
    os.makedirs(os.path.dirname(out_md), exist_ok=True)
    with open(out_md, "w") as f:
        f.write("\n".join(L) + "\n")
    print(f"Wrote {out_md}")
    return out_md


if __name__ == "__main__":
    run()
