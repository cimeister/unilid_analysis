"""Post-training gate and effect measurement for the fp64 retrains (Exp 42).

Compares each Apertus model trained under the corrupted float32 E-step against its
retrain under the double-precision E-step (fork commits d0208d9, c5921a2), and
re-runs the degeneracy gate on the new models.

Per model pair, all computed row by row from the packed weight matrices (no
scoring):
- degenerate rows before and after (fewer than MIN_ESTIMATED entries above the
  row minimum), with the flagged language lists;
- how many rows changed materially (max absolute log-probability difference over
  the row above 1 and above 5 nats);
- row entropy before and after, and the languages whose entropy moved most;
- explicit rows for the diagnosed victims: azj_Latn (the collapse) and the
  Cree-syllabics cluster (the vocabulary-coverage class, which the fix must NOT
  change in kind).

Gates: both models in a pair must carry the same language list in the same order,
and the same vocabulary width; abort otherwise.

Output: outputs/tables/fp64_retrain_check.md.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from analysis.transfer_sweep import _load_model_data

SCR = "/capstor/scratch/cscs/cmeister747/unilid_analysis"
PAIRS = {
    "131k": (f"{SCR}/glotlid_apertus131k.unilid",
             f"{SCR}/glotlid_apertus131k_fp64.unilid"),
    "200k": (f"{SCR}/glotlid_apertus200k.unilid",
             f"{SCR}/glotlid_apertus200k_fp64.unilid"),
}
MIN_ESTIMATED = 100          # the Exp 35 degeneracy threshold
BIG_DELTA = 1.0              # nats: "materially changed" row
HUGE_DELTA = 5.0             # nats
CREE = ["csw_Cans", "cwd_Cans", "crj_Cans", "crk_Cans", "crl_Cans", "crm_Cans"]
WATCH = ["azj_Latn", "tat_Latn", "quc_Latn", "fra_Latn"] + CREE
OUT_MD = "outputs/tables/fp64_retrain_check.md"


def _row_stats(row: np.ndarray) -> tuple[int, float]:
    """(entries above the row minimum, entropy in nats)."""
    above = int((row > row.min()).sum())
    p = np.exp(row)
    s = p.sum()
    if s <= 0:
        raise RuntimeError("row has non-positive total mass")
    p = p / s
    nz = p > 0
    return above, float(-(p[nz] * np.log(p[nz])).sum())


def run(out_md: str = OUT_MD) -> str:
    L = ["# fp64 retrain check: corrupted vs corrected trainer (Exp 42)\n",
         "Row-level comparison of each Apertus model against its retrain under the "
         "double-precision E-step. All numbers read from the packed weight "
         "matrices; no scoring. A row is degenerate when fewer than "
         f"{MIN_ESTIMATED} entries lie above the row minimum.\n"]
    for tag, (old_path, new_path) in PAIRS.items():
        for p in (old_path, new_path):
            if not os.path.exists(p):
                raise FileNotFoundError(f"model missing: {p}")
        Wo, langs_o, _ = _load_model_data(old_path)
        Wn, langs_n, _ = _load_model_data(new_path)
        if langs_o != langs_n:
            raise RuntimeError(f"{tag}: language lists differ between the models")
        if Wo.shape != Wn.shape:
            raise RuntimeError(f"{tag}: shapes differ {Wo.shape} vs {Wn.shape}")
        n_lang, vocab = Wo.shape

        deg_o, deg_n, big, huge = [], [], 0, 0
        ent_o = np.zeros(n_lang)
        ent_n = np.zeros(n_lang)
        maxdelta = np.zeros(n_lang)
        for i in range(n_lang):
            ro = np.asarray(Wo[i], dtype=np.float64)
            rn = np.asarray(Wn[i], dtype=np.float64)
            ao, ent_o[i] = _row_stats(ro)
            an, ent_n[i] = _row_stats(rn)
            if ao < MIN_ESTIMATED:
                deg_o.append(langs_o[i])
            if an < MIN_ESTIMATED:
                deg_n.append(langs_o[i])
            d = float(np.abs(ro - rn).max())
            maxdelta[i] = d
            if d > BIG_DELTA:
                big += 1
            if d > HUGE_DELTA:
                huge += 1

        L.append(f"## {tag} ({vocab:,} vocabulary entries, {n_lang:,} languages)\n")
        L.append(f"- Degenerate rows: {len(deg_o)} before, {len(deg_n)} after.")
        gone = sorted(set(deg_o) - set(deg_n))
        stayed = sorted(set(deg_o) & set(deg_n))
        new_deg = sorted(set(deg_n) - set(deg_o))
        L.append(f"  - Repaired by the fix: {', '.join(gone) if gone else 'none'}.")
        L.append(f"  - Still degenerate (expected: the vocabulary-coverage class, "
                 f"unrelated to the bug): {', '.join(stayed) if stayed else 'none'}.")
        L.append(f"  - Newly degenerate: "
                 f"{', '.join(new_deg) if new_deg else 'none'}.")
        L.append(f"- Rows changed by more than {BIG_DELTA} nat at their largest "
                 f"entry: {big:,} of {n_lang:,}; by more than {HUGE_DELTA} nats: "
                 f"{huge:,}.")
        L.append(f"- Row entropy: mean {ent_o.mean():.4f} -> {ent_n.mean():.4f} "
                 f"nats.\n")
        order = np.argsort(-np.abs(ent_n - ent_o))[:8]
        L.append("Largest entropy movements:\n")
        L.append("| lang | entropy before | entropy after | max |delta| (nats) |")
        L.append("|---|---|---|---|")
        for i in order:
            L.append(f"| {langs_o[i]} | {ent_o[i]:.3f} | {ent_n[i]:.3f} | "
                     f"{maxdelta[i]:.2f} |")
        L.append("\nDiagnosed victims and controls:\n")
        L.append("| lang | above-min before | above-min after | entropy before | "
                 "entropy after |")
        L.append("|---|---|---|---|---|")
        for lang in WATCH:
            if lang not in langs_o:
                continue
            i = langs_o.index(lang)
            ao, _ = _row_stats(np.asarray(Wo[i], dtype=np.float64))
            an, _ = _row_stats(np.asarray(Wn[i], dtype=np.float64))
            L.append(f"| {lang} | {ao:,} | {an:,} | {ent_o[i]:.3f} | "
                     f"{ent_n[i]:.3f} |")
        L.append("")
        del Wo, Wn

    os.makedirs(os.path.dirname(out_md), exist_ok=True)
    with open(out_md, "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))
    return out_md


if __name__ == "__main__":
    run()
