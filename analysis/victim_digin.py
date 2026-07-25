"""Dig-ins required by the outlier-tolerant collapse clause (Exp 32).

Cases (from the 2026-07-24 dual-track report and Exp 30):
1. llb_Latn under learned_bias (natural-track flag) and the gt_margin lowmid class
   (llb, arq_Arab, skr_Arab, vmk_Latn): per-victim mechanism on the veto instrument:
   false-positive inflow (new sources) versus recall loss (new destinations).
2. sbs_Latn and mev_Latn under gt_min on the balanced test draw (uniform-track
   flags): same decomposition on the draw.
3. The 131k azj/tat pathology (Exp 30): weight-row statistics for tat_Latn and
   azj_Latn in the 100k vs 131k matrices (entropy, floor, plateau size) and the
   destination profile of azj's test lines under each model.

Descriptive counting only; every instrument mask reproduces the recorded totals.
Output: outputs/tables/victim_digins.md.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from analysis.config import TOTAL_LINES
from analysis.metric_decomposition import _per_lang_stats
from analysis.balanced_split import OUT_DIR as DRAW_DIR, SEEDS, TEST_SEED
from analysis.full_test_eval import SCRATCH_DIR
from analysis.full_test_eval_131k import (SCRATCH_DIR as SCRATCH_131K,
                                          MODEL_131K)
from analysis.transfer_sweep import _load_model_data

PRF_CSV = "outputs/diagnostic/full_test_per_lang_prf.csv"
OUT_MD = "outputs/tables/victim_digins.md"
TOP = 5


def _flows(y, base, cand, li, mask, langs):
    """Mechanism decomposition for language li between two prediction sets on mask:
    new FP sources (lines newly predicted li) and new recall destinations (true-li
    lines newly lost)."""
    m_new_fp = mask & (cand == li) & (base != li) & (y != li)
    m_lost = mask & (y == li) & (base == li) & (cand != li)
    m_fp_gone = mask & (base == li) & (cand != li) & (y != li)
    src = pd.Series([langs[i] for i in y[m_new_fp]]).value_counts().head(TOP)
    dst = pd.Series([langs[i] for i in cand[m_lost]]).value_counts().head(TOP)
    return {
        "new_fp": int(m_new_fp.sum()), "fp_removed": int(m_fp_gone.sum()),
        "recall_lost": int(m_lost.sum()),
        "true_n": int((mask & (y == li)).sum()),
        "src": dict(src), "dst": dict(dst)}


def run(out_md: str = OUT_MD) -> str:
    prf = pd.read_csv(PRF_CSV)
    langs = prf.lang.tolist()
    n_lang = len(langs)
    idx = {l: i for i, l in enumerate(langs)}
    y = np.asarray(np.lib.format.open_memmap(
        os.path.join(SCRATCH_DIR, "y_true.npy"), mode="r"))
    load = lambda p, d=SCRATCH_DIR: np.asarray(
        np.lib.format.open_memmap(os.path.join(d, p), mode="r"))
    pb = load("pred_baseline.npy")
    kept = y >= 0

    excl = np.zeros(TOTAL_LINES, bool)
    for s in [SEEDS[0], TEST_SEED]:
        excl[np.load(os.path.join(DRAW_DIR, f"val_lines_seed{s}.npy"))] = True
    veto = kept & ~excl
    if int(veto.sum()) != 45_004_014:
        raise RuntimeError(f"veto mask {int(veto.sum()):,} != recorded 45,004,014")
    tmask = np.zeros(TOTAL_LINES, bool)
    tmask[np.load(os.path.join(DRAW_DIR, f"val_lines_seed{TEST_SEED}.npy"))] = True
    if int(tmask.sum()) != 185_204:
        raise RuntimeError("test-draw mask size mismatch")

    L = ["# Victim dig-ins (Exp 32): flagged outliers and the 131k pathology\n"]

    L.append("## 1. llb_Latn under learned_bias; the gt_margin lowmid class "
             "(veto instrument)\n")
    L.append("| victim | config | true n | new FPs (top sources) | FPs removed | "
             "recall lost (top destinations) |")
    L.append("|---|---|---|---|---|---|")
    cases = [("llb_Latn", "learned_bias", load("pred_learned_bias.npy"))]
    pgm = load("pred_gt_margin.npy")
    for v in ["llb_Latn", "arq_Arab", "skr_Arab", "vmk_Latn"]:
        cases.append((v, "gt_margin", pgm))
    for v, cname, cand in cases:
        f = _flows(y, pb, cand, idx[v], veto, langs)
        src = ", ".join(f"{k} ({c})" for k, c in f["src"].items()) or "none"
        dst = ", ".join(f"{k} ({c})" for k, c in f["dst"].items()) or "none"
        L.append(f"| {v} | {cname} | {f['true_n']:,} | {f['new_fp']:,} ({src}) | "
                 f"{f['fp_removed']:,} | {f['recall_lost']:,} ({dst}) |")

    L.append("\n## 2. sbs_Latn and mev_Latn under gt_min (balanced test draw)\n")
    L.append("| victim | true n | new FPs (top sources) | FPs removed | "
             "recall lost (top destinations) |")
    L.append("|---|---|---|---|---|")
    pgt = load("pred_gt_min.npy")
    for v in ["sbs_Latn", "mev_Latn"]:
        f = _flows(y, pb, pgt, idx[v], tmask, langs)
        src = ", ".join(f"{k} ({c})" for k, c in f["src"].items()) or "none"
        dst = ", ".join(f"{k} ({c})" for k, c in f["dst"].items()) or "none"
        L.append(f"| {v} | {f['true_n']:,} | {f['new_fp']:,} ({src}) | "
                 f"{f['fp_removed']:,} | {f['recall_lost']:,} ({dst}) |")

    L.append("\n## 3. The 131k azj/tat pathology\n")
    w100, langs100, _ = _load_model_data()
    w131, langs131, _ = _load_model_data(MODEL_131K)
    if langs100 != langs or langs131 != langs:
        raise RuntimeError("model language order mismatch")
    L.append("| lang | model | entropy (nats) | floor | plateau size | "
             "plateau mass |")
    L.append("|---|---|---|---|---|---|")
    for lname in ["tat_Latn", "azj_Latn"]:
        li = idx[lname]
        for mname, W in [("100k", w100), ("131k", w131)]:
            row = np.asarray(W[li], dtype=np.float64)
            p = np.exp(row)
            H = float(-(p * row).sum())
            floor = float(row.min())
            plateau = row == floor
            L.append(f"| {lname} | {mname} | {H:.3f} | {floor:.2f} | "
                     f"{int(plateau.sum()):,} | {float(p[plateau].sum()):.4e} |")
    p131 = load("pred_baseline131k.npy", SCRATCH_131K)
    azj = idx["azj_Latn"]
    m_azj = kept & (y == azj)
    for mname, parr in [("100k", pb), ("131k", p131)]:
        dest = pd.Series([langs[i] for i in parr[m_azj]]).value_counts().head(4)
        acc = float((parr[m_azj] == azj).mean())
        L.append(f"\nazj_Latn test lines ({int(m_azj.sum()):,}) under {mname}: "
                 f"recall {acc:.4f}; top destinations "
                 + ", ".join(f"{k} ({c:,})" for k, c in dest.items()) + ".")

    os.makedirs(os.path.dirname(out_md), exist_ok=True)
    with open(out_md, "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))
    return out_md


if __name__ == "__main__":
    run()
