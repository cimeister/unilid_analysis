"""Composition candidate gt_margin (pre-registered 2026-07-24, EXPERIMENTS_RESULTS):
gt_min weights (Exp 28) plus the head-targeted margin gate (Exp 26 rule) with tau_L
recalibrated under the gt_min matrix.

Pipeline (all constants inherited from their pre-registrations, nothing swept):
1. Rebuild the gt_min matrix deterministically (build_gt_weights, all gates) and
   verify its sha256 against fingerprint_gt.json (the matrix that produced
   pred_gt_min.npy).
2. Cache it in the scorer; recalibrate tau_L for the 96 tail languages on their own
   train lines (MARGIN_Q=5th percentile of self-won margins, MIN_CALIB_LINES=200
   exclusion logged, CALIB_MAX=2000, CALIB_SEED=0).
3. Affected lines = kept lines with a tail prediction in pred_gt_min.npy; score them
   top-TOPK_MARGIN under gt_min (top-1 agreement gate >= 0.99 vs the memmap); where
   the margin is below tau[top1], reassign to the highest-scoring candidate with
   N >= HEAD_N in the top list, else keep the gt_min prediction.
4. Write pred_gt_margin.npy (bit-identical to pred_gt_min off the gated lines),
   tau CSV, and a build report. Adoption judged by two_sided_report (both tracks).

Modularity: tau uses each language's own train data plus the shared gt_min model;
no draw data is fit (CONFIG_FIT_DRAWS entry stays empty).
"""
from __future__ import annotations

import hashlib
import json
import os

import numpy as np
import pandas as pd

from analysis.config import TEST_FILE, TOTAL_LINES
from analysis.transfer_sweep import _load_model_data, _load_unilid_model
from analysis.full_test_eval import SCRATCH_DIR, _parse_line
from analysis.full_test_gt import build_gt_weights, GT_CSV
from analysis.full_test_margin import HEAD_N
from analysis.margin_diagnostic import (_topk_batch, _gap, PRF_CSV, TOPK_MARGIN,
                                        MARGIN_Q, MIN_CALIB_LINES, CALIB_MAX,
                                        CALIB_SEED, CORPUS_DIR)

OUT_PRED = os.path.join(SCRATCH_DIR, "pred_gt_margin.npy")
OUT_TAU = "outputs/diagnostic/tau_per_lang_gtmin.csv"
OUT_MD = "outputs/tables/gt_margin_build.md"


def run(gate: str = "tail") -> str:
    """gate="tail": the Exp 31 candidate gt_margin (gated labels = tail, N<1k).
    gate="nonhead": the Exp 32 pre-registered candidate gt_margin_all (gated labels
    = every language with N < HEAD_N; the dig-in showed every victim suffers FP
    inflow at a non-head label, so the defense must cover tail AND lowmid)."""
    if gate not in ("tail", "nonhead"):
        raise ValueError(f"unknown gate {gate!r}")
    prf = pd.read_csv(PRF_CSV)
    langs = prf.lang.tolist()
    n_lang = len(langs)
    N = prf.N.values
    tail_idx = (np.where(N < 1_000)[0] if gate == "tail"
                else np.where(N < HEAD_N)[0])
    name = "gt_margin" if gate == "tail" else "gt_margin_all"
    out_pred = (OUT_PRED if gate == "tail"
                else os.path.join(SCRATCH_DIR, "pred_gt_margin_all.npy"))
    out_tau = (OUT_TAU if gate == "tail"
               else "outputs/diagnostic/tau_per_lang_gtmin_all.csv")
    out_md = (OUT_MD if gate == "tail"
              else "outputs/tables/gt_margin_all_build.md")

    weights, langs_m, _m = _load_model_data()
    if langs_m != langs:
        raise RuntimeError("model language order differs from the PRF CSV")
    W = np.array(weights, dtype=np.float32)
    del weights
    gt = pd.read_csv(GT_CSV)
    wgt = build_gt_weights(W, gt, langs)
    with open(os.path.join(SCRATCH_DIR, "fingerprint_gt.json")) as f:
        fp = json.load(f)
    sha = hashlib.sha256(wgt.tobytes()).hexdigest()
    if sha != fp["sha256_wgt"]:
        raise RuntimeError("rebuilt gt_min matrix does not match the fingerprint of "
                           "the matrix that produced pred_gt_min.npy")
    del W

    y = np.asarray(np.lib.format.open_memmap(
        os.path.join(SCRATCH_DIR, "y_true.npy"), mode="r"))
    pg = np.asarray(np.lib.format.open_memmap(
        os.path.join(SCRATCH_DIR, "pred_gt_min.npy"), mode="r"))
    affected = np.where((pg >= 0) & np.isin(pg, tail_idx))[0]
    print(f"{len(affected):,} lines carry a gated gt_min prediction")

    model = _load_unilid_model()
    print("Caching gt_min weights...", flush=True)
    model.model.set_weight_sets(wgt.tolist())
    del wgt

    # --- tau recalibration under gt_min ---
    rng = np.random.default_rng(CALIB_SEED)
    tau = {}
    calib_rows = []
    for li in tail_idx:
        lang = langs[li]
        path = os.path.join(CORPUS_DIR, f"{lang}_train.txt")
        if not os.path.exists(path):
            raise FileNotFoundError(f"corpus file missing: {path}")
        with open(path) as f:
            lines = [l.rstrip("\n") for l in f if l.rstrip("\n")]
        if len(lines) > CALIB_MAX:
            lines = [lines[k] for k in
                     sorted(rng.choice(len(lines), CALIB_MAX, replace=False))]
        cdrop = []
        _cpos, ctopk = _topk_batch(model, lines, cdrop)
        wins = [c for c in ctopk if c and c[0][0] == li]
        gaps = np.array([_gap(c) for c in wins])
        gaps = gaps[np.isfinite(gaps)]
        excluded = len(gaps) < MIN_CALIB_LINES
        tau[int(li)] = (float("-inf") if excluded
                        else float(np.percentile(gaps, MARGIN_Q)))
        calib_rows.append({"lang": lang, "n_scoreable": len(ctopk),
                           "n_self_won": len(wins), "tau": tau[int(li)],
                           "excluded": excluded})
    os.makedirs(os.path.dirname(out_tau), exist_ok=True)
    pd.DataFrame(calib_rows).to_csv(out_tau, index=False)
    n_excluded = sum(r["excluded"] for r in calib_rows)

    # --- score affected lines and apply the head-targeted gate ---
    want = set(affected.tolist())
    texts = {}
    with open(TEST_FILE) as fh:
        for i in range(TOTAL_LINES):
            line = fh.readline()
            if i not in want:
                continue
            _label, text = _parse_line(line)
            texts[i] = text
    if len(texts) != len(affected):
        raise RuntimeError(f"collected {len(texts)} texts for {len(affected)} lines")
    drop = []
    pos, topk = _topk_batch(model, [texts[i] for i in affected], drop)
    if drop:
        raise RuntimeError(f"{len(drop)} affected lines empty after preprocess")
    agree = np.mean([topk[k][0][0] == int(pg[affected[pos[k]]])
                     for k in range(len(topk))])
    if agree < 0.99:
        raise RuntimeError(f"top-1 agreement with pred_gt_min {agree:.4f} < 0.99")

    pred = np.array(pg, dtype=np.int16)
    n_disagree = n_gated = n_reassigned = n_to_true = n_no_target = 0
    for k, cands in enumerate(topk):
        line = int(affected[pos[k]])
        top1 = int(cands[0][0])
        if top1 != int(pg[line]):
            n_disagree += 1
            continue
        t = tau[top1]
        n_gated += 1
        if not (len(cands) >= 2 and _gap(cands) < t):
            continue
        head_cands = [int(c[0]) for c in cands[1:] if N[int(c[0])] >= HEAD_N]
        if not head_cands:
            n_no_target += 1
            continue
        pred[line] = np.int16(head_cands[0])
        n_reassigned += 1
        n_to_true += int(head_cands[0] == int(y[line]))
    np.save(out_pred, pred)

    L = [f"# {name} candidate build (pre-registered composition; gate={gate})\n",
         f"- gt_min matrix rebuilt and fingerprint-verified (sha {sha[:16]}...).",
         f"- tau recalibrated under gt_min: {n_excluded} of {len(tail_idx)} gated "
         f"languages excluded (< {MIN_CALIB_LINES} self-won train lines); values in "
         f"{out_tau}.",
         f"- Gated gt_min-predicted lines scored: {len(affected):,} (top-1 agreement "
         f"{agree:.4f}; {n_disagree} disagreeing lines left ungated).",
         f"- Reassigned to a head candidate: {n_reassigned:,} of {n_gated:,} gated "
         f"lines; {n_to_true:,} land on the true label; {n_no_target:,} below-tau "
         f"lines kept for lack of a head candidate in the top-{TOPK_MARGIN}.",
         f"- All other lines bit-identical to pred_gt_min.npy. Output: {out_pred}."]
    os.makedirs(os.path.dirname(out_md), exist_ok=True)
    with open(out_md, "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))
    return out_pred


if __name__ == "__main__":
    run()
