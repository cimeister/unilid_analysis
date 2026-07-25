"""Full-test margin-gated candidate predictions (config margin_q5, plan B3 follow-up).

Builds the prediction memmap for the margin rule found viable in Exp 26: for every
line whose baseline prediction is a gated tail language, if the recomputed top1/top2
score gap is below that language's tau_L (outputs/diagnostic/tau_per_lang.csv,
calibrated on the language's own training data only, Exp 26), predict the runner-up
instead. All other lines keep the saved baseline prediction bit-identically.

Coverage fact (gated at runtime): every kept line with a tail baseline prediction is
either an FP-into-tail line or a true-tail line, the two sets scored in Exp 26, so
only ~30k lines need scoring here.

Determinism note: the gate is applied only where the recomputed top-1 equals the
saved baseline prediction; the known scorer self-agreement is ~0.995, and the
disagreeing remainder keeps the baseline prediction ungated (counted and reported).

Output: pred_margin_q5.npy in the full-test scratch dir (same layout as the other
config memmaps) plus a build report. Adoption is judged afterwards by
analysis/two_sided_report.py with margin_q5 added as a candidate.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from analysis.config import TEST_FILE, TOTAL_LINES
from analysis.transfer_sweep import _load_unilid_model
from analysis.full_test_eval import SCRATCH_DIR, _parse_line
from analysis.margin_diagnostic import (_topk_batch, _gap, PRF_CSV, OUT_CSV as TAU_CSV,
                                        EXPECTED_FP, EXPECTED_TRUE_TAIL, TOPK_MARGIN)

OUT_PRED = os.path.join(SCRATCH_DIR, "pred_margin_q5.npy")
OUT_MD = "outputs/tables/full_test_margin_build.md"
HEAD_N = 18_000   # reassignment-target threshold for target="head"; the established
                  # head stratum bound (full_test_eval.lang_flags)


def run(target: str = "runner_up") -> str:
    """target="runner_up": Exp 26 rule, reassign to cands[1] (config margin_q5).
    target="head": pre-registered follow-up (Exp 26 addendum): reassign to the
    highest-scoring candidate with N >= HEAD_N in the top-5; if none, keep the
    baseline prediction (config margin_q5_head). Motivated by the measured szy_Latn
    mechanism: runner-up reassignment moved 82 of pwn_Latn's caught FPs onto its
    Formosan neighbor."""
    if target not in ("runner_up", "head"):
        raise ValueError(f"unknown target {target!r}")
    name = "margin_q5" if target == "runner_up" else "margin_q5_head"
    out_pred = (OUT_PRED if target == "runner_up"
                else os.path.join(SCRATCH_DIR, f"pred_{name}.npy"))
    out_md = (OUT_MD if target == "runner_up"
              else f"outputs/tables/full_test_{name}_build.md")
    prf = pd.read_csv(PRF_CSV)
    langs = prf.lang.tolist()
    N = prf.N.values
    tail_idx = np.where(N < 1_000)[0]

    taus = pd.read_csv(TAU_CSV)
    tau_by_lang = dict(zip(taus.lang, taus.tau))
    missing = [langs[i] for i in tail_idx if langs[i] not in tau_by_lang]
    if missing:
        raise RuntimeError(f"tau missing for {len(missing)} tail languages, first: "
                           f"{missing[0]}; run margin_diagnostic first")
    tau = {int(i): float(tau_by_lang[langs[i]]) for i in tail_idx}

    y = np.asarray(np.lib.format.open_memmap(
        os.path.join(SCRATCH_DIR, "y_true.npy"), mode="r"))
    pb = np.asarray(np.lib.format.open_memmap(
        os.path.join(SCRATCH_DIR, "pred_baseline.npy"), mode="r"))
    fp_mask = np.isin(pb, tail_idx) & (pb != y) & (pb >= 0) & (y >= 0)
    tt_mask = np.isin(y, tail_idx)
    if int(fp_mask.sum()) != EXPECTED_FP or int(tt_mask.sum()) != EXPECTED_TRUE_TAIL:
        raise RuntimeError("FP/true-tail counts do not match the Exp 24 record")
    affected = np.where((pb >= 0) & np.isin(pb, tail_idx))[0]
    if not np.all(fp_mask[affected] | tt_mask[affected]):
        raise RuntimeError("a tail-predicted line is neither FP nor true-tail; "
                           "coverage claim broken")
    print(f"{len(affected):,} lines carry a tail baseline prediction")

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

    model = _load_unilid_model()
    drop = []
    pos, topk = _topk_batch(model, [texts[i] for i in affected], drop)
    if drop:
        raise RuntimeError(f"{len(drop)} tail-predicted lines empty after preprocess")
    agree = np.mean([topk[k][0][0] == int(pb[affected[pos[k]]])
                     for k in range(len(topk))])
    if agree < 0.99:
        raise RuntimeError(f"top-1 agreement {agree:.4f} < 0.99")

    pred = np.array(pb, dtype=np.int16)
    n_disagree = n_gated = n_reassigned = n_to_true = n_no_target = 0
    for k, cands in enumerate(topk):
        line = int(affected[pos[k]])
        top1 = int(cands[0][0])
        if top1 != int(pb[line]):
            n_disagree += 1
            continue
        t = tau[top1]
        n_gated += 1
        if not (len(cands) >= 2 and _gap(cands) < t):
            continue
        if target == "runner_up":
            dest = int(cands[1][0])
        else:
            head_cands = [int(c[0]) for c in cands[1:] if N[int(c[0])] >= HEAD_N]
            if not head_cands:
                n_no_target += 1
                continue
            dest = head_cands[0]
        pred[line] = np.int16(dest)
        n_reassigned += 1
        n_to_true += int(dest == int(y[line]))
    np.save(out_pred, pred)

    L = [f"# {name} candidate build (plan B3 follow-up, Exp 26 rule, "
         f"target={target})\n",
         f"- Tail-predicted lines scored: {len(affected):,} (top-1 agreement "
         f"{agree:.4f}; {n_disagree} disagreeing lines left ungated at the baseline "
         "prediction).",
         f"- Reassigned: {n_reassigned:,} of {n_gated:,} gated lines; {n_to_true:,} "
         f"reassignments land on the true label."
         + (f" Below-tau lines kept at baseline for lack of a head candidate in the "
            f"top-{TOPK_MARGIN}: {n_no_target:,}."
            if target == "head" else ""),
         f"- All other lines are bit-identical to pred_baseline.npy. Output: "
         f"{out_pred}.",
         "- tau source: outputs/diagnostic/tau_per_lang.csv (Exp 26, job 2883715); "
         "excluded languages have tau = -inf and are never gated."]
    os.makedirs(os.path.dirname(out_md), exist_ok=True)
    with open(out_md, "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))
    return out_pred


if __name__ == "__main__":
    run()
