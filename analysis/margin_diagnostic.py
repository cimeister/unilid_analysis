"""Margin diagnostic for the per-language decision threshold (plan B3).

Question: on lines the baseline routes into a tail label, is the score gap between
the winning tail language and the runner-up separable from the gaps on genuine lines
of that language? If yes, a per-language margin tau_L, calibrated on L's OWN training
lines only (modular: L's data plus the shared model), can reassign false positives to
the runner-up at a recall cost bounded by the calibration quantile.

Design (constants pre-registered in the 2026-07-23 plan, not swept):
- Margin = top1 minus top2 score from top_k_of_cached_weight_sets_batch(pre, TOPK_MARGIN).
- tau_L = the MARGIN_Q-th percentile of margins on L's train lines that L itself wins
  (top1 == L). Train-side own-recall loss <= MARGIN_Q% by construction; the honest
  test-side recall cost is measured separately on the true-tail test lines.
- Languages with fewer than MIN_CALIB_LINES scoreable train lines are EXCLUDED from
  gating (tau = -inf, never suppressed) and listed in the report; explicit and
  auditable, never a silent default.
- Calibration lines are a deterministic random sample (CALIB_SEED) of up to CALIB_MAX
  lines per language from the language's corpus file.

Scored line sets (from the saved baseline memmap; texts from one streaming pass over
the test file):
- FP-into-tail: pred is a tail language, pred != true (expected 22,522, gated);
- true-tail: true label is a tail language (expected 7,735, gated).
Scorer agreement gate: recomputed top-1 must match the memmap prediction on >= 99% of
the FP-into-tail lines.

Report per absorbing tail language (and aggregate): FP catch rate at tau_L, fraction
of caught FPs whose true label is the runner-up (recovery), margin AUC (genuine train
vs FP), test-side genuine suppression rate and its runner-up destinations (tail-to-
tail cascade check). Outputs outputs/tables/margin_diagnostic.md and
outputs/diagnostic/tau_per_lang.csv. This is a diagnostic: no predictions are changed
here and nothing is adopted from this script.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from analysis.config import TEST_FILE, TOTAL_LINES
from analysis.transfer_sweep import _load_unilid_model
from analysis.full_test_eval import SCRATCH_DIR, _parse_line

PRF_CSV = "outputs/diagnostic/full_test_per_lang_prf.csv"
CORPUS_DIR = "/capstor/scratch/cscs/cmeister747/unilid_analysis/results_apertus200k/corpus"
OUT_MD = "outputs/tables/margin_diagnostic.md"
OUT_CSV = "outputs/diagnostic/tau_per_lang.csv"
TOPK_MARGIN = 5
MARGIN_Q = 5                 # percentile for tau_L (own-recall bound, pre-registered)
MIN_CALIB_LINES = 200        # below this, a language is excluded from gating (logged)
CALIB_MAX = 2_000
CALIB_SEED = 0
EXPECTED_FP = 22_522         # recorded in outputs/tables/metric_decomposition.md
EXPECTED_TRUE_TAIL = 7_735
SCORE_CHUNK = 5_000


def _topk_batch(model, texts, drop_log):
    """Preprocess + top-k score a list of texts; returns (kept_positions, topk)."""
    pre, pos = [], []
    for i, t in enumerate(texts):
        p = model.preprocess(t)
        if p:
            pre.append(p)
            pos.append(i)
        else:
            drop_log.append(i)
    out = []
    for lo in range(0, len(pre), SCORE_CHUNK):
        out.extend(model.model.top_k_of_cached_weight_sets_batch(
            pre[lo:lo + SCORE_CHUNK], TOPK_MARGIN))
    return pos, out


def _gap(cands):
    return cands[0][1] - cands[1][1] if len(cands) >= 2 else float("inf")


def _auc(pos_vals, neg_vals):
    """Rank AUC: P(genuine margin > FP margin), ties counted half."""
    if not len(pos_vals) or not len(neg_vals):
        return float("nan")
    both = np.concatenate([pos_vals, neg_vals])
    ranks = pd.Series(both).rank().values
    r_pos = ranks[:len(pos_vals)].sum()
    n_p, n_n = len(pos_vals), len(neg_vals)
    return (r_pos - n_p * (n_p + 1) / 2) / (n_p * n_n)


def run(out_md: str = OUT_MD, out_csv: str = OUT_CSV) -> str:
    prf = pd.read_csv(PRF_CSV)
    langs = prf.lang.tolist()
    n_lang = len(langs)
    N = prf.N.values
    tail_idx = np.where(N < 1_000)[0]
    tail_set = set(int(i) for i in tail_idx)

    y = np.asarray(np.lib.format.open_memmap(
        os.path.join(SCRATCH_DIR, "y_true.npy"), mode="r"))
    pb = np.asarray(np.lib.format.open_memmap(
        os.path.join(SCRATCH_DIR, "pred_baseline.npy"), mode="r"))
    fp_mask = np.isin(pb, tail_idx) & (pb != y) & (pb >= 0) & (y >= 0)
    tt_mask = np.isin(y, tail_idx)
    if int(fp_mask.sum()) != EXPECTED_FP:
        raise RuntimeError(f"FP-into-tail count {int(fp_mask.sum())} != recorded "
                           f"{EXPECTED_FP}")
    if int(tt_mask.sum()) != EXPECTED_TRUE_TAIL:
        raise RuntimeError(f"true-tail count {int(tt_mask.sum())} != recorded "
                           f"{EXPECTED_TRUE_TAIL}")

    need = np.where(fp_mask | tt_mask)[0]
    want = set(need.tolist())
    texts = {}
    with open(TEST_FILE) as fh:
        for i in range(TOTAL_LINES):
            line = fh.readline()
            if i not in want:
                continue
            _label, text = _parse_line(line)
            texts[i] = text
    if len(texts) != len(need):
        raise RuntimeError(f"collected {len(texts)} texts for {len(need)} lines")

    model = _load_unilid_model()

    # --- score the test-side sets ---
    fp_lines = np.where(fp_mask)[0]
    drop = []
    fp_pos, fp_topk = _topk_batch(model, [texts[i] for i in fp_lines], drop)
    if drop:
        raise RuntimeError(f"{len(drop)} FP lines empty after preprocess; the memmap "
                           "scored them non-empty, scorer state differs")
    agree = np.mean([fp_topk[k][0][0] == int(pb[fp_lines[fp_pos[k]]])
                     for k in range(len(fp_topk))])
    if agree < 0.99:
        raise RuntimeError(f"top-1 agreement with the saved baseline predictions "
                           f"{agree:.4f} < 0.99 on FP lines")
    print(f"scored {len(fp_topk):,} FP lines (top-1 agreement {agree:.4f})")

    tt_lines = np.where(tt_mask)[0]
    tt_drop = []
    tt_pos, tt_topk = _topk_batch(model, [texts[i] for i in tt_lines], tt_drop)
    print(f"scored {len(tt_topk):,} true-tail lines "
          f"({len(tt_drop)} empty after preprocess, excluded)")

    # --- calibration on each tail language's own train lines ---
    rng = np.random.default_rng(CALIB_SEED)
    tau = {}
    calib_rows = []
    genuine_gaps = {}
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
        tau[li] = float("-inf") if excluded else float(np.percentile(gaps, MARGIN_Q))
        genuine_gaps[li] = gaps
        calib_rows.append({
            "lang": lang, "n_corpus_lines": len(lines),
            "n_scoreable": len(ctopk), "n_self_won": len(wins),
            "train_top1_rate": len(wins) / len(ctopk) if ctopk else 0.0,
            "tau": tau[li], "excluded": excluded})
    pd.DataFrame(calib_rows).to_csv(out_csv, index=False)
    excluded_langs = [r["lang"] for r in calib_rows if r["excluded"]]

    # --- per-language and aggregate effects ---
    fp_by_lang = {}
    for k, cands in enumerate(fp_topk):
        line = fp_lines[fp_pos[k]]
        li = int(pb[line])
        fp_by_lang.setdefault(li, []).append((line, cands))
    per_lang = []
    tot = {"fp": 0, "caught": 0, "recovered": 0, "gen": 0, "supp": 0,
           "supp_to_tail": 0}
    for li in tail_idx:
        entries = fp_by_lang.get(int(li), [])
        gcaps = genuine_gaps[li]
        t = tau[li]
        caught = [(line, c) for line, c in entries if _gap(c) < t]
        recovered = [1 for line, c in caught
                     if len(c) >= 2 and c[1][0] == int(y[line])]
        fgaps = np.array([_gap(c) for _line, c in entries])
        fgaps = fgaps[np.isfinite(fgaps)]
        # test-side genuine suppression for L
        gen = [(tt_lines[tt_pos[k]], c) for k, c in enumerate(tt_topk)
               if c and c[0][0] == li and int(y[tt_lines[tt_pos[k]]]) == li]
        supp = [c for _line, c in gen if _gap(c) < t]
        supp_to_tail = sum(1 for c in supp if len(c) >= 2 and c[1][0] in tail_set)
        per_lang.append({
            "lang": langs[li], "n_fp": len(entries), "tau": t,
            "excluded": not np.isfinite(t),
            "fp_catch": len(caught) / len(entries) if entries else float("nan"),
            "recover": sum(recovered) / len(caught) if caught else float("nan"),
            "auc": _auc(gcaps, fgaps),
            "n_genuine_test_won": len(gen),
            "test_suppressed": len(supp) / len(gen) if gen else float("nan"),
            "supp_to_tail": supp_to_tail})
        tot["fp"] += len(entries)
        tot["caught"] += len(caught)
        tot["recovered"] += sum(recovered)
        tot["gen"] += len(gen)
        tot["supp"] += len(supp)
        tot["supp_to_tail"] += supp_to_tail

    dfp = pd.DataFrame(per_lang).sort_values("n_fp", ascending=False)
    L = ["# Margin diagnostic (per-language decision threshold, plan B3)\n",
         f"Margin = top1 minus top2 of {TOPK_MARGIN}-best scores. tau_L = "
         f"{MARGIN_Q}th percentile of L's own train-line margins (lines L wins); "
         f"calibration sample <= {CALIB_MAX} lines, seed {CALIB_SEED}; languages "
         f"with < {MIN_CALIB_LINES} scoreable self-won train lines are excluded "
         f"from gating (tau = -inf).",
         f"Excluded languages ({len(excluded_langs)}): "
         + (", ".join(excluded_langs) if excluded_langs else "none") + ".\n",
         f"## Aggregate (all {len(tail_idx)} tail languages)\n",
         f"- FP-into-tail lines scored: {tot['fp']:,}; caught at tau: "
         f"{tot['caught']:,} ({tot['caught'] / tot['fp']:.1%}); caught lines whose "
         f"runner-up is the true label: {tot['recovered']:,} "
         f"({tot['recovered'] / max(tot['caught'], 1):.1%} of caught).",
         f"- Genuine true-tail test lines won by their own language: {tot['gen']:,}; "
         f"suppressed at tau: {tot['supp']:,} "
         f"({tot['supp'] / max(tot['gen'], 1):.1%}); of the suppressed, "
         f"{tot['supp_to_tail']:,} have another tail language as runner-up "
         "(cascade check).",
         "\n## Per-language table (sorted by FP count; NaN = no FPs or no wins)\n",
         dfp.round(4).to_markdown(index=False)]
    os.makedirs(os.path.dirname(out_md), exist_ok=True)
    with open(out_md, "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L[:12]))
    print(f"\nWrote {out_md} and {out_csv}")
    return out_md


if __name__ == "__main__":
    run()
