"""Learned per-language bias (calibration), generalizing the frequency prior (Exp 14).

Fits a free per-language offset b_L by maximizing the regularized conditional likelihood
of the true label under a softmax over each example's top-k candidate scores:
    minimize  -sum_x log softmax_{L in topk(x)}(score_L(x) + b_L)[true(x)]  + reg * ||b||^2
This is convex in b (bias-only multinomial logistic, restricted to the per-example candidate
set), solved with L-BFGS. The reg strength is selected on val; the test half is scored once,
exactly, with the Rust biased scorer (full 1940-way argmax + b). Modular: b_L is one scalar
per language. Compare against the gamma=0.5 frequency prior (+0.0058 macro-F1, Exp 14).

Needs the custom Rust build (top_k + biased scorers) + ~250 GB cache -> SLURM.
"""
from __future__ import annotations

import os

import numpy as np
from scipy.optimize import minimize

from analysis.config import DEFAULT_SAMPLE_SIZE
from analysis.transfer_sweep import (
    _load_model_data, _load_train_counts, _load_unilid_model, _stream_sampled_texts,
)
from analysis.sample_data import load_sample
from analysis.hierarchical_pool import (
    _strata, _macro_f1, _bootstrap_delta, passes_guard, GUARD_STRATA, GUARD_TOL,
    VAL_MASK, DIAG_CSV,
)

TOPK = 20
# L2 strength, selected on val. 5.0 and 7.0 added 2026-07-10 with the all-strata guard:
# the guarded region's boundary lies between reg=3 (fails on magnets) and reg=10 (passes),
# so the original grid would select its own endpoint. Recorded in EXPERIMENTAL_SETUP.md.
REGS = [0.3, 1.0, 3.0, 5.0, 7.0, 10.0]
OUT_DIR = "outputs"


def _flatten_topk(topk_lists, y_true_idx, n_lang):
    """Ragged top-k -> flat (cand_idx, cand_score, seg, true_flat_pos[, present])."""
    cand_idx, cand_score, seg = [], [], []
    true_flat_pos = np.full(len(topk_lists), -1, dtype=np.int64)
    pos = 0
    for e, cands in enumerate(topk_lists):
        ti = y_true_idx[e]
        for (idx, sc) in cands:
            cand_idx.append(idx); cand_score.append(sc); seg.append(e)
            if idx == ti:
                true_flat_pos[e] = pos
            pos += 1
    return (np.asarray(cand_idx, np.int64), np.asarray(cand_score, np.float64),
            np.asarray(seg, np.int64), true_flat_pos)


def _make_loss(cand_idx, cand_score, seg, true_flat_pos, n_seg, n_lang, reg):
    present = true_flat_pos >= 0
    true_pos = true_flat_pos[present]

    def loss_grad(b):
        logit = cand_score + b[cand_idx]
        segmax = np.full(n_seg, -np.inf)
        np.maximum.at(segmax, seg, logit)
        ex = np.exp(logit - segmax[seg])
        segsum = np.zeros(n_seg)
        np.add.at(segsum, seg, ex)
        p = ex / segsum[seg]                                  # softmax prob per candidate
        # loss = -sum log p[true] over present examples + reg||b||^2
        loss = -np.sum(np.log(np.maximum(p[true_pos], 1e-300))) + reg * np.dot(b, b)
        grad = np.zeros(n_lang)
        np.add.at(grad, cand_idx, p)                          # +P(L|x)
        np.add.at(grad, cand_idx[true_pos], -1.0)             # -1 for the true candidate
        grad += 2.0 * reg * b
        return loss, grad
    return loss_grad


def run(sample_size: int = DEFAULT_SAMPLE_SIZE, out_dir: str = OUT_DIR):
    import pandas as pd
    os.makedirs(os.path.join(out_dir, "tables"), exist_ok=True)
    weights, langs, lang_to_idx = _load_model_data()
    train_counts = _load_train_counts()
    N = np.array([train_counts.get(l, 0) for l in langs], dtype=np.float64)
    n_lang = len(langs)
    diag = pd.read_csv(DIAG_CSV)

    texts = _stream_sampled_texts(sample_size)
    data = load_sample(sample_size)
    y_true = np.array(data["y_true"])
    val = np.load(VAL_MASK); test = ~val
    strata = _strata(y_true, langs, lang_to_idx, N, diag)
    y_true_idx = np.array([lang_to_idx.get(l, -1) for l in y_true])

    model = _load_unilid_model()
    print("Preprocessing...")
    pre = [model.preprocess(t) for t in texts]
    # valid positions (non-empty preprocess)
    val_pos = [i for i in np.where(val)[0] if pre[i]]
    test_pos = np.where(test)[0]

    print("Extracting val top-k for the fit...")
    val_topk = model.model.top_k_of_cached_weight_sets_batch([pre[i] for i in val_pos], TOPK)
    ci, cs, seg, tfp = _flatten_topk(val_topk, y_true_idx[val_pos], n_lang)
    n_seg = len(val_pos)
    recall_at_k = (tfp >= 0).mean()
    print(f"  val top-{TOPK} recall of true label: {recall_at_k:.4f}")

    yv = y_true[val_pos]
    val_masks = {st: strata[st][val_pos] for st in strata}

    def val_strata_for_b(b):
        # vectorized argmax over each example's top-k of (score + b): within a seg, the
        # last entry after lexsort by (seg, logit) is the max-logit candidate.
        logit = cs + b[ci]
        order = np.lexsort((logit, seg))
        seg_sorted, idx_sorted = seg[order], ci[order]
        lasts = np.searchsorted(seg_sorted, np.arange(n_seg), side="right") - 1
        vp = np.array([langs[i] for i in idx_sorted[lasts]], dtype=object)
        return {st: _macro_f1(yv, vp, val_masks[st]) for st in strata}

    val_base = val_strata_for_b(np.zeros(n_lang))   # b=0 reproduces the unbiased argmax
    best = {"reg": None, "b": None, "val_overall": -1}
    rows = [{"reg": 0.0, "val_recall@k": round(recall_at_k, 4),
             **{f"val_{k}": round(v, 4) for k, v in val_base.items()}}]
    for reg in REGS:
        lg = _make_loss(ci, cs, seg, tfp, n_seg, n_lang, reg)
        res = minimize(lg, np.zeros(n_lang), jac=True, method="L-BFGS-B",
                       options={"maxiter": 200, "maxfun": 300})
        b = res.x.astype(np.float64)
        mvr = val_strata_for_b(b)
        rows.append({"reg": reg, "val_recall@k": round(recall_at_k, 4),
                     **{f"val_{k}": round(v, 4) for k, v in mvr.items()}})
        if passes_guard(mvr, val_base) and mvr["overall"] > best["val_overall"]:
            best = {"reg": reg, "b": b, "val_overall": mvr["overall"]}
        print(f"  reg={reg}: val " + " ".join(f"{k}={mvr[k]:.4f}" for k in
                                              ("overall", *GUARD_STRATA)))

    if best["b"] is None:
        # no reg is eligible: select the baseline (b = 0) and report the negative result,
        # matching prior_sweep's gamma=0 behaviour (no silent fallback to a fitted b)
        print("No reg passed the guard; selecting the baseline (b = 0).")
        best = {"reg": None, "b": np.zeros(n_lang), "val_overall": val_base["overall"]}

    b = best["b"].astype(np.float32)
    print(f"\nSelected reg={best['reg']}. Scoring test with the biased scorer...")
    test_texts = [pre[i] for i in test_pos]
    base_batch = model.model.best_of_cached_weight_sets_biased_batch(test_texts, [0.0] * n_lang)
    learn_batch = model.model.best_of_cached_weight_sets_biased_batch(test_texts, b.tolist())
    if len(base_batch) != len(test_texts) or len(learn_batch) != len(test_texts):
        raise RuntimeError(f"biased scorer returned {len(base_batch)}/{len(learn_batch)} "
                           f"results for {len(test_texts)} inputs; zip would misalign")
    base_pred = np.array(["?"] * len(y_true), dtype=object)
    learn_pred = np.array(["?"] * len(y_true), dtype=object)
    for j, (idx, _t, _s) in zip(test_pos, base_batch):
        base_pred[j] = langs[idx]
    for j, (idx, _t, _s) in zip(test_pos, learn_batch):
        learn_pred[j] = langs[idx]

    lines = ["# Learned per-language bias — TEST evaluation\n",
             f"Top-{TOPK} val recall of true label: {recall_at_k:.4f}. Selected reg={best['reg']}"
             " (None means no reg passed the guard; baseline b=0 evaluated).",
             f"Selection guard: val overall must improve and no stratum "
             f"({'/'.join(GUARD_STRATA)}) may drop more than {GUARD_TOL} vs baseline.",
             "Compare to the gamma=0.5 frequency prior (Exp 14): overall macro-F1 +0.0058.\n",
             "| stratum | base macroF1 | learned macroF1 | delta | 95% CI |",
             "|---|---|---|---|---|"]
    for st in ["overall", "magnets", "tail", "twins", "head"]:
        bm = _macro_f1(y_true[test], base_pred[test], strata[st][test])
        lm = _macro_f1(y_true[test], learn_pred[test], strata[st][test])
        d, lo, hi = _bootstrap_delta(y_true[test], base_pred[test], learn_pred[test], strata[st][test])
        lines.append(f"| {st} | {bm:.4f} | {lm:.4f} | {d:+.4f} | [{lo:+.4f}, {hi:+.4f}] |")
    acc_b = (base_pred[test] == y_true[test]).mean()
    acc_l = (learn_pred[test] == y_true[test]).mean()
    lines.append(f"\nOverall accuracy: {acc_b:.4f} -> {acc_l:.4f} ({acc_l-acc_b:+.4f}).")
    np.save(os.path.join(out_dir, "tables", "learned_bias.npy"), b)

    md = pd.DataFrame(rows).to_markdown(index=False) + "\n\n" + "\n".join(lines)
    out_path = os.path.join(out_dir, "tables", "learned_prior.md")
    with open(out_path, "w") as f:
        f.write(md)
    print("\n" + "\n".join(lines))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    run()
