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
# Prior-centered regularization (plan item 3, added 2026-07-18): penalize
# ||b - gamma*log(N+1)||^2 so languages with few val examples shrink toward the
# frequency-prior default instead of toward 0, and a newly added language gets the
# starting bias gamma*log(N_new). gamma=0.0 reproduces the plain L2 of Exp 14.
# Grid recorded in EXPERIMENTAL_SETUP.md. Caution from Exp 16: the anchor itself
# (pure frequency prior) costs tail -0.0182 on the full test set, so large gamma is
# not presumed safe; the guard decides.
PRIOR_GAMMAS = [0.0, 0.25, 0.5]
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


def _make_loss(cand_idx, cand_score, seg, true_flat_pos, n_seg, n_lang, reg, c0):
    """Regularized softmax NLL; the L2 penalty is centered on c0 (zeros = plain L2).

    The loss conditions on the true label being inside the example's top-k (examples
    where it is not have no defined candidate likelihood), so the softmax soft counts
    in the gradient must be restricted to those same examples. The original Exp 14
    code accumulated +P(L|x) over ALL examples' candidates, which is not the gradient
    of this loss whenever top-k recall < 1 (found by review 2026-07-18, verified by
    finite differences; recall here is 0.9971)."""
    present = true_flat_pos >= 0
    true_pos = true_flat_pos[present]
    cand_present = present[seg]           # candidates of examples with the true label in top-k

    def loss_grad(b):
        logit = cand_score + b[cand_idx]
        segmax = np.full(n_seg, -np.inf)
        np.maximum.at(segmax, seg, logit)
        ex = np.exp(logit - segmax[seg])
        segsum = np.zeros(n_seg)
        np.add.at(segsum, seg, ex)
        p = ex / segsum[seg]                                  # softmax prob per candidate
        d = b - c0
        # loss = -sum log p[true] over present examples + reg||b - c0||^2
        loss = -np.sum(np.log(np.maximum(p[true_pos], 1e-300))) + reg * np.dot(d, d)
        grad = np.zeros(n_lang)
        np.add.at(grad, cand_idx[cand_present], p[cand_present])   # +P(L|x), present only
        np.add.at(grad, cand_idx[true_pos], -1.0)             # -1 for the true candidate
        grad += 2.0 * reg * d
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
    best = {"reg": None, "gamma": None, "b": None, "val_overall": -1}
    rows = [{"gamma": None, "reg": 0.0, "val_recall@k": round(recall_at_k, 4),
             **{f"val_{k}": round(v, 4) for k, v in val_base.items()}}]
    for gamma in PRIOR_GAMMAS:
        c0 = gamma * np.log(N + 1.0)
        for reg in REGS:
            lg = _make_loss(ci, cs, seg, tfp, n_seg, n_lang, reg, c0)
            res = minimize(lg, np.zeros(n_lang), jac=True, method="L-BFGS-B",
                           options={"maxiter": 200, "maxfun": 300})
            b = res.x.astype(np.float64)
            mvr = val_strata_for_b(b)
            rows.append({"gamma": gamma, "reg": reg, "val_recall@k": round(recall_at_k, 4),
                         **{f"val_{k}": round(v, 4) for k, v in mvr.items()}})
            if passes_guard(mvr, val_base) and mvr["overall"] > best["val_overall"]:
                best = {"reg": reg, "gamma": gamma, "b": b, "val_overall": mvr["overall"]}
            print(f"  gamma={gamma} reg={reg}: val " +
                  " ".join(f"{k}={mvr[k]:.4f}" for k in ("overall", *GUARD_STRATA)))

    if best["b"] is None:
        # no (gamma, reg) is eligible: select the baseline (b = 0) and report the negative
        # result, matching prior_sweep's gamma=0 behaviour (no silent fallback to a fitted b)
        print("No (gamma, reg) passed the guard; selecting the baseline (b = 0).")
        best = {"reg": None, "gamma": None, "b": np.zeros(n_lang),
                "val_overall": val_base["overall"]}

    b = best["b"].astype(np.float32)
    print(f"\nSelected gamma={best['gamma']} reg={best['reg']}. "
          f"Scoring test with the biased scorer...")
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

    lines = ["# Learned per-language bias, prior-centered regularizer — TEST evaluation\n",
             f"Regularizer: reg*||b - gamma*log(N+1)||^2; gamma=0 is the plain L2 of Exp 14.",
             f"Top-{TOPK} val recall of true label: {recall_at_k:.4f}. Selected "
             f"gamma={best['gamma']} reg={best['reg']}"
             " (None/None means nothing passed the guard; baseline b=0 evaluated).",
             f"Selection guard: val overall must improve and no stratum "
             f"({'/'.join(GUARD_STRATA)}) may drop more than {GUARD_TOL} vs baseline.",
             "Reference points: plain-L2 reg=5.0 gave test overall +0.0112 (Exp 14 revised); "
             "full-test numbers for it are in Exp 16.\n",
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
    # tagged filenames: the Exp 14 artifacts (learned_prior.md, learned_bias.npy) stay
    # the record for the plain-L2 result; downstream scripts read learned_bias.npy
    np.save(os.path.join(out_dir, "tables", "learned_bias_centered.npy"), b)

    md = pd.DataFrame(rows).to_markdown(index=False) + "\n\n" + "\n".join(lines)
    out_path = os.path.join(out_dir, "tables", "learned_prior_centered.md")
    with open(out_path, "w") as f:
        f.write(md)
    print("\n" + "\n".join(lines))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    run()
