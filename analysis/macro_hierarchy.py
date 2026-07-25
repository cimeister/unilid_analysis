"""Macrolanguage-hierarchical decision rule (plan item 13, added 2026-07-18).

40% of val false-positive mass sits on head-level dialect sinks inside macrolanguage
clusters (ars, glk, arz; diagnostic analysis 2026-07-17), and ~20% of the Exp 10 error
budget is arbitrary macrolanguage splits. Treat the variety within a macrolanguage as a
latent variable: the score of a group is the marginal over its members,
    score(G) = logsumexp_{L in G} score(L)   (uniform within-group prior),
the argmax group wins, and the prediction is the best-scoring member within it.
Grouping is the SIL macrolanguage table (member iso -> macro iso, the mapping already
used by the CommonLID evaluation); languages outside any macrolanguage are singleton
groups, for which the rule reduces to the plain argmax. Over the 1,940 languages: 83
multi-member groups covering 289 languages (ara 11, msa 12, zho 8, que 27, zap 30).

Parameter-free (no fitting, no tuning); the guard is still applied as the accept/reject
criterion. Scores come from the top-K candidates per example
(`top_k_of_cached_weight_sets_batch`); members outside the top-K contribute
exp(score - max) < e^-gap to the marginal and are negligible at K=50. Reported
alongside the primary stratified macro-F1: macro-aware accuracy (prediction counts as
correct if it maps to the true label's macrolanguage), which quantifies how much
remaining error is within-macro confusion.
"""
from __future__ import annotations

import os
from collections import defaultdict

import numpy as np

from analysis.config import DEFAULT_SAMPLE_SIZE
from analysis.transfer_sweep import (
    _load_model_data, _load_train_counts, _load_unilid_model, _stream_sampled_texts,
)
from analysis.sample_data import load_sample
from analysis.commonlid_eval import _load_macro_member_to_macro
from analysis.hierarchical_pool import (
    _strata, _macro_f1, _bootstrap_delta, passes_guard, GUARD_STRATA, GUARD_TOL,
    VAL_MASK, DIAG_CSV,
)

TOPK = 50          # candidate depth for the group marginal (top-20 true-label recall is
                   # already 0.9971; 50 leaves the marginal's truncation error negligible)
OUT_DIR = "outputs"


def _group_ids(langs: list) -> np.ndarray:
    """Integer group id per language index: SIL macrolanguage of the iso part, else the
    iso itself (singleton). Script variants of the same iso share a group."""
    m2M = _load_macro_member_to_macro()
    keys = [m2M.get(l.rsplit("_", 1)[0], l.rsplit("_", 1)[0]) for l in langs]
    uniq = {k: g for g, k in enumerate(sorted(set(keys)))}
    return np.array([uniq[k] for k in keys], dtype=np.int32)


def hierarchical_predict(topk_lists, langs, gid) -> tuple[list, list]:
    """(flat argmax prediction, hierarchical prediction) per example, both as language
    strings ('' for examples with no candidates, scored as wrong)."""
    flat, hier = [], []
    for cands in topk_lists:
        if not cands:
            flat.append("")
            hier.append("")
            continue
        best_idx, best_sc = max(cands, key=lambda c: c[1])
        flat.append(langs[best_idx])
        by_group = defaultdict(list)
        for idx, sc in cands:
            by_group[int(gid[idx])].append((idx, sc))
        best_g, best_gs = None, -np.inf
        for g, members in by_group.items():
            scores = np.array([s for _i, s in members], dtype=np.float64)
            m = scores.max()
            gs = m + np.log(np.exp(scores - m).sum())
            if gs > best_gs:
                best_gs, best_g = gs, g
        widx, _ = max(by_group[best_g], key=lambda c: c[1])
        hier.append(langs[widx])
    return flat, hier


def run(sample_size: int = DEFAULT_SAMPLE_SIZE, out_dir: str = OUT_DIR):
    import pandas as pd
    os.makedirs(os.path.join(out_dir, "tables"), exist_ok=True)
    weights, langs, lang_to_idx = _load_model_data()
    del weights
    train_counts = _load_train_counts()
    N = np.array([train_counts.get(l, 0) for l in langs], dtype=np.float64)
    diag = pd.read_csv(DIAG_CSV)
    gid = _group_ids(langs)
    n_groups = len(set(gid.tolist()))
    multi = sum(1 for g, c in zip(*np.unique(gid, return_counts=True)) if c > 1)
    print(f"{n_groups} groups; {multi} multi-member")

    texts = _stream_sampled_texts(sample_size)
    data = load_sample(sample_size)
    y_true = np.array(data["y_true"])
    if len(texts) != len(y_true):
        raise RuntimeError(f"text/y_true mismatch {len(texts)} vs {len(y_true)}")
    val = np.load(VAL_MASK)
    test = ~val
    strata = _strata(y_true, langs, lang_to_idx, N, diag)

    model = _load_unilid_model()
    print("Preprocessing + top-k extraction...")
    pre = [model.preprocess(t) for t in texts]
    nonempty = [p if p else "" for p in pre]
    # top_k on an empty string returns an empty candidate list (scored as wrong),
    # matching the empty-text handling of the sweep scripts
    topk = model.model.top_k_of_cached_weight_sets_batch(nonempty, TOPK)
    if len(topk) != len(texts):
        raise RuntimeError(f"top-k returned {len(topk)} results for {len(texts)} inputs")

    flat, hier = hierarchical_predict(topk, langs, gid)
    base_pred = np.array(flat, dtype=object)
    hier_pred = np.array(hier, dtype=object)
    agree = (base_pred == np.array(data["pred_UniLID"])).mean()
    print(f"  top-1 agreement with recorded UniLID preds: {agree:.4f}")
    if agree < 0.99:
        raise RuntimeError(f"top-1 agreement {agree:.4f} < 0.99; model/scorer changed")

    def strat_row(pred, mask_split):
        return {st: _macro_f1(y_true[mask_split], pred[mask_split], strata[st][mask_split])
                for st in strata}

    val_base = strat_row(base_pred, val)
    val_hier = strat_row(hier_pred, val)
    rows = [{"config": "baseline", **{f"val_{k}": v for k, v in val_base.items()}},
            {"config": "macro_marginal", **{f"val_{k}": v for k, v in val_hier.items()}}]
    accepted = passes_guard(rows[1], rows[0], prefix="val_")
    print(f"\nGuard: {'PASS' if accepted else 'FAIL'} "
          f"(single parameter-free config; guard is accept/reject only)")

    # macro-aware accuracy: prediction correct if it maps to the true label's macro
    m2M = _load_macro_member_to_macro()

    def macro_aware_acc(pred, mask):
        yt = y_true[mask]
        yp = pred[mask]
        ok = 0
        for t, p in zip(yt, yp):
            ti, pi = t.rsplit("_", 1)[0], p.rsplit("_", 1)[0] if p else ""
            ok += (ti == pi) or (m2M.get(ti, ti) == m2M.get(pi, pi))
        return ok / len(yt)

    lines = ["# Macrolanguage-hierarchical decision — TEST evaluation\n",
             f"score(G) = logsumexp over members (top-{TOPK} candidates); prediction = "
             f"best member of the argmax group. Parameter-free (no tuning), so the table "
             f"reports the hierarchical-vs-baseline deltas unconditionally; the guard "
             f"verdict on val ({'PASS' if accepted else 'FAIL'}) is the accept/reject "
             f"criterion for adopting the rule. Top-1 agreement {agree:.4f}.",
             f"Selection guard: val overall must improve and no stratum "
             f"({'/'.join(GUARD_STRATA)}) may drop more than {GUARD_TOL} vs baseline.\n",
             "| stratum | base macroF1 | hierarchical macroF1 | delta | 95% CI |",
             "|---|---|---|---|---|"]
    for st in ["overall", "magnets", "tail", "twins", "head"]:
        b = _macro_f1(y_true[test], base_pred[test], strata[st][test])
        nw = _macro_f1(y_true[test], hier_pred[test], strata[st][test])
        d, lo, hi = _bootstrap_delta(y_true[test], base_pred[test], hier_pred[test],
                                     strata[st][test])
        lines.append(f"| {st} | {b:.4f} | {nw:.4f} | {d:+.4f} | [{lo:+.4f}, {hi:+.4f}] |")
    acc_b = (base_pred[test] == y_true[test]).mean()
    acc_h = (hier_pred[test] == y_true[test]).mean()
    lines.append(f"\nOverall exact accuracy: {acc_b:.4f} -> {acc_h:.4f} "
                 f"({acc_h-acc_b:+.4f}).")
    lines.append(f"Macro-aware accuracy (test half): baseline "
                 f"{macro_aware_acc(base_pred, test):.4f} -> hierarchical "
                 f"{macro_aware_acc(hier_pred, test):.4f}.")

    md = pd.DataFrame(rows).round(4).to_markdown(index=False) + "\n\n" + "\n".join(lines)
    out_path = os.path.join(out_dir, "tables", "macro_hierarchy.md")
    with open(out_path, "w") as f:
        f.write(md)
    print("\n" + "\n".join(lines))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    run()
