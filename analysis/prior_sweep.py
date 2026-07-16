"""Per-language prior / calibration offset, the redirect after Stage 1's negative result.

Adds a per-language bias b_L to each language's summed score before the argmax:
    pred = argmax_L [ sum_t log p_L(token_t) + b_L ]
implemented with the Rust `best_of_cached_weight_sets_biased_batch` (one constant per
language, NOT a per-token edit). The constant matters most for SHORT texts (few tokens,
small |score|), which is exactly the weak-evidence regime where the rare-attractor problem
is worst, and it is selective: a magnet still wins on its own text (large likelihood
margin) but loses the marginal cases it was stealing.

Prior family swept here: b_L = gamma * log(N_L + 1)  (a frequency prior P(L) ~ N_L^gamma;
gamma = 0 is the no-op baseline). Tuned on the val half, scored once on the test half.
Modular: b_L is one scalar per language, so adding a language is unchanged.

Needs the custom Rust build (biased scorer) + ~250 GB to cache all 1,940 weight sets -> SLURM.
"""
from __future__ import annotations

import os

import numpy as np

from analysis.config import DEFAULT_SAMPLE_SIZE
from analysis.transfer_sweep import (
    _load_model_data, _load_train_counts, _load_unilid_model, _stream_sampled_texts,
)
from analysis.sample_data import load_sample
from analysis.hierarchical_pool import (
    _strata, _macro_f1, _bootstrap_delta, passes_guard, GUARD_STRATA, GUARD_TOL,
    VAL_MASK, DIAG_CSV,
)

GAMMAS = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0]
OUT_DIR = "outputs"


def run(sample_size: int = DEFAULT_SAMPLE_SIZE, out_dir: str = OUT_DIR,
        model_path: str | None = None, tag: str = ""):
    import pandas as pd
    os.makedirs(os.path.join(out_dir, "tables"), exist_ok=True)
    mp_args = {} if model_path is None else {"model_path": model_path}
    weights, langs, lang_to_idx = _load_model_data(**mp_args)
    train_counts = _load_train_counts()
    N = np.array([train_counts.get(l, 0) for l in langs], dtype=np.float64)
    logN = np.log(N + 1.0)
    diag = pd.read_csv(DIAG_CSV)

    texts = _stream_sampled_texts(sample_size)
    data = load_sample(sample_size)
    y_true = np.array(data["y_true"])
    if len(texts) != len(y_true):
        raise RuntimeError(f"text/y_true mismatch {len(texts)} vs {len(y_true)}")
    val = np.load(VAL_MASK); test = ~val
    strata = _strata(y_true, langs, lang_to_idx, N, diag)

    model = _load_unilid_model(**mp_args)
    # preprocess once (shared across all gammas); keep mapping back to original positions
    print("Preprocessing texts...")
    pre, valid_idx = [], []
    for i, t in enumerate(texts):
        p = model.preprocess(t)
        if p:
            pre.append(p); valid_idx.append(i)
    valid_idx = np.array(valid_idx)

    def predict_with_bias(gamma):
        biases = (gamma * logN).astype(np.float32).tolist()
        batch = model.model.best_of_cached_weight_sets_biased_batch(pre, biases)
        pred = np.array([""] * len(texts), dtype=object)
        for j, (idx, _t, _s) in zip(valid_idx, batch):
            pred[j] = langs[idx]
        return pred

    def strat_row(pred, mask_split):
        return {st: _macro_f1(y_true[mask_split], pred[mask_split], strata[st][mask_split])
                for st in strata}

    rows, preds_by_g, vr_by_g = [], {}, {}
    val_base = None
    for g in GAMMAS:
        print(f"gamma={g} ...")
        pred = predict_with_bias(g)
        preds_by_g[g] = pred
        vr = strat_row(pred, val)
        vr_by_g[g] = vr
        if g == 0.0:
            val_base = vr
            # sanity: gamma=0 must reproduce the unbiased baseline predictions
            agree = (pred[valid_idx] == np.array(data["pred_UniLID"])[valid_idx]).mean()
            print(f"  gamma=0 agreement with recorded UniLID preds: {agree:.4f}")
        rows.append({"gamma": g, **{f"val_{k}": round(v, 4) for k, v in vr.items()}})
        print("  val:", {k: round(v, 4) for k, v in vr.items()})

    # select best gamma on val overall macro-F1, guarded on all strata (passes_guard);
    # guard and argmax use the UNROUNDED values (rows are rounded for display only)
    eligible = [g for g in GAMMAS if g != 0.0 and passes_guard(vr_by_g[g], val_base)]
    best_g = max(eligible, key=lambda g: vr_by_g[g]["overall"]) if eligible else 0.0
    print(f"\nBest gamma (val, guarded): {best_g}")

    base_pred, best_pred = preds_by_g[0.0], preds_by_g[best_g]
    lines = ["# Per-language frequency prior (b_L = gamma*log N_L) — TEST evaluation\n",
             f"Best gamma on val: **{best_g}** (gamma=0 means no eligible operating point).",
             f"Selection guard: val overall must improve and no stratum "
             f"({'/'.join(GUARD_STRATA)}) may drop more than {GUARD_TOL} vs gamma=0.\n",
             "| stratum | base macroF1 | prior macroF1 | delta | 95% CI |",
             "|---|---|---|---|---|"]
    for st in ["overall", "magnets", "tail", "twins", "head"]:
        b = _macro_f1(y_true[test], base_pred[test], strata[st][test])
        nw = _macro_f1(y_true[test], best_pred[test], strata[st][test])
        d, lo, hi = _bootstrap_delta(y_true[test], base_pred[test], best_pred[test], strata[st][test])
        lines.append(f"| {st} | {b:.4f} | {nw:.4f} | {d:+.4f} | [{lo:+.4f}, {hi:+.4f}] |")
    # also report accuracy delta (overall, test)
    acc_b = (base_pred[test] == y_true[test]).mean()
    acc_n = (best_pred[test] == y_true[test]).mean()
    lines.append(f"\nOverall accuracy: {acc_b:.4f} -> {acc_n:.4f} ({acc_n-acc_b:+.4f}).")

    df = pd.DataFrame(rows)
    md = df.to_markdown(index=False) + "\n\n" + "\n".join(lines)
    out_path = os.path.join(out_dir, "tables", f"prior_sweep{tag}.md")
    with open(out_path, "w") as f:
        f.write(md)
    print("\n" + "\n".join(lines))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    run()
