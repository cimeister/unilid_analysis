"""Downward floor equalization (plan item 14, added 2026-07-18).

Every mass-adding intervention failed in the same direction (Exp 9/13/18/19): giving
unseen tokens more probability increases theft. Exp 10's diagnosis points the opposite
way: the per-language unseen-token floor is resource-tied (an exact plateau over
74,617-99,810 of 100k entries per row; levels -13.2 to -19.9, corr(floor, log10 N)
= -0.966), so low-resource languages under-penalize unseen material relative to the
head. This experiment equalizes the penalty DOWNWARD: each language's floor plateau is
clamped to min(floor_L, F) for one global constant F, so weak floors (small N) are
lowered toward or past head levels and no entry is ever raised. Exp 6 is the mirror
image (it raised sub-floor entries up to a shared value) and was negative; the downward
direction is untested.

Observed-token estimates and special tokens are bit-identical (only exact-minimum
plateau entries change, and only downward; specials sit at the row maximum). No
renormalization (the scorer sums unnormalized log-weights). Modular: one shared global
constant, no per-language fitting of any kind. F is swept; selection on the val half
via the all-strata guard; test half scored once.
"""
from __future__ import annotations

import gc
import os

import numpy as np

from analysis.config import DEFAULT_SAMPLE_SIZE
from analysis.transfer_sweep import (
    _load_model_data, _load_train_counts, _load_unilid_model, _stream_sampled_texts,
    predict_all,
)
from analysis.sample_data import load_sample
from analysis.hierarchical_pool import (
    _strata, _macro_f1, _bootstrap_delta, passes_guard, GUARD_STRATA, GUARD_TOL,
    VAL_MASK, DIAG_CSV,
)

# Global target floors (log-prob). Row minimums span -19.9 to -13.2; -17/-19 lower the
# low-resource floors toward head levels, -21/-23 probe over-penalization past them.
FLOORS = [-17.0, -19.0, -21.0, -23.0]
SPECIAL_P = 0.2
OUT_DIR = "outputs"


def build_equalized_weights(W: np.ndarray, target: float) -> tuple[np.ndarray, int]:
    """Clamp each row's exact floor plateau to min(floor, target). Returns the new
    matrix and the number of modified languages."""
    out = np.array(W, dtype=np.float32)
    n_mod = 0
    for i in range(W.shape[0]):
        row = W[i]
        floor = row.min()
        if floor > target:
            out[i, row == floor] = np.float32(target)
            n_mod += 1
    if not np.isfinite(out).all():
        raise RuntimeError("non-finite weights after floor equalization")
    return out, n_mod


def run(sample_size: int = DEFAULT_SAMPLE_SIZE, out_dir: str = OUT_DIR):
    import pandas as pd
    os.makedirs(os.path.join(out_dir, "tables"), exist_ok=True)
    weights, langs, lang_to_idx = _load_model_data()
    train_counts = _load_train_counts()
    N = np.array([train_counts.get(l, 0) for l in langs], dtype=np.float64)
    W = np.array(weights, dtype=np.float32)
    diag = pd.read_csv(DIAG_CSV)

    floors = W.min(axis=1)
    corr = float(np.corrcoef(floors, np.log10(N + 1.0))[0, 1])
    print(f"row floors: min={floors.min():.3f} max={floors.max():.3f}; "
          f"corr(floor, log10 N) = {corr:.3f}")
    if corr > -0.5:
        raise RuntimeError(f"floor-resource correlation {corr:.3f} is not strongly "
                           f"negative; the premise of this experiment does not hold")
    special_cols = np.where(np.all(np.abs(np.exp(W.astype(np.float64))
                                          - SPECIAL_P) < 1e-4, axis=0))[0]
    if len(special_cols) != 4:
        raise RuntimeError(f"expected 4 special columns at p={SPECIAL_P}, "
                           f"found {len(special_cols)}")

    texts = _stream_sampled_texts(sample_size)
    data = load_sample(sample_size)
    y_true = np.array(data["y_true"])
    if len(texts) != len(y_true):
        raise RuntimeError(f"text/y_true mismatch {len(texts)} vs {len(y_true)}")
    val = np.load(VAL_MASK)
    test = ~val
    strata = _strata(y_true, langs, lang_to_idx, N, diag)

    model = _load_unilid_model()
    print("Baseline prediction...")
    base_pred = np.array(predict_all(texts, model))
    agree = (base_pred == np.array(data["pred_UniLID"])).mean()
    print(f"  baseline agreement with recorded UniLID preds: {agree:.4f}")
    if agree < 0.99:
        raise RuntimeError(f"baseline agreement {agree:.4f} < 0.99; model/scorer changed")

    def strat_row(pred, mask_split):
        return {st: _macro_f1(y_true[mask_split], pred[mask_split], strata[st][mask_split])
                for st in strata}

    val_base = strat_row(base_pred, val)
    rows = [{"config": "baseline", "n_modified": 0,
             **{f"val_{k}": v for k, v in val_base.items()}}]
    preds_by_cfg = {"baseline": base_pred}
    for F in FLOORS:
        cfg = f"floor{F:g}"
        print(f"Config {cfg}...")
        w_new, n_mod = build_equalized_weights(W, F)
        if not np.array_equal(w_new[:, special_cols], W[:, special_cols]):
            raise RuntimeError("special-token columns were modified")
        model.model.set_weight_sets(w_new.tolist())
        del w_new
        gc.collect()
        pred = np.array(predict_all(texts, model))
        preds_by_cfg[cfg] = pred
        vr = strat_row(pred, val)
        rows.append({"config": cfg, "n_modified": n_mod,
                     **{f"val_{k}": v for k, v in vr.items()}})
        print(f"    modified {n_mod} languages; val macro-F1:",
              {k: round(v, 4) for k, v in vr.items()})

    eligible = [r for r in rows if r["config"] != "baseline"
                and passes_guard(r, rows[0], prefix="val_")]
    best = max(eligible, key=lambda r: r["val_overall"])["config"] if eligible else "baseline"
    print(f"\nBest config (val, guarded): {best}")

    best_pred = preds_by_cfg[best]
    lines = ["# Downward floor equalization — TEST evaluation\n",
             f"Best config selected on val: **{best}** (baseline means nothing passed the "
             f"guard). Baseline agreement {agree:.4f}.",
             f"Floor plateaus clamped to min(floor_L, F); observed tokens and specials "
             f"bit-identical; no entry is ever raised.",
             f"Selection guard: val overall must improve and no stratum "
             f"({'/'.join(GUARD_STRATA)}) may drop more than {GUARD_TOL} vs baseline.\n",
             "| stratum | base macroF1 | equalized macroF1 | delta | 95% CI |",
             "|---|---|---|---|---|"]
    for st in ["overall", "magnets", "tail", "twins", "head"]:
        b = _macro_f1(y_true[test], base_pred[test], strata[st][test])
        nw = _macro_f1(y_true[test], best_pred[test], strata[st][test])
        d, lo, hi = _bootstrap_delta(y_true[test], base_pred[test], best_pred[test],
                                     strata[st][test])
        lines.append(f"| {st} | {b:.4f} | {nw:.4f} | {d:+.4f} | [{lo:+.4f}, {hi:+.4f}] |")
    acc_b = (base_pred[test] == y_true[test]).mean()
    acc_n = (best_pred[test] == y_true[test]).mean()
    lines.append(f"\nOverall accuracy: {acc_b:.4f} -> {acc_n:.4f} ({acc_n-acc_b:+.4f}).")

    md = pd.DataFrame(rows).round(4).to_markdown(index=False) + "\n\n" + "\n".join(lines)
    out_path = os.path.join(out_dir, "tables", "floor_equalization.md")
    with open(out_path, "w") as f:
        f.write(md)
    print("\n" + "\n".join(lines))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    run()
