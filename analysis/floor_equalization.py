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
# The same grid on the corrected scale. The clamp sets an ABSOLUTE target in log
# space, and the special-token correction raised every real token by
# log 5 = 1.6094 nats, so asking the published question of a corrected model
# means sweeping the published grid shifted by the same amount. Sweeping the
# unshifted grid would ask a different question: at -21 a corrected row's unseen
# tokens are pushed 1.609 nats further below its seen ones than the released
# model's were.
FLOORS_CORRECTED = [f + float(np.log(5.0)) for f in FLOORS]
OUT_DIR = "outputs"

# The per-row probability of each of the four special tokens in a matrix trained
# before UNILID 0.3.0. It is the signature of the special-token defect, not a
# property of the model: the trainer read HuggingFace's score-0 specials as
# probability 1.0, so four of them normalized to exactly 1/5 each and left the
# real tokens 0.2 of the mass.
#
# This is NOT how special columns are located any more. Detection by probability
# only worked while that probability was 0.2, and it silently finds nothing on a
# corrected model; use _special_columns() below, which reads the vocabulary.
# SPECIAL_P survives solely as the assertion value for the Good-Turing family
# (analysis/gt_counts.py, full_test_gt.py, full_test_bgfloor.py), whose arithmetic
# is derived from the 0.2 budget throughout. Those scripts abort when the
# assertion fails, which is the behaviour wanted: their derivation does not hold
# for a corrected matrix and must be redone before they can be run against one.
SPECIAL_P = 0.2


def build_equalized_weights(W: np.ndarray, target: float,
                            special_idx=()) -> tuple[np.ndarray, int]:
    """Clamp each row's exact floor plateau to min(floor, target). Returns the new
    matrix and the number of modified languages.

    ``special_idx`` names columns to leave out of the minimum, and is required
    for matrices produced by UNILID 0.3.0 or later, which park the special tokens
    at the training floor below every real token. Without it the row minimum is
    the special tokens, the unseen-token plateau is never found, and this
    function reports zero modified languages while doing nothing. Matrices
    written earlier hold their special tokens near the top of the row, so naming
    them changes nothing for those."""
    out = np.array(W, dtype=np.float32)
    real = np.ones(W.shape[1], dtype=bool)
    real[list(special_idx)] = False
    if not real.any():
        raise RuntimeError("every column was named special; nothing to clamp")
    real_cols = np.flatnonzero(real)
    n_mod = 0
    for i in range(W.shape[0]):
        row = W[i]
        floor = row[real_cols].min()
        if floor > target:
            out[i, real_cols[row[real_cols] == floor]] = np.float32(target)
            n_mod += 1
    if not np.isfinite(out).all():
        raise RuntimeError("non-finite weights after floor equalization")
    return out, n_mod


def _special_columns(model_path: str) -> list[int]:
    """Special-token column indices, read from the model's own vocabulary.

    Detecting them by their probability instead only worked while that
    probability was the 0.2 the defect produced. A corrected model, or anything
    trained by 0.3.0, parks them at the training floor.
    """
    import json as _json
    import sys as _sys
    unilid_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "UNILID")
    if unilid_dir not in _sys.path:
        _sys.path.insert(0, unilid_dir)
    from unilid.constants import SPECIAL_TOKENS
    from unilid.model_io import load_unilid_raw

    tok_json, weights, _langs = load_unilid_raw(model_path)
    del weights
    text = tok_json if isinstance(tok_json, str) else tok_json.decode("utf-8")
    vocab = [t for t, _ in _json.loads(text)["model"]["vocab"]]
    return [vocab.index(t) for t in SPECIAL_TOKENS.values() if t in vocab]


def run(sample_size: int = DEFAULT_SAMPLE_SIZE, out_dir: str = OUT_DIR,
        model_path: str = None, floor_grid: list = None):
    import pandas as pd
    from analysis.transfer_sweep import UNILID_MODEL_PATH

    model_path = model_path or UNILID_MODEL_PATH
    floor_grid = list(floor_grid) if floor_grid else list(FLOORS)
    print(f"model {model_path}\nsweeping c over {floor_grid}", flush=True)
    os.makedirs(os.path.join(out_dir, "tables"), exist_ok=True)
    weights, langs, lang_to_idx = _load_model_data(model_path)
    train_counts = _load_train_counts()
    N = np.array([train_counts.get(l, 0) for l in langs], dtype=np.float64)
    W = np.array(weights, dtype=np.float32)
    diag = pd.read_csv(DIAG_CSV)

    special_cols = np.array(_special_columns(model_path), dtype=np.int64)
    if len(special_cols) != 4:
        raise RuntimeError(f"expected 4 special columns from the vocabulary, "
                           f"found {len(special_cols)}")
    real_cols = np.setdiff1d(np.arange(W.shape[1]), special_cols)
    floors = W[:, real_cols].min(axis=1)
    corr = float(np.corrcoef(floors, np.log10(N + 1.0))[0, 1])
    print(f"row floors: min={floors.min():.3f} max={floors.max():.3f}; "
          f"corr(floor, log10 N) = {corr:.3f}")
    if corr > -0.5:
        raise RuntimeError(f"floor-resource correlation {corr:.3f} is not strongly "
                           f"negative; the premise of this experiment does not hold")

    texts = _stream_sampled_texts(sample_size)
    data = load_sample(sample_size)
    y_true = np.array(data["y_true"])
    if len(texts) != len(y_true):
        raise RuntimeError(f"text/y_true mismatch {len(texts)} vs {len(y_true)}")
    val = np.load(VAL_MASK)
    test = ~val
    strata = _strata(y_true, langs, lang_to_idx, N, diag)

    model = _load_unilid_model(model_path)
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
    for F in floor_grid:
        cfg = f"floor{F:g}"
        print(f"Config {cfg}...")
        w_new, n_mod = build_equalized_weights(W, F, special_cols)
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


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", dest="model_path", default=None)
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    ap.add_argument("--floors", default=None,
                    help="comma-separated grid for c. Default: the published "
                         "grid. Use --corrected-grid for the same grid shifted "
                         "by log 5, which is the published question asked of a "
                         "special-token-corrected model.")
    ap.add_argument("--corrected-grid", action="store_true")
    a = ap.parse_args(argv)
    if a.floors and a.corrected_grid:
        raise SystemExit("pass --floors or --corrected-grid, not both")
    grid = ([float(x) for x in a.floors.split(",")] if a.floors
            else FLOORS_CORRECTED if a.corrected_grid else None)
    run(sample_size=a.sample_size, out_dir=a.out_dir,
        model_path=a.model_path, floor_grid=grid)


if __name__ == "__main__":
    main()
