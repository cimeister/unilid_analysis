"""How far do the calibration's fixed numbers move under the correction?

The correction adds log 5 = 1.6094 nats to every real token of every row. The
constants that live in nats therefore cannot be carried over unexamined: the
unseen-token constant c is an absolute target in log space, and the proximity
bound is a score difference. This probes how far the optimum moves before
committing to a full re-derivation, on a subsample rather than the full pool.

Selection is done on the VALIDATION half of the seed-42 500k draw, never the
test half, matching the protocol the paper states.

  python -m analysis.probe_calibration_shift -o outputs/rerelease/probe_c.json
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "UNILID"))

from analysis.config import DEFAULT_SAMPLE_SIZE  # noqa: E402
from analysis.floor_equalization import build_equalized_weights  # noqa: E402
from analysis.hierarchical_pool import VAL_MASK  # noqa: E402
from analysis.sample_data import load_sample  # noqa: E402
from analysis.transfer_sweep import (  # noqa: E402
    UNILID_MODEL_PATH, _stream_sampled_texts,
)

CORRECTED = ("/capstor/scratch/cscs/cmeister747/unilid_analysis/corrected/"
             "glotlidc_corrected.unilid")
TEXT_CACHE = ("/capstor/scratch/cscs/cmeister747/unilid_analysis/"
              "probe_sample_texts.pkl")
# Enough to separate configurations that differ by a few tenths of a point,
# small enough that a two-model, eight-value sweep finishes in minutes.
N_PROBE = 60_000
PROBE_SEED = 20260817
# Centred on the released model's c = -21 and wide enough to contain both that
# optimum and one shifted by +1.609.
C_GRID = [-27.0, -25.0, -23.0, -21.0, -19.5, -19.0, -17.5, -16.0, -14.0]
BATCH = 20_000


def _texts_for_sample() -> list[str]:
    if os.path.exists(TEXT_CACHE):
        with open(TEXT_CACHE, "rb") as f:
            return pickle.load(f)
    texts = _stream_sampled_texts(DEFAULT_SAMPLE_SIZE)
    with open(TEXT_CACHE, "wb") as f:
        pickle.dump(texts, f, protocol=4)
    return texts


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    f1s = []
    for lab in np.unique(y_true):
        tp = int(((y_true == lab) & (y_pred == lab)).sum())
        fp = int(((y_true != lab) & (y_pred == lab)).sum())
        fn = int(((y_true == lab) & (y_pred != lab)).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return float(np.mean(f1s))


def predict(model, texts):
    out = []
    for start in range(0, len(texts), BATCH):
        out.extend(lang for lang, _t, _s in
                   model.predict_batch(texts[start:start + BATCH]))
    return np.array(out, dtype=object)


def sweep(model_path: str, texts, y_true, tag: str) -> list[dict]:
    import json as _json

    from unilid.constants import SPECIAL_TOKENS
    from unilid.model_io import UnilidModel, load_unilid_raw

    tok_json, weights, _langs = load_unilid_raw(model_path)
    tok_text = tok_json if isinstance(tok_json, str) else tok_json.decode("utf-8")
    vocab = [t for t, _ in _json.loads(tok_text)["model"]["vocab"]]
    # Required from 0.3.0: the special tokens sit at the training floor there, so
    # a row minimum over the whole row is them rather than the unseen tokens.
    special_idx = [vocab.index(t) for t in SPECIAL_TOKENS.values() if t in vocab]
    W = np.asarray(weights).astype(np.float32)
    del weights
    model = UnilidModel(model_path, calibrated=False)

    rows = []
    base_pred = predict(model, texts)
    rows.append({"c": None, "n_modified": 0, "macro_f1": macro_f1(y_true, base_pred),
                 "accuracy": float((base_pred == y_true).mean())})
    print(f"  [{tag}] no clamp        macro F1 {rows[0]['macro_f1']:.4f}  "
          f"accuracy {rows[0]['accuracy']:.4f}", flush=True)

    for c in C_GRID:
        w_new, n_mod = build_equalized_weights(W, c, special_idx)
        model.model.set_weight_sets_numpy(w_new)
        del w_new
        gc.collect()
        pred = predict(model, texts)
        row = {"c": c, "n_modified": int(n_mod),
               "macro_f1": macro_f1(y_true, pred),
               "accuracy": float((pred == y_true).mean())}
        rows.append(row)
        print(f"  [{tag}] c={c:>7.2f}  modified {n_mod:>5}  "
              f"macro F1 {row['macro_f1']:.4f}  accuracy {row['accuracy']:.4f}",
              flush=True)

    del model, W
    gc.collect()
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--output", default=None)
    args = ap.parse_args(argv)

    data = load_sample(DEFAULT_SAMPLE_SIZE)
    y_true_all = np.array(data["y_true"], dtype=object)
    val = np.load(VAL_MASK)
    texts_all = _texts_for_sample()
    if len(texts_all) != len(y_true_all):
        raise RuntimeError(f"{len(texts_all)} texts against {len(y_true_all)} labels")

    val_idx = np.flatnonzero(val)
    rng = np.random.default_rng(PROBE_SEED)
    pick = np.sort(rng.choice(val_idx, size=min(N_PROBE, len(val_idx)),
                              replace=False))
    texts = [texts_all[i] for i in pick]
    y_true = y_true_all[pick]
    print(f"probe: {len(texts):,} lines drawn from the {len(val_idx):,}-line "
          f"validation half, {len(set(y_true.tolist())):,} labels", flush=True)

    out = {"n_probe": len(texts), "seed": PROBE_SEED, "c_grid": C_GRID}
    for tag, path in (("released", UNILID_MODEL_PATH), ("corrected", CORRECTED)):
        print(f"\n=== {tag}: {path}", flush=True)
        out[tag] = sweep(path, texts, y_true, tag)

    print("\n=== optimum by macro F1 on the validation subsample ===")
    best = {}
    for tag in ("released", "corrected"):
        clamped = [r for r in out[tag] if r["c"] is not None]
        b = max(clamped, key=lambda r: r["macro_f1"])
        best[tag] = b
        print(f"  {tag:10} c = {b['c']:>7.2f}  macro F1 {b['macro_f1']:.4f}")
    shift = best["corrected"]["c"] - best["released"]["c"]
    print(f"  shift {shift:+.3f} nats against log 5 = 1.609")
    out["best"] = best
    out["shift"] = shift

    if args.output:
        Path(args.output).write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
