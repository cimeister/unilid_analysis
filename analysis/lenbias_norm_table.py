"""The tab:lenbias-norm table: length-normalized scoring, by input length.

The paper's table has three prediction columns per length bin: Original, Raw
rescore, Normalized. It had no reproducible generator. analysis/normalized_predict.py
was extended into an eleven-value alpha sweep, and the sweep's tables do not carry
the Original column or the agreement check that makes the Raw rescore column mean
anything, so the published table cannot be regenerated from it. This produces
exactly that table.

- Raw rescore is alpha = 0, the unnormalized scorer reached through the same
  normalization code path. Its purpose is to validate the implementation: it must
  reproduce the model's own predictions exactly.
- Normalized is alpha = 1, dividing the summed token log-probabilities by the
  segmentation length.
- Original is the model's own plain-scorer predictions. For the released model
  those are pred_UniLID in the seed-42 sample pickle (the source of the published
  table, outputs/tables/normalized_comparison.md). For any other model they come
  from that model's own full-pool run, pred_baseline.npy.

The sample is the GOLDEN SUBSET: the test half of the seed-42 500,000-line draw,
250,000 lines. The published table used all 500,000, but half of those are the
validation lines the full-pool runs exclude, so a corrected model has no
plain-scorer prediction for them and the Original column could not be filled. The
test half is inside the scored pool, which keeps the three columns on one
instrument and keeps the validation half out of a reported number. It is the same
subset both release gates use.

  python -m analysis.lenbias_norm_table -o outputs/tables
  python -m analysis.lenbias_norm_table --model CORRECTED.unilid -o outputs_corrected/tables
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "UNILID"))

from analysis.config import DEFAULT_SAMPLE_SIZE, LENGTH_BINS, LENGTH_LABELS
from analysis.full_test_eval import _sample_line_indices
from analysis.normalized_predict import (UNILID_MODEL_PATH, _load_unilid_model,
                                         _stream_sampled_texts, predict_all)
from analysis.sample_data import load_sample
from analysis.transfer_sweep import _load_model_data

RAW_ALPHA = 0.0
NORM_ALPHA = 1.0
# The Raw rescore column exists to show the normalization code path reproduces
# the plain scorer. Anything below exact agreement means the two paths differ and
# the Normalized column cannot be attributed to normalization alone.
RAW_AGREEMENT_MIN = 1.0


def _by_length(pred: np.ndarray, y_true: np.ndarray, lengths: np.ndarray):
    """Accuracy per length bin, plus the overall figure."""
    out = []
    for i, label in enumerate(LENGTH_LABELS):
        lo, hi = LENGTH_BINS[i], LENGTH_BINS[i + 1]
        m = (lengths >= lo) & (lengths < hi)
        out.append({"bin": label, "n": int(m.sum()),
                    "accuracy": float((pred[m] == y_true[m]).mean())
                    if m.any() else None})
    out.append({"bin": "Overall", "n": int(len(y_true)),
                "accuracy": float((pred == y_true).mean())})
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", dest="model_path", default=UNILID_MODEL_PATH)
    ap.add_argument("-o", "--out-dir", default="outputs/tables")
    ap.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    ap.add_argument("--baseline-pred", default=None,
                    help="pred_baseline.npy from this model's own full-pool run; "
                         "required unless the model is the released one")
    a = ap.parse_args(argv)

    is_default = os.path.abspath(a.model_path) == os.path.abspath(UNILID_MODEL_PATH)
    print(f"model: {a.model_path}", flush=True)

    data = load_sample(a.sample_size)
    y_all = np.array(data["y_true"])
    len_all = np.array(data["text_lengths"])
    texts_all = _stream_sampled_texts(a.sample_size)
    if len(texts_all) != len(y_all):
        raise RuntimeError(f"{len(texts_all)} texts against {len(y_all)} labels")

    # The golden subset: odd positions of the draw are the test half, the same
    # split analysis/full_test_eval.py and the release gates use.
    test_pos = (np.arange(a.sample_size) % 2) == 1
    y_true = y_all[test_pos]
    lengths = len_all[test_pos]
    texts = [t for t, keep in zip(texts_all, test_pos) if keep]
    del texts_all
    abs_idx = _sample_line_indices()[test_pos]
    print(f"golden subset: {len(texts):,} lines (test half of the "
          f"{a.sample_size:,}-line seed-42 draw)", flush=True)

    # Original: this model's own plain-scorer predictions.
    if is_default:
        pred_orig = np.array(data["pred_UniLID"])[test_pos]
    else:
        if not a.baseline_pred:
            raise SystemExit(
                "--baseline-pred is required for a non-default model: the "
                "Original column must be that model's own plain-scorer "
                "predictions, and filling it from the sample pickle would put "
                "the released model's predictions in a row about this one")
        _w, langs, _m = _load_model_data(a.model_path)
        del _w
        codes = np.asarray(np.load(a.baseline_pred, mmap_mode="r"))[abs_idx]
        if (codes < 0).any():
            raise SystemExit(
                f"{int((codes < 0).sum()):,} of {len(codes):,} golden-subset "
                f"lines carry a sentinel in {a.baseline_pred}; that run did not "
                f"score this subset")
        pred_orig = np.array([langs[c] for c in codes], dtype=object)

    model = _load_unilid_model(a.model_path)
    print(f"scoring alpha={RAW_ALPHA} (raw rescore)...", flush=True)
    pred_raw = np.array(predict_all(texts, model, alpha=RAW_ALPHA))
    print(f"scoring alpha={NORM_ALPHA} (normalized)...", flush=True)
    pred_norm = np.array(predict_all(texts, model, alpha=NORM_ALPHA))

    columns = {}
    agreement = float((pred_orig == pred_raw).mean())
    print(f"raw-rescore agreement with the plain scorer: {agreement:.6f}", flush=True)
    if agreement < RAW_AGREEMENT_MIN:
        raise RuntimeError(
            f"the alpha={RAW_ALPHA} rescore reproduces only {agreement:.6f} of "
            f"the plain scorer's predictions, not all of them. The Raw rescore "
            f"column exists to show the two code paths agree; until they do, the "
            f"Normalized column cannot be attributed to normalization alone.")
    columns["Original"] = _by_length(pred_orig, y_true, lengths)
    columns["Raw rescore"] = _by_length(pred_raw, y_true, lengths)
    columns["Normalized"] = _by_length(pred_norm, y_true, lengths)

    names = list(columns)
    rows = []
    for i in range(len(LENGTH_LABELS) + 1):
        rows.append({"bin": columns[names[0]][i]["bin"],
                     "n": columns[names[0]][i]["n"],
                     **{c: columns[c][i]["accuracy"] for c in names}})

    header = "| Length (chars) | N | " + " | ".join(names) + " |"
    sep = "|---" * (len(names) + 2) + "|"
    lines = ["# tab:lenbias-norm: length-normalized scoring by input length", "",
             f"Model: `{a.model_path}`. Sample: {len(texts):,} lines, the test "
             f"half of the seed-42 {a.sample_size:,}-line draw (the golden "
             f"subset; the validation half is excluded).",
             f"Raw rescore is alpha={RAW_ALPHA}, Normalized is alpha={NORM_ALPHA} "
             f"(score divided by segmentation length).", ""]
    lines.append(f"Raw rescore reproduces the plain scorer exactly "
                 f"({agreement:.6f} agreement), which is the implementation "
                 f"check.")
    lines += ["", header, sep]
    for r in rows:
        cells = " | ".join("--" if r[c] is None else f"{r[c]:.3f}" for c in names)
        lines.append(f"| {r['bin']} | {r['n']:,} | {cells} |")

    os.makedirs(a.out_dir, exist_ok=True)
    md_path = os.path.join(a.out_dir, "lenbias_norm.md")
    Path(md_path).write_text("\n".join(lines) + "\n")
    json_path = os.path.join(a.out_dir, "lenbias_norm.json")
    Path(json_path).write_text(json.dumps(
        {"model": a.model_path, "sample_size": a.sample_size,
         "raw_alpha": RAW_ALPHA, "norm_alpha": NORM_ALPHA,
         "raw_agreement_with_recorded": agreement, "rows": rows}, indent=2))
    print("\n".join(lines))
    print(f"\nWrote {md_path} and {json_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
