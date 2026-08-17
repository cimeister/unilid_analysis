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
- Original is the model's recorded predictions. For the released model those are
  pred_UniLID in the seed-42 sample pickle, and the agreement with the Raw
  rescore column is reported as the implementation check. For any other model
  there is no recorded column, so Original is omitted and the caption says so
  rather than silently reusing another model's predictions.

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
from analysis.normalized_predict import (UNILID_MODEL_PATH, _load_unilid_model,
                                         _stream_sampled_texts, predict_all)
from analysis.sample_data import load_sample

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
    a = ap.parse_args(argv)

    is_default = os.path.abspath(a.model_path) == os.path.abspath(UNILID_MODEL_PATH)
    print(f"model: {a.model_path}"
          f"{'' if is_default else '  (not the released model)'}", flush=True)

    data = load_sample(a.sample_size)
    y_true = np.array(data["y_true"])
    lengths = np.array(data["text_lengths"])
    texts = _stream_sampled_texts(a.sample_size)
    if len(texts) != len(y_true):
        raise RuntimeError(f"{len(texts)} texts against {len(y_true)} labels")

    model = _load_unilid_model(a.model_path)
    print(f"scoring alpha={RAW_ALPHA} (raw rescore)...", flush=True)
    pred_raw = np.array(predict_all(texts, model, alpha=RAW_ALPHA))
    print(f"scoring alpha={NORM_ALPHA} (normalized)...", flush=True)
    pred_norm = np.array(predict_all(texts, model, alpha=NORM_ALPHA))

    columns = {}
    agreement = None
    if is_default:
        pred_orig = np.array(data["pred_UniLID"])
        agreement = float((pred_orig == pred_raw).mean())
        print(f"raw-rescore agreement with the recorded predictions: "
              f"{agreement:.6f}", flush=True)
        if agreement < RAW_AGREEMENT_MIN:
            raise RuntimeError(
                f"the alpha={RAW_ALPHA} rescore reproduces only {agreement:.6f} of "
                f"the recorded UniLID predictions, not all of them. The Raw "
                f"rescore column exists to show the two code paths agree; until "
                f"they do, the Normalized column cannot be attributed to "
                f"normalization alone.")
        columns["Original"] = _by_length(pred_orig, y_true, lengths)
    else:
        print("no recorded prediction column exists for this model; the Original "
              "column is omitted rather than filled from another model", flush=True)
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
             f"Model: `{a.model_path}`. Sample: {a.sample_size:,} lines "
             f"(seed-42 draw).",
             f"Raw rescore is alpha={RAW_ALPHA}, Normalized is alpha={NORM_ALPHA} "
             f"(score divided by segmentation length).", ""]
    if agreement is not None:
        lines.append(f"Raw rescore reproduces the recorded predictions exactly "
                     f"({agreement:.6f} agreement), which is the implementation "
                     f"check.")
    else:
        lines.append("The Original column is omitted: no recorded prediction "
                     "column exists for this model.")
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
