"""What does the special-token correction do to the paper's base-model metrics?

Scores the released and the corrected weights on the golden subset (the test half
of the seed-42 500k draw, 250,000 lines) in BASE mode, and reports macro F1,
macro FPR and accuracy against the recorded gold labels. This is the cheap
estimate that decides whether the full 45.6M-line passes are worth starting, and
in which direction the paper's numbers move.

Base mode only: the calibration's thresholds were estimated on the old scale and
do not apply to corrected weights until they are re-derived.

  python -m analysis.correction_effect -o outputs/rerelease/correction_effect.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "UNILID"))

from analysis.release_gates import (  # noqa: E402
    EMPTY, STORE, _collect_texts, _golden_indices, _predict,
)
from unilid.model_io import UnilidModel  # noqa: E402

Y_TRUE = f"{STORE}/full_test_eval/y_true.npy"
RELEASED = f"{STORE}/glotlidc.unilid"
CORRECTED = ("/capstor/scratch/cscs/cmeister747/unilid_analysis/corrected/"
             "glotlidc_corrected.unilid")


def macro_metrics(pred: np.ndarray, gold: np.ndarray, n_labels: int) -> dict:
    """Macro F1 and macro FPR over every label present in the gold labels."""
    labels = np.unique(gold)
    f1s, fprs = [], []
    for lab in labels:
        is_gold = gold == lab
        is_pred = pred == lab
        tp = int((is_gold & is_pred).sum())
        fp = int((~is_gold & is_pred).sum())
        fn = int((is_gold & ~is_pred).sum())
        tn = len(gold) - tp - fp - fn
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
        fprs.append(fp / (fp + tn) if fp + tn else 0.0)
    return {"macro_f1": float(np.mean(f1s)), "macro_fpr": float(np.mean(fprs)),
            "accuracy": float((pred == gold).mean()), "n_labels": len(labels)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--output", default=None)
    args = ap.parse_args(argv)

    idx = _golden_indices()
    print(f"golden subset: {len(idx):,} lines", flush=True)
    gold = np.load(Y_TRUE)[idx]
    if (gold < 0).any():
        raise RuntimeError("gold labels contain negative sentinels on the "
                           "golden subset; the subset derivation is wrong")
    texts = _collect_texts(idx)

    out = {}
    for tag, path in (("released", RELEASED), ("corrected", CORRECTED)):
        model = UnilidModel(path, calibrated=False)
        lang_to_idx = {lang: i for i, lang in enumerate(model.langs)}
        print(f"scoring {tag} ({path})", flush=True)
        pred = _predict(model, texts, lang_to_idx)
        n_empty = int((pred == EMPTY).sum())
        out[tag] = macro_metrics(pred, gold, len(model.langs))
        out[tag]["empty_predictions"] = n_empty
        out[f"{tag}_pred"] = pred
        print(f"  {tag}: macro F1 {out[tag]['macro_f1']:.4f}  "
              f"macro FPR {out[tag]['macro_fpr']:.3e}  "
              f"accuracy {out[tag]['accuracy']:.4f}", flush=True)
        del model

    a, b = out.pop("released_pred"), out.pop("corrected_pred")
    changed = int((a != b).sum())
    was, now = a == gold, b == gold
    out["comparison"] = {
        "predictions_changed": changed,
        "fraction_changed": changed / len(gold),
        "fixed_by_correction": int((~was & now).sum()),
        "broken_by_correction": int((was & ~now).sum()),
        "delta_macro_f1": out["corrected"]["macro_f1"] - out["released"]["macro_f1"],
        "delta_macro_fpr": out["corrected"]["macro_fpr"] - out["released"]["macro_fpr"],
        "delta_accuracy": out["corrected"]["accuracy"] - out["released"]["accuracy"],
    }
    print(json.dumps(out["comparison"], indent=2))
    if args.output:
        Path(args.output).write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
