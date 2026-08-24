"""Evaluate a .unilid model on WiLI-2018, and gate that instrument against the paper.

No WiLI tooling existed in this repository: `grep -rln wili analysis/*.py` was
empty before this file. `UNILID/eval.py` cannot serve, because it reports
accuracy, macro F1, macro precision and macro recall but **no macro FPR**
(`eval.py:309-316`; `grep -rn fpr UNILID/` returns nothing), and every WiLI table
in the paper reports FPR.

Metrics come from `analysis.metrics.compute_metrics`, whose FPR is the
convention recorded in `analysis/paper_eval.py:133-138`:
FPR_l = FP_l / (FP_l + TN_l) with TN_l = n - support_l - FP_l. The two agree
identically: compute_metrics uses tn = n - tp - fp - fn, and support = tp + fn.
Both average unweighted over languages with test support.

Conventions, stated because they change the third decimal:
- a text that preprocesses to empty is scored as WRONG rather than dropped, which
  is what analysis/full_test_eval.py does for the GlotLID-C pool. `UNILID/eval.py`
  instead drops such rows, so its denominators differ.
- a gold label outside the model's label set is likewise kept and scored as wrong.
Both counts are reported, so a run that silently changed the denominator is
visible.

Gate mode compares against published cells and exits non-zero on a mismatch, so it
can be used to decide whether this instrument reproduces the paper before any
number measured with it is trusted.

  python -m analysis.wili_eval --model M.unilid -o out.json
  python -m analysis.wili_eval --model M.unilid --expect-f1 0.960 \
      --expect-fpr 1.859e-4 --expect-acc 0.9565
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

from analysis.metrics import compute_metrics  # noqa: E402

WILI_DIR = ("/capstor/scratch/cscs/cmeister747/unilid_analysis/wili_assets/"
            "wili-2018")
BATCH = 10_000
# Published cells round to three decimals for F1 and accuracy and to four
# significant figures for FPR, so a reproduction is accepted at that resolution.
F1_TOL = 5e-4
ACC_TOL = 5e-4
FPR_REL_TOL = 0.01


def load_split(wili_dir: str, split: str):
    x = Path(wili_dir) / f"x_{split}.txt"
    y = Path(wili_dir) / f"y_{split}.txt"
    for p in (x, y):
        if not p.is_file():
            raise SystemExit(f"missing WiLI file: {p}")
    texts = x.read_text(encoding="utf-8", errors="replace").splitlines()
    labels = y.read_text(encoding="utf-8", errors="replace").splitlines()
    if len(texts) != len(labels):
        raise SystemExit(f"{len(texts):,} texts against {len(labels):,} labels")
    return texts, labels


def predict_all(model, texts, verbose=True):
    """Predict a label for every text, with wili_eval's exact conventions.

    Returns (preds, n_empty). A text that preprocesses to empty yields the
    sentinel "<EMPTY>" rather than being dropped, so it is kept in the
    denominator and scored as wrong. Shared with analysis/wili_length_accuracy.py
    so the two instruments cannot drift apart.
    """
    preds, n_empty = [], 0
    for start in range(0, len(texts), BATCH):
        for lang, _tok, _sc in model.predict_batch(texts[start:start + BATCH]):
            if not lang:
                n_empty += 1
                preds.append("<EMPTY>")     # kept, scored as wrong
            else:
                preds.append(lang)
    if verbose:
        print(f"  {n_empty:,} line(s) empty after preprocess, kept and scored "
              f"as wrong", flush=True)
    return preds, n_empty


def out_of_set_labels(model, gold, verbose=True):
    """Gold labels the model cannot emit; kept and scored as wrong."""
    out_of_set = sorted({g for g in gold if g not in set(model.langs)})
    if out_of_set and verbose:
        print(f"  {len(out_of_set)} gold label(s) absent from the model, kept and "
              f"scored as wrong: {out_of_set[:5]}", flush=True)
    return out_of_set


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--wili-dir", default=WILI_DIR)
    ap.add_argument("--split", default="test", choices=("test", "train"))
    ap.add_argument("-o", "--output", default=None)
    ap.add_argument("--expect-f1", type=float, default=None)
    ap.add_argument("--expect-fpr", type=float, default=None)
    ap.add_argument("--expect-acc", type=float, default=None)
    a = ap.parse_args(argv)

    from unilid.model_io import UnilidModel

    texts, gold = load_split(a.wili_dir, a.split)
    print(f"WiLI {a.split}: {len(texts):,} lines, "
          f"{len(set(gold)):,} labels", flush=True)

    model = UnilidModel(a.model, calibrated=False)
    out_of_set = out_of_set_labels(model, gold)
    preds, n_empty = predict_all(model, texts)

    m = compute_metrics(np.array(gold, dtype=object), np.array(preds, dtype=object))
    res = {"model": os.path.abspath(a.model), "split": a.split,
           "total_samples": len(texts), "n_labels": len(set(gold)),
           "n_empty_after_preprocess": n_empty,
           "n_gold_labels_absent_from_model": len(out_of_set),
           "accuracy": m["accuracy"], "macro_f1": m["macro_f1"],
           "macro_fpr": m["macro_fpr"]}
    print(f"\naccuracy   {m['accuracy']:.4f}")
    print(f"macro F1   {m['macro_f1']:.4f}")
    print(f"macro FPR  {m['macro_fpr']:.4e}")

    checks = []
    if a.expect_f1 is not None:
        ok = abs(m["macro_f1"] - a.expect_f1) <= F1_TOL
        checks.append(("macro F1", m["macro_f1"], a.expect_f1, ok))
    if a.expect_acc is not None:
        ok = abs(m["accuracy"] - a.expect_acc) <= ACC_TOL
        checks.append(("accuracy", m["accuracy"], a.expect_acc, ok))
    if a.expect_fpr is not None:
        ok = abs(m["macro_fpr"] - a.expect_fpr) <= FPR_REL_TOL * a.expect_fpr
        checks.append(("macro FPR", m["macro_fpr"], a.expect_fpr, ok))
    if checks:
        print("\ngate against the published cells:")
        for name, got, want, ok in checks:
            print(f"  {name:10} got {got:.6g}  expected {want:.6g}  "
                  f"{'MATCH' if ok else 'MISMATCH'}")
        res["gate"] = [{"metric": n, "got": g, "expected": w, "match": bool(o)}
                       for n, g, w, o in checks]
        res["gate_passed"] = all(o for *_, o in checks)

    if a.output:
        Path(a.output).write_text(json.dumps(res, indent=2))
    if checks and not res["gate_passed"]:
        print("\nGATE FAILED: this instrument does not reproduce the published "
              "cells, so nothing measured with it can be trusted yet.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
