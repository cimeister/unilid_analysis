"""The tail under both metric views, for the corrected model.

Exp 24 established that the two views disagree about the tail structurally, and
that the ranking of configurations INVERTS between them:

- the within-stratum view (what every stratum row of the full-test tables
  reports) restricts truth and predictions to examples whose true label is in the
  stratum, so a head-true line predicted as a tail language is excluded. It shows
  the recall cost of the clamp.
- global per-language F1 over the same languages counts those false positives. It
  shows the precision gain.

For a deployment where a tail language absorbing head-language text is the
expensive failure, the second view is the relevant one, and it is the view the
paper's own FPR argument appeals to. This computes both for the corrected model
so the clamp can be judged on either.

  python -m analysis.tail_views_corrected -o outputs/rerelease/tail_views_corrected.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
from analysis.transfer_sweep import _load_model_data, _load_train_counts

SCRATCH = "/capstor/scratch/cscs/cmeister747/unilid_analysis/full_test_eval_corrected"
CORRECTED = ("/capstor/scratch/cscs/cmeister747/unilid_analysis/corrected/"
             "glotlidc_corrected.unilid")
TAIL_MAX_N = 1_000          # the strata definition used throughout: tail is N < 1000
# Recorded Exp 24 values for the RELEASED model, for side-by-side reading.
RELEASED = {"baseline": {"within": 0.9132, "global_f1": 0.5618, "prec": 0.459,
                         "rec": 0.874, "fp_into_tail": 22522},
            "floor21": {"within": 0.8928, "global_f1": 0.7655, "prec": 0.763,
                        "rec": 0.842, "fp_into_tail": 9103}}


def views(y: np.ndarray, pred: np.ndarray, tail_mask: np.ndarray) -> dict:
    """Within-stratum tail F1, and global per-language tail F1/precision/recall."""
    tail_ids = np.flatnonzero(tail_mask)
    is_tail_true = tail_mask[y]

    # within-stratum: restrict BOTH truth and prediction to tail-true examples
    yw, pw = y[is_tail_true], pred[is_tail_true]
    f1s = []
    for lid in tail_ids:
        tp = int(((yw == lid) & (pw == lid)).sum())
        fp = int(((yw != lid) & (pw == lid)).sum())
        fn = int(((yw == lid) & (pw != lid)).sum())
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * p * r / (p + r) if p + r else 0.0)
    within = float(np.mean(f1s))

    # global: the same languages, but every false positive counted wherever it came from
    gf1, gp, gr = [], [], []
    fp_into_tail = 0
    for lid in tail_ids:
        tp = int(((y == lid) & (pred == lid)).sum())
        fp = int(((y != lid) & (pred == lid)).sum())
        fn = int(((y == lid) & (pred != lid)).sum())
        fp_into_tail += fp
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        gp.append(p); gr.append(r)
        gf1.append(2 * p * r / (p + r) if p + r else 0.0)
    return {"within_stratum_f1": within, "global_f1": float(np.mean(gf1)),
            "global_precision": float(np.mean(gp)), "global_recall": float(np.mean(gr)),
            "fp_into_tail_labels": int(fp_into_tail),
            "n_tail_languages": len(tail_ids),
            "n_tail_true_examples": int(is_tail_true.sum())}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-o", "--output", default=None)
    a = ap.parse_args(argv)

    _w, langs, _m = _load_model_data(CORRECTED)
    del _w
    counts = _load_train_counts()
    N = np.array([counts.get(l, 0) for l in langs])
    tail_mask = N < TAIL_MAX_N

    y = np.asarray(np.load(f"{SCRATCH}/y_true.npy", mmap_mode="r"))
    kept = y >= 0
    yk = y[kept]
    out = {"tail_max_n": TAIL_MAX_N, "released_recorded": RELEASED, "corrected": {}}
    for tag, fname in (("baseline", "pred_baseline.npy"),
                       ("floor17", "pred_floor21.npy")):
        pred = np.asarray(np.load(f"{SCRATCH}/{fname}", mmap_mode="r"))[kept]
        out["corrected"][tag] = views(yk, pred, tail_mask)
        del pred

    print(f"tail = {out['corrected']['baseline']['n_tail_languages']} languages with "
          f"N < {TAIL_MAX_N:,}, {out['corrected']['baseline']['n_tail_true_examples']:,} "
          f"true examples\n")
    hdr = f"{'config':<22}{'within-stratum':>15}{'global F1':>11}{'precision':>11}{'recall':>9}{'FPs into tail':>15}"
    print(hdr); print("-" * len(hdr))
    for tag, r in RELEASED.items():
        print(f"{'released ' + tag:<22}{r['within']:>15.4f}{r['global_f1']:>11.4f}"
              f"{r['prec']:>11.3f}{r['rec']:>9.3f}{r['fp_into_tail']:>15,}")
    for tag, r in out["corrected"].items():
        print(f"{'corrected ' + tag:<22}{r['within_stratum_f1']:>15.4f}"
              f"{r['global_f1']:>11.4f}{r['global_precision']:>11.3f}"
              f"{r['global_recall']:>9.3f}{r['fp_into_tail_labels']:>15,}")
    if a.output:
        Path(a.output).write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
