"""The cross-language relation between the unseen-token plateau and corpus size.

Each row of the model contains a large block of entries at the row's exact
minimum: the tokens the estimator never observed for that language. The paper's
appendix attributes that value to the training-time probability floor of 1e-12.
It is not that: every observed plateau sits well above log(1e-12) = -27.631, and
the sp training path never applies the floor to a real token.

This fits the relation the value actually follows, from the committed Exp 27
counting pass (outputs/diagnostic/gt_counts.csv), which recorded for every
language its plateau value, its Viterbi token count T, and its corpus line count.
It is the reference the single-language subsample experiment
(analysis/plateau_vs_corpus_size.py) is compared against.

The relation is measured ACROSS 1,940 different languages, so corpus size is
confounded with language identity here. Separating them is what the subsample
experiment is for.

  python -m analysis.plateau_reference_fit -o outputs/rerelease/plateau_reference_fit.json
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
GT_COUNTS = REPO / "outputs/diagnostic/gt_counts.csv"
TRAINING_FLOOR = float(np.log(1e-12))


def fit(x: np.ndarray, y: np.ndarray) -> dict:
    slope, intercept = np.polyfit(x, y, 1)
    pred = slope * x + intercept
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return {"slope": float(slope), "intercept": float(intercept),
            "corr": float(np.corrcoef(x, y)[0, 1]),
            "r_squared": 1.0 - ss_res / ss_tot}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--output", default=None)
    args = ap.parse_args(argv)

    langs, plateau, T, lines = [], [], [], []
    with open(GT_COUNTS) as f:
        for row in csv.DictReader(f):
            langs.append(row["lang"])
            plateau.append(float(row["floor"]))
            T.append(int(row["T"]))
            lines.append(int(row["corpus_lines"]))
    plateau = np.array(plateau)
    T = np.array(T, dtype=np.float64)
    lines = np.array(lines, dtype=np.float64)
    print(f"{len(langs):,} languages from {GT_COUNTS.name}")

    if (plateau <= TRAINING_FLOOR).any():
        raise SystemExit("a plateau reached the training floor; the claim that "
                         "the floor is never reached is false for this model")
    print(f"plateau range {plateau.min():.2f} to {plateau.max():.2f}; "
          f"training floor {TRAINING_FLOOR:.3f}; closest approach "
          f"{(plateau.min() - TRAINING_FLOOR):.2f} nats above it")

    out = {"n_languages": len(langs), "training_floor": TRAINING_FLOOR,
           "plateau_min": float(plateau.min()), "plateau_max": float(plateau.max()),
           "plateau_median": float(np.median(plateau)),
           "vs_log10_tokens": fit(np.log10(T), plateau),
           "vs_log10_lines": fit(np.log10(lines), plateau)}

    for name in ("vs_log10_tokens", "vs_log10_lines"):
        f = out[name]
        print(f"{name:18} plateau = {f['intercept']:.3f} {f['slope']:+.3f} * x   "
              f"corr {f['corr']:.4f}  R^2 {f['r_squared']:.3f}")

    # Median plateau by corpus-size band, the descriptive form quoted in the docs.
    bands = [(0, 1_000), (1_000, 50_000), (50_000, np.inf)]
    out["median_by_band"] = []
    for lo, hi in bands:
        m = (lines >= lo) & (lines < hi)
        if m.any():
            rec = {"lines_from": lo, "lines_to": None if hi == np.inf else hi,
                   "n": int(m.sum()), "median_plateau": float(np.median(plateau[m]))}
            out["median_by_band"].append(rec)
            print(f"  lines [{lo:,}, {rec['lines_to'] or 'inf'}): n={rec['n']:>4}  "
                  f"median plateau {rec['median_plateau']:.3f}")

    if args.output:
        Path(args.output).write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
