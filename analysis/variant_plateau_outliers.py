"""Flag rows whose unseen-token plateau is far below what corpus size predicts.

B0 (analysis/plateau_vs_corpus_size.py) established that a row's unseen-token
plateau is near-deterministic in the language's corpus size, falling about 2 nats
per decade. That makes it a diagnostic: fit the relation WITHIN a model, and a row
sitting many standard deviations below its own expectation has something wrong
that corpus size does not explain.

This is a sharper instrument than analysis/degeneracy_scan.py for one failure
mode. The degeneracy scan counts how many tokens sit off the plateau, which
catches a row that collapsed wholesale; it does not catch a row that kept plenty
of estimated tokens but had its most frequent ones deleted. That is exactly the
signature of the fixed-vocabulary EM bug (EXPERIMENTAL_SETUP.md, "The
fixed-vocabulary EM bug"), where 32-bit expected counts overflowed and the fork's
guard mapped non-finite counts to zero, "deleting exactly the most frequent
tokens and leaving a plausible-looking but collapsed model". The recorded trigger
was the longest line in all 1,940 corpora, in the Azerbaijani corpus.

Two causes must not be confused, matching the degeneracy scan's own note:
deterministic vocabulary coverage (a script with few pieces in the inventory,
which shows up in every model built on a vocabulary that lacks them) and genuine
training failure (which shows up in one model and not another built from the same
corpora).

  python -m analysis.variant_plateau_outliers MODEL.unilid [...] -o out.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "UNILID"))

from unilid.constants import SPECIAL_TOKENS  # noqa: E402
from unilid.model_io import load_unilid_raw  # noqa: E402

TRAIN_COUNTS = ("/capstor/store/cscs/swissai/a0229/cmeister/unilid_analysis/"
                "glotlid_unilid/glotlid_train_counts.json")
ROW_BLOCK = 128
# Residuals beyond this many standard deviations are reported. The within-model
# residual spread is about 0.4 nats, so this is roughly 2 nats: far outside what
# the corpus-size relation leaves unexplained, and far inside the 8-nat
# departures the EM bug produced.
OUTLIER_SD = 5.0
# The fit is made robust by dropping the worst 1% of residuals before refitting,
# so a handful of corrupted rows cannot drag the line toward themselves.
ROBUST_DROP_PCT = 99


def plateaus(path: str):
    tok, weights, langs = load_unilid_raw(path)
    text = tok if isinstance(tok, str) else tok.decode("utf-8")
    vocab = [t for t, _ in json.loads(text)["model"]["vocab"]]
    W = np.asarray(weights)
    sidx = np.array([vocab.index(t) for t in SPECIAL_TOKENS.values() if t in vocab])
    pl = np.empty(W.shape[0])
    for s in range(0, W.shape[0], ROW_BLOCK):
        blk = np.array(W[s:s + ROW_BLOCK], dtype=np.float32)
        blk[:, sidx] = np.inf          # specials excluded from the row minimum
        pl[s:s + blk.shape[0]] = blk.min(axis=1)
        del blk
    n_vocab = int(W.shape[1])
    del W, weights
    return pl, langs, n_vocab


def analyze(path: str, counts: dict) -> dict:
    pl, langs, n_vocab = plateaus(path)
    N = np.array([counts.get(l, 0) for l in langs], dtype=float)
    ok = N > 0
    x, y = np.log10(N[ok]), pl[ok]
    names = [l for l, o in zip(langs, ok) if o]

    slope, intercept = np.polyfit(x, y, 1)
    resid = y - (slope * x + intercept)
    keep = np.abs(resid) < np.percentile(np.abs(resid), ROBUST_DROP_PCT)
    slope, intercept = np.polyfit(x[keep], y[keep], 1)
    resid = y - (slope * x + intercept)
    sd = float(resid[keep].std())

    flags = [{"lang": names[i], "n_l": int(N[ok][i]), "plateau": float(y[i]),
              "expected": float(slope * x[i] + intercept),
              "residual": float(resid[i]), "sd_below": float(resid[i] / sd)}
             for i in np.argsort(resid) if resid[i] / sd <= -OUTLIER_SD]

    print(f"\n{Path(path).name}  (vocab {n_vocab:,})")
    print(f"  plateau = {intercept:.3f} {slope:+.3f} * log10(N_L), "
          f"residual sd {sd:.3f} nats")
    print(f"  {len(flags)} row(s) more than {OUTLIER_SD} sd below expectation:")
    for f in flags:
        print(f"     {f['lang']:10} N={f['n_l']:>7,}  plateau {f['plateau']:>8.3f}  "
              f"expected {f['expected']:>8.3f}  {f['sd_below']:>6.1f} sd")
    return {"model": path, "vocab_size": n_vocab, "slope": float(slope),
            "intercept": float(intercept), "residual_sd": sd,
            "outlier_sd_threshold": OUTLIER_SD, "outliers": flags}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("models", nargs="+")
    ap.add_argument("-o", "--output", default=None)
    a = ap.parse_args(argv)
    counts = json.loads(Path(TRAIN_COUNTS).read_text())
    results = [analyze(m, counts) for m in a.models]

    # A language flagged in every model is a vocabulary-coverage effect; one
    # flagged in a single model points at that model's training run.
    if len(results) > 1:
        sets = [{f["lang"] for f in r["outliers"]} for r in results]
        shared = set.intersection(*sets)
        print(f"\nflagged in every model (vocabulary coverage): {sorted(shared)}")
        for r, s in zip(results, sets):
            only = sorted(s - shared)
            if only:
                print(f"flagged ONLY in {Path(r['model']).name}: {only}")
    if a.output:
        Path(a.output).write_text(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
