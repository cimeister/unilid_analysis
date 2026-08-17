"""How the special-token correction moves the Viterbi segmentation.

A language scores a text as the sum of log p(token | language) over that
language's own Viterbi segmentation. The correction adds log 5 = 1.6094 nats to
every real token, so a segmentation with n tokens gains n * log 5. Two
consequences follow from the scoring rule, and this script measures both rather
than asserting them:

1. The max-plus DP maximizes sum(log p_i) + n * log 5, so a positive per-token
   constant favors segmentations with more tokens.
2. The gain differs across languages for the same text, because the token count
   under each language's own segmentation differs. The correction is therefore
   not a constant offset on every candidate and can change the argmax.

  python -m analysis.segmentation_shift -o outputs/rerelease/segmentation_shift.json
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "UNILID"))

from analysis.transfer_sweep import UNILID_MODEL_PATH  # noqa: E402

CORRECTED = ("/capstor/scratch/cscs/cmeister747/unilid_analysis/corrected/"
             "glotlidc_corrected.unilid")
TEXT_CACHE = ("/capstor/scratch/cscs/cmeister747/unilid_analysis/"
              "probe_sample_texts.pkl")
N_LINES = 3_000
SEED = 7
LOG5 = float(np.log(5.0))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--output", default=None)
    ap.add_argument("--n-lines", type=int, default=N_LINES)
    args = ap.parse_args(argv)

    from unilid.model_io import UnilidModel

    with open(TEXT_CACHE, "rb") as f:
        texts = pickle.load(f)
    rng = np.random.default_rng(SEED)
    pick = np.sort(rng.choice(len(texts), args.n_lines, replace=False))
    sub = [texts[i] for i in pick]

    got = {}
    for tag, path in (("released", UNILID_MODEL_PATH), ("corrected", CORRECTED)):
        model = UnilidModel(path, calibrated=False)
        got[tag] = [(lang, len(toks), score)
                    for lang, toks, score in model.predict_batch(sub)]
        del model

    a, b = got["released"], got["corrected"]
    n_a = np.array([r[1] for r in a])
    n_b = np.array([r[1] for r in b])
    s_a = np.array([r[2] for r in a])
    s_b = np.array([r[2] for r in b])
    same_lang = np.array([a[i][0] == b[i][0] for i in range(len(a))])
    same_seg = n_a == n_b
    keep = same_lang & same_seg
    resid = float(np.abs((s_b - s_a) - n_a * LOG5)[keep].max()) if keep.any() else None

    out = {
        "n_lines": len(sub), "seed": SEED, "log5": LOG5,
        "predictions_changed": int((~same_lang).sum()),
        "segmentation_changed": int((~same_seg).sum()),
        "segmentation_finer": int((n_b > n_a).sum()),
        "segmentation_coarser": int((n_b < n_a).sum()),
        "mean_tokens_released": float(n_a.mean()),
        "mean_tokens_corrected": float(n_b.mean()),
        "n_same_lang_and_segmentation": int(keep.sum()),
        "max_abs_score_delta_minus_n_log5": resid,
    }
    for k, v in out.items():
        print(f"{k:38} {v}")
    if args.output:
        Path(args.output).write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
