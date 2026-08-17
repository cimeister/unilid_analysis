"""Move the probability mass off the special tokens of a released .unilid file.

Per-language training before UNILID 0.3.0 gave each of the four special tokens a
pre-normalization log-probability of 0.0, i.e. probability 1.0. Since
SentencePiece emits normalized log-probabilities, the real tokens summed to 1 and
the four specials added 4, so every stored row is the correct row divided by 5.

No special token's stored weight is read when scoring: the scorer takes its
unknown-token score from a model-wide constant (min_score - K_UNK_PENALTY in
model.rs) and <s>/</s>/<pad> are reachable only by text containing those literal
substrings. The mass is therefore pure loss, taken from the tokens that decide
predictions.

This rewrites each row to put all of its mass on the real tokens and parks the
special tokens at the training floor. The result is what UNILID 0.3.0 produces
when training the same corpus with the same base tokenizer; that equivalence is
gated separately by analysis/gate_correction.py, which retrains languages and
compares.

Writes a version-1 container. Any calibration must be re-derived against the
corrected weights, not carried over.

  python -m analysis.correct_special_token_mass IN.unilid -o OUT.unilid
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "UNILID"))

from unilid.constants import MIN_TOKEN_LOG_PROB, SPECIAL_TOKENS  # noqa: E402
from unilid.model_io import load_unilid_raw, write_unilid  # noqa: E402

# The rows of one model must be on one scale, or there is nothing to correct
# uniformly and the file was built by mixing training methods.
MAX_REAL_MASS_SPREAD = 1.01
ROW_BLOCK = 64


def special_indices(tok_json) -> tuple[list[int], list[str]]:
    text = tok_json if isinstance(tok_json, str) else tok_json.decode("utf-8")
    vocab = [t for t, _ in json.loads(text)["model"]["vocab"]]
    return [vocab.index(t) for t in SPECIAL_TOKENS.values() if t in vocab], vocab


def real_token_mass(weights, real_idx: np.ndarray) -> np.ndarray:
    out = np.empty(weights.shape[0], dtype=np.float64)
    for start in range(0, weights.shape[0], ROW_BLOCK):
        chunk = np.asarray(weights[start:start + ROW_BLOCK])[:, real_idx]
        out[start:start + ROW_BLOCK] = np.exp(chunk.astype(np.float64)).sum(axis=1)
    return out


def correct(in_path: Path, out_path: Path) -> dict:
    tok_json, weights, langs = load_unilid_raw(in_path)
    spec_idx, vocab = special_indices(tok_json)
    if len(spec_idx) != len(SPECIAL_TOKENS):
        raise RuntimeError(
            f"expected {len(SPECIAL_TOKENS)} special tokens in the vocabulary, "
            f"found {len(spec_idx)}: {[vocab[i] for i in spec_idx]}")
    real_mask = np.ones(len(vocab), dtype=bool)
    real_mask[spec_idx] = False
    real_idx = np.flatnonzero(real_mask)

    w = np.asarray(weights).astype(np.float32).copy()
    del weights

    before_real = real_token_mass(w, real_idx)
    before_spec = np.exp(np.asarray(w[:, spec_idx], dtype=np.float64)).sum(axis=1)
    lo, hi = float(before_real.min()), float(before_real.max())
    if lo <= 0.0 or not np.isfinite(hi):
        raise RuntimeError(f"rows put no usable mass on real tokens "
                           f"(min {lo:.6g}, max {hi:.6g})")
    if hi / lo > MAX_REAL_MASS_SPREAD:
        raise RuntimeError(
            f"the rows are not on one scale: real-token mass ranges from "
            f"{lo:.6g} to {hi:.6g} across {len(langs):,} languages, so no single "
            f"correction applies. This file was probably built by mixing "
            f"per-language training methods; correct it per row only after "
            f"establishing why")

    shift = (-np.log(before_real)).astype(np.float32)
    for start in range(0, w.shape[0], ROW_BLOCK):
        block = w[start:start + ROW_BLOCK]
        block[:, real_idx] += shift[start:start + ROW_BLOCK, None]
    w[:, spec_idx] = np.float32(MIN_TOKEN_LOG_PROB)

    after_real = real_token_mass(w, real_idx)
    after_spec = np.exp(np.asarray(w[:, spec_idx], dtype=np.float64)).sum(axis=1)
    if not np.allclose(after_real, 1.0, rtol=1e-4):
        raise RuntimeError(
            f"corrected rows do not sum to one over the real tokens "
            f"(min {after_real.min():.6g}, max {after_real.max():.6g})")
    if not np.isfinite(w).all():
        raise RuntimeError("corrected weights contain non-finite values")

    write_unilid(out_path, tok_json, langs, w, calibration=None)

    return {
        "input": str(in_path), "output": str(out_path),
        "languages": len(langs), "vocab_size": len(vocab),
        "special_indices": spec_idx,
        "real_mass_before": [lo, hi],
        "special_mass_before": [float(before_spec.min()), float(before_spec.max())],
        "real_mass_after": [float(after_real.min()), float(after_real.max())],
        "special_mass_after": [float(after_spec.min()), float(after_spec.max())],
        "shift_nats": [float(shift.min()), float(shift.max())],
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("model")
    ap.add_argument("-o", "--output", required=True)
    args = ap.parse_args(argv)

    summary = correct(Path(args.model), Path(args.output))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
