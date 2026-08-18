"""Does a .unilid model carry the special-token defect, and where are its specials?

Row-blocked so it works on the LLM-tokenizer variants: their vocabularies run to
about 150,000 entries, and materializing exp() over the whole float64 matrix at
once needs several gigabytes per temporary.

Reports the defect signature (each of the four special tokens at exactly 1/5, so
0.8 special mass and 0.2 real mass per row) and the special columns' positions,
which are NOT the same across models: the base GlotLID-C model puts them first,
Mistral-Nemo at 0, 1, 2 and 10, DeepSeek3.2 at the end of a 128,819-entry
vocabulary.

  python -m analysis.inspect_variant_models MODEL.unilid [MODEL2.unilid ...] -o out.json
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

ROW_BLOCK = 64
# The defect puts each of four specials at exactly 1/5. Tolerance is loose enough
# for float32 storage and tight enough that a corrected model (real mass 1.0)
# cannot be mistaken for a defective one.
DEFECT_TOL = 1e-4


def inspect(path: str) -> dict:
    tok, weights, langs = load_unilid_raw(path)
    text = tok if isinstance(tok, str) else tok.decode("utf-8")
    vocab = [t for t, _ in json.loads(text)["model"]["vocab"]]
    W = np.asarray(weights)
    special_idx = [vocab.index(t) for t in SPECIAL_TOKENS.values() if t in vocab]
    real_idx = np.setdiff1d(np.arange(W.shape[1]), np.array(special_idx))

    sm_lo = sm_hi = rm_lo = rm_hi = pl_lo = pl_hi = None
    for start in range(0, W.shape[0], ROW_BLOCK):
        blk = np.asarray(W[start:start + ROW_BLOCK], dtype=np.float64)
        sm = np.exp(blk[:, special_idx]).sum(axis=1)
        rm = np.exp(blk[:, real_idx]).sum(axis=1)
        pl = blk[:, real_idx].min(axis=1)
        sm_lo = sm.min() if sm_lo is None else min(sm_lo, sm.min())
        sm_hi = sm.max() if sm_hi is None else max(sm_hi, sm.max())
        rm_lo = rm.min() if rm_lo is None else min(rm_lo, rm.min())
        rm_hi = rm.max() if rm_hi is None else max(rm_hi, rm.max())
        pl_lo = pl.min() if pl_lo is None else min(pl_lo, pl.min())
        pl_hi = pl.max() if pl_hi is None else max(pl_hi, pl.max())
        del blk

    defect = bool(abs(sm_lo - 0.8) < DEFECT_TOL and abs(sm_hi - 0.8) < DEFECT_TOL)
    out = {"model": path, "n_languages": len(langs), "vocab_size": int(W.shape[1]),
           "special_columns": [int(i) for i in special_idx],
           "special_mass_min": float(sm_lo), "special_mass_max": float(sm_hi),
           "real_mass_min": float(rm_lo), "real_mass_max": float(rm_hi),
           "plateau_min": float(pl_lo), "plateau_max": float(pl_hi),
           "defect_present": defect}
    print(f"\n{Path(path).name}")
    print(f"  languages {out['n_languages']:,}  vocab {out['vocab_size']:,}")
    print(f"  special columns {out['special_columns']}")
    print(f"  special mass per row  {sm_lo:.6f} to {sm_hi:.6f}")
    print(f"  real mass per row     {rm_lo:.6f} to {rm_hi:.6f}")
    print(f"  unseen plateau        {pl_lo:.3f} to {pl_hi:.3f}")
    print(f"  DEFECT PRESENT: {defect}")
    del W, weights
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("models", nargs="+")
    ap.add_argument("-o", "--output", default=None)
    a = ap.parse_args(argv)
    results = [inspect(m) for m in a.models]
    if a.output:
        Path(a.output).write_text(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
