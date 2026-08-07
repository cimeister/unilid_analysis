"""Post-pack degeneracy gate for the Mistral-Nemo retrain (task 3, "post-pack
degeneracy scan invocation").

SETUP.md gotcha 8 and EXPERIMENTAL_SETUP.md "Per-language training pipeline"
both require `analysis/degeneracy_scan.py` (or an equivalent row-level scan)
to run on every newly packed model before evaluation, flagging rows with
fewer than MIN_ESTIMATED entries above the row minimum.

This is a standalone script rather than an added entry in
`analysis/degeneracy_scan.py`'s MODELS dict for two reasons, both worth
flagging back rather than silently deciding:
  1. Task 3 asked for new files only.
  2. `analysis/degeneracy_scan.py`'s MODELS dict is itself stale: it points at
     glotlid_apertus{131k,200k}.unilid (the pre-fp64, since-superseded
     models), not the glotlid_apertus{131k,200k}_fp64.unilid models that are
     the current artifacts of record. Silently adding a fourth entry next to
     two already-wrong ones would make the drift worse, not better. Whether
     to fix degeneracy_scan.py's MODELS dict (separately, on request) is an
     open question for the user, not resolved here.

MIN_ESTIMATED is imported from analysis.degeneracy_scan so this scan's
degeneracy threshold cannot drift from the canonical one.

Output: outputs/tables/degenerate_rows_mistralnemo.md, with the flagged set
per language and script composition (no per-language F1 column: no evaluation
of this model exists yet at preparation time).
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from analysis.degeneracy_scan import MIN_ESTIMATED
from analysis.mistralnemo_constants import DEGENERACY_OUT_MD, PACKED_MODEL_PATH
from analysis.transfer_sweep import _load_model_data

LANG_DIAGNOSTIC_CSV = "outputs/diagnostic/lang_diagnostic.csv"


def run(out_md: str = DEGENERACY_OUT_MD) -> str:
    if not os.path.exists(PACKED_MODEL_PATH):
        raise FileNotFoundError(f"packed model missing: {PACKED_MODEL_PATH}")
    if not os.path.exists(LANG_DIAGNOSTIC_CSV):
        raise FileNotFoundError(f"language diagnostic table missing: {LANG_DIAGNOSTIC_CSV}")

    diag = pd.read_csv(LANG_DIAGNOSTIC_CSV)
    script = dict(zip(diag.lang, diag.script))

    W, langs, _ = _load_model_data(PACKED_MODEL_PATH)
    vocab = W.shape[1]

    flagged = []
    for i in range(len(langs)):
        row = np.asarray(W[i], dtype=np.float64)
        est = int(vocab - (row == row.min()).sum())
        if est < MIN_ESTIMATED:
            flagged.append((langs[i], est))

    L = [f"# Degenerate-row scan: mistralnemo_fp64 (fewer than {MIN_ESTIMATED} "
         f"estimated tokens)\n",
         f"Model: {PACKED_MODEL_PATH}\n",
         f"## mistralnemo_fp64 ({vocab:,} vocab, {len(langs):,} languages): "
         f"{len(flagged)} flagged rows\n"]
    if flagged:
        L.append("| lang | estimated tokens | script |")
        L.append("|---|---|---|")
        for l, est in flagged:
            L.append(f"| {l} | {est} | {script.get(l, '?')} |")
    L.append("")

    os.makedirs(os.path.dirname(out_md), exist_ok=True)
    with open(out_md, "w") as f:
        f.write("\n".join(L) + "\n")
    print(f"Wrote {out_md}")
    print(f"{len(flagged)} of {len(langs)} rows flagged")
    return out_md


if __name__ == "__main__":
    run()
