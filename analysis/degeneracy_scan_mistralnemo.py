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

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "UNILID"))
from unilid.constants import SPECIAL_TOKENS  # noqa: E402
from unilid.model_io import load_unilid_raw  # noqa: E402
import json  # noqa: E402

LANG_DIAGNOSTIC_CSV = "outputs/diagnostic/lang_diagnostic.csv"


def run(out_md: str = DEGENERACY_OUT_MD,
        model_path: str = PACKED_MODEL_PATH) -> str:
    """model_path defaults to the released Mistral-Nemo container; a corrected
    run passes both a non-default model AND a non-default out_md so the
    released record is never overwritten. LANG_DIAGNOSTIC_CSV stays shared for
    any model: only its `script` column (the writing system per language, a
    corpus-metadata fact, not a weight-derived one) is read here."""
    if model_path != PACKED_MODEL_PATH and out_md == DEGENERACY_OUT_MD:
        raise RuntimeError(
            f"model {model_path} is not the released container but out_md is "
            f"the released record ({DEGENERACY_OUT_MD}); pass an out_md under "
            f"that model's own output root")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"packed model missing: {model_path}")
    if not os.path.exists(LANG_DIAGNOSTIC_CSV):
        raise FileNotFoundError(f"language diagnostic table missing: {LANG_DIAGNOSTIC_CSV}")

    diag = pd.read_csv(LANG_DIAGNOSTIC_CSV)
    script = dict(zip(diag.lang, diag.script))

    W, langs, _ = _load_model_data(model_path)
    vocab = W.shape[1]

    # Estimated-token counts run over REAL columns only. The original scan
    # counted entries above the whole-row minimum, which is the plateau only
    # while the four specials sit at log-prob 0.0 (the defect); in a corrected
    # container the specials hold the row MINIMUM instead, every real entry is
    # "above minimum", and the whole-row scan flags nothing. Restricting to
    # real columns measures the same thing (entries above the plateau) in both
    # generations. Against the released record this shifts each count down by
    # exactly 4 (the specials no longer counted as estimated).
    tok, _, _raw_langs = load_unilid_raw(model_path)
    text = tok if isinstance(tok, str) else tok.decode("utf-8")
    tokens = [t for t, _score in json.loads(text)["model"]["vocab"]]
    if len(tokens) != vocab:
        raise RuntimeError(
            f"{model_path}: tokenizer vocabulary has {len(tokens)} entries but "
            f"the weight matrix has {vocab} columns")
    spec = set(SPECIAL_TOKENS.values())
    special_idx = [i for i, t in enumerate(tokens) if t in spec]
    if len(special_idx) != len(SPECIAL_TOKENS):
        raise RuntimeError(
            f"{model_path}: found {len(special_idx)} special columns, expected "
            f"{len(SPECIAL_TOKENS)}")
    real_idx = np.setdiff1d(np.arange(vocab), np.array(special_idx))

    flagged = []
    for i in range(len(langs)):
        row = np.asarray(W[i], dtype=np.float64)[real_idx]
        est = int(len(real_idx) - (row == row.min()).sum())
        if est < MIN_ESTIMATED:
            flagged.append((langs[i], est))

    L = [f"# Degenerate-row scan: mistralnemo_fp64 (fewer than {MIN_ESTIMATED} "
         f"estimated tokens)\n",
         f"Model: {model_path}\n",
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
    import argparse
    ap = argparse.ArgumentParser(description="degenerate-row scan, Mistral-Nemo")
    ap.add_argument("--model", default=PACKED_MODEL_PATH,
                    help="packed .unilid container (default: the released one)")
    ap.add_argument("--out-md", default=DEGENERACY_OUT_MD,
                    help="output markdown path; mandatory non-default when "
                         "--model is non-default")
    a = ap.parse_args()
    run(out_md=a.out_md, model_path=a.model)
