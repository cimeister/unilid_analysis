"""Label audit for the largest floor-21 residual pair (plan B2): pnt_Grek <- ell_Grek.

Samples AUDIT_N of the test lines labeled ell_Grek that floor-21 still predicts as
pnt_Grek (2,644 such lines per Exp 24) and writes them out verbatim for manual
judgment: how much of this residual is model error versus corpus label noise (Pontic
Greek text inside the ell_Grek corpus is linguistically plausible). The answer bounds
what any decision-margin method can recover on this pair.

Deterministic sample (AUDIT_SEED); texts are pulled in one streaming pass over the
test file and each line's label string is validated against y_true.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from analysis.config import TEST_FILE, TOTAL_LINES
from analysis.full_test_eval import SCRATCH_DIR, _parse_line

PRF_CSV = "outputs/diagnostic/full_test_per_lang_prf.csv"
OUT_MD = "outputs/tables/label_audit_pnt_ell.md"
PRED_LANG = "pnt_Grek"
TRUE_LANG = "ell_Grek"
EXPECTED_RESIDUAL = 2_644          # recorded in outputs/tables/metric_decomposition.md
AUDIT_N = 50
AUDIT_SEED = 0


def run(out_md: str = OUT_MD) -> str:
    prf = pd.read_csv(PRF_CSV)
    langs = prf.lang.tolist()
    idx_pred, idx_true = langs.index(PRED_LANG), langs.index(TRUE_LANG)

    y = np.asarray(np.lib.format.open_memmap(
        os.path.join(SCRATCH_DIR, "y_true.npy"), mode="r"))
    pf = np.asarray(np.lib.format.open_memmap(
        os.path.join(SCRATCH_DIR, "pred_floor21.npy"), mode="r"))
    pb = np.asarray(np.lib.format.open_memmap(
        os.path.join(SCRATCH_DIR, "pred_baseline.npy"), mode="r"))
    lines = np.where((pf == idx_pred) & (y == idx_true))[0]
    if len(lines) != EXPECTED_RESIDUAL:
        raise RuntimeError(f"{PRED_LANG}<-{TRUE_LANG} residual count {len(lines)} != "
                           f"recorded {EXPECTED_RESIDUAL}; memmap state changed")
    rng = np.random.default_rng(AUDIT_SEED)
    sample = np.sort(rng.choice(lines, size=AUDIT_N, replace=False))
    also_baseline = int((pb[sample] == idx_pred).sum())

    want = set(sample.tolist())
    texts = {}
    with open(TEST_FILE) as fh:
        for i in range(TOTAL_LINES):
            line = fh.readline()
            if i not in want:
                continue
            label, text = _parse_line(line)
            if label != TRUE_LANG:
                raise RuntimeError(f"line {i}: label {label!r} != {TRUE_LANG}")
            texts[i] = text
    if len(texts) != AUDIT_N:
        raise RuntimeError(f"collected {len(texts)} texts for {AUDIT_N} sampled lines")

    L = [f"# Label audit: {AUDIT_N} of the {EXPECTED_RESIDUAL:,} lines labeled "
         f"{TRUE_LANG} that floor-21 predicts as {PRED_LANG}\n",
         f"Deterministic sample (seed {AUDIT_SEED}); {also_baseline}/{AUDIT_N} of the "
         f"sampled lines are also predicted {PRED_LANG} by the baseline. For manual "
         f"judgment: mark each line as (a) standard Modern Greek (model error), "
         f"(b) Pontic Greek or Pontic-influenced (label noise for this pair), or "
         f"(c) unverifiable/too short.\n"]
    for i in sample.tolist():
        L.append(f"**line {i}**")
        L.append("```")
        L.append(texts[i])
        L.append("```")
    os.makedirs(os.path.dirname(out_md), exist_ok=True)
    with open(out_md, "w") as f:
        f.write("\n".join(L) + "\n")
    print(f"Wrote {out_md} ({AUDIT_N} lines; {also_baseline} also flipped by baseline)")
    return out_md


if __name__ == "__main__":
    run()
