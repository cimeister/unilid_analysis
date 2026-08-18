"""The GlotLID-C cells of tab:lid_main, released against corrected.

full_test_eval.py prints macro F1 and accuracy but not macro FPR, which the
paper's Table 1 also reports. Both are recomputed here from the stored
predictions rather than rescored, so this is cheap and traceable to the memmaps
the two full-pool runs wrote.

Scope: the base (uncalibrated) UniLID row only. The calibrated row needs the
gated predictions, which depend on the floor-c pass and the thresholds.

  python -m analysis.corrected_lid_main_cells -o outputs/rerelease/lid_main_cells.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
from analysis.metrics import compute_metrics

RELEASED_SCRATCH = "/capstor/scratch/cscs/cmeister747/unilid_analysis/full_test_eval"
CORRECTED_SCRATCH = ("/capstor/scratch/cscs/cmeister747/unilid_analysis/"
                     "full_test_eval_corrected")


def cells(scratch: str, tag: str) -> dict:
    y = np.asarray(np.load(f"{scratch}/y_true.npy", mmap_mode="r"))
    p = np.asarray(np.load(f"{scratch}/pred_baseline.npy", mmap_mode="r"))
    kept = y >= 0
    yk, pk = y[kept], p[kept]
    m = compute_metrics(yk, pk)
    out = {"tag": tag, "n_lines": int(kept.sum()),
           "macro_f1": m["macro_f1"], "macro_fpr": m["macro_fpr"],
           "accuracy": m["accuracy"]}
    print(f"{tag:10} n={out['n_lines']:,}  macro F1 {out['macro_f1']:.4f}  "
          f"macro FPR {out['macro_fpr']:.3e}  accuracy {out['accuracy']:.4f}",
          flush=True)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--output", default=None)
    a = ap.parse_args(argv)
    rel = cells(RELEASED_SCRATCH, "released")
    cor = cells(CORRECTED_SCRATCH, "corrected")
    print(f"\ntab:lid_main UniLID row, GlotLID-C columns:")
    print(f"  F1  {rel['macro_f1']:.3f} -> {cor['macro_f1']:.3f}")
    print(f"  FPR {rel['macro_fpr']:.2e} -> {cor['macro_fpr']:.2e}")
    out = {"released": rel, "corrected": cor,
           "delta_macro_f1": cor["macro_f1"] - rel["macro_f1"],
           "delta_macro_fpr": cor["macro_fpr"] - rel["macro_fpr"]}
    if a.output:
        Path(a.output).write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
