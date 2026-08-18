"""Do the published DeepSeek3.2 and Qwen3 cells reproduce from their recorded predictions?

The co-author's Drive folder holds a y_pred.txt per variant, one label per line
aligned to the full GlotLID-C test file. tab:lid_main's rows for those variants
were computed from them, on all 45,627,279 lines (the table caption states that
the carried-over rows use the full file, while the UniLID and calibrated rows use
the 45,377,279-line scored pool).

This recomputes those cells from the recorded predictions. It matters before the
variants are corrected and re-evaluated: without it, a difference between a new
corrected number and the published one could be the correction OR a difference
between this repository's evaluation and the one that produced the table, and
those two would be indistinguishable. Reproducing the published cell first rules
the second out.

  python -m analysis.variant_recorded_preds -o outputs/rerelease/variant_recorded_preds.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
from analysis.config import TEST_FILE, TOTAL_LINES
from analysis.full_test_eval import _parse_line
from analysis.metrics import compute_metrics

DRIVE = Path("/capstor/scratch/cscs/cmeister747/unilid_analysis/drive_models")
# label -> published tab:lid_main GlotLID-C macro F1 cell, for comparison
VARIANTS = {
    "deepseek_v3.2": {"pred": DRIVE / "deepseek_v3.2_glotlid_y_pred.txt",
                      "published_f1": 0.909, "published_fpr": 2.08e-5},
    "qwen3_8b": {"pred": DRIVE / "qwen3_8b_glotlid_y_pred.txt",
                 "published_f1": 0.904, "published_fpr": 2.55e-5},
}
# The published cells are printed to three decimals, so a reproduction is
# accepted when it rounds to the same value.
ROUNDING = 3


def gold_labels() -> list[str]:
    out = []
    with open(TEST_FILE) as f:
        for line in f:
            out.append(_parse_line(line)[0])
    if len(out) != TOTAL_LINES:
        raise RuntimeError(f"{len(out):,} gold labels, expected {TOTAL_LINES:,}")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-o", "--output", default=None)
    a = ap.parse_args(argv)

    print(f"reading gold labels from {TEST_FILE}...", flush=True)
    gold = np.array(gold_labels(), dtype=object)
    print(f"  {len(gold):,} lines, {len(set(gold.tolist())):,} labels", flush=True)

    out = {}
    for name, spec in VARIANTS.items():
        path = spec["pred"]
        if not path.is_file():
            raise FileNotFoundError(f"{path} missing")
        print(f"\n{name}: reading {path.name}...", flush=True)
        pred = np.array(path.read_text().splitlines(), dtype=object)
        if len(pred) != len(gold):
            raise RuntimeError(f"{len(pred):,} predictions against {len(gold):,} "
                               f"gold labels")
        m = compute_metrics(gold, pred)
        rep = {"macro_f1": m["macro_f1"], "macro_fpr": m["macro_fpr"],
               "accuracy": m["accuracy"], "n_lines": len(pred),
               "published_f1": spec["published_f1"],
               "published_fpr": spec["published_fpr"]}
        rep["f1_reproduces"] = (round(m["macro_f1"], ROUNDING)
                                == round(spec["published_f1"], ROUNDING))
        out[name] = rep
        print(f"  macro F1 {m['macro_f1']:.4f} (published {spec['published_f1']:.3f})"
              f"  {'REPRODUCES' if rep['f1_reproduces'] else 'DIFFERS'}")
        print(f"  macro FPR {m['macro_fpr']:.3e} (published {spec['published_fpr']:.3g})")
        print(f"  accuracy {m['accuracy']:.4f}")
        del pred

    ok = all(v["f1_reproduces"] for v in out.values())
    print(f"\n{'PASS' if ok else 'FAIL'}: "
          f"{sum(v['f1_reproduces'] for v in out.values())}/{len(out)} published "
          f"macro F1 cells reproduce from the recorded predictions")
    if a.output:
        Path(a.output).write_text(json.dumps(out, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
