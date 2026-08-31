"""Scoring-path gate on GlotLID-C: does this repository's scorer reproduce the
recorded per-line predictions of the released generation?

The GlotLID-C full-label-set pass costs ~12 h per model (1,940 rows x 45,627,279
lines), too much to run purely to re-derive a cell the sweep already reproduced
from the recorded prediction files to 7 significant digits
(outputs/rerelease/cld_subset_convention_sweep.json, "validation_metric_core").
What is NOT yet gated on GlotLID-C is the SCORER: that
`UnilidModel(..., calibrated=False).predict_batch(..., forward=False)` in this
environment reproduces the rank-1 label the co-author recorded in
`<model>_y_pred.txt`.

This script gates exactly that, on a strided sample: every STRIDE-th line of the
45,627,279-line test file, scored with the FULL model, compared label-for-label
with the same line of the recorded y_pred file. A single mismatch is reported
with its line index; the exit status is nonzero on any mismatch.

Both files are read in one pass and consumed in lockstep, so the alignment is
positional and cannot drift: eval_glotlid.py writes one prediction per FILTERED
input line, and the released run's filter kept all 45,627,279 lines (the model
covers every gold label in the file), which the line-count check below asserts.
"""
import argparse
import json
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (os.path.join(REPO, "UNILID"), REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from analysis.config import TEST_FILE, TOTAL_LINES  # noqa: E402
from unilid.model_io import UnilidModel  # noqa: E402

BATCH = 10_000


def main():
    p = argparse.ArgumentParser()
    p.add_argument("model")
    p.add_argument("--ypred", required=True)
    p.add_argument("--stride", type=int, default=450)
    p.add_argument("--out", required=True)
    p.add_argument("--max-lines", type=int, default=None,
                   help="smoke-test only: stop after N test lines and skip the "
                        "45,627,279-line and equal-length checks")
    a = p.parse_args()

    for path, what in ((a.model, "model"), (a.ypred, "y_pred"), (TEST_FILE, "test file")):
        if not os.path.exists(path):
            raise SystemExit(f"FATAL: {what} missing at {path}")

    model = UnilidModel(a.model, calibrated=False)

    n_lines = 0
    n_sampled = 0
    n_mismatch = 0
    first_mismatches = []
    buf_t, buf_p, buf_i = [], [], []
    t0 = time.perf_counter()

    def flush():
        nonlocal n_mismatch
        if not buf_t:
            return
        res = model.predict_batch(buf_t, forward=False)
        if len(res) != len(buf_t):
            raise SystemExit("FATAL: scorer returned a short batch")
        for (pred, _t, _s), recorded, idx in zip(res, buf_p, buf_i):
            if pred is None:
                pred = "NONE"
            if pred != recorded:
                n_mismatch += 1
                if len(first_mismatches) < 20:
                    first_mismatches.append(
                        {"line": idx, "mine": pred, "recorded": recorded})
        buf_t.clear(); buf_p.clear(); buf_i.clear()

    with open(TEST_FILE, "r", encoding="utf-8", errors="ignore") as ft, \
         open(a.ypred, "r", encoding="utf-8", errors="ignore") as fp:
        for line, pred_line in zip(ft, fp):
            line = line.strip()
            if not line or not line.startswith("__label__"):
                raise SystemExit(
                    f"FATAL: test line {n_lines} is blank or unlabelled; the "
                    f"positional alignment with y_pred assumes neither occurs")
            n_lines += 1
            if (n_lines - 1) % a.stride:
                continue
            parts = line.split(" ", 1)
            buf_t.append(parts[1] if len(parts) == 2 else "")
            buf_p.append(pred_line.strip())
            buf_i.append(n_lines - 1)
            n_sampled += 1
            if len(buf_t) == BATCH:
                flush()
                print(f"  read {n_lines:,} / sampled {n_sampled:,} / "
                      f"mismatches {n_mismatch} "
                      f"({n_sampled / (time.perf_counter() - t0):,.0f} scored/s)",
                      flush=True)
            if a.max_lines is not None and n_lines >= a.max_lines:
                break
        flush()
        # zip() stops at the shorter file; count whatever is left in each so a
        # length mismatch is reported rather than silently truncating the gate.
        extra_test = 0 if a.max_lines else sum(1 for _ in ft)
        extra_pred = 0 if a.max_lines else sum(1 for _ in fp)

    if a.max_lines:
        print(f"SMOKE MODE: stopped at {n_lines:,} lines; length checks skipped")
    elif extra_test or extra_pred:
        raise SystemExit(
            f"FATAL: test file and y_pred have different lengths "
            f"({extra_test} extra test lines, {extra_pred} extra y_pred lines "
            f"after {n_lines:,} paired lines)")
    if not a.max_lines and n_lines != TOTAL_LINES:
        raise SystemExit(
            f"FATAL: read {n_lines:,} test lines, expected {TOTAL_LINES:,}; "
            f"either the file changed or y_pred is shorter (zip stops at the "
            f"shorter file)")

    out = {
        "model": os.path.realpath(a.model),
        "ypred": os.path.realpath(a.ypred),
        "test_file": os.path.realpath(TEST_FILE),
        "stride": a.stride,
        "test_lines": n_lines,
        "sampled": n_sampled,
        "mismatches": n_mismatch,
        "agreement": 1.0 - n_mismatch / n_sampled if n_sampled else None,
        "first_mismatches": first_mismatches,
        "elapsed_s": time.perf_counter() - t0,
    }
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps({k: v for k, v in out.items()
                      if k != "first_mismatches"}, indent=2))
    if n_mismatch:
        print(f"SCORING-PATH GATE FAIL: {n_mismatch} / {n_sampled} mismatches")
        raise SystemExit(1)
    print(f"SCORING-PATH GATE PASS: {n_sampled:,} / {n_sampled:,} agree")


if __name__ == "__main__":
    main()
