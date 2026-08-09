"""Measure how baseline UniLID errors concentrate on predictions into the two
calibration groups (camera-ready F2; feeds one sentence in section 4 of the paper).

Quantity: on the 45,377,279-line scored pool, the share of baseline false positives
whose PREDICTED language is (a) in the under-18,000-training-sample group, (b) one of
the four high-entropy languages (tau_flat4.csv), and (c) their union. Each share is
reported against the support-based null (the union's share of scored lines by gold
label), since the union spans 1,084 of 1,940 languages and a bare share would invite
a per-label-null misreading (pre-run review finding 5, 2026-08-09).

Inputs (all existing artifacts of record):
- SCRATCH/full_test_eval/y_true.npy, pred_baseline.npy  (int16 .npy files,
  45,627,279 lines; y_true < 0 marks lines outside the scored pool)
- outputs/diagnostic/paper_eval_per_lang_f1_fullpool.csv  (E1; per-language N,
  support, fp_baseline)
- outputs/diagnostic/tau_floor21_gate.csv  (the mechanism's own under-18k group)
- outputs/diagnostic/tau_flat4.csv  (the four high-entropy languages)

Wiring gates, all blocking (abort loudly, no fallback):
G1 language list from _load_model_data has 1,940 entries and matches the CSV's set.
G2 scored-pool size (y_true >= 0) equals EXPECTED_KEPT.
G3 per-language support on the scored pool equals the CSV support column exactly.
G4 per-language baseline false-positive counts equal the CSV fp_baseline column
   exactly, and their total equals the error count (no EMPTY predictions).
G5 tau_flat4.csv has exactly 4 languages, none excluded, each with N >= HEAD_N.
G6 the under-HEAD_N set from the CSV equals the language set of tau_floor21_gate.csv.

Output: outputs/tables/baseline_error_concentration.md
"""
import csv
import os
import sys

import numpy as np

from analysis.config import TOTAL_LINES
from analysis.full_test_margin import HEAD_N
from analysis.metric_decomposition import EXPECTED_KEPT
from analysis.transfer_sweep import _load_model_data

SCRATCH = "/capstor/scratch/cscs/cmeister747/unilid_analysis/full_test_eval"
CSV = "outputs/diagnostic/paper_eval_per_lang_f1_fullpool.csv"
TAU_GATE_CSV = "outputs/diagnostic/tau_floor21_gate.csv"
FLAT4_CSV = "outputs/diagnostic/tau_flat4.csv"
OUT = "outputs/tables/baseline_error_concentration.md"


def main():
    _w, langs, _m = _load_model_data()
    if len(langs) != 1940:
        sys.exit(f"ABORT G1: _load_model_data returned {len(langs)} languages, expected 1,940")
    lang_to_idx = {l: i for i, l in enumerate(langs)}

    with open(CSV) as f:
        rows = list(csv.DictReader(f))
    if set(r["lang"] for r in rows) != set(langs):
        sys.exit("ABORT G1: language set of the E1 CSV does not match _load_model_data")
    n_by_idx = np.zeros(1940, dtype=np.int64)
    support_csv = np.zeros(1940, dtype=np.int64)
    fp_csv = np.zeros(1940, dtype=np.int64)
    for r in rows:
        i = lang_to_idx[r["lang"]]
        n_by_idx[i] = int(r["N"])
        support_csv[i] = int(r["support"])
        fp_csv[i] = int(r["fp_baseline"])

    y = np.load(os.path.join(SCRATCH, "y_true.npy"), mmap_mode="r")
    p = np.load(os.path.join(SCRATCH, "pred_baseline.npy"), mmap_mode="r")
    if y.dtype != np.int16 or p.dtype != np.int16:
        sys.exit(f"ABORT: dtypes {y.dtype}/{p.dtype}, expected int16")
    if y.shape[0] != TOTAL_LINES or p.shape[0] != TOTAL_LINES:
        sys.exit(f"ABORT: memmap lengths {y.shape[0]}/{p.shape[0]}, expected {TOTAL_LINES}")

    scored = y >= 0
    n_scored = int(scored.sum())
    if n_scored != EXPECTED_KEPT:
        sys.exit(f"ABORT G2: scored pool has {n_scored} lines, expected {EXPECTED_KEPT}")

    ys = np.asarray(y[scored])
    ps = np.asarray(p[scored])
    support = np.bincount(ys, minlength=1940)
    if not np.array_equal(support, support_csv):
        bad = int(np.argmax(support != support_csv))
        sys.exit(f"ABORT G3: support mismatch at {langs[bad]}: {support[bad]} vs CSV {support_csv[bad]}")

    err = ps != ys
    n_err = int(err.sum())
    fp = np.bincount(ps[err], minlength=1940)
    if not np.array_equal(fp, fp_csv):
        bad = int(np.argmax(fp != fp_csv))
        sys.exit(f"ABORT G4: fp mismatch at {langs[bad]}: {fp[bad]} vs CSV {fp_csv[bad]}")
    total_fp = int(fp.sum())
    if total_fp != n_err:
        sys.exit(f"ABORT G4: fp total {total_fp} != error count {n_err} "
                 f"(EMPTY predictions present; the denominator convention breaks)")

    with open(FLAT4_CSV) as f:
        flat_rows = list(csv.DictReader(f))
    flat_langs = [r["lang"] for r in flat_rows]
    if len(flat_langs) != 4:
        sys.exit(f"ABORT G5: tau_flat4.csv has {len(flat_langs)} rows, expected 4")
    for r in flat_rows:
        if r.get("excluded", "False") not in ("False", "false", ""):
            sys.exit(f"ABORT G5: flat language {r['lang']} is marked excluded")
        if n_by_idx[lang_to_idx[r["lang"]]] < HEAD_N:
            sys.exit(f"ABORT G5: flat language {r['lang']} has N < {HEAD_N}")
    flat_idx = np.zeros(1940, dtype=bool)
    for l in flat_langs:
        flat_idx[lang_to_idx[l]] = True

    under = n_by_idx < HEAD_N
    with open(TAU_GATE_CSV) as f:
        gate_set = set(r["lang"] for r in csv.DictReader(f))
    under_set = {langs[i] for i in np.where(under)[0]}
    if gate_set != under_set:
        diff = gate_set.symmetric_difference(under_set)
        sys.exit(f"ABORT G6: under-{HEAD_N} set differs from tau_floor21_gate.csv by {sorted(diff)[:5]}")

    union = under | flat_idx
    fp_under = int(fp[under].sum())
    fp_flat = int(fp[flat_idx].sum())
    fp_union = int(fp[union].sum())
    sup_union = int(support[union].sum())

    lines = [
        "# Baseline false-positive concentration into the two calibration groups (F2)",
        "",
        "Instrument: 45,377,279-line scored pool; pred_baseline.npy vs y_true.npy.",
        "Gates G1-G6 passed (support and fp columns match the E1 CSV exactly; fp total",
        "equals the error count; group A matches tau_floor21_gate.csv).",
        "",
        f"- total baseline false positives: {total_fp:,}",
        f"- into the under-{HEAD_N:,} group ({int(under.sum())} languages): "
        f"{fp_under:,} ({fp_under/total_fp:.4f})",
        f"- into the high-entropy four ({', '.join(flat_langs)}): "
        f"{fp_flat:,} ({fp_flat/total_fp:.4f})",
        f"- union ({int(union.sum())} languages): {fp_union:,} ({fp_union/total_fp:.4f})",
        f"- union support on the scored pool (gold lines): {sup_union:,} "
        f"({sup_union/EXPECTED_KEPT:.4f} of the pool)",
        f"- concentration ratio (fp share / support share): "
        f"{(fp_union/total_fp)/(sup_union/EXPECTED_KEPT):.2f}",
    ]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
