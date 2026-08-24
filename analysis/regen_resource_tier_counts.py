"""Regenerate the model-dependent columns of the paper's resource-tier table
(paper/submission.tex, tab:resource-tier) on the scored pool.

Two outputs, from two different inputs:

  tables/resource_tier_ntest.md
      The $N_{\\text{test}}$ column, from the per-language CSV
      analysis/paper_eval.py wrote for THIS run's model
      (<out-root>/diagnostic/paper_eval_per_lang_f1_fullpool.csv: camera-ready E1
      artifact; per-language training count N and scored-pool support). The
      published column was computed on the full 45,627,279-line test file; the
      table's F1 cells reproduce on the 45,377,279-line scored pool (E4), so the
      counts are re-derived on the same pool. Format and content unchanged from
      the 2026-08-09 camera-ready pass: a default run reproduces that file
      byte for byte.

  tables/resource_tier_fpr.md   (added 2026-08-24)
      The remaining four model-dependent columns of tab:resource-tier -- UniLID
      F1, UniLID FPR, fastText F1, fastText FPR -- in the WITHIN-STRATUM view,
      recomputed from this run's prediction memmaps under <scratch>/.

Why within-stratum, and how that was established (2026-08-24). The paper's
resource-tier F1 columns were already known to be the within-stratum view
(EXPERIMENTS_CHRONOLOGICAL.md, "provenance of the paper's appendix breakdown
tables resolved", 2026-08-07; analysis/paper_breakdowns.py gates that column).
The FPR columns had no recorded provenance and were reproduced here for the
first time. Measured against the RELEASED model's memmaps, the global view (each
language's false positives over the whole 45,377,279-line pool) misses every
published cell by up to an order of magnitude, while the within-stratum macro
FPR -- restrict to lines whose TRUE label is in the tier, then average
FP_L / (N_tier - support_L) over the tier's languages -- reproduces all twelve
published cells (six UniLID, six fastText) to their printed precision. The
micro form (summed FP over summed negatives) agrees with the macro form to the
same printed precision in every cell, so the published cells do not distinguish
them; the macro form is reported because every other F1/FPR pair in this repo's
tables is macro-averaged, and the micro form is printed alongside for the record.

Gates (abort on any mismatch, no substitution):
- the per-language CSV holds exactly 1,940 languages;
- support sums to exactly 45,377,279 (the scored pool);
- per-tier language counts equal the published 56/40/458/526/398/462;
- (FPR part) the kept-line count of <scratch>/y_true.npy equals 45,377,279;
- (FPR part) ALIGNMENT: per-language F1 and false-positive counts recomputed
  from the memmaps equal the per-language CSV's own f1_*/fp_* columns. This ties
  the memmaps and the CSV to the same model, so a mismatched pair aborts instead
  of producing a table that mixes two models' halves.
The first three are corpus properties (the label inventory, the test pool and the
per-language TRAINING counts), not model properties, so they bind for every
model. The alignment gate is internal to the run and also binds for every model.

Published-cell reproduction gate: the recomputed within-stratum F1/FPR cells are
compared against paper/submission.tex's published tab:resource-tier cells. That
comparison BINDS THE RELEASED MODEL ONLY (a mismatch exits nonzero and the .md
says so). Under any other --model it is computed and reported in full but is
INFORMATIONAL: those published cells are measurements of the released model, so
a difference is the expected outcome of a cross-model comparison, not a
reproduction failure. Precedent and wording: analysis/paper_breakdowns.py
(_cross_model_message / _breakdowns_exit_code).

Usage:
    python -m analysis.regen_resource_tier_counts
    python -m analysis.regen_resource_tier_counts \\
        --model  /capstor/scratch/.../corrected/glotlidc_corrected.unilid \\
        --scratch-dir /capstor/scratch/.../full_test_eval_corrected \\
        --out-dir outputs_corrected_round
"""
# ---------------------------------------------------------------------------
# INPUT INVENTORY (2026-08-24; the classification is what a non-default --model
# changes and what it must not).
#
# (a) MODEL-DERIVED -- must come from the run's own scratch / output root, and must
#     abort naming the artifact when it is absent there:
#       - <out-root>/diagnostic/paper_eval_per_lang_f1_fullpool.csv. Its f1_*/fp_*
#         columns are this model's; a released-model copy would put released
#         numbers into a corrected model's table. Taken from the same --out-dir
#         this script writes to, never from outputs/.
#       - <scratch>/y_true.npy
#       - <scratch>/pred_baseline.npy, <scratch>/pred_fasttext.npy.
#         pred_fasttext.npy is an EXTERNAL model's predictions, but it is still a
#         per-line array that has to be positionally aligned with THIS run's
#         y_true, so it is required in the run's own scratch root, never borrowed.
#
# (b) CONFIG CONSTANTS: POOL, TIERS, EXPECTED_NLANG, EXPECTED_NROWS. PAPER_CELLS is
#     the exception: those are published RELEASED-model numbers and nothing is ever
#     substituted from them; see the gate note above for when they bind.
#
# NOT READ, here or transitively: any tau CSV, any fingerprint json, the .unilid
# weight file itself (this script scores nothing; it reads saved arrays).
# ---------------------------------------------------------------------------
import argparse
import csv
import os
import sys

import numpy as np
import pandas as pd

from analysis.metric_decomposition import EXPECTED_KEPT, _per_lang_stats
from analysis.model_context import (DEFAULT_OUT_ROOT, add_arguments, resolve,
                                    resolve_out_root)

SRC_REL = "diagnostic/paper_eval_per_lang_f1_fullpool.csv"
OUT_REL = "tables/resource_tier_ntest.md"
OUT_FPR_REL = "tables/resource_tier_fpr.md"

# The default-root spellings, kept as module constants because they are the paths
# the 2026-08-09 camera-ready pass recorded.
SRC = os.path.join(DEFAULT_OUT_ROOT, SRC_REL)
OUT = os.path.join(DEFAULT_OUT_ROOT, OUT_REL)
OUT_FPR = os.path.join(DEFAULT_OUT_ROOT, OUT_FPR_REL)

POOL = 45_377_279
EXPECTED_NROWS = 1940
TIERS = [("<500", 0, 500), ("500-1k", 500, 1_000), ("1k-12k", 1_000, 12_000),
         ("12k-18k", 12_000, 18_000), ("18k-35k", 18_000, 35_000),
         ("35k+", 35_000, None)]
EXPECTED_NLANG = [56, 40, 458, 526, 398, 462]

if POOL != EXPECTED_KEPT:
    raise RuntimeError(
        f"POOL ({POOL:,}) and analysis.metric_decomposition.EXPECTED_KEPT "
        f"({EXPECTED_KEPT:,}) disagree about the scored pool")

# The two systems tab:resource-tier prints, as (column name, prediction memmap).
SYSTEMS = [("UniLID", "baseline"), ("fastText", "fasttext")]

# Published paper/submission.tex tab:resource-tier, read 2026-08-24:
# tier -> (UniLID F1, UniLID FPR, fastText F1, fastText FPR).
PAPER_CELLS = {
    "<500":    (0.871, 7.2e-5,  0.915, 1.15e-4),
    "500-1k":  (0.975, 1.5e-5,  0.964, 1.9e-5),
    "1k-12k":  (0.990, 8.0e-6,  0.979, 8.0e-6),
    "12k-18k": (0.997, 2.0e-6,  0.986, 1.0e-5),
    "18k-35k": (0.992, 7.0e-6,  0.981, 1.6e-5),
    "35k+":    (0.958, 5.3e-5,  0.942, 9.1e-5),
}
# MAGIC NUMBERS (defined here 2026-08-24, both for the published-cell comparison
# only, and neither used anywhere else):
#   F1_GATE_TOL      absolute, and the same 0.005 analysis/paper_breakdowns.py
#                    pre-registered for the same published F1 column.
#   FPR_GATE_REL_TOL relative. The published FPR cells are printed at one or two
#                    significant figures (8.0e-6, 2.0e-6, 7.0e-6 against measured
#                    8.174e-6, 1.925e-6, 6.718e-6), so an absolute or
#                    significant-figure test cannot express "agrees at printed
#                    precision". The largest relative gap over the twelve released
#                    cells is 4.0%; 5% is the smallest round bound above it.
F1_GATE_TOL = 0.005
FPR_GATE_REL_TOL = 0.05


def _tier_of(n: int) -> str:
    for name, lo, hi in TIERS:
        if n >= lo and (hi is None or n < hi):
            return name
    raise RuntimeError(f"N={n} did not match any tier")


def _require(path: str, what: str) -> str:
    if not os.path.exists(path):
        sys.exit(f"ABORT: required {what} missing: {path}")
    return path


def run_ntest(src: str, out: str) -> list:
    """The N_test column. Byte-for-byte the 2026-08-09 output under the defaults."""
    rows = list(csv.DictReader(open(src)))
    if len(rows) != EXPECTED_NROWS:
        sys.exit(f"ABORT: expected 1,940 languages in {src}, found {len(rows)}")
    total = sum(int(r["support"]) for r in rows)
    if total != POOL:
        sys.exit(f"ABORT: support sums to {total}, expected {POOL}")

    lines = ["# resource-tier N_test on the scored pool (45,377,279 lines)", "",
             f"Source: {src}", "", "| tier | n_lang | N_test |", "|---|---|---|"]
    for (name, lo, hi), expect in zip(TIERS, EXPECTED_NLANG):
        sel = [r for r in rows if int(r["N"]) >= lo and (hi is None or int(r["N"]) < hi)]
        if len(sel) != expect:
            sys.exit(f"ABORT: tier {name} has {len(sel)} languages, published table has {expect}")
        lines.append(f"| {name} | {len(sel)} | {sum(int(r['support']) for r in sel)} |")

    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    return lines


def run_fpr(src: str, scratch_dir: str, out: str, ctx) -> bool:
    """The UniLID/fastText F1 and FPR columns, within-stratum. Returns whether the
    published-cell comparison passed (meaningful only for the released model)."""
    df = pd.read_csv(src)
    if len(df) != EXPECTED_NROWS:
        sys.exit(f"ABORT: expected 1,940 languages in {src}, found {len(df)}")
    n_lang = len(df)

    y = np.asarray(np.load(_require(os.path.join(scratch_dir, "y_true.npy"),
                                    "true-label memmap"), mmap_mode="r"))
    kept = y >= 0
    n_kept = int(kept.sum())
    if n_kept != POOL:
        sys.exit(f"ABORT: {scratch_dir}/y_true.npy has {n_kept:,} kept lines, "
                 f"expected {POOL:,}")
    yk = y[kept]

    preds = {}
    for _name, cfg in SYSTEMS:
        path = _require(os.path.join(scratch_dir, f"pred_{cfg}.npy"),
                        f"prediction memmap for config {cfg!r}")
        preds[cfg] = np.asarray(np.load(path, mmap_mode="r"))[kept]

    # ALIGNMENT GATE: these memmaps and this CSV must be the same model's.
    for _name, cfg in SYSTEMS:
        _p, _r, f1, _tp, fp, _fn = _per_lang_stats(preds[cfg], yk, n_lang)
        d_f1 = float(np.abs(f1 - df[f"f1_{cfg}"].values).max())
        d_fp = float(np.abs(fp - df[f"fp_{cfg}"].values).max())
        if d_f1 > 1e-9 or d_fp != 0:
            sys.exit(f"ABORT: alignment gate failed for config {cfg!r}: "
                     f"per-language F1/FP recomputed from {scratch_dir} differ from "
                     f"{src} (max |dF1| = {d_f1:.3e}, max |dFP| = {d_fp:.0f}). The "
                     "prediction memmaps and the per-language CSV are not the same "
                     "model's; refusing to build a table from a mixed pair.")

    tier_of = np.array([_tier_of(int(v)) for v in df.N.values])
    cells, extra = {}, {}
    for name, lo, hi in TIERS:
        idx = np.where(tier_of == name)[0]
        m = np.isin(yk, idx)
        n_tier = int(m.sum())
        row, xrow = {"n_lang": len(idx), "n_test": n_tier}, {}
        for col, cfg in SYSTEMS:
            _p, _r, f1, tp, fp, fn = _per_lang_stats(preds[cfg][m], yk[m], n_lang)
            neg = n_tier - (tp + fn)[idx]
            row[f"{col}_f1"] = float(f1[idx].mean())
            row[f"{col}_fpr"] = float(np.mean(fp[idx] / neg))
            xrow[f"{col}_fpr_micro"] = float(fp[idx].sum() / neg.sum())
        cells[name], extra[name] = row, xrow

    for (name, _lo, _hi), expect in zip(TIERS, EXPECTED_NLANG):
        if cells[name]["n_lang"] != expect:
            sys.exit(f"ABORT: tier {name} has {cells[name]['n_lang']} languages, "
                     f"published table has {expect}")
    if sum(c["n_test"] for c in cells.values()) != POOL:
        sys.exit("ABORT: tier N_test does not sum to the scored pool")

    binding = ctx.is_default_model
    gate_rows, mismatches = [], []
    for name, _lo, _hi in TIERS:
        c, p = cells[name], PAPER_CELLS[name]
        checks = [("UniLID F1", c["UniLID_f1"], p[0], "f1"),
                  ("UniLID FPR", c["UniLID_fpr"], p[1], "fpr"),
                  ("fastText F1", c["fastText_f1"], p[2], "f1"),
                  ("fastText FPR", c["fastText_fpr"], p[3], "fpr")]
        for what, ours, pub, kind in checks:
            if kind == "f1":
                ok = abs(ours - pub) <= F1_GATE_TOL
                fmt = f"{ours:.4f}", f"{pub:.3f}", f"{ours - pub:+.4f}"
            else:
                ok = abs(ours - pub) <= FPR_GATE_REL_TOL * pub
                fmt = f"{ours:.4e}", f"{pub:.3g}", f"{(ours - pub) / pub:+.1%}"
            if not ok:
                mismatches.append(f"{name}/{what}")
            gate_rows.append([name, what, fmt[0], fmt[1], fmt[2],
                              "OK" if ok else "MISMATCH"])
    passed = not mismatches

    md = ["# resource-tier F1 and FPR on the scored pool "
          f"({POOL:,} lines), within-stratum view", ""]
    if not binding:
        md += [f"**NON-DEFAULT MODEL RUN.** Every number below was computed from "
               f"`{ctx.model_path}` and its predictions under `{ctx.scratch_dir}`, "
               f"not from the released model (`{ctx.default_model_path}`), and must "
               f"not be read as a restatement of the released model's table.", ""]
    md += [f"Sources: {src} (label set, training counts, alignment gate); "
           f"{scratch_dir}/y_true.npy, "
           + ", ".join(f"{scratch_dir}/pred_{c}.npy" for _n, c in SYSTEMS) + ".", "",
           "View: examples restricted to true labels inside the tier "
           "(cross-tier false positives excluded). F1 is the mean per-language F1 "
           "over the tier's languages; FPR is the mean per-language "
           "FP_L / (N_test - support_L) over the same languages.", "",
           "| tier | n_lang | N_test | UniLID F1 | UniLID FPR | fastText F1 | "
           "fastText FPR |", "|---|---|---|---|---|---|---|"]
    for name, _lo, _hi in TIERS:
        c = cells[name]
        md.append(f"| {name} | {c['n_lang']} | {c['n_test']:,} | "
                  f"{c['UniLID_f1']:.4f} | {c['UniLID_fpr']:.4e} | "
                  f"{c['fastText_f1']:.4f} | {c['fastText_fpr']:.4e} |")

    md += ["", "## Micro form, for the record (not the reported cells)", "",
           "Summed false positives over summed negatives, same restricted lines.",
           "", "| tier | UniLID FPR (micro) | fastText FPR (micro) |",
           "|---|---|---|"]
    for name, _lo, _hi in TIERS:
        x = extra[name]
        md.append(f"| {name} | {x['UniLID_fpr_micro']:.4e} | "
                  f"{x['fastText_fpr_micro']:.4e} |")

    head = ("## Reproduction gate against paper/submission.tex, tab:resource-tier"
            if binding else
            "## INFORMATIONAL comparison against paper/submission.tex, "
            "tab:resource-tier (NOT a gate)")
    md += ["", head, ""]
    if not binding:
        md += ["Those published cells are measurements of the RELEASED model and "
               f"this run scored `{ctx.model_path}`, so a difference here is an "
               "expected cross-model difference, not a regression and not a "
               "reproduction failure. The comparison is computed and reported in "
               "full, it withholds nothing, and it does not set the exit code.", ""]
    md += [f"Tolerances: F1 {F1_GATE_TOL} absolute, FPR "
           f"{FPR_GATE_REL_TOL:.0%} relative (the published FPR cells are printed "
           "at one or two significant figures).", "",
           "| tier | cell | ours | published | diff | status |",
           "|---|---|---|---|---|---|"]
    md += ["| " + " | ".join(str(v) for v in r) + " |" for r in gate_rows]
    md += [""]
    if binding:
        md.append("Reproduction gate: " + ("PASSED." if passed else
                  f"FAILED for {mismatches}; do not publish."))
    else:
        md.append("Comparison outcome (informational): "
                  + ("every cell happens to agree with the published table."
                     if passed else f"cells {mismatches} differ from the published "
                     "table, as expected for a different model."))

    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))
    return passed


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Regenerate tab:resource-tier's N_test and F1/FPR columns "
                    "on the scored pool.")
    add_arguments(ap)
    ap.add_argument("--out-dir", dest="out_dir", default=None,
                    help="report root (default: outputs/); required when --model "
                         "is not the released model")
    ap.add_argument("--part", choices=["all", "ntest", "fpr"], default="all")
    args = ap.parse_args(argv)

    purpose = "resource-tier N_test / FPR regeneration"
    ctx = resolve(args.model_path, args.scratch_dir, purpose=purpose)
    out_root = resolve_out_root(ctx, args.out_dir, purpose=purpose)
    if not ctx.is_default_model:
        print(f"resource-tier regeneration against {ctx.describe()}\n"
              f"  reports {out_root}", flush=True)

    src = _require(os.path.join(out_root, SRC_REL), "per-language E1 CSV")
    passed = True
    if args.part in ("all", "ntest"):
        run_ntest(src, os.path.join(out_root, OUT_REL))
    if args.part in ("all", "fpr"):
        print()
        passed = run_fpr(src, ctx.scratch_dir,
                         os.path.join(out_root, OUT_FPR_REL), ctx)
    return 0 if (passed or not ctx.is_default_model) else 1


if __name__ == "__main__":
    sys.exit(main())
