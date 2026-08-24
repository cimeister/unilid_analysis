"""Accuracy by input length on WiLI-2018, the instrument behind `tab:length_accuracy`.

The published table (`paper/tables/length_accuracy.tex`) reports one metric only:
accuracy in percent, per length bucket, for UniLID and for fastText, plus an
Overall row over all 117,500 test lines. This script reproduces the UniLID
column. There is no fastText WiLI model in the release assets
(`/capstor/scratch/cscs/cmeister747/unilid_analysis/wili_assets/` holds three
.unilid files and the WiLI corpus, nothing else), so the fastText column is
printed as published reference only and is never recomputed here.

Length definition, established by reproducing the six published bucket counts
before any accuracy was computed (see WILI_LENGTH_BINS): length is
`len(raw_line)`, i.e. Unicode code points of the raw line as read from
x_test.txt, with no preprocessing and no stripping. Candidates tested:
  raw chars      7845 / 26652 / 31449 / 29494 / 18142 / 3918  -> EXACT
  utf-8 bytes    2947 / 16851 / 25389 / 32660 / 27363 / 12290 -> off by thousands
  stripped chars identical to raw (no test line has leading/trailing whitespace)
The shortest test line is 140 chars, so nothing falls below the first bucket,
matching the caption's claim that all WiLI samples exceed 100 characters.

Scoring conventions are wili_eval's, imported rather than restated, so the two
instruments cannot drift: a line that preprocesses to empty is kept and scored
wrong; a gold label outside the model's label set is kept and scored wrong.
`analysis/wili_eval.py` stores no per-line predictions in its -o JSON (only
aggregates), and no per-line dump exists under outputs/, so predictions are
recomputed here via wili_eval.predict_all.

Do NOT use analysis.config.LENGTH_BINS here: that is the GlotLID-C binning
[0, 30, 75, 150, 300, inf], a different corpus and a different table.

  python -m analysis.wili_length_accuracy                 # three retrained models
  python -m analysis.wili_length_accuracy --stored --gate # instrument gate
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "UNILID"))

from analysis.wili_eval import WILI_DIR, load_split, out_of_set_labels, predict_all  # noqa: E402

SCRATCH = "/capstor/scratch/cscs/cmeister747/unilid_analysis"

# The three fp64 retrains that replace the defective stored originals.
RETRAINED_MODELS = [
    f"{SCRATCH}/wili_100k_500_fp64.unilid",
    f"{SCRATCH}/deepseek_v3.2_wili_fp64.unilid",
    f"{SCRATCH}/qwen3_8b_wili_fp64.unilid",
]
# The stored (defective) originals, kept for the instrument-reproduction check:
# wili_100k_500.unilid is the model whose aggregate cells reproduce the paper
# exactly (outputs/rerelease/wili_instrument_gate.json).
STORED_MODELS = [
    f"{SCRATCH}/wili_assets/wili_100k_500.unilid",
    f"{SCRATCH}/wili_assets/deepseek_v3.2_wili.unilid",
    f"{SCRATCH}/wili_assets/qwen3_8b_wili.unilid",
]

# Buckets exactly as printed in paper/tables/length_accuracy.tex, as
# (label, lo_inclusive, hi_inclusive) over len(raw_line) in characters.
# hi=None means unbounded. Any line below the first lo would abort (see below).
WILI_LENGTH_BINS = [
    ("101-150", 101, 150),
    ("151-200", 151, 200),
    ("201-300", 201, 300),
    ("301-500", 301, 500),
    ("501-1000", 501, 1000),
    ("1000+", 1001, None),
]
# Sample counts as printed in that table; the length definition is only accepted
# if it reproduces all six exactly.
PUBLISHED_BIN_COUNTS = {
    "101-150": 7845, "151-200": 26652, "201-300": 31449,
    "301-500": 29494, "501-1000": 18142, "1000+": 3918,
}
PUBLISHED_TOTAL = 117500
# Published accuracy cells, percent, from the same table.
PUBLISHED_ACC_UNILID = {
    "101-150": 93.10, "151-200": 94.17, "201-300": 95.86,
    "301-500": 96.78, "501-1000": 96.53, "1000+": 96.53, "Overall": 95.65,
}
PUBLISHED_ACC_FASTTEXT = {
    "101-150": 90.73, "151-200": 92.56, "201-300": 94.58,
    "301-500": 96.03, "501-1000": 96.25, "1000+": 96.30, "Overall": 94.54,
}
# Cells are published to two decimal places in percent, so a reproduction is
# accepted within half of that last digit.
ACC_CELL_TOL_PCT = 0.005


def bin_of(length: int) -> str:
    for label, lo, hi in WILI_LENGTH_BINS:
        if length >= lo and (hi is None or length <= hi):
            return label
    raise ValueError(
        f"length {length} falls below the first bucket "
        f"({WILI_LENGTH_BINS[0][1]}); the published table has no bucket for it")


def bin_texts(texts):
    """Assign every text to a bucket and verify the published counts."""
    assign = [bin_of(len(t)) for t in texts]
    counts = {label: 0 for label, _, _ in WILI_LENGTH_BINS}
    for b in assign:
        counts[b] += 1
    if len(texts) == PUBLISHED_TOTAL and counts != PUBLISHED_BIN_COUNTS:
        bad = {k: (counts[k], PUBLISHED_BIN_COUNTS[k])
               for k in PUBLISHED_BIN_COUNTS if counts[k] != PUBLISHED_BIN_COUNTS[k]}
        raise SystemExit(
            "length binning does not reproduce paper/tables/length_accuracy.tex; "
            f"got vs published: {bad}")
    return assign, counts


def accuracy_by_bin(assign, gold, preds):
    hit = {label: 0 for label, _, _ in WILI_LENGTH_BINS}
    tot = {label: 0 for label, _, _ in WILI_LENGTH_BINS}
    for b, g, p in zip(assign, gold, preds):
        tot[b] += 1
        if g == p:
            hit[b] += 1
    rows = {}
    for label, _, _ in WILI_LENGTH_BINS:
        rows[label] = {"samples": tot[label],
                       "correct": hit[label],
                       "accuracy_pct": 100.0 * hit[label] / tot[label]
                       if tot[label] else float("nan")}
    n_hit, n_tot = sum(hit.values()), sum(tot.values())
    rows["Overall"] = {"samples": n_tot, "correct": n_hit,
                       "accuracy_pct": 100.0 * n_hit / n_tot}
    return rows


def markdown_table(rows, gate=None):
    labels = [l for l, _, _ in WILI_LENGTH_BINS] + ["Overall"]
    head = "| Length (chars) | Samples | Acc. (%) |"
    sep = "| --- | ---: | ---: |"
    if gate:
        head += " Published (%) | Delta (pp) |"
        sep += " ---: | ---: |"
    out = [head, sep]
    for l in labels:
        r = rows[l]
        line = f"| {l} | {r['samples']:,} | {r['accuracy_pct']:.2f} |"
        if gate:
            pub = PUBLISHED_ACC_UNILID[l]
            line += f" {pub:.2f} | {r['accuracy_pct'] - pub:+.2f} |"
        out.append(line)
    return "\n".join(out)


def run_model(model_path, texts, gold, assign, wili_dir, split):
    from unilid.model_io import UnilidModel

    if not Path(model_path).is_file():
        raise SystemExit(f"missing model file: {model_path}")
    print(f"\n=== {model_path}", flush=True)
    model = UnilidModel(model_path, calibrated=False)
    out_of_set = out_of_set_labels(model, gold)
    preds, n_empty = predict_all(model, texts)
    rows = accuracy_by_bin(assign, gold, preds)
    return {
        "model": os.path.abspath(model_path),
        "split": split,
        "wili_dir": wili_dir,
        "length_definition": "len(raw_line) in Unicode characters, unpreprocessed",
        "total_samples": len(texts),
        "n_empty_after_preprocess": n_empty,
        "n_gold_labels_absent_from_model": len(out_of_set),
        "bins": rows,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=None,
                    help="model paths; default is the three fp64 retrains")
    ap.add_argument("--stored", action="store_true",
                    help="use the stored (defective) originals instead")
    ap.add_argument("--gate", action="store_true",
                    help="compare each cell against the published UniLID column "
                         "and exit non-zero on a mismatch")
    ap.add_argument("--wili-dir", default=WILI_DIR)
    ap.add_argument("--split", default="test", choices=("test", "train"))
    ap.add_argument("--limit", type=int, default=None,
                    help="smoke test only: score the first N lines. Disables the "
                         "published-count check and writes no output.")
    ap.add_argument("--outdir", default=str(REPO / "outputs" / "rerelease"))
    a = ap.parse_args(argv)

    if a.models and a.stored:
        raise SystemExit("--models and --stored are mutually exclusive")
    models = a.models or (STORED_MODELS if a.stored else RETRAINED_MODELS)

    texts, gold = load_split(a.wili_dir, a.split)
    if a.limit:
        texts, gold = texts[:a.limit], gold[:a.limit]
        print(f"SMOKE TEST: first {len(texts):,} lines only; the published-count "
              f"check is skipped and no JSON is written.", flush=True)
    print(f"WiLI {a.split}: {len(texts):,} lines, {len(set(gold)):,} labels",
          flush=True)

    assign, counts = bin_texts(texts)
    if not a.limit:
        print("length bucket counts reproduce the published table exactly:")
        for label, _, _ in WILI_LENGTH_BINS:
            print(f"  {label:>9}: {counts[label]:>7,}")

    failures = []
    for mp in models:
        res = run_model(mp, texts, gold, assign, a.wili_dir, a.split)
        if a.gate and not a.limit:
            cells = []
            for label in [l for l, _, _ in WILI_LENGTH_BINS] + ["Overall"]:
                got = res["bins"][label]["accuracy_pct"]
                want = PUBLISHED_ACC_UNILID[label]
                ok = abs(got - want) <= ACC_CELL_TOL_PCT
                cells.append({"bin": label, "got_pct": got, "published_pct": want,
                              "delta_pp": got - want, "match": bool(ok)})
            res["gate"] = cells
            res["gate_passed"] = all(c["match"] for c in cells)
            if not res["gate_passed"]:
                failures.append(mp)
        print()
        print(markdown_table(res["bins"], gate=a.gate and not a.limit))
        if a.gate and not a.limit:
            print(f"  gate: {'PASSED' if res['gate_passed'] else 'FAILED'}")
        if not a.limit:
            outdir = Path(a.outdir)
            outdir.mkdir(parents=True, exist_ok=True)
            name = Path(mp).stem
            if a.stored:
                name += "_stored"
            out = outdir / f"wili_length_accuracy_{name}.json"
            out.write_text(json.dumps(res, indent=2))
            print(f"  wrote {out}")

    if failures:
        print("\nGATE FAILED for: " + ", ".join(failures))
        print("This instrument does not reproduce the published cells, so nothing "
              "measured with it can be trusted yet.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
