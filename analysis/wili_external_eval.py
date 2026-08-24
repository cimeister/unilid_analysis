"""Evaluate a WiLI-trained .unilid model on Tatoeba or UDHR, and gate that
instrument against `tab:tatoeba_udhr_comparison`.

This is the external-benchmark sibling of `analysis/wili_eval.py`, which covers
the in-domain WiLI test set. It deliberately imports `predict_all` and
`out_of_set_labels` from that module rather than re-implementing them, so the
two instruments cannot drift apart. Metrics come from
`analysis.metrics.compute_metrics` (macro F1 averaged over gold languages,
FPR_l = FP_l / (FP_l + TN_l), accuracy), the same call `wili_eval` makes.

`unilid_resources/eval_udhr.py` is NOT used as the instrument: it computes its
own macro F1/FPR inline from a confusion Counter. Only its data-loading
convention is reused (see `load_udhr` below), which is the part the paper's
number depends on.

Conventions, inherited from wili_eval and stated because they change the third
decimal:
- a text that preprocesses to empty is scored as WRONG rather than dropped;
- a gold label outside the model's label set is kept and scored as wrong when
  --scope all, and excluded from the evaluation when --scope model-langs.
Both counts are reported.

Evaluation scope. `paper/submission.tex:1131` states for Tatoeba: "Models are
evaluated on the entire dataset for the subset of languages that are in their
label set. For evaluations with models trained on WiLI, this results in an
evaluation on approximately ~12M samples across 201 languages." That is the
--scope model-langs reading, and it is the default. --scope all is provided so
the alternative reading can be measured rather than assumed.

  python -m analysis.wili_external_eval --bench udhr --stored --gate
  python -m analysis.wili_external_eval --bench tatoeba --stored --gate
  python -m analysis.wili_external_eval --bench tatoeba --fp64
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "UNILID"))

from analysis.metrics import compute_metrics  # noqa: E402
from analysis.wili_eval import (  # noqa: E402
    F1_TOL, FPR_REL_TOL, predict_all, out_of_set_labels,
)

SCRATCH = "/capstor/scratch/cscs/cmeister747/unilid_analysis"

# Tatoeba, from the project's own release asset `tatoeba.zip`. The zip carries
# no README; the split is determined by measurement, not by name. Filtered to
# the stored WiLI model's 235 labels, tatoeba_full.txt yields exactly 201
# languages and 11,848,300 rows, matching submission.tex:1131 ("~12M samples
# across 201 languages"); tatoeba_test.txt yields 197 languages / 2,371,336
# rows and cannot be the published split.
TATOEBA_DIR = f"{SCRATCH}/wili_assets/tatoeba"
TATOEBA_FULL = f"{TATOEBA_DIR}/tatoeba_full.txt"
TATOEBA_TEST = f"{TATOEBA_DIR}/tatoeba_test.txt"
TATOEBA_SPLITS = {"full": TATOEBA_FULL, "test": TATOEBA_TEST}
# fastText line prefix in the Tatoeba dumps: "__label__<iso639-3> <text>".
LABEL_PREFIX = "__label__"

# UDHR, HF dataset cis-lmu/udhr-lid at the revision recorded in
# outputs/tables/external_bench_mapping.md. NOT external_bench/udhr_eval.tsv,
# which is filtered to GlotLID-C's 366 lang_Script labels.
UDHR_CSV = ("/capstor/store/cscs/swissai/a0229/cmeister/huggingface/hub/"
            "datasets--cis-lmu--udhr-lid/snapshots/"
            "6908db2a27c296158da7e69782d15df911652184/udhr-lid.csv")

STORED_MODELS = [f"{SCRATCH}/wili_assets/wili_100k_500.unilid",
                 f"{SCRATCH}/wili_assets/deepseek_v3.2_wili.unilid",
                 f"{SCRATCH}/wili_assets/qwen3_8b_wili.unilid"]
FP64_MODELS = [f"{SCRATCH}/wili_100k_500_fp64.unilid",
               f"{SCRATCH}/deepseek_v3.2_wili_fp64.unilid",
               f"{SCRATCH}/qwen3_8b_wili_fp64.unilid"]

# Published cells of tab:tatoeba_udhr_comparison, \unilid row (the \fasttext row
# is not re-runnable here: no WiLI-trained fastText model exists in this repo).
# The \unilid row of that table is the wili_100k_500 model, the same model that
# carries the WiLI-trained cells of every other WiLI table in the paper.
# The published \unilid row is the wili_100k_500 model (the WiLI-trained model
# behind every other WiLI table in the paper). --gate therefore compares only
# that model, or its _fp64 retrain, against these cells; the DeepSeek and Qwen
# WiLI models have no published cell in this table and are reported ungated.
PUBLISHED_ROW_MODEL = "wili_100k_500"
FP64_SUFFIX = "_fp64"

PUBLISHED = {"tatoeba": {"macro_f1": 0.414, "macro_fpr": 9.61e-4, "n_langs": 201},
             "udhr": {"macro_f1": 0.868, "macro_fpr": 5.88e-4, "n_langs": 142}}

# Lines held in memory at once while streaming Tatoeba (11.8M rows). Predictions
# and gold labels are kept for the whole run (interned, ~8 bytes each); only the
# texts of one chunk are alive. A multiple of wili_eval.BATCH (10,000).
STREAM_CHUNK = 200_000


def load_tatoeba(path: str):
    """Yield (texts, gold) chunks of STREAM_CHUNK lines from a fastText dump."""
    p = Path(path)
    if not p.is_file():
        raise SystemExit(f"missing Tatoeba file: {p}")
    texts, gold = [], []
    with p.open(encoding="utf-8", errors="replace") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.rstrip("\n")
            if not line.startswith(LABEL_PREFIX):
                raise SystemExit(
                    f"{p}:{lineno} does not start with {LABEL_PREFIX!r}: "
                    f"{line[:60]!r}")
            head, _, text = line.partition(" ")
            gold.append(sys.intern(head[len(LABEL_PREFIX):]))
            texts.append(text)
            if len(texts) == STREAM_CHUNK:
                yield texts, gold
                texts, gold = [], []
    if texts:
        yield texts, gold


def load_udhr(path: str):
    """Load UDHR as one (texts, gold) chunk, with eval_udhr.py's conventions.

    keep_default_na/na_filter are load-bearing: "nan" is a real ISO 639-3 code
    (Min Nan Chinese, 58 rows), and the default reader coerces it to a missing
    value, silently dropping a language from the macro average. Gold is the bare
    iso639-3 code, because the WiLI label set is bare codes with no script
    suffix; that matching yields the published 142 shared languages.
    """
    import pandas as pd
    p = Path(path)
    if not p.is_file():
        raise SystemExit(f"missing UDHR csv: {p}")
    df = pd.read_csv(p, keep_default_na=False, na_filter=False)
    for col in ("sentence", "iso639-3"):
        if col not in df.columns:
            raise SystemExit(f"{p} has no column {col!r}; columns are "
                             f"{list(df.columns)}")
    gold = [sys.intern(s.strip()) for s in df["iso639-3"].astype(str)]
    texts = df["sentence"].astype(str).tolist()
    yield texts, gold


def evaluate(model, chunks, scope: str, verbose=True):
    """Score a model over (texts, gold) chunks. Returns a result dict."""
    model_langs = set(model.langs)
    gold_all, pred_all = [], []
    n_empty = n_dropped = n_lines = 0
    out_of_set = set()
    t0 = time.time()
    for texts, gold in chunks:
        n_lines += len(texts)
        if scope == "model-langs":
            keep = [i for i, g in enumerate(gold) if g in model_langs]
            n_dropped += len(gold) - len(keep)
            if len(keep) != len(gold):
                out_of_set.update(g for g in gold if g not in model_langs)
                texts = [texts[i] for i in keep]
                gold = [gold[i] for i in keep]
            if not texts:
                continue
        else:
            out_of_set.update(out_of_set_labels(model, gold, verbose=False))
        preds, n_e = predict_all(model, texts, verbose=False)
        n_empty += n_e
        gold_all.extend(gold)
        pred_all.extend(preds)
        if verbose:
            el = time.time() - t0
            print(f"  {n_lines:,} lines read, {len(gold_all):,} scored, "
                  f"{el:,.0f}s ({len(gold_all)/max(el, 1e-9):,.0f}/s)",
                  flush=True)
    if not gold_all:
        raise SystemExit("no rows survived the scope filter; nothing to score")
    m = compute_metrics(np.array(gold_all, dtype=object),
                        np.array(pred_all, dtype=object))
    return {"scope": scope, "n_lines_in_file": n_lines,
            "total_samples": len(gold_all),
            "n_languages_scored": len(set(gold_all)),
            "n_empty_after_preprocess": n_empty,
            "n_rows_outside_model_label_set": n_dropped,
            "n_gold_labels_outside_model_label_set": len(out_of_set),
            "accuracy": m["accuracy"], "macro_f1": m["macro_f1"],
            "macro_fpr": m["macro_fpr"],
            "elapsed_s": time.time() - t0}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bench", required=True, choices=("tatoeba", "udhr"))
    ap.add_argument("--models", nargs="+", default=None)
    ap.add_argument("--stored", action="store_true",
                    help="evaluate the three stored (pre-retrain) WiLI models")
    ap.add_argument("--fp64", action="store_true",
                    help="evaluate the three retrained fp64 WiLI models")
    ap.add_argument("--tatoeba-split", default="full",
                    choices=tuple(TATOEBA_SPLITS))
    ap.add_argument("--udhr-csv", default=UDHR_CSV)
    ap.add_argument("--scope", default="model-langs",
                    choices=("model-langs", "all"))
    ap.add_argument("--limit", type=int, default=None,
                    help="smoke test only: score at most N lines of the file. "
                         "Recorded in the json; such a run is NOT comparable "
                         "to a published cell.")
    ap.add_argument("--out-dir", default=str(REPO / "outputs" / "rerelease"))
    ap.add_argument("--gate", action="store_true",
                    help="compare against the published cell for this bench")
    a = ap.parse_args(argv)

    chosen = [bool(a.models), a.stored, a.fp64]
    if sum(chosen) != 1:
        raise SystemExit("give exactly one of --models, --stored, --fp64")
    models = a.models or (STORED_MODELS if a.stored else FP64_MODELS)
    for mp in models:
        if not Path(mp).is_file():
            raise SystemExit(f"missing model: {mp}")

    from unilid.model_io import UnilidModel

    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for mp in models:
        name = Path(mp).stem
        print(f"\n=== {a.bench}  {name}  scope={a.scope}", flush=True)
        model = UnilidModel(mp, calibrated=False)
        if a.bench == "tatoeba":
            src = TATOEBA_SPLITS[a.tatoeba_split]
            chunks = load_tatoeba(src)
        else:
            src = a.udhr_csv
            chunks = load_udhr(src)
        if a.limit is not None:
            chunks = _limited(chunks, a.limit)
        res = evaluate(model, chunks, a.scope)
        res.update({"bench": a.bench, "model": os.path.abspath(mp),
                    "model_name": name, "source": src, "limit": a.limit})
        if a.bench == "tatoeba":
            res["tatoeba_split"] = a.tatoeba_split
        print(f"  total {res['total_samples']:,} in "
              f"{res['n_languages_scored']} languages; "
              f"{res['n_empty_after_preprocess']:,} empty-after-preprocess; "
              f"{res['n_rows_outside_model_label_set']:,} rows dropped by scope")
        print(f"  accuracy   {res['accuracy']:.4f}")
        print(f"  macro F1   {res['macro_f1']:.4f}")
        print(f"  macro FPR  {res['macro_fpr']:.4e}")

        # Two different things share the --gate flag and must not be confused.
        # The STORED wili_100k_500 is the model behind the published row, so a
        # mismatch there means the INSTRUMENT is wrong and the run must fail.
        # Its _fp64 retrain is a different model: comparing it to the published
        # cell measures the retrain, and a difference is a result, not a fault.
        comparable = name.removesuffix(FP64_SUFFIX) == PUBLISHED_ROW_MODEL
        is_reproduction = comparable and not name.endswith(FP64_SUFFIX)
        if a.gate and not comparable:
            print(f"  no published cell for {name} in "
                  f"tab:tatoeba_udhr_comparison; reported ungated")
        if a.gate and comparable:
            pub = PUBLISHED[a.bench]
            checks = [("macro F1", res["macro_f1"], pub["macro_f1"],
                       abs(res["macro_f1"] - pub["macro_f1"]) <= F1_TOL),
                      ("macro FPR", res["macro_fpr"], pub["macro_fpr"],
                       abs(res["macro_fpr"] - pub["macro_fpr"])
                       <= FPR_REL_TOL * pub["macro_fpr"]),
                      ("n languages", res["n_languages_scored"],
                       pub["n_langs"],
                       res["n_languages_scored"] == pub["n_langs"])]
            print("  reproduction gate against the published cells:"
                  if is_reproduction else
                  "  comparison against the published cells (retrained model; "
                  "a difference here is a result, not an instrument fault):")
            for n, got, want, ok in checks:
                gs = f"{got:.6g}" if isinstance(got, float) else str(got)
                ws = f"{want:.6g}" if isinstance(want, float) else str(want)
                print(f"    {n:12} got {gs:12} expected {ws:12} "
                      f"{'MATCH' if ok else 'MISMATCH'}")
            res["is_reproduction_gate"] = is_reproduction
            res["gate"] = [{"metric": n, "got": g, "expected": w,
                            "match": bool(o)} for n, g, w, o in checks]
            res["gate_passed"] = all(o for *_, o in checks)

        suffix = "" if a.limit is None else f"_limit{a.limit}"
        op = out_dir / f"wili_{a.bench}_{name}{suffix}.json"
        op.write_text(json.dumps(res, indent=2))
        print(f"  wrote {op}")
        rows.append(res)

    print(f"\n{'model':28} {'N':>12} {'langs':>6} {'macroF1':>9} "
          f"{'macroFPR':>11} {'acc':>8}")
    for r in rows:
        print(f"{r['model_name']:28} {r['total_samples']:12,} "
              f"{r['n_languages_scored']:6} {r['macro_f1']:9.4f} "
              f"{r['macro_fpr']:11.4e} {r['accuracy']:8.4f}")
    if a.gate and not all(r.get("gate_passed", True) for r in rows
                          if r.get("is_reproduction_gate")):
        print("\nGATE FAILED: this instrument does not reproduce the published "
              "cells, so nothing measured with it can be trusted yet.")
        return 1
    return 0


def _limited(chunks, limit):
    seen = 0
    for texts, gold in chunks:
        if seen + len(texts) > limit:
            texts, gold = texts[:limit - seen], gold[:limit - seen]
        seen += len(texts)
        if texts:
            yield texts, gold
        if seen >= limit:
            return


if __name__ == "__main__":
    sys.exit(main())
