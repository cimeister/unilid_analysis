"""CLD3-subset (and full-label-set) evaluation of a .unilid model on the three
benchmarks of `tab:lid_main`.

WHAT THIS COMPUTES, AND WHY IT IS DEFINED THIS WAY
--------------------------------------------------
`tab:lid_main` has two halves. The left half evaluates over a benchmark's full
label set (1,940 / 366 / 190 `lang_Script` labels); the right half evaluates over
"the subsets of the benchmarks that \\cld has label coverage for" (83 / 80 / 77
*languages*, bare ISO 639-3). The convention behind the right half was
reconstructed in outputs/rerelease/cld_subset_convention_sweep.{json,md}
(2026-08-31) and the remaining ambiguity -- how predictions get confined to the
subset -- was resolved by the author on 2026-08-26:

    Under the UniLID algorithm a restricted argmax over the full model and a
    model trained on only the subset languages are the SAME computation. Each
    language's row is estimated individually over the SHARED base vocabulary, so
    a subset model is just the subset of rows of the full model. (Verified
    empirically; see the equivalence micro-check recorded alongside this
    module's outputs.)

    That equivalence holds only while the base vocabulary is held fixed. A
    subset model trained from scratch, whose base tokenizer is fitted to the
    subset corpora alone, is a DIFFERENT model and is not what this computes.

So the subset convention is, in full:

  1. Label set: every label of the model whose bare ISO 639-3 code
     (`label.split("_", 1)[0]`, the `--lang-only` collapse of
     unilid_resources/eval_glotlid.py:38) is in the subset definition file.
     This is more rows than the printed language count -- 99 rows for the 83
     GlotLID-C languages, 94 for 80 UDHR, 93 for 77 FLORES on the released
     1,940-language model -- because several ISO codes carry more than one
     script.
  2. Model: loaded restricted to exactly those labels
     (`UnilidModel(..., languages=...)`, i.e. `model_io.subset_rows`), so the
     argmax cannot leave the subset. Uncalibrated (`calibrated=False`): the
     published \\unilid row is the base model, the calibrated row is separate.
  3. Line pool: the `only_model_langs` filter of the paper team's own scripts
     (eval_glotlid.py:47-54, eval_udhr.py:43-46, eval_flores.py:38-39) applied
     to that restricted model -- i.e. exactly the lines whose gold bare ISO is
     in the subset.
  4. Metric: the paper team's own macro F1 / macro FPR core, over bare ISO
     labels.

Running the paper team's `eval_glotlid.py --lang-only` against a subset-trained
model is precisely steps 1-4; this module is that computation without the
retraining, which is what the author's equivalence licenses.

THE METRIC CORE
---------------
Transcribed from unilid_resources/eval_glotlid.py:79-97 (identical arithmetic in
eval_udhr.py:72-90 and eval_flores.py:70-88), and validated to 7 significant
digits against the paper team's own recorded output in
outputs/rerelease/cld_subset_convention_sweep.json ("validation_metric_core":
fastText 0.9443269255825798 / 2.706307431053109e-05):

    tp   = confusion[(L, L)]
    fn   = sum over p != L of confusion[(L, p)]
    fp   = sum over g != L of confusion[(g, L)]
    tn   = total - tp - fn - fp              # so fp + tn == total - support_L
    fpr  = fp / (fp + tn)
    f1   = harmonic mean of tp/(tp+fp) and tp/(tp+fn)
    macro_f1 / macro_fpr = unweighted mean over the label universe

The three scripts differ in ONE respect, the label universe of that mean:

    eval_glotlid.py:79   all_langs  = labels seen as gold OR as prediction
    eval_udhr.py:72      gold_langs = labels seen as gold
    eval_flores.py:70    gold_langs = labels seen as gold

`--label-universe` defaults per benchmark to whichever the paper team's own
script for that benchmark uses. The two coincide whenever every predicted label
also has gold support, which is the normal case under confinement; the summary
records `n_pred_only_labels` so a divergence is visible rather than silent.

DECODE
------
Viterbi (`predict_batch(forward=False)`), the default of all three eval scripts.
This is the decode behind the carried \\unilid row: the original submission's
GlotLID-C cell (.929 / 2.03e-5, git 27883d5) reproduces the Viterbi prediction
file `glotlid_unilid/glotlidc_y_pred.txt` at .9292 / 2.0265e-5, not the marginal
decode's .9311 / 1.989e-5.

BENCHMARK INPUTS
----------------
GlotLID-C: the fastText-format test file, parsed exactly as eval_glotlid.py:19-31
(`__label__<code> <text>`, split on the FIRST space; blank and unlabelled lines
skipped). The caption of `tab:lid_main` states the carried rows were computed on
all 45,627,279 lines, so the gate and every carried-row comparison use the full
file, never the 45,377,279-line scored pool.

UDHR and FLORES: the pre-built `label<TAB>text` TSVs of
analysis/external_bench_eval.py's BENCH_REGISTRY, whose row and label counts are
imported from that registry rather than restated here, and which are the pools
on which the released model reproduces the published full-set cells (.859 /
1.43e-4 and .932 / 2.78e-4). Split on the FIRST tab only: exactly one UDHR row
carries a literal tab inside its text field (external_bench_eval.py's module
docstring).

NO SILENT FALLBACKS
-------------------
Every input is required and checked: the model file, the benchmark file, the
subset file. Row and label counts are asserted against the registry. A subset
code that no model label carries is an abort, not a skip. `--out` has no default,
so a run can never overwrite another model's record by omission.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from collections import Counter

from analysis.config import TEST_FILE, TOTAL_LINES
from analysis.external_bench_eval import BENCH_REGISTRY

# --- constants defined in this module -------------------------------------

# eval_glotlid.py:59 / eval_udhr.py:53 / eval_flores.py:51 all batch the scorer
# at this size; kept identical so the batching cannot itself change a result.
PREDICT_BATCH = 10_000

# Progress cadence for the 23M-line GlotLID-C pass. Reporting only; no effect on
# any number.
PROGRESS_EVERY = 1_000_000

# The label eval_*.py substitutes when the scorer returns no prediction
# (eval_glotlid.py:68-69, eval_udhr.py:62-63, eval_flores.py:60-61). Reproduced
# so that an unscorable line stays in the pool and counts as an error, exactly as
# it does in the paper team's scripts.
NONE_LABEL = "NONE"

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

BENCHES = {
    "glotlidc": {
        "reader": "fasttext",
        "path": TEST_FILE,
        "expected_rows": TOTAL_LINES,
        "expected_labels": 1940,
        "label_universe": "goldpred",   # eval_glotlid.py:79
        "subset_file": "unilid_resources/glotlidc_cld3subset_83.txt",
        "full_label_file": "unilid_resources/glotlidc_full_1940.txt",
    },
    "udhr": {
        "reader": "tsv",
        "path": BENCH_REGISTRY["udhr"]["tsv_path"],
        "expected_rows": BENCH_REGISTRY["udhr"]["expected_rows"],
        "expected_labels": BENCH_REGISTRY["udhr"]["expected_labels"],
        "label_universe": "gold",       # eval_udhr.py:72
        "subset_file": "unilid_resources/udhr_cld3subset_80.txt",
        "full_label_file": "unilid_resources/udhr_full_366.txt",
    },
    "flores": {
        "reader": "tsv",
        "path": BENCH_REGISTRY["flores"]["tsv_path"],
        "expected_rows": BENCH_REGISTRY["flores"]["expected_rows"],
        "expected_labels": BENCH_REGISTRY["flores"]["expected_labels"],
        "label_universe": "gold",       # eval_flores.py:70
        "subset_file": "unilid_resources/flores_cld3subset_77.txt",
        "full_label_file": "unilid_resources/flores_full_190.txt",
    },
}


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

def _require(path: str, what: str) -> str:
    if not os.path.exists(path):
        raise SystemExit(f"FATAL: {what} missing at {path}")
    return path


def _read_codes(path: str) -> list[str]:
    with open(path, encoding="utf-8") as f:
        codes = [ln.strip() for ln in f if ln.strip()]
    if len(codes) != len(set(codes)):
        raise SystemExit(f"FATAL: duplicate entries in {path}")
    return codes


_LABEL_RE = re.compile(r"^__label__(\S+)")


def _iter_fasttext(path: str, limit: int | None):
    """eval_glotlid.py:19-31, verbatim in behaviour, but yielded rather than
    accumulated: the GlotLID-C test file is 7.1 GB and materialising it as
    Python strings costs far more RAM than the model itself."""
    n = 0
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = _LABEL_RE.match(line)
            if not m:
                continue
            parts = line.split(" ", 1)
            yield m.group(1), (parts[1] if len(parts) == 2 else "")
            n += 1
            if limit is not None and n >= limit:
                return


def _iter_tsv(path: str, limit: int | None):
    """`label<TAB>text`, split on the FIRST tab only (one UDHR row has an
    embedded tab; analysis/external_bench_eval.py's docstring)."""
    n = 0
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.rstrip("\n")
            if not line:
                continue
            if "\t" not in line:
                raise SystemExit(f"FATAL: {path}:{i} has no tab separator")
            lab, txt = line.split("\t", 1)
            yield lab, txt
            n += 1
            if limit is not None and n >= limit:
                return


def iter_bench(bench: str, limit: int | None):
    reg = BENCHES[bench]
    path = _require(reg["path"], f"{bench} benchmark file")
    it = _iter_fasttext(path, limit) if reg["reader"] == "fasttext" \
        else _iter_tsv(path, limit)
    return it, path


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def _add_unilid_to_path():
    unilid_dir = os.path.join(REPO_ROOT, "UNILID")
    if unilid_dir not in sys.path:
        sys.path.insert(0, unilid_dir)


def model_langs_of(model_path: str) -> list[str]:
    """The model's language list, read from the file header only. Uses the
    memmap loader so choosing the subset rows never pushes the full weight
    matrix into the Rust cache."""
    _add_unilid_to_path()
    from unilid.model_io import load_unilid_raw
    _base, _weights, langs = load_unilid_raw(model_path)
    return list(langs)


def load_model(model_path: str, languages: list[str] | None):
    """Base-mode load, optionally restricted to `languages`.

    calibrated=False: the published \\unilid row is the uncalibrated model. The
    package default is calibrated=True, which would either abort (version-1 file,
    no bundled calibration) or silently evaluate a different system.
    """
    _add_unilid_to_path()
    from unilid.model_io import UnilidModel
    return UnilidModel(model_path, calibrated=False, languages=languages)


def subset_labels_of(model_langs: list[str], subset_codes: list[str],
                     subset_file: str) -> list[str]:
    """Model labels whose bare ISO is in the subset. Aborts on any subset code
    the model cannot express: a silently dropped language would shrink the macro
    average's denominator and change the printed cell."""
    want = set(subset_codes)
    keep = [l for l in model_langs if l.split("_", 1)[0] in want]
    have = {l.split("_", 1)[0] for l in keep}
    missing = sorted(want - have)
    if missing:
        raise SystemExit(
            f"FATAL: {len(missing)} of the {len(subset_codes)} codes in "
            f"{subset_file} have no label in this model: {missing}")
    return keep


# ---------------------------------------------------------------------------
# Metric core -- unilid_resources/eval_glotlid.py:79-97
# ---------------------------------------------------------------------------

def metrics_from_confusion(confusion: Counter, label_universe: str):
    gold_labels = set(g for g, _ in confusion)
    pred_labels = set(p for _, p in confusion)
    if label_universe == "goldpred":
        universe = sorted(gold_labels | pred_labels)      # eval_glotlid.py:79
    elif label_universe == "gold":
        universe = sorted(gold_labels)                    # eval_udhr.py:72
    else:
        raise SystemExit(f"FATAL: unknown label universe {label_universe!r}")

    correct = sum(v for (g, p), v in confusion.items() if g == p)
    total = sum(confusion.values())
    accuracy = correct / total if total else 0

    per_lang = {}
    for lang in universe:
        tp = confusion.get((lang, lang), 0)
        fn = sum(v for (g, p), v in confusion.items() if g == lang and p != lang)
        fp = sum(v for (g, p), v in confusion.items() if p == lang and g != lang)
        tn = total - tp - fn - fp
        prec = tp / (tp + fp) if (tp + fp) else 0
        rec = tp / (tp + fn) if (tp + fn) else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
        fpr = fp / (fp + tn) if (fp + tn) else 0
        per_lang[lang] = {"precision": prec, "recall": rec, "f1": f1,
                          "fpr": fpr, "support": tp + fn}

    macro_f1 = sum(v["f1"] for v in per_lang.values()) / len(per_lang) if per_lang else 0
    macro_fpr = sum(v["fpr"] for v in per_lang.values()) / len(per_lang) if per_lang else 0
    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "macro_fpr": macro_fpr,
        "total_samples": total,
        "correct": correct,
        "num_languages": len(universe),
        "n_gold_labels": len(gold_labels),
        "n_pred_only_labels": len(pred_labels - gold_labels),
        "pred_only_labels": sorted(pred_labels - gold_labels)[:20],
    }, per_lang


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", REPO_ROOT, "rev-parse", "HEAD"],
            text=True, stderr=subprocess.DEVNULL).strip()
    except Exception as exc:                       # provenance only
        return f"unavailable: {exc}"


def run(model_path: str, bench: str, mode: str, out_path: str,
        limit: int | None = None, per_lang_out: str | None = None,
        label_universe: str | None = None, pred_out: str | None = None):
    reg = BENCHES[bench]
    _require(model_path, "model")
    lang_only = (mode == "subset")
    universe = label_universe or reg["label_universe"]

    rows, bench_path = iter_bench(bench, limit)

    if mode == "subset":
        subset_file = _require(os.path.join(REPO_ROOT, reg["subset_file"]),
                               f"{bench} subset definition")
        subset_codes = _read_codes(subset_file)
        # Read the model's language list from the header only, then load once,
        # already restricted: loading the full model first would push a
        # ~776 MB weight matrix into the Rust cache for nothing.
        full_langs = model_langs_of(model_path)
        keep = subset_labels_of(full_langs, subset_codes, subset_file)
        model = load_model(model_path, keep)
        model_langs = list(model.langs)
        if len(model_langs) != len(keep):
            raise SystemExit(
                f"FATAL: restricted model carries {len(model_langs)} rows, "
                f"expected {len(keep)}")
    elif mode == "full":
        subset_file, subset_codes = None, None
        model = load_model(model_path, None)
        model_langs = list(model.langs)
        full_langs = list(model_langs)
    else:
        raise SystemExit(f"FATAL: unknown mode {mode!r}")

    def extract(label: str) -> str:
        return label.split("_", 1)[0] if lang_only else label

    # only_model_langs (eval_glotlid.py:47-54), applied to the stream: a row is
    # kept iff its gold label, under the same collapse the metric uses, is in
    # the evaluated label set. Batches are then formed from the KEPT rows in
    # file order, exactly as eval_glotlid.py:63 forms them from its filtered
    # list -- the batching is identical, only the buffering differs.
    check = {extract(l) for l in model_langs} if lang_only else set(model_langs)

    confusion = Counter()
    n_none = 0
    n_read = 0
    n_kept = 0
    gold_labels_seen = set()
    buf_t: list[str] = []
    buf_g: list[str] = []
    # Per-line record, banked only when --pred-out is given, so a convention
    # question can be re-asked later without re-scoring. Full `lang_Script`
    # labels, not the bare-ISO collapse, so the record is convention-neutral.
    bank_gold: list[str] | None = [] if pred_out else None
    bank_pred: list[str] | None = [] if pred_out else None
    t0 = time.perf_counter()

    def flush():
        nonlocal n_none
        if not buf_t:
            return
        results = model.predict_batch(buf_t, forward=False)
        if len(results) != len(buf_t):
            raise SystemExit(
                f"FATAL: scorer returned {len(results)} results for "
                f"{len(buf_t)} inputs (at kept row {n_kept:,})")
        for (pred_full, _tokens, _score), gold in zip(results, buf_g):
            if pred_full is None:
                pred_full = NONE_LABEL
                n_none += 1
            confusion[(extract(gold), extract(pred_full))] += 1
            if bank_gold is not None:
                bank_gold.append(gold)
                bank_pred.append(pred_full)
        buf_t.clear()
        buf_g.clear()

    for gold, text in rows:
        n_read += 1
        gold_labels_seen.add(gold)
        if extract(gold) not in check:
            continue
        n_kept += 1
        buf_t.append(text)
        buf_g.append(gold)
        if len(buf_t) == PREDICT_BATCH:
            flush()
            if n_kept % PROGRESS_EVERY < PREDICT_BATCH:
                print(f"  read {n_read:,} / scored {n_kept:,} "
                      f"({n_kept / (time.perf_counter() - t0):,.0f} kept-lines/s)",
                      flush=True)
    flush()
    elapsed = time.perf_counter() - t0

    print(f"Read {n_read:,} rows; filtered to {n_kept:,} in the evaluated "
          f"label set ({elapsed:.1f}s)", flush=True)
    if not n_kept:
        raise SystemExit("FATAL: the line filter left no rows")
    if limit is None:
        if n_read != reg["expected_rows"]:
            raise SystemExit(
                f"FATAL: {bench} has {n_read:,} rows, expected "
                f"{reg['expected_rows']:,} (BENCHES[{bench!r}]['expected_rows'])")
        if len(gold_labels_seen) != reg["expected_labels"]:
            raise SystemExit(
                f"FATAL: {bench} carries {len(gold_labels_seen)} distinct gold "
                f"labels, expected {reg['expected_labels']}")

    summary, per_lang = metrics_from_confusion(confusion, universe)
    summary.update({
        "model": os.path.realpath(model_path),
        "model_arg": model_path,
        "bench": bench,
        "bench_path": bench_path,
        "mode": mode,
        "lang_only": lang_only,
        "label_universe": universe,
        "subset_file": subset_file,
        "n_subset_codes": len(subset_codes) if subset_codes else None,
        "n_model_rows_evaluated": len(model_langs),
        "n_model_rows_total": len(full_langs),
        "n_none_predictions": n_none,
        "rows_before_filter": n_read,
        "n_gold_labels_in_file": len(gold_labels_seen),
        "limit": limit,
        "inference_time_s": elapsed,
        "samples_per_sec": n_kept / elapsed if elapsed else None,
        "predict_batch": PREDICT_BATCH,
        "decode": "viterbi (forward=False)",
        "calibrated": False,
        "git_commit": _git_commit(),
        "argv": sys.argv,
    })

    print(f"\nAccuracy: {summary['accuracy']:.6f} "
          f"({summary['correct']:,}/{summary['total_samples']:,})")
    print(f"Macro F1:  {summary['macro_f1']:.7f}")
    print(f"Macro FPR: {summary['macro_fpr']:.7e}")
    print(f"Languages: {summary['num_languages']}")

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {out_path}")
    if per_lang_out:
        os.makedirs(os.path.dirname(os.path.abspath(per_lang_out)), exist_ok=True)
        with open(per_lang_out, "w") as f:
            json.dump(per_lang, f, indent=2)
        print(f"Wrote {per_lang_out}")
    if pred_out:
        import numpy as np
        vocab = sorted(set(bank_gold) | set(bank_pred))
        code = {l: i for i, l in enumerate(vocab)}
        if len(vocab) > np.iinfo(np.int16).max:
            raise SystemExit(f"FATAL: {len(vocab)} labels exceed int16 codes")
        os.makedirs(os.path.dirname(os.path.abspath(pred_out)), exist_ok=True)
        np.savez_compressed(
            pred_out,
            labels=np.array(vocab, dtype=object),
            gold=np.fromiter((code[l] for l in bank_gold), np.int16, len(bank_gold)),
            pred=np.fromiter((code[l] for l in bank_pred), np.int16, len(bank_pred)),
        )
        print(f"Wrote {pred_out} ({len(bank_gold):,} rows, {len(vocab)} labels)")
    return summary, per_lang


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("model", help=".unilid model path")
    p.add_argument("--bench", required=True, choices=sorted(BENCHES))
    p.add_argument("--mode", required=True, choices=["subset", "full"],
                   help="subset: restricted argmax over the CLD3-subset label "
                        "set, bare-ISO metric. full: the benchmark's own full "
                        "label set, lang_Script metric.")
    p.add_argument("--out", required=True, help="summary JSON path (no default)")
    p.add_argument("--per-lang-out", default=None, help="per-label JSON path")
    p.add_argument("--pred-out", default=None,
                   help="bank the per-line (gold, prediction) pairs to this "
                        ".npz, as full lang_Script labels, so a later "
                        "convention question needs no re-scoring")
    p.add_argument("--label-universe", default=None, choices=["goldpred", "gold"],
                   help="override the per-benchmark default taken from the "
                        "paper team's own eval script")
    p.add_argument("--limit", type=int, default=None,
                   help="smoke-test only: stop reading after N benchmark rows "
                        "(disables the row/label count checks)")
    a = p.parse_args(argv)
    run(a.model, a.bench, a.mode, a.out, limit=a.limit,
        per_lang_out=a.per_lang_out, label_universe=a.label_universe,
        pred_out=a.pred_out)


if __name__ == "__main__":
    main()
