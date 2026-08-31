"""Assemble the corrected generation's \\unilid CLD3-subset cells from the
subset-vocabulary models, against the cells parsed out of
paper/tables/lid_main.tex.

WHAT THIS REPORTS
-----------------
The author answered on 2026-08-31 that the plain \\unilid row's CLD3-subset
columns came from models whose BASE TOKENIZER was trained on that subset of
languages (outputs/rerelease/cld_subset_gate_2026-08-31.md section 10.1), and
ruled on the same day: "Just approximately reproduce them. Again, there is no
need to stick exactly to what was done in the first version."

So this is NOT a reproduction check and no cell here is a pass/fail. It is a new
measurement of the same quantity under stated defaults, on the corrected
generation (UNILID 0.3.0, patched fp64 spm_train, the shared corpus draw), put
beside the published cell so the size and direction of the difference are on the
record.

Published values are PARSED from the .tex via analysis.cld_subset_gate_report,
never transcribed here, so the comparison cannot drift from the table it sits
beside. The restricted-argmax measurements of the same three cells are read from
outputs/rerelease/cld_subset/, so the two candidate conventions appear side by
side on one line.

NO SILENT FALLBACKS
-------------------
The corpus manifest is REQUIRED and a missing one aborts, because every count in
this record is checked against it. The four per-run artifacts (training summary,
container inspection, subset run, restricted-argmax run) are optional only in
the sense that this record can be rendered before the jobs land: each is named
under "Not yet measured" and its cells are left empty. No value is ever filled
in from another subset, another convention or a carried number, and every
artifact that IS present is checked against the manifest before it is used.

  python -m analysis.cld3_regenerated_report OUT.json OUT.md
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from analysis.cld_subset_gate_report import (  # noqa: E402
    ORIG_COMMIT, TEX, cell, parse_tex,
)
from analysis.build_cld3_subset_corpus import SUBSETS  # noqa: E402
from analysis.preflight_cld3_subset import check_base_provenance  # noqa: E402

RUNS = os.path.join(REPO, "outputs/rerelease/cld3_subset_models")
RERELEASE = os.path.join(REPO, "outputs/rerelease")
# The restricted-argmax runs of the same three cells over the FULL released
# model, measured by the 2026-08-31 gate. GlotLID-C is the one the gate had not
# measured at the time this file was written; absent -> that column of the
# comparison is left empty rather than filled.
GATE_RUNS = os.path.join(REPO, "outputs/rerelease/cld_subset")
TRAINING_ROOT = "/capstor/scratch/cscs/cmeister747/unilid_analysis"

# subset -> (benchmark, printed language count, the gate's restricted-argmax run)
SUBSET_BENCH = {
    "83": ("glotlidc", 83, "released_glotlidc_subset.json"),
    "80": ("udhr",     80, "released_udhr_subset.json"),
    "77": ("flores",   77, "released_flores_subset.json"),
}

# SLURM jobs that produced these runs. Facts of this session, recorded here so
# the record names them; nothing is computed from them.
JOBS = {
    "83": {"train": 3246937, "eval": 3247404},
    "80": {"train": 3246939, "eval": 3247405},
    "77": {"train": 3246941, "eval": 3247406},
}
# Superseded before running: 3246702/3246704/3246706 (train) and
# 3246703/3246705/3246708 (eval), submitted 2026-08-31 20:2x and cancelled the
# same evening. They were replaced, not re-run: their copy of
# slurm_cld3_subset_train.sh lacked RAYON_NUM_THREADS (the base fit would have
# oversubscribed 288 rayon threads onto 64 allocated CPUs), lacked the
# row-count and special-token gates that now run before and after packing, and
# carried a 4 h limit set from an extrapolation that the one large measurement
# of the same code path contradicts. No output of theirs exists or is used.
SUPERSEDED_JOBS = {
    "83": {"train": 3246702, "eval": 3246703},
    "80": {"train": 3246704, "eval": 3246705},
    "77": {"train": 3246706, "eval": 3246708},
}
# The eval jobs were then resubmitted a second time, also before running:
# 3246938/3246940/3246942 carried --time overrides (5 h / 1 h / 1 h) given at
# submission, so two of the three contradicted the 5 h their own script
# declares. Resubmitting without the override costs nothing -- they are held on
# afterok and cannot start before their training job finishes -- and makes the
# jobs and slurm_cld3_subset_eval.sh agree about what governed them.
SUPERSEDED_EVAL_JOBS = [3246938, 3246940, 3246942]

# Every default this job family chose, with the one line of reasoning each.
#
# This list is PROSE, and only three of the values in it are enforced anywhere:
# vocab_size, base method "hf" and byte_level, all asserted by
# check_base_provenance against the base tokenizer's own sidecar. seed,
# per_lang_counts_method, max_sentence_length, max_base_samples_per_lang and
# lang_batch_size are copied from training_summary.json into this record's JSON
# under subsets.<n>.training and are not compared against anything. Read a
# disagreement between this list and that JSON as the JSON being right.
DEFAULTS = [
    ("three models, one per benchmark subset (83 / 80 / 77)",
     "the author's answer says the base tokenizer was trained on \"that subset "
     "of languages\", and the three columns have three different subsets, so "
     "each column gets the vocabulary fitted to its own"),
    ("all lang_Script corpora of each subset's ISO codes (99 / 94 / 93)",
     "the subset definitions are bare ISO codes and the corpora are "
     "language-script pairs; the established evaluation convention is the "
     "--lang-only bare-ISO collapse, so one row per lang_Script with "
     "predictions collapsed at scoring time"),
    ("vocabulary 100,000, fitted from scratch on the subset corpora",
     "paper/submission.tex states \"we use a vocabulary size of 100k\" unless "
     "otherwise specified, and nothing in the record states a subset-model "
     "size; the base sample is 0.87-0.93 M lines, far above any size at which "
     "the trainer's own constraints would bind"),
    ("base method hf, per-language method sp, byte-level, seed 42, "
     "10,000 base samples per language, max_sentence_length 1,000,000",
     "train.py's own defaults, unchanged; max_sentence_length is the repo "
     "default that keeps every training line, which every GlotLID-C-scale run "
     "here has used"),
    ("--lang-batch-size 20",
     "what every full-scale run in this repository uses; orchestration only, it "
     "changes no number"),
    ("corpus draw results_apertus200k/corpus",
     "the monolithic train.txt the released model was trained from is gone and "
     "the author ruled a fresh draw acceptable; this is the shared draw every "
     "retrain since has used"),
    ("UNILID 0.3.0 at the current working tree, patched fp64 spm_train",
     "a corrected-generation model must carry the special-token correction "
     "(real-token mass 1.0) and the fp64 EM; the spm_train build is gated by "
     "sha256 in the job and the special-token mass is gated after packing, "
     "before the container is evaluated"),
    ("evaluation: analysis/cld_subset_eval.py --mode subset on the subset's own "
     "benchmark",
     "the paper team's own macro F1 / macro FPR core, their only_model_langs "
     "line filter, the --lang-only bare-ISO collapse and the Viterbi decode -- "
     "i.e. exactly what running their eval_*.py --lang-only against these "
     "models would compute"),
]


# The five files that govern this run. None is committed, so `git rev-parse
# HEAD` does NOT identify the code that produced these numbers, and the two
# analysis modules are imported from the working tree AT RUN TIME -- an edit
# between submission and dequeue would silently change what the queued jobs do
# while the frozen sbatch script stayed the same. Hashing them at submission
# time and re-hashing them here makes that drift visible instead of silent.
# outputs/rerelease/cld3_governing_files.json is the submission-time snapshot.
# This module is deliberately NOT in the list. It reads the results; it does not
# govern the runs, it is expected to change while they are queued, and hashing
# the file that writes the hash makes the check circular.
GOVERNING_FILES = [
    "analysis/build_cld3_subset_corpus.py",   # built the corpora and manifests
    "analysis/preflight_cld3_subset.py",      # imported by the jobs at run time
    "slurm_cld3_subset_train.sh",             # frozen at submit; hash provable
    "slurm_cld3_subset_eval.sh",              # against `scontrol write batch_script`
]
GOVERNING_SNAPSHOT = os.path.join(RERELEASE, "cld3_governing_files.json")
HASH_BLOCK = 1 << 20


def sha256_of(rel: str) -> str:
    h = hashlib.sha256()
    with open(os.path.join(REPO, rel), "rb") as f:
        while True:
            b = f.read(HASH_BLOCK)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def governing_file_state() -> dict:
    """Current hashes, and how they compare with the submission-time snapshot.

    A difference is NOT an abort: this report is regenerated after the jobs
    land, and the author may legitimately have fixed the report itself in
    between. It is recorded and printed, so a changed preflight or corpus
    builder is read as what it is -- the queued jobs ran something other than
    what this file describes.
    """
    now = {rel: sha256_of(rel) for rel in GOVERNING_FILES}
    out = {"current": now, "snapshot_path": GOVERNING_SNAPSHOT}
    if not os.path.exists(GOVERNING_SNAPSHOT):
        out["snapshot"] = None
        out["changed_since_submission"] = None
        return out
    snap = json.load(open(GOVERNING_SNAPSHOT))
    out["snapshot"] = snap
    out["changed_since_submission"] = sorted(
        rel for rel in GOVERNING_FILES
        if snap.get("files", {}).get(rel) != now[rel])
    return out


def _git_commit():
    try:
        return subprocess.check_output(
            ["git", "-C", REPO, "rev-parse", "HEAD"], text=True,
            stderr=subprocess.DEVNULL).strip()
    except Exception as exc:
        return f"unavailable: {exc}"


def _load(path, what, required=True):
    if not os.path.exists(path):
        if required:
            raise SystemExit(f"FATAL: {what} missing at {path}")
        return None
    with open(path) as f:
        return json.load(f)


def collect(subset):
    """Everything measured for one subset, or a stated absence."""
    bench, n_langs, gate_name = SUBSET_BENCH[subset]
    out = {"subset": subset, "bench": bench, "printed_languages": n_langs,
           "jobs": JOBS[subset]}

    out["corpus_manifest"] = _load(
        os.path.join(RERELEASE, f"cld3_subset_corpus_manifest_{subset}.json"),
        f"subset-{subset} corpus manifest")

    ts = os.path.join(TRAINING_ROOT, f"results_cld3sub{subset}",
                      "training_summary.json")
    out["training_summary"] = _load(ts, f"subset-{subset} training summary",
                                    required=False)
    out["training_summary_path"] = ts

    insp = os.path.join(RERELEASE, f"cld3sub{subset}_inspect.json")
    out["inspect"] = _load(insp, f"subset-{subset} container inspection",
                           required=False)
    out["inspect_path"] = insp

    run = os.path.join(RUNS, f"cld3sub{subset}_{bench}_subset.json")
    out["run"] = _load(run, f"subset-{subset} {bench} run", required=False)
    out["run_path"] = run

    gate = os.path.join(GATE_RUNS, gate_name)
    out["restricted_argmax"] = _load(
        gate, f"restricted-argmax {bench} run", required=False)
    out["restricted_argmax_path"] = gate
    return out


def _fmt_f1(x):
    return "--" if x is None else f"{x:.5f}"


def _fmt_fpr(x):
    return "--" if x is None else f"{x:.4e}"


def _published(rows, bench):
    return (cell(rows, "unilid", bench, "subset", "f1"),
            cell(rows, "unilid", bench, "subset", "fpr"))


def build(collected):
    rows_pub = parse_tex(TEX, ORIG_COMMIT)
    rows_cur = parse_tex(TEX)
    doc = {
        "generated": "2026-09-01",
        "git_commit": _git_commit(),
        "ruling": ("Author, 2026-08-31: \"Just approximately reproduce them. "
                   "Again, there is no need to stick exactly to what was done "
                   "in the first version.\" Exact reproduction of the published "
                   "cells is explicitly NOT the bar; faithfulness to the "
                   "published procedure is."),
        "answer": ("Author, 2026-08-31: \"The base tokenizer used for the CLD3 "
                   "columns in the plain implementation of UniLID were trained "
                   "on that subset of languages, so it is indeed a different "
                   "vocabulary than the model used for the full dataset "
                   "columns.\""),
        "defaults": [{"choice": c, "reason": r} for c, r in DEFAULTS],
        "published_tex": TEX,
        "published_commit": ORIG_COMMIT,
        "governing_files": governing_file_state(),
        "subsets": {},
        "missing": [],
    }
    for subset, c in collected.items():
        bench = c["bench"]
        pf1, pfpr = _published(rows_pub, bench)
        cf1, cfpr = _published(rows_cur, bench)
        entry = {
            "bench": bench,
            "printed_languages": c["printed_languages"],
            "jobs": c["jobs"],
            "published_f1": pf1, "published_fpr": pfpr,
            "current_tex_f1": cf1, "current_tex_fpr": cfpr,
            # Section 4 prints "no \corrrev edit touched a subset cell, so the
            # current file agrees". That sentence is now enforced rather than
            # asserted: parse_tex(TEX) reads the WORKING TREE, so an uncommitted
            # edit to lid_main.tex would otherwise make the printed claim false
            # with nothing reporting it.
            "published_equals_current": (pf1 == cf1 and pfpr == cfpr),
            "corpus": {
                "n_corpora": c["corpus_manifest"]["n_corpora"],
                "n_codes": c["corpus_manifest"]["n_codes"],
                "total_lines": c["corpus_manifest"]["total_lines"],
                "total_bytes": c["corpus_manifest"]["total_bytes"],
                "base_sample_lines":
                    c["corpus_manifest"]["base_sample_lines"],
                "dir": c["corpus_manifest"]["out_dir"],
            },
        }
        ts = c["training_summary"]
        if ts:
            entry["training"] = {
                "vocab_size": ts["method"]["vocab_size"],
                "base_training_method": ts["method"]["base_training_method"],
                "per_lang_counts_method": ts["method"]["per_lang_counts_method"],
                "byte_level": ts["method"]["byte_level"],
                "seed": ts["method"]["seed"],
                "max_sentence_length": ts["method"].get("max_sentence_length"),
                "max_base_samples_per_lang":
                    ts["method"]["sampling"]["max_base_samples_per_lang"],
                "languages": ts["source"]["num_languages"],
                "total_samples": ts["source"]["total_samples"],
                "base_tokenizer_seconds": ts["timing"]["base_tokenizer_seconds"],
                "language_tokenizers_seconds":
                    ts["timing"]["language_tokenizers_seconds"],
                "total_seconds": ts["timing"]["total_seconds"],
                "base_tokenizer_reused": ts["timing"]["base_tokenizer_reused"],
            }
            if ts["source"]["num_languages"] != entry["corpus"]["n_corpora"]:
                raise SystemExit(
                    f"FATAL: subset {subset} trained "
                    f"{ts['source']['num_languages']} languages, the corpus "
                    f"manifest lists {entry['corpus']['n_corpora']} corpora")
            # The twin of the base-fit cross-check below, on the OTHER of the
            # two line counts. train.py's reuse_corpus_from_dir counts the
            # corpus in text mode (train.py:254-256); the manifest counts it in
            # binary. They agree on this data -- no corpus carries a bare CR --
            # but printing the manifest's figure as "training lines" while the
            # rows were estimated over train.py's would be the same unchecked
            # duplication that the base-fit counts had.
            if ts["source"]["total_samples"] != entry["corpus"]["total_lines"]:
                raise SystemExit(
                    f"FATAL: subset {subset} was trained on "
                    f"{ts['source']['total_samples']:,} lines, the corpus "
                    f"manifest records {entry['corpus']['total_lines']:,}. The "
                    f"rows were not estimated over the corpus this record "
                    f"describes.")
            # "The base vocabulary was fitted to THIS subset's corpora" is the
            # one property that makes these subset models, so it is checked
            # against the base tokenizer's own record of the files it consumed.
            # training_summary's base_tokenizer_reused flag cannot answer it: a
            # run resumed after a walltime kill reports true for a base that
            # was fitted from scratch on the right corpora in the killed run.
            base = os.path.join(TRAINING_ROOT, f"results_cld3sub{subset}",
                                "tokenizers", "langspec_base_tokenizer.json")
            entry["base_vocabulary"] = check_base_provenance(
                base, c["corpus_manifest"]["labels"], ts["method"]["vocab_size"])
            entry["base_vocabulary"]["refitted_in_final_run"] = \
                not ts["timing"]["base_tokenizer_reused"]
            # Two independent counts of the same quantity: the manifest's
            # prediction from train.py's documented --max-base-samples-per-lang
            # default, and what the fit actually consumed. Printing both without
            # comparing them would hide a changed sampling cap.
            cap = ts["method"]["sampling"]["max_base_samples_per_lang"]
            if cap != c["corpus_manifest"]["max_base_samples_per_lang"]:
                raise SystemExit(
                    f"FATAL: subset {subset} was trained with "
                    f"--max-base-samples-per-lang {cap}, the corpus manifest "
                    f"predicted the base-fit size at "
                    f"{c['corpus_manifest']['max_base_samples_per_lang']}")
            predicted = c["corpus_manifest"]["base_sample_lines"]
            actual = entry["base_vocabulary"]["base_fit_lines"]
            if actual != predicted:
                raise SystemExit(
                    f"FATAL: subset {subset}'s base vocabulary was fitted on "
                    f"{actual:,} lines, the corpus manifest predicts "
                    f"{predicted:,} at a cap of {cap:,} lines per language. "
                    f"The base fit did not see the corpus this record describes.")
        else:
            doc["missing"].append(f"subset {subset}: {c['training_summary_path']}")

        ins = c["inspect"]
        if ins:
            # analysis.inspect_variant_models writes a LIST, one record per
            # model argument; these jobs pass exactly one.
            if not isinstance(ins, list) or len(ins) != 1:
                raise SystemExit(
                    f"FATAL: {c['inspect_path']} holds "
                    f"{len(ins) if isinstance(ins, list) else type(ins)} "
                    f"records, expected exactly 1")
            rec = ins[0]
            entry["container"] = {
                "n_languages": rec["n_languages"],
                "vocab_size": rec["vocab_size"],
                "special_columns": rec["special_columns"],
                "real_mass_min": rec["real_mass_min"],
                "real_mass_max": rec["real_mass_max"],
                "plateau_min": rec["plateau_min"],
                "plateau_max": rec["plateau_max"],
                "defect_present": rec["defect_present"],
                "model": os.path.realpath(rec["model"]),
            }
            if rec["n_languages"] != entry["corpus"]["n_corpora"]:
                raise SystemExit(
                    f"FATAL: subset {subset}'s container carries "
                    f"{rec['n_languages']} rows, the corpus manifest lists "
                    f"{entry['corpus']['n_corpora']} corpora")
            if rec["defect_present"]:
                raise SystemExit(
                    f"FATAL: subset {subset}'s container carries the "
                    f"special-token defect; it is not a corrected-generation "
                    f"model. See {c['inspect_path']}")
        else:
            doc["missing"].append(f"subset {subset}: {c['inspect_path']}")

        run = c["run"]
        if run:
            if run["mode"] != "subset" or not run["lang_only"]:
                raise SystemExit(
                    f"FATAL: {c['run_path']} is mode={run['mode']!r} "
                    f"lang_only={run['lang_only']}, not the subset convention")
            # The defect and row-count gates above describe the INSPECTED
            # container; the cells below come from the EVALUATED one. Without
            # this they could be different files and every check would still
            # pass. cld_subset_eval writes os.path.realpath, inspect writes its
            # argv path, so both are normalised before comparing.
            if entry.get("container") and \
                    entry["container"]["model"] != os.path.realpath(run["model"]):
                raise SystemExit(
                    f"FATAL: subset {subset} inspected "
                    f"{entry['container']['model']} but evaluated "
                    f"{os.path.realpath(run['model'])}. The container the "
                    f"defect and row-count gates cleared is not the container "
                    f"that produced these cells.")
            if run["n_model_rows_evaluated"] != run["n_model_rows_total"]:
                raise SystemExit(
                    f"FATAL: {c['run_path']} evaluated "
                    f"{run['n_model_rows_evaluated']} of "
                    f"{run['n_model_rows_total']} rows. A subset-trained "
                    f"model's label set IS the subset; the restriction must be "
                    f"a no-op.")
            if run["n_model_rows_total"] != entry["corpus"]["n_corpora"]:
                raise SystemExit(
                    f"FATAL: {c['run_path']} scored a container of "
                    f"{run['n_model_rows_total']} rows, the corpus manifest "
                    f"lists {entry['corpus']['n_corpora']} corpora. A container "
                    f"short one script variant still covers every bare ISO code, "
                    f"so neither the evaluator's subset check nor the job's "
                    f"row-count assertion would have caught it.")
            if run["num_languages"] != c["printed_languages"]:
                raise SystemExit(
                    f"FATAL: {c['run_path']} averaged over "
                    f"{run['num_languages']} labels, the printed column says "
                    f"{c['printed_languages']} languages. On GlotLID-C the "
                    f"label universe is gold-or-predicted, so an unscorable "
                    f"line adds the substituted {'NONE'!r} label and lands "
                    f"here; the run's own JSON is already written and this "
                    f"report can be re-run after the cause is recorded.")
            entry["measured"] = {
                "macro_f1": run["macro_f1"],
                "macro_fpr": run["macro_fpr"],
                "accuracy": run["accuracy"],
                "errors": run["total_samples"] - run["correct"],
                "lines_scored": run["total_samples"],
                "rows_before_filter": run["rows_before_filter"],
                "n_model_rows_total": run["n_model_rows_total"],
                "n_model_rows_evaluated": run["n_model_rows_evaluated"],
                "num_languages": run["num_languages"],
                "n_none_predictions": run["n_none_predictions"],
                "decode": run["decode"],
                "label_universe": run["label_universe"],
                "bench_path": run["bench_path"],
                "model": run["model"],
                "run_path": os.path.relpath(c["run_path"], REPO),
            }
        else:
            doc["missing"].append(f"subset {subset}: {c['run_path']}")

        ra = c["restricted_argmax"]
        if ra:
            # Checked exactly as strictly as the subset run above: it goes into
            # the same published-comparison table, so it gets the same guard
            # against being a differently-scoped run that happens to be on disk.
            if ra["mode"] != "subset" or not ra["lang_only"]:
                raise SystemExit(
                    f"FATAL: {c['restricted_argmax_path']} is "
                    f"mode={ra['mode']!r} lang_only={ra['lang_only']}, not the "
                    f"subset convention")
            if ra["num_languages"] != c["printed_languages"]:
                raise SystemExit(
                    f"FATAL: {c['restricted_argmax_path']} averaged over "
                    f"{ra['num_languages']} labels, the printed column says "
                    f"{c['printed_languages']} languages")
            if ra["n_model_rows_evaluated"] >= ra["n_model_rows_total"]:
                raise SystemExit(
                    f"FATAL: {c['restricted_argmax_path']} evaluated all "
                    f"{ra['n_model_rows_total']} of its model's rows. A "
                    f"RESTRICTED argmax must drop rows; this run is not the "
                    f"full-model comparison the table claims it is.")
            entry["restricted_argmax"] = {
                "macro_f1": ra["macro_f1"],
                "macro_fpr": ra["macro_fpr"],
                "errors": ra["total_samples"] - ra["correct"],
                "lines_scored": ra["total_samples"],
                "n_model_rows_evaluated": ra["n_model_rows_evaluated"],
                "n_model_rows_total": ra["n_model_rows_total"],
                "run_path": os.path.relpath(c["restricted_argmax_path"], REPO),
            }
        else:
            doc["missing"].append(
                f"subset {subset} (restricted-argmax comparison, not this "
                f"session's work): {c['restricted_argmax_path']}")
        doc["subsets"][subset] = entry
    return doc


CONSISTENCY = r"""## 6. A single convention for the right half of `tab:lid_main`

The right half now has three candidate conventions and four kinds of row, so
this section states one arrangement that is internally consistent and says what
each row costs under it.

**The convention.** For every row, the CLD3-subset cell is *the model's own
label set restricted to the subset, scored with the paper team's macro F1 /
macro FPR core over bare ISO 639-3 labels, on the benchmark lines whose gold
bare ISO is in the subset, Viterbi decode.* The rows differ only in how the
model's label set gets restricted to the subset, and there are exactly two
mechanisms, chosen by whether the vocabulary can be refitted:

- **Refit** (the \unilid row, and the calibrated row on top of it). The base
  vocabulary is fitted to the subset corpora and the rows are estimated over it.
  This is what the author's answer describes and what section 4 above measures.
- **Restrict** (the three variant rows). Their base vocabulary is a fixed LLM
  tokenizer and cannot be refitted, so the only available restriction is a
  restricted argmax over the full model's rows. This is what jobs 3244447-3244450
  compute.

These are not the same computation -- the 2026-08-26 equivalence holds only
while the base vocabulary is fixed, which is precisely what Refit changes -- so
a table that mixes them has to say so. The caption is the place: one sentence
that the \unilid and calibrated rows' subset cells come from models whose
vocabulary was fitted to each subset, and the variant rows' from the full
variant model restricted to the subset labels, because an LLM vocabulary cannot
be refitted.

**Row by row.**

| row | subset cells under the proposal | status |
|---|---|---|
| \cld, \fasttext, \glotlid | external systems; their own label sets already are, or are restricted to, the subset | unchanged, no action |
| \unilid | Refit: the three models measured in section 4 | **regenerated here** |
| \unilid (calibrated) | Refit: the same three models, with the calibration refitted on each | **not run**; see below |
| \unilid-Mistral-Nemo, -DeepSeek3.2, -Qwen3 | Restrict: restricted argmax over each variant's container | jobs 3244447-3244450 cover the released and carried containers; corrected-generation variants would need the same three passes each |

**What the calibrated row costs.** Its three subset F1 cells now in the file
(.975 / .986 / .992) were computed in this repository under a third convention
that is neither Refit nor Restrict: the test lines are filtered to the subset,
the predictions are NOT restricted, and each bare ISO code is mapped to its
largest-training-corpus `lang_Script` variant
(`outputs/tables/paper_eval_cld3_subset.md` and
`..._external.md`). Its three FPR cells are printed as `--`, with the caption
saying why. Under this proposal that row becomes Refit: the calibration is
refitted on each subset container and the same
`analysis/cld_subset_eval.py --mode subset` pass is run against the calibrated
model. That is three cheap fits plus three evaluation passes, and it needs two
code changes this session did not make: the calibrated row's own fitting
procedure has to be re-pointed at these containers, and
`analysis/cld_subset_eval.py` hardcodes `calibrated=False` at
`load_model()` (deliberately -- the published \unilid row is the uncalibrated
model), so it needs a flag rather than an edit. Until both are done, adopting
the proposal
for the \unilid row alone would put a Refit \unilid row directly above a
third-convention calibrated row -- worse than today, not better. **The two rows
move together or not at all.**

**Why not one mechanism for all seven rows.** Restrict for every row is
available today and needs no training, but it contradicts the author's answer
about how the \unilid row's published cells were produced, and the gate already
measured that it moves the published \unilid UDHR and FLORES cells (0.996 and
0.996 against the printed .992 and .997). Refit for every row is impossible:
the variant rows' vocabularies are fixed LLM tokenizers. So the split above is
forced, and the only real choice is whether to say so in the caption or to leave
the reader to assume one convention.

**The one thing this proposal does not settle.** Whether the published cells are
*replaced* by these numbers or *carried* is the author's call, not a
measurement. Section 4 gives the sizes involved. Replacing them makes the right
half the same generation as the left half and the same convention across the
\unilid rows; carrying them leaves six cells from an unavailable generation
beside corrected ones. Option 2 of the gate record's section 10.6 (carry, and
say so in the caption) remains available and costs nothing; what this session
adds is that option 1 is now costed and half-executed rather than hypothetical.
"""


def write_md(doc, path):
    o = []
    a = o.append
    a("# The \\unilid CLD3-subset cells, regenerated for the corrected generation")
    a("")
    a("Generated 2026-09-01. Measurements: "
      "`outputs/rerelease/cld3_regenerated_2026-09-01.json` and the per-run "
      "summaries under `outputs/rerelease/cld3_subset_models/`.")
    a("")
    a("**These are new measurements, not a reproduction check.** " + doc["ruling"])
    a("")
    a("## 1. What was built, and why this shape")
    a("")
    a("The answer this responds to:")
    a("")
    a("> " + doc["answer"].split(": ", 1)[1].strip('"'))
    a("")
    a("A model whose base vocabulary is fitted to the subset languages is not a "
      "row subset of the full model, so the 2026-08-26 restricted-argmax "
      "equivalence does not reach it and no such model was on this filesystem "
      "(`outputs/rerelease/cld_subset_gate_2026-08-31.md` section 5). Three "
      "were trained here, one per benchmark subset.")
    a("")
    a("## 2. Every default chosen, and its reason")
    a("")
    a("| choice | reason |")
    a("|---|---|")
    for d in doc["defaults"]:
        a(f"| {d['choice']} | {d['reason']} |")
    a("")
    a("None of these was escalated. The author's ruling licenses stated "
      "defaults over reconstruction, and each line above is the default the "
      "paper, the trainer or this repository's own prior runs already state.")
    a("")
    a("## 3. Corpus, vocabulary and container, per subset")
    a("")
    a("| subset | benchmark | ISO codes | `lang_Script` corpora | training lines | base-fit lines | vocabulary | rows | real-token mass | defect |")
    a("|---|---|---|---|---|---|---|---|---|---|")
    for s in ("83", "80", "77"):
        e = doc["subsets"].get(s)
        if not e:
            continue
        c = e["corpus"]
        t = e.get("training") or {}
        k = e.get("container") or {}
        vocab = f"{t['vocab_size']:,}" if t else "--"
        rows = f"{k['n_languages']}" if k else "--"
        rm = (f"{k['real_mass_min']:.6f}-{k['real_mass_max']:.6f}" if k else "--")
        defect = str(k["defect_present"]) if k else "--"
        a(f"| {s} | {e['bench']} | {c['n_codes']} | {c['n_corpora']} | "
          f"{c['total_lines']:,} | {c['base_sample_lines']:,} | {vocab} | "
          f"{rows} | {rm} | {defect} |")
    a("")
    a("Each base vocabulary is checked against its own sidecar metadata, which "
      "records the exact file list the fit consumed, so \"fitted on the subset "
      "corpora\" is a verified property of the artifact rather than an "
      "assumption about the run:")
    a("")
    a("| subset | corpora the base was fitted on | lines | fitted at |")
    a("|---|---|---|---|")
    for s in ("83", "80", "77"):
        b = (doc["subsets"].get(s) or {}).get("base_vocabulary")
        if not b:
            a(f"| {s} | -- | -- | -- |")
            continue
        lines = f"{b['base_fit_lines']:,}" if b["base_fit_lines"] else "--"
        a(f"| {s} | {b['n_files']} | {lines} | {b['created_at']} |")
    a("")
    a("Wall clock, from each run's own `training_summary.json`:")
    a("")
    a("| subset | base vocabulary fit | row estimation | total |")
    a("|---|---|---|---|")
    for s in ("83", "80", "77"):
        t = (doc["subsets"].get(s) or {}).get("training")
        if not t:
            a(f"| {s} | -- | -- | -- |")
            continue
        a(f"| {s} | {t['base_tokenizer_seconds'] / 3600:.2f} h | "
          f"{t['language_tokenizers_seconds'] / 3600:.2f} h | "
          f"{t['total_seconds'] / 3600:.2f} h |")
    a("")
    a("## 4. The regenerated cells, beside the published ones")
    a("")
    drift = [s for s, e in doc["subsets"].items()
             if not e["published_equals_current"]]
    agree = ("no `\\corrrev` edit touched a subset cell, so the current file "
             "agrees" if not drift else
             "**the current file DISAGREES on subset "
             + ", ".join(sorted(drift)) + "**, so these two generations of the "
             "table no longer carry the same subset cells")
    a("`published` is parsed from `paper/tables/lid_main.tex` at git "
      f"`{doc['published_commit']}` (the original submission; {agree}). "
      "`restricted argmax` is the same cell measured over the FULL released "
      "model restricted to the subset labels -- the other candidate convention, "
      "from `outputs/rerelease/cld_subset/`.")
    a("")
    a("| column | published F1 | **subset-model F1** | restricted-argmax F1 | published FPR | **subset-model FPR** | restricted-argmax FPR | lines scored | errors |")
    a("|---|---|---|---|---|---|---|---|---|")
    for s in ("83", "80", "77"):
        e = doc["subsets"].get(s)
        if not e:
            continue
        m = e.get("measured") or {}
        r = e.get("restricted_argmax") or {}
        col = f"{e['bench']}, {e['printed_languages']} languages"
        lines = f"{m['lines_scored']:,}" if m else "--"
        errs = f"{m['errors']:,}" if m else "--"
        mf1 = f"**{_fmt_f1(m['macro_f1'])}**" if m else "--"
        mfpr = f"**{_fmt_fpr(m['macro_fpr'])}**" if m else "--"
        a(f"| {col} | {e['published_f1']} | {mf1} | "
          f"{_fmt_f1(r.get('macro_f1'))} | {e['published_fpr']} | "
          f"{mfpr} | {_fmt_fpr(r.get('macro_fpr'))} | "
          f"{lines} | {errs} |")
    a("")
    a("Differences from the published cells are expected and are not failures. "
      "A vocabulary fitted to 77-83 languages, a fresh corpus draw, UNILID "
      "0.3.0's special-token correction and the patched fp64 spm_train each "
      "move the number, and the author ruled that approximate reproduction is "
      "the bar.")
    a("")
    a("## 5. Provenance")
    a("")
    gf = doc["governing_files"]
    n_gov = len(GOVERNING_FILES)
    a(f"- Repository commit: `{doc['git_commit']}` -- **but none of the "
      f"{n_gov} files that govern this run is committed**, so that hash does "
      "not identify this code. `analysis/preflight_cld3_subset.py` and "
      "`analysis/build_cld3_subset_corpus.py` are imported from the working "
      "tree at run time, so an edit between submission and dequeue changes "
      "what the queued jobs do; the two `.sh` files are frozen by sbatch at "
      "submission and their hashes are provable against `scontrol write "
      "batch_script <jobid>`. The submission-time sha256 of all "
      f"{n_gov} is recorded in "
      f"`{os.path.relpath(gf['snapshot_path'], REPO)}` and re-checked whenever "
      "this record is regenerated.")
    if gf["changed_since_submission"] is None:
        a("  - No submission-time snapshot was found, so no drift check was "
          "possible.")
    elif gf["changed_since_submission"]:
        a("  - **CHANGED since submission: "
          + ", ".join(f"`{r}`" for r in gf["changed_since_submission"])
          + ".** The queued jobs did not necessarily run what these files now "
            "contain; the frozen sbatch script is recoverable with `scontrol "
            "write batch_script <jobid>`.")
    else:
        a(f"  - All {n_gov} unchanged since submission.")
    a("- Corpus construction: `analysis/build_cld3_subset_corpus.py`; manifests "
      "at `outputs/rerelease/cld3_subset_corpus_manifest_{83,80,77}.json`.")
    a("- Preflight: `analysis/preflight_cld3_subset.py`.")
    a("- Training: `slurm_cld3_subset_train.sh`, account infra01, "
      "`--export=ALL,SUBSET=<n>`.")
    a("- Evaluation: `slurm_cld3_subset_eval.sh` calling "
      "`analysis/cld_subset_eval.py --mode subset`.")
    a("- This record: `analysis/cld3_regenerated_report.py`.")
    a("- SLURM jobs: " + "; ".join(
        f"subset {s} train {j['train']}, eval {j['eval']}"
        for s, j in JOBS.items()) + ".")
    a("- Superseded before running, and replaced rather than re-run: " +
      "; ".join(f"{j['train']}/{j['eval']}" for j in SUPERSEDED_JOBS.values()) +
      ". Their copy of the training script pinned no rayon thread count, ran no "
      "row-count or special-token gate around packing, and carried a 4 h limit "
      "taken from an extrapolation the one large measurement of the same code "
      "path contradicts. No output of theirs exists or is used.")
    a("- Also superseded before running: eval jobs " +
      ", ".join(str(j) for j in SUPERSEDED_EVAL_JOBS) + ", which carried "
      "`--time` overrides given at submission, so two of the three "
      "contradicted the walltime their own script declares. Resubmitted "
      "without the override; they are held on `afterok` and could not have "
      "started, so this cost no queue position.")
    a("")
    a("Sizing measurement, taken on the login node on 2026-08-31 before the "
      "walltime was set: the same `StandardUnigramLMTokenizer(em_mode=\"hf\", "
      "vocab_size=100000, byte_level=True)` fit that train.py runs, on the "
      "subset-83 corpus truncated to 250 and to 1,000 lines per language, took "
      "36.8 s on 4.04 MB and 87.8 s on 16.16 MB. That is sublinear "
      "(exponent 0.63), so the 151.8 MB base sample extrapolates to 6 min under "
      "the fitted power law and 14 min under a linear reading -- an order of "
      "magnitude cheaper than the 3,734 s the same fit took on WiLI's 64.1 MB "
      "(every byte figure here is 10^6 bytes; the WiLI corpus is the 61.1 MiB "
      "`du` reports), which is the figure "
      "`outputs/rerelease/cld_subset_gate_2026-08-31.md` section 10.5 "
      "extrapolated its 1.5-2 h per-model estimate from.")
    a("")
    a("That extrapolation is NOT what the walltime was set from. It runs 9.4x "
      "beyond its largest fitted point, two points spanning 4x cannot separate "
      "a fixed per-run cost from a size-dependent one, and it predicts 206 s "
      "for the WiLI fit that measured 3,734 s. Scaling linearly from the WiLI "
      "measurement instead gives about 2.5 h for the base fit plus 0.2-0.6 h of "
      "row estimation, at the 4,134-12,995 lines/s recorded in "
      "`results_deepseek_v32_fp64` and `results_apertus200k`. The jobs were "
      "given the partition's 12 h maximum, set by the pessimistic reading. The "
      "probe's value here is a lower bound and a sanity check on the corpus "
      "size, not a schedule.")
    if doc["missing"]:
        a("")
        a("### Not yet measured")
        a("")
        for m in doc["missing"]:
            a(f"- `{m}`")
        a("")
        a("Regenerate this record with "
          "`python3 -m analysis.cld3_regenerated_report "
          "outputs/rerelease/cld3_regenerated_2026-09-01.json "
          "outputs/rerelease/cld3_regenerated_2026-09-01.md` once they land.")
    a("")
    a(CONSISTENCY)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(o) + "\n")


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 2:
        raise SystemExit(__doc__.strip().splitlines()[-1].strip())
    json_out, md_out = argv
    collected = {s: collect(s) for s in SUBSETS}
    doc = build(collected)
    os.makedirs(os.path.dirname(os.path.abspath(json_out)), exist_ok=True)
    with open(json_out, "w") as f:
        json.dump(doc, f, indent=2)
    write_md(doc, md_out)
    print(f"Wrote {json_out}")
    print(f"Wrote {md_out}")
    if doc["missing"]:
        print(f"\n{len(doc['missing'])} artifacts not yet present:")
        for m in doc["missing"]:
            print(f"  {m}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
