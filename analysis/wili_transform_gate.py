"""Does transforming a stored WiLI model reproduce its fp64 retrain?

The three WiLI models published on the GitHub release carry the special-token
defect: every real token's stored log-probability is depressed by log 5, so each
row puts 0.2 of its mass on the real tokens and 0.8 on the four specials.
`analysis/correct_special_token_mass.py` undoes that arithmetically. The three
retrains that finished 2026-08-22 redo the same training with two changes:
UNILID 0.3.0's corrected special-token handling, and the patched fp64 spm_train
(fork commits d0208d9 + c5921a2) that fixes an fp32 expected-count overflow on
very long training lines.

This compares the retrained matrix against the TRANSFORMED stored matrix, per
language. A language that agrees is evidence that the transformation reproduces
the retrain there. A language that DISAGREES is reported as "differs beyond the
transferred thresholds" and nothing stronger. It is NOT by itself a measurement
of fp32-EM corruption, for two reasons stated so they are not lost:

  1. The two sides come from different trainer builds (stored: pre-0.3.0 with
     fp32 spm_train; retrained: 0.3.0 with the patched fp64 spm_train), so the
     benign variance of this comparison has never been measured. The three
     thresholds are TRANSFERRED from `analysis/gate_correction.py`, which
     calibrated them on a SAME-trainer-build comparison whose only benign source
     of variance was a different subsampling draw of the same corpus. They are
     not re-derived here.
  2. Attributing a disagreement to the stored side requires an fp32 null arm
     (retrain with the unpatched fp32 build and compare that to the transformed
     stored model). Until that exists, a FAIL localizes a difference, not a
     cause.

Corpus identity is ASSUMED, not verified. The retrains drew
`wili_corpus_shared` (see slurm_wili_train_fp64.sh), and the published models
are assumed to have been trained on the same text: the GitHub release supplies
the WiLI corpus, and the WiLI-2018 train/test split is deterministic (it ships
as fixed x_train/y_train files rather than being resampled). No record ties the
released models to these exact bytes. The corpus directory, its per-language
file count, its total size and a content-hash manifest digest are written into
every output JSON so a later run can tell whether the assumption was made
against the same bytes.

Nothing is scored: both containers are read with `load_unilid_raw`, the same
raw-weight reader `analysis/inspect_variant_models.py` uses, so no calibration
question arises. Both containers are still required to be version 1.

Rows are aligned by LANGUAGE NAME, never by position: WiLI row order is the
tokenizer-filename sort, which puts `nds-nl` before `nds` and first diverges
from sorted() at index 146.

Exit codes:
  0  every gated model PASSED
  1  the gate ran to completion and at least one model FAILED
  2  the run aborted (missing or mismatched artifact, bad selection, or any
     unexpected exception). A non-zero exit is never ambiguous between
     "measured a failure" and "could not measure".

  python -m analysis.wili_transform_gate
  python -m analysis.wili_transform_gate --models qwen3_8b_wili
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
import subprocess
import sys
import tempfile
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "UNILID"))

from analysis.correct_special_token_mass import correct  # noqa: E402
from analysis.gate_correction import (  # noqa: E402
    MAX_ABS_SIGNED_MEAN,
    MAX_MASS_WEIGHTED_DIFF,
    MIN_CORRELATION,
)
from unilid.constants import (  # noqa: E402
    MISSING_TOKEN_FILL_DETECTION_BOUND,
    SPECIAL_TOKENS,
)
from unilid.model_io import (  # noqa: E402
    HEADER_FMT,
    HEADER_SIZE,
    MAGIC,
    VERSION_BASE,
    load_unilid_raw,
)

SCRATCH = Path("/capstor/scratch/cscs/cmeister747/unilid_analysis")
# Where the stored (defective) release models were recorded by
# analysis/inspect_variant_models.py. Their paths are read from this file rather
# than hardcoded, so the gate cannot run against a model the inspection never saw.
STORED_INSPECT = REPO / "outputs" / "rerelease" / "wili_models_inspect.json"
# The training-label file of the WiLI-2018 corpus drawn from the GitHub release;
# the same directory analysis/wili_eval.py:46 evaluates against.
WILI_LABELS = SCRATCH / "wili_assets" / "wili-2018" / "y_train.txt"
OUT_DIR = REPO / "outputs" / "rerelease"
# The corpus the three retrains were trained on. Kept in step with the submit
# script by CORPUS_DECL below rather than trusted from memory.
WILI_CORPUS_DIR = SCRATCH / "wili_corpus_shared"
TRAIN_SCRIPT = REPO / "slurm_wili_train_fp64.sh"
CORPUS_DECL = 'CORPUS="${SCR}/wili_corpus_shared"'
CORPUS_GLOB = "*_train.txt"

# Model stem -> retrained (corrected + fp64) container. The stem is also the key
# used against STORED_INSPECT and the output filename suffix.
RETRAINED = {
    "wili_100k_500": SCRATCH / "wili_100k_500_fp64.unilid",
    "deepseek_v3.2_wili": SCRATCH / "deepseek_v3.2_wili_fp64.unilid",
    "qwen3_8b_wili": SCRATCH / "qwen3_8b_wili_fp64.unilid",
}
ALL_MODELS = list(RETRAINED)

# --- constants defined by this script -------------------------------------
# Per-row real-token mass must be 1.0 in BOTH matrices being compared: the
# transformed one by construction, the retrained one because UNILID 0.3.0 no
# longer hands the specials the base tokenizer's score-0 entries. The measured
# fp64 retrains sit within 1.4e-7 of 1.0 (outputs/rerelease/wili_*_fp64_inspect.json)
# and the defect sits at 0.2, so this tolerance separates the two by four orders
# of magnitude while absorbing float32 storage over ~150,000 columns.
REAL_MASS_TOL = 1e-4
# How many languages to list per worst-offender ranking.
TOP_N = 10
# Chunk size for streaming a corpus file into sha256.
HASH_BLOCK = 1 << 20
# Default home for the temporary transformed container. Session scratch rather
# than /tmp: the container is a copy of the input model, up to ~150 MB.
DEFAULT_WORK_DIR = Path(
    "/iopsstor/scratch/cscs/cmeister747/claude-65594/"
    "-users-cmeister747-unilid-analysis/"
    "02781e2f-8ce1-4f2d-82b4-b4b357bcdd9a/scratchpad/wili_transform_gate_work")
# Exit codes; see the module docstring.
EXIT_PASS = 0
EXIT_GATE_FAIL = 1
EXIT_ABORT = 2
# --------------------------------------------------------------------------


def die(msg: str):
    """Abort the run. Never used for a measured gate failure."""
    print(f"FATAL: {msg}", file=sys.stderr, flush=True)
    raise SystemExit(EXIT_ABORT)


def container_version(path: Path) -> int:
    """Read the .unilid header version without touching the weights."""
    with open(path, "rb") as f:
        header = f.read(HEADER_SIZE)
    if len(header) != HEADER_SIZE:
        die(f"{path} is shorter than a .unilid header ({len(header)} bytes)")
    magic, version, _n, _v, _b, _l = struct.unpack(HEADER_FMT, header)
    if magic != MAGIC:
        die(f"{path} is not a .unilid container (bad magic {magic!r})")
    return int(version)


def vocab_tokens(tok_json) -> list:
    """Ordered vocabulary token list, exactly as gate_correction.py reads it."""
    text = tok_json if isinstance(tok_json, str) else tok_json.decode("utf-8")
    return [t for t, _ in json.loads(text)["model"]["vocab"]]


def load_container(path: Path, label: str):
    """(vocab, weights, langs, version) from a version-1 .unilid file, or abort."""
    if not path.is_file():
        die(f"{label} missing: {path}")
    version = container_version(path)
    if version != VERSION_BASE:
        die(f"{label} {path} is container version {version}; this gate reads "
            f"raw weights and requires version {VERSION_BASE} (an uncalibrated "
            f"base model). Refusing to compare a container whose calibration "
            f"section would be silently ignored.")
    tok_json, weights, langs = load_unilid_raw(path)
    vocab = vocab_tokens(tok_json)
    if len(vocab) != weights.shape[1]:
        die(f"{label} {path}: vocabulary has {len(vocab):,} tokens but the "
            f"weight matrix is {weights.shape[1]:,} columns wide")
    dupes = sorted({l for l in langs if langs.count(l) > 1})
    if dupes:
        die(f"{label} {path}: duplicate language labels {dupes[:5]}; row "
            f"alignment by name would be ambiguous")
    return vocab, weights, langs, version


def real_token_mask(vocab: list, label: str) -> np.ndarray:
    """Mirror of gate_correction.py's real-token mask (name membership)."""
    spec = {v for v in SPECIAL_TOKENS.values()}
    mask = np.array([t not in spec for t in vocab])
    n_special = int((~mask).sum())
    if n_special != len(SPECIAL_TOKENS):
        die(f"{label}: masked out {n_special} special-token columns but "
            f"{len(SPECIAL_TOKENS)} special tokens are defined "
            f"({list(SPECIAL_TOKENS.values())}); the vocabulary either omits one "
            f"or repeats one, and the mask would not match the correction's "
            f"first-occurrence indices")
    return mask


def compare_ordered_vocab(a: list, b: list, label_a: str, label_b: str) -> dict:
    """Exact ordered token-list comparison; aborts on the first divergence."""
    if len(a) != len(b):
        die(f"vocabulary sizes differ: {label_a} has {len(a):,} tokens, "
            f"{label_b} has {len(b):,}")
    for i, (ta, tb) in enumerate(zip(a, b)):
        if ta != tb:
            die(f"vocabularies diverge at index {i}: {label_a} has {ta!r}, "
                f"{label_b} has {tb!r}. The retrain was supposed to carry the "
                f"stored model's base tokenizer verbatim.")
    return {"method": "ordered token list from the container's base tokenizer",
            "tokens_compared": len(a), "identical": True}


def read_label_set(path: Path) -> set:
    if not path.is_file():
        die(f"WiLI training labels missing: {path}")
    # No errors= handling: a decode fault in the label file must raise.
    labels = {l.strip() for l in
              path.read_text(encoding="utf-8").splitlines()}
    labels.discard("")
    if not labels:
        die(f"WiLI training labels file is empty: {path}")
    return labels


def require_same_label_set(a: set, b: set, label_a: str, label_b: str) -> None:
    if a == b:
        return
    only_a = sorted(a - b)
    only_b = sorted(b - a)
    die(f"label sets differ between {label_a} and {label_b}: "
        f"{len(only_a)} only in {label_a} ({only_a[:10]}), "
        f"{len(only_b)} only in {label_b} ({only_b[:10]})")


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(HASH_BLOCK), b""):
            h.update(block)
    return h.hexdigest()


def corpus_manifest() -> dict:
    """Content-hash manifest of the training corpus the retrains drew.

    The digest is sha256 over the newline-joined "name:size:sha256" lines of the
    per-language files in sorted name order, so it is deterministic and depends
    on content, never on mtime.
    """
    if not TRAIN_SCRIPT.is_file():
        die(f"retrain submit script missing: {TRAIN_SCRIPT} (needed to confirm "
            f"which corpus directory the retrains drew)")
    if CORPUS_DECL not in TRAIN_SCRIPT.read_text(encoding="utf-8"):
        die(f"{TRAIN_SCRIPT} no longer contains {CORPUS_DECL!r}. This gate's "
            f"WILI_CORPUS_DIR ({WILI_CORPUS_DIR}) is derived from that line; "
            f"the two have drifted apart and the corpus recorded in the output "
            f"would not be the one the retrains used.")
    if not WILI_CORPUS_DIR.is_dir():
        die(f"training corpus directory missing: {WILI_CORPUS_DIR}")
    files = sorted(WILI_CORPUS_DIR.glob(CORPUS_GLOB))
    if not files:
        die(f"no {CORPUS_GLOB} files under {WILI_CORPUS_DIR}")
    lines, total = [], 0
    for p in files:
        size = p.stat().st_size
        total += size
        lines.append(f"{p.name}:{size}:{file_sha256(p)}")
    digest = hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()
    return {
        "directory": str(WILI_CORPUS_DIR),
        "declared_in": f"{TRAIN_SCRIPT}: {CORPUS_DECL}",
        "n_files": len(files),
        "total_bytes": total,
        "manifest_sha256": digest,
        "manifest_definition": ("sha256 of the newline-joined "
                                "'name:size:sha256' lines of the per-language "
                                "files in sorted name order"),
        "identity_with_published_training_corpus": (
            "ASSUMED, not verified. Evidence: the GitHub release supplies the "
            "WiLI corpus and the WiLI-2018 train/test split is deterministic "
            "(fixed x_train/y_train files, not resampled). No record ties the "
            "released models to these exact bytes."),
    }


def provenance(stored_path: Path, retrained_path: Path) -> dict:
    try:
        head = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                              capture_output=True, text=True,
                              check=True).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "-C", str(REPO), "status", "--porcelain"],
            capture_output=True, text=True, check=True).stdout.strip())
    except (OSError, subprocess.CalledProcessError) as exc:
        die(f"cannot read the git revision of {REPO}: {exc}")
    out = {"git_head": head, "git_worktree_dirty": dirty,
           "run_utc": datetime.now(timezone.utc).isoformat()}
    for tag, p in (("stored", stored_path), ("retrained", retrained_path)):
        st = p.stat()
        out[f"{tag}_st_size"] = st.st_size
        out[f"{tag}_st_mtime"] = st.st_mtime
    return out


def row_metrics(retrained_real: np.ndarray, transformed_real: np.ndarray) -> dict:
    """Deviation metrics for one language, mirrored from gate_correction.py.

    There the comparison is `got` (a fresh retrain) against `corrected` (the
    transformed row): delta is retrain minus reference, the mass weights come
    from the REFERENCE row's probabilities, and the correlation is over the
    real tokens only. Here the retrained container supplies `got` and the
    transformed stored container supplies `corrected`, so both arguments are
    already restricted to the real-token columns.

    `mass_weighted_diff_retrain_weighted` is the same sum with the weights taken
    from the RETRAINED side instead. It is diagnostic only and gates nothing:
    reference-side weighting is blind to deviation on tokens the reference gives
    little mass but the retrain gives much, and recording both makes that
    asymmetry visible in the artifact.
    """
    delta = retrained_real - transformed_real
    p_ref = np.exp(transformed_real)
    p_retrain = np.exp(retrained_real)
    corr = float(np.corrcoef(retrained_real, transformed_real)[0, 1])
    row = {"signed_mean": float(delta.mean()),
           "mass_weighted_diff": float((p_ref * np.abs(delta)).sum()),
           "mass_weighted_diff_retrain_weighted":
               float((p_retrain * np.abs(delta)).sum()),
           "median_abs_diff": float(np.median(np.abs(delta))),
           "max_abs_diff": float(np.abs(delta).max()),
           "correlation": corr}
    row["passes_signed_mean"] = abs(row["signed_mean"]) <= MAX_ABS_SIGNED_MEAN
    row["passes_mass_weighted"] = row["mass_weighted_diff"] <= MAX_MASS_WEIGHTED_DIFF
    row["passes_correlation"] = corr >= MIN_CORRELATION
    row["passed"] = (row["passes_signed_mean"] and row["passes_mass_weighted"]
                     and row["passes_correlation"])
    return row


def row_checks(row: np.ndarray, mask: np.ndarray, lang: str, label: str) -> float:
    """Abort on assembly fill or wrong mass; return the row's real-token mass."""
    real = row[mask]
    n_fill = int((real <= MISSING_TOKEN_FILL_DETECTION_BOUND).sum())
    if n_fill:
        die(f"{label}: language {lang} has {n_fill:,} real-token entries at or "
            f"below {MISSING_TOKEN_FILL_DETECTION_BOUND:g} (the container-assembly "
            f"fill for tokens absent from that language's tokenizer file). No "
            f"deviation metric is meaningful against such a row.")
    mass = float(np.exp(real).sum())
    if abs(mass - 1.0) > REAL_MASS_TOL:
        die(f"{label}: language {lang} puts {mass:.8g} of its mass on the real "
            f"tokens, not 1.0 within {REAL_MASS_TOL:g}. Both matrices in this "
            f"comparison must be on the corrected scale.")
    return mass


def resolve_stored(name: str) -> dict:
    """Find this model's stored-container record in the inspection output."""
    if not STORED_INSPECT.is_file():
        die(f"stored-model inspection missing: {STORED_INSPECT}")
    records = json.loads(STORED_INSPECT.read_text())
    hits = [r for r in records if Path(r["model"]).stem == name]
    if len(hits) != 1:
        die(f"expected exactly one record for {name!r} in {STORED_INSPECT}, "
            f"found {len(hits)}: {[r['model'] for r in hits]}")
    rec = hits[0]
    if not rec.get("defect_present"):
        die(f"{rec['model']} is recorded in {STORED_INSPECT} with "
            f"defect_present={rec.get('defect_present')!r}. The special-token "
            f"transformation applies only to a defective model; refusing to "
            f"transform a model that was not measured as defective.")
    return rec


def gate_model(name: str, work_root: Path, corpus: dict) -> dict:
    stored_rec = resolve_stored(name)
    stored_path = Path(stored_rec["model"])
    retrained_path = RETRAINED[name]

    print(f"\n=== {name} ===", flush=True)
    print(f"  stored    {stored_path}")
    print(f"  retrained {retrained_path}")

    s_vocab, s_weights, s_langs, s_ver = load_container(stored_path, "stored model")
    # The stored matrix itself is never read here: correct() re-reads it from
    # disk and this gate compares only the transformed result.
    del s_weights
    r_vocab, r_weights, r_langs, r_ver = load_container(
        retrained_path, "retrained model")

    # (a) label SET equality between the two containers, and (c) against WiLI.
    corpus_labels = read_label_set(WILI_LABELS)
    require_same_label_set(set(s_langs), set(r_langs),
                           f"stored {stored_path.name}",
                           f"retrained {retrained_path.name}")
    require_same_label_set(set(s_langs), corpus_labels,
                           f"stored {stored_path.name}",
                           f"WiLI y_train labels {WILI_LABELS}")

    # (b) identical ordered token inventories.
    vocab_cmp = compare_ordered_vocab(s_vocab, r_vocab,
                                      f"stored {stored_path.name}",
                                      f"retrained {retrained_path.name}")
    mask = real_token_mask(s_vocab, f"{name} vocabulary")
    print(f"  {len(s_langs):,} languages, {len(s_vocab):,} vocabulary entries, "
          f"{int(mask.sum()):,} real tokens", flush=True)

    # Transform the stored matrix with the correction script's own arithmetic.
    work = work_root / name
    work.mkdir(parents=True, exist_ok=True)
    transformed_path = work / f"{name}_transformed.unilid"
    print(f"  transforming stored model -> {transformed_path}", flush=True)
    transform_summary = correct(stored_path, transformed_path)

    t_vocab, t_weights, t_langs, t_ver = load_container(
        transformed_path, "transformed stored model")
    compare_ordered_vocab(s_vocab, t_vocab, f"stored {stored_path.name}",
                          "transformed stored model")
    if t_langs != s_langs:
        die(f"the transformation changed the language list of {stored_path}")

    rows = []
    for lang in sorted(corpus_labels):
        t_row = np.asarray(t_weights[t_langs.index(lang)]).astype(np.float64)
        r_row = np.asarray(r_weights[r_langs.index(lang)]).astype(np.float64)
        t_mass = row_checks(t_row, mask, lang, f"transformed {name}")
        r_mass = row_checks(r_row, mask, lang, f"retrained {name}")

        row = {"lang": lang, **row_metrics(r_row[mask], t_row[mask]),
               "real_mass_transformed": t_mass, "real_mass_retrained": r_mass}
        rows.append(row)
        del t_row, r_row

    del t_weights
    transformed_path.unlink()

    def top(key, reverse=True, absolute=False):
        keyed = sorted(rows, key=lambda r: abs(r[key]) if absolute else r[key],
                       reverse=reverse)[:TOP_N]
        return [{"lang": r["lang"], key: r[key],
                 "signed_mean": r["signed_mean"],
                 "abs_signed_mean": abs(r["signed_mean"])} for r in keyed]

    failed = [r["lang"] for r in rows if not r["passed"]]
    result = {
        "model": name,
        "run_complete": True,
        "stored": str(stored_path),
        "retrained": str(retrained_path),
        "provenance": provenance(stored_path, retrained_path),
        "stored_container_version": s_ver,
        "retrained_container_version": r_ver,
        "transformed_container_version": t_ver,
        "n_languages": len(rows),
        "vocab_size": len(s_vocab),
        "n_real_tokens": int(mask.sum()),
        "row_alignment": "by language name (langs.index), never by position",
        "token_inventory_comparison": vocab_cmp,
        "label_set_sources": {"wili_y_train": str(WILI_LABELS),
                              "stored_inspection": str(STORED_INSPECT)},
        "training_corpus": corpus,
        "transform_summary": transform_summary,
        "thresholds": {"max_abs_signed_mean": MAX_ABS_SIGNED_MEAN,
                       "max_mass_weighted_diff": MAX_MASS_WEIGHTED_DIFF,
                       "min_correlation": MIN_CORRELATION},
        "threshold_provenance": {
            "source": "analysis/gate_correction.py:48-50 (module constants)",
            "calibrated_on": ("the GlotLID-C base model, comparing retrains "
                              "produced by the SAME trainer build; the only "
                              "benign source of variance there was a different "
                              "subsampling draw of the same corpus"),
            "re_derived_here": False,
            "this_comparison": ("cross trainer build: stored = pre-0.3.0 with "
                                "fp32 spm_train, retrained = 0.3.0 with the "
                                "patched fp64 spm_train. The benign variance of "
                                "this comparison has not been measured."),
            "interpretation_of_fail": (
                "differs beyond the transferred thresholds. NOT a measurement "
                "of fp32-EM corruption in the stored model: attributing the "
                "difference requires an fp32 null arm (retrain with the "
                "unpatched fp32 build and compare that to the transformed "
                "stored model)."),
        },
        "non_gating_columns": ["mass_weighted_diff_retrain_weighted",
                               "median_abs_diff", "max_abs_diff"],
        "threshold_failures": {
            "signed_mean": [r["lang"] for r in rows if not r["passes_signed_mean"]],
            "mass_weighted_diff": [r["lang"] for r in rows
                                   if not r["passes_mass_weighted"]],
            "correlation": [r["lang"] for r in rows if not r["passes_correlation"]],
        },
        "worst_offenders": {
            "abs_signed_mean": top("signed_mean", absolute=True),
            "mass_weighted_diff": top("mass_weighted_diff"),
            "mass_weighted_diff_retrain_weighted":
                top("mass_weighted_diff_retrain_weighted"),
            "max_abs_diff": top("max_abs_diff"),
            "lowest_correlation": top("correlation", reverse=False),
        },
        "languages": rows,
        "failed_languages": failed,
        "passed": not failed,
    }

    print(f"\n  --- {name}: {'PASS' if result['passed'] else 'FAIL'} "
          f"({len(rows) - len(failed)}/{len(rows)} languages) ---")
    for tname, langs_failing in result["threshold_failures"].items():
        print(f"  {tname:20} {len(langs_failing):>4} failing")
    print("  worst by mass-weighted diff:")
    for r in result["worst_offenders"]["mass_weighted_diff"]:
        full = next(x for x in rows if x["lang"] == r["lang"])
        print(f"    {full['lang']:10} signed mean {full['signed_mean']:+.3e}  "
              f"mass-weighted {full['mass_weighted_diff']:.3e}  "
              f"max |delta| {full['max_abs_diff']:.3f}  "
              f"corr {full['correlation']:.6f}  "
              f"{'PASS' if full['passed'] else 'FAIL'}")
    if failed:
        print(f"  failing languages ({len(failed)}): {failed[:20]}")
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--models", default=",".join(ALL_MODELS),
                    help="comma-separated model stems (default: all three)")
    ap.add_argument("--work-dir", default=None,
                    help="directory to hold the temporary transformed container; "
                         "up to about 150 MB is needed there. An explicit "
                         "directory must already exist. Default: "
                         f"{DEFAULT_WORK_DIR}")
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    args = ap.parse_args(argv)

    names = [n.strip() for n in args.models.split(",") if n.strip()]
    if not names:
        die(f"--models {args.models!r} selects no model. Pass one or more of "
            f"{ALL_MODELS}, or omit --models to gate all three. Refusing to "
            f"report a verdict over an empty selection.")
    unknown = [n for n in names if n not in RETRAINED]
    if unknown:
        die(f"unknown model stem(s) {unknown}; known: {ALL_MODELS}")
    dupes = sorted({n for n in names if names.count(n) > 1})
    if dupes:
        die(f"--models repeats {dupes}; each model may be gated once")
    out_dir = Path(args.out_dir)
    if not out_dir.is_dir():
        die(f"output directory missing: {out_dir}")
    if args.work_dir is None:
        work_dir = DEFAULT_WORK_DIR
        work_dir.mkdir(parents=True, exist_ok=True)
    else:
        work_dir = Path(args.work_dir)
        if not work_dir.is_dir():
            die(f"--work-dir does not exist: {work_dir}")

    corpus = corpus_manifest()
    print(f"training corpus {corpus['directory']}: {corpus['n_files']} files, "
          f"{corpus['total_bytes']:,} bytes, manifest "
          f"{corpus['manifest_sha256'][:16]}...", flush=True)

    # Every model is gated before anything is written, so an abort partway
    # through cannot leave a PASS-looking artifact from a run that never
    # finished. Each record additionally carries run_complete.
    results = []
    with tempfile.TemporaryDirectory(dir=work_dir) as tmp:
        for name in names:
            results.append(gate_model(name, Path(tmp), corpus))

    if not results:
        die("no model produced a result; refusing to report a verdict")

    for res in results:
        out_path = out_dir / f"wili_transform_gate_{res['model']}.json"
        out_path.write_text(json.dumps(res, indent=2))
        print(f"wrote {out_path}")

    print("\n=== verdict ===")
    for res in results:
        print(f"  {res['model']:20} {'PASS' if res['passed'] else 'FAIL'}  "
              f"({res['n_languages'] - len(res['failed_languages'])}/"
              f"{res['n_languages']} languages)")
    passed = all(r["passed"] for r in results)
    print(f"  overall: {'PASS' if passed else 'FAIL'}")
    print("  a FAIL means 'differs beyond the transferred thresholds', not "
          "'corruption'; see threshold_provenance in the output JSON.")
    return EXIT_PASS if passed else EXIT_GATE_FAIL


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        print("FATAL: aborted on an unexpected exception (see traceback above)",
              file=sys.stderr, flush=True)
        sys.exit(EXIT_ABORT)
