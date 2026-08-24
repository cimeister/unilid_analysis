"""E2: UDHR and FLORES-200 external-benchmark transfer test (EXPERIMENTS_PLAN.md,
"Camera-ready evaluation program (2026-08-06)", E2; conventions fixed in
EXPERIMENTAL_SETUP.md, "Camera-ready reporting conventions"). No refitting: every
constant used below (HEAD_N, RES_CAP, D3_PROX, FLOOR_TARGET, and the tau CSVs) is the
promoted configuration's own training-derived value, applied unchanged to these two
external benchmarks (the transfer statement of record in EXPERIMENTAL_SETUP.md).

Two benchmark TSVs, already built and label-mapped by the acquisition stage
(outputs/tables/external_bench_mapping.md; label mapping user-approved before any
scoring): `label<TAB>text`, UTF-8, no header, gold labels already in `lang_Script`
form matching the model's own 1,940-language codes directly (no further remapping
needed here). Exactly one row, in the UDHR TSV, carries a literal tab inside the text
field itself (the FLORES-200 file has none), so every line is split on the FIRST tab
only, never on every tab.

Three stages, dispatched by `--stage`:

STAGE "score" (`--stage score --bench {udhr,flores}`): loads the full UniLID model
(_load_unilid_model, analysis/transfer_sweep.py) and must run on SLURM, never on a
login node; this module asserts nothing about the host (no hostname check), this is
documentation only. Scores every text twice: once under the unmodified weight matrix
W (best_of_cached_weight_sets_batch, the baseline prediction), and once under the
sha256-verified floor-21 matrix W_f21 = build_equalized_weights(W, FLOOR_TARGET)[0]
(top_k_of_cached_weight_sets_batch, TOPK_MARGIN candidates per row: the floor-21
top-1 is the floor-21 prediction, and the full top-5 ids/scores are banked so stage
"eval" can replay the promoted configuration's re-examination gate without any further
model access). Persists gold indices, the baseline prediction, and the banked top-5
candidates to a scratch .npz plus a JSON provenance sidecar. Pattern source for the
scoring loop, the SCORE_CHUNK batching, and the floor-21 swap-and-verify:
analysis/commonlid_carried.py (which does the same three things for the CommonLID
out-of-domain check). Pattern source for the sha256-verify-against-fingerprint idiom:
the same file plus analysis/gate_variants.py's _build_verified_floor21_matrix.

STAGE "eval" (`--stage eval --bench {udhr,flores}`): reads the .npz, no model load.
Re-derives the promoted configuration's gate (floor21_gate, i.e. "gate_flat4_prox21")
from the banked top-5 candidates, mirroring analysis/gate_variants.py's
_flat4_prox21_two_step_pred semantics exactly: group A is every row whose floor-21
prediction has fewer than HEAD_N training lines AND has a non-excluded threshold row
in tau_floor21_gate.csv AND whose top1-minus-top2 score margin is below that
threshold; group B is every row whose floor-21 prediction is one of the flat-four
languages (tau_flat4.csv) AND below ITS threshold. The two groups are disjoint by
construction (the flat-four all have N >= HEAD_N, asserted below, not assumed) and
each group's re-examination walks a FRESH copy of the floor-21 prediction (so a group
A move can never be seen, or fed into, group B's walk, and vice versa) before the two
groups' diffs are merged into one array. The per-row replacement walk itself is
analysis/gate_variants.py's own `_walk_replacement(D3_PROX, RES_CAP, ids_row,
scores_row, N)`, imported and called directly (not reimplemented), so the acceptance
rule for a replacement candidate is bit-identical to the promoted configuration's.
The tau CSV reader is also imported directly (`_load_tau_csv`): same six-column
format as both tau_*.csv files here, and it already enforces the dtype/duplicate/
exact-language-set checks this module would otherwise have to reimplement. The
group-A/group-B masking, threshold gate, replacement walk, and merge live in one
shared helper, `_gate_walk_and_merge`, so stage "eval" and stage "selfcheck" run the
identical code path.

The floor-21 prediction used throughout this module is the top-k pass's own rank-1
candidate under the floor-21 matrix (top5_ids[:, 0], the same matrix the margins are
scored from). Because of that definition, analysis/gate_variants.py's own agree_mask
carve-out (only re-examining rows whose independently recomputed top-1 agrees with a
separately-scored floor-21 prediction) is vacuous for stage "eval": the two values
are one and the same by construction, and there is no separate best_of-scored
floor-21 pass for these external benchmarks the way there is in the internal
full-test pipeline. Stage "selfcheck" below reintroduces the carve-out explicitly,
because it replays gate_variants.py's own banked arrays, where the two values are
genuinely independent.

STAGE "selfcheck" (`--stage selfcheck`, no `--bench`): reads no benchmark .npz and
loads no model. Replays analysis/gate_variants.py's own banked stage-"topk" outputs
(gate_topk_lines.npy, gate_topk_ids.npy, gate_topk_scores.npy,
gate_topk_fingerprint.json, all under analysis.full_test_eval.SCRATCH_DIR) plus
pred_floor21.npy, pred_gate_flat4_prox21.npy, the N column of
outputs/diagnostic/full_test_per_lang_prf.csv, and both tau CSVs, through this
module's own `_gate_walk_and_merge` (the identical function stage "eval" calls). The
base prediction for each banked line is pred_floor21.npy at that line; only rows
whose banked top-1 candidate (gate_topk_ids[:, 0]) agrees with pred_floor21.npy at
that line are gated (gate_variants.py's own agree_mask carve-out, mirrored explicitly
here since it does not hold automatically the way it does for stage "eval"; see
above). The check requires the resulting full-length prediction array to equal
pred_gate_flat4_prox21.npy on every line except, at most, the banked rows where the
banked top-1 disagrees with pred_floor21.npy (gate_topk_fingerprint.json records
top1_agree_overall 0.99999955, so this set is expected to be a tiny share of the
banked rows); it reports the disagreement count either way, aborts with counts and
the first 10 differing line indices on any other mismatch, and prints SELFCHECK OK on
success. This proves this module's own reimplementation of the promoted
configuration's gate agrees with the production build it stands in for, not merely
with itself.

ACCEPTANCE GATE (evaluated before any gated-configuration number is computed, per
pre-registration): the recomputed baseline macro F1 must reproduce the paper's UniLID
baseline cell for this benchmark within ACCEPTANCE_TOL. On failure: writes a gate
report with the measured value, the paper value, and a baseline-only per-label CSV,
then exits nonzero WITHOUT computing or writing any floor-21 or gated number (the
discrepancy goes to the user, not folded into a "close enough" table). On success:
writes the full per-label CSV (baseline, floor21, gated) and the report table, and
removes any stale gate-failure report left over from a prior failing run (a failure
report from a superseded run must not linger and be mistaken for the current state).
The gate BINDS THE RELEASED MODEL ONLY. BENCH_REGISTRY's `paper_baseline_f1` is the
cell published for the released model, so holding another model to it would compare
two different models and call the expected difference a failure; worse, it would make
the corrected round's own UDHR/FLORES cells -- the whole deliverable of such a run --
unreachable. For a non-default model the same comparison is computed at the same
tolerance and printed and written in full, labelled INFORMATIONAL, NOT A GATE, and it
withholds nothing and does not exit nonzero (analysis/paper_breakdowns.py's
`gates_binding` rule, mirrored here; see `_cross_model_baseline_message`).

FLOOR TARGET. The clamp constant this module's two stages use is the one recorded in
the model's own <scratch-dir>/fingerprint_floor21.json, not the module constant
FLOOR_TARGET. That constant (-21.0, analysis.full_test_floor21) is the RELEASED
chain's guard-selected value; the floor target is a measured per-model selection and
the corrected base model's own round-grid sweep selected -17.0. Stage "score" already
read the fingerprint; stage "eval" checks the scored sidecar's recorded floor_target
against the SAME record (`_expected_floor_target`, `_check_sidecar_floor_target`),
which is what analysis/mistralnemo_eval.py's `_resolve_floor_target` does for that
chain. For the default model the expected value IS the module constant and the check
and its message are exactly what they were.

Metrics (`_per_label_metrics`, self-contained in numpy, no analysis.metrics or
analysis.metric_decomposition import): per-language TP/FP/FN/support are computed
over the FULL 1,940-language space (the same TP/FP/FN convention as
analysis/metric_decomposition.py's `_per_lang_stats` and analysis/paper_eval.py's
`_macro_fpr`, reimplemented here rather than imported), then sliced down to the
benchmark's own label set (label_idx = the distinct values in the gold array, which
for both of these benchmarks equals the TSV's full label set by construction). A
prediction landing on a language outside that label set therefore contributes no FP
to any benchmark label (its FP count lands at an index this module never reports) but
its row still counts as a false negative for its true (gold) label; the count of such
out-of-label-set predictions is tracked separately (`n_out_of_set`) and reported, not
silently dropped. Macro F1 is the unweighted mean of per-label F1 over the benchmark
label set. Macro FPR follows the repo convention (analysis/paper_eval.py's
_macro_fpr, and FPR_SCALE/FPR_HEADER are imported from there, not redefined here):
FPR_l = FP_l / (FP_l + TN_l), TN_l = n_rows - support_l - FP_l, unweighted mean,
reported multiplied by FPR_SCALE (1e5). Empty-after-preprocess rows keep the EMPTY (-1)
sentinel (analysis/full_test_eval.EMPTY) as their prediction under every
configuration, so they score as wrong for whichever system is being measured (the
repo's EMPTY convention); the count is carried from the score stage and re-verified
against the loaded arrays before it is trusted.

Hard rules followed throughout: no silent fallback (every missing/mismatched artifact
aborts, naming the artifact and, where two values are compared, both values); every
numeric constant used is imported from its origin module, never re-typed as a literal
(see the "Constants imported, never redefined here" list right after the imports).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd

from analysis.commonlid_carried import SCORE_CHUNK
from analysis.config import TOTAL_LINES
from analysis.floor_equalization import (build_equalized_weights,
                                         _special_columns,
                                         verify_one_sided_clamp)
from analysis.format_utils import to_markdown
from analysis.full_test_eval import EMPTY
from analysis.model_context import add_arguments, resolve, resolve_out_root
from analysis.full_test_floor21 import FLOOR_TARGET
from analysis.full_test_margin import HEAD_N
from analysis.gate_variants import (D3_PROX, TAU_FLAT4_CSV, TAU_FLOOR21_GATE_CSV,
                                    TAU_FLAT4_CSV_NAME, TAU_FLOOR21_GATE_CSV_NAME,
                                    _load_tau_csv, _walk_replacement)
from analysis.hierarchical_pool import RES_CAP
from analysis.margin_diagnostic import PRF_CSV, TOPK_MARGIN
from analysis.paper_eval import FPR_HEADER, FPR_SCALE
from analysis.transfer_sweep import (UNILID_MODEL_PATH, _load_model_data,
                                     _load_unilid_model)

# Constants imported, never redefined here: SCORE_CHUNK (analysis.commonlid_carried),
# TOTAL_LINES (analysis.config), FLOOR_TARGET (analysis.full_test_floor21), HEAD_N
# (analysis.full_test_margin), RES_CAP (analysis.hierarchical_pool), D3_PROX,
# TAU_FLOOR21_GATE_CSV, TAU_FLAT4_CSV (analysis.gate_variants, D3_PROX is defined
# there; the two CSV paths are also defined there and reused unchanged), TOPK_MARGIN,
# PRF_CSV (analysis.margin_diagnostic), FPR_HEADER, FPR_SCALE (analysis.paper_eval),
# EMPTY (analysis.full_test_eval), UNILID_MODEL_PATH (analysis.transfer_sweep).

# --- constants defined in this module ---

# Root of the pre-built external-benchmark TSVs and the score-stage scratch outputs
# (outputs/tables/external_bench_mapping.md, "Output files written").
EXTERNAL_BENCH_DIR = "/capstor/scratch/cscs/cmeister747/unilid_analysis/external_bench"

# Pre-registration (EXPERIMENTS_PLAN.md E2 / EXPERIMENTAL_SETUP.md "Camera-ready
# reporting conventions"): the acceptance gate's tolerance against the paper's
# published UniLID baseline cells.
ACCEPTANCE_TOL = 0.005

# Benchmark registry. expected_rows/expected_labels/paper_baseline_f1 are the
# pre-registered values (EXPERIMENTS_PLAN.md E2): UDHR reproduces the paper's label
# count exactly (366); FLORES-200 here is the original FLORES-200 devtest release
# (not flores_plus), which reproduces the paper's 190 exactly (addendum,
# outputs/tables/external_bench_mapping.md).
BENCH_REGISTRY = {
    "udhr": {
        "tsv_path": os.path.join(EXTERNAL_BENCH_DIR, "udhr_eval.tsv"),
        "expected_rows": 24_115,
        "expected_labels": 366,
        "paper_baseline_f1": 0.859,
    },
    "flores": {
        "tsv_path": os.path.join(EXTERNAL_BENCH_DIR, "flores200_eval.tsv"),
        "expected_rows": 192_280,
        "expected_labels": 190,
        "paper_baseline_f1": 0.932,
    },
}

# The flat-four re-examined set (analysis/gate_variants.py's own hardcoded check,
# "expected exactly 4 (sco_Latn, bjn_Latn, arg_Latn, vls_Latn)") has exactly this many
# rows in TAU_FLAT4_CSV; mirrored here as the same literal, for the same reason.
N_FLAT4 = 4

# The floor-21 fingerprint written by the base model's own floor-21 stage, by name.
# Stage "score" reads its `floor_target` to build the clamped matrix and stage "eval"
# reads the same field of the same file to check the scored sidecar against, so the
# two stages provably agree on the clamp; spelled once here rather than as a literal
# in each stage. Same file and same field as analysis/mistralnemo_eval.py's
# BASE_FLOOR21_FP_NAME.
FINGERPRINT_FLOOR21_NAME = "fingerprint_floor21.json"

OUT_TABLES_DIR = "outputs/tables"
OUT_DIAG_DIR = "outputs/diagnostic/external_bench"

# ---------------------------------------------------------------- model context
# Configured once rather than threaded through each free function, the same
# pattern as analysis/gate_variants.py. Unconfigured, it falls back to the
# released model and its own output root, which is the historical behaviour.
_CTX = None
# Gate-threshold CSV paths for stages eval/selfcheck. Default: the released
# tree's literals (historical behaviour); configure(out_dir=...) repoints both
# into that output tree. _load_gate_thresholds refuses to read the released
# literals for a non-default model.
_TAU1_CSV = TAU_FLOOR21_GATE_CSV
_TAU2_CSV = TAU_FLAT4_CSV


def configure(model_path: str = None, scratch_dir: str = None,
              out_dir: str = None):
    global _CTX, OUT_TABLES_DIR, OUT_DIAG_DIR, _TAU1_CSV, _TAU2_CSV
    _CTX = resolve(model_path, scratch_dir, purpose="external benchmark scoring")
    # The repo-side counterpart of that guard. OUT_TABLES_DIR/OUT_DIAG_DIR default
    # to the released model's published record: outputs/tables/external_bench_*.md
    # are where lid_main.tex's UDHR/FLORES cells come from, and the per-label CSVs
    # under outputs/diagnostic/external_bench/ are the E2 detail behind them. A
    # non-default model with no --out-dir would write the baseline per-label CSV
    # over that record before any later guard fires. resolve_out_root refuses a
    # non-default model paired with the default root, with anything inside it, or
    # with a store-backed root, exactly as in analysis/paper_eval.py and
    # analysis/paper_breakdowns.py; for the default model it returns the root and
    # never raises, so nothing about a default run changes.
    resolve_out_root(_CTX, out_dir, purpose="external benchmark scoring")
    if out_dir:
        OUT_TABLES_DIR = os.path.join(out_dir, "tables")
        OUT_DIAG_DIR = os.path.join(out_dir, "diagnostic/external_bench")
        # The gate thresholds must come from the same output tree as the model
        # being evaluated: the released-tree literals imported from
        # gate_variants would silently pair a corrected model with the
        # released model's thresholds.
        _TAU1_CSV = os.path.join(out_dir, TAU_FLOOR21_GATE_CSV_NAME)
        _TAU2_CSV = os.path.join(out_dir, TAU_FLAT4_CSV_NAME)
    else:
        # Restore the module defaults unconditionally: configure() may be
        # called more than once in one process (a driver configuring a
        # corrected model and then the default one), and without this branch
        # the corrected tau/output paths from the earlier call would stay
        # live under the default context. The non-default guard in
        # _load_gate_thresholds does not fire for a default model, so the
        # stale pairing would pass silently.
        OUT_TABLES_DIR = "outputs/tables"
        OUT_DIAG_DIR = "outputs/diagnostic/external_bench"
        _TAU1_CSV = TAU_FLOOR21_GATE_CSV
        _TAU2_CSV = TAU_FLAT4_CSV
    print(f"external benchmarks against {_CTX.describe()}\n"
          f"  tables under {OUT_TABLES_DIR}", flush=True)
    return _CTX


def _ctx():
    global _CTX
    if _CTX is None:
        _CTX = resolve(None, None, purpose="external benchmark scoring")
    return _CTX


def _bench_out_dir() -> str:
    """Where scored_<bench>.npz and its sidecar go.

    The released model keeps EXTERNAL_BENCH_DIR itself, so its recorded E2
    artifacts stay where every earlier record points. Any other model gets a
    subdirectory named for it. Without this a second model silently overwrites
    the first's scored arrays: the benchmark TSVs and the scored npz share one
    directory, and the npz name carries only the benchmark, not the model.
    """
    if _ctx().is_default_model:
        return EXTERNAL_BENCH_DIR
    stem = os.path.splitext(os.path.basename(_ctx().model_path))[0]
    return os.path.join(EXTERNAL_BENCH_DIR, f"scored_{stem}")


# ---------------------------------------------------------------------------
# Small shared helpers
# ---------------------------------------------------------------------------

def _scored_npz_path(bench: str) -> str:
    return os.path.join(_bench_out_dir(), f"scored_{bench}.npz")


def _scored_meta_path(bench: str) -> str:
    return os.path.join(_bench_out_dir(), f"scored_{bench}_meta.json")


def _gate_failure_md_path(bench: str) -> str:
    return os.path.join(OUT_TABLES_DIR, f"external_bench_{bench}_gate_failure.md")


def _expected_floor_target() -> tuple[float, str]:
    """(clamp target, provenance) the scored sidecar's `floor_target` must equal.

    Default model: the module constant, unchanged. The released chain's own
    fingerprint records -21.0, every recorded E2 artifact was scored at it, and the
    check and its message stay exactly what they were.

    Any other model: the `floor_target` recorded in that model's own
    <scratch-dir>/fingerprint_floor21.json -- the same file, field and directory
    stage "score" reads to build the clamped matrix (run_score, below). The floor
    target is a measured per-model selection, not a universal constant: the
    corrected base model's round-grid sweep selected -17.0. A missing file or a
    missing field aborts naming the artifact; falling back to the module constant
    here is what made the eval stage reachable only for the released model.
    """
    if _ctx().is_default_model:
        return FLOOR_TARGET, "analysis.full_test_floor21"
    fp_path = os.path.join(_ctx().scratch_dir, FINGERPRINT_FLOOR21_NAME)
    if not os.path.exists(fp_path):
        raise RuntimeError(
            f"model {_ctx().model_path} is not the released model, so the clamp "
            f"target must come from its own {FINGERPRINT_FLOOR21_NAME}, but "
            f"{fp_path} does not exist. That file is the record of the floor sweep "
            f"that selected the value, and it is what stage \"score\" built the "
            f"clamped matrix from. The module constant FLOOR_TARGET "
            f"({FLOOR_TARGET}, analysis.full_test_floor21) is the RELEASED chain's "
            f"guard-selected value and must not stand in for it.")
    with open(fp_path) as f:
        recorded = json.load(f).get("floor_target")
    if recorded is None:
        raise RuntimeError(
            f"{fp_path} records no floor_target field, so the clamp target stage "
            f"\"score\" built at cannot be verified for {_ctx().model_path}.")
    return float(recorded), fp_path


def _check_sidecar_floor_target(recorded, meta_path: str) -> tuple[float, str]:
    """Stage "eval"'s floor-target check, against `_expected_floor_target`.

    Kept a pure function of (recorded value, sidecar path) and the configured
    context so both branches can be fired without running a stage, the same reason
    analysis/mistralnemo_eval.py's `_resolve_floor_target` and
    analysis/paper_breakdowns.py's `_publish_tex` are pure. Returns the expected
    value and its provenance (the report's constants list states them); aborts
    naming both numbers on a mismatch."""
    expected, provenance = _expected_floor_target()
    if recorded != expected:
        if _ctx().is_default_model:
            raise RuntimeError(f"{meta_path} records floor_target {recorded}, "
                               f"expected FLOOR_TARGET {expected} "
                               "(analysis.full_test_floor21)")
        raise RuntimeError(
            f"{meta_path} records floor_target {recorded}, expected {expected} "
            f"(the floor_target recorded in {provenance}, this model's own "
            f"floor-21 fingerprint). The scored arrays were built at a different "
            f"clamp than this model's own record selected, so the floor-21 and "
            f"gated numbers from them would not be this model's.")
    return expected, provenance


def _cross_model_baseline_message(bench: str, measured: float, paper_value: float,
                                  diff: float, model_path: str) -> str:
    """The non-default-model form of the acceptance-gate line.

    BENCH_REGISTRY's `paper_baseline_f1` is the cell published for the RELEASED
    model. Comparing another model's recomputed baseline against it is a cross-model
    comparison: a difference is the expected outcome, not a reproduction failure.
    The comparison is still computed at the same tolerance and reported in full; it
    withholds no number and fails no exit code. Wording and rule mirror
    analysis/paper_breakdowns.py's `_cross_model_message`."""
    verdict = "differs from" if diff > ACCEPTANCE_TOL else "agrees with"
    return (f"INFORMATIONAL, NOT A GATE: the recomputed {bench} baseline macro F1 "
            f"{measured:.6f} {verdict} the paper's published UniLID baseline cell "
            f"{paper_value} (absolute difference {diff:.6f}, ACCEPTANCE_TOL "
            f"{ACCEPTANCE_TOL}). That published cell is a measurement of the "
            f"RELEASED model and this run scored {model_path}, so a difference "
            f"here is an expected cross-model difference, not a regression and not "
            f"a reproduction failure. Every configuration below WAS computed and "
            f"written, with this run's own numbers.")


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def _git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                             text=True, check=True)
    except (subprocess.CalledProcessError, OSError) as e:
        raise RuntimeError(f"git rev-parse HEAD failed: {e}") from e
    return out.stdout.strip()


def _read_tsv(path: str, expected_rows: int) -> tuple[list[str], list[str]]:
    """Reads a label<TAB>text TSV, splitting each line on the FIRST tab only (one
    row, in the UDHR TSV, carries a literal tab inside the text field itself; the
    FLORES-200 file has none; see the module docstring). Aborts if a line has no tab
    at all, or if the row count does not match `expected_rows`."""
    labels: list[str] = []
    texts: list[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for lineno, raw in enumerate(f, start=1):
            parts = raw.split("\t", 1)
            if len(parts) != 2:
                raise RuntimeError(f"{path}:{lineno}: no tab found; expected "
                                   "label<TAB>text")
            labels.append(parts[0])
            texts.append(parts[1].rstrip("\n"))
    if len(labels) != expected_rows:
        raise RuntimeError(f"{path} has {len(labels):,} rows, expected "
                           f"{expected_rows:,}")
    return labels, texts


def _validate_labels(labels: list[str], lang_to_idx: dict, expected_labels: int,
                     tsv_path: str) -> None:
    distinct = sorted(set(labels))
    if len(distinct) != expected_labels:
        raise RuntimeError(f"{tsv_path} has {len(distinct)} distinct labels, "
                           f"expected {expected_labels}")
    bad = [l for l in distinct if l not in lang_to_idx]
    if bad:
        shown = bad[:10]
        raise RuntimeError(
            f"{tsv_path} has {len(bad)} label(s) not in the 1,940-language set: "
            f"{shown}" + (" ..." if len(bad) > 10 else ""))


def _per_label_metrics(pred: np.ndarray, y: np.ndarray, label_idx: np.ndarray,
                       n_lang: int, n_rows: int) -> dict:
    """Self-contained (numpy only) per-label precision/recall/F1/FPR, restricted to
    `label_idx` (the benchmark's own label set: for both external benchmarks this is
    exactly the distinct values of the gold array `y`, i.e. every benchmark label has
    at least one gold row).

    TP/FP/FN/support are accumulated over the FULL n_lang-language space first (the
    same convention as analysis/metric_decomposition.py's _per_lang_stats and
    analysis/paper_eval.py's _macro_fpr, reimplemented here rather than imported so
    this function has no dependency beyond numpy), then sliced down to `label_idx`.
    A prediction landing outside `label_idx` therefore contributes no FP to any
    reported label (its FP lands at an index this function never slices out) while
    still counting as a false negative for its true label via the `fn` vector; the
    count of such out-of-set predictions is returned separately as `n_out_of_set`
    rather than folded into any per-label count.

    pred == -1 (EMPTY, analysis.full_test_eval.EMPTY: empty-after-preprocess) is
    excluded from `valid`, so it is never a true positive or a false positive for any
    label; it is a false negative for its own true label (the repo's EMPTY
    convention: an empty row scores as wrong), and it is NOT counted in
    `n_out_of_set` (that count is for valid-but-outside-the-benchmark-set
    predictions only; empty rows are reported separately by the caller from the
    score-stage empty count).

    TN_l = n_rows - support_l - FP_l (n_rows is the benchmark's total row count, the
    repo's macro-FPR convention, analysis/paper_eval.py's _macro_fpr)."""
    valid = pred >= 0
    correct = valid & (pred == y)
    tp_full = np.bincount(y[correct], minlength=n_lang).astype(np.float64)
    fp_full = np.bincount(pred[valid & ~correct], minlength=n_lang).astype(np.float64)
    fn_full = np.bincount(y[~correct], minlength=n_lang).astype(np.float64)
    support_full = np.bincount(y, minlength=n_lang).astype(np.float64)

    tp = tp_full[label_idx]
    fp = fp_full[label_idx]
    fn = fn_full[label_idx]
    support = support_full[label_idx]

    prec = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    rec = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(tp),
                   where=(prec + rec) > 0)

    tn = n_rows - support - fp
    denom = fp + tn
    fpr = np.divide(fp, denom, out=np.zeros_like(tp), where=denom > 0)

    in_label_set = np.isin(pred, label_idx)
    n_out_of_set = int((valid & ~in_label_set).sum())

    return {
        "precision": prec, "recall": rec, "f1": f1,
        "support": support.astype(np.int64), "fp": fp.astype(np.int64), "fpr": fpr,
        "macro_f1": float(f1.mean()), "macro_fpr": float(fpr.mean()),
        "n_out_of_set": n_out_of_set,
    }


def _write_per_label_csv(path: str, label_idx: np.ndarray, langs: list[str],
                         N: np.ndarray, stats_by_config: dict) -> None:
    """Columns: label, N, support, f1_<config> (one per config, in `stats_by_config`
    order), fp_<config> (same order). `support` is asserted identical across configs
    (it is purely gold-derived) rather than assumed."""
    out = {"label": [langs[i] for i in label_idx], "N": N[label_idx].astype(int)}
    support_ref = None
    for cfg, st in stats_by_config.items():
        if support_ref is None:
            support_ref = st["support"]
            out["support"] = support_ref
        elif not np.array_equal(st["support"], support_ref):
            raise RuntimeError(f"support vector for config {cfg!r} differs from "
                               "the first config's; support is gold-derived and "
                               "must be identical across configs")
    for cfg, st in stats_by_config.items():
        out[f"f1_{cfg}"] = st["f1"]
    for cfg, st in stats_by_config.items():
        out[f"fp_{cfg}"] = st["fp"]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pd.DataFrame(out).to_csv(path, index=False)


# ---------------------------------------------------------------------------
# STAGE "score" (SLURM only: loads the full model)
# ---------------------------------------------------------------------------

def _score_baseline(model, pre: list[str], vidx: list[int], n_rows: int) -> np.ndarray:
    """best_of_cached_weight_sets_batch, SCORE_CHUNK at a time, over the
    already-preprocessed non-empty texts (`pre`), scattered back to their original
    row positions (`vidx`). Rows never scored (empty after preprocess) keep EMPTY."""
    out = np.full(n_rows, EMPTY, dtype=np.int16)
    for lo in range(0, len(pre), SCORE_CHUNK):
        chunk = pre[lo:lo + SCORE_CHUNK]
        batch = model.model.best_of_cached_weight_sets_batch(chunk)
        if len(batch) != len(chunk):
            raise RuntimeError(f"scorer returned {len(batch)} results for "
                               f"{len(chunk)} inputs (baseline pass, chunk at {lo})")
        for k, (idx, _tokens, _score) in enumerate(batch):
            out[vidx[lo + k]] = idx
        print(f"  baseline: {min(lo + SCORE_CHUNK, len(pre)):,} / {len(pre):,}",
              flush=True)
    return out


def _score_top5(model, pre: list[str], vidx: list[int],
                n_rows: int) -> tuple[np.ndarray, np.ndarray, int]:
    """top_k_of_cached_weight_sets_batch(chunk, TOPK_MARGIN), SCORE_CHUNK at a time.
    Returns (top5_ids, top5_scores, n_short_cands). Both arrays have shape
    (n_rows, TOPK_MARGIN); rows never scored (empty after preprocess) keep the -1 id
    sentinel fill (never read downstream since those rows also carry EMPTY at index
    0, guarded explicitly by the "valid" mask in stage "eval"). The scores sentinel
    fill is -inf, not 0.0, matching analysis.gate_variants's own gate_topk_scores.npy
    convention for a short candidate list: an unfilled -inf slot can never win a
    top1-minus-top2 margin comparison, and _walk_replacement already skips any
    ids_row entry < 0 outright, so the two conventions never disagree. n_short_cands
    counts rows that returned fewer than TOPK_MARGIN candidates (written into the
    score-stage meta sidecar)."""
    ids_out = np.full((n_rows, TOPK_MARGIN), -1, dtype=np.int16)
    scores_out = np.full((n_rows, TOPK_MARGIN), -np.inf, dtype=np.float32)
    n_short_cands = 0
    for lo in range(0, len(pre), SCORE_CHUNK):
        chunk = pre[lo:lo + SCORE_CHUNK]
        topk = model.model.top_k_of_cached_weight_sets_batch(chunk, TOPK_MARGIN)
        if len(topk) != len(chunk):
            raise RuntimeError(f"top-k scorer returned {len(topk)} results for "
                               f"{len(chunk)} inputs (floor-21 pass, chunk at {lo})")
        for k, cands in enumerate(topk):
            row = vidx[lo + k]
            if len(cands) > TOPK_MARGIN:
                raise RuntimeError(f"top-k scorer returned {len(cands)} candidates "
                                   f"for row {row}, expected at most {TOPK_MARGIN}")
            if len(cands) < TOPK_MARGIN:
                n_short_cands += 1
            for j, (cid, score) in enumerate(cands):
                ids_out[row, j] = cid
                scores_out[row, j] = score
        print(f"  floor21 top-{TOPK_MARGIN}: "
              f"{min(lo + SCORE_CHUNK, len(pre)):,} / {len(pre):,}", flush=True)
    return ids_out, scores_out, n_short_cands


def run_score(bench: str) -> str:
    """STAGE "score". Must run on SLURM (loads the full ~1,940-language model); this
    function makes no host assertion, the requirement is documentation only (module
    docstring, and slurm_external_bench.sh)."""
    if bench not in BENCH_REGISTRY:
        raise RuntimeError(f"unknown bench {bench!r}, expected one of "
                           f"{list(BENCH_REGISTRY)}")
    reg = BENCH_REGISTRY[bench]

    labels, texts = _read_tsv(reg["tsv_path"], reg["expected_rows"])
    n_rows = len(texts)

    weights, langs, lang_to_idx = _load_model_data(_ctx().model_path)
    n_lang = len(langs)
    W = np.array(weights, dtype=np.float32)
    del weights

    _validate_labels(labels, lang_to_idx, reg["expected_labels"], reg["tsv_path"])
    y = np.array([lang_to_idx[l] for l in labels], dtype=np.int16)

    tsv_sha = _sha256_file(reg["tsv_path"])
    git_commit = _git_commit()

    print(f"Loading model ({_ctx().model_path})...", flush=True)
    model = _load_unilid_model(_ctx().model_path)
    if model.langs != langs:
        raise RuntimeError("_load_unilid_model's language list differs from "
                           "_load_model_data's; the two loaders read the model "
                           "file inconsistently")

    # Preprocessing convention reused from analysis/commonlid_carried.py: apply the
    # model's own preprocess() (tokenizer normalizer, then pre_tokenizer), and treat
    # a falsy result (None or empty string) as "empty after preprocess". Both the
    # baseline pass and the floor-21 pass reuse the same (pre, vidx) so the two
    # passes agree exactly on which rows were scored.
    pre: list[str] = []
    vidx: list[int] = []
    for i, t in enumerate(texts):
        p = model.preprocess(t)
        if p:
            pre.append(p)
            vidx.append(i)
    n_empty = n_rows - len(pre)
    print(f"{n_rows:,} rows, {n_empty:,} empty after preprocess "
          f"({len(pre):,} scored)", flush=True)

    print("baseline pass (unmodified W)...", flush=True)
    pred_baseline = _score_baseline(model, pre, vidx, n_rows)

    fp_path = os.path.join(_ctx().scratch_dir, FINGERPRINT_FLOOR21_NAME)
    with open(fp_path) as f:
        fp = json.load(f)
    # The constant comes from the fingerprint, so this cannot be built at a
    # different c than the full-pool predictions it is compared against. The
    # sha256 check below then verifies the rebuild.
    target = float(fp.get("floor_target", FLOOR_TARGET))
    print(f"building and verifying the clamped matrix at c = {target}...",
          flush=True)
    special_cols = _special_columns(_ctx().model_path)
    if len(special_cols) != 4:
        raise RuntimeError(
            f"{_ctx().model_path}: found {len(special_cols)} special columns, "
            f"expected 4 (_special_columns drops a token absent from the "
            f"vocabulary silently, so a short list means a missing special)")
    w21, n_mod = build_equalized_weights(W, target, special_idx=special_cols)
    verify_one_sided_clamp(W, target, special_cols, n_mod)
    sha_w = hashlib.sha256(W.tobytes()).hexdigest()
    if sha_w != fp["sha256_base_W"]:
        raise RuntimeError(
            f"loaded base weight matrix does not match the recorded fingerprint: "
            f"loaded sha256 {sha_w}, recorded sha256_base_W {fp['sha256_base_W']} "
            f"(from {fp_path}); this run's model file may differ from the one the "
            "internal full-test pipeline fingerprinted")
    sha_w21 = hashlib.sha256(w21.tobytes()).hexdigest()
    if sha_w21 != fp["sha256_w21"]:
        raise RuntimeError(
            f"rebuilt floor-21 matrix does not match the recorded fingerprint: "
            f"rebuilt sha256 {sha_w21}, recorded sha256_w21 {fp['sha256_w21']} "
            f"(from {fp_path})")
    model.model.set_weight_sets(w21.tolist())
    del W, w21

    print(f"floor-21 top-{TOPK_MARGIN} pass...", flush=True)
    top5_ids, top5_scores, n_short_cands = _score_top5(model, pre, vidx, n_rows)

    del model

    meta = {
        "bench": bench,
        "git_commit": git_commit,
        "tsv_path": reg["tsv_path"],
        "tsv_sha256": tsv_sha,
        "n_rows": n_rows,
        "n_labels": reg["expected_labels"],
        "n_empty": n_empty,
        "model_path": _ctx().model_path,
        "matrix_shas": {
            "sha256_W_loaded": sha_w,
            "sha256_w21_computed": sha_w21,
            "sha256_w21_fingerprint": fp["sha256_w21"],
            "fingerprint_floor21_path": fp_path,
        },
        "preprocessing": (
            "model.preprocess(text): UnilidModel.preprocess (UNILID/unilid/"
            "model_io.py) applies the base tokenizer's normalizer, then its "
            "pre_tokenizer; a row is treated as empty when the result is None or "
            "an empty string, the same test analysis/commonlid_carried.py uses "
            "(`if p:` on the preprocessed value). Empty rows are excluded from the "
            "batches sent to the scorer and keep the EMPTY (-1) sentinel "
            "(analysis.full_test_eval.EMPTY) as their prediction under both the "
            "baseline pass and the floor-21 top-k pass."
        ),
        "score_chunk": SCORE_CHUNK,
        "topk_margin": TOPK_MARGIN,
        "floor_target": target,
        "n_short_cands": n_short_cands,
    }

    os.makedirs(_bench_out_dir(), exist_ok=True)
    npz_path = _scored_npz_path(bench)
    npz_tmp = npz_path.replace(".npz", ".tmp.npz")
    np.savez_compressed(npz_tmp, y=y, pred_baseline=pred_baseline, top5_ids=top5_ids,
                        top5_scores=top5_scores, n_empty=np.int64(n_empty))
    os.replace(npz_tmp, npz_path)

    meta_path = _scored_meta_path(bench)
    meta_tmp = meta_path + ".tmp"
    with open(meta_tmp, "w") as f:
        json.dump(meta, f, indent=2)
    os.replace(meta_tmp, meta_path)

    print(f"Wrote {npz_path}, {meta_path}")
    print(f"SCORE OK ({bench}): {n_rows:,} rows, {n_empty:,} empty, "
          f"{reg['expected_labels']} labels, {n_short_cands:,} rows with fewer "
          f"than {TOPK_MARGIN} saved candidates.")
    return npz_path


# ---------------------------------------------------------------------------
# Shared: group-A/group-B masking, threshold gate, replacement walk, and merge.
# Used by both run_eval (stage "eval") and run_selfcheck (stage "selfcheck"), so
# the selfcheck exercises the exact code path it is checking, not a reimplementation
# of it.
# ---------------------------------------------------------------------------

def _load_gate_thresholds(langs: list[str], N: np.ndarray) -> dict:
    """Loads TAU_FLOOR21_GATE_CSV (group A: languages with N < HEAD_N) and
    TAU_FLAT4_CSV (group B: the flat-four languages with N >= HEAD_N), and the
    group B language-index array, mirroring analysis/gate_variants.py's
    _run_apply_flat4_prox21 exactly. Shared by run_eval and run_selfcheck so both
    read the promoted configuration's own thresholds through one path.

    The expected group-A language count is read directly from TAU_FLOOR21_GATE_CSV's
    own row count, never a hardcoded literal: a tau CSV that silently gained or lost
    a re-examined language would otherwise pass unnoticed.

    Returns a dict: tau1, sha_tau1, tau2, sha_tau2, step2_idx, step1_langs,
    step2_langs."""
    n_lang = len(langs)
    lang_to_pos = {l: i for i, l in enumerate(langs)}

    if not _ctx().is_default_model and _TAU1_CSV == TAU_FLOOR21_GATE_CSV:
        raise RuntimeError(
            f"model {_ctx().model_path} is not the released model but the gate "
            f"thresholds would be read from the released tree "
            f"({TAU_FLOOR21_GATE_CSV}); pass --out-dir pointing at the output "
            f"root that holds this model's own tau CSVs")

    step1_langs = {langs[i] for i in range(n_lang) if N[i] < HEAD_N}
    tau1_row_count = len(pd.read_csv(_TAU1_CSV))
    if len(step1_langs) != tau1_row_count:
        raise RuntimeError(f"{len(step1_langs)} languages with N < HEAD_N "
                           f"({HEAD_N:,}), expected {tau1_row_count} (the "
                           f"data-row count of {_TAU1_CSV})")
    tau1, sha_tau1 = _load_tau_csv(_TAU1_CSV, langs, step1_langs)

    flat4_df = pd.read_csv(_TAU2_CSV)
    if len(flat4_df) != N_FLAT4:
        raise RuntimeError(f"{_TAU2_CSV} has {len(flat4_df)} rows, expected "
                           f"exactly {N_FLAT4} (the flat-four languages)")
    step2_langs = set(flat4_df.lang)
    for lang in step2_langs:
        n_l = int(N[lang_to_pos[lang]])
        if n_l < HEAD_N:
            raise RuntimeError(
                f"flat-four language {lang} has N={n_l:,} < HEAD_N ({HEAD_N:,}); "
                "expected all four to have N >= HEAD_N so group A and group B "
                "are disjoint by construction")
    tau2, sha_tau2 = _load_tau_csv(_TAU2_CSV, langs, step2_langs)

    if step1_langs & step2_langs:
        raise RuntimeError("group A (N < HEAD_N) and group B (flat-four) "
                           "language sets overlap; expected disjoint")
    step2_idx = np.array(sorted(lang_to_pos[l] for l in step2_langs), dtype=np.int64)

    return {"tau1": tau1, "sha_tau1": sha_tau1, "tau2": tau2, "sha_tau2": sha_tau2,
           "step2_idx": step2_idx, "step1_langs": step1_langs,
           "step2_langs": step2_langs}


def _gate_walk_and_merge(base_pred: np.ndarray, ids: np.ndarray, scores: np.ndarray,
                         N: np.ndarray, tau1: np.ndarray, tau2: np.ndarray,
                         step2_idx: np.ndarray,
                         agree_mask: np.ndarray | None = None
                         ) -> tuple[np.ndarray, dict, dict]:
    """Group-A/group-B masking, threshold gate, replacement walk
    (analysis.gate_variants._walk_replacement), and merge: the promoted
    configuration's (gate_flat4_prox21) own re-examination logic, mirroring
    analysis.gate_variants._run_apply_flat4_prox21 / _flat4_prox21_two_step_pred
    exactly. Shared by run_eval (stage "eval") and run_selfcheck (stage
    "selfcheck"), so the check exercises the exact code path it is checking.

    `base_pred`: one floor-21 prediction per row of `ids`/`scores`. In run_eval this
    IS ids[:, 0] by construction (this module's floor-21 prediction is defined as
    the top-k pass's own rank-1 candidate; see the module docstring), so every row
    trivially agrees with its own top-1 and `agree_mask` is left None (every row
    treated as agreeing). In run_selfcheck, `base_pred` is the separately-scored
    pred_floor21.npy value at each banked line, which the topk-stage's own
    recomputation (ids[:, 0]) agrees with on only a share of rows
    (gate_topk_fingerprint.json's top1_agree_overall); `agree_mask` there is the
    explicit ids[:, 0] == base_pred comparison, mirroring
    analysis.gate_variants.py's own agree_mask carve-out.

    `agree_mask`, when given, restricts which in-set rows are gated (eligible for
    the below-tau re-examination): a row in a group's language set whose agree_mask
    entry is False is left unchanged and is never walked, matching
    analysis.gate_variants.py's own disagreement handling exactly.

    Returns (pred_gated, stats_a, stats_b): pred_gated is base_pred with group A's
    and group B's accepted replacements applied; stats_a/stats_b carry
    n_examined/n_moved/n_blocked_by_proximity/n_no_cand for each group."""
    n_rows = base_pred.shape[0]
    valid = base_pred >= 0
    if agree_mask is None:
        agree_mask = np.ones(n_rows, dtype=bool)

    gap = scores[:, 0] - scores[:, 1]

    in_set1 = np.zeros(n_rows, dtype=bool)
    in_set1[valid] = N[base_pred[valid]] < HEAD_N
    in_set2 = np.zeros(n_rows, dtype=bool)
    in_set2[valid] = np.isin(base_pred[valid], step2_idx)
    if (in_set1 & in_set2).any():
        raise RuntimeError("a row's floor-21 prediction falls in both group A's "
                           "and group B's language sets; expected disjoint")

    gated1 = in_set1 & agree_mask
    gated2 = in_set2 & agree_mask

    below1 = np.zeros(n_rows, dtype=bool)
    below1[valid] = gated1[valid] & (gap[valid] < tau1[base_pred[valid]])
    below2 = np.zeros(n_rows, dtype=bool)
    below2[valid] = gated2[valid] & (gap[valid] < tau2[base_pred[valid]])
    if (below1 & below2).any():
        raise RuntimeError("a row is flagged for re-examination under both "
                           "group A and group B; expected disjoint")

    def _run_step(mask: np.ndarray) -> tuple[np.ndarray, dict]:
        # Fresh copy of base_pred per group: a group A move must never be visible
        # to, or feed, group B's walk (and vice versa).
        pred = base_pred.copy()
        n_moved = n_blocked = n_no_cand = 0
        for r in np.where(mask)[0].tolist():
            outcome, cid = _walk_replacement(D3_PROX, RES_CAP, ids[r], scores[r], N)
            if outcome == "moved":
                pred[r] = cid
                n_moved += 1
            elif outcome == "blocked_by_proximity":
                n_blocked += 1
            elif outcome == "no_cand":
                n_no_cand += 1
            else:
                raise RuntimeError(f"_walk_replacement returned unknown outcome "
                                   f"{outcome!r}")
        n_examined = int(mask.sum())
        if n_examined != n_moved + n_blocked + n_no_cand:
            raise RuntimeError(
                f"re-examination bookkeeping mismatch: {n_examined:,} examined "
                f"!= {n_moved:,} moved + {n_blocked:,} blocked_by_proximity + "
                f"{n_no_cand:,} no_cand")
        return pred, {"n_examined": n_examined, "n_moved": n_moved,
                     "n_blocked_by_proximity": n_blocked, "n_no_cand": n_no_cand}

    pred1, stats_a = _run_step(below1)
    pred2, stats_b = _run_step(below2)
    diff1 = pred1 != base_pred
    diff2 = pred2 != base_pred
    if (diff1 & diff2).any():
        raise RuntimeError("group A's and group B's moves landed on the same "
                           "row(s); expected disjoint re-examined-row sets")
    pred_gated = base_pred.copy()
    pred_gated[diff1] = pred1[diff1]
    pred_gated[diff2] = pred2[diff2]
    return pred_gated, stats_a, stats_b


# ---------------------------------------------------------------------------
# STAGE "eval" (login node: no model load, reads the .npz)
# ---------------------------------------------------------------------------

def run_eval(bench: str) -> str:
    if bench not in BENCH_REGISTRY:
        raise RuntimeError(f"unknown bench {bench!r}, expected one of "
                           f"{list(BENCH_REGISTRY)}")
    reg = BENCH_REGISTRY[bench]

    npz_path = _scored_npz_path(bench)
    meta_path = _scored_meta_path(bench)
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"scored npz missing: {npz_path}; run "
                                f"--stage score --bench {bench} first")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"scored meta sidecar missing: {meta_path}")
    with open(meta_path) as f:
        meta = json.load(f)

    current_sha = _sha256_file(reg["tsv_path"])
    if current_sha != meta["tsv_sha256"]:
        raise RuntimeError(
            f"{reg['tsv_path']} sha256 has drifted since the score stage ran: "
            f"scored under {meta['tsv_sha256']}, currently {current_sha}; rerun "
            f"--stage score --bench {bench} before evaluating")

    with np.load(npz_path) as z:
        y = z["y"].astype(np.int64)
        pred_baseline = z["pred_baseline"].astype(np.int64)
        top5_ids = z["top5_ids"].astype(np.int64)
        # NOT upcast to float64: analysis.gate_variants stores and walks its own
        # candidate scores as float32 throughout (gate_topk_scores.npy,
        # _walk_replacement's scores_row), so the top1-minus-top2 margin and the
        # D3_PROX proximity comparison here must use the identical float32
        # arithmetic, bit-for-bit, or run_selfcheck's reproduction of
        # pred_gate_flat4_prox21.npy could differ from a rounding difference in the
        # gap/threshold comparisons rather than a real logic mismatch.
        top5_scores = z["top5_scores"]
        n_empty = int(z["n_empty"])

    n_rows = len(y)
    if n_rows != reg["expected_rows"]:
        raise RuntimeError(f"{npz_path} has {n_rows:,} rows, expected "
                           f"{reg['expected_rows']:,}")
    if top5_scores.dtype != np.float32:
        raise RuntimeError(f"{npz_path}: top5_scores has dtype "
                           f"{top5_scores.dtype}, expected float32 (the score "
                           "stage's own _score_top5 output dtype)")
    if pred_baseline.shape != (n_rows,):
        raise RuntimeError(f"pred_baseline shape {pred_baseline.shape} != "
                           f"({n_rows},)")
    if top5_ids.shape != (n_rows, TOPK_MARGIN):
        raise RuntimeError(f"top5_ids shape {top5_ids.shape} != "
                           f"({n_rows}, {TOPK_MARGIN})")
    if top5_scores.shape != (n_rows, TOPK_MARGIN):
        raise RuntimeError(f"top5_scores shape {top5_scores.shape} != "
                           f"({n_rows}, {TOPK_MARGIN})")
    if int((pred_baseline == EMPTY).sum()) != n_empty:
        raise RuntimeError(
            f"{npz_path}: pred_baseline has "
            f"{int((pred_baseline == EMPTY).sum()):,} EMPTY entries, recorded "
            f"n_empty is {n_empty:,}")
    if int((top5_ids[:, 0] == EMPTY).sum()) != n_empty:
        raise RuntimeError(
            f"{npz_path}: top5_ids[:, 0] has "
            f"{int((top5_ids[:, 0] == EMPTY).sum()):,} EMPTY entries, recorded "
            f"n_empty is {n_empty:,}")

    # --- meta sidecar vs. this run's own constants: catches an eval run against a
    # score-stage npz built under a different floor/top-k/row/label configuration ---
    floor_target, floor_target_provenance = _check_sidecar_floor_target(
        meta["floor_target"], meta_path)
    if meta["topk_margin"] != TOPK_MARGIN:
        raise RuntimeError(f"{meta_path} records topk_margin {meta['topk_margin']}, "
                           f"expected TOPK_MARGIN {TOPK_MARGIN} "
                           "(analysis.margin_diagnostic)")
    if meta["n_rows"] != n_rows:
        raise RuntimeError(f"{meta_path} records n_rows {meta['n_rows']:,}, but "
                           f"{npz_path} has {n_rows:,} rows")
    if meta["n_labels"] != reg["expected_labels"]:
        raise RuntimeError(f"{meta_path} records n_labels {meta['n_labels']}, "
                           f"expected {reg['expected_labels']} "
                           f"(BENCH_REGISTRY[{bench!r}]['expected_labels'])")

    # --- canonical language order + training-line counts N ---
    weights, langs, _lang_to_idx = _load_model_data(_ctx().model_path)
    del weights
    n_lang = len(langs)
    prf = pd.read_csv(PRF_CSV)
    if prf.lang.tolist() != langs:
        raise RuntimeError(f"{PRF_CSV} language order does not match "
                           "_load_model_data's canonical language list")
    N = prf.N.values.astype(np.int64)

    label_idx = np.unique(y)
    if len(label_idx) != reg["expected_labels"]:
        raise RuntimeError(f"{bench}: {len(label_idx)} distinct gold labels in "
                           f"{npz_path}, expected {reg['expected_labels']}")

    os.makedirs(OUT_DIAG_DIR, exist_ok=True)
    os.makedirs(OUT_TABLES_DIR, exist_ok=True)

    # --- acceptance gate: baseline only, before any gated-configuration number ---
    # Does the gate BIND this run? reg["paper_baseline_f1"] is the cell published
    # for the RELEASED model, so only the released model can be held to it; under
    # any other model the difference is the point of the run, and the comparison
    # withholds no number and fails no exit code. Same rule and same test as
    # analysis/paper_breakdowns.py's `gates_binding`.
    gates_binding = _ctx().is_default_model
    baseline_stats = _per_label_metrics(pred_baseline, y, label_idx, n_lang, n_rows)
    baseline_macro_f1 = baseline_stats["macro_f1"]
    diff = abs(baseline_macro_f1 - reg["paper_baseline_f1"])

    baseline_csv_path = os.path.join(OUT_DIAG_DIR, f"{bench}_baseline_per_label.csv")
    _write_per_label_csv(baseline_csv_path, label_idx, langs, N,
                         {"baseline": baseline_stats})

    if diff > ACCEPTANCE_TOL and gates_binding:
        gate_md_path = _gate_failure_md_path(bench)
        L = [
            f"# External benchmark E2 ({bench}): baseline acceptance gate FAILED\n",
            "Pre-registration: EXPERIMENTS_PLAN.md \"Camera-ready evaluation "
            "program (2026-08-06)\", E2; EXPERIMENTAL_SETUP.md \"Camera-ready "
            "reporting conventions\". Acceptance gate: the recomputed baseline "
            f"macro F1 must reproduce the paper's UniLID baseline cell for "
            f"{bench} ({reg['paper_baseline_f1']}) within ACCEPTANCE_TOL "
            f"({ACCEPTANCE_TOL}), evaluated before any floor-21 or gated-"
            "configuration number.\n",
            f"- Measured baseline macro F1: {baseline_macro_f1:.6f} (over the "
            f"{len(label_idx)} labels present in gold, {n_rows:,} rows).",
            f"- Paper value: {reg['paper_baseline_f1']}.",
            f"- Absolute difference: {diff:.6f} (tolerance {ACCEPTANCE_TOL}).",
            f"- Out-of-label-set baseline predictions: "
            f"{baseline_stats['n_out_of_set']:,} of {n_rows:,} rows.",
            f"- Empty-after-preprocess rows: {n_empty:,} of {n_rows:,}.",
            f"\nPer-label baseline F1: {baseline_csv_path}.",
            "\nNo floor-21 or gated-configuration number was computed. Per "
            "pre-registration, this discrepancy goes to the user before any "
            "further scoring.",
        ]
        with open(gate_md_path, "w") as f:
            f.write("\n".join(L) + "\n")
        print("\n".join(L))
        print(f"\nWrote {gate_md_path}, {baseline_csv_path}")
        sys.exit(1)

    # --- the run proceeds: the gate passed, or it does not bind this model ---
    if gates_binding:
        acceptance_line = (
            f"Acceptance gate: baseline macro F1 {baseline_macro_f1:.6f} vs paper "
            f"{reg['paper_baseline_f1']} (diff {diff:.6f}, tolerance "
            f"{ACCEPTANCE_TOL}) -- PASSED.\n")
    else:
        acceptance_line = _cross_model_baseline_message(
            bench, baseline_macro_f1, reg["paper_baseline_f1"], diff,
            _ctx().model_path) + "\n"
        print(acceptance_line.rstrip("\n"), flush=True)

    # --- remove any stale failure report from a prior run, so it is never mistaken
    # for the current state ---
    stale_gate_md_path = _gate_failure_md_path(bench)
    if os.path.exists(stale_gate_md_path):
        os.remove(stale_gate_md_path)
        if gates_binding:
            print(f"Removed stale {stale_gate_md_path} (the acceptance gate now "
                  "passes).")
        else:
            print(f"Removed stale {stale_gate_md_path} (it is a gate failure from "
                  "a run where the gate was binding; under this model the same "
                  "comparison is informational and every configuration below was "
                  "computed).")

    # --- floor-21 prediction is top5_ids[:, 0] (the top-k pass's own rank-1
    # candidate under the floor-21 matrix; see the module docstring) ---
    pred_floor21 = top5_ids[:, 0].astype(np.int64)
    floor21_stats = _per_label_metrics(pred_floor21, y, label_idx, n_lang, n_rows)

    # --- build the gated (floor21_gate / "gate_flat4_prox21") prediction, mirroring
    # analysis/gate_variants.py's _flat4_prox21_two_step_pred exactly, via the
    # shared _load_gate_thresholds / _gate_walk_and_merge helpers (also used by
    # run_selfcheck) ---
    thresholds = _load_gate_thresholds(langs, N)
    tau1, sha_tau1 = thresholds["tau1"], thresholds["sha_tau1"]
    tau2, sha_tau2 = thresholds["tau2"], thresholds["sha_tau2"]
    step2_idx = thresholds["step2_idx"]
    step1_langs = thresholds["step1_langs"]

    # agree_mask left None: pred_floor21 IS top5_ids[:, 0], so every row trivially
    # agrees with its own top-1; gate_variants.py's agree_mask carve-out is vacuous
    # here (see the module docstring).
    pred_gated, stats_a, stats_b = _gate_walk_and_merge(
        pred_floor21, top5_ids, top5_scores, N, tau1, tau2, step2_idx)

    gated_stats = _per_label_metrics(pred_gated, y, label_idx, n_lang, n_rows)

    # --- outputs ---
    per_label_csv_path = os.path.join(OUT_DIAG_DIR, f"{bench}_per_label.csv")
    _write_per_label_csv(per_label_csv_path, label_idx, langs, N,
                         {"baseline": baseline_stats, "floor21": floor21_stats,
                          "gated": gated_stats})

    metrics_rows = [
        ["baseline", f"{baseline_stats['macro_f1']:.4f}",
         f"{baseline_stats['macro_fpr'] * FPR_SCALE:.4f}",
         f"{baseline_stats['n_out_of_set']:,}"],
        ["floor21", f"{floor21_stats['macro_f1']:.4f}",
         f"{floor21_stats['macro_fpr'] * FPR_SCALE:.4f}",
         f"{floor21_stats['n_out_of_set']:,}"],
        ["gated", f"{gated_stats['macro_f1']:.4f}",
         f"{gated_stats['macro_fpr'] * FPR_SCALE:.4f}",
         f"{gated_stats['n_out_of_set']:,}"],
    ]
    metrics_md = to_markdown(
        metrics_rows, ["config", "macro F1", FPR_HEADER, "n_out_of_set"],
        caption=f"{bench}: macro F1 and macro FPR over the {len(label_idx)} "
                f"benchmark labels, {n_rows:,} rows")

    exam_rows = [
        ["A (N < HEAD_N)", f"{stats_a['n_examined']:,}", f"{stats_a['n_moved']:,}",
         f"{stats_a['n_blocked_by_proximity']:,}", f"{stats_a['n_no_cand']:,}"],
        ["B (flat-four)", f"{stats_b['n_examined']:,}", f"{stats_b['n_moved']:,}",
         f"{stats_b['n_blocked_by_proximity']:,}", f"{stats_b['n_no_cand']:,}"],
    ]
    exam_md = to_markdown(
        exam_rows,
        ["group", "examined", "moved", "blocked_by_proximity", "no_cand"],
        caption="Re-examination accounting (gated configuration)")

    try:
        git_commit = _git_commit()
    except RuntimeError:
        git_commit = "unknown"

    L = [
        f"# External benchmark E2 ({bench}): baseline / floor-21 / gated\n",
        "Pre-registration: EXPERIMENTS_PLAN.md \"Camera-ready evaluation program "
        "(2026-08-06)\", E2; EXPERIMENTAL_SETUP.md \"Camera-ready reporting "
        "conventions\" (no-refitting transfer test: every constant below is the "
        "promoted configuration's own training-derived value, applied unchanged "
        "to this benchmark).\n",
        f"Benchmark: {bench}, {n_rows:,} rows, {len(label_idx)} labels "
        f"(paper: {reg['expected_labels']}). {n_empty:,} rows empty after "
        "preprocess (scored as wrong under every configuration, the repo's EMPTY "
        "convention).\n",
        acceptance_line,
        "The floor-21 prediction reported here is the top-k pass's own rank-1 "
        "candidate under the floor-21 matrix (the same matrix the margins are "
        "scored from); analysis/gate_variants.py's own agree_mask carve-out is "
        "vacuous under that definition, and there is no separate best_of-scored "
        "floor-21 pass for this benchmark.\n",
        metrics_md,
        exam_md,
        "## Constants used\n",
        f"- ACCEPTANCE_TOL = {ACCEPTANCE_TOL} (this module; pre-registration)",
        f"- HEAD_N = {HEAD_N:,} (analysis.full_test_margin)",
        f"- RES_CAP = {RES_CAP:,} (analysis.hierarchical_pool)",
        f"- D3_PROX = {D3_PROX} (analysis.gate_variants)",
        f"- FLOOR_TARGET = {floor_target} ({floor_target_provenance})",
        f"- TOPK_MARGIN = {TOPK_MARGIN} (analysis.margin_diagnostic)",
        f"- {_TAU1_CSV} (group A thresholds, {len(step1_langs):,} "
        f"languages, sha256 {sha_tau1[:16]}...)",
        f"- {_TAU2_CSV} (group B thresholds, {N_FLAT4} languages, sha256 "
        f"{sha_tau2[:16]}...)",
        f"\nPer-label detail: {per_label_csv_path}.",
        f"\nGit commit: {git_commit}.",
    ]
    out_md_path = os.path.join(OUT_TABLES_DIR, f"external_bench_{bench}.md")
    with open(out_md_path, "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\nWrote {out_md_path}, {per_label_csv_path}")
    print(f"EVAL OK ({bench}): baseline F1 {baseline_stats['macro_f1']:.4f} / "
         f"floor21 F1 {floor21_stats['macro_f1']:.4f} / "
         f"gated F1 {gated_stats['macro_f1']:.4f}; baseline FPR "
         f"{baseline_stats['macro_fpr'] * FPR_SCALE:.4f}e-5 / floor21 FPR "
         f"{floor21_stats['macro_fpr'] * FPR_SCALE:.4f}e-5 / gated FPR "
         f"{gated_stats['macro_fpr'] * FPR_SCALE:.4f}e-5; group A "
         f"{stats_a['n_examined']:,} examined / {stats_a['n_moved']:,} moved; "
         f"group B {stats_b['n_examined']:,} examined / {stats_b['n_moved']:,} "
         "moved.")
    return out_md_path


# ---------------------------------------------------------------------------
# STAGE "selfcheck" (login node: no model load, no --bench; replays
# analysis/gate_variants.py's own banked stage-"topk" arrays through this module's
# _gate_walk_and_merge)
# ---------------------------------------------------------------------------

def run_selfcheck() -> None:
    """STAGE "selfcheck". Reads no benchmark .npz and loads no model. Replays
    analysis/gate_variants.py's own banked stage-"topk" outputs through this
    module's `_gate_walk_and_merge` (the identical function run_eval calls), and
    requires the result to reproduce gate_variants.py's actual production output,
    pred_gate_flat4_prox21.npy, on every line except, at most, the banked rows whose
    recomputed top-1 disagrees with the separately-scored pred_floor21.npy at that
    line (gate_variants.py's own agree_mask carve-out, mirrored here explicitly;
    see the module docstring). Aborts with counts and the first 10 differing line
    indices on any other mismatch; prints SELFCHECK OK on success."""
    lines_path = os.path.join(_ctx().scratch_dir, "gate_topk_lines.npy")
    ids_path = os.path.join(_ctx().scratch_dir, "gate_topk_ids.npy")
    scores_path = os.path.join(_ctx().scratch_dir, "gate_topk_scores.npy")
    fp_path = os.path.join(_ctx().scratch_dir, "gate_topk_fingerprint.json")
    pred_floor21_path = os.path.join(_ctx().scratch_dir, "pred_floor21.npy")
    pred_gated_path = os.path.join(_ctx().scratch_dir, "pred_gate_flat4_prox21.npy")
    for p in (lines_path, ids_path, scores_path, fp_path, pred_floor21_path,
             pred_gated_path, PRF_CSV, _TAU1_CSV, _TAU2_CSV):
        if not os.path.exists(p):
            raise FileNotFoundError(f"required artifact missing: {p}")

    with open(fp_path) as f:
        fp = json.load(f)

    lines = np.load(lines_path)
    ids = np.load(ids_path).astype(np.int64)
    # NOT upcast to float64, same reasoning as run_eval: gate_topk_scores.npy is
    # itself float32 (analysis.gate_variants), and the gap/D3_PROX arithmetic below
    # must match that source bit-for-bit for this check to mean anything.
    scores = np.load(scores_path)
    n_affected = lines.shape[0]
    if lines.ndim != 1 or lines.dtype != np.int64:
        raise RuntimeError(f"{lines_path} has unexpected shape/dtype "
                           f"{lines.shape}/{lines.dtype}")
    if n_affected != fp["n_affected"]:
        raise RuntimeError(f"{lines_path} has {n_affected} lines but {fp_path} "
                           f"records n_affected={fp['n_affected']}")
    if scores.dtype != np.float32:
        raise RuntimeError(f"{scores_path} has dtype {scores.dtype}, expected "
                           "float32 (analysis.gate_variants's own stage-\"topk\" "
                           "output dtype)")
    expect_shape = (n_affected, TOPK_MARGIN)
    if ids.shape != expect_shape or scores.shape != expect_shape:
        raise RuntimeError(f"gate_topk_ids/scores shape mismatch: ids {ids.shape}, "
                           f"scores {scores.shape}, expected {expect_shape}")

    weights, langs, _lang_to_idx = _load_model_data(_ctx().model_path)
    del weights
    n_lang = len(langs)
    prf = pd.read_csv(PRF_CSV)
    if prf.lang.tolist() != langs:
        raise RuntimeError(f"{PRF_CSV} language order does not match "
                           "_load_model_data's canonical language list")
    N = prf.N.values.astype(np.int64)
    if n_lang != fp["n_lang"]:
        raise RuntimeError(f"{PRF_CSV} has {n_lang} languages, "
                           f"gate_topk_fingerprint.json records {fp['n_lang']}")

    pred_floor21_full = np.asarray(
        np.lib.format.open_memmap(pred_floor21_path, mode="r")).astype(np.int64)
    if pred_floor21_full.shape != (TOTAL_LINES,):
        raise RuntimeError(f"{pred_floor21_path} shape {pred_floor21_full.shape} "
                           f"!= ({TOTAL_LINES},)")
    pred_gated_ref = np.asarray(
        np.lib.format.open_memmap(pred_gated_path, mode="r")).astype(np.int64)
    if pred_gated_ref.shape != (TOTAL_LINES,):
        raise RuntimeError(f"{pred_gated_path} shape {pred_gated_ref.shape} != "
                           f"({TOTAL_LINES},)")

    base_line_pred = pred_floor21_full[lines]
    if (base_line_pred < 0).any():
        raise RuntimeError(f"{pred_floor21_path} has a negative prediction at one "
                           "or more banked gate_topk_lines.npy indices")

    # agree_mask: gate_variants.py's own carve-out. Here, unlike run_eval,
    # base_line_pred (pred_floor21.npy) and ids[:, 0] (the topk stage's own
    # recomputation under a cached, re-verified floor-21 matrix) are genuinely
    # independently derived, so they need not agree on every row.
    top1 = ids[:, 0]
    agree_mask = top1 == base_line_pred
    n_disagree = int((~agree_mask).sum())
    measured_agree = float(agree_mask.mean())
    print(f"banked top-1 agreement with {pred_floor21_path}: {measured_agree:.8f} "
          f"({n_affected - n_disagree:,}/{n_affected:,} rows); "
          f"gate_topk_fingerprint.json records top1_agree_overall "
          f"{fp.get('top1_agree_overall', 'n/a')}.")

    thresholds = _load_gate_thresholds(langs, N)
    tau1, tau2 = thresholds["tau1"], thresholds["tau2"]
    step2_idx = thresholds["step2_idx"]

    pred_banked_gated, stats_a, stats_b = _gate_walk_and_merge(
        base_line_pred, ids, scores, N, tau1, tau2, step2_idx, agree_mask=agree_mask)

    selfcheck_pred = np.array(pred_floor21_full, dtype=np.int64)
    selfcheck_pred[lines] = pred_banked_gated

    diff_mask = selfcheck_pred != pred_gated_ref
    n_diff = int(diff_mask.sum())

    disagree_line_mask = np.zeros(TOTAL_LINES, dtype=bool)
    disagree_line_mask[lines[~agree_mask]] = True

    bad_diff_mask = diff_mask & ~disagree_line_mask
    n_bad_diff = int(bad_diff_mask.sum())
    if n_bad_diff:
        bad_idx = np.where(bad_diff_mask)[0][:10].tolist()
        raise RuntimeError(
            f"SELFCHECK FAILED: {n_bad_diff:,} of {TOTAL_LINES:,} lines differ "
            f"from {pred_gated_path} outside the {n_disagree:,}-row banked "
            f"top-1-disagreement set (first 10 differing line indices: {bad_idx}); "
            f"{n_diff:,} lines differ from {pred_gated_path} in total. Group A: "
            f"{stats_a['n_examined']:,} examined / {stats_a['n_moved']:,} moved. "
            f"Group B: {stats_b['n_examined']:,} examined / "
            f"{stats_b['n_moved']:,} moved.")

    n_diff_within_disagree = int((diff_mask & disagree_line_mask).sum())
    print(f"SELFCHECK OK: {n_affected:,} banked rows replayed through this "
          f"module's own _gate_walk_and_merge; {n_diff:,} of {TOTAL_LINES:,} "
          f"lines differ from {pred_gated_path}, all within the {n_disagree:,}-row "
          f"banked top-1-disagreement set ({n_diff_within_disagree:,} of those "
          "actually differ; the remainder still reproduce because a disagreeing "
          "row is only excused from the equality requirement, not required to "
          f"differ). Group A: {stats_a['n_examined']:,} examined / "
          f"{stats_a['n_moved']:,} moved. Group B: {stats_b['n_examined']:,} "
          f"examined / {stats_b['n_moved']:,} moved.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="E2 external-benchmark (UDHR / FLORES-200) transfer test: "
                    "stage 'score' loads the full model and must run on SLURM; "
                    "stage 'eval' reads the scored .npz and runs on the login "
                    "node; stage 'selfcheck' replays analysis/gate_variants.py's "
                    "own banked arrays through this module's gate logic and needs "
                    "neither the model nor --bench.")
    parser.add_argument("--stage", required=True,
                        choices=["score", "eval", "selfcheck"])
    parser.add_argument("--out-dir", default=None)
    add_arguments(parser)
    parser.add_argument("--bench", choices=list(BENCH_REGISTRY),
                        help="required for --stage score/eval; ignored for "
                             "--stage selfcheck")
    args = parser.parse_args()
    configure(args.model_path, args.scratch_dir, args.out_dir)
    if args.stage == "selfcheck":
        if args.bench is not None:
            print("--bench is ignored for --stage selfcheck", flush=True)
        run_selfcheck()
    else:
        if args.bench is None:
            raise RuntimeError(f"--bench is required for --stage {args.stage}")
        if args.stage == "score":
            run_score(args.bench)
        else:
            run_eval(args.bench)


if __name__ == "__main__":
    main()
