"""E3: Mistral-Nemo variant evaluation (EXPERIMENTS_PLAN.md, "Camera-ready
evaluation program (2026-08-06)", E3; conventions fixed in EXPERIMENTAL_SETUP.md,
"Camera-ready reporting conventions"). Training is finished (job 3028465): the
packed model is analysis.mistralnemo_constants.PACKED_MODEL_PATH
(glotlid_mistralnemo_fp64.unilid), a per-language fixed-vocabulary EM retrain
against the pinned Mistral-Nemo-Base-2407 tokenizer snapshot
(analysis.mistralnemo_constants.SNAPSHOT_HASH), reusing the recorded Apertus
fp64 pipeline exactly. This module builds the evaluation pipeline: baseline
scoring, the flat-language rule (recomputed for this model's own weight
matrix), floor-21 tau recalibration, the gated prediction, and the final
metrics report, mirroring the Apertus branch precedent (analysis/
full_test_eval_131k.py, jobs 2883222/2911700) and the promoted gate's own
machinery (analysis/gate_variants.py, analysis/external_bench_eval.py,
analysis/solo_gates.py) with NO refitting: HEAD_N, RES_CAP, D3_PROX,
FLOOR_TARGET, MARGIN_Q, and the calibration constants are the promoted
configuration's own training-derived values, applied unchanged to this
retrain (the same no-refitting transfer-test philosophy EXPERIMENTAL_SETUP.md
records for E2). Only the flat-language SET and the per-language tau values
are recomputed, because both depend on the variant's own weight matrix.

Six CLI stages, dispatched by --stage:

STAGE "baseline" (SLURM, 100G): one full-pool scoring pass of the variant
under its unmodified weight matrix, chunked and resumable exactly like
analysis/full_test_eval.py. y_true.npy is REUSED READ-ONLY from
analysis.full_test_eval.SCRATCH_DIR (the base model's scratch dir): the test
file and line space are identical for every model variant (Apertus 131k/200k
precedent, analysis/full_test_eval_131k.py), so the recorded label indices
are valid for the variant too, gated on the variant's language order being
IDENTICAL, position for position, to the canonical order
analysis.transfer_sweep._load_model_data() returns for the base model
(abort, not a label-string join, matching full_test_eval_131k.py's own
choice). Output: pred_nemo_baseline.npy (int16), a fingerprint json (model
file sha256, weight-matrix-only sha256, tokenizer snapshot hash, langs sha,
chunk size), and a progress json for chunk-level resume.

STAGE "calibval" (SLURM, 100G; can run after "baseline" in the same job):
scores the variant, unmodified matrix, on the retired 250,000-line
validation half of the seed-42 500k sample (the EXCLUDED=-2 lines of
y_true.npy) ONLY. This feeds the flat-language rule's magnet_ratio
(analysis/diagnostic.py's construction: false positives into a language on
this held-out half, divided by its true support there plus one), the same
validation half analysis/diagnostic.py itself uses for the base model
(never the test half, never used for anything else). Output:
pred_nemo_calibval.npy and calibval_line_indices.npy, both length 250,000,
same order.

STAGE "flatrule" (login node): recomputes zH (within-script entropy
z-score) for the variant's own weight matrix, reusing
analysis.diagnostic._probs_and_logprobs by import, with entropy H computed
directly as -(P * logP).sum(axis=1) rather than via
analysis.diagnostic._sym_kl_matrix (verified bit-identical to that
function's own entropy return, without its unused n_lang x n_lang symKL
matmul), and recomputes
support_val/fp_val/magnet_ratio from the calibval predictions via
analysis.diagnostic._empirical_magnet, also reused by import. Applies the
recorded category rule verbatim (is_magnet = (zH > ZH_MAGNET and
magnet_ratio > MAGNET_RATIO_MIN) or zH > ZH_EXTREME, all three constants
imported from analysis.diagnostic, never redefined). The variant's flat set
is every language with is_magnet True AND N >= HEAD_N (the languages the
gate's step 2 re-examines; step 1, N < HEAD_N, already covers every N <
HEAD_N language regardless of flatness). Unlike the base model's flat-four
(sco_Latn, bjn_Latn, arg_Latn, vls_Latn, fixed by the promoted
configuration's own pre-registration), the variant's flat-set SIZE is not
assumed: it is whatever the recomputed rule yields for this weight matrix,
including possibly zero, and the "tau" and "topk" stages below both handle
an empty flat set without special-casing. Output:
outputs/diagnostic/mistralnemo_flat_set.csv and
outputs/tables/mistralnemo_flat_set.md. This stage has no acceptance gate
of its own (it is descriptive, not gated), but its output CSV's sha256 is
embedded in the "tau" and "topk" stages' own fingerprints, so a later stage
that reads a flat set built by different code, or missing entirely, is
caught there.

STAGE "tau" (SLURM, 100G): builds the variant's floor-21 matrix
(analysis.floor_equalization.build_equalized_weights at FLOOR_TARGET) and
writes fingerprint_floor21_mistralnemo.json (base and floor-21 matrix
sha256, mirroring analysis/full_test_floor21.py's own fp dict). Then
per-language tau recalibration under that matrix, mirroring
analysis/solo_gates.py's run("floor21") exactly for group A (every language
with N < HEAD_N, size-adaptive quantile q_L = MARGIN_Q * (1 -
min(N,HEAD_N)/HEAD_N)) and analysis/gate_variants.py's
_calibrate_flat4_tau5 exactly for group B (the variant's own flat set from
the "flatrule" stage, fixed MARGIN_Q-th percentile, abort (not exclude) on
a calibration shortfall, the same rationale: group B is defined as N >=
HEAD_N, so a shortfall there indicates a wiring error). Calibration
constants (CALIB_MAX, CALIB_SEED, MIN_CALIB_LINES, MARGIN_Q, TOPK_MARGIN,
CORPUS_DIR) are imported from analysis.margin_diagnostic, unchanged.
Outputs: outputs/diagnostic/tau_mistralnemo_floor21_gate.csv (group A) and
outputs/diagnostic/tau_mistralnemo_flat.csv (group B), same six columns as
the base model's tau CSVs (lang, n_scoreable, n_self_won, tau, excluded,
cause).

STAGE "topk" (SLURM, 100G): ONE full-pool pass under the variant's floor-21
matrix. Unlike the base model's pipeline, where a full best_of pass
(analysis/full_test_floor21.py) and a separate top-5-banking pass over an
already-known affected subset (analysis/gate_variants.py's _run_topk) are
two different jobs, this stage folds both into one pass: every kept line is
scored with top_k_of_cached_weight_sets_batch(TOPK_MARGIN), rank 0 of that
call is written to pred_nemo_floor21.npy for EVERY kept line (the floor-21
prediction used throughout this module IS the top-k pass's own rank-1
candidate, the same definition analysis/external_bench_eval.py uses for its
benchmarks, stated explicitly here per the pre-registration's "state which
and be consistent"), and the FULL top-TOPK_MARGIN candidate list is banked
(gate_topk_lines_nemo.npy/gate_topk_ids_nemo.npy/gate_topk_scores_nemo.npy)
only for lines whose rank-1 falls in the expanded label set (N < HEAD_N, or
the variant's flat set from "flatrule"), mirroring the ARRAY SHAPES and
sentinel conventions of analysis.gate_variants._run_topk (affected-only
banking bounds memory; -1/-inf fill for short candidate lists;
n_short_cands tracked) while replacing its two-pass ancestry with one.
Because both the floor-21 prediction and the banked top-1 come from the
same call at the same line, they are identical by construction, not merely
in ~99.99% agreement the way two independently-scored passes are in the
base model's pipeline; this stage asserts that identity as np.array_equal,
a strictly stronger check than analysis.gate_variants._run_topk's
TOP1_AGREE_MIN bar. Chunked (CHUNK_LINES, matching analysis/full_test_eval.py)
and resumable at chunk granularity: each chunk's affected-row banking is
written to its own file under a per-chunk scratch subdirectory as it
completes, and a finalization step (idempotent, run at the end of every
invocation) concatenates whatever chunks are done once all chunks are done.
The base model's affected-line-count sanity range (2,000,000-2,600,000,
analysis.gate_variants._run_topk) is NOT reused here: it was measured for
that model's own expanded label set and would be a wrongly-transplanted
magic number for a different flat set; only a basic 0 < n_affected < n_kept
sanity check applies.

STAGE "eval" (login node): builds the gated prediction via
analysis.external_bench_eval._gate_walk_and_merge, imported and called
unmodified (D3_PROX, RES_CAP, HEAD_N are the promoted configuration's own
constants, applied unchanged; only the two tau arrays and the flat-set
index array are this variant's own). base_pred at a banked line is
pred_nemo_floor21.npy at that line (mirroring
analysis.external_bench_eval.run_selfcheck's agree_mask construction, since
here too the top-5 arrays were only banked for a subset of lines, not
every line, unlike run_eval's external-benchmark case where every row is
banked); by the "topk" stage's own construction this must equal the banked
rank-1 exactly, and this stage re-asserts that identity rather than
assuming it (a live cross-file integrity check: it would only fail if one
of the two files were regenerated independently after the other). Then:
full-pool and judge-part (seed-301 split re-derived exactly as
analysis/paper_eval.py does, reusing its constants from
analysis.combined_evidence and analysis.balanced_split unchanged) macro F1 +
macro FPR for {nemo_baseline, nemo_floor21, nemo_gated}; a paired bootstrap
(B=10,000, seed 0, analysis.combined_evidence.BOOT_B/BOOT_SEED) of
(nemo_gated - nemo_baseline) on the judge part, mirroring
analysis/paper_eval.py's bootstrap block. A comparability row states the
paper's UniLID-Mistral-Nemo full-pool cell (paper/tables/lid_main.tex:
F1 .912, FPR 1.84e-5, raw scientific notation, NOT the repo's internal
x1e5-scaled table convention) next to the measured nemo_baseline full-pool
cell; this is a RECORDED MEASUREMENT, not a gate (the paper's row is the
paper team's own training run; this one is an independent retrain from the
same recipe, so rough proximity is expected, not equality, per the
pre-registration). Gates that DO abort the run: language order; the
y_true.npy reuse checks (shape, no UNSEEN, exactly 250,000 EXCLUDED lines);
EXPECTED_KEPT (analysis.metric_decomposition); judge/derivation split sizes
against the stored seed-301 record; the sentinel guard (no value < -1 on any
kept line of any of the three prediction arrays); the weight-matrix sha
binding between the baseline and floor-21 fingerprints; the topk stage's own
fingerprint (gate_topk_fingerprint_nemo.json) against the freshly computed
flat-set/tau CSV shas, langs_sha256, head_n, topk_margin, and the banked
array shapes; gate-group membership for every banked line's floor-21 base
prediction. The 32 degeneracy-flagged
rows (outputs/tables/degenerate_rows_mistralnemo.md,
analysis/degeneracy_scan_mistralnemo.py) are carried as a listed caveat in
the report and flagged in the per-language CSVs, not gated on: the
adjudication already on record (EXPERIMENTS_CHRONOLOGICAL.md, 2026-08-07)
treats them as an accepted model property (base-vocab script coverage), not
a defect to reject on. Outputs: outputs/tables/mistralnemo_eval.md,
outputs/tables/mistralnemo_eval.tex, and
outputs/diagnostic/mistralnemo_per_lang_f1_{fullpool,judge}.csv.

New constants introduced in this module are listed and justified just after
the imports, mirroring analysis/mistralnemo_constants.py's own convention.
Every constant with a precedent value (HEAD_N, RES_CAP, D3_PROX,
FLOOR_TARGET, MARGIN_Q, CALIB_MAX, CALIB_SEED, MIN_CALIB_LINES, TOPK_MARGIN,
ZH_MAGNET, ZH_EXTREME, MAGNET_RATIO_MIN, BOOT_B, BOOT_SEED, the seed-301
split constants) is IMPORTED from its origin module, never re-typed as a
literal here.

Which model this scores, where it writes, and at what floor (2026-08-23):
--model / --scratch-dir / --base-scratch / --out-dir / --floor-target, resolved
through analysis.model_context by configure(). With no flags nothing changes:
this chain's own packed model, its own scratch root, the outputs/ tree and
FLOOR_TARGET = analysis.full_test_floor21.FLOOR_TARGET, byte for byte as before.
See the input inventory below for what a non-default model moves and what it
deliberately does not.
"""
# ---------------------------------------------------------------------------
# INPUT INVENTORY (checked line by line 2026-08-23, following
# analysis/paper_eval.py's own precedent; the classification is what a
# non-default --model changes and what it must not).
#
# (a) MODEL-DERIVED -- must come from the run's own model / scratch root /
#     output root, and must abort naming the artifact when it is absent there:
#       - PACKED_MODEL_PATH, the .unilid file itself                [--model]
#       - everything under SCRATCH_DIR_NEMO: the three prediction memmaps, the
#         baseline/calibval/floor-21/top-k fingerprints, the progress files and
#         the per-chunk top-k banking       [--scratch-dir, via resolve()]
#       - <out-root>/diagnostic/mistralnemo_flat_set.csv, both tau CSVs, and
#         the four reports/tables this module writes. The flat set is recomputed
#         from THIS model's weight matrix, the tau values are calibrated under
#         THIS model's floor-21 matrix, and the tables are this model's numbers;
#         all nine move with --out-dir       [--out-dir, via resolve_out_root()]
#       - <out-root>/tables/degenerate_rows_mistralnemo.md (DEGENERACY_OUT_MD),
#         READ by the eval stage for the degeneracy caveat. It is a scan of the
#         packed model's own weight rows, so it is model-derived and moves with
#         --out-dir; analysis/degeneracy_scan_mistralnemo.py must be run for the
#         run's own model first. It is NEVER read from the released tree while a
#         different model is scored: the eval stage aborts naming the missing
#         path instead.
#       - FLOOR_TARGET, the floor-21 clamp constant. Not a file, but the same
#         hazard: a measured per-model selection whose module default belongs to
#         the released chain, so a non-default model must state it explicitly
#         (--floor-target) and it is cross-checked against the record that
#         selected it                                       [--floor-target]
#
# (b) CORPUS-DERIVED / MODEL-INVARIANT -- keeps its shared location under a
#     non-default model, each with the reason it cannot carry model information:
#       - <base-scratch>/y_true.npy: the label index per test line, a property of
#         the test file and the canonical language order, not of any model
#         (verified bit-identical between the released and corrected base runs
#         over all 45,627,279 entries, 2026-08-18). --base-scratch moves it
#         anyway so a corrected run reads nothing from the released tree.
#       - VAL_MASK (outputs/diagnostic/val_mask.npy): the position-parity mask
#         over the seed-42 500k sample. Re-derived here from DEFAULT_SAMPLE_SIZE
#         and required to match bit-for-bit before use (_val_lines_sorted), so a
#         divergence aborts rather than importing another run's split.
#       - DRAW_DIR/val_lines_seed{101,201}.npy: line-index draws over the test
#         file, drawn from the corpus by analysis/balanced_split.py.
#       - SPLIT_PATH (rule_split_seed301.npz, under the RELEASED base model's
#         scratch root and not moved by --base-scratch): re-derived in
#         _load_judge_split from THIS run's own kept pool plus the two draws and
#         RULE_SPLIT_SEED/RULE_SPLIT_FRACTION, and required to match
#         bit-for-bit before use, so it cannot import the released run's line set.
#       - CORPUS_DIR/{lang}_train.txt (analysis.margin_diagnostic): the training
#         corpus the tau stage calibrates own-train margins on.
#       - analysis.transfer_sweep._load_train_counts(): per-language TRAINING
#         line counts, a corpus property.
#       - _canonical_langs(): the BASE model's language order, deliberately the
#         base model's, because it is the order y_true.npy's indices are aligned
#         to; the variant's own order is then required to be identical position
#         for position (_verify_variant_langs) or the run aborts.
#
# (c) CONFIG CONSTANTS: HEAD_N, RES_CAP, D3_PROX, MARGIN_Q, CALIB_MAX,
#     CALIB_SEED, MIN_CALIB_LINES, TOPK_MARGIN, ZH_MAGNET, ZH_EXTREME,
#     MAGNET_RATIO_MIN, BOOT_B, BOOT_SEED, the seed-301 split sizes, TOTAL_LINES,
#     EXPECTED_KEPT. Per the E3 pre-registration these transfer UNCHANGED to any
#     variant (the no-refitting transfer test), so they are not per-model.
#     PAPER_MISTRALNEMO_F1_FULLPOOL / _FPR_FULLPOOL are the exception: they are
#     the paper team's own Mistral-Nemo cell, compared against as a recorded
#     measurement, never as a gate.
#
# NOT READ, here or transitively: PRF_CSV, outputs/diagnostic/lang_diagnostic.csv
# (read by analysis/degeneracy_scan_mistralnemo.py, not by this module), and no
# artifact of the base model's own gate chain (pred_floor21*.npy,
# fingerprint_floor21.json, gate_topk_*.npy, tau_flat4.csv) -- this module builds
# and calibrates its own floor-21 matrix and its own tau CSVs from scratch. The
# ONE exception is the read-only floor-target cross-check below, which opens
# <base-scratch>/fingerprint_floor21.json for its `floor_target` field alone.
# ---------------------------------------------------------------------------
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import subprocess

import numpy as np
import pandas as pd

from analysis.balanced_split import OUT_DIR as DRAW_DIR, SEEDS, TEST_SEED
from analysis.carried_set_comparison import EXPECTED_REMAINDER
from analysis.combined_evidence import (BOOT_B, BOOT_SEED, EXPECTED_DERIVATION,
                                        EXPECTED_JUDGE, RULE_SPLIT_FRACTION,
                                        RULE_SPLIT_SEED, SPLIT_PATH)
from analysis.config import DEFAULT_SAMPLE_SIZE, TEST_FILE, TOTAL_LINES
from analysis.diagnostic import (MAGNET_RATIO_MIN, ZH_EXTREME, ZH_MAGNET,
                                 _empirical_magnet, _probs_and_logprobs)
from analysis.external_bench_eval import _gate_walk_and_merge
from analysis.floor_equalization import (build_equalized_weights,
                                         _special_columns,
                                         verify_one_sided_clamp)
from analysis.format_utils import to_latex, to_markdown
from analysis.full_test_eval import (CHUNK_LINES, EMPTY, EXCLUDED, SCRATCH_DIR
                                     as BASE_SCRATCH, UNSEEN, _parse_line,
                                     _sample_line_indices)
from analysis.full_test_floor21 import FLOOR_TARGET as DEFAULT_FLOOR_TARGET
from analysis.full_test_margin import HEAD_N
from analysis.gate_variants import (D3_PROX, SCORE_BATCH_MAX, _load_tau_csv,
                                    _read_wanted_lines)
from analysis.hierarchical_pool import RES_CAP, VAL_MASK
from analysis.margin_diagnostic import (CALIB_MAX, CALIB_SEED, CORPUS_DIR,
                                        MARGIN_Q, MIN_CALIB_LINES, TOPK_MARGIN,
                                        _gap, _topk_batch)
from analysis.metric_decomposition import EXPECTED_KEPT, _per_lang_stats
from analysis.mistralnemo_constants import (DEGENERACY_OUT_MD,
                                            EXPECTED_TOKENIZER_SHA256,
                                            PACKED_MODEL_PATH, SNAPSHOT_HASH,
                                            TOKENIZER_JSON)
from analysis.model_context import (DEFAULT_OUT_ROOT, UnsafeModelContext,
                                    add_arguments, resolve, resolve_out_root)
from analysis.paper_eval import (FPR_HEADER, FPR_SCALE, FULLPOOL_INSTRUMENT,
                                 JUDGE_INSTRUMENT, _macro_fpr)
from analysis.sample_data import load_sample
from analysis.transfer_sweep import (_load_model_data, _load_train_counts,
                                     _load_unilid_model)

# ---------------------------------------------------------------------------
# New constants (none existed before the E3 evaluation build; every one is
# flagged in the implementing agent's report per the project's magic-number
# rule). Constants with a precedent value live in the modules imported above
# and are reused, not redefined, here.
# ---------------------------------------------------------------------------

# All writable scoring state for this variant. Distinct from BASE_SCRATCH
# (full_test_eval/, the base model's own scratch dir, read-only from this
# module's point of view except for y_true.npy reuse).
# The Mistral-Nemo vocabulary's special-token ids (preflight_mistralnemo's
# verified mapping: <unk> 0, <s> 1, </s> 2, <pad> 10). NOT the contiguous 0:4
# of the base model's packing; column 3 here is an ordinary token.
SPECIAL_COLS_NEMO = (0, 1, 2, 10)

SCRATCH_DIR_NEMO = ("/capstor/scratch/cscs/cmeister747/unilid_analysis/"
                    "full_test_eval_mistralnemo")

# Retired validation half of the seed-42 500k sample: exactly half of
# DEFAULT_SAMPLE_SIZE by the position-parity split analysis/full_test_eval.py
# and analysis/diagnostic.py both use. Not a bare literal: derived from the
# imported DEFAULT_SAMPLE_SIZE so it cannot silently drift from the sample
# size those two modules assume.
EXPECTED_VAL_LINES = DEFAULT_SAMPLE_SIZE // 2

# Scratch-side artifact names (SCRATCH_DIR_NEMO).
FP_BASELINE_PATH = os.path.join(SCRATCH_DIR_NEMO, "fingerprint_baseline.json")
PROGRESS_BASELINE_PATH = os.path.join(SCRATCH_DIR_NEMO, "progress_baseline.json")
PRED_NEMO_BASELINE = os.path.join(SCRATCH_DIR_NEMO, "pred_nemo_baseline.npy")

FP_CALIBVAL_PATH = os.path.join(SCRATCH_DIR_NEMO, "fingerprint_calibval.json")
CALIBVAL_IDX_PATH = os.path.join(SCRATCH_DIR_NEMO, "calibval_line_indices.npy")
PRED_NEMO_CALIBVAL = os.path.join(SCRATCH_DIR_NEMO, "pred_nemo_calibval.npy")

FP_FLOOR21_NEMO_PATH = os.path.join(SCRATCH_DIR_NEMO,
                                    "fingerprint_floor21_mistralnemo.json")

PROGRESS_TOPK_PATH = os.path.join(SCRATCH_DIR_NEMO, "progress_topk.json")
TOPK_CHUNKS_DIR = os.path.join(SCRATCH_DIR_NEMO, "topk_partial_nemo")
# Written before the first chunk of a topk run is scored (atomic tmp+replace)
# and re-verified against the current run's fp_base on every subsequent
# invocation, so a resumed run whose configuration changed since the partial
# chunk files were written is caught before progress_topk.json's done-chunk
# list is trusted, rather than silently mixing chunks scored under two
# different configurations.
CHUNKS_FP_PATH = os.path.join(TOPK_CHUNKS_DIR, "chunks_fingerprint.json")
GATE_TOPK_LINES_NEMO = os.path.join(SCRATCH_DIR_NEMO, "gate_topk_lines_nemo.npy")
GATE_TOPK_IDS_NEMO = os.path.join(SCRATCH_DIR_NEMO, "gate_topk_ids_nemo.npy")
GATE_TOPK_SCORES_NEMO = os.path.join(SCRATCH_DIR_NEMO, "gate_topk_scores_nemo.npy")
GATE_TOPK_FP_NEMO = os.path.join(SCRATCH_DIR_NEMO,
                                 "gate_topk_fingerprint_nemo.json")
PRED_NEMO_FLOOR21 = os.path.join(SCRATCH_DIR_NEMO, "pred_nemo_floor21.npy")
PRED_NEMO_GATED = os.path.join(SCRATCH_DIR_NEMO, "pred_nemo_gated.npy")

# ---------------------------------------------------------------------------
# Model context
#
# SCRATCH_DIR_NEMO is itself a symlink into /capstor/store/cscs/swissai, and
# every path above is derived from it at import time, so a run against corrected
# Mistral-Nemo weights would have written through those symlinks and replaced the
# E3 artifacts of record. configure() re-resolves the pair through
# analysis.model_context (which refuses a non-default model paired with a
# store-backed root) and re-derives the paths. The defaults are this chain's own
# packed model and root, not the base model's, so the ordinary invocation is
# unaffected.
# ---------------------------------------------------------------------------
# The same hazard has two more axes, both fixed here:
#
#   * the nine repo-side artifacts below were bare "outputs/..." literals. Those
#     nine files are the RELEASED model's E3 record and outputs/tables/
#     mistralnemo_eval.md is cited by paper/tables/calibrated_nemo.tex, so the
#     flatrule/tau/eval stages would have overwritten the published record in
#     place for any --model. They now resolve under an --out-dir routed through
#     analysis.model_context.resolve_out_root, which refuses a non-default model
#     paired with the default root, anything inside it, or a store-backed root.
#
#   * FLOOR_TARGET was imported from analysis.full_test_floor21 and used
#     unchanged by the tau/topk/eval stages. That module constant (-21.0) is the
#     RELEASED chain's guard-selected value; the corrected chain's own round-grid
#     sweep selected a different one (-17.0, recorded in
#     full_test_eval_corrected/fingerprint_floor21.json, 2026-08-18). A corrected
#     run would therefore have built and calibrated against the wrong clamp and
#     produced floor-21/gated numbers indistinguishable in the tables from a
#     correctly-targeted run. It is now --floor-target, mandatory for a
#     non-default model and cross-checked against the record that selected it.
# ---------------------------------------------------------------------------
DEFAULT_PACKED_MODEL_PATH = PACKED_MODEL_PATH
DEFAULT_SCRATCH_DIR_NEMO = SCRATCH_DIR_NEMO
DEFAULT_BASE_SCRATCH = BASE_SCRATCH

# The floor-21 clamp target actually in force. Rebound by configure(); the
# imported DEFAULT_FLOOR_TARGET stays the released chain's own value and is what
# a run with no --floor-target gets, so the default run is unchanged. It is a
# module global rather than a parameter threaded through the stage functions for
# the same reason PACKED_MODEL_PATH/SCRATCH_DIR_NEMO are: every stage entry point
# in this module takes no arguments and reads its configuration from globals.
FLOOR_TARGET = DEFAULT_FLOOR_TARGET

# The base model's floor-21 fingerprint, by name only. Read for exactly one
# field (`floor_target`) by _resolve_floor_target, to cross-check the value a run
# was given against the record of the sweep that selected it.
BASE_FLOOR21_FP_NAME = "fingerprint_floor21.json"


def _resolve_floor_target(ctx, floor_target: float = None,
                          base_scratch: str = None) -> float:
    """The floor-21 clamp target this run builds and calibrates at.

    The module default is the RELEASED chain's guard-selected -21.0. It is not a
    universal constant: it is a per-model measured selection, and the corrected
    base model's own round-grid sweep selected -17.0. So:

      * the default model keeps the module default (nothing changes);
      * any other model MUST state the value, because inheriting the released
        chain's constant would produce floor-21 and gated numbers for the wrong
        clamp, indistinguishable in the output tables from a correct run;
      * whenever --base-scratch is given, the value is cross-checked against the
        `floor_target` recorded in that root's own fingerprint_floor21.json, the
        record of the sweep that selected it. A mismatch names both numbers and
        aborts; a missing fingerprint aborts too for a non-default model, rather
        than proceeding on an unverified constant.
    """
    if floor_target is None:
        if not ctx.is_default_model:
            raise UnsafeModelContext(
                f"refusing to run {ctx.model_path} without an explicit "
                f"--floor-target.\n"
                f"  module default: {DEFAULT_FLOOR_TARGET} "
                f"(analysis.full_test_floor21.FLOOR_TARGET)\n"
                "That default is the RELEASED chain's guard-selected clamp, not "
                "a universal constant: the floor target is a measured "
                "per-model selection (the corrected base model's own round-grid "
                "sweep selected -17.0, recorded in its "
                f"{BASE_FLOOR21_FP_NAME}). Building this variant's floor-21 "
                "matrix at the released chain's value would produce floor-21 "
                "and gated numbers for the wrong clamp that look exactly like "
                "correct ones in the tables. State the value explicitly.")
        floor_target = DEFAULT_FLOOR_TARGET
    floor_target = float(floor_target)

    if base_scratch is None:
        if not ctx.is_default_model:
            raise UnsafeModelContext(
                f"--floor-target {floor_target} was given for "
                f"{ctx.model_path} with no --base-scratch, so there is nothing "
                f"to check it against.\n"
                "The floor target is a measured selection and its record is the "
                f"base run's own {BASE_FLOOR21_FP_NAME}. Pass --base-scratch "
                "pointing at that model's base scratch root so the value can be "
                "verified before any matrix is built.")
        return floor_target

    fp_path = os.path.join(base_scratch, BASE_FLOOR21_FP_NAME)
    if not os.path.exists(fp_path):
        if not ctx.is_default_model:
            raise UnsafeModelContext(
                f"cannot verify --floor-target {floor_target} for "
                f"{ctx.model_path}: {fp_path} does not exist.\n"
                "That file is the record of the floor sweep that selected the "
                "value. Run the base model's floor-21 stage against "
                f"{base_scratch} first, or correct --base-scratch. This run "
                "will not proceed on an unverified clamp target.")
        return floor_target
    with open(fp_path) as f:
        recorded = json.load(f).get("floor_target")
    if recorded is None:
        raise RuntimeError(
            f"{fp_path} records no floor_target field, so --floor-target "
            f"{floor_target} cannot be verified against it.")
    if float(recorded) != floor_target:
        raise UnsafeModelContext(
            f"floor-target mismatch against the base run's own record.\n"
            f"  --floor-target: {floor_target}\n"
            f"  {fp_path} records floor_target: {float(recorded)}\n"
            "The floor target is a measured per-model selection; these two "
            "disagree, so one of them is wrong and this run would build its "
            "floor-21 matrix at a clamp the base sweep did not select.")
    return floor_target


def configure(model_path: str = None, scratch_dir: str = None,
              base_scratch: str = None, out_dir: str = None,
              floor_target: float = None):
    """Resolve the (packed model, scratch root, output root, floor target) and
    re-derive every path.

    ``base_scratch`` names the base model's output root, which this chain reads
    y_true.npy from. That array is model-independent (verified 2026-08-18:
    bit-identical between the released and corrected base runs over all
    45,627,279 entries), so the default is safe, but a corrected run should point
    at its own root so it reads nothing from the released model's directory. It
    is also where the floor-target cross-check finds its record.

    ``out_dir`` is the repo-side counterpart of ``scratch_dir``: the nine tables,
    CSVs and .tex fragments this chain writes, plus the degeneracy scan it reads.
    ``floor_target`` is the floor-21 clamp; see _resolve_floor_target.
    """
    global PACKED_MODEL_PATH, SCRATCH_DIR_NEMO, BASE_SCRATCH
    global FP_BASELINE_PATH, PROGRESS_BASELINE_PATH, PRED_NEMO_BASELINE
    global FP_CALIBVAL_PATH, CALIBVAL_IDX_PATH, PRED_NEMO_CALIBVAL
    global FP_FLOOR21_NEMO_PATH, PROGRESS_TOPK_PATH, TOPK_CHUNKS_DIR
    global CHUNKS_FP_PATH, GATE_TOPK_LINES_NEMO, GATE_TOPK_IDS_NEMO
    global GATE_TOPK_SCORES_NEMO, GATE_TOPK_FP_NEMO
    global PRED_NEMO_FLOOR21, PRED_NEMO_GATED
    global OUT_ROOT, FLAT_SET_CSV, FLAT_SET_MD, TAU_FLOOR21_NEMO_CSV
    global TAU_FLAT_NEMO_CSV, TAU_BUILD_MD, OUT_MD, OUT_TEX
    global OUT_CSV_FULLPOOL, OUT_CSV_JUDGE, DEGENERACY_MD
    global FLOOR_TARGET

    ctx = resolve(model_path, scratch_dir,
                  default_model=DEFAULT_PACKED_MODEL_PATH,
                  default_scratch=DEFAULT_SCRATCH_DIR_NEMO,
                  purpose="Mistral-Nemo variant scoring")
    PACKED_MODEL_PATH = ctx.model_path
    SCRATCH_DIR_NEMO = d = ctx.scratch_dir
    BASE_SCRATCH = base_scratch or DEFAULT_BASE_SCRATCH
    FP_BASELINE_PATH = os.path.join(d, "fingerprint_baseline.json")
    PROGRESS_BASELINE_PATH = os.path.join(d, "progress_baseline.json")
    PRED_NEMO_BASELINE = os.path.join(d, "pred_nemo_baseline.npy")
    FP_CALIBVAL_PATH = os.path.join(d, "fingerprint_calibval.json")
    CALIBVAL_IDX_PATH = os.path.join(d, "calibval_line_indices.npy")
    PRED_NEMO_CALIBVAL = os.path.join(d, "pred_nemo_calibval.npy")
    FP_FLOOR21_NEMO_PATH = os.path.join(d, "fingerprint_floor21_mistralnemo.json")
    PROGRESS_TOPK_PATH = os.path.join(d, "progress_topk.json")
    TOPK_CHUNKS_DIR = os.path.join(d, "topk_partial_nemo")
    CHUNKS_FP_PATH = os.path.join(TOPK_CHUNKS_DIR, "chunks_fingerprint.json")
    GATE_TOPK_LINES_NEMO = os.path.join(d, "gate_topk_lines_nemo.npy")
    GATE_TOPK_IDS_NEMO = os.path.join(d, "gate_topk_ids_nemo.npy")
    GATE_TOPK_SCORES_NEMO = os.path.join(d, "gate_topk_scores_nemo.npy")
    GATE_TOPK_FP_NEMO = os.path.join(d, "gate_topk_fingerprint_nemo.json")
    PRED_NEMO_FLOOR21 = os.path.join(d, "pred_nemo_floor21.npy")
    PRED_NEMO_GATED = os.path.join(d, "pred_nemo_gated.npy")

    OUT_ROOT = r = resolve_out_root(ctx, out_dir,
                                    purpose="Mistral-Nemo variant reporting")
    FLAT_SET_CSV = out_path("flat_set_csv", r)
    FLAT_SET_MD = out_path("flat_set_md", r)
    TAU_FLOOR21_NEMO_CSV = out_path("tau_floor21_csv", r)
    TAU_FLAT_NEMO_CSV = out_path("tau_flat_csv", r)
    TAU_BUILD_MD = out_path("tau_build_md", r)
    OUT_MD = out_path("eval_md", r)
    OUT_TEX = out_path("eval_tex", r)
    OUT_CSV_FULLPOOL = out_path("per_lang_fullpool_csv", r)
    OUT_CSV_JUDGE = out_path("per_lang_judge_csv", r)
    DEGENERACY_MD = out_path("degeneracy_md", r)

    FLOOR_TARGET = _resolve_floor_target(ctx, floor_target, base_scratch)

    print(f"Mistral-Nemo chain against {ctx.describe()}\n  base y_true from {BASE_SCRATCH}"
          f"\n  reports {OUT_ROOT}\n  floor target {FLOOR_TARGET}", flush=True)
    return ctx

# Repo-side artifact names, stated relative to an output root so that --out-dir
# moves the whole set together (analysis/paper_eval.py's OUT_REL/out_path
# precedent). The module-level constants below keep the exact strings they had
# before --out-dir existed -- os.path.join("outputs", "tables/x.md") is
# "outputs/tables/x.md" -- because they are printed into the reports themselves.
OUT_REL = {
    # Written by "flatrule".
    "flat_set_csv": "diagnostic/mistralnemo_flat_set.csv",
    "flat_set_md": "tables/mistralnemo_flat_set.md",
    # Written by "tau", read back by "topk" and "eval".
    "tau_floor21_csv": "diagnostic/tau_mistralnemo_floor21_gate.csv",
    "tau_flat_csv": "diagnostic/tau_mistralnemo_flat.csv",
    "tau_build_md": "tables/mistralnemo_tau_build.md",
    # Written by "eval".
    "eval_md": "tables/mistralnemo_eval.md",
    "eval_tex": "tables/mistralnemo_eval.tex",
    "per_lang_fullpool_csv": "diagnostic/mistralnemo_per_lang_f1_fullpool.csv",
    "per_lang_judge_csv": "diagnostic/mistralnemo_per_lang_f1_judge.csv",
    # READ, not written, by "eval": analysis/degeneracy_scan_mistralnemo.py's
    # scan of THIS model's own weight rows. Model-derived, so it sits in the
    # output root and a run with its own --out-dir looks for its own copy.
    "degeneracy_md": "tables/degenerate_rows_mistralnemo.md",
}


def out_path(name: str, out_dir: str = None) -> str:
    """Path of one of this chain's artifacts under `out_dir` (default: outputs/)."""
    return os.path.join(out_dir or DEFAULT_OUT_ROOT, OUT_REL[name])


# analysis.mistralnemo_constants owns the degeneracy scan's path and names it
# with its own literal; if that literal ever moves, this module's
# out-root-relative copy would silently point somewhere else under --out-dir.
# Fail at import instead (analysis/paper_eval.py's gate-A check, same rationale).
if out_path("degeneracy_md") != DEGENERACY_OUT_MD:
    raise RuntimeError(
        f"degeneracy-scan path disagrees: analysis.mistralnemo_constants."
        f"DEGENERACY_OUT_MD is {DEGENERACY_OUT_MD!r} but this module's "
        f"out-root-relative form resolves to {out_path('degeneracy_md')!r}; "
        f"update OUT_REL['degeneracy_md'].")

OUT_ROOT = DEFAULT_OUT_ROOT
FLAT_SET_CSV = out_path("flat_set_csv")
FLAT_SET_MD = out_path("flat_set_md")
TAU_FLOOR21_NEMO_CSV = out_path("tau_floor21_csv")
TAU_FLAT_NEMO_CSV = out_path("tau_flat_csv")
TAU_BUILD_MD = out_path("tau_build_md")
OUT_MD = out_path("eval_md")
OUT_TEX = out_path("eval_tex")
OUT_CSV_FULLPOOL = out_path("per_lang_fullpool_csv")
OUT_CSV_JUDGE = out_path("per_lang_judge_csv")
DEGENERACY_MD = out_path("degeneracy_md")

# The paper's UniLID-Mistral-Nemo cell, GlotLID-C test (full 1,940-label
# pool), verified against paper/tables/lid_main.tex ("\unilid-Mistral-Nemo"
# row, "Full GlotLID-C label set" / "GlotLID-C test (1940 labels)" columns):
# "& .912 & \textbf{1.84e-5}". Raw (unscaled) values, matching the paper's
# own scientific-notation display, NOT the repo's internal x1e5-scaled table
# convention (FPR_SCALE, analysis.paper_eval) used elsewhere in this
# module's own tables. This is the paper team's own training run of the
# Mistral-Nemo variant, not the artifact this module evaluates (an
# independent retrain from the same recipe); the comparison below is a
# recorded measurement, not a pass/fail gate, per the pre-registration.
PAPER_MISTRALNEMO_F1_FULLPOOL = 0.912
PAPER_MISTRALNEMO_FPR_FULLPOOL = 1.84e-5

# Display names for LaTeX output (to_latex escapes only percent signs; raw
# names with underscores would break compilation), mirroring
# analysis/paper_eval.py's DISPLAY/_disp convention.
DISPLAY = {"nemo_baseline": "baseline", "nemo_floor21": "floor-21",
          "nemo_gated": "gated"}
CONFIGS = ("nemo_baseline", "nemo_floor21", "nemo_gated")


def _disp(c: str) -> str:
    return DISPLAY.get(c, c.replace("_", "-"))


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _sha256_file(path: str) -> str:
    with open(path, "rb") as f:
        return _sha256_bytes(f.read())


def _git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                             text=True, check=True)
    except (subprocess.CalledProcessError, OSError) as e:
        raise RuntimeError(f"git rev-parse HEAD failed: {e}") from e
    return out.stdout.strip()


def _floor_target_provenance() -> str:
    """Where the floor target in force came from, for the reports that state it.
    Renders identically to the pre---floor-target text for a default run."""
    return ("analysis.full_test_floor21" if FLOOR_TARGET == DEFAULT_FLOOR_TARGET
            else "--floor-target override")


def _check_fingerprint_floor_target(fp: dict, path: str) -> None:
    """The floor-21 matrix a later stage consumes must have been built at the
    clamp this run is configured for.

    _build_and_fingerprint_floor21_nemo enforces this by whole-dict equality
    whenever it rebuilds the matrix, but the topk stage's already-finalized early
    return and the eval stage both read the fingerprint WITHOUT rebuilding, so
    they check the one field here. For a run whose scratch state and
    --floor-target agree this is a no-op."""
    recorded = fp.get("floor_target")
    if recorded is None:
        raise RuntimeError(
            f"{path} records no floor_target field, so the floor-21 matrix it "
            f"describes cannot be paired with this run's clamp target "
            f"{FLOOR_TARGET}.")
    if float(recorded) != float(FLOOR_TARGET):
        raise RuntimeError(
            f"floor-target mismatch: {path} records floor_target "
            f"{float(recorded)}, this run is configured for {FLOOR_TARGET}. "
            "The floor-21 matrix, the tau values calibrated under it and the "
            "banked top-k candidates all belong to the recorded clamp; rerun "
            "the tau/topk stages with the matching --floor-target, or point "
            "--scratch-dir at the root built for this one.")


def _canonical_langs() -> list[str]:
    """The base model's own language order (analysis.transfer_sweep's default
    model path), the order y_true.npy's label indices and every downstream
    per-language array in this module are positionally aligned to."""
    weights, langs, _lang_to_idx = _load_model_data()
    del weights
    return langs


def _verify_variant_langs(canonical: list[str]) -> tuple[list[str], dict]:
    """Loads the variant's own language list and aborts if it differs, in
    order, from `canonical`. A label-string join across differing orders is
    not implemented (matching analysis/full_test_eval_131k.py's own choice):
    the reused y_true.npy indices would silently mean something else."""
    weights, langs, lang_to_idx = _load_model_data(PACKED_MODEL_PATH)
    del weights
    if langs != canonical:
        raise RuntimeError(
            f"variant language order ({PACKED_MODEL_PATH}) differs from the "
            "canonical order _load_model_data() returns for the base model; "
            "the reused y_true.npy label indices would be invalid for this "
            "model and a label-string join is not implemented")
    return langs, lang_to_idx


def _load_base_y_true() -> np.ndarray:
    """Reads y_true.npy from BASE_SCRATCH read-only. Valid for the variant
    because the test file and line space are identical across every model
    variant (the Apertus 131k/200k precedent) and the language order gate
    above guarantees the label indices mean the same thing."""
    path = os.path.join(BASE_SCRATCH, "y_true.npy")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} missing; the base model's full-pool baseline run "
            "(analysis/full_test_eval.py) must have completed before this "
            "module can reuse its y_true.npy")
    y = np.asarray(np.lib.format.open_memmap(path, mode="r"))
    if y.shape != (TOTAL_LINES,):
        raise RuntimeError(f"{path} has shape {y.shape}, expected "
                           f"({TOTAL_LINES},)")
    if int((y == UNSEEN).sum()) != 0:
        raise RuntimeError(f"{path} contains UNSEEN lines; the base model's "
                           "run is incomplete")
    return y


def _val_lines_sorted() -> np.ndarray:
    """Reproduces the seed-42 sample's validation half (position-parity
    split), cross-checked against the saved val_mask.npy, and asserts the
    count is exactly EXPECTED_VAL_LINES (250,000). Ascending order (the same
    order `_sample_line_indices()` and `load_sample()`'s y_true/pred arrays
    already carry, since `sample_idx` is built with `sorted(...)`)."""
    sample_idx = _sample_line_indices()
    parity_val = (np.arange(DEFAULT_SAMPLE_SIZE) % 2) == 0
    if not np.array_equal(np.load(VAL_MASK), parity_val):
        raise RuntimeError(f"{VAL_MASK} does not match the position-parity "
                           "split")
    val_lines = sample_idx[parity_val]
    if len(val_lines) != EXPECTED_VAL_LINES:
        raise RuntimeError(f"retired validation half has {len(val_lines):,} "
                           f"lines, expected exactly {EXPECTED_VAL_LINES:,}")
    return val_lines


def _read_degenerate_langs(path: str = None) -> list[str]:
    """Parses the `lang` column out of analysis/degeneracy_scan_mistralnemo.py's
    own markdown table (`| lang | estimated tokens | script |` rows) rather
    than hardcoding a duplicate list of the 32 flagged codes: the file is the
    single source of truth.

    The default is bound at CALL time (DEGENERACY_MD, which configure() re-derives
    under --out-dir), not at def time: a scan of the model's own weight rows is
    model-derived, so a non-default model must read its own copy or abort."""
    path = DEGENERACY_MD if path is None else path
    if not os.path.exists(path):
        raise FileNotFoundError(f"degeneracy scan report missing: {path}")
    langs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line.startswith("|"):
                continue
            cells = [c.strip() for c in line.strip("|").split("|")]
            if len(cells) != 3 or cells[0] in ("lang", "---"):
                continue
            if set(cells[1]) <= set("-"):
                continue
            langs.append(cells[0])
    return langs


# ---------------------------------------------------------------------------
# STAGE "baseline"
# ---------------------------------------------------------------------------

def run_baseline() -> str:
    os.makedirs(SCRATCH_DIR_NEMO, exist_ok=True)
    if not os.path.exists(PACKED_MODEL_PATH):
        raise FileNotFoundError(f"packed model missing: {PACKED_MODEL_PATH}")

    canonical = _canonical_langs()
    langs, lang_to_idx = _verify_variant_langs(canonical)
    n_lang = len(langs)
    print(f"language order verified: {n_lang} languages match the canonical "
         "order.", flush=True)

    weights, _langs_m, _m = _load_model_data(PACKED_MODEL_PATH)
    W = np.array(weights, dtype=np.float32)
    del weights
    sha_weight_matrix = _sha256_bytes(W.tobytes())
    del W
    sha_model_file = _sha256_file(PACKED_MODEL_PATH)

    # Measured, not copied: tokenizer_json_sha256 is computed by hashing the
    # tokenizer.json file at its pinned snapshot path (TOKENIZER_JSON,
    # analysis.mistralnemo_constants) so this field records an actual
    # measurement of the file the model was trained against, not a restated
    # copy of the EXPECTED_TOKENIZER_SHA256 constant. Still gated against
    # that constant (the pin recorded at preflight time, 2026-08-07): a
    # mismatch means the snapshot on disk has drifted since preflight.
    if not os.path.exists(TOKENIZER_JSON):
        raise FileNotFoundError(
            f"tokenizer.json missing at the pinned snapshot path: "
            f"{TOKENIZER_JSON}")
    sha_tokenizer_json = _sha256_file(TOKENIZER_JSON)
    if sha_tokenizer_json != EXPECTED_TOKENIZER_SHA256:
        raise RuntimeError(
            f"{TOKENIZER_JSON} sha256 ({sha_tokenizer_json[:16]}...) does "
            f"not match EXPECTED_TOKENIZER_SHA256 "
            f"({EXPECTED_TOKENIZER_SHA256[:16]}..., "
            "analysis.mistralnemo_constants); the tokenizer snapshot has "
            "drifted from the pin recorded at preflight time")

    fp = {
        "model_file_sha256": sha_model_file,
        "weight_matrix_sha256": sha_weight_matrix,
        "tokenizer_snapshot_hash": SNAPSHOT_HASH,
        "tokenizer_json_sha256": sha_tokenizer_json,
        "langs_sha256": _sha256_bytes("|".join(langs).encode()),
        "chunk_lines": CHUNK_LINES,
        "total_lines": TOTAL_LINES,
    }
    if os.path.exists(FP_BASELINE_PATH):
        with open(FP_BASELINE_PATH) as f:
            prev = json.load(f)
        if prev != fp:
            bad = sorted(k for k in fp if prev.get(k) != fp[k])
            raise RuntimeError(
                f"scratch state in {SCRATCH_DIR_NEMO} was produced under a "
                f"different configuration (mismatched: {bad}); clear the "
                "baseline files there or restore the original inputs")
    else:
        with open(FP_BASELINE_PATH + ".tmp", "w") as f:
            json.dump(fp, f)
        os.replace(FP_BASELINE_PATH + ".tmp", FP_BASELINE_PATH)

    y_mm = np.lib.format.open_memmap(os.path.join(BASE_SCRATCH, "y_true.npy"),
                                     mode="r")
    if y_mm.shape != (TOTAL_LINES,):
        raise RuntimeError(f"y_true.npy shape {y_mm.shape} != ({TOTAL_LINES},)")

    # val_lines: the shared _val_lines_sorted() helper (single source for the
    # position-parity split's validation half, including its own VAL_MASK
    # cross-check), rather than reimplementing that derivation inline.
    val_lines = set(_val_lines_sorted().tolist())
    sample_idx = _sample_line_indices()
    parity_val = (np.arange(DEFAULT_SAMPLE_SIZE) % 2) == 0
    sample_test_lines = sample_idx[~parity_val]
    pickle_y = np.array(load_sample(DEFAULT_SAMPLE_SIZE)["y_true"])[~parity_val]
    expect_label = dict(zip(sample_test_lines.tolist(), pickle_y.tolist()))

    if os.path.exists(PRED_NEMO_BASELINE):
        pred_mm = np.lib.format.open_memmap(PRED_NEMO_BASELINE, mode="r+")
        if pred_mm.shape != (TOTAL_LINES,):
            raise RuntimeError(f"{PRED_NEMO_BASELINE} shape {pred_mm.shape} != "
                               f"({TOTAL_LINES},)")
    else:
        pred_mm = np.lib.format.open_memmap(PRED_NEMO_BASELINE, mode="w+",
                                            dtype=np.int16, shape=(TOTAL_LINES,))
        pred_mm[:] = UNSEEN
        pred_mm.flush()

    done_chunks = set()
    if os.path.exists(PROGRESS_BASELINE_PATH):
        with open(PROGRESS_BASELINE_PATH) as f:
            done_chunks = set(json.load(f))

    n_chunks = (TOTAL_LINES + CHUNK_LINES - 1) // CHUNK_LINES
    model = None
    with open(TEST_FILE) as fh:
        for chunk in range(n_chunks):
            lo = chunk * CHUNK_LINES
            hi = min(lo + CHUNK_LINES, TOTAL_LINES)
            if chunk in done_chunks:
                for _ in range(hi - lo):
                    fh.readline()
                continue
            if model is None:
                print("Loading the Mistral-Nemo variant model (unmodified "
                     "matrix)...", flush=True)
                model = _load_unilid_model(PACKED_MODEL_PATH)
                if model.langs != langs:
                    raise RuntimeError(
                        "_load_unilid_model's language list differs from "
                        "_load_model_data's for the variant; the two loaders "
                        "read the model file inconsistently")
            lines = [fh.readline() for _ in range(hi - lo)]

            keep_pos, texts = [], []
            for j, line in enumerate(lines):
                i = lo + j
                if i in val_lines:
                    if y_mm[i] != EXCLUDED:
                        raise RuntimeError(f"line {i}: val line not EXCLUDED "
                                           "in the reused y_true memmap")
                    continue
                label, text = _parse_line(line)
                exp = expect_label.get(i)
                if exp is not None and label != exp:
                    raise RuntimeError(f"alignment mismatch at line {i}: "
                                       f"parsed {label!r}, sample pickle has "
                                       f"{exp!r}")
                li = lang_to_idx.get(label)
                if li is None or y_mm[i] != li:
                    raise RuntimeError(
                        f"line {i}: label {label!r} (idx {li}) does not "
                        f"match the reused y_true value {int(y_mm[i])}")
                keep_pos.append(i)
                texts.append(text)

            pre, valid = [], []
            for k, t in enumerate(texts):
                p = model.preprocess(t)
                if p:
                    pre.append(p)
                    valid.append(k)
            out = np.full(len(texts), EMPTY, dtype=np.int16)
            if pre:
                batch = model.model.best_of_cached_weight_sets_batch(pre)
                if len(batch) != len(pre):
                    raise RuntimeError(f"chunk {chunk}: scorer returned "
                                       f"{len(batch)} results for {len(pre)} "
                                       "inputs")
                for k, (idx, _t, _s) in zip(valid, batch):
                    out[k] = idx
            pred_mm[np.asarray(keep_pos, dtype=np.int64)] = out
            pred_mm.flush()
            done_chunks.add(chunk)
            with open(PROGRESS_BASELINE_PATH + ".tmp", "w") as f:
                json.dump(sorted(done_chunks), f)
            os.replace(PROGRESS_BASELINE_PATH + ".tmp", PROGRESS_BASELINE_PATH)
            print(f"baseline chunk {chunk + 1}/{n_chunks} done "
                 f"({hi - lo} lines)", flush=True)

    pred = np.asarray(pred_mm)
    y = np.asarray(y_mm)
    kept = y >= 0
    if int((pred[kept] == UNSEEN).sum()) != 0:
        raise RuntimeError("unprocessed kept lines remain in pred_nemo_baseline")
    print(f"STAGE baseline done: {int(kept.sum()):,} kept lines scored; "
         f"fingerprint at {FP_BASELINE_PATH}, predictions at "
         f"{PRED_NEMO_BASELINE}.")
    return PRED_NEMO_BASELINE


# ---------------------------------------------------------------------------
# STAGE "calibval"
# ---------------------------------------------------------------------------

def run_calibval() -> str:
    os.makedirs(SCRATCH_DIR_NEMO, exist_ok=True)
    if not os.path.exists(PACKED_MODEL_PATH):
        raise FileNotFoundError(f"packed model missing: {PACKED_MODEL_PATH}")

    canonical = _canonical_langs()
    langs, lang_to_idx = _verify_variant_langs(canonical)

    val_lines = _val_lines_sorted()   # ascending, length EXPECTED_VAL_LINES
    y_base = _load_base_y_true()
    n_not_excluded = int((y_base[val_lines] != EXCLUDED).sum())
    if n_not_excluded:
        raise RuntimeError(
            f"{n_not_excluded:,} of {len(val_lines):,} validation-half line "
            "indices are not EXCLUDED in the base model's y_true.npy; the "
            "retired validation half and the reused y_true.npy EXCLUDED "
            "sentinel have diverged")
    print(f"retired validation half: {len(val_lines):,} lines (verified "
         f"== EXPECTED_VAL_LINES {EXPECTED_VAL_LINES:,}, all EXCLUDED in "
         "y_true.npy).", flush=True)

    weights, _langs_m, _m = _load_model_data(PACKED_MODEL_PATH)
    W = np.array(weights, dtype=np.float32)
    del weights
    sha_weight_matrix = _sha256_bytes(W.tobytes())
    del W

    fp = {
        "weight_matrix_sha256": sha_weight_matrix,
        "langs_sha256": _sha256_bytes("|".join(langs).encode()),
        "n_val_lines": int(len(val_lines)),
    }
    if os.path.exists(FP_CALIBVAL_PATH):
        with open(FP_CALIBVAL_PATH) as f:
            prev = json.load(f)
        if prev == fp and os.path.exists(PRED_NEMO_CALIBVAL) and \
                os.path.exists(CALIBVAL_IDX_PATH):
            print(f"existing {PRED_NEMO_CALIBVAL} matches the current "
                 "fingerprint; skipping rescoring.")
            return PRED_NEMO_CALIBVAL
        if prev != fp:
            bad = sorted(k for k in fp if prev.get(k) != fp[k])
            raise RuntimeError(
                f"calibval scratch state mismatch ({bad}); clear the "
                f"calibval files in {SCRATCH_DIR_NEMO} or investigate what "
                "changed")

    want = set(int(i) for i in val_lines.tolist())
    raw_lines, n_read = _read_wanted_lines(TEST_FILE, want)
    if n_read != TOTAL_LINES:
        raise RuntimeError(f"read {n_read} lines from {TEST_FILE}, expected "
                           f"{TOTAL_LINES}")
    if len(raw_lines) != len(val_lines):
        raise RuntimeError(f"collected {len(raw_lines)} lines of text for "
                           f"{len(val_lines)} validation line indices")

    texts = []
    for i in val_lines.tolist():
        _label, text = _parse_line(raw_lines.pop(int(i)).decode("utf-8"))
        texts.append(text)
    del raw_lines

    print("Loading the Mistral-Nemo variant model (unmodified matrix) for "
         "calibval scoring...", flush=True)
    model = _load_unilid_model(PACKED_MODEL_PATH)
    if model.langs != langs:
        raise RuntimeError("_load_unilid_model's language list differs from "
                           "_load_model_data's for the variant")

    pre, valid = [], []
    for k, t in enumerate(texts):
        p = model.preprocess(t)
        if p:
            pre.append(p)
            valid.append(k)
    out = np.full(len(texts), EMPTY, dtype=np.int16)
    for lo in range(0, len(pre), SCORE_BATCH_MAX):
        hi = min(lo + SCORE_BATCH_MAX, len(pre))
        batch = model.model.best_of_cached_weight_sets_batch(pre[lo:hi])
        if len(batch) != hi - lo:
            raise RuntimeError(f"calibval scorer returned {len(batch)} "
                               f"results for {hi - lo} inputs")
        for k, (idx, _t, _s) in zip(valid[lo:hi], batch):
            out[k] = idx
        print(f"calibval scored [{lo:,}:{hi:,}) of {len(pre):,}", flush=True)

    np.save(PRED_NEMO_CALIBVAL, out)
    np.save(CALIBVAL_IDX_PATH, val_lines.astype(np.int64))
    with open(FP_CALIBVAL_PATH + ".tmp", "w") as f:
        json.dump(fp, f)
    os.replace(FP_CALIBVAL_PATH + ".tmp", FP_CALIBVAL_PATH)

    n_empty = int((out == EMPTY).sum())
    print(f"STAGE calibval done: {len(val_lines):,} lines scored, "
         f"{n_empty:,} empty after preprocess; wrote {PRED_NEMO_CALIBVAL} "
         f"and {CALIBVAL_IDX_PATH}.")
    return PRED_NEMO_CALIBVAL


# ---------------------------------------------------------------------------
# STAGE "flatrule" (login node)
# ---------------------------------------------------------------------------

def run_flatrule() -> str:
    os.makedirs(os.path.dirname(FLAT_SET_CSV), exist_ok=True)
    os.makedirs(os.path.dirname(FLAT_SET_MD), exist_ok=True)
    for p in (PRED_NEMO_CALIBVAL, CALIBVAL_IDX_PATH, FP_BASELINE_PATH,
             FP_CALIBVAL_PATH):
        if not os.path.exists(p):
            raise FileNotFoundError(f"{p} missing; run --stage baseline / "
                                    "calibval first")

    canonical = _canonical_langs()
    langs, _lang_to_idx = _verify_variant_langs(canonical)
    n_lang = len(langs)
    train_counts = _load_train_counts()
    N = np.array([train_counts.get(l, 0) for l in langs], dtype=np.float64)
    scripts = np.array([l.rsplit("_", 1)[-1] if "_" in l else "Unknown"
                        for l in langs])

    weights, _langs_m, _m = _load_model_data(PACKED_MODEL_PATH)
    # Weight-matrix sha binding: the matrix loaded here for zH must be the
    # identical unmodified matrix the baseline and calibval scoring passes
    # used, not merely a re-read of the same PACKED_MODEL_PATH file (which
    # could have been repacked between runs). Both fingerprints are checked
    # against this stage's own freshly-computed sha, not against each other,
    # so either one drifting independently is caught with this stage named
    # as the point of detection.
    sha_w = _sha256_bytes(np.array(weights, dtype=np.float32).tobytes())
    with open(FP_BASELINE_PATH) as f:
        fp_baseline = json.load(f)
    with open(FP_CALIBVAL_PATH) as f:
        fp_calibval = json.load(f)
    mismatched = [name for name, fp in (("baseline", fp_baseline),
                                        ("calibval", fp_calibval))
                 if fp["weight_matrix_sha256"] != sha_w]
    if mismatched:
        raise RuntimeError(
            f"weight-matrix sha mismatch: the matrix loaded here for zH "
            f"(sha256 {sha_w[:16]}...) does not match the recorded "
            f"weight_matrix_sha256 in the {mismatched} fingerprint(s) "
            f"({FP_BASELINE_PATH} / {FP_CALIBVAL_PATH}); the baseline and "
            "calibval scoring passes and this stage's own zH computation "
            "must all use the identical unmodified matrix")

    print("Computing probabilities / entropy for the variant's weight "
         "matrix (reusing analysis.diagnostic._probs_and_logprobs)...",
         flush=True)
    P, logP = _probs_and_logprobs(weights)
    del weights
    # H = -(P * logP).sum(axis=1): analysis.diagnostic._sym_kl_matrix
    # computes this identical quantity internally (as -neg_entropy, where
    # neg_entropy = (P * logP).sum(axis=1)) before deriving its symKL matrix
    # from it; the reviewer verified the two are bit-identical. Computed
    # directly here instead of calling _sym_kl_matrix, which would also
    # build an n_lang x n_lang symKL matrix via a P @ logP.T matmul the
    # flat-magnet rule never uses (is_magnet has top priority in
    # analysis/diagnostic.py's classification and does not depend on the
    # twin/tight_lowres/isolated_tail branches, the only consumers of symKL
    # there); avoiding that matmul is the point of computing H directly. P
    # and logP are freed immediately after.
    H = -(P * logP).sum(axis=1)
    del P, logP

    zH = np.zeros(n_lang)
    for s in np.unique(scripts):
        m = scripts == s
        if m.sum() < 3:
            continue
        med = np.median(H[m])
        mad = np.median(np.abs(H[m] - med)) * 1.4826 + 1e-9
        zH[m] = (H[m] - med) / mad

    calib_idx = np.load(CALIBVAL_IDX_PATH)
    calib_pred = np.load(PRED_NEMO_CALIBVAL)
    if calib_pred.shape != calib_idx.shape:
        raise RuntimeError(f"{PRED_NEMO_CALIBVAL} shape {calib_pred.shape} != "
                           f"{CALIBVAL_IDX_PATH} shape {calib_idx.shape}")
    if len(calib_idx) != EXPECTED_VAL_LINES:
        raise RuntimeError(f"{CALIBVAL_IDX_PATH} has {len(calib_idx):,} "
                           f"lines, expected {EXPECTED_VAL_LINES:,}")

    # True labels for the calibval lines: re-derived from the seed-42
    # sample pickle's y_true, restricted to the validation-half positions,
    # the same source analysis/diagnostic.py itself reads for the base
    # model (never re-parsing TEST_FILE for labels the pickle already
    # carries). calib_idx is cross-checked directly against the shared
    # _val_lines_sorted() helper (single source for the position-parity
    # split's validation half, including its own VAL_MASK cross-check)
    # rather than reimplementing that derivation inline; parity_val is still
    # needed locally to positionally slice the sample pickle's y_true.
    parity_val = (np.arange(DEFAULT_SAMPLE_SIZE) % 2) == 0
    if not np.array_equal(_val_lines_sorted(), calib_idx):
        raise RuntimeError(
            f"{CALIBVAL_IDX_PATH} does not match _val_lines_sorted(); "
            "the calibval line order has drifted from the sample draw")
    true_labels = np.array(load_sample(DEFAULT_SAMPLE_SIZE)["y_true"])[parity_val]

    # EMPTY (-1) predictions are mapped to a sentinel string that cannot
    # collide with any lang_Script code, so an empty-after-preprocess line
    # contributes no false positive to any language (matching the repo's
    # "empty scores as wrong" convention: it is a false negative for its
    # true label via _empirical_magnet's own true_c/tp_c accounting, and
    # never a counted prediction for anyone).
    pred_labels = np.array(
        [langs[i] if i >= 0 else "<EMPTY>" for i in calib_pred.tolist()],
        dtype=object)

    S, FP, ratio = _empirical_magnet(langs, true_labels, pred_labels)
    support_val = np.array([S[l] for l in langs])
    fp_val = np.array([FP[l] for l in langs])
    magnet_ratio = np.array([ratio[l] for l in langs])

    is_magnet = ((zH > ZH_MAGNET) & (magnet_ratio > MAGNET_RATIO_MIN)) | \
        (zH > ZH_EXTREME)
    flat_mask = is_magnet & (N >= HEAD_N)
    flat_idx = np.where(flat_mask)[0]

    rows = [{"lang": langs[i], "N": int(N[i]), "zH": round(float(zH[i]), 4),
            "support_val": int(support_val[i]), "fp_val": int(fp_val[i]),
            "magnet_ratio": round(float(magnet_ratio[i]), 4)}
           for i in flat_idx]
    pd.DataFrame(rows, columns=["lang", "N", "zH", "support_val", "fp_val",
                                "magnet_ratio"]).to_csv(FLAT_SET_CSV,
                                                        index=False)

    n_magnet_total = int(is_magnet.sum())
    L = [
        "# Mistral-Nemo variant: flat-language rule (E3 pre-registration)\n",
        f"zH (within-script entropy z-score) and magnet_ratio recomputed for "
        f"the variant's own weight matrix, reusing "
        f"analysis.diagnostic._probs_and_logprobs / ._empirical_magnet by "
        f"import (entropy H computed directly as -(P * logP).sum(axis=1), "
        f"verified bit-identical to _sym_kl_matrix's own entropy return, "
        f"without that function's unused symKL matmul). Rule (imported "
        f"constants, unchanged): "
        f"is_magnet = (zH > ZH_MAGNET={ZH_MAGNET} and magnet_ratio > "
        f"MAGNET_RATIO_MIN={MAGNET_RATIO_MIN}) or zH > ZH_EXTREME={ZH_EXTREME}. "
        f"magnet_ratio, support_val, and fp_val come from "
        f"{PRED_NEMO_CALIBVAL} (the retired 250,000-line validation half),"
        f" never the test pool.\n",
        f"- Languages flagged is_magnet by this rule: {n_magnet_total:,} of "
        f"{n_lang:,}.",
        f"- Flat set (is_magnet AND N >= HEAD_N={HEAD_N:,}), the languages "
        f"the gate's step 2 re-examines: {len(flat_idx)}.",
        "",
    ]
    if len(flat_idx):
        L.append("| lang | N | zH | support_val | fp_val | magnet_ratio |")
        L.append("|---|---|---|---|---|---|")
        for r in rows:
            L.append(f"| {r['lang']} | {r['N']} | {r['zH']} | "
                     f"{r['support_val']} | {r['fp_val']} | "
                     f"{r['magnet_ratio']} |")
    else:
        L.append("No language met the flat-set condition (is_magnet AND "
                 "N >= HEAD_N) for this weight matrix.")
    L.append("")
    with open(FLAT_SET_MD, "w") as f:
        f.write("\n".join(L))
    print("\n".join(L))
    print(f"\nWrote {FLAT_SET_CSV} and {FLAT_SET_MD}")
    return FLAT_SET_CSV


# ---------------------------------------------------------------------------
# Shared: floor-21 matrix build + fingerprint (write-if-absent,
# verify-if-present), used by both "tau" (first writer) and "topk" (reader).
# ---------------------------------------------------------------------------

def _build_and_fingerprint_floor21_nemo(langs: list[str]) -> tuple[np.ndarray, str, str]:
    if not os.path.exists(FP_BASELINE_PATH):
        raise FileNotFoundError(f"{FP_BASELINE_PATH} missing; run --stage "
                                "baseline first")
    with open(FP_BASELINE_PATH) as f:
        fp_baseline = json.load(f)

    weights, langs_m, _m = _load_model_data(PACKED_MODEL_PATH)
    if langs_m != langs:
        raise RuntimeError("variant model language order does not match the "
                           "caller's language list")
    W = np.array(weights, dtype=np.float32)
    del weights

    # special_idx is REQUIRED here: 0.3.0-packed (corrected) containers park
    # the four specials at the training floor BELOW every real token, so
    # without it build_equalized_weights takes the row minimum over the
    # specials, reports zero modified rows, and returns the matrix unchanged
    # -- an unclamped baseline masquerading as the floor-21 matrix
    # (review finding, 2026-08-23). On the released container (specials at
    # log 0.2, the row MAXIMUM) passing special_idx is a measured no-op.
    special_cols = _special_columns(PACKED_MODEL_PATH)
    if sorted(special_cols) != sorted(SPECIAL_COLS_NEMO):
        raise RuntimeError(
            f"{PACKED_MODEL_PATH}: special columns {sorted(special_cols)} "
            f"read from the vocabulary differ from the recorded "
            f"SPECIAL_COLS_NEMO {sorted(SPECIAL_COLS_NEMO)}")
    real_cols = np.setdiff1d(np.arange(W.shape[1]), np.array(special_cols))
    matrix, n_mod = build_equalized_weights(W, FLOOR_TARGET, special_cols)
    verify_one_sided_clamp(W, FLOOR_TARGET, special_cols, n_mod,
                           label="mistralnemo ")
    # The recorded mechanism (Exp 20) is a DOWNWARD clamp, min(floor_L, F),
    # nothing raised, so a row whose natural floor already sits at or below
    # FLOOR_TARGET is legitimately left unchanged. The base model had no such
    # row (all floors above -21, median -17.66) and its precedent gate
    # asserted n_mod == n_lang; this variant has rows with deeper natural
    # floors (measured 2026-08-08: khm_Khmr -21.232, ory_Orya -21.016, both
    # healthy rows), so the precise invariant is asserted instead: every
    # UNmodified row's pre-existing floor must be <= FLOOR_TARGET. Any other
    # skip reason is a wiring error and aborts.
    if n_mod != len(langs):
        row_mins = W[:, real_cols].min(axis=1)
        unmodified = np.where((matrix == W).all(axis=1))[0]
        illegitimate = [i for i in unmodified if row_mins[i] > FLOOR_TARGET]
        if illegitimate:
            names = ", ".join(f"{langs[i]} (floor {row_mins[i]:.4f})"
                              for i in illegitimate[:10])
            raise RuntimeError(
                f"floor {FLOOR_TARGET} left {len(illegitimate)} row(s) "
                f"unmodified whose natural floor is ABOVE the target, which "
                f"the downward clamp can never do: {names}")
        # json round-trips tuples as lists, so build lists directly or the
        # fingerprint equality check fails on type rather than content.
        skipped = [[langs[i], float(row_mins[i])] for i in unmodified]
        print(f"floor {FLOOR_TARGET}: {n_mod} of {len(langs)} rows clamped; "
              f"{len(skipped)} row(s) already at or below the target and "
              f"left unchanged per the downward-clamp mechanism: "
              + ", ".join(f"{l} ({m:.4f})" for l, m in skipped), flush=True)
    else:
        skipped = []
    # Special-token columns for THIS vocabulary are 0 (<unk>), 1 (<s>),
    # 2 (</s>), 10 (<pad>) (the preflight's recorded id mapping), NOT the
    # contiguous 0:4 of the base model's packing. Column 3 here is an
    # ordinary token that can legitimately sit at a row minimum and be
    # clamped (measured 2026-08-09: exactly one row). The gate asserts the
    # actual special columns are untouched AND hold the packing convention's
    # p=0.2 value.
    for c in SPECIAL_COLS_NEMO:
        if not np.array_equal(matrix[:, c], W[:, c]):
            raise RuntimeError(f"special-token column {c} was modified by "
                               "the floor clamp")
        col = W[:, c]
        real_min = W[:, real_cols].min(axis=1)
        # Generation-aware packing check: defective (pre-0.3.0) containers
        # store each special at p = 0.2 (the row maximum); corrected (0.3.0)
        # containers park them at the training floor, at or below every real
        # token. Either is a valid packing; anything else is a wiring error.
        defective_packing = np.allclose(col, np.log(0.2), atol=1e-4)
        corrected_packing = bool((col <= real_min).all())
        if not (defective_packing or corrected_packing):
            raise RuntimeError(
                f"special-token column {c} holds neither the defective "
                f"packing's log(0.2) value nor the 0.3.0 packing's "
                f"at-or-below-every-real-token value")
    sha_w = _sha256_bytes(W.tobytes())
    sha_w21 = _sha256_bytes(matrix.tobytes())
    del W

    # Weight-matrix sha binding: the base matrix built into the floor-21
    # matrix here must be the identical unmodified matrix the baseline
    # scoring pass used, not merely another read of the same
    # PACKED_MODEL_PATH file (which could have been repacked in between).
    if fp_baseline["weight_matrix_sha256"] != sha_w:
        raise RuntimeError(
            f"weight-matrix sha mismatch: the base matrix built here "
            f"(sha256 {sha_w[:16]}...) does not match {FP_BASELINE_PATH}'s "
            f"recorded weight_matrix_sha256 "
            f"({fp_baseline['weight_matrix_sha256'][:16]}...); the baseline "
            "scoring pass and this floor-21 build used different "
            "underlying weight matrices")

    fp = {"sha256_base_W": sha_w, "sha256_w21": sha_w21,
         "floor_target": FLOOR_TARGET,
         "langs_sha256": _sha256_bytes("|".join(langs).encode()),
         "rows_below_floor_target": skipped}
    if os.path.exists(FP_FLOOR21_NEMO_PATH):
        with open(FP_FLOOR21_NEMO_PATH) as f:
            prev = json.load(f)
        if prev != fp:
            bad = sorted(k for k in fp if prev.get(k) != fp[k])
            raise RuntimeError(f"{FP_FLOOR21_NEMO_PATH} mismatch ({bad}); "
                               "the rebuilt floor-21 matrix does not match "
                               "the recorded fingerprint")
    else:
        os.makedirs(SCRATCH_DIR_NEMO, exist_ok=True)
        with open(FP_FLOOR21_NEMO_PATH + ".tmp", "w") as f:
            json.dump(fp, f)
        os.replace(FP_FLOOR21_NEMO_PATH + ".tmp", FP_FLOOR21_NEMO_PATH)
    return matrix, sha_w, sha_w21


# ---------------------------------------------------------------------------
# STAGE "tau"
# ---------------------------------------------------------------------------

def _calibrate_group(model, langs: list[str], N: np.ndarray, idx: np.ndarray,
                     q_of, out_csv: str, abort_on_shortfall: bool) -> None:
    """Shared per-language tau calibration loop (own-train margins under the
    already-cached matrix), used for both group A (size-adaptive quantile,
    exclude-and-log on shortfall, mirroring analysis/solo_gates.py's
    run("floor21")) and group B (fixed MARGIN_Q-th percentile, abort on
    shortfall, mirroring analysis/gate_variants.py's _calibrate_flat4_tau5).
    `q_of(lang_idx) -> float` returns the percentile to use for that
    language; `abort_on_shortfall` selects which of the two conventions
    applies."""
    rng = np.random.default_rng(CALIB_SEED)
    calib_rows = []
    for li in idx.tolist():
        lang = langs[li]
        path = os.path.join(CORPUS_DIR, f"{lang}_train.txt")
        if not os.path.exists(path):
            raise FileNotFoundError(f"corpus file missing: {path}")
        with open(path) as f:
            lines = [l.rstrip("\n") for l in f if l.rstrip("\n")]
        if len(lines) > CALIB_MAX:
            lines = [lines[k] for k in
                     sorted(rng.choice(len(lines), CALIB_MAX, replace=False))]
        cdrop = []
        _cpos, ctopk = _topk_batch(model, lines, cdrop)
        wins = [c for c in ctopk if c and c[0][0] == li]
        gaps = np.array([_gap(c) for c in wins])
        gaps = gaps[np.isfinite(gaps)]
        low_calib = len(gaps) < MIN_CALIB_LINES
        if low_calib and abort_on_shortfall:
            raise RuntimeError(
                f"{lang} has {len(gaps)} finite winning calibration margins, "
                f"below MIN_CALIB_LINES={MIN_CALIB_LINES}; this language is "
                f"in the flat set with N={int(N[li]):,} >= HEAD_N={HEAD_N:,}, "
                "so a shortfall indicates a wiring error, not a genuine "
                "low-resource case, and aborts rather than excluding it")
        q_l = q_of(li)
        zero_strength = q_l <= 0.0
        excluded = low_calib or zero_strength
        cause = ("low_calibration" if low_calib
                else "zero_strength" if zero_strength else "")
        tau = float("-inf") if excluded else float(np.percentile(gaps, q_l))
        calib_rows.append({"lang": lang, "n_scoreable": len(ctopk),
                          "n_self_won": len(wins), "tau": tau,
                          "excluded": excluded, "cause": cause})
        print(f"tau calibration: {lang} tau={tau:.4f} nats "
             f"({len(gaps):,} finite winning margins of {len(wins):,} "
             f"self-won, {len(ctopk):,} scoreable of {len(lines):,} sampled)",
             flush=True)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    pd.DataFrame(calib_rows, columns=["lang", "n_scoreable", "n_self_won",
                                      "tau", "excluded", "cause"]
                ).to_csv(out_csv, index=False)


def run_tau() -> str:
    # Both directories, not just tables/: _calibrate_group writes the two tau
    # CSVs under diagnostic/ before TAU_BUILD_MD's own makedirs is reached, so a
    # fresh --out-dir would otherwise fail partway through the calibration.
    os.makedirs(os.path.join(OUT_ROOT, "diagnostic"), exist_ok=True)
    os.makedirs(os.path.join(OUT_ROOT, "tables"), exist_ok=True)
    if not os.path.exists(FLAT_SET_CSV):
        raise FileNotFoundError(f"{FLAT_SET_CSV} missing; run --stage "
                                "flatrule first")

    canonical = _canonical_langs()
    langs, lang_to_idx = _verify_variant_langs(canonical)
    train_counts = _load_train_counts()
    N = np.array([train_counts.get(l, 0) for l in langs], dtype=np.float64)

    matrix, sha_w, sha_w21 = _build_and_fingerprint_floor21_nemo(langs)
    print(f"floor-21 matrix built and fingerprinted at "
         f"{FP_FLOOR21_NEMO_PATH} (sha256_base_W {sha_w[:16]}..., "
         f"sha256_w21 {sha_w21[:16]}...).", flush=True)

    tail_idx = np.where(N < HEAD_N)[0]
    flat_df = pd.read_csv(FLAT_SET_CSV)
    flat_langs = set(flat_df.lang) if len(flat_df) else set()
    for lang in flat_langs:
        n_l = int(N[lang_to_idx[lang]])
        if n_l < HEAD_N:
            raise RuntimeError(f"flat-set language {lang} has N={n_l:,} < "
                               f"HEAD_N ({HEAD_N:,}); the flat set must be "
                               "disjoint from the N < HEAD_N group by "
                               "construction (analysis/mistralnemo_eval.py's "
                               "run_flatrule)")
    flat_idx = np.array(sorted(lang_to_idx[l] for l in flat_langs),
                        dtype=np.int64)

    model = _load_unilid_model(PACKED_MODEL_PATH)
    if model.langs != langs:
        raise RuntimeError("_load_unilid_model's language list differs from "
                           "_load_model_data's for the variant")
    print("Caching the variant's floor-21 weights for tau calibration...",
         flush=True)
    model.model.set_weight_sets(matrix.tolist())
    del matrix

    q_group_a = lambda li: MARGIN_Q * (1.0 - min(float(N[li]), float(HEAD_N)) / HEAD_N)
    _calibrate_group(model, langs, N, tail_idx, q_group_a,
                     TAU_FLOOR21_NEMO_CSV, abort_on_shortfall=False)
    print(f"Wrote {TAU_FLOOR21_NEMO_CSV} ({len(tail_idx)} group-A languages, "
         "N < HEAD_N).")

    q_group_b = lambda li: float(MARGIN_Q)
    if len(flat_idx):
        _calibrate_group(model, langs, N, flat_idx, q_group_b,
                         TAU_FLAT_NEMO_CSV, abort_on_shortfall=True)
    else:
        pd.DataFrame([], columns=["lang", "n_scoreable", "n_self_won", "tau",
                                  "excluded", "cause"]).to_csv(
            TAU_FLAT_NEMO_CSV, index=False)
    print(f"Wrote {TAU_FLAT_NEMO_CSV} ({len(flat_idx)} group-B languages, "
         "the variant's flat set).")

    L = ["# Mistral-Nemo variant: tau recalibration under the floor-21 "
        "matrix (E3 pre-registration)\n",
        f"floor-21 matrix: FLOOR_TARGET={FLOOR_TARGET} "
        f"({_floor_target_provenance()}), fingerprint at "
        f"{FP_FLOOR21_NEMO_PATH}.\n",
        f"Group A (N < HEAD_N={HEAD_N:,}): {len(tail_idx)} languages, "
        f"size-adaptive quantile q_L = MARGIN_Q * (1 - min(N,HEAD_N)/HEAD_N), "
        f"MARGIN_Q={MARGIN_Q} (analysis.margin_diagnostic); exclude-and-log "
        f"on calibration shortfall (mirrors analysis/solo_gates.py's "
        "run(\"floor21\")). Output: " + TAU_FLOOR21_NEMO_CSV + ".",
        f"\nGroup B (the variant's flat set, N >= HEAD_N): {len(flat_idx)} "
        f"languages, fixed {MARGIN_Q}th percentile; abort (not exclude) on "
        "calibration shortfall (mirrors "
        "analysis.gate_variants._calibrate_flat4_tau5). Output: " +
        TAU_FLAT_NEMO_CSV + ".",
        f"\nCalibration constants (analysis.margin_diagnostic, unchanged): "
        f"CALIB_MAX={CALIB_MAX:,}, CALIB_SEED={CALIB_SEED}, "
        f"MIN_CALIB_LINES={MIN_CALIB_LINES}, TOPK_MARGIN={TOPK_MARGIN}, "
        f"corpus files at {CORPUS_DIR}/{{lang}}_train.txt.",
        ""]
    os.makedirs(os.path.dirname(TAU_BUILD_MD), exist_ok=True)
    with open(TAU_BUILD_MD, "w") as f:
        f.write("\n".join(L))
    print("\n".join(L))
    return TAU_FLOOR21_NEMO_CSV


# ---------------------------------------------------------------------------
# STAGE "topk"
# ---------------------------------------------------------------------------

def _finalize_topk(fp_base: dict, n_chunks: int) -> None:
    """Concatenates every completed chunk's partial banking file (written
    under TOPK_CHUNKS_DIR by the chunk loop) into the final
    gate_topk_{lines,ids,scores}_nemo.npy arrays and fingerprint. Idempotent:
    if the final arrays and fingerprint already exist and match, does
    nothing. `n_chunks` is the caller's own count of expected chunks
    (TOTAL_LINES split at CHUNK_LINES): the chunk file list under
    TOPK_CHUNKS_DIR is asserted to be exactly [chunk_{i:05d}.npz for i in
    range(n_chunks)], not merely whatever chunk_*.npz files happen to be on
    disk, so a stray or missing chunk file is caught here rather than
    silently under- or over-counting affected lines."""
    expected_files = [os.path.join(TOPK_CHUNKS_DIR, f"chunk_{i:05d}.npz")
                      for i in range(n_chunks)]
    chunk_files = sorted(glob.glob(os.path.join(TOPK_CHUNKS_DIR, "chunk_*.npz")))
    if chunk_files != expected_files:
        missing = sorted(set(expected_files) - set(chunk_files))
        extra = sorted(set(chunk_files) - set(expected_files))
        raise RuntimeError(
            f"{TOPK_CHUNKS_DIR} does not contain exactly the {n_chunks:,} "
            f"expected chunk_{{i:05d}}.npz files; missing {len(missing)} "
            f"(e.g. {missing[:3]}), extra {len(extra)} (e.g. {extra[:3]})")

    lines_parts, ids_parts, scores_parts = [], [], []
    n_short_cands_total = 0
    n_inf_margin_total = 0
    for cf in chunk_files:
        with np.load(cf) as z:
            lines_parts.append(z["lines"])
            ids_parts.append(z["ids"])
            scores_parts.append(z["scores"])
            n_short_cands_total += int(z["n_short_cands"])
            n_inf_margin_total += int(z["n_inf_margin"])
    lines = (np.concatenate(lines_parts) if lines_parts
            else np.zeros(0, dtype=np.int64))
    ids = (np.concatenate(ids_parts) if ids_parts
          else np.zeros((0, TOPK_MARGIN), dtype=np.int16))
    scores = (np.concatenate(scores_parts) if scores_parts
             else np.zeros((0, TOPK_MARGIN), dtype=np.float32))
    order = np.argsort(lines)
    lines, ids, scores = lines[order], ids[order], scores[order]

    if len(np.unique(lines)) != len(lines):
        raise RuntimeError(
            f"finalized gate_topk_lines_nemo has {len(lines):,} entries but "
            f"only {len(np.unique(lines)):,} unique line indices after "
            f"concatenating {n_chunks:,} chunk files under "
            f"{TOPK_CHUNKS_DIR}; a line was banked by more than one chunk")

    n_affected = len(lines)
    fp = dict(fp_base)
    fp["n_affected"] = int(n_affected)
    fp["n_short_cands"] = n_short_cands_total
    fp["n_inf_margin"] = n_inf_margin_total

    if os.path.exists(GATE_TOPK_FP_NEMO):
        with open(GATE_TOPK_FP_NEMO) as f:
            prev = json.load(f)
        if prev == fp and all(os.path.exists(p) for p in
                              (GATE_TOPK_LINES_NEMO, GATE_TOPK_IDS_NEMO,
                               GATE_TOPK_SCORES_NEMO)):
            print(f"existing gate_topk_*_nemo arrays match the current "
                 f"fingerprint; skipping finalization ({fp['n_affected']:,} "
                 f"affected lines, {fp['n_short_cands']:,} short candidate "
                 f"lists, {fp['n_inf_margin']:,} with fewer than 2).")
            return
        if prev != fp:
            bad = sorted(k for k in fp if prev.get(k) != fp[k])
            raise RuntimeError(f"gate_topk finalization mismatch ({bad}); "
                               f"clear {GATE_TOPK_FP_NEMO} and the "
                               "gate_topk_*_nemo.npy files to rebuild")

    np.save(GATE_TOPK_LINES_NEMO, lines)
    np.save(GATE_TOPK_IDS_NEMO, ids)
    np.save(GATE_TOPK_SCORES_NEMO, scores)
    with open(GATE_TOPK_FP_NEMO + ".tmp", "w") as f:
        json.dump(fp, f)
    os.replace(GATE_TOPK_FP_NEMO + ".tmp", GATE_TOPK_FP_NEMO)
    print(f"Finalized topk banking: {n_affected:,} affected lines "
         f"({n_short_cands_total:,} with fewer than {TOPK_MARGIN} saved "
         f"candidates, {n_inf_margin_total:,} of those with fewer than 2, "
         "following the recorded margin-gate convention, "
         "analysis.margin_diagnostic's _gap()). Wrote "
         f"{GATE_TOPK_LINES_NEMO}, {GATE_TOPK_IDS_NEMO}, "
         f"{GATE_TOPK_SCORES_NEMO}, {GATE_TOPK_FP_NEMO}.")


def run_topk() -> str:
    os.makedirs(SCRATCH_DIR_NEMO, exist_ok=True)
    os.makedirs(TOPK_CHUNKS_DIR, exist_ok=True)
    for p in (FLAT_SET_CSV, TAU_FLOOR21_NEMO_CSV, TAU_FLAT_NEMO_CSV):
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"{p} missing; the gate stage refuses to bank candidates "
                "without a recorded flat set and both tau CSVs (run "
                "--stage flatrule and --stage tau first)")

    canonical = _canonical_langs()
    langs, lang_to_idx = _verify_variant_langs(canonical)
    n_lang = len(langs)
    train_counts = _load_train_counts()
    N = np.array([train_counts.get(l, 0) for l in langs], dtype=np.float64)

    low_n_mask = N < HEAD_N
    flat_df = pd.read_csv(FLAT_SET_CSV)
    flat_langs = set(flat_df.lang) if len(flat_df) else set()
    flat_mask = np.array([l in flat_langs for l in langs], dtype=bool)
    # Boolean lookup, not an index array: the chunk loop below tests
    # membership for every scored line (up to ~45M times), and an `in`
    # check against an index array would be an O(len(expanded_idx)) linear
    # scan per line; expanded_mask[lang_id] is O(1).
    expanded_mask = low_n_mask | flat_mask

    # Floor-21 fingerprint read (not rebuild): the "tau" stage (a required
    # precondition, checked above via TAU_FLOOR21_NEMO_CSV/TAU_FLAT_NEMO_CSV)
    # already built and fingerprinted the floor-21 matrix via
    # _build_and_fingerprint_floor21_nemo. Reading its sha256_base_W /
    # sha256_w21 here, rather than rebuilding the matrix, lets the
    # already-finalized-run and chunk-fingerprint checks below run (and
    # potentially return or abort) before this stage pays the cost of
    # loading the model and rebuilding the matrix; _build_and_fingerprint_
    # floor21_nemo re-verifies these same values once the matrix actually
    # is rebuilt, further down, for the case where real work remains.
    if not os.path.exists(FP_FLOOR21_NEMO_PATH):
        raise FileNotFoundError(f"{FP_FLOOR21_NEMO_PATH} missing; run "
                                "--stage tau first")
    with open(FP_FLOOR21_NEMO_PATH) as f:
        fp_floor21 = json.load(f)
    _check_fingerprint_floor_target(fp_floor21, FP_FLOOR21_NEMO_PATH)
    sha_w = fp_floor21["sha256_base_W"]
    sha_w21 = fp_floor21["sha256_w21"]

    # Provenance/ordering guard: the flat-set CSV and both tau CSVs must
    # exist (checked above) and their content is embedded in this stage's
    # own fingerprint by sha256, so a topk resume against a flat set or tau
    # CSV that changed since the arrays were first banked is caught by the
    # standard fingerprint-mismatch abort in _finalize_topk, even though
    # this stage's own affected-line selection (expanded_mask, just above)
    # depends only on N and flat_mask, not on the tau values themselves.
    # No "floor_target" key here, deliberately. It would bind the clamp into
    # this fingerprint, but sha256_w21 -- the hash of the floor-21 matrix bytes
    # themselves -- already binds it strictly more tightly, and the fingerprint
    # is compared to the stored one by whole-dict equality, so adding a key would
    # make every already-completed run (including the RELEASED model's, whose
    # gate_topk_fingerprint_nemo.json has the pre-existing 17 keys) abort on
    # resume instead of returning idempotently. The clamp is checked directly
    # against FP_FLOOR21_NEMO_PATH's own floor_target field, just above.
    fp_base = {
        "sha256_base_W": sha_w,
        "sha256_w21": sha_w21,
        "langs_sha256": _sha256_bytes("|".join(langs).encode()),
        "flat_set_csv_sha256": _sha256_file(FLAT_SET_CSV),
        "tau_floor21_csv_sha256": _sha256_file(TAU_FLOOR21_NEMO_CSV),
        "tau_flat_csv_sha256": _sha256_file(TAU_FLAT_NEMO_CSV),
        "head_n": HEAD_N,
        "topk_margin": TOPK_MARGIN,
        "n_lang": n_lang,
        "n_low_n": int(low_n_mask.sum()),
        "n_flat": int(flat_mask.sum()),
        "n_expanded": int(expanded_mask.sum()),
        "total_lines": TOTAL_LINES,
        "chunk_lines": CHUNK_LINES,
    }

    # Already-finalized-run early return: gate_topk_fingerprint_nemo.json is
    # written only by _finalize_topk, at the very end of a fully-completed
    # run. Checked here, ABOVE the matrix build below, so a completed stage
    # returns immediately without loading the model or rebuilding the
    # floor-21 matrix. n_affected/n_short_cands/n_inf_margin are excluded
    # from the comparison: they are outputs of a completed run, not part of
    # the configuration fp_base describes.
    if os.path.exists(GATE_TOPK_FP_NEMO):
        with open(GATE_TOPK_FP_NEMO) as f:
            prev = json.load(f)
        prev_no_n = {k: v for k, v in prev.items()
                    if k not in ("n_affected", "n_short_cands",
                                 "n_inf_margin")}
        if prev_no_n == fp_base and all(
                os.path.exists(p) for p in
                (GATE_TOPK_LINES_NEMO, GATE_TOPK_IDS_NEMO,
                 GATE_TOPK_SCORES_NEMO, PRED_NEMO_FLOOR21)):
            print("existing topk outputs match the current fingerprint; "
                 "skipping rescoring.")
            return GATE_TOPK_LINES_NEMO
        if prev_no_n != fp_base:
            bad = sorted(k for k in fp_base if prev_no_n.get(k) != fp_base[k])
            raise RuntimeError(f"topk scratch state mismatch ({bad}); clear "
                               f"the topk files in {SCRATCH_DIR_NEMO} or "
                               "investigate what changed")

    # Partial-chunk fingerprint (TOPK_CHUNKS_DIR/chunks_fingerprint.json):
    # written atomically before the first chunk of a run is scored, and
    # verified on every subsequent invocation before done_chunks (from
    # PROGRESS_TOPK_PATH) is honoured. Distinct from the already-finalized-
    # run check above: gate_topk_fingerprint_nemo.json is only written at
    # the very end (_finalize_topk), so a run interrupted mid-way and
    # resumed under a changed configuration would otherwise have nothing
    # checking the partial chunk_*.npz files already on disk under
    # TOPK_CHUNKS_DIR against the new configuration before they get
    # concatenated in at finalization.
    if os.path.exists(CHUNKS_FP_PATH):
        with open(CHUNKS_FP_PATH) as f:
            prev_chunks_fp = json.load(f)
        if prev_chunks_fp != fp_base:
            bad = sorted(k for k in fp_base
                        if prev_chunks_fp.get(k) != fp_base[k])
            raise RuntimeError(
                f"{CHUNKS_FP_PATH} does not match the current configuration "
                f"(mismatched: {bad}); {PROGRESS_TOPK_PATH} and the partial "
                f"chunk files under {TOPK_CHUNKS_DIR} were produced under a "
                "different configuration and cannot be resumed from; clear "
                f"{TOPK_CHUNKS_DIR} and {PROGRESS_TOPK_PATH} to restart, or "
                "investigate what changed")
    else:
        with open(CHUNKS_FP_PATH + ".tmp", "w") as f:
            json.dump(fp_base, f)
        os.replace(CHUNKS_FP_PATH + ".tmp", CHUNKS_FP_PATH)

    # Only now build (and cache) the actual floor-21 matrix: both checks
    # above are satisfied, so real scoring work remains and the model-load /
    # matrix-build cost is justified. _build_and_fingerprint_floor21_nemo
    # re-verifies sha_w/sha_w21 against FP_FLOOR21_NEMO_PATH itself.
    matrix, sha_w, sha_w21 = _build_and_fingerprint_floor21_nemo(langs)

    y_mm = np.lib.format.open_memmap(os.path.join(BASE_SCRATCH, "y_true.npy"),
                                     mode="r")
    if y_mm.shape != (TOTAL_LINES,):
        raise RuntimeError(f"y_true.npy shape {y_mm.shape} != ({TOTAL_LINES},)")

    if os.path.exists(PRED_NEMO_FLOOR21):
        pred_mm = np.lib.format.open_memmap(PRED_NEMO_FLOOR21, mode="r+")
        if pred_mm.shape != (TOTAL_LINES,):
            raise RuntimeError(f"{PRED_NEMO_FLOOR21} shape {pred_mm.shape} "
                               f"!= ({TOTAL_LINES},)")
    else:
        pred_mm = np.lib.format.open_memmap(PRED_NEMO_FLOOR21, mode="w+",
                                            dtype=np.int16, shape=(TOTAL_LINES,))
        pred_mm[:] = UNSEEN
        pred_mm.flush()

    done_chunks = set()
    if os.path.exists(PROGRESS_TOPK_PATH):
        with open(PROGRESS_TOPK_PATH) as f:
            done_chunks = set(json.load(f))

    n_chunks = (TOTAL_LINES + CHUNK_LINES - 1) // CHUNK_LINES
    model = None
    with open(TEST_FILE) as fh:
        for chunk in range(n_chunks):
            lo = chunk * CHUNK_LINES
            hi = min(lo + CHUNK_LINES, TOTAL_LINES)
            chunk_npz = os.path.join(TOPK_CHUNKS_DIR, f"chunk_{chunk:05d}.npz")
            if chunk in done_chunks:
                for _ in range(hi - lo):
                    fh.readline()
                continue
            if model is None:
                print("Loading the variant model + caching floor-21 "
                     "weights for the combined topk pass...", flush=True)
                model = _load_unilid_model(PACKED_MODEL_PATH)
                if model.langs != langs:
                    raise RuntimeError(
                        "_load_unilid_model's language list differs from "
                        "_load_model_data's for the variant")
                model.model.set_weight_sets(matrix.tolist())
                del matrix
            lines = [fh.readline() for _ in range(hi - lo)]

            keep_pos, texts = [], []
            for j, line in enumerate(lines):
                i = lo + j
                if y_mm[i] < 0:
                    continue
                _label, text = _parse_line(line)
                keep_pos.append(i)
                texts.append(text)

            pre, valid = [], []
            for k, t in enumerate(texts):
                p = model.preprocess(t)
                if p:
                    pre.append(p)
                    valid.append(k)
            top1_out = np.full(len(texts), EMPTY, dtype=np.int16)
            aff_lines, aff_ids, aff_scores = [], [], []
            n_short_cands_chunk = 0
            n_inf_margin_chunk = 0
            for lo2 in range(0, len(pre), SCORE_BATCH_MAX):
                hi2 = min(lo2 + SCORE_BATCH_MAX, len(pre))
                topk = model.model.top_k_of_cached_weight_sets_batch(
                    pre[lo2:hi2], TOPK_MARGIN)
                if len(topk) != hi2 - lo2:
                    raise RuntimeError(f"chunk {chunk}: top-k scorer "
                                       f"returned {len(topk)} results for "
                                       f"{hi2 - lo2} inputs")
                for k, cands in enumerate(topk):
                    pos_in_texts = valid[lo2 + k]
                    if len(cands) > TOPK_MARGIN:
                        raise RuntimeError(
                            f"chunk {chunk}: top-k scorer returned "
                            f"{len(cands)} candidates, expected at most "
                            f"{TOPK_MARGIN}")
                    if not cands:
                        continue
                    top1 = int(cands[0][0])
                    top1_out[pos_in_texts] = np.int16(top1)
                    if len(cands) < TOPK_MARGIN:
                        n_short_cands_chunk += 1
                        # Recorded margin-gate convention (analysis/
                        # margin_diagnostic.py's _gap(), used operationally
                        # in analysis/full_test_margin.py and mirrored by
                        # analysis.gate_variants._run_topk): a line with
                        # fewer than 2 candidates has no top1-minus-top2
                        # margin to compute, so it is treated as having an
                        # infinite margin and is never moved by any
                        # variant's threshold gate.
                        if len(cands) < 2:
                            n_inf_margin_chunk += 1
                    if expanded_mask[top1]:
                        ids_row = np.full(TOPK_MARGIN, -1, dtype=np.int16)
                        scores_row = np.full(TOPK_MARGIN, -np.inf,
                                             dtype=np.float32)
                        for jj, (cid, cscore) in enumerate(cands):
                            ids_row[jj] = np.int16(cid)
                            scores_row[jj] = np.float32(cscore)
                        aff_lines.append(keep_pos[pos_in_texts])
                        aff_ids.append(ids_row)
                        aff_scores.append(scores_row)

            pred_mm[np.asarray(keep_pos, dtype=np.int64)] = top1_out
            pred_mm.flush()

            if aff_lines:
                np.savez(chunk_npz,
                        lines=np.asarray(aff_lines, dtype=np.int64),
                        ids=np.asarray(aff_ids, dtype=np.int16),
                        scores=np.asarray(aff_scores, dtype=np.float32),
                        n_short_cands=n_short_cands_chunk,
                        n_inf_margin=n_inf_margin_chunk)
            else:
                np.savez(chunk_npz,
                        lines=np.zeros(0, dtype=np.int64),
                        ids=np.zeros((0, TOPK_MARGIN), dtype=np.int16),
                        scores=np.zeros((0, TOPK_MARGIN), dtype=np.float32),
                        n_short_cands=n_short_cands_chunk,
                        n_inf_margin=n_inf_margin_chunk)

            done_chunks.add(chunk)
            with open(PROGRESS_TOPK_PATH + ".tmp", "w") as f:
                json.dump(sorted(done_chunks), f)
            os.replace(PROGRESS_TOPK_PATH + ".tmp", PROGRESS_TOPK_PATH)
            print(f"topk chunk {chunk + 1}/{n_chunks} done ({hi - lo} "
                 f"lines, {len(aff_lines)} affected, "
                 f"{n_short_cands_chunk} short candidate lists)", flush=True)

    pred_all = np.asarray(pred_mm)
    y = np.asarray(y_mm)
    kept = y >= 0
    if int((pred_all[kept] == UNSEEN).sum()) != 0:
        raise RuntimeError("unprocessed kept lines remain in pred_nemo_floor21")
    n_kept = int(kept.sum())
    print(f"combined topk pass done: {n_kept:,} kept lines scored.")

    _finalize_topk(fp_base, n_chunks)

    lines_final = np.load(GATE_TOPK_LINES_NEMO)
    ids_final = np.load(GATE_TOPK_IDS_NEMO)
    n_affected = len(lines_final)
    if not (0 < n_affected < n_kept):
        raise RuntimeError(
            f"{n_affected:,} affected lines banked, expected strictly "
            f"between 0 and the {n_kept:,} kept lines; this basic sanity "
            "bound (not the base model's historical 2.0M-2.6M range, which "
            "was calibrated for a different expanded label set) failed")
    # By construction (rank-1 of the same call written to both
    # pred_nemo_floor21 and gate_topk_ids' first column), the two must be
    # bit-identical, a strictly stronger check than
    # analysis.gate_variants._run_topk's TOP1_AGREE_MIN bar.
    if not np.array_equal(ids_final[:, 0], pred_all[lines_final]):
        raise RuntimeError(
            "banked top-1 candidates disagree with pred_nemo_floor21.npy at "
            "the same lines; both come from the same call in the same pass "
            "and must be bit-identical")
    print(f"identity check passed: banked rank-1 == pred_nemo_floor21 at "
         f"all {n_affected:,} affected lines.")
    print(f"STAGE topk done. Outputs: {PRED_NEMO_FLOOR21}, "
         f"{GATE_TOPK_LINES_NEMO}, {GATE_TOPK_IDS_NEMO}, "
         f"{GATE_TOPK_SCORES_NEMO}, {GATE_TOPK_FP_NEMO}.")
    return GATE_TOPK_LINES_NEMO


# ---------------------------------------------------------------------------
# STAGE "eval" (login node)
# ---------------------------------------------------------------------------

def _load_gate_thresholds_nemo(langs: list[str], N: np.ndarray) -> dict:
    """Variant analogue of analysis.external_bench_eval._load_gate_thresholds,
    pointed at this module's own tau CSVs and flat-set CSV. Unlike that
    function (which hardcodes the base model's flat-four at exactly 4 via
    N_FLAT4), the variant's group B size is whatever run_flatrule found,
    including zero; both are handled without a fixed-count assertion."""
    n_lang = len(langs)
    lang_to_pos = {l: i for i, l in enumerate(langs)}

    step1_langs = {langs[i] for i in range(n_lang) if N[i] < HEAD_N}
    tau1_row_count = len(pd.read_csv(TAU_FLOOR21_NEMO_CSV))
    if len(step1_langs) != tau1_row_count:
        raise RuntimeError(f"{len(step1_langs)} languages with N < HEAD_N "
                           f"({HEAD_N:,}), expected {tau1_row_count} (the "
                           f"data-row count of {TAU_FLOOR21_NEMO_CSV})")
    tau1, sha_tau1 = _load_tau_csv(TAU_FLOOR21_NEMO_CSV, langs, step1_langs)

    flat_df = pd.read_csv(FLAT_SET_CSV)
    step2_langs = set(flat_df.lang) if len(flat_df) else set()
    for lang in step2_langs:
        n_l = int(N[lang_to_pos[lang]])
        if n_l < HEAD_N:
            raise RuntimeError(
                f"flat-set language {lang} has N={n_l:,} < HEAD_N "
                f"({HEAD_N:,}); expected the flat set to be disjoint from "
                "the N < HEAD_N group so group A and group B are disjoint "
                "by construction")
    if step2_langs:
        tau2, sha_tau2 = _load_tau_csv(TAU_FLAT_NEMO_CSV, langs, step2_langs)
    else:
        # _load_tau_csv requires the CSV's "excluded" column to have bool
        # dtype; an empty CSV (zero data rows) round-trips through pandas'
        # CSV writer/reader as dtype object (no values to infer bool from),
        # which _load_tau_csv correctly rejects for a non-empty expected set
        # but which is exactly TAU_FLAT_NEMO_CSV's correct content when the
        # variant's flat set is empty (STAGE tau writes exactly this; see
        # run_tau). Handled directly here rather than routed through
        # _load_tau_csv, whose dtype gate is not meant to be relaxed.
        if not os.path.exists(TAU_FLAT_NEMO_CSV):
            raise FileNotFoundError(f"tau CSV missing: {TAU_FLAT_NEMO_CSV}")
        tau2 = np.full(n_lang, -np.inf, dtype=np.float64)
        sha_tau2 = _sha256_file(TAU_FLAT_NEMO_CSV)

    if step1_langs & step2_langs:
        raise RuntimeError("group A (N < HEAD_N) and group B (the variant's "
                           "flat set) language sets overlap; expected "
                           "disjoint")
    step2_idx = np.array(sorted(lang_to_pos[l] for l in step2_langs),
                         dtype=np.int64)

    return {"tau1": tau1, "sha_tau1": sha_tau1, "tau2": tau2,
           "sha_tau2": sha_tau2, "step2_idx": step2_idx,
           "step1_langs": step1_langs, "step2_langs": step2_langs}


def _load_judge_split(kept: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Re-derives the seed-301 rule split exactly as analysis/paper_eval.py
    does, over the SAME kept pool (y_true.npy is reused unchanged from the
    base model, so the kept mask, and therefore this split, is identical for
    the variant). Returns (derive_idx, judge_idx, remainder_mask)."""
    val101_path = os.path.join(DRAW_DIR, f"val_lines_seed{SEEDS[0]}.npy")
    test201_path = os.path.join(DRAW_DIR, f"val_lines_seed{TEST_SEED}.npy")
    for p in (val101_path, test201_path, SPLIT_PATH):
        if not os.path.exists(p):
            raise FileNotFoundError(f"required artifact missing: {p}")
    val101 = np.load(val101_path)
    test201 = np.load(test201_path)
    if np.intersect1d(val101, test201).size:
        raise RuntimeError("test draw overlaps the working val draw")
    excl = np.zeros(TOTAL_LINES, bool)
    excl[val101] = True
    excl[test201] = True
    if int((kept & excl).sum()) != int(excl.sum()):
        raise RuntimeError("draw lines fall outside the kept pool")
    remainder_mask = kept & ~excl

    if int(remainder_mask.sum()) != EXPECTED_REMAINDER:
        raise RuntimeError(f"remainder {int(remainder_mask.sum()):,} != "
                           f"{EXPECTED_REMAINDER:,}")

    remainder_idx = np.where(remainder_mask)[0]
    u = np.random.default_rng(RULE_SPLIT_SEED).random(remainder_idx.size)
    derive_idx = remainder_idx[u < RULE_SPLIT_FRACTION]
    judge_idx = remainder_idx[u >= RULE_SPLIT_FRACTION]
    if len(derive_idx) != EXPECTED_DERIVATION:
        raise RuntimeError(f"derivation part {len(derive_idx):,} != "
                           f"EXPECTED_DERIVATION {EXPECTED_DERIVATION:,}")
    if len(judge_idx) != EXPECTED_JUDGE:
        raise RuntimeError(f"judge part {len(judge_idx):,} != EXPECTED_JUDGE "
                           f"{EXPECTED_JUDGE:,}")

    with np.load(SPLIT_PATH) as stored:
        if not (np.array_equal(stored["derive_idx"], derive_idx)
                and np.array_equal(stored["judge_idx"], judge_idx)):
            raise RuntimeError(f"split stored at {SPLIT_PATH} does not match "
                               "the recomputed seed-301 split")
    return derive_idx, judge_idx, remainder_mask


def run_eval() -> str:
    os.makedirs(os.path.join(OUT_ROOT, "tables"), exist_ok=True)
    os.makedirs(os.path.join(OUT_ROOT, "diagnostic"), exist_ok=True)

    canonical = _canonical_langs()
    langs, lang_to_idx = _verify_variant_langs(canonical)
    n_lang = len(langs)
    train_counts = _load_train_counts()
    N = np.array([train_counts.get(l, 0) for l in langs], dtype=np.int64)

    y = _load_base_y_true()
    kept = y >= 0
    n_kept = int(kept.sum())
    if n_kept != EXPECTED_KEPT:
        raise RuntimeError(f"kept pool {n_kept:,} != EXPECTED_KEPT "
                           f"({EXPECTED_KEPT:,})")
    n_excluded = int((y == EXCLUDED).sum())
    if n_excluded != EXPECTED_VAL_LINES:
        raise RuntimeError(f"y_true.npy has {n_excluded:,} EXCLUDED lines, "
                           f"expected exactly {EXPECTED_VAL_LINES:,}")
    yk = y[kept]

    preds = {}
    for cfg, path in (("nemo_baseline", PRED_NEMO_BASELINE),
                      ("nemo_floor21", PRED_NEMO_FLOOR21)):
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} missing; run the corresponding "
                                    "stage first")
        arr = np.asarray(np.lib.format.open_memmap(path, mode="r"))
        if arr.shape != (TOTAL_LINES,):
            raise RuntimeError(f"{path} shape {arr.shape} != ({TOTAL_LINES},)")
        n_bad = int((arr[kept] < -1).sum())
        if n_bad:
            raise RuntimeError(
                f"{path} has {n_bad:,} sentinel values below -1 on the kept "
                "pool (UNSEEN or EXCLUDED); the pass that produced it is "
                "incomplete")
        preds[cfg] = arr.astype(np.int64)

    # Weight-matrix sha binding across stages: the baseline scoring pass's
    # own matrix and the matrix built into the floor-21 build must be the
    # identical unmodified matrix, not merely two independent reads of
    # PACKED_MODEL_PATH (which could have been repacked between the two
    # stages' runs). _build_and_fingerprint_floor21_nemo already enforces
    # this at "tau"/"topk" build time; this is a cross-file integrity
    # re-check at eval time against the two already-written fingerprints
    # directly, no rebuild.
    for p in (FP_BASELINE_PATH, FP_FLOOR21_NEMO_PATH):
        if not os.path.exists(p):
            raise FileNotFoundError(f"{p} missing; run the corresponding "
                                    "stage first")
    with open(FP_BASELINE_PATH) as f:
        fp_baseline = json.load(f)
    with open(FP_FLOOR21_NEMO_PATH) as f:
        fp_floor21 = json.load(f)
    # The clamp this report is about to state as a constant must be the clamp the
    # floor-21 matrix behind the numbers was actually built at.
    _check_fingerprint_floor_target(fp_floor21, FP_FLOOR21_NEMO_PATH)
    if fp_baseline["weight_matrix_sha256"] != fp_floor21["sha256_base_W"]:
        raise RuntimeError(
            f"weight-matrix sha mismatch: {FP_BASELINE_PATH}'s recorded "
            f"weight_matrix_sha256 "
            f"({fp_baseline['weight_matrix_sha256'][:16]}...) does not "
            f"match {FP_FLOOR21_NEMO_PATH}'s recorded sha256_base_W "
            f"({fp_floor21['sha256_base_W'][:16]}...); the baseline "
            "scoring pass and the floor-21 matrix build used different "
            "underlying weight matrices")

    for p in (GATE_TOPK_LINES_NEMO, GATE_TOPK_IDS_NEMO, GATE_TOPK_SCORES_NEMO,
             GATE_TOPK_FP_NEMO):
        if not os.path.exists(p):
            raise FileNotFoundError(f"{p} missing; run --stage topk first")
    gate_lines = np.load(GATE_TOPK_LINES_NEMO)
    gate_ids = np.load(GATE_TOPK_IDS_NEMO).astype(np.int64)
    gate_scores = np.load(GATE_TOPK_SCORES_NEMO)
    if gate_scores.dtype != np.float32:
        raise RuntimeError(f"{GATE_TOPK_SCORES_NEMO} has dtype "
                           f"{gate_scores.dtype}, expected float32")

    # gate_topk_fingerprint_nemo.json vs current state: catches an eval run
    # against topk outputs banked under a different flat set, tau CSVs,
    # language order, or gate constants than are currently in effect.
    with open(GATE_TOPK_FP_NEMO) as f:
        fp_topk = json.load(f)
    fresh_topk_fields = {
        "flat_set_csv_sha256": _sha256_file(FLAT_SET_CSV),
        "tau_floor21_csv_sha256": _sha256_file(TAU_FLOOR21_NEMO_CSV),
        "tau_flat_csv_sha256": _sha256_file(TAU_FLAT_NEMO_CSV),
        "langs_sha256": _sha256_bytes("|".join(langs).encode()),
        "head_n": HEAD_N,
        "topk_margin": TOPK_MARGIN,
    }
    bad_topk = sorted(k for k, v in fresh_topk_fields.items()
                      if fp_topk.get(k) != v)
    if bad_topk:
        raise RuntimeError(
            f"{GATE_TOPK_FP_NEMO} does not match the current state "
            f"(mismatched: {bad_topk}); the topk stage's outputs were "
            "produced under a different flat set, tau CSVs, language "
            "order, or gate constants than are currently in effect")
    n_affected_fp = fp_topk["n_affected"]
    if len(gate_lines) != n_affected_fp:
        raise RuntimeError(
            f"{GATE_TOPK_LINES_NEMO} has {len(gate_lines):,} lines, but "
            f"{GATE_TOPK_FP_NEMO} records n_affected={n_affected_fp:,}")
    if gate_ids.shape != (n_affected_fp, TOPK_MARGIN):
        raise RuntimeError(
            f"{GATE_TOPK_IDS_NEMO} shape {gate_ids.shape} != "
            f"({n_affected_fp:,}, {TOPK_MARGIN}) "
            "(fp_topk['n_affected'], TOPK_MARGIN)")
    if gate_scores.shape != (n_affected_fp, TOPK_MARGIN):
        raise RuntimeError(
            f"{GATE_TOPK_SCORES_NEMO} shape {gate_scores.shape} != "
            f"({n_affected_fp:,}, {TOPK_MARGIN}) "
            "(fp_topk['n_affected'], TOPK_MARGIN)")

    thresholds = _load_gate_thresholds_nemo(langs, N)
    tau1, tau2 = thresholds["tau1"], thresholds["tau2"]
    step2_idx = thresholds["step2_idx"]

    # Gate-group membership: every banked line's floor-21 base prediction
    # must fall in exactly one of group A (N < HEAD_N) or group B (the flat
    # set) -- the two are disjoint by construction (_load_gate_thresholds_
    # nemo aborts otherwise) -- because the topk stage only banks candidates
    # for lines whose rank-1 prediction is in this expanded set (run_topk's
    # own expanded_mask). A banked line outside both groups would mean the
    # topk stage's own banking logic and this stage's gate-group definition
    # have diverged.
    low_n_mask = N < HEAD_N
    flat_mask = np.zeros(n_lang, dtype=bool)
    flat_mask[step2_idx] = True
    expanded_mask = low_n_mask | flat_mask

    base_line_pred = preds["nemo_floor21"][gate_lines]
    if (base_line_pred < 0).any():
        raise RuntimeError(f"{PRED_NEMO_FLOOR21} has a negative prediction "
                           "at one or more banked line indices")
    if not expanded_mask[base_line_pred].all():
        n_outside = int((~expanded_mask[base_line_pred]).sum())
        raise RuntimeError(
            f"{n_outside:,} of {len(gate_lines):,} banked lines have a "
            "floor-21 base prediction outside both group A (N < HEAD_N) "
            "and group B (the flat set); every banked line's base "
            "prediction must fall in exactly one gate group")
    agree_mask = gate_ids[:, 0] == base_line_pred
    n_disagree = int((~agree_mask).sum())
    if n_disagree:
        raise RuntimeError(
            f"{n_disagree:,} of {len(gate_lines):,} banked lines have a "
            "top-1 candidate that disagrees with pred_nemo_floor21.npy at "
            "the same line; by the topk stage's own single-pass "
            "construction these must be identical, so this indicates one "
            "of the two files was regenerated independently of the other")
    print(f"agree_mask check passed: banked top-1 matches "
         f"{PRED_NEMO_FLOOR21} on all {len(gate_lines):,} banked lines "
         "(mirrors analysis.gate_variants' own agree-mask carve-out; here "
         "the carve-out is vacuous by construction, so this is an "
         "assertion, not a filter).")

    pred_banked_gated, stats_a, stats_b = _gate_walk_and_merge(
        base_line_pred, gate_ids, gate_scores, N, tau1, tau2, step2_idx,
        agree_mask=agree_mask)
    pred_nemo_gated = np.array(preds["nemo_floor21"], dtype=np.int64)
    pred_nemo_gated[gate_lines] = pred_banked_gated
    np.save(PRED_NEMO_GATED, pred_nemo_gated.astype(np.int16))
    preds["nemo_gated"] = pred_nemo_gated
    n_bad_gated = int((pred_nemo_gated[kept] < -1).sum())
    if n_bad_gated:
        raise RuntimeError(f"pred_nemo_gated has {n_bad_gated:,} sentinel "
                           "values below -1 on the kept pool")
    n_gated_diff = int((pred_nemo_gated[kept]
                       != preds["nemo_floor21"][kept]).sum())

    n_empty_cfg = {c: int((preds[c][kept] == EMPTY).sum()) for c in CONFIGS}

    tau1_df = pd.read_csv(TAU_FLOOR21_NEMO_CSV)
    tau2_df = pd.read_csv(TAU_FLAT_NEMO_CSV)
    n_excluded_tau1 = int(tau1_df["excluded"].sum()) if len(tau1_df) else 0
    n_excluded_tau2 = int(tau2_df["excluded"].sum()) if len(tau2_df) else 0

    exam_rows = [
        ["A (N < HEAD_N)", f"{stats_a['n_examined']:,}",
         f"{stats_a['n_moved']:,}", f"{stats_a['n_blocked_by_proximity']:,}",
         f"{stats_a['n_no_cand']:,}"],
        ["B (flat set)", f"{stats_b['n_examined']:,}",
         f"{stats_b['n_moved']:,}", f"{stats_b['n_blocked_by_proximity']:,}",
         f"{stats_b['n_no_cand']:,}"],
    ]
    exam_md = to_markdown(
        exam_rows,
        ["group", "examined", "moved", "blocked_by_proximity", "no_cand"],
        caption="Re-examination accounting (gated configuration)")

    derive_idx, judge_idx, remainder_mask = _load_judge_split(kept)
    yj = y[judge_idx]
    n_judge = len(judge_idx)
    print(f"seed-{RULE_SPLIT_SEED} judge split reproduced: "
         f"{len(derive_idx):,} derivation / {len(judge_idx):,} judge; "
         f"matches {SPLIT_PATH}.")

    stats_fullpool = {c: _per_lang_stats(preds[c][kept], yk, n_lang)
                     for c in CONFIGS}
    stats_judge = {c: _per_lang_stats(preds[c][judge_idx], yj, n_lang)
                  for c in CONFIGS}
    support_fullpool = np.bincount(yk, minlength=n_lang).astype(float)
    support_judge = np.bincount(yj, minlength=n_lang).astype(int)

    macro_f1_fullpool = {c: float(stats_fullpool[c][2].mean()) for c in CONFIGS}
    macro_f1_judge = {c: float(stats_judge[c][2].mean()) for c in CONFIGS}
    macro_fpr_fullpool = {
        c: float(_macro_fpr(stats_fullpool[c][4], support_fullpool,
                            n_kept).mean()) for c in CONFIGS}
    macro_fpr_judge = {
        c: float(_macro_fpr(stats_judge[c][4], support_judge.astype(float),
                            n_judge).mean()) for c in CONFIGS}

    # --- paired bootstrap, judge part: (nemo_gated - nemo_baseline) ---
    f1_judge_vec = {c: stats_judge[c][2] for c in CONFIGS}
    rng = np.random.default_rng(BOOT_SEED)
    resample = rng.integers(0, n_lang, size=(BOOT_B, n_lang))
    boots = {c: f1_judge_vec[c][resample].mean(axis=1) for c in CONFIGS}
    d = boots["nemo_gated"] - boots["nemo_baseline"]
    boot_point = float(f1_judge_vec["nemo_gated"].mean()
                       - f1_judge_vec["nemo_baseline"].mean())
    boot_lo, boot_hi = (float(v) for v in np.percentile(d, [2.5, 97.5]))

    # --- comparability row (recorded measurement, not a gate) ---
    diff_f1 = macro_f1_fullpool["nemo_baseline"] - PAPER_MISTRALNEMO_F1_FULLPOOL
    diff_fpr = (macro_fpr_fullpool["nemo_baseline"]
               - PAPER_MISTRALNEMO_FPR_FULLPOOL)

    # --- degeneracy caveat ---
    degenerate_langs = _read_degenerate_langs()
    degenerate_set = set(degenerate_langs)

    git_commit = _git_commit()

    def _write_per_lang_csv(path, stats, support):
        out = {"lang": langs, "N": N,
              "degenerate": [l in degenerate_set for l in langs]}
        for c in CONFIGS:
            out[f"f1_{c}"] = stats[c][2]
            out[f"fp_{c}"] = stats[c][4].astype(int)
        out["support"] = support
        pd.DataFrame(out).to_csv(path, index=False)

    _write_per_lang_csv(OUT_CSV_FULLPOOL, stats_fullpool,
                       support_fullpool.astype(int))
    _write_per_lang_csv(OUT_CSV_JUDGE, stats_judge, support_judge)

    # --- report ---
    L = [
        "# E3: Mistral-Nemo variant evaluation\n",
        "Pre-registration: EXPERIMENTS_PLAN.md, \"Camera-ready evaluation "
        "program (2026-08-06)\", E3. Model: "
        f"`{os.path.basename(PACKED_MODEL_PATH)}` "
        f"({PACKED_MODEL_PATH}). Configurations: nemo_baseline (unmodified "
        "matrix), nemo_floor21 (the variant's own floor-21 matrix, no "
        "gate), nemo_gated (floor-21 plus the promoted configuration's "
        "two-step re-examination, both tau sets recalibrated for this "
        "matrix, D3_PROX/RES_CAP/HEAD_N unchanged).\n",
        "## Gates passed\n",
        f"- Language order: the variant's {n_lang:,}-language list matches "
        "the canonical order.",
        f"- y_true.npy reuse: shape ({TOTAL_LINES:,},), no UNSEEN, "
        f"{n_excluded:,} EXCLUDED lines (== EXPECTED_VAL_LINES "
        f"{EXPECTED_VAL_LINES:,}).",
        f"- Full kept pool: {n_kept:,} lines (== EXPECTED_KEPT "
        f"{EXPECTED_KEPT:,}).",
        f"- Seed-{RULE_SPLIT_SEED} judge split: {len(derive_idx):,} "
        f"derivation / {len(judge_idx):,} judge (== EXPECTED_DERIVATION "
        f"{EXPECTED_DERIVATION:,} / EXPECTED_JUDGE {EXPECTED_JUDGE:,}); "
        f"matches the stored record at {SPLIT_PATH}.",
        "- Sentinel guard: no value < -1 on the kept pool for "
        "nemo_baseline, nemo_floor21, or nemo_gated.",
        "- Banked-array identity: gate_topk_ids[:,0] == pred_nemo_floor21 "
        "at every banked line (asserted at both the topk and eval stages).",
        f"- Weight-matrix sha: {FP_BASELINE_PATH}'s weight_matrix_sha256 "
        f"matches {FP_FLOOR21_NEMO_PATH}'s sha256_base_W.",
        f"- {GATE_TOPK_FP_NEMO}: flat_set/tau CSV shas, langs_sha256, "
        "head_n, topk_margin, n_affected, and gate_ids/gate_scores shape "
        "all match the current state.",
        "- Gate-group membership: every banked line's floor-21 base "
        "prediction falls in exactly one of group A (N < HEAD_N) or "
        "group B (the flat set).",
        "",
        to_markdown(
            [[c, f"{macro_f1_fullpool[c]:.4f}",
             f"{macro_fpr_fullpool[c] * FPR_SCALE:.4f}"] for c in CONFIGS],
            ["config", "macro F1", FPR_HEADER],
            caption=FULLPOOL_INSTRUMENT),
        to_markdown(
            [[c, f"{macro_f1_judge[c]:.4f}",
             f"{macro_fpr_judge[c] * FPR_SCALE:.4f}"] for c in CONFIGS],
            ["config", "macro F1", FPR_HEADER],
            caption=JUDGE_INSTRUMENT),
        f"- EMPTY (-1) predictions on the {FULLPOOL_INSTRUMENT}: " +
        ", ".join(f"{c} {n_empty_cfg[c]:,}" for c in CONFIGS) + ".",
        "",
        "## Re-examination accounting\n",
        exam_md,
        f"\nGroup A tau CSV ({TAU_FLOOR21_NEMO_CSV}): {n_excluded_tau1:,} "
        f"of {len(tau1_df):,} languages excluded (low_calibration or "
        f"zero_strength). Group B tau CSV ({TAU_FLAT_NEMO_CSV}): "
        f"{n_excluded_tau2:,} of {len(tau2_df):,} languages excluded.",
        f"\n{n_gated_diff:,} of {n_kept:,} kept lines have nemo_gated != "
        "nemo_floor21 (the lines the two-step re-examination actually "
        "moved).",
        f"\nTopk stage candidate-list shortfall (persisted in "
        f"{GATE_TOPK_FP_NEMO}): {fp_topk['n_short_cands']:,} of "
        f"{fp_topk['n_affected']:,} affected lines returned fewer than "
        f"{TOPK_MARGIN} saved candidates; {fp_topk['n_inf_margin']:,} of "
        "those returned fewer than 2 and are treated as having infinite "
        "margin (never moved), following the recorded margin-gate "
        "convention (analysis.margin_diagnostic's _gap()).\n",
        f"## Paired bootstrap, {JUDGE_INSTRUMENT}\n",
        f"B={BOOT_B:,}, seed={BOOT_SEED}, percentile 95% interval, paired "
        f"resample over the {n_lang:,} language positions. "
        "(nemo_gated - nemo_baseline) judge-part mean F1: "
        f"{boot_point:+.4f} [{boot_lo:+.4f}, {boot_hi:+.4f}].\n",
        "## Comparability to the paper's own Mistral-Nemo row (recorded "
        "measurement, not a gate)\n",
        "The paper's UniLID-Mistral-Nemo full-pool cell "
        "(paper/tables/lid_main.tex) is F1 "
        f"{PAPER_MISTRALNEMO_F1_FULLPOOL}, FPR "
        f"{PAPER_MISTRALNEMO_FPR_FULLPOOL:.2e} (raw scientific notation, "
        f"not the x1e5-scaled convention used in the table above), "
        f"computed over the paper team's own N = {TOTAL_LINES:,} lines. "
        f"This module's nemo_baseline full-pool cell is computed over the "
        f"{FULLPOOL_INSTRUMENT}: the two instruments differ by the "
        f"{n_excluded:,} retired validation lines (== EXPECTED_VAL_LINES "
        f"{EXPECTED_VAL_LINES:,}), so that difference is part of any gap "
        "below, not attributable to the retrain alone. That row is "
        "the paper team's own training run of the variant; this module "
        "evaluates an independent retrain from the same recipe, so rough "
        "proximity is expected, not equality. Measured nemo_baseline "
        f"full-pool: F1 {macro_f1_fullpool['nemo_baseline']:.4f} (diff "
        f"{diff_f1:+.4f}), FPR "
        f"{macro_fpr_fullpool['nemo_baseline']:.4e} (diff "
        f"{diff_fpr:+.4e}).\n",
        "## Degeneracy caveat\n",
        f"{len(degenerate_langs)} of {n_lang:,} rows are flagged degenerate "
        f"(fewer than 100 estimated tokens; "
        f"{DEGENERACY_MD}), adjudicated as an accepted model property "
        "(base-vocab script coverage in minority scripts: Ethiopic, "
        "Canadian syllabics, Syriac, Tibetan, and similar; "
        "EXPERIMENTS_CHRONOLOGICAL.md, 2026-08-07), not gated on here. "
        "Per-language F1 for these rows is carried in the `degenerate` "
        f"column of {OUT_CSV_FULLPOOL} and {OUT_CSV_JUDGE} for the reader "
        "to inspect directly.\n",
        "## Constants used\n",
        f"- HEAD_N = {HEAD_N:,} (analysis.full_test_margin)",
        f"- RES_CAP = {RES_CAP:,} (analysis.hierarchical_pool)",
        f"- D3_PROX = {D3_PROX} (analysis.gate_variants)",
        f"- FLOOR_TARGET = {FLOOR_TARGET} ({_floor_target_provenance()})",
        f"- TOPK_MARGIN = {TOPK_MARGIN} (analysis.margin_diagnostic)",
        f"- {TAU_FLOOR21_NEMO_CSV} (group A, {len(thresholds['step1_langs'])} "
        f"languages, sha256 {thresholds['sha_tau1'][:16]}...)",
        f"- {TAU_FLAT_NEMO_CSV} (group B, {len(thresholds['step2_langs'])} "
        f"languages, sha256 {thresholds['sha_tau2'][:16]}...)",
        f"\nPer-language detail: {OUT_CSV_FULLPOOL} ({FULLPOOL_INSTRUMENT}), "
        f"{OUT_CSV_JUDGE} ({JUDGE_INSTRUMENT}).",
        f"\nGit commit: {git_commit}.",
    ]
    with open(OUT_MD, "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))

    tex_rows_fullpool = [[_disp(c), macro_f1_fullpool[c],
                         f"{macro_fpr_fullpool[c] * FPR_SCALE:.4f}"]
                        for c in CONFIGS]
    tex_fullpool = to_latex(
        tex_rows_fullpool, ["Method", "Macro F1", FPR_HEADER],
        caption=f"Mistral-Nemo variant, {FULLPOOL_INSTRUMENT}. FPR values "
                "are multiplied by 1e5.",
        label="tab:mistralnemo_eval_fullpool",
        col_formats=["str", "metric", "str"])
    tex_rows_judge = [[_disp(c), macro_f1_judge[c],
                      f"{macro_fpr_judge[c] * FPR_SCALE:.4f}"] for c in CONFIGS]
    tex_judge = to_latex(
        tex_rows_judge, ["Method", "Macro F1", FPR_HEADER],
        caption=f"Mistral-Nemo variant, {JUDGE_INSTRUMENT}. FPR values are "
                "multiplied by 1e5.",
        label="tab:mistralnemo_eval_judge",
        col_formats=["str", "metric", "str"])
    tex_boot = to_latex(
        [[_disp("nemo_gated"), boot_point, boot_lo, boot_hi]],
        ["Comparator", "Mean diff", "CI low", "CI high"],
        caption=f"Paired bootstrap, gated minus baseline, {JUDGE_INSTRUMENT}.",
        label="tab:mistralnemo_eval_bootstrap",
        col_formats=["str", "metric", "metric", "metric"])
    with open(OUT_TEX, "w") as f:
        f.write("\n\n".join([tex_fullpool, tex_judge, tex_boot]) + "\n")

    print(f"\nWrote {OUT_MD}, {OUT_TEX}, {OUT_CSV_FULLPOOL}, {OUT_CSV_JUDGE}, "
         f"{PRED_NEMO_GATED}")
    return OUT_MD


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

STAGES = {
    "baseline": run_baseline,
    "calibval": run_calibval,
    "flatrule": run_flatrule,
    "tau": run_tau,
    "topk": run_topk,
    "eval": run_eval,
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="E3 Mistral-Nemo variant evaluation pipeline. Run stages "
                    "in order: baseline, calibval (SLURM, needs the full "
                    "model), flatrule (login node), tau, topk (SLURM), eval "
                    "(login node).")
    parser.add_argument("--stage", required=True, choices=list(STAGES))
    add_arguments(parser)
    parser.add_argument("--base-scratch", default=None,
                        help="the base model's output root, read for y_true.npy "
                             "and for the floor-target cross-check "
                             "(default: the released base model's)")
    parser.add_argument("--out-dir", default=None,
                        help="root for the tables/ and diagnostic/ files the "
                             "flatrule, tau and eval stages write and read back "
                             f"(default: {DEFAULT_OUT_ROOT}); required, and "
                             "required to be outside the default root, when "
                             "--model is not this chain's own packed model")
    parser.add_argument("--floor-target", type=float, default=None,
                        help="floor-21 clamp target the tau/topk/eval stages "
                             f"build and calibrate at (default: "
                             f"{DEFAULT_FLOOR_TARGET}, analysis.full_test_"
                             "floor21.FLOOR_TARGET); REQUIRED when --model is "
                             "not this chain's own packed model, because the "
                             "value is a measured per-model selection, not a "
                             "universal constant")
    args = parser.parse_args()
    configure(args.model_path, args.scratch_dir, args.base_scratch,
              out_dir=args.out_dir, floor_target=args.floor_target)
    STAGES[args.stage]()


if __name__ == "__main__":
    main()
