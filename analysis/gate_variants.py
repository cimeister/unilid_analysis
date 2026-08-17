"""Infrastructure for Experiments 47 to 49 (candidate directions 1 to 3 in
EXPERIMENTS_PLAN.md, section "Candidate directions from the post-promotion error
analysis").

The promoted configuration (floor21_gate) does two things to every test line. First,
in the model's weight matrix, every entry equal to a language's row minimum (a token
never seen in that language's training data) is lowered to FLOOR_TARGET, -21
(analysis.full_test_floor21). Second, a decision-time re-examination step: for every
line whose predicted language has fewer than HEAD_N, 18,000, training lines
(analysis.full_test_margin), if the winning score exceeds the second-place score by
less than a threshold calibrated on that language's own training lines, the
prediction moves to the highest-scoring alternative among the top five candidates
whose training corpus has at least RES_CAP, 100,000, lines
(analysis.hierarchical_pool). Directions 1 to 3 each propose a different
re-examination rule over the same floor-21 matrix: a different threshold, a
different set of re-examined languages, or a different acceptance condition on the
replacement candidate. This module scores the candidates those rules need once,
then applies each rule as cheap post-processing over the saved scores, so a new
variant costs no further model scoring.

Two stages, run separately through run() (see also __main__ below).

Stage "topk" (run("topk")) makes one pass over the full test pool. For every kept
line (a line whose true label was scored, i.e. y_true.npy >= 0) whose floor21
prediction (pred_floor21.npy) falls in an expanded label set, it saves the top five
(TOPK_MARGIN, analysis.margin_diagnostic) candidate language indices and their
scores under the floor-21 weight matrix. The expanded label set is the union of two
groups of languages: every language with fewer than HEAD_N training lines (the
promoted configuration's own re-examined set, which direction 1 keeps unchanged),
and every language whose category in outputs/diagnostic/lang_diagnostic.csv
(analysis.diagnostic's flat_magnet/tight_lowres/twin/isolated_tail/head/mid
classification) is flat_magnet, regardless of its training-line count. flat_magnet
covers 118 languages and is a strict subset of the 279 languages with flatness
score zH at or above ZH_MAGNET, 1.5 (analysis.diagnostic); four flat_magnet
languages, sco_Latn, arg_Latn, bjn_Latn, and vls_Latn, have at least HEAD_N
training lines and are exactly the languages direction 2 (Experiment 48) proposes
to add to the re-examined set. The expanded label set covers 1,084 languages and is
expected to affect 2,236,864 kept test lines; the topk stage aborts if the measured
count falls outside 2,000,000 to 2,600,000.
Outputs, all in the resolved output root (see configure()): gate_topk_lines.npy, gate_topk_ids.npy,
gate_topk_scores.npy, and gate_topk_fingerprint.json.

Stage "apply" (run("apply", variant=NAME)) reads those three saved arrays and the
fingerprint, and builds one full-pool prediction file for a named rule variant
declared in the VARIANTS dict below. No test-pool scoring happens in this stage:
gate_topk_ids.npy and gate_topk_scores.npy already carry every candidate score this
stage needs. One variant, "flat4_tau5" (direction 2), does its own scoring inside
this stage, but only of calibration lines drawn from training corpora, to compute
its per-language thresholds; that scoring is separate from, and does not modify,
gate_topk_*. Every variant declares a "base_pred" field naming the output-root
prediction file its output is a copy-and-selectively-modify of: the first variant,
"shared9_bar18k", uses pred_floor21.npy; the second, "flat4_tau5", uses
pred_floor21_gate.npy (the promoted configuration's own predictions); the third,
"flat4_prox21", uses pred_floor21.npy again (both of the promoted configuration's
re-examination steps are re-run fresh from there, not chained through
pred_floor21_gate.npy, so the added proximity condition described below applies to
step 1's moves too, not only step 2's). "flat4_prox21" does not run through the
generic single-threshold apply logic the other two variants share; it has its own
dedicated function, _run_apply_flat4_prox21, dispatched from _run_apply by variant
name (see that function's docstring).

The first variant, "shared9_bar18k", implements direction 1 (Experiment 47,
pre-registered in EXPERIMENTS_RESULTS.md): the re-examined set is exactly the
languages with fewer than HEAD_N training lines (the flat-distribution languages
added to the topk-stage candidate universe by the flat_magnet-category criterion
are excluded from this variant's re-examined set); for every affected line whose
floor21 prediction is in that set and whose saved top-1 candidate agrees with
pred_floor21.npy, if the
top-1 score exceeds the second-place score by less than SHARED_TAU_V1, 9.0 nats,
the prediction moves to the highest-scoring saved candidate among ranks 2 to 5 whose
training corpus has at least HEAD_N, 18,000, lines, in place of the promoted
configuration's own replacement bar of RES_CAP, 100,000, lines. Lines whose saved
top-1 disagrees with pred_floor21.npy are left unchanged and counted separately, not
re-examined.

The second variant, "flat4_tau5", implements direction 2 (Experiment 48,
pre-registered in EXPERIMENTS_RESULTS.md): the re-examined set is exactly the four
flat_magnet-category languages with at least HEAD_N training lines (sco_Latn,
bjn_Latn, arg_Latn, vls_Latn), re-examined on top of the promoted configuration's
own predictions (base_pred = pred_floor21_gate.npy) rather than pred_floor21.npy.
Because all four have N at or above HEAD_N, none of them is in floor21_gate's own
re-examined set (lines predicted into them pass through floor21_gate unchanged);
and because all four have N below RES_CAP, 100,000, floor21_gate never moves a line
into them either (its replacement bar requires N >= RES_CAP). The set
of lines predicted into one of the four is therefore identical under
pred_floor21_gate.npy and pred_floor21.npy, which this module asserts by count
equality rather than assuming (see _run_apply). The threshold is calibrated
separately per language (_calibrate_flat4_tau5): the floor-21 matrix is rebuilt and
sha256-verified (via _build_verified_floor21_matrix, the helper also used by stage
"topk"), cached with set_weight_sets, and for each of the four languages a sample of
up to CALIB_MAX training lines is scored; tau_L is the fixed MARGIN_Q-th (5th)
percentile of the finite margins on the lines L itself wins, aborting (not
excluding) if fewer than MIN_CALIB_LINES win, since these four languages have large
corpora and predicted-line counts and a shortfall would indicate a wiring error.
Thresholds are written to outputs/diagnostic/tau_flat4.csv (same columns as the
other tau CSVs). The acceptance bar for a replacement candidate is RES_CAP, 100,000,
the promoted configuration's own bar (unchanged, unlike direction 1).

The third variant, "flat4_prox21", implements direction 3 (Experiment 49,
pre-registered in EXPERIMENTS_RESULTS.md): a full reimplementation of the promoted
configuration's own two re-examination steps (step 1: languages with N < HEAD_N,
thresholds read directly from tau_floor21_gate.csv, the CSV analysis/solo_gates.py's
run("floor21") already wrote; step 2: the same four flat_magnet languages as
flat4_tau5, thresholds read directly from tau_flat4.csv), plus one added acceptance
condition on the shared replacement-candidate walk: a candidate must have a saved
score within D3_PROX, 21.0 nats, of the saved top-1 score, in addition to the
existing N >= RES_CAP bar. Because step 1 is re-run from pred_floor21.npy rather
than resumed from pred_floor21_gate.npy, the D3_PROX condition governs step 1's
moves as well as step 2's. Both tau CSVs are read directly (_load_tau_csv), not
recalibrated: this variant reuses the two prior variants' own calibration outputs
rather than re-deriving them. A mandatory self-check (inside
_run_apply_flat4_prox21, before the conditioned build is trusted) reruns the
identical two-step walk with the D3_PROX condition disabled and requires the result
to be bit-identical to pred_gate_flat4_tau5.npy in the output root (flat4_tau5's own
output) over the full array, aborting otherwise.

Pattern sources: analysis/solo_gates.py (floor-21 matrix rebuild and sha256
verification against fingerprint_floor21.json, affected-line selection, model
loading, _topk_batch usage, the per-line reassignment loop over saved candidates,
and for flat4_tau5 specifically, its per-language calibration loop: rng creation,
corpus sampling, own-win margin collection); analysis/mixed_matrix.py (fingerprint
and chunked resumable full-pool conventions); analysis/full_test_eval.py
(the output root, memmap sentinels, _parse_line).

Constants imported, never redefined here: HEAD_N (analysis.full_test_margin),
FLOOR_TARGET (analysis.full_test_floor21), TOPK_MARGIN, PRF_CSV, MARGIN_Q,
MIN_CALIB_LINES, CALIB_MAX, CALIB_SEED, CORPUS_DIR, and _gap
(analysis.margin_diagnostic), RES_CAP and DIAG_CSV (analysis.hierarchical_pool),
ZH_MAGNET (analysis.diagnostic).
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd

from analysis.config import TEST_FILE, TOTAL_LINES
from analysis.transfer_sweep import _load_model_data, _load_unilid_model
from analysis.full_test_eval import _parse_line
from analysis.full_test_margin import HEAD_N
from analysis.full_test_floor21 import FLOOR_TARGET
from analysis.floor_equalization import build_equalized_weights, _special_columns
from analysis.model_context import resolve
from analysis.hierarchical_pool import RES_CAP, DIAG_CSV
from analysis.margin_diagnostic import (_topk_batch, _gap, PRF_CSV, TOPK_MARGIN,
                                        MARGIN_Q, MIN_CALIB_LINES, CALIB_MAX,
                                        CALIB_SEED, CORPUS_DIR)
from analysis.diagnostic import ZH_MAGNET

# ---------------------------------------------------------------- model context
#
# This module's stages share their model and their output root across a dozen
# free functions, so the pair is configured once here rather than threaded
# through every signature. configure() is called by main(); anything that runs
# without it falls back to the released model and its own output root, which is
# the historical behaviour. The guard in analysis.model_context refuses to pair a
# non-default model with a store-backed root, which is what this directory is.
_CTX = None
_OUT_DIR = "outputs"


def configure(model_path: str = None, scratch_dir: str = None,
              out_dir: str = "outputs"):
    global _CTX, _OUT_DIR
    _CTX = resolve(model_path, scratch_dir, purpose="gate-variant scoring")
    _OUT_DIR = out_dir
    print(f"gate variants against {_CTX.describe()}\n  tables under {out_dir}",
          flush=True)
    return _CTX


def _ctx():
    global _CTX
    if _CTX is None:
        _CTX = resolve(None, None, purpose="gate-variant scoring")
    return _CTX


def _scratch(*parts) -> str:
    return os.path.join(_ctx().scratch_dir, *parts)


def _out(*parts) -> str:
    return os.path.join(_OUT_DIR, *parts)


# Bounds peak memory for the topk stage's saved-candidate accumulation: a prior
# single 2.17-million-line batch through _topk_batch was killed on the login node.
SCORE_BATCH_MAX = 200_000
# Block size for the chunked binary read of TEST_FILE in the topk stage. Reading the
# 45.6-million-line file in large binary blocks and splitting on newlines in bulk,
# instead of a per-line readline() loop, was measured (review, warm cache, on the
# real file) at 10.0 seconds for this helper, against 18.3 seconds for a plain
# readline() scan and 19.8 seconds for a readline() scan with a per-line membership
# check. The helper is kept for the roughly 2x saving over the membership-check scan
# and its verified correctness.
READ_BLOCK_BYTES = 64 * 1024 * 1024
# Top-1 agreement floor between the recomputed candidates and pred_floor21.npy on
# affected lines, the same bar analysis/solo_gates.py uses for its floor21 rebuild.
TOP1_AGREE_MIN = 0.99
# Experiment 47 pre-registration (EXPERIMENTS_RESULTS.md, direction 1): the shared
# re-examination threshold, in natural-log units. The sweep behind this value
# (outputs/diagnostic/gate_threshold_sweep_20260730.csv) found the derivation-part
# optimum flat between 7 and 12; this value is fixed, not swept, in this module.
SHARED_TAU_V1 = 9.0
# Experiment 48 pre-registration (EXPERIMENTS_RESULTS.md, direction 2): output path
# for the flat4_tau5 variant's four per-language calibrated thresholds, written by
# _calibrate_flat4_tau5 in the same format as the other tau_*.csv files (e.g.
# outputs/diagnostic/tau_floor21_gate.csv).
TAU_FLAT4_CSV_NAME = "diagnostic/tau_flat4.csv"
# Experiment 49 pre-registration (EXPERIMENTS_RESULTS.md, direction 3): input path
# for the flat4_prox21 variant's step-1 per-language thresholds. This CSV already
# exists (written by analysis/solo_gates.py's run("floor21"), the promoted
# configuration's own build); flat4_prox21 reads it directly rather than
# recalibrating, since the promoted configuration's step 1 is being re-run
# unchanged, only with the added proximity condition on the replacement walk.
TAU_FLOOR21_GATE_CSV_NAME = "diagnostic/tau_floor21_gate.csv"
# The default-root spellings, kept because analysis/external_bench_eval.py and
# analysis/commonlid_calibrated.py import them by name. Inside this module the
# configurable _out(..._NAME) form is used instead, so a corrected-model run
# writes its thresholds under its own output root.
TAU_FLAT4_CSV = os.path.join("outputs", TAU_FLAT4_CSV_NAME)
TAU_FLOOR21_GATE_CSV = os.path.join("outputs", TAU_FLOOR21_GATE_CSV_NAME)
# Experiment 49 pre-registration (direction 3): score-proximity bound on a
# replacement candidate, in natural-log units (top1_score - candidate_score must be
# at or below this to be accepted). Chosen on the derivation part by a grid search
# from 0.5 to 100 in steps of 1; the optimum plateau spans roughly 15 to 35, all
# within 0.0003 of each other, so 21.0 is representative of the plateau, not a
# finely-tuned optimum.
D3_PROX = 21.0


def _build_verified_floor21_matrix(langs: list) -> tuple[np.ndarray, str, str]:
    """Loads the model's weight matrix, verifies it against fingerprint_floor21
    .json's sha256_base_W, builds the floor-21 clamp (build_equalized_weights at
    FLOOR_TARGET), and verifies the result against the fingerprint's sha256_w21.
    `langs` must be the caller's own language order (checked against the model's);
    the caller does not need to separately verify either sha256. Returns
    (matrix, sha256_base_W_hex, sha256_w21_hex). Shared by stage "topk" (_run_topk)
    and the flat4_tau5 calibration helper (_calibrate_flat4_tau5), both of which
    need a bit-identical rebuild of the same floor-21 matrix."""
    n_lang = len(langs)
    weights, langs_m, _lang_to_idx = _load_model_data(_ctx().model_path)
    if langs_m != langs:
        raise RuntimeError("model language order does not match the caller's "
                           f"language list ({PRF_CSV})")
    W = np.array(weights, dtype=np.float32)
    del weights

    fp21_path = _scratch("fingerprint_floor21.json")
    with open(fp21_path) as f:
        fp21 = json.load(f)
    sha_w = hashlib.sha256(W.tobytes()).hexdigest()
    if sha_w != fp21["sha256_base_W"]:
        raise RuntimeError(f"loaded W does not match sha256_base_W in {fp21_path}")

    # Special columns by name; from 0.3.0 they are the row minimum and
    # would otherwise be selected as the plateau, disabling the clamp.
    matrix, n_mod = build_equalized_weights(
        W, FLOOR_TARGET, special_idx=_special_columns(_ctx().model_path))
    if n_mod != n_lang:
        raise RuntimeError(f"floor {FLOOR_TARGET} modified {n_mod} of {n_lang} "
                           "rows; expected all of them (row floors all exceed the "
                           "target, the precedent this model has always shown)")
    if not np.array_equal(matrix[:, :4], W[:, :4]):
        raise RuntimeError("special-token columns (0:4) were modified by the clamp")
    sha_w21 = hashlib.sha256(matrix.tobytes()).hexdigest()
    if sha_w21 != fp21["sha256_w21"]:
        raise RuntimeError(f"rebuilt floor-21 matrix does not match sha256_w21 in "
                           f"{fp21_path}")
    del W
    return matrix, sha_w, sha_w21


def _calibrate_flat4_tau5(langs: list, N: np.ndarray, reexam_idx: np.ndarray) -> dict:
    """Experiment 48 (direction 2) per-language threshold calibration for the
    flat4_tau5 variant, run inside the apply stage (no scoring happens in stage
    "topk" for this variant beyond what it already does for every language in the
    expanded label set). `reexam_idx` must be exactly the four flat-distribution,
    at-or-above-HEAD_N language indices (sco_Latn, bjn_Latn, arg_Latn, vls_Latn per
    the pre-registration); this is checked by the caller (_run_apply) before this
    function is invoked, and re-asserted here as a defensive invariant before the
    expensive matrix rebuild and model load.

    Rebuilds and sha256-verifies the floor-21 weight matrix (_build_verified_
    floor21_matrix, the same fingerprinted path stage "topk" and
    analysis/solo_gates.py use), caches it with set_weight_sets, then for each of
    the four languages samples up to CALIB_MAX training lines (seed CALIB_SEED, the
    same sampling pattern as analysis/solo_gates.py's calibration loop: one rng
    instance shared across languages, sorted np.random.Generator.choice indices)
    from its own CORPUS_DIR/{lang}_train.txt, scores the top TOPK_MARGIN candidates
    with _topk_batch, and sets tau_L to the MARGIN_Q-th percentile (fixed at 5, NOT
    the size-adaptive formula analysis/solo_gates.py uses for its own gate) of the
    finite top1-minus-top2 margins on the lines L itself wins (top1 == L).

    Aborts if any of the four has fewer than MIN_CALIB_LINES winning lines. This is
    an abort, not analysis/solo_gates.py's exclude-and-log convention: these four
    languages have 13,911 to 89,109 predicted test lines apiece and large training
    corpora, so a calibration shortfall here would indicate a wiring error (wrong
    corpus file, wrong cached matrix, wrong language index), not a genuine
    low-resource case.

    Writes the four thresholds to tau_flat4.csv under the output root,
    with the same columns as the other
    tau_*.csv files (lang, n_scoreable, n_self_won, tau, excluded, cause); "excluded"
    is always False and "cause" always empty here because a language that would
    need excluding instead aborts the run, per the above. Returns
    {language_index: tau} for exactly the four languages in reexam_idx."""
    reexam_idx = [int(i) for i in reexam_idx]
    if len(reexam_idx) != 4:
        raise RuntimeError("flat4_tau5 calibration expects exactly 4 re-examined "
                           f"languages, got {len(reexam_idx)}: "
                           f"{[langs[i] for i in reexam_idx]}")

    matrix, _sha_w, _sha_w21 = _build_verified_floor21_matrix(langs)
    model = _load_unilid_model(_ctx().model_path)
    print("Caching floor-21 weights for flat4_tau5 calibration...", flush=True)
    model.model.set_weight_sets(matrix.tolist())
    del matrix

    rng = np.random.default_rng(CALIB_SEED)
    tau = {}
    calib_rows = []
    for li in reexam_idx:
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
        if len(gaps) < MIN_CALIB_LINES:
            raise RuntimeError(
                f"{lang} has {len(gaps)} finite winning calibration margins, below "
                f"MIN_CALIB_LINES={MIN_CALIB_LINES}; flat4_tau5's four languages "
                "have large corpora and predicted-line counts, so this indicates a "
                "wiring error, not a genuine calibration shortfall, and aborts "
                "rather than excluding the language")
        t = float(np.percentile(gaps, MARGIN_Q))
        tau[li] = t
        calib_rows.append({"lang": lang, "n_scoreable": len(ctopk),
                           "n_self_won": len(wins), "tau": t,
                           "excluded": False, "cause": ""})
        print(f"flat4_tau5 calibration: {lang} tau={t:.4f} nats "
              f"({len(gaps):,} finite winning margins of {len(wins):,} self-won "
              f"lines, {len(ctopk):,} scoreable of {len(lines):,} sampled)",
              flush=True)

    os.makedirs(os.path.dirname(_out(TAU_FLAT4_CSV_NAME)), exist_ok=True)
    pd.DataFrame(calib_rows).to_csv(_out(TAU_FLAT4_CSV_NAME), index=False)
    print(f"Wrote {_out(TAU_FLAT4_CSV_NAME)}")
    return tau


def _load_tau_csv(path: str, langs: list, expected_langs: set) -> tuple[np.ndarray, str]:
    """Reads a tau_*.csv (the format written by analysis/solo_gates.py and
    _calibrate_flat4_tau5: columns lang, n_scoreable, n_self_won, tau, excluded,
    cause) and returns (tau_arr, sha256_hex). tau_arr has one entry per language in
    `langs` (same order), filled with -inf for every language not present in the
    CSV. For a language present in the CSV, tau_arr is set to -inf if its
    "excluded" column is True, else to its "tau" column value; this is asserted
    explicitly here (not merely assumed) even though both existing tau CSVs already
    write -inf into "tau" for excluded rows themselves, because a variant reading a
    third-party CSV should not depend on that convention holding silently.

    Aborts if the set of languages present in the CSV does not exactly equal
    `expected_langs`: a tau CSV that is missing a re-examined language, or that
    carries an extra one, would silently leave a threshold at -inf (never
    re-examined) or apply a threshold to a language never scored, either of which
    is a wiring error, not a case to tolerate."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"tau CSV missing: {path}")
    with open(path, "rb") as f:
        sha = hashlib.sha256(f.read()).hexdigest()
    df = pd.read_csv(path)
    missing_cols = [c for c in ("lang", "tau", "excluded") if c not in df.columns]
    if missing_cols:
        raise RuntimeError(f"{path} is missing required column(s) {missing_cols}")
    if df.excluded.dtype != bool:
        raise RuntimeError(
            f"{path} column 'excluded' has dtype {df.excluded.dtype}, expected "
            "bool; a non-bool dtype (e.g. an object column of string 'True'/"
            "'False', or ambiguous 0/1 integers) would silently corrupt the "
            "-inf/tau selection below")
    dup_langs = df.lang[df.lang.duplicated()].tolist()
    if dup_langs:
        raise RuntimeError(
            f"{path} has duplicate lang value(s): {dup_langs}; each language "
            "must appear at most once, or the tau_arr assignment below would "
            "silently keep only the last matching row's value")
    csv_langs = set(df.lang)
    if csv_langs != expected_langs:
        raise RuntimeError(
            f"{path} languages do not match the expected re-examined set: "
            f"{len(csv_langs - expected_langs)} language(s) in the CSV but not "
            f"expected, {len(expected_langs - csv_langs)} expected but missing "
            "from the CSV")
    lang_to_idx = {lang: i for i, lang in enumerate(langs)}
    tau_arr = np.full(len(langs), -np.inf, dtype=np.float64)
    for row in df.itertuples():
        tau_arr[lang_to_idx[row.lang]] = -np.inf if row.excluded else float(row.tau)
    return tau_arr, sha


VARIANTS = {
    "shared9_bar18k": {
        "experiment": 47,
        "description": (
            "Direction 1 (Experiment 47): one shared re-examination threshold of "
            f"{SHARED_TAU_V1} nats in place of the promoted configuration's "
            "1,080 per-language calibrated thresholds, and a replacement-candidate "
            f"minimum lowered from RES_CAP ({RES_CAP:,} training lines, the "
            f"promoted configuration's own bar) to HEAD_N ({HEAD_N:,} training "
            "lines)."
        ),
        # base_pred: the output-root prediction file this variant's output is a
        # copy-and-selectively-modify of. Direction 1 re-examines pred_floor21.npy
        # directly (the same array stage "topk" anchored its affected-line
        # selection on).
        "base_pred": "pred_floor21.npy",
        # Re-examined set: languages with N < HEAD_N only. Takes (N, zH, category);
        # zH and category are ignored here (the flat-distribution additions to the
        # topk-stage candidate universe are direction 2's concern, not direction
        # 1's).
        "reexamined_set": lambda N, zH, category: N < HEAD_N,
        "reexamined_set_desc": (
            f"languages with N < HEAD_N ({HEAD_N:,} training lines) only; the "
            "flat-distribution languages added to the topk-stage candidate universe "
            "by the category == 'flat_magnet' criterion are excluded from this "
            "variant's re-examined set."
        ),
        # No fixed expectation on how many languages fall under N < HEAD_N (roughly
        # 1,080; not asserted to an exact count the way flat4_tau5's four are).
        "expected_reexam_langs": None,
        "threshold": SHARED_TAU_V1,
        "calibrate": None,
        "tau_csv": None,
        "replacement_min_n": HEAD_N,
        "replacement_min_n_desc": (
            f"HEAD_N ({HEAD_N:,} training lines), lowered from the promoted "
            f"configuration's own bar of RES_CAP ({RES_CAP:,} training lines)."
        ),
        # Optional per-candidate acceptance function, signature (candidate_lang_idx,
        # N) -> bool. None means the default: N[candidate_lang_idx] >=
        # replacement_min_n. Later variants (direction 3) can override this with a
        # custom condition without changing the apply code below.
        "accept": None,
    },
    "flat4_tau5": {
        "experiment": 48,
        "description": (
            "Direction 2 (Experiment 48): re-examine every line whose promoted-"
            "configuration prediction (pred_floor21_gate.npy) is one of the four "
            "flat_magnet-category languages with at least HEAD_N training lines "
            "(sco_Latn, bjn_Latn, arg_Latn, vls_Latn), using a threshold calibrated "
            f"separately for each of the four (the fixed {MARGIN_Q}th percentile, "
            "MARGIN_Q, of that language's own training-line margins under the "
            "floor-21 matrix, not the promoted configuration's size-adaptive "
            f"formula) and the promoted configuration's own replacement-candidate "
            f"minimum, RES_CAP ({RES_CAP:,} training lines, unchanged from "
            "direction 1's lowered HEAD_N bar)."
        ),
        "base_pred": "pred_floor21_gate.npy",
        # Re-examined set: the flat_magnet-category languages with N >= HEAD_N
        # (exactly 4 by construction of the category; see EXPERIMENTS_RESULTS.md's
        # Experiment 48 pre-registration). N and zH from PRF_CSV/DIAG_CSV order;
        # category is diag.category.values, passed in by _run_apply.
        "reexamined_set": lambda N, zH, category: (
            (category == "flat_magnet") & (N >= HEAD_N)),
        "reexamined_set_desc": (
            "the four flat_magnet-category languages (outputs/diagnostic/"
            f"lang_diagnostic.csv) with N >= HEAD_N ({HEAD_N:,} training lines): "
            "sco_Latn, bjn_Latn, arg_Latn, vls_Latn."
        ),
        "expected_reexam_langs": 4,
        # threshold=None + calibrate=callable selects the per-language calibrated
        # path in _run_apply, instead of one fixed scalar nats value.
        "threshold": None,
        "calibrate": _calibrate_flat4_tau5,
        "tau_csv": _out(TAU_FLAT4_CSV_NAME),
        "replacement_min_n": RES_CAP,
        "replacement_min_n_desc": (
            f"RES_CAP ({RES_CAP:,} training lines), the promoted configuration's "
            "own replacement-candidate bar (unchanged, unlike direction 1's "
            "lowered bar)."
        ),
        "accept": None,
    },
    "flat4_prox21": {
        "experiment": 49,
        "description": (
            "Direction 3 (Experiment 49): a full reimplementation of the promoted "
            "configuration's two re-examination steps (step 1: languages with "
            f"N < HEAD_N ({HEAD_N:,} training lines), thresholds from "
            f"{_out(TAU_FLOOR21_GATE_CSV_NAME)}; step 2: the four flat_magnet-category "
            f"languages with N >= HEAD_N, thresholds from {_out(TAU_FLAT4_CSV_NAME)}), both "
            "re-run fresh from pred_floor21.npy (not chained through "
            "pred_floor21_gate.npy) so the added condition below applies to step "
            "1's moves too. Added condition: a replacement candidate must also "
            f"have a saved score within D3_PROX ({D3_PROX} nats) of the saved "
            "top-1 score, in addition to the existing N >= RES_CAP "
            f"({RES_CAP:,} training lines) bar; a candidate failing either check "
            "is skipped and the walk continues to the next of the saved "
            f"top-{TOPK_MARGIN} candidates; a line with no acceptable candidate "
            "keeps its pre-move (pred_floor21.npy) prediction. This variant does "
            "not go through the generic single-threshold apply logic below "
            "(_run_apply): it layers two different tau sources and a shared "
            "score-proximity acceptance condition the generic path does not "
            "support, so it has its own dedicated apply function, "
            "_run_apply_flat4_prox21, dispatched from _run_apply by variant name. "
            "The fields below are still declared, and still pass the module-level "
            "validation loop, for documentation and consistency with the other "
            "variants; 'reexamined_set' and 'accept' are not called by "
            "_run_apply_flat4_prox21 (see its docstring for what is actually run "
            "in their place)."
        ),
        "base_pred": "pred_floor21.npy",
        # Documents the union of both steps' re-examined languages (the same
        # union the topk stage's expanded_mask uses); _run_apply_flat4_prox21
        # recomputes step 1's and step 2's masks separately (they need separate
        # tau sources), not via this lambda.
        "reexamined_set": lambda N, zH, category: (
            (N < HEAD_N) | ((category == "flat_magnet") & (N >= HEAD_N))),
        "reexamined_set_desc": (
            f"the union of step 1 (languages with N < HEAD_N, {HEAD_N:,} training "
            "lines) and step 2 (the four flat_magnet-category languages with "
            "N >= HEAD_N: sco_Latn, bjn_Latn, arg_Latn, vls_Latn); the same union "
            "the topk stage's expanded label set uses."
        ),
        # Not used for a generic exact-count check (_run_apply_flat4_prox21 checks
        # step 2's count is exactly 4 directly); left None per the "disables the
        # exact-language-count check" convention.
        "expected_reexam_langs": None,
        "threshold": None,
        "calibrate": None,
        # Third tau_csv pattern (see the validation loop below): two pre-computed
        # CSVs, read directly by _load_tau_csv, no fresh calibration in this
        # module.
        "tau_csv": (_out(TAU_FLOOR21_GATE_CSV_NAME), _out(TAU_FLAT4_CSV_NAME)),
        "replacement_min_n": RES_CAP,
        "replacement_min_n_desc": (
            f"RES_CAP ({RES_CAP:,} training lines), the promoted configuration's "
            "own replacement-candidate bar, unchanged, plus the added D3_PROX "
            f"({D3_PROX} nats) score-proximity condition described above."
        ),
        # The N >= replacement_min_n check is still applied (inside
        # _run_apply_flat4_prox21's own walk), but the added score-proximity
        # condition cannot be expressed as an (candidate_lang_idx, N) -> bool
        # callable, so this field is not used for flat4_prox21; left None per
        # convention.
        "accept": None,
    },
}
for _vname, _ventry in VARIANTS.items():
    for _key, _hint in (
        ("accept", "None selects the default N >= replacement_min_n rule"),
        ("base_pred", "the output-root prediction file this variant's output is a "
                      "copy-and-selectively-modify of"),
        ("expected_reexam_langs", "None disables the exact-language-count check"),
        ("calibrate", "None when 'threshold' is a fixed scalar nats value; a "
                      "(langs, N, reexam_idx) -> {lang_idx: tau} callable when "
                      "'threshold' is None"),
        ("tau_csv", "None when 'threshold' is a fixed scalar (no calibration CSV "
                    "is written)"),
        ("replacement_min_n_desc", "plain-language description of "
                                   "replacement_min_n for the build markdown"),
    ):
        if _key not in _ventry:
            raise RuntimeError(f"VARIANTS[{_vname!r}] must declare {_key!r} "
                               f"explicitly ({_hint})")
    # Three mutually exclusive threshold/calibrate/tau_csv patterns:
    #  1. threshold is a fixed scalar nats value; calibrate and tau_csv both None
    #     (e.g. shared9_bar18k).
    #  2. threshold is None and calibrate is a (langs, N, reexam_idx) -> {lang_idx:
    #     tau} callable that calibrates and writes one tau CSV inside the apply
    #     stage; tau_csv names that single CSV path (e.g. flat4_tau5).
    #  3. threshold and calibrate are both None; tau_csv is a 2-tuple/list of CSV
    #     paths read directly, pre-computed, with no fresh calibration in this
    #     module (flat4_prox21, Experiment 49: it reads two existing tau CSVs from
    #     prior variants' builds).
    _threshold = _ventry["threshold"]
    _calibrate = _ventry["calibrate"]
    _tau_csv = _ventry["tau_csv"]
    if _threshold is not None:
        if _calibrate is not None or _tau_csv is not None:
            raise RuntimeError(f"VARIANTS[{_vname!r}]: 'threshold' is a fixed "
                               "scalar; 'calibrate' and 'tau_csv' must both be "
                               "None")
    elif _calibrate is not None:
        if not isinstance(_tau_csv, str):
            raise RuntimeError(f"VARIANTS[{_vname!r}]: 'calibrate' is set; "
                               "'tau_csv' must be a single CSV path (str)")
    elif isinstance(_tau_csv, (tuple, list)):
        if len(_tau_csv) != 2 or not all(isinstance(p, str) for p in _tau_csv):
            raise RuntimeError(f"VARIANTS[{_vname!r}]: the pre-computed-CSV "
                               "pattern requires 'tau_csv' to be a 2-tuple/list of "
                               "CSV path strings")
    else:
        raise RuntimeError(f"VARIANTS[{_vname!r}] must set exactly one of: "
                           "'threshold' (fixed scalar nats value), 'calibrate' "
                           "(per-language callable, with 'tau_csv' a single CSV "
                           "path), or leave both None with 'tau_csv' a 2-tuple of "
                           "pre-computed CSV paths")
del _vname, _ventry, _key, _hint, _threshold, _calibrate, _tau_csv


def _read_wanted_lines(path: str, wanted: set) -> tuple[dict, int]:
    """Chunked binary read of `path`. Returns ({line_index: raw_line_bytes} for
    every line index in `wanted`, total number of lines read). Reads the file in
    READ_BLOCK_BYTES blocks and splits on newlines in bulk; only wanted lines are
    kept in memory. See the module docstring for why this replaces a per-line
    readline() loop."""
    found = {}
    idx = 0
    leftover = b""
    with open(path, "rb") as f:
        while True:
            block = f.read(READ_BLOCK_BYTES)
            if not block:
                break
            data = leftover + block
            parts = data.split(b"\n")
            leftover = parts.pop()
            for raw in parts:
                if idx in wanted:
                    found[idx] = raw
                idx += 1
    if leftover:
        if idx in wanted:
            found[idx] = leftover
        idx += 1
    return found, idx


def _load_topk_arrays():
    """Reads the three stage-"topk" output arrays plus their fingerprint. Aborts if
    any of the four files is missing, or if the loaded arrays' shapes disagree with
    what the fingerprint itself recorded."""
    lines_path = _scratch("gate_topk_lines.npy")
    ids_path = _scratch("gate_topk_ids.npy")
    scores_path = _scratch("gate_topk_scores.npy")
    fp_path = _scratch("gate_topk_fingerprint.json")
    missing = [p for p in (lines_path, ids_path, scores_path, fp_path)
               if not os.path.exists(p)]
    if missing:
        raise RuntimeError(f"gate_topk output(s) missing: {missing}; run "
                           "run('topk') first")
    with open(fp_path) as f:
        fp = json.load(f)
    lines = np.load(lines_path)
    ids = np.load(ids_path)
    scores = np.load(scores_path)
    if lines.ndim != 1 or lines.dtype != np.int64:
        raise RuntimeError(f"{lines_path} has unexpected shape/dtype "
                           f"{lines.shape}/{lines.dtype}")
    n_affected = lines.shape[0]
    if n_affected != fp["n_affected"]:
        raise RuntimeError(f"{lines_path} has {n_affected} lines but {fp_path} "
                           f"records n_affected={fp['n_affected']}")
    expect_shape = (n_affected, TOPK_MARGIN)
    if ids.shape != expect_shape or scores.shape != expect_shape:
        raise RuntimeError(f"gate_topk_ids/scores shape mismatch: ids {ids.shape}, "
                           f"scores {scores.shape}, expected {expect_shape}")
    return lines, ids, scores, fp


def _run_topk() -> str:
    """Stage "topk": scores the top-TOPK_MARGIN floor-21 candidates for every kept
    line whose floor21 prediction falls in the expanded label set, and saves them
    to the output root. Idempotent: if gate_topk_lines/ids/scores.npy already exist and
    match gate_topk_fingerprint.json, prints and returns without rescoring; if a
    fingerprint exists but does not match, aborts naming what changed."""
    prf = pd.read_csv(PRF_CSV)
    langs = prf.lang.tolist()
    N = prf.N.values
    n_lang = len(langs)

    diag = pd.read_csv(DIAG_CSV)
    if diag.lang.tolist() != langs:
        raise RuntimeError(f"{DIAG_CSV} language order does not match {PRF_CSV}")

    low_n_mask = N < HEAD_N
    flat_mask = (diag.category == "flat_magnet").values
    expanded_mask = low_n_mask | flat_mask
    expanded_idx = np.where(expanded_mask)[0]

    matrix, sha_w, sha_w21 = _build_verified_floor21_matrix(langs)

    y = np.asarray(np.lib.format.open_memmap(
        _scratch("y_true.npy"), mode="r"))
    if y.shape != (TOTAL_LINES,):
        raise RuntimeError(f"y_true.npy shape {y.shape} != ({TOTAL_LINES},)")
    pf21 = np.asarray(np.lib.format.open_memmap(
        _scratch("pred_floor21.npy"), mode="r"))
    if pf21.shape != (TOTAL_LINES,):
        raise RuntimeError(f"pred_floor21.npy shape {pf21.shape} != ({TOTAL_LINES},)")

    affected = np.where((y >= 0) & np.isin(pf21, expanded_idx))[0]
    n_affected = len(affected)
    if not (2_000_000 <= n_affected <= 2_600_000):
        raise RuntimeError(
            f"{n_affected:,} lines carry a floor21 prediction in the expanded "
            f"label set (N < {HEAD_N:,} or category == 'flat_magnet'); expected "
            "between 2,000,000 and 2,600,000 (2,236,864 measured in the reviewed "
            f"run). {PRF_CSV} or {DIAG_CSV} may not match what the reviewed run "
            "used, or expanded_mask was built from the wrong criterion.")
    print(f"{n_affected:,} lines carry a floor21 prediction in the expanded label "
          f"set (N < {HEAD_N:,} or category == 'flat_magnet').")

    langs_sha = hashlib.sha256("|".join(langs).encode()).hexdigest()
    fp = {
        "sha256_base_W": sha_w,
        "sha256_w21": sha_w21,
        "sha256_expanded_mask": hashlib.sha256(expanded_mask.tobytes()).hexdigest(),
        "langs_sha256": langs_sha,
        "head_n": HEAD_N,
        "zh_magnet": ZH_MAGNET,
        "topk_margin": TOPK_MARGIN,
        "n_lang": n_lang,
        "n_low_n": int(low_n_mask.sum()),
        "n_flat": int(flat_mask.sum()),
        "n_expanded": int(expanded_mask.sum()),
        "n_affected": n_affected,
        "total_lines": TOTAL_LINES,
    }

    lines_path = _scratch("gate_topk_lines.npy")
    ids_path = _scratch("gate_topk_ids.npy")
    scores_path = _scratch("gate_topk_scores.npy")
    fp_path = _scratch("gate_topk_fingerprint.json")
    arrays_exist = all(os.path.exists(p) for p in (lines_path, ids_path, scores_path))

    if os.path.exists(fp_path):
        with open(fp_path) as f:
            prev = json.load(f)
        if prev == fp:
            if arrays_exist:
                print(f"existing gate_topk_* arrays in {_ctx().scratch_dir} match the "
                      "current fingerprint; skipping rescoring.")
                return lines_path
            raise RuntimeError(f"{fp_path} matches the current fingerprint but one "
                               "or more of gate_topk_lines.npy / gate_topk_ids.npy / "
                               f"gate_topk_scores.npy is missing from {_ctx().scratch_dir}; "
                               "investigate before rerunning")
        bad = sorted(k for k in fp if prev.get(k) != fp[k])
        raise RuntimeError(f"gate_topk scratch state mismatch ({bad}); clear the "
                           f"gate_topk_* files in {_ctx().scratch_dir} or investigate what "
                           "changed")
    elif arrays_exist:
        raise RuntimeError(f"gate_topk_lines/ids/scores.npy exist in {_ctx().scratch_dir} "
                           f"but {fp_path} is missing; they cannot be verified as "
                           "produced under the current inputs; remove the stale "
                           "arrays to force a rescore or restore the fingerprint")

    # Model load and weight-set caching complete before the wanted-lines read so
    # `matrix`'s memory is freed (del matrix) before raw_lines' dict allocation
    # begins, instead of the two large allocations overlapping in peak memory.
    model = _load_unilid_model(_ctx().model_path)
    print("Caching floor-21 weights...", flush=True)
    model.model.set_weight_sets(matrix.tolist())
    del matrix

    want = set(affected.tolist())
    raw_lines, n_read = _read_wanted_lines(TEST_FILE, want)
    del want
    if n_read != TOTAL_LINES:
        raise RuntimeError(f"read {n_read} lines from {TEST_FILE}, expected "
                           f"{TOTAL_LINES}")
    if len(raw_lines) != n_affected:
        raise RuntimeError(f"collected {len(raw_lines)} lines of text for "
                           f"{n_affected} affected line indices")

    # ids_arr/scores_arr are pre-filled with sentinels (-1 / -inf), not left
    # uninitialized: a line whose saved candidate list is short (see below) keeps
    # -1 in its unfilled id slots and -inf in its unfilled score slots.
    ids_arr = np.full((n_affected, TOPK_MARGIN), -1, dtype=np.int16)
    scores_arr = np.full((n_affected, TOPK_MARGIN), -np.inf, dtype=np.float32)
    n_short_cands = 0
    n_inf_margin = 0
    for lo in range(0, n_affected, SCORE_BATCH_MAX):
        hi = min(lo + SCORE_BATCH_MAX, n_affected)
        batch_lines = affected[lo:hi]
        # .pop() consumes raw_lines as each line is used, so memory for a finished
        # batch's raw text is freed rather than held for the rest of the run.
        texts = [_parse_line(raw_lines.pop(int(i)).decode("utf-8"))[1]
                 for i in batch_lines]
        drop = []
        pos, topk = _topk_batch(model, texts, drop)
        if drop:
            raise RuntimeError(f"{len(drop)} affected lines empty after preprocess "
                               f"in batch [{lo}:{hi})")
        for k, cands in enumerate(topk):
            # Recorded margin-gate convention (analysis/margin_diagnostic.py's
            # _gap(), used operationally in analysis/full_test_margin.py): a line
            # with fewer than 2 candidates has no top1-minus-top2 margin to compute,
            # so it is treated as having an infinite margin and is never moved by
            # any variant's threshold gate, not raised on.
            if len(cands) < TOPK_MARGIN:
                n_short_cands += 1
                if len(cands) < 2:
                    n_inf_margin += 1
            row = lo + pos[k]
            for j, (cand_idx, cand_score) in enumerate(cands):
                ids_arr[row, j] = np.int16(cand_idx)
                scores_arr[row, j] = np.float32(cand_score)
        print(f"scored [{lo:,}:{hi:,}) of {n_affected:,} affected lines", flush=True)
    if raw_lines:
        raise RuntimeError(f"{len(raw_lines)} raw lines were never consumed by the "
                           "batch loop; the affected-line partition is broken")
    if n_short_cands:
        print(f"{n_short_cands:,} of {n_affected:,} affected lines returned fewer "
              f"than {TOPK_MARGIN} saved candidates; {n_inf_margin:,} of those "
              "returned fewer than 2 and are treated as having infinite margin "
              "(never moved), following the recorded margin-gate convention "
              "(analysis/margin_diagnostic.py's _gap()).")

    pf21_aff = pf21[affected]
    low_n_row_mask = low_n_mask[pf21_aff]
    flat_added_row_mask = flat_mask[pf21_aff] & ~low_n_mask[pf21_aff]
    n_low_n_rows = int(low_n_row_mask.sum())
    n_flat_added_rows = int(flat_added_row_mask.sum())
    if n_low_n_rows == 0:
        raise RuntimeError("no affected line's floor21 prediction falls in the "
                           "N < HEAD_N subset; the expanded label set is broken")

    agree = float(np.mean(ids_arr[:, 0] == pf21_aff))
    agree_low_n = float(np.mean(ids_arr[low_n_row_mask, 0]
                                == pf21_aff[low_n_row_mask]))
    agree_flat_added = (float(np.mean(ids_arr[flat_added_row_mask, 0]
                                      == pf21_aff[flat_added_row_mask]))
                        if n_flat_added_rows else float("nan"))
    print(f"top-1 agreement with pred_floor21 on affected lines: {agree:.4f} "
          f"overall ({n_affected:,} lines); {agree_low_n:.4f} on the N < "
          f"{HEAD_N:,} subset ({n_low_n_rows:,} lines); {agree_flat_added:.4f} on "
          f"the added flat_magnet-only subset ({n_flat_added_rows:,} lines).")
    if agree < TOP1_AGREE_MIN:
        raise RuntimeError(f"top-1 agreement with pred_floor21 {agree:.4f} < "
                           f"{TOP1_AGREE_MIN} overall")
    if agree_low_n < TOP1_AGREE_MIN:
        raise RuntimeError(f"top-1 agreement with pred_floor21 {agree_low_n:.4f} < "
                           f"{TOP1_AGREE_MIN} on the N < {HEAD_N:,} subset")

    fp["top1_agree_overall"] = agree
    fp["top1_agree_low_n"] = agree_low_n
    fp["top1_agree_flat_added"] = agree_flat_added
    fp["n_low_n_rows"] = n_low_n_rows
    fp["n_flat_added_rows"] = n_flat_added_rows
    fp["n_short_cands"] = n_short_cands
    fp["n_inf_margin"] = n_inf_margin

    np.save(lines_path, affected.astype(np.int64))
    np.save(ids_path, ids_arr)
    np.save(scores_path, scores_arr)
    with open(fp_path + ".tmp", "w") as f:
        json.dump(fp, f)
    os.replace(fp_path + ".tmp", fp_path)
    print(f"Wrote {lines_path}, {ids_path}, {scores_path}, {fp_path}")
    return lines_path


def _walk_replacement(prox_limit: float | None, replacement_min_n: int,
                      ids_row: np.ndarray, scores_row: np.ndarray,
                      N: np.ndarray) -> tuple[str, int | None]:
    """Shared replacement-candidate walk for _run_apply_flat4_prox21's two steps.
    Walks `ids_row`/`scores_row` (one gate_topk row: index 0 is the saved top-1,
    indices 1..TOPK_MARGIN-1 are the saved rank-2..TOPK_MARGIN candidates, in saved
    rank order) starting at index 1. A candidate at index j is accepted if
    N[ids_row[j]] >= replacement_min_n and, when `prox_limit` is not None,
    scores_row[0] - scores_row[j] <= prox_limit (an unfilled slot, ids_row[j] ==
    -1, is skipped). The first accepted candidate is taken. `replacement_min_n`
    is fed in by the caller from the variant entry's own "replacement_min_n"
    field (RES_CAP for every current variant that reaches this function; see
    VARIANTS), not hardcoded here.

    Returns (outcome, candidate_or_None):
    - ("moved", cid): the first candidate meeting every active condition.
    - ("blocked_by_proximity", None): at least one candidate met
      N >= replacement_min_n, but none of those also met the D3_PROX bound. Only
      reachable when prox_limit is not None, since with prox_limit=None the first
      N >= replacement_min_n candidate is always accepted immediately.
    - ("no_cand", None): no candidate met N >= replacement_min_n at all,
      regardless of `prox_limit`.

    With prox_limit=None this reproduces, candidate-for-candidate, the replacement
    rule analysis/solo_gates.py's run("floor21") and this module's flat4_tau5
    variant both use (first saved rank-2..TOPK_MARGIN candidate with
    N >= replacement_min_n); this identity is what _run_apply_flat4_prox21's
    mandatory self-check verifies before the D3_PROX-conditioned build is
    trusted."""
    top1_score = scores_row[0]
    saw_min_n = False
    for j in range(1, TOPK_MARGIN):
        cid = int(ids_row[j])
        if cid < 0:
            continue  # unfilled candidate slot (topk stage short list); skip
        if N[cid] < replacement_min_n:
            continue
        saw_min_n = True
        if prox_limit is not None and (top1_score - scores_row[j]) > prox_limit:
            continue
        return "moved", cid
    if saw_min_n:
        return "blocked_by_proximity", None
    return "no_cand", None


def _flat4_prox21_two_step_pred(prox_limit: float | None, replacement_min_n: int,
                                below1: np.ndarray, below2: np.ndarray,
                                lines: np.ndarray, ids: np.ndarray,
                                scores: np.ndarray, N: np.ndarray, y: np.ndarray,
                                base_pred_arr: np.ndarray) -> dict:
    """Runs _walk_replacement over every gate_topk row flagged by `below1` (step 1:
    N < HEAD_N languages) and `below2` (step 2: the four flat_magnet, N >= HEAD_N
    languages) separately, each starting from a fresh copy of `base_pred_arr`
    (pred_floor21.npy), then merges the two sets of line-level diffs into one
    TOTAL_LINES-length prediction array. below1 and below2 index disjoint sets of
    gate_topk rows (step 1's and step 2's re-examined languages are disjoint by
    construction; see the module docstring and _run_apply_flat4_prox21), so the
    merge cannot conflict; this is asserted, not assumed. `replacement_min_n` is
    passed straight through to every _walk_replacement call (the caller reads it
    from the variant entry's own "replacement_min_n" field).

    Returns a dict with keys "pred" (the merged TOTAL_LINES array) and, for each of
    "step1", "step2", "combined": "n_moved", "n_to_true", "n_no_cand",
    "n_blocked_prox"."""
    def _run_step(below_mask):
        pred = np.array(base_pred_arr, dtype=np.int16)
        n_moved = n_to_true = n_no_cand = n_blocked = 0
        for r in np.where(below_mask)[0].tolist():
            outcome, cid = _walk_replacement(prox_limit, replacement_min_n,
                                             ids[r], scores[r], N)
            if outcome == "moved":
                line = int(lines[r])
                pred[line] = np.int16(cid)
                n_moved += 1
                if cid == int(y[line]):
                    n_to_true += 1
            elif outcome == "blocked_by_proximity":
                n_blocked += 1
            else:
                n_no_cand += 1
        if prox_limit is None and n_blocked:
            raise RuntimeError(f"{n_blocked} lines reported blocked_by_proximity "
                               "with prox_limit=None; _walk_replacement should "
                               "never return that outcome when the proximity "
                               "condition is disabled")
        return pred, {"n_moved": n_moved, "n_to_true": n_to_true,
                     "n_no_cand": n_no_cand, "n_blocked_prox": n_blocked}

    pred1, stats1 = _run_step(below1)
    pred2, stats2 = _run_step(below2)
    diff1 = pred1 != base_pred_arr
    diff2 = pred2 != base_pred_arr
    if (diff1 & diff2).any():
        raise RuntimeError("step 1 and step 2 modified the same line(s); their "
                           "re-examined-set languages are supposed to be disjoint")
    merged = np.array(base_pred_arr, dtype=np.int16)
    merged[diff1] = pred1[diff1]
    merged[diff2] = pred2[diff2]
    combined = {k: stats1[k] + stats2[k] for k in stats1}
    return {"pred": merged, "step1": stats1, "step2": stats2, "combined": combined}


def _run_apply_flat4_prox21() -> str:
    """Dedicated apply path for the flat4_prox21 variant (Experiment 49, direction
    3), dispatched from _run_apply by variant name because this variant is a full
    reimplementation of the promoted configuration's two re-examination steps, not
    an instance of the generic single-threshold logic the rest of _run_apply runs
    for every other VARIANTS entry: it needs two different tau sources (step 1
    from tau_floor21_gate.csv, step 2 from tau_flat4.csv, both under the
    output root) and a shared
    score-proximity acceptance condition (D3_PROX) that the generic path's
    (candidate_lang_idx, N) -> bool 'accept' callable cannot express.

    Both steps are re-run fresh from pred_floor21.npy (never from
    pred_floor21_gate.npy), so the added D3_PROX condition governs step 1's moves
    as well as step 2's, per the pre-registration.

    Mandatory self-check before the conditioned build is trusted: the identical
    two-step walk with the proximity condition disabled (prox_limit=None) must
    reproduce pred_gate_flat4_tau5.npy in the output root bit-identically
    over the full
    TOTAL_LINES-length array (np.array_equal); this aborts, not warns, on any
    mismatch. Only after that passes is the D3_PROX=21.0 conditioned build run and
    saved."""
    entry = VARIANTS["flat4_prox21"]

    lines, ids, scores, fp = _load_topk_arrays()
    if fp["head_n"] != HEAD_N:
        raise RuntimeError(f"gate_topk_fingerprint.json head_n={fp['head_n']} != "
                           f"the current HEAD_N={HEAD_N}; the topk-stage output was "
                           "built under a different HEAD_N than this apply run")

    prf = pd.read_csv(PRF_CSV)
    langs = prf.lang.tolist()
    N = prf.N.values
    if len(langs) != fp["n_lang"]:
        raise RuntimeError(f"{PRF_CSV} has {len(langs)} languages, "
                           f"gate_topk_fingerprint.json records {fp['n_lang']}")
    langs_sha = hashlib.sha256("|".join(langs).encode()).hexdigest()
    if langs_sha != fp["langs_sha256"]:
        raise RuntimeError(f"{PRF_CSV} language list does not match "
                           "gate_topk_fingerprint.json's langs_sha256; the language "
                           "table changed since stage 'topk' ran")

    diag = pd.read_csv(DIAG_CSV)
    if diag.lang.tolist() != langs:
        raise RuntimeError(f"{DIAG_CSV} language order does not match {PRF_CSV}")
    category = diag.category.values

    low_n_mask = N < HEAD_N
    flat_mask = (category == "flat_magnet")
    expanded_mask = low_n_mask | flat_mask
    expanded_sha = hashlib.sha256(expanded_mask.tobytes()).hexdigest()
    if expanded_sha != fp["sha256_expanded_mask"]:
        raise RuntimeError(f"expanded mask recomputed from {PRF_CSV}/{DIAG_CSV} "
                           f"(sha256 {expanded_sha}) does not match "
                           "gate_topk_fingerprint.json's sha256_expanded_mask "
                           f"({fp['sha256_expanded_mask']}); one of those CSVs "
                           "changed since stage 'topk' ran")

    step1_mask = low_n_mask
    step2_mask = flat_mask & (N >= HEAD_N)
    if (step1_mask & step2_mask).any():
        raise RuntimeError("step 1 and step 2 re-examined-set language masks "
                           "overlap; expected disjoint (step 1 is N < HEAD_N, step "
                           "2 is N >= HEAD_N)")
    n_step2_langs = int(step2_mask.sum())
    if n_step2_langs != 4:
        raise RuntimeError(f"step 2's re-examined set has {n_step2_langs} "
                           "language(s), expected exactly 4 (sco_Latn, bjn_Latn, "
                           "arg_Latn, vls_Latn)")

    y = np.asarray(np.lib.format.open_memmap(
        _scratch("y_true.npy"), mode="r"))
    if y.shape != (TOTAL_LINES,):
        raise RuntimeError(f"y_true.npy shape {y.shape} != ({TOTAL_LINES},)")
    base_pred_path = _scratch(entry["base_pred"])
    base_pred_arr = np.asarray(np.lib.format.open_memmap(base_pred_path, mode="r"))
    if base_pred_arr.shape != (TOTAL_LINES,):
        raise RuntimeError(f"{base_pred_path} shape {base_pred_arr.shape} != "
                           f"({TOTAL_LINES},)")

    n_affected = len(lines)
    base_line_pred = base_pred_arr[lines]
    if (base_line_pred < 0).any():
        raise RuntimeError(f"{base_pred_path} has a negative prediction at one or "
                           "more gate_topk_lines.npy indices; numpy fancy indexing "
                           "with a negative int16 index silently wraps to the end "
                           "of the language array instead of failing")

    in_set1 = step1_mask[base_line_pred]
    in_set2 = step2_mask[base_line_pred]
    if (in_set1 & in_set2).any():
        raise RuntimeError("internal count mismatch: a line falls in both step "
                           "1's and step 2's in-set masks; this indicates a bug, "
                           "since the two steps' re-examined-set languages are "
                           "disjoint by construction")
    n_in_set1 = int(in_set1.sum())
    n_in_set2 = int(in_set2.sum())
    if n_in_set1 + n_in_set2 != n_affected:
        raise RuntimeError(
            f"n_in_set1({n_in_set1:,}) + n_in_set2({n_in_set2:,}) != "
            f"n_affected({n_affected:,}); step 1's and step 2's in-set masks are "
            "expected to partition the full banked expanded set (every affected "
            "line's base prediction falls in exactly one step's re-examined "
            "languages)")

    step1_lang_set = {langs[i] for i in np.where(step1_mask)[0]}
    step2_lang_set = {langs[i] for i in np.where(step2_mask)[0]}
    tau1, sha_tau1 = _load_tau_csv(_out(TAU_FLOOR21_GATE_CSV_NAME), langs, step1_lang_set)
    tau2, sha_tau2 = _load_tau_csv(_out(TAU_FLAT4_CSV_NAME), langs, step2_lang_set)

    top1 = ids[:, 0]
    agree_mask = top1 == base_line_pred
    gap = scores[:, 0] - scores[:, 1]

    def _step_below(in_set, tau_arr):
        disagree = in_set & ~agree_mask
        gated = in_set & agree_mask
        below = gated & (gap < tau_arr[base_line_pred])
        return disagree, gated, below

    disagree1, gated1, below1 = _step_below(in_set1, tau1)
    disagree2, gated2, below2 = _step_below(in_set2, tau2)
    n_disagree1, n_gated1, n_below1 = (int(disagree1.sum()), int(gated1.sum()),
                                       int(below1.sum()))
    n_disagree2, n_gated2, n_below2 = (int(disagree2.sum()), int(gated2.sum()),
                                       int(below2.sum()))

    # --- mandatory self-check: reproduce pred_gate_flat4_tau5.npy with the ---
    # --- proximity condition disabled, before trusting the conditioned build ---
    ref_path = _scratch("pred_gate_flat4_tau5.npy")
    if not os.path.exists(ref_path):
        raise RuntimeError(f"self-check reference missing: {ref_path}; run "
                           "Experiment 48's flat4_tau5 apply stage first")
    reference = np.asarray(np.lib.format.open_memmap(ref_path, mode="r"))
    if reference.shape != (TOTAL_LINES,):
        raise RuntimeError(f"{ref_path} shape {reference.shape} != "
                           f"({TOTAL_LINES},)")

    none_result = _flat4_prox21_two_step_pred(
        None, entry["replacement_min_n"], below1, below2, lines, ids, scores, N, y,
        base_pred_arr)
    self_check_pred = none_result["pred"]
    if not np.array_equal(self_check_pred, reference):
        n_mismatch = int((self_check_pred != reference).sum())
        mismatch_idx = np.where(self_check_pred != reference)[0][:10].tolist()
        raise RuntimeError(
            "flat4_prox21 self-check FAILED: the two-step walk with the "
            f"proximity condition disabled differs from {ref_path} on "
            f"{n_mismatch:,} of {TOTAL_LINES:,} lines (first differing indices: "
            f"{mismatch_idx}); the conditioned (D3_PROX={D3_PROX}) build is not "
            "trusted until this reproduces bit-identically. Aborting without "
            "building the conditioned output.")
    print(f"flat4_prox21 self-check passed: the D=None two-step walk is "
          f"bit-identical to {ref_path} on all {TOTAL_LINES:,} lines.")

    # --- conditioned (D3_PROX) build, only reached after the self-check passes ---
    result = _flat4_prox21_two_step_pred(
        D3_PROX, entry["replacement_min_n"], below1, below2, lines, ids, scores, N,
        y, base_pred_arr)
    pred = result["pred"]

    # Partition assertion, both the self-check pass (prox_limit=None) and the
    # conditioned (D3_PROX) build: every re-examined line (n_below, the count of
    # gate_topk rows that fell below tau and were walked) must land in exactly one
    # of moved, kept-no-candidate, or blocked-by-proximity. A mismatch means the
    # walk's outcome bookkeeping in _flat4_prox21_two_step_pred/_walk_replacement
    # dropped or double-counted a row.
    for check_name, res in (("self-check (prox_limit=None)", none_result),
                            ("conditioned (D3_PROX)", result)):
        for k, nb in (("step1", n_below1), ("step2", n_below2)):
            s = res[k]
            if nb != s["n_moved"] + s["n_no_cand"] + s["n_blocked_prox"]:
                raise RuntimeError(
                    f"{check_name} {k}: n_below={nb:,} does not equal "
                    f"n_moved({s['n_moved']:,}) + n_no_cand({s['n_no_cand']:,}) + "
                    f"n_blocked_prox({s['n_blocked_prox']:,}); the re-examined "
                    "rows for this step do not partition into moved, kept-no-"
                    "candidate, and blocked-by-proximity outcomes")

    n_diff = int((pred != base_pred_arr).sum())
    n_moved = result["combined"]["n_moved"]
    if n_diff != n_moved:
        raise RuntimeError(f"{n_diff:,} lines differ from {entry['base_pred']} but "
                           f"n_moved={n_moved:,}; the moved-set count does not "
                           f"match the actual diff against {entry['base_pred']}")

    out_pred = _scratch("pred_gate_flat4_prox21.npy")
    np.save(out_pred, pred)

    try:
        git_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, OSError) as e:
        raise RuntimeError(f"git rev-parse HEAD failed: {e}") from e

    meta = {
        "topk_fingerprint": fp,
        "variant": "flat4_prox21",
        "base_pred": entry["base_pred"],
        "d3_prox": D3_PROX,
        "replacement_min_n": RES_CAP,
        "tau_csv": {
            "step1_floor21_gate": {"path": _out(TAU_FLOOR21_GATE_CSV_NAME),
                                   "sha256": sha_tau1},
            "step2_flat4": {"path": _out(TAU_FLAT4_CSV_NAME), "sha256": sha_tau2},
        },
        "reexamined_set_desc": entry["reexamined_set_desc"],
        "self_check": {
            "reference": ref_path,
            "n_lines_compared": TOTAL_LINES,
            "bit_identical": True,
        },
        "counts": {
            "n_affected": n_affected,
            "step1": {"n_in_set": n_in_set1, "n_disagree": n_disagree1,
                     "n_gated": n_gated1, "n_gap_ok": n_gated1 - n_below1,
                     "n_below": n_below1, **result["step1"]},
            "step2": {"n_in_set": n_in_set2, "n_disagree": n_disagree2,
                     "n_gated": n_gated2, "n_gap_ok": n_gated2 - n_below2,
                     "n_below": n_below2, **result["step2"]},
            "combined": {
                "n_in_set": n_in_set1 + n_in_set2,
                "n_disagree": n_disagree1 + n_disagree2,
                "n_gated": n_gated1 + n_gated2,
                "n_gap_ok": (n_gated1 - n_below1) + (n_gated2 - n_below2),
                "n_below": n_below1 + n_below2,
                **result["combined"],
            },
        },
        "git_commit": git_commit,
    }
    out_meta = _scratch("pred_gate_flat4_prox21_meta.json")
    with open(out_meta + ".tmp", "w") as f:
        json.dump(meta, f)
    os.replace(out_meta + ".tmp", out_meta)

    c = meta["counts"]
    lines_out = [
        "# flat4_prox21 candidate build (Experiment 49)\n",
        entry["description"],
        "",
        f"- Reproduction check passed: the two-step walk, run with the D3_PROX "
        f"condition disabled, is bit-identical to {ref_path} across all "
        f"{TOTAL_LINES:,} lines (np.array_equal over the full array).",
        f"- Constants: HEAD_N = {HEAD_N:,} training lines (step 1's re-examined-"
        f"set boundary); RES_CAP = {RES_CAP:,} training lines (the replacement-"
        f"candidate minimum, both steps); TOPK_MARGIN = {TOPK_MARGIN} saved "
        f"candidates per affected line; D3_PROX = {D3_PROX} nats (the added "
        "score-proximity bound, derivation-part grid search, plateau 15 to 35). "
        f"Base predictions: {entry['base_pred']}.",
        f"- Tau sources: step 1 from {_out(TAU_FLOOR21_GATE_CSV_NAME)} (sha256 "
        f"{sha_tau1[:16]}...); step 2 from {_out(TAU_FLAT4_CSV_NAME)} (sha256 "
        f"{sha_tau2[:16]}...).",
        f"- Affected: {n_affected:,} lines carry a saved floor21-prediction "
        "candidate list (the topk-stage expanded label set).",
        "",
        "| step | in-set | disagreements | gated (in-set, agreeing) | gap-ok "
        "(kept) | re-examined (below tau) | moved | moved-to-true | "
        "kept-no-candidate | blocked-by-proximity |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for label, key in (("step 1 (N < HEAD_N)", "step1"),
                       ("step 2 (flat4, N >= HEAD_N)", "step2"),
                       ("combined", "combined")):
        s = c[key]
        lines_out.append(
            f"| {label} | {s['n_in_set']:,} | {s['n_disagree']:,} | "
            f"{s['n_gated']:,} | {s['n_gap_ok']:,} | {s['n_below']:,} | "
            f"{s['n_moved']:,} | {s['n_to_true']:,} | {s['n_no_cand']:,} | "
            f"{s['n_blocked_prox']:,} |")
    lines_out += [
        "",
        f"- Disagreements: lines whose saved top-1 candidate disagrees with "
        f"{entry['base_pred']} are left unchanged and counted, not re-examined "
        "(the recorded top-1-agreement convention).",
        f"- Blocked-by-proximity: re-examined lines where at least one candidate "
        f"ranked 2 to {TOPK_MARGIN} met the N >= RES_CAP bar, but none of those "
        f"also met the D3_PROX = {D3_PROX} nats bound, so the line keeps its "
        f"{entry['base_pred']} prediction (distinct from kept-no-candidate: no "
        "candidate met the N >= RES_CAP bar at all).",
        f"- Moved: {c['combined']['n_moved']:,} lines total move to a replacement "
        f"candidate; {c['combined']['n_to_true']:,} of those land on the true "
        "label recorded in y_true.npy.",
        f"- All lines outside the moved set are bit-identical to "
        f"{entry['base_pred']} ({n_diff:,} lines differ, verified equal to "
        f"n_moved={n_moved:,}). Output: {out_pred}; metadata: {out_meta}.",
    ]
    out_md = "outputs/tables/gate_flat4_prox21_build.md"
    os.makedirs(os.path.dirname(out_md), exist_ok=True)
    with open(out_md, "w") as f:
        f.write("\n".join(lines_out) + "\n")
    print("\n".join(lines_out))
    print(f"\nWrote {out_pred}, {out_meta}, and {out_md}")
    return out_pred


def _run_apply(variant: str) -> str:
    """Stage "apply": post-processes the saved stage-"topk" arrays into a full-pool
    prediction file for one named entry in VARIANTS. No scoring happens here.

    "flat4_prox21" (Experiment 49) is dispatched to its own dedicated function,
    _run_apply_flat4_prox21, instead of running the generic single-threshold logic
    below: see that function's docstring for why."""
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; must be one of "
                         f"{sorted(VARIANTS)}")
    if variant == "flat4_prox21":
        return _run_apply_flat4_prox21()
    entry = VARIANTS[variant]

    lines, ids, scores, fp = _load_topk_arrays()
    if fp["head_n"] != HEAD_N:
        raise RuntimeError(f"gate_topk_fingerprint.json head_n={fp['head_n']} != "
                           f"the current HEAD_N={HEAD_N}; the topk-stage output was "
                           "built under a different HEAD_N than this apply run")

    prf = pd.read_csv(PRF_CSV)
    langs = prf.lang.tolist()
    N = prf.N.values
    if len(langs) != fp["n_lang"]:
        raise RuntimeError(f"{PRF_CSV} has {len(langs)} languages, "
                           f"gate_topk_fingerprint.json records {fp['n_lang']}")
    langs_sha = hashlib.sha256("|".join(langs).encode()).hexdigest()
    if langs_sha != fp["langs_sha256"]:
        raise RuntimeError(f"{PRF_CSV} language list does not match "
                           "gate_topk_fingerprint.json's langs_sha256; the language "
                           "table changed since stage 'topk' ran")

    diag = pd.read_csv(DIAG_CSV)
    if diag.lang.tolist() != langs:
        raise RuntimeError(f"{DIAG_CSV} language order does not match {PRF_CSV}")
    zH = diag.zH.values
    category = diag.category.values

    low_n_mask = N < HEAD_N
    flat_mask = (diag.category == "flat_magnet").values
    expanded_mask = low_n_mask | flat_mask
    expanded_sha = hashlib.sha256(expanded_mask.tobytes()).hexdigest()
    if expanded_sha != fp["sha256_expanded_mask"]:
        raise RuntimeError(f"expanded mask recomputed from {PRF_CSV}/{DIAG_CSV} "
                           f"(sha256 {expanded_sha}) does not match "
                           "gate_topk_fingerprint.json's sha256_expanded_mask "
                           f"({fp['sha256_expanded_mask']}); one of those CSVs "
                           "changed since stage 'topk' ran")

    y = np.asarray(np.lib.format.open_memmap(
        _scratch("y_true.npy"), mode="r"))
    if y.shape != (TOTAL_LINES,):
        raise RuntimeError(f"y_true.npy shape {y.shape} != ({TOTAL_LINES},)")
    pf21 = np.asarray(np.lib.format.open_memmap(
        _scratch("pred_floor21.npy"), mode="r"))
    if pf21.shape != (TOTAL_LINES,):
        raise RuntimeError(f"pred_floor21.npy shape {pf21.shape} != ({TOTAL_LINES},)")

    n_affected = len(lines)
    pf21_line_pred = pf21[lines]
    if (pf21_line_pred < 0).any():
        raise RuntimeError("pred_floor21.npy has a negative prediction at one or "
                           "more gate_topk_lines.npy indices; numpy fancy indexing "
                           "with a negative int16 index silently wraps to the end "
                           "of the language array instead of failing")

    reexam_lang_mask = entry["reexamined_set"](N, zH, category)
    if (reexam_lang_mask & ~expanded_mask).any():
        raise RuntimeError(f"variant {variant!r}'s re-examined-set mask includes "
                           "language(s) outside the topk-stage expanded label set; "
                           "the topk stage never saved candidates for those lines")
    reexam_idx = np.where(reexam_lang_mask)[0]
    expected_reexam_langs = entry["expected_reexam_langs"]
    if (expected_reexam_langs is not None
            and len(reexam_idx) != expected_reexam_langs):
        raise RuntimeError(f"variant {variant!r}'s re-examined set has "
                           f"{len(reexam_idx)} language(s), expected "
                           f"{expected_reexam_langs}")
    if expected_reexam_langs is not None:
        print(f"variant {variant!r} re-examined-set languages "
              f"({len(reexam_idx)}): {[langs[int(i)] for i in reexam_idx]}")

    # base_pred: the array this variant's output is a copy-and-selectively-modify
    # of. It need not be pred_floor21.npy (flat4_tau5 uses pred_floor21_gate.npy),
    # but the topk-stage arrays (lines/ids/scores/pf21_line_pred, computed above)
    # are always anchored on pred_floor21.npy, so a variant with a different
    # base_pred must verify its re-examined-set languages are predicted identically
    # under both arrays before reusing pf21-derived masks against base_pred lines.
    base_pred_name = entry["base_pred"]
    base_pred_path = _scratch(base_pred_name)
    base_pred_arr = np.asarray(np.lib.format.open_memmap(base_pred_path, mode="r"))
    if base_pred_arr.shape != (TOTAL_LINES,):
        raise RuntimeError(f"{base_pred_path} shape {base_pred_arr.shape} != "
                           f"({TOTAL_LINES},)")
    if base_pred_name != "pred_floor21.npy":
        pf21_in = np.isin(pf21, reexam_idx)
        base_in = np.isin(base_pred_arr, reexam_idx)
        either = pf21_in | base_in
        if not np.array_equal(pf21[either], base_pred_arr[either]):
            raise RuntimeError(
                "pred_floor21 and the variant base disagree on lines predicted "
                "into the re-examined set; per-language thresholds would be "
                "misapplied")
        pf21_in_reexam = int(pf21_in.sum())
        base_in_reexam = int(base_in.sum())
        print(f"pred_floor21.npy and {base_pred_name} agree on {pf21_in_reexam:,} "
              "lines predicted into the re-examined-set languages (count equality "
              "verified).")

    in_set_mask = reexam_lang_mask[pf21_line_pred]
    n_in_set = int(in_set_mask.sum())

    top1 = ids[:, 0]
    agree_mask = top1 == pf21_line_pred
    disagree_mask = in_set_mask & ~agree_mask
    n_disagree = int(disagree_mask.sum())

    gated_mask = in_set_mask & agree_mask
    n_gated = int(gated_mask.sum())
    if n_in_set != n_disagree + n_gated:
        raise RuntimeError("internal count mismatch: in-set != disagreements + "
                           "gated; this indicates a bug in the apply-stage masks")

    # scores[:, 1] is -inf for any row the topk stage saved fewer than 2 candidates
    # for (see gate_topk_fingerprint.json's n_inf_margin), so gap is +inf there and
    # below_mask is always false for it: the recorded margin-gate convention
    # (analysis/margin_diagnostic.py's _gap()) falls out of this subtraction
    # without a special case.
    gap = scores[:, 0] - scores[:, 1]
    threshold = entry["threshold"]
    calibrate = entry["calibrate"]
    if threshold is None:
        # Per-language calibrated threshold (flat4_tau5 / direction 2): calibrate()
        # does its own model load, floor-21 matrix rebuild, and corpus scoring, and
        # returns {language_index: tau} for exactly reexam_idx's languages.
        tau_by_lang = calibrate(langs, N, reexam_idx)
        missing = [int(i) for i in reexam_idx if int(i) not in tau_by_lang]
        if missing:
            raise RuntimeError(f"calibrate() for variant {variant!r} returned no "
                               f"threshold for language index/indices {missing}")
        tau_arr = np.full(len(langs), -np.inf, dtype=np.float64)
        for li, t in tau_by_lang.items():
            tau_arr[li] = t
        per_line_threshold = tau_arr[pf21_line_pred]
        below_mask = gated_mask & (gap < per_line_threshold)
        threshold_desc = ("per-language calibrated (" + entry["tau_csv"] + "): " +
                          "; ".join(f"{langs[li]}={t:.4f} nats"
                                   for li, t in sorted(tau_by_lang.items())))
        threshold_for_meta = {langs[li]: t for li, t in sorted(tau_by_lang.items())}
    else:
        below_mask = gated_mask & (gap < threshold)
        threshold_desc = f"{threshold} nats (shared across the re-examined set)"
        threshold_for_meta = threshold
    n_below = int(below_mask.sum())
    n_gap_ok = n_gated - n_below

    replacement_min_n = entry["replacement_min_n"]
    accept = entry["accept"]
    if accept is None:
        accept_fn = lambda cid: N[cid] >= replacement_min_n  # noqa: E731
    else:
        accept_fn = lambda cid: accept(cid, N)  # noqa: E731

    pred = np.array(base_pred_arr, dtype=np.int16)
    n_moved = n_to_true = n_no_cand = 0
    for r in np.where(below_mask)[0].tolist():
        line = int(lines[r])
        replacement = None
        for j in range(1, TOPK_MARGIN):
            cid = int(ids[r, j])
            if cid < 0:
                continue  # unfilled candidate slot (topk stage short list); skip
            if accept_fn(cid):
                replacement = cid
                break
        if replacement is None:
            n_no_cand += 1
            continue
        pred[line] = np.int16(replacement)
        n_moved += 1
        if replacement == int(y[line]):
            n_to_true += 1
    if n_below != n_moved + n_no_cand:
        raise RuntimeError("internal count mismatch: below-threshold != moved + "
                           "kept-no-candidate; this indicates a bug in the "
                           "replacement loop")

    n_diff = int((pred != base_pred_arr).sum())
    if n_diff != n_moved:
        raise RuntimeError(f"{n_diff:,} lines differ from {base_pred_name} but "
                           f"n_moved={n_moved:,}; the moved-set count does not "
                           f"match the actual diff against {base_pred_name}")

    out_pred = _scratch(f"pred_gate_{variant}.npy")
    np.save(out_pred, pred)

    try:
        git_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, OSError) as e:
        raise RuntimeError(f"git rev-parse HEAD failed: {e}") from e

    meta = {
        "topk_fingerprint": fp,
        "variant": variant,
        "base_pred": base_pred_name,
        "threshold": threshold_for_meta,
        "tau_csv": entry["tau_csv"],
        "replacement_min_n": replacement_min_n,
        "reexamined_set_desc": entry["reexamined_set_desc"],
        "counts": {
            "n_affected": n_affected,
            "n_in_set": n_in_set,
            "n_disagree": n_disagree,
            "n_gated": n_gated,
            "n_gap_ok": n_gap_ok,
            "n_below": n_below,
            "n_moved": n_moved,
            "n_to_true": n_to_true,
            "n_no_cand": n_no_cand,
        },
        "git_commit": git_commit,
    }
    if threshold is None:
        meta["calibration"] = {
            "MARGIN_Q": MARGIN_Q,
            "CALIB_MAX": CALIB_MAX,
            "CALIB_SEED": CALIB_SEED,
            "MIN_CALIB_LINES": MIN_CALIB_LINES,
        }
    out_meta = _scratch(f"pred_gate_{variant}_meta.json")
    with open(out_meta + ".tmp", "w") as f:
        json.dump(meta, f)
    os.replace(out_meta + ".tmp", out_meta)

    lines_out = [
        f"# {variant} candidate build (Experiment {entry['experiment']})\n",
        entry["description"],
        "",
        f"- Constants: HEAD_N = {HEAD_N:,} training lines (the topk-stage low-N "
        f"criterion); flat_magnet category from {DIAG_CSV} (the topk-stage "
        f"flatness criterion; ZH_MAGNET = {ZH_MAGNET} is one of several inputs to "
        "that classification in analysis/diagnostic.py, not applied directly "
        f"here); TOPK_MARGIN = {TOPK_MARGIN} saved candidates per affected line. "
        f"Base predictions for this variant: {base_pred_name}. This variant's "
        f"re-examination threshold: {threshold_desc}. Replacement-candidate "
        f"minimum: {entry['replacement_min_n_desc']}",
        f"- Re-examined set for this variant: {entry['reexamined_set_desc']}",
    ]
    if isinstance(threshold_for_meta, dict):
        lines_out.append(
            f"- Per-language calibrated thresholds ({entry['tau_csv']}, the "
            f"{MARGIN_Q}th percentile of each language's own finite winning "
            "calibration margins under the floor-21 matrix): " +
            "; ".join(f"{lang} {t:.4f} nats"
                     for lang, t in threshold_for_meta.items()) + ".")
    lines_out += [
        f"- Affected: {n_affected:,} lines carry a saved floor21-prediction "
        "candidate list (the topk-stage expanded label set, always keyed on "
        "pred_floor21.npy regardless of this variant's base_pred).",
        f"- In-set: {n_in_set:,} of {n_affected:,} affected lines have a floor21 "
        "prediction in this variant's re-examined set.",
        f"- Top1-disagreements: {n_disagree:,} in-set lines whose saved top-1 "
        "candidate disagrees with pred_floor21.npy; left unchanged.",
        f"- Of the remaining {n_gated:,} in-set, agreeing lines: {n_gap_ok:,} have "
        f"a top1-minus-top2 score gap at or above their threshold and are kept "
        f"unchanged; {n_below:,} fall below threshold and are re-examined.",
        f"- Moved: {n_moved:,} of the {n_below:,} re-examined lines move to a "
        f"replacement candidate ranked 2 to {TOPK_MARGIN} with N >= "
        f"{replacement_min_n:,} training lines.",
        f"- Moved-to-true: {n_to_true:,} of the {n_moved:,} moved lines land on "
        "the true label recorded in y_true.npy.",
        f"- Kept-no-candidate: {n_no_cand:,} re-examined lines have no candidate "
        f"ranked 2 to {TOPK_MARGIN} meeting the acceptance condition, and keep the "
        f"{base_pred_name} prediction.",
        f"- Short candidate lists: the topk stage recorded {fp['n_short_cands']:,} "
        f"affected lines with fewer than {TOPK_MARGIN} saved candidates, of which "
        f"{fp['n_inf_margin']:,} had fewer than 2 and are treated as having "
        "infinite margin (never moved), following the recorded margin-gate "
        "convention (analysis/margin_diagnostic.py's _gap()).",
        f"- All lines outside the moved set are bit-identical to {base_pred_name} "
        f"({n_diff:,} lines differ, verified equal to n_moved). Output: "
        f"{out_pred}; metadata: {out_meta}.",
    ]
    out_md = _out(f"tables/gate_{variant}_build.md")
    os.makedirs(os.path.dirname(out_md), exist_ok=True)
    with open(out_md, "w") as f:
        f.write("\n".join(lines_out) + "\n")
    print("\n".join(lines_out))
    print(f"\nWrote {out_pred}, {out_meta}, and {out_md}")
    return out_pred


def run(stage: str, variant: str | None = None, model_path: str = None,
        scratch_dir: str = None, out_dir: str = "outputs") -> str:
    configure(model_path, scratch_dir, out_dir)
    if stage == "topk":
        return _run_topk()
    if stage == "apply":
        if variant is None:
            raise ValueError("stage='apply' requires a variant name")
        return _run_apply(variant)
    raise ValueError(f"unknown stage {stage!r}; must be 'topk' or 'apply'")


def main(argv=None):
    import argparse
    from analysis.model_context import add_arguments
    ap = argparse.ArgumentParser(description="gate variant stages")
    ap.add_argument("stage", choices=("topk", "apply"))
    ap.add_argument("variant", nargs="?", default=None)
    ap.add_argument("--out-dir", default="outputs")
    add_arguments(ap)
    a = ap.parse_args(argv)
    if a.stage == "apply" and a.variant is None:
        raise SystemExit("stage 'apply' requires a variant name")
    if a.stage == "topk" and a.variant is not None:
        raise SystemExit("stage 'topk' takes no variant name")
    run(a.stage, variant=a.variant, model_path=a.model_path,
        scratch_dir=a.scratch_dir, out_dir=a.out_dir)


if __name__ == "__main__":
    main()
