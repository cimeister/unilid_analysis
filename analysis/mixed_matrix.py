"""Step-3 implementation of the per-language combined method, rule v1
(EXPERIMENTS_PLAN.md, "Plan: per-language combined method", with its
2026-07-29 amendments; rule version 1 fixed by analysis/mixed_assign.py and
signed off by the user 2026-07-30).

Builds the mixed weight matrix from the pre-registered, signed-off
outputs/diagnostic/mixed_assignments.csv (build_mixed_weights, used by both
scoring stages below), scores the full pool once under that matrix (stage A),
then applies the per-language adaptive margin gate to the gate_on languages
only (stage B), plus a solotau variant that isolates the tau-recalibration
component of the mixed-minus-solo difference (stage B_solotau). A "noop"
stage exercises the exact stage-A scoring loop under the unmodified matrix W
as a pre-registered end-to-end check before any pass ever runs under
W_mixed.

Structural clone of three existing modules, faithfully:
- analysis/full_test_gt.py supplies the stage-A pattern: the chunked scoring
  loop over TEST_FILE, the fingerprint gate (atomic write, mismatch on resume
  aborts), the progress-json resume protocol, and the alignment gates
  (val-mask parity, pickle label cross-check, y_true index equality).
- analysis/solo_gates.py supplies the stage-B pattern: gate over a matrix
  that is neither the model's own cached weights nor an unmodified rebuild,
  tau recalibration under that matrix with a cause column
  (low_calibration/zero_strength), and the RES_CAP reassignment bar.
- analysis/floor_equalization.py supplies build_equalized_weights, used
  unchanged to build the floor-21 row source.

Row treatments under rule v1 (outputs/diagnostic/mixed_assignments.csv,
1,940 rows): 860 languages "unmod" (gate_on False), 1,080 languages
"floor21" (gate_on True). No language is assigned "gt" under this rule
version; build_mixed_weights only builds the Good-Turing matrix if the
assignments CSV ever calls for it, to avoid holding a fourth full weight
matrix in memory (alongside W, W_f21, and W_mixed) when nothing uses it.

HAZARD, read before touching build_mixed_weights: floor-21 rows are a
downward-only clamp of the row's unseen-token plateau (build_equalized_weights,
analysis/floor_equalization.py) and are NEVER renormalized back to a row sum
of 1.0. Renormalizing would add a per-token constant shift to every seen-token
log-weight in the row, which is a different method than the one that was
scored, judged, and carried (analysis/full_test_floor21.py, Exp 20). The
row-sum gate below checks the AS-CLAMPED sum against its analytically correct
value (below 1.0, since only the plateau mass changed), never against 1.0.

Evaluation (the per-language F1 comparison against the carried set and the
adoption rule, on the judge part of the seed-301 split) is
analysis/mixed_eval.py's job, not this module's; stages A and B/B_solotau
only build matrices, score, gate, and report raw counts.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd

from analysis.config import DEFAULT_SAMPLE_SIZE, TEST_FILE, TOTAL_LINES
from analysis.sample_data import load_sample
from analysis.transfer_sweep import _load_model_data, _load_unilid_model
from analysis.hierarchical_pool import VAL_MASK, RES_CAP
from analysis.full_test_eval import (
    CHUNK_LINES, SCRATCH_DIR, UNSEEN, EXCLUDED, EMPTY,
    _sample_line_indices, _parse_line,
)
from analysis.full_test_floor21 import FLOOR_TARGET
from analysis.full_test_gt import build_gt_weights, GT_CSV
from analysis.full_test_margin import HEAD_N
from analysis.floor_equalization import build_equalized_weights
from analysis.margin_diagnostic import (_topk_batch, _gap, PRF_CSV, TOPK_MARGIN,
                                        MARGIN_Q, MIN_CALIB_LINES, CALIB_MAX,
                                        CALIB_SEED, CORPUS_DIR)

ASSIGNMENTS_CSV = "outputs/diagnostic/mixed_assignments.csv"
EXPECTED_N_LANG = 1_940
ROW_TREATMENTS = ("unmod", "floor21", "gt")
ROW_SUM_TOL = 5e-4                # per-row |actual - expected| gate, floor21 rows
AGREE_MIN = 0.99                  # top-1 agreement floor vs pred_mixed_nogate.npy

TAU_CSV = "outputs/diagnostic/tau_mixed.csv"
TAU_JSON = "outputs/diagnostic/tau_mixed.json"
SOLOTAU_CSV = "outputs/diagnostic/tau_floor21_gate.csv"
OUT_MD = "outputs/tables/mixed_matrix_build.md"
STAGE_B_HEADING = "## Stage B (tau recalibrated under the mixed matrix)"
STAGE_B_SOLOTAU_HEADING = ("## Stage B_solotau (tau read from the floor21_gate "
                           "solo reference)")


def build_mixed_weights():
    """Builds W_mixed from outputs/diagnostic/mixed_assignments.csv against the
    model's own weight matrix W. Returns (W_mixed, assignments dataframe,
    sha256 of W_mixed, sha256 of the assignments CSV file bytes).

    Gates, in order: assignments CSV has exactly EXPECTED_N_LANG rows; its
    language column equals the model's own language order; every
    row_treatment is one of ROW_TREATMENTS; no gate_on language has
    N >= HEAD_N (N read from the assignments CSV, not recomputed). W is then
    sha-verified against fingerprint_floor21.json's sha256_base_W (the matrix
    that produced pred_baseline.npy); the floor-21 source matrix is rebuilt
    with build_equalized_weights and sha-verified against the same file's
    sha256_w21. The Good-Turing source matrix is built only if some language
    is assigned "gt" (none under rule v1) and is then sha-verified against
    fingerprint_gt.json's sha256_wgt.

    After assembly: every unmod row is asserted bit-identical to W, every
    floor21 row bit-identical to the floor-21 source, the four special-token
    columns (0:4) bit-identical to W, and every entry finite. floor21 rows
    additionally pass a row-sum gate against their analytically correct
    as-clamped sum (see the HAZARD note in the module docstring); unmod rows
    need no separate row-sum gate because the bit-identity assertion already
    covers them exactly.
    """
    with open(ASSIGNMENTS_CSV, "rb") as f:
        assign_bytes = f.read()
    assign_sha = hashlib.sha256(assign_bytes).hexdigest()
    assign = pd.read_csv(ASSIGNMENTS_CSV)
    if len(assign) != EXPECTED_N_LANG:
        raise RuntimeError(f"{ASSIGNMENTS_CSV} has {len(assign)} rows, expected "
                           f"{EXPECTED_N_LANG}")

    weights, langs, _lang_to_idx = _load_model_data()
    if list(assign.lang) != langs:
        raise RuntimeError(f"{ASSIGNMENTS_CSV} language order does not match the "
                           "model language list from _load_model_data()")
    n_lang = len(langs)

    bad_treatment = ~assign.row_treatment.isin(ROW_TREATMENTS)
    if bad_treatment.any():
        bad_vals = sorted(assign.loc[bad_treatment, "row_treatment"].unique())
        raise RuntimeError(f"{ASSIGNMENTS_CSV} row_treatment contains values "
                           f"outside {ROW_TREATMENTS}: {bad_vals}")

    N = assign["N"].values
    bad_gate = assign.gate_on.values & (N >= HEAD_N)
    if bad_gate.any():
        bad_langs = assign.loc[bad_gate, "lang"].tolist()
        raise RuntimeError(f"{int(bad_gate.sum())} gate_on language(s) have "
                           f"N >= HEAD_N ({HEAD_N}), first: {bad_langs[0]}")

    # --- source matrix W, sha-verified against the floor-21 fingerprint's own
    # record of the matrix that produced pred_baseline.npy ---
    W = np.array(weights, dtype=np.float32)
    del weights
    fp21_path = os.path.join(SCRATCH_DIR, "fingerprint_floor21.json")
    with open(fp21_path) as f:
        fp21 = json.load(f)
    sha_w = hashlib.sha256(W.tobytes()).hexdigest()
    if sha_w != fp21["sha256_base_W"]:
        raise RuntimeError(f"loaded W does not match sha256_base_W in {fp21_path}")

    # --- floor-21 source matrix, always built (rule v1 assigns it to 1,080
    # of 1,940 languages) ---
    W_f21, n_mod = build_equalized_weights(W, FLOOR_TARGET)
    if n_mod != n_lang:
        raise RuntimeError(f"floor {FLOOR_TARGET} modified {n_mod} of {n_lang} "
                           "rows; expected every row's own floor to exceed the "
                           "target (the precedent this model has always shown)")
    sha_w21 = hashlib.sha256(W_f21.tobytes()).hexdigest()
    if sha_w21 != fp21["sha256_w21"]:
        raise RuntimeError(f"rebuilt floor-21 matrix does not match sha256_w21 in "
                           f"{fp21_path}")

    # --- gt_counts.csv, needed for the floor21 row-sum gate below regardless
    # of whether any language is assigned "gt" ---
    gt_counts = pd.read_csv(GT_CSV)
    if gt_counts.lang.tolist() != langs:
        raise RuntimeError(f"{GT_CSV} language order does not match the model "
                           "language list")

    treat = assign.row_treatment.values
    is_unmod = treat == "unmod"
    is_floor21 = treat == "floor21"
    is_gt = treat == "gt"

    # --- Good-Turing source matrix, built only if the rule ever assigns "gt"
    # (memory: a third full weight matrix is not held unless something needs
    # it; no language is assigned "gt" under rule v1) ---
    if is_gt.any():
        W_gt = build_gt_weights(W, gt_counts, langs)
        fpgt_path = os.path.join(SCRATCH_DIR, "fingerprint_gt.json")
        with open(fpgt_path) as f:
            fpgt = json.load(f)
        sha_wgt = hashlib.sha256(W_gt.tobytes()).hexdigest()
        if sha_wgt != fpgt["sha256_wgt"]:
            raise RuntimeError(f"rebuilt gt matrix does not match sha256_wgt in "
                               f"{fpgt_path}")
    else:
        W_gt = None

    # --- assemble W_mixed row by row from the source matrix per row_treatment ---
    W_mixed = np.empty_like(W)
    W_mixed[is_unmod] = W[is_unmod]
    W_mixed[is_floor21] = W_f21[is_floor21]
    if is_gt.any():
        W_mixed[is_gt] = W_gt[is_gt]

    if not np.array_equal(W_mixed[is_unmod], W[is_unmod]):
        raise RuntimeError("unmod rows in W_mixed are not bit-identical to W")
    if not np.array_equal(W_mixed[is_floor21], W_f21[is_floor21]):
        raise RuntimeError("floor21 rows in W_mixed are not bit-identical to the "
                           "floor-21 source matrix")
    del W_f21, W_gt
    if not np.array_equal(W_mixed[:, :4], W[:, :4]):
        raise RuntimeError("special-token columns (0:4) were changed by the "
                           "mixed assembly")
    if not np.isfinite(W_mixed).all():
        raise RuntimeError("non-finite weights in the mixed matrix")

    # Per-treatment row-sum gate. unmod rows are already covered exactly by the
    # bit-identity assertion above. floor21 rows: the row's exp-sum before any
    # treatment is 1.0 (four specials at 0.2 each plus the seen+plateau budget
    # of 0.2, analysis/full_test_gt.py); the plateau mass is lowered from
    # plateau_mass_L to plateau_size_L * exp(FLOOR_TARGET) and nothing else
    # moves, so the new sum is exactly 1 - (plateau_mass_L
    # - plateau_size_L * exp(FLOOR_TARGET)). See the HAZARD note in the module
    # docstring: this gate must NOT be satisfied by renormalizing the row.
    plateau_mass = gt_counts["plateau_mass"].values.astype(np.float64)
    plateau_size = gt_counts["plateau_size"].values.astype(np.float64)
    row_sum = np.exp(W_mixed.astype(np.float64)).sum(axis=1)
    expect_floor21 = 1.0 - (plateau_mass - plateau_size * np.exp(FLOOR_TARGET))
    diff21 = np.abs(row_sum[is_floor21] - expect_floor21[is_floor21])
    if diff21.size and (diff21 > ROW_SUM_TOL).any():
        bad_i = np.where(is_floor21)[0][diff21 > ROW_SUM_TOL]
        raise RuntimeError(f"{len(bad_i)} floor21 row(s) fail the row-sum gate, "
                           f"first: {langs[bad_i[0]]} (max |diff| "
                           f"{diff21.max():.2e} > {ROW_SUM_TOL:.0e})")

    sha_mixed = hashlib.sha256(W_mixed.tobytes()).hexdigest()
    print(f"W_mixed built: {int(is_unmod.sum())} unmod, {int(is_floor21.sum())} "
          f"floor21, {int(is_gt.sum())} gt rows of {n_lang} (sha256 "
          f"{sha_mixed[:16]}...).")
    return W_mixed, assign, sha_mixed, assign_sha


def _run_noop() -> None:
    """Pre-registered end-to-end scorer check: score the first CHUNK_LINES
    lines of TEST_FILE using W itself (not W_mixed) through the exact stage-A
    code path (set_weight_sets caching, preprocessing, batching, the same
    three alignment gates), and compare predictions on that chunk's kept
    non-val lines against pred_baseline.npy. Aborts unless bit-identical.
    W is first sha-verified against fingerprint_floor21.json's sha256_base_W,
    the same check build_mixed_weights performs. Writes nothing permanent: no
    memmap, no progress file."""
    weights, langs, lang_to_idx = _load_model_data()
    n_lang = len(langs)
    W = np.array(weights, dtype=np.float32)
    del weights

    fp21_path = os.path.join(SCRATCH_DIR, "fingerprint_floor21.json")
    with open(fp21_path) as f:
        fp21 = json.load(f)
    sha_w = hashlib.sha256(W.tobytes()).hexdigest()
    if sha_w != fp21["sha256_base_W"]:
        raise RuntimeError(f"loaded W does not match sha256_base_W in {fp21_path}")

    base_mm = np.lib.format.open_memmap(
        os.path.join(SCRATCH_DIR, "pred_baseline.npy"), mode="r")
    if base_mm.shape != (TOTAL_LINES,):
        raise RuntimeError(f"pred_baseline.npy shape {base_mm.shape} != "
                           f"({TOTAL_LINES},)")
    y_mm = np.lib.format.open_memmap(
        os.path.join(SCRATCH_DIR, "y_true.npy"), mode="r")
    if y_mm.shape != (TOTAL_LINES,):
        raise RuntimeError(f"y_true.npy shape {y_mm.shape} != ({TOTAL_LINES},)")

    sample_idx = _sample_line_indices()
    parity_val = (np.arange(DEFAULT_SAMPLE_SIZE) % 2) == 0
    if not np.array_equal(np.load(VAL_MASK), parity_val):
        raise RuntimeError("saved val_mask.npy does not match position-parity split")
    val_lines = set(sample_idx[parity_val].tolist())
    sample_test_lines = sample_idx[~parity_val]
    pickle_y = np.array(load_sample(DEFAULT_SAMPLE_SIZE)["y_true"])[~parity_val]
    expect_label = dict(zip(sample_test_lines.tolist(), pickle_y.tolist()))

    n_chunk0 = min(CHUNK_LINES, TOTAL_LINES)
    model = _load_unilid_model()
    print("Loading model + caching W (noop check: not W_mixed)...", flush=True)
    model.model.set_weight_sets(W.tolist())
    del W

    with open(TEST_FILE) as fh:
        lines = [fh.readline() for _ in range(n_chunk0)]

    keep_pos, texts = [], []
    for i, line in enumerate(lines):
        if i in val_lines:
            if y_mm[i] != EXCLUDED:
                raise RuntimeError(f"line {i}: val line not EXCLUDED in the saved "
                                   f"y_true memmap")
            continue
        label, text = _parse_line(line)
        exp = expect_label.get(i)
        if exp is not None and label != exp:
            raise RuntimeError(f"alignment mismatch at line {i}: parsed "
                               f"{label!r}, pickle has {exp!r}")
        li = lang_to_idx.get(label)
        if li is None or y_mm[i] != li:
            raise RuntimeError(f"line {i}: label {label!r} (idx {li}) does not "
                               f"match saved y_true {int(y_mm[i])}")
        keep_pos.append(i)
        texts.append(text)

    pre, valid = [], []
    for k, t in enumerate(texts):
        p = model.preprocess(t)
        if p:
            pre.append(p)
            valid.append(k)
    zero_bias = [0.0] * n_lang
    out = np.full(len(texts), EMPTY, dtype=np.int16)
    if pre:
        batch = model.model.best_of_cached_weight_sets_biased_batch(pre, zero_bias)
        if len(batch) != len(pre):
            raise RuntimeError(f"noop chunk: scorer returned {len(batch)} results "
                               f"for {len(pre)} inputs")
        for k, (idx, _t, _s) in zip(valid, batch):
            out[k] = idx

    keep_pos = np.asarray(keep_pos, dtype=np.int64)
    base_chunk = np.array(base_mm[keep_pos])
    if int((base_chunk == UNSEEN).sum()) != 0:
        raise RuntimeError("pred_baseline.npy has UNSEEN entries on chunk-0 kept "
                           "lines")
    if not np.array_equal(out, base_chunk):
        n_diff = int((out != base_chunk).sum())
        raise RuntimeError(f"noop check failed: {n_diff} of {len(out)} chunk-0 "
                           "predictions under W differ from pred_baseline.npy")
    print(f"noop check passed: {len(out):,} chunk-0 kept non-val lines bit-"
          "identical to pred_baseline.npy under W through the stage-A code path.")


def _run_A() -> None:
    """Full-pool scoring under W_mixed. Structural clone of
    analysis/full_test_gt.py's run(): fingerprint gate, resumable chunked
    memmap, the same three alignment gates, zero-bias batch scorer. No
    post-loop metric report; evaluation is analysis/mixed_eval.py's job."""
    W_mixed, assign, sha_mixed, sha_assign = build_mixed_weights()
    langs = assign.lang.tolist()
    lang_to_idx = {l: i for i, l in enumerate(langs)}
    n_lang = len(langs)

    fp21_path = os.path.join(SCRATCH_DIR, "fingerprint_floor21.json")
    with open(fp21_path) as f:
        fp21 = json.load(f)

    fp = {"sha256_base_W": fp21["sha256_base_W"],
          "sha256_mixed": sha_mixed,
          "assignments_csv_sha256": sha_assign,
          "langs_sha256": hashlib.sha256("|".join(langs).encode()).hexdigest(),
          "chunk_lines": CHUNK_LINES, "total_lines": TOTAL_LINES}
    fp_path = os.path.join(SCRATCH_DIR, "fingerprint_mixed.json")
    if os.path.exists(fp_path):
        with open(fp_path) as f:
            prev = json.load(f)
        if prev != fp:
            bad = sorted(k for k in fp if prev.get(k) != fp[k])
            raise RuntimeError(f"mixed scratch state mismatch ({bad}); clear the "
                               f"mixed files in {SCRATCH_DIR} or restore inputs")
    else:
        with open(fp_path + ".tmp", "w") as f:
            json.dump(fp, f)
        os.replace(fp_path + ".tmp", fp_path)

    y_mm = np.lib.format.open_memmap(
        os.path.join(SCRATCH_DIR, "y_true.npy"), mode="r")
    if y_mm.shape != (TOTAL_LINES,):
        raise RuntimeError(f"y_true.npy shape {y_mm.shape} != ({TOTAL_LINES},)")
    y_all = np.asarray(y_mm)
    if int((y_all == UNSEEN).sum()) != 0:
        raise RuntimeError("y_true memmap contains UNSEEN lines; Exp 16 run incomplete")

    sample_idx = _sample_line_indices()
    parity_val = (np.arange(DEFAULT_SAMPLE_SIZE) % 2) == 0
    if not np.array_equal(np.load(VAL_MASK), parity_val):
        raise RuntimeError("saved val_mask.npy does not match position-parity split")
    val_lines = set(sample_idx[parity_val].tolist())
    sample_test_lines = sample_idx[~parity_val]
    pickle_y = np.array(load_sample(DEFAULT_SAMPLE_SIZE)["y_true"])[~parity_val]
    expect_label = dict(zip(sample_test_lines.tolist(), pickle_y.tolist()))

    pred_path = os.path.join(SCRATCH_DIR, "pred_mixed_nogate.npy")
    if os.path.exists(pred_path):
        pred_mm = np.lib.format.open_memmap(pred_path, mode="r+")
        if pred_mm.shape != (TOTAL_LINES,):
            raise RuntimeError(f"pred_mixed_nogate memmap has shape {pred_mm.shape}")
    else:
        pred_mm = np.lib.format.open_memmap(pred_path, mode="w+", dtype=np.int16,
                                            shape=(TOTAL_LINES,))
        pred_mm[:] = UNSEEN
        pred_mm.flush()
    progress_path = os.path.join(SCRATCH_DIR, "progress_mixed.json")
    done_chunks = set()
    if os.path.exists(progress_path):
        with open(progress_path) as f:
            done_chunks = set(json.load(f))

    n_chunks = (TOTAL_LINES + CHUNK_LINES - 1) // CHUNK_LINES
    model = None
    zero_bias = [0.0] * n_lang
    with open(TEST_FILE) as fh:
        for chunk in range(n_chunks):
            lo = chunk * CHUNK_LINES
            hi = min(lo + CHUNK_LINES, TOTAL_LINES)
            if chunk in done_chunks:
                for _ in range(hi - lo):
                    fh.readline()
                continue
            if model is None:
                print("Loading model + caching mixed weights...", flush=True)
                model = _load_unilid_model()
                model.model.set_weight_sets(W_mixed.tolist())
                del W_mixed
            lines = [fh.readline() for _ in range(hi - lo)]

            keep_pos, texts = [], []
            for j, line in enumerate(lines):
                i = lo + j
                if i in val_lines:
                    if y_mm[i] != EXCLUDED:
                        raise RuntimeError(f"line {i}: val line not EXCLUDED in "
                                           f"the saved y_true memmap")
                    continue
                label, text = _parse_line(line)
                exp = expect_label.get(i)
                if exp is not None and label != exp:
                    raise RuntimeError(f"alignment mismatch at line {i}: parsed "
                                       f"{label!r}, pickle has {exp!r}")
                li = lang_to_idx.get(label)
                if li is None or y_mm[i] != li:
                    raise RuntimeError(f"line {i}: label {label!r} (idx {li}) "
                                       f"does not match saved y_true "
                                       f"{int(y_mm[i])}")
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
                batch = model.model.best_of_cached_weight_sets_biased_batch(
                    pre, zero_bias)
                if len(batch) != len(pre):
                    raise RuntimeError(f"chunk {chunk}: scorer returned "
                                       f"{len(batch)} results for {len(pre)} "
                                       f"inputs")
                for k, (idx, _t, _s) in zip(valid, batch):
                    out[k] = idx
            pred_mm[np.asarray(keep_pos, dtype=np.int64)] = out
            pred_mm.flush()
            done_chunks.add(chunk)
            with open(progress_path + ".tmp", "w") as f:
                json.dump(sorted(done_chunks), f)
            os.replace(progress_path + ".tmp", progress_path)
            print(f"chunk {chunk + 1}/{n_chunks} done", flush=True)

    print("Computing completion counts...", flush=True)
    y = np.asarray(y_mm)
    kept = y >= 0
    predm = np.asarray(pred_mm)
    if int((predm[kept] == UNSEEN).sum()) != 0:
        raise RuntimeError("unprocessed kept lines remain in pred_mixed_nogate")
    print(f"stage A complete: {int(kept.sum()):,} kept lines scored under the "
          f"mixed matrix (sha256 {sha_mixed[:16]}...). Output: {pred_path}")


def _recalibrate_or_load_tau(model, langs: list, N: np.ndarray, gate_idx: np.ndarray,
                             sha_mixed: str, sha_assign: str):
    """Stage B's tau source. If TAU_CSV already exists and is a complete
    recalibration for the current gate_on set (same language set, expected
    columns), it is loaded and the affected-line scoring pass proceeds
    without recalibrating (restart case). Otherwise this recalibrates tau
    under the already-cached mixed matrix on each gate_on language's own
    train lines, exactly mirroring analysis/solo_gates.py's calibration loop
    and cause logic, and writes TAU_CSV before returning (so a restart after
    this point sees a complete file and skips recalibration).

    TAU_CSV alone does not identify which matrix it was calibrated under, so
    a resume must not trust it on file existence alone: TAU_JSON is written
    alongside TAU_CSV recording the sha256_mixed and assignments_csv_sha256
    of the build that produced it, and the resume path aborts unless
    TAU_JSON exists and both values match the current build's sha_mixed and
    sha_assign. Without this, a TAU_CSV calibrated under a superseded matrix
    (e.g. after the assignments CSV changed) would be silently reused, which
    would look correct while answering a different question."""
    n_gated = len(gate_idx)
    if os.path.exists(TAU_CSV):
        if not os.path.exists(TAU_JSON):
            raise RuntimeError(f"{TAU_CSV} exists but {TAU_JSON} is missing; "
                               f"{TAU_CSV} cannot be verified as calibrated "
                               "under the current mixed matrix and may be "
                               "calibrated under a different matrix; delete "
                               f"{TAU_CSV} to force recalibration or restore "
                               f"{TAU_JSON}")
        with open(TAU_JSON) as f:
            tau_meta = json.load(f)
        if (tau_meta.get("sha256_mixed") != sha_mixed
                or tau_meta.get("assignments_csv_sha256") != sha_assign):
            raise RuntimeError(f"{TAU_CSV} is calibrated under a different "
                               f"matrix than the current build ({TAU_JSON}'s "
                               "sha256_mixed/assignments_csv_sha256 do not "
                               f"match); delete {TAU_CSV} and {TAU_JSON} to "
                               "force recalibration or investigate why they "
                               "differ")
        tau_df = pd.read_csv(TAU_CSV)
        required = {"lang", "n_scoreable", "n_self_won", "tau", "excluded", "cause"}
        missing_cols = required - set(tau_df.columns)
        gate_langs = [langs[li] for li in gate_idx]
        if missing_cols or len(tau_df) != n_gated or sorted(tau_df.lang) != sorted(gate_langs):
            raise RuntimeError(f"{TAU_CSV} exists but is not a complete "
                               f"recalibration for the current gate_on set "
                               f"({len(tau_df)} rows, columns "
                               f"{sorted(tau_df.columns)}; expected {n_gated} "
                               f"rows over exactly the gate_on languages, "
                               f"columns {sorted(required)}); delete it to force "
                               "recalibration or investigate why it differs")
        by_lang = tau_df.set_index("lang")
        tau = {int(li): float(by_lang.loc[langs[li], "tau"]) for li in gate_idx}
        n_excluded = int(by_lang.loc[gate_langs, "excluded"].sum())
        cause_counts = by_lang.loc[gate_langs, "cause"].fillna("").value_counts()
        print(f"Loaded existing {TAU_CSV} ({n_gated} rows); skipping "
              "recalibration.")
        return tau, n_excluded, cause_counts, f"loaded from an existing, complete {TAU_CSV}"

    rng = np.random.default_rng(CALIB_SEED)
    tau = {}
    calib_rows = []
    for li in gate_idx:
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
        q_l = MARGIN_Q * (1.0 - min(float(N[li]), float(HEAD_N)) / HEAD_N)
        zero_strength = q_l <= 0.0
        excluded = low_calib or zero_strength
        cause = ("low_calibration" if low_calib
                 else "zero_strength" if zero_strength else "")
        tau[int(li)] = (float("-inf") if excluded
                        else float(np.percentile(gaps, q_l)))
        calib_rows.append({"lang": lang, "n_scoreable": len(ctopk),
                           "n_self_won": len(wins), "tau": tau[int(li)],
                           "excluded": excluded, "cause": cause})
    os.makedirs(os.path.dirname(TAU_CSV), exist_ok=True)
    tau_df = pd.DataFrame(calib_rows)
    tau_df.to_csv(TAU_CSV + ".tmp", index=False)
    os.replace(TAU_CSV + ".tmp", TAU_CSV)
    tau_meta = {"sha256_mixed": sha_mixed, "assignments_csv_sha256": sha_assign}
    with open(TAU_JSON + ".tmp", "w") as f:
        json.dump(tau_meta, f)
    os.replace(TAU_JSON + ".tmp", TAU_JSON)
    n_excluded = sum(r["excluded"] for r in calib_rows)
    cause_counts = tau_df["cause"].fillna("").value_counts()
    print(f"Wrote {TAU_CSV} and {TAU_JSON} ({len(calib_rows)} rows) before "
          "the affected-line scoring pass.")
    return tau, n_excluded, cause_counts, "recalibrated under the mixed matrix"


def _load_solotau(langs: list, gate_idx: np.ndarray, assign: pd.DataFrame):
    """Stage B_solotau's tau source: outputs/diagnostic/tau_floor21_gate.csv,
    the solo-gate reference build's own tau values (rule v1's gate_on
    languages are exactly the languages assigned row_treatment "floor21", so
    floor21_gate is their solo reference). Aborts if gate_idx is not exactly
    the floor21 row_treatment index set, since the solotau decomposition is
    only defined when every gated language's solo reference is the floor-21
    build. No recalibration; aborts if any gate_on language is missing from
    that CSV."""
    floor21_idx = np.where(assign.row_treatment.values == "floor21")[0]
    if not np.array_equal(gate_idx, floor21_idx):
        raise RuntimeError("gate_on index set does not equal the floor21 "
                           "row_treatment index set; the solotau "
                           "decomposition is only defined when every gated "
                           "language's solo reference is the floor-21 build")
    solo_df = pd.read_csv(SOLOTAU_CSV)
    required = {"lang", "tau", "excluded", "cause"}
    missing_cols = required - set(solo_df.columns)
    if missing_cols:
        raise RuntimeError(f"{SOLOTAU_CSV} missing required columns: "
                           f"{sorted(missing_cols)}")
    by_lang = solo_df.set_index("lang")
    gate_langs = [langs[li] for li in gate_idx]
    missing_langs = [l for l in gate_langs if l not in by_lang.index]
    if missing_langs:
        raise RuntimeError(f"{len(missing_langs)} gate_on language(s) missing "
                           f"from {SOLOTAU_CSV}, first: {missing_langs[0]}")
    tau = {int(li): float(by_lang.loc[langs[li], "tau"]) for li in gate_idx}
    n_excluded = int(by_lang.loc[gate_langs, "excluded"].sum())
    cause_counts = by_lang.loc[gate_langs, "cause"].fillna("").value_counts()
    return tau, n_excluded, cause_counts


def _write_report(heading: str, body_lines: list) -> None:
    """Writes `heading`'s section into OUT_MD. A previous instance of the SAME
    heading is replaced (idempotent re-run); the other stage's section, if an
    earlier run already wrote it, is preserved. Section order in the file is
    always Stage B before Stage B_solotau, regardless of which stage ran
    first (the spec allows appending or a second section; this keeps a fixed
    order under either invocation sequence)."""
    section = "\n".join([heading, ""] + body_lines)
    sections = {}
    if os.path.exists(OUT_MD):
        with open(OUT_MD) as f:
            content = f.read()
        for h in (STAGE_B_HEADING, STAGE_B_SOLOTAU_HEADING):
            i = content.find(h)
            if i == -1:
                continue
            j = content.find("\n## ", i + 1)
            sections[h] = (content[i:j] if j != -1 else content[i:]).rstrip("\n")
    sections[heading] = section
    ordered = [sections[h] for h in (STAGE_B_HEADING, STAGE_B_SOLOTAU_HEADING)
              if h in sections]
    new_content = ("# Mixed-matrix build report (per-language combined method, "
                  "rule v1)\n\n" + "\n\n".join(ordered) + "\n")
    os.makedirs(os.path.dirname(OUT_MD), exist_ok=True)
    with open(OUT_MD, "w") as f:
        f.write(new_content)


def _run_gate(solotau: bool) -> str:
    """Stage B (solotau=False) or Stage B_solotau (solotau=True). Clone of
    analysis/solo_gates.py's gate exactly, differences: gated set = gate_on
    languages from the mixed assignment (not a fixed N-threshold tail_idx);
    matrix = W_mixed; base predictions = pred_mixed_nogate.npy (stage A must
    be complete); tau source differs by stage (see _recalibrate_or_load_tau
    / _load_solotau); reassignment bar RES_CAP; top-1 agreement >= AGREE_MIN."""
    W_mixed, assign, sha_mixed, sha_assign = build_mixed_weights()
    langs = assign.lang.tolist()
    n_lang = len(langs)
    N = assign["N"].values.astype(np.float64)
    gate_idx = np.where(assign.gate_on.values)[0]
    if len(gate_idx) == 0:
        raise RuntimeError("no gate_on language in the assignments CSV; nothing "
                           "to gate")

    # Provenance cross-check against the base per-language table every other
    # gate script in this project trusts for language order (solo_gates.py's
    # first validation).
    prf = pd.read_csv(PRF_CSV)
    if prf.lang.tolist() != langs:
        raise RuntimeError(f"{PRF_CSV} language order does not match the "
                           "assignments CSV / model language list")
    if not np.array_equal(assign["N"].values, prf.N.values):
        raise RuntimeError(f"{PRF_CSV} N values do not match the assignments "
                           f"CSV N values; the mixed and floor21_gate builds "
                           "must use identical N")

    fp_path = os.path.join(SCRATCH_DIR, "fingerprint_mixed.json")
    if not os.path.exists(fp_path):
        raise RuntimeError(f"{fp_path} missing; run stage A first")
    with open(fp_path) as f:
        fp = json.load(f)
    if sha_mixed != fp["sha256_mixed"]:
        raise RuntimeError(f"rebuilt mixed matrix does not match sha256_mixed in "
                           f"{fp_path}; it would not be bit-identical to the "
                           "matrix that produced pred_mixed_nogate.npy")
    if sha_assign != fp["assignments_csv_sha256"]:
        raise RuntimeError(f"assignments CSV sha256 does not match {fp_path}'s "
                           "recorded value; the assignments file changed since "
                           "stage A ran")

    y = np.asarray(np.lib.format.open_memmap(
        os.path.join(SCRATCH_DIR, "y_true.npy"), mode="r"))
    if y.shape != (TOTAL_LINES,):
        raise RuntimeError(f"y_true.npy shape {y.shape} != ({TOTAL_LINES},)")
    base_path = os.path.join(SCRATCH_DIR, "pred_mixed_nogate.npy")
    if not os.path.exists(base_path):
        raise RuntimeError(f"{base_path} missing; run stage A first")
    pg = np.asarray(np.lib.format.open_memmap(base_path, mode="r"))
    if pg.shape != (TOTAL_LINES,):
        raise RuntimeError(f"{base_path} shape {pg.shape} != ({TOTAL_LINES},)")
    kept = y >= 0
    if int((pg[kept] == UNSEEN).sum()) != 0:
        raise RuntimeError(f"{base_path} has UNSEEN entries on kept lines; "
                           "stage A is not complete")

    affected = np.where((pg >= 0) & np.isin(pg, gate_idx))[0]
    print(f"{len(affected):,} lines carry a gate_on mixed-nogate prediction")

    model = _load_unilid_model()
    print("Caching mixed weights...", flush=True)
    model.model.set_weight_sets(W_mixed.tolist())
    del W_mixed

    if solotau:
        tau, n_excluded, cause_counts = _load_solotau(langs, gate_idx, assign)
        tau_note = (f"read from {SOLOTAU_CSV} (rule v1's gate_on solo reference "
                   "is floor21_gate), no recalibration")
        out_pred = os.path.join(SCRATCH_DIR, "pred_mixed_solotau.npy")
        section_heading = STAGE_B_SOLOTAU_HEADING
    else:
        tau, n_excluded, cause_counts, tau_note = _recalibrate_or_load_tau(
            model, langs, N, gate_idx, sha_mixed, sha_assign)
        out_pred = os.path.join(SCRATCH_DIR, "pred_mixed.npy")
        section_heading = STAGE_B_HEADING

    want = set(affected.tolist())
    texts = {}
    with open(TEST_FILE) as fh:
        for i in range(TOTAL_LINES):
            line = fh.readline()
            if i not in want:
                continue
            _label, text = _parse_line(line)
            texts[i] = text
    if len(texts) != len(affected):
        raise RuntimeError(f"collected {len(texts)} texts for {len(affected)} "
                           "lines")
    drop = []
    pos, topk = _topk_batch(model, [texts[i] for i in affected], drop)
    if drop:
        raise RuntimeError(f"{len(drop)} affected lines empty after preprocess")
    agree = (np.mean([topk[k][0][0] == int(pg[affected[pos[k]]])
                      for k in range(len(topk))]) if topk else 1.0)
    if agree < AGREE_MIN:
        raise RuntimeError(f"top-1 agreement with pred_mixed_nogate.npy "
                           f"{agree:.4f} < {AGREE_MIN}")

    pred = np.array(pg, dtype=np.int16)
    n_disagree = n_gated = n_reassigned = n_to_true = n_no_target = 0
    for k, cands in enumerate(topk):
        line = int(affected[pos[k]])
        top1 = int(cands[0][0])
        if top1 != int(pg[line]):
            n_disagree += 1
            continue
        t = tau[top1]
        n_gated += 1
        if not (len(cands) >= 2 and _gap(cands) < t):
            continue
        head_cands = [int(c[0]) for c in cands[1:] if N[int(c[0])] >= RES_CAP]
        if not head_cands:
            n_no_target += 1
            continue
        pred[line] = np.int16(head_cands[0])
        n_reassigned += 1
        n_to_true += int(head_cands[0] == int(y[line]))
    np.save(out_pred, pred)

    cause_str = ", ".join(f"{c or 'not_excluded'}: {n}"
                          for c, n in cause_counts.items())
    body = [
        f"- mixed matrix rebuilt and fingerprint-verified against {fp_path} "
        f"(sha {sha_mixed[:16]}...); assignments CSV sha256-verified.",
        f"- gate_on set: {len(gate_idx)} of {n_lang} languages.",
        f"- tau {tau_note}: {n_excluded} of {len(gate_idx)} gated languages "
        f"excluded ({cause_str}).",
        f"- Gate_on mixed-nogate-predicted lines scored: {len(affected):,} "
        f"(top-1 agreement {agree:.4f}; {n_disagree} disagreeing lines left "
        "ungated).",
        f"- Of {n_gated:,} agreeing gated lines: reassigned to a head "
        f"candidate {n_reassigned:,}; {n_to_true:,} land on the true label; "
        f"{n_no_target:,} below-tau lines kept for lack of a head candidate "
        f"in the top-{TOPK_MARGIN}.",
        f"- All other lines bit-identical to pred_mixed_nogate.npy. Output: "
        f"{out_pred}.",
    ]
    _write_report(section_heading, body)
    print("\n".join(body))
    print(f"Wrote {out_pred} and updated {OUT_MD}")
    return out_pred


def run(stage: str) -> None:
    if stage == "noop":
        _run_noop()
    elif stage == "A":
        _run_A()
    elif stage == "B":
        _run_gate(solotau=False)
    elif stage == "B_solotau":
        _run_gate(solotau=True)
    else:
        raise ValueError(f"unknown stage {stage!r}; must be one of 'noop', "
                         "'A', 'B', 'B_solotau'")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: python -m analysis.mixed_matrix "
                         "{noop|A|B|B_solotau}")
    run(sys.argv[1])
