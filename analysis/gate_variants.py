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
Outputs, all in SCRATCH_DIR: gate_topk_lines.npy, gate_topk_ids.npy,
gate_topk_scores.npy, and gate_topk_fingerprint.json.

Stage "apply" (run("apply", variant=NAME)) reads those three saved arrays and the
fingerprint, and builds one full-pool prediction file for a named rule variant
declared in the VARIANTS dict below. No further scoring happens in this stage. The
first variant, "shared9_bar18k", implements direction 1 (Experiment 47,
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

Pattern sources: analysis/solo_gates.py (floor-21 matrix rebuild and sha256
verification against fingerprint_floor21.json, affected-line selection, model
loading, _topk_batch usage, the per-line reassignment loop over saved candidates);
analysis/mixed_matrix.py (fingerprint and chunked resumable full-pool conventions);
analysis/full_test_eval.py (SCRATCH_DIR, memmap sentinels, _parse_line).

Constants imported, never redefined here: HEAD_N (analysis.full_test_margin),
FLOOR_TARGET (analysis.full_test_floor21), TOPK_MARGIN and PRF_CSV
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
from analysis.full_test_eval import SCRATCH_DIR, _parse_line
from analysis.full_test_margin import HEAD_N
from analysis.full_test_floor21 import FLOOR_TARGET
from analysis.floor_equalization import build_equalized_weights
from analysis.hierarchical_pool import RES_CAP, DIAG_CSV
from analysis.margin_diagnostic import _topk_batch, PRF_CSV, TOPK_MARGIN
from analysis.diagnostic import ZH_MAGNET

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
        # Re-examined set: languages with N < HEAD_N only. Takes (N, zH); zH is
        # ignored here (the flat-distribution additions to the topk-stage candidate
        # universe are direction 2's concern, not direction 1's).
        "reexamined_set": lambda N, zH: N < HEAD_N,
        "reexamined_set_desc": (
            f"languages with N < HEAD_N ({HEAD_N:,} training lines) only; the "
            "flat-distribution languages added to the topk-stage candidate universe "
            "by the category == 'flat_magnet' criterion are excluded from this "
            "variant's re-examined set."
        ),
        "threshold": SHARED_TAU_V1,
        "replacement_min_n": HEAD_N,
        # Optional per-candidate acceptance function, signature (candidate_lang_idx,
        # N) -> bool. None means the default: N[candidate_lang_idx] >=
        # replacement_min_n. Later variants (directions 2 and 3) can override this
        # with a custom condition without changing the apply code below.
        "accept": None,
    },
}
for _vname, _ventry in VARIANTS.items():
    if "accept" not in _ventry:
        raise RuntimeError(f"VARIANTS[{_vname!r}] must declare 'accept' explicitly "
                           "(None selects the default N >= replacement_min_n rule)")
del _vname, _ventry


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
    lines_path = os.path.join(SCRATCH_DIR, "gate_topk_lines.npy")
    ids_path = os.path.join(SCRATCH_DIR, "gate_topk_ids.npy")
    scores_path = os.path.join(SCRATCH_DIR, "gate_topk_scores.npy")
    fp_path = os.path.join(SCRATCH_DIR, "gate_topk_fingerprint.json")
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
    to SCRATCH_DIR. Idempotent: if gate_topk_lines/ids/scores.npy already exist and
    match gate_topk_fingerprint.json, prints and returns without rescoring; if a
    fingerprint exists but does not match, aborts naming what changed."""
    prf = pd.read_csv(PRF_CSV)
    langs = prf.lang.tolist()
    N = prf.N.values
    n_lang = len(langs)

    weights, langs_m, _lang_to_idx = _load_model_data()
    if langs_m != langs:
        raise RuntimeError(f"model language order does not match {PRF_CSV}")

    diag = pd.read_csv(DIAG_CSV)
    if diag.lang.tolist() != langs:
        raise RuntimeError(f"{DIAG_CSV} language order does not match {PRF_CSV}")

    low_n_mask = N < HEAD_N
    flat_mask = (diag.category == "flat_magnet").values
    expanded_mask = low_n_mask | flat_mask
    expanded_idx = np.where(expanded_mask)[0]

    W = np.array(weights, dtype=np.float32)
    del weights
    fp21_path = os.path.join(SCRATCH_DIR, "fingerprint_floor21.json")
    with open(fp21_path) as f:
        fp21 = json.load(f)
    sha_w = hashlib.sha256(W.tobytes()).hexdigest()
    if sha_w != fp21["sha256_base_W"]:
        raise RuntimeError(f"loaded W does not match sha256_base_W in {fp21_path}")

    matrix, n_mod = build_equalized_weights(W, FLOOR_TARGET)
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

    y = np.asarray(np.lib.format.open_memmap(
        os.path.join(SCRATCH_DIR, "y_true.npy"), mode="r"))
    if y.shape != (TOTAL_LINES,):
        raise RuntimeError(f"y_true.npy shape {y.shape} != ({TOTAL_LINES},)")
    pf21 = np.asarray(np.lib.format.open_memmap(
        os.path.join(SCRATCH_DIR, "pred_floor21.npy"), mode="r"))
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

    lines_path = os.path.join(SCRATCH_DIR, "gate_topk_lines.npy")
    ids_path = os.path.join(SCRATCH_DIR, "gate_topk_ids.npy")
    scores_path = os.path.join(SCRATCH_DIR, "gate_topk_scores.npy")
    fp_path = os.path.join(SCRATCH_DIR, "gate_topk_fingerprint.json")
    arrays_exist = all(os.path.exists(p) for p in (lines_path, ids_path, scores_path))

    if os.path.exists(fp_path):
        with open(fp_path) as f:
            prev = json.load(f)
        if prev == fp:
            if arrays_exist:
                print(f"existing gate_topk_* arrays in {SCRATCH_DIR} match the "
                      "current fingerprint; skipping rescoring.")
                return lines_path
            raise RuntimeError(f"{fp_path} matches the current fingerprint but one "
                               "or more of gate_topk_lines.npy / gate_topk_ids.npy / "
                               f"gate_topk_scores.npy is missing from {SCRATCH_DIR}; "
                               "investigate before rerunning")
        bad = sorted(k for k in fp if prev.get(k) != fp[k])
        raise RuntimeError(f"gate_topk scratch state mismatch ({bad}); clear the "
                           f"gate_topk_* files in {SCRATCH_DIR} or investigate what "
                           "changed")
    elif arrays_exist:
        raise RuntimeError(f"gate_topk_lines/ids/scores.npy exist in {SCRATCH_DIR} "
                           f"but {fp_path} is missing; they cannot be verified as "
                           "produced under the current inputs; remove the stale "
                           "arrays to force a rescore or restore the fingerprint")

    # Model load and weight-set caching complete before the wanted-lines read so
    # `matrix`'s memory is freed (del matrix) before raw_lines' dict allocation
    # begins, instead of the two large allocations overlapping in peak memory.
    model = _load_unilid_model()
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


def _run_apply(variant: str) -> str:
    """Stage "apply": post-processes the saved stage-"topk" arrays into a full-pool
    prediction file for one named entry in VARIANTS. No scoring happens here."""
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; must be one of "
                         f"{sorted(VARIANTS)}")
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
        os.path.join(SCRATCH_DIR, "y_true.npy"), mode="r"))
    if y.shape != (TOTAL_LINES,):
        raise RuntimeError(f"y_true.npy shape {y.shape} != ({TOTAL_LINES},)")
    pf21 = np.asarray(np.lib.format.open_memmap(
        os.path.join(SCRATCH_DIR, "pred_floor21.npy"), mode="r"))
    if pf21.shape != (TOTAL_LINES,):
        raise RuntimeError(f"pred_floor21.npy shape {pf21.shape} != ({TOTAL_LINES},)")

    n_affected = len(lines)
    pf21_line_pred = pf21[lines]
    if (pf21_line_pred < 0).any():
        raise RuntimeError("pred_floor21.npy has a negative prediction at one or "
                           "more gate_topk_lines.npy indices; numpy fancy indexing "
                           "with a negative int16 index silently wraps to the end "
                           "of the language array instead of failing")

    reexam_lang_mask = entry["reexamined_set"](N, zH)
    if (reexam_lang_mask & ~expanded_mask).any():
        raise RuntimeError(f"variant {variant!r}'s re-examined-set mask includes "
                           "language(s) outside the topk-stage expanded label set; "
                           "the topk stage never saved candidates for those lines")
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
    below_mask = gated_mask & (gap < threshold)
    n_below = int(below_mask.sum())
    n_gap_ok = n_gated - n_below

    replacement_min_n = entry["replacement_min_n"]
    accept = entry["accept"]
    if accept is None:
        accept_fn = lambda cid: N[cid] >= replacement_min_n  # noqa: E731
    else:
        accept_fn = lambda cid: accept(cid, N)  # noqa: E731

    pred = np.array(pf21, dtype=np.int16)
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

    n_diff = int((pred != pf21).sum())
    if n_diff != n_moved:
        raise RuntimeError(f"{n_diff:,} lines differ from pred_floor21.npy but "
                           f"n_moved={n_moved:,}; the moved-set count does not "
                           "match the actual diff against pred_floor21.npy")

    out_pred = os.path.join(SCRATCH_DIR, f"pred_gate_{variant}.npy")
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
        "threshold": threshold,
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
    out_meta = os.path.join(SCRATCH_DIR, f"pred_gate_{variant}_meta.json")
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
        f"here); TOPK_MARGIN = {TOPK_MARGIN} saved candidates per affected line; "
        f"this variant's re-examination threshold is {threshold} nats; RES_CAP = "
        f"{RES_CAP:,} training lines is the promoted configuration's own "
        "replacement-candidate bar, not used by this variant.",
        f"- Re-examined set for this variant: {entry['reexamined_set_desc']}",
        f"- Affected: {n_affected:,} lines carry a saved floor21-prediction "
        "candidate list (the topk-stage expanded label set).",
        f"- In-set: {n_in_set:,} of {n_affected:,} affected lines have a floor21 "
        "prediction in this variant's re-examined set.",
        f"- Top1-disagreements: {n_disagree:,} in-set lines whose saved top-1 "
        "candidate disagrees with pred_floor21.npy; left unchanged.",
        f"- Of the remaining {n_gated:,} in-set, agreeing lines: {n_gap_ok:,} have "
        f"a top1-minus-top2 score gap at or above {threshold} nats and are kept "
        f"unchanged; {n_below:,} fall below {threshold} nats and are re-examined.",
        f"- Moved: {n_moved:,} of the {n_below:,} re-examined lines move to a "
        f"replacement candidate ranked 2 to {TOPK_MARGIN} with N >= "
        f"{replacement_min_n:,} training lines.",
        f"- Moved-to-true: {n_to_true:,} of the {n_moved:,} moved lines land on "
        "the true label recorded in y_true.npy.",
        f"- Kept-no-candidate: {n_no_cand:,} re-examined lines have no candidate "
        f"ranked 2 to {TOPK_MARGIN} meeting the acceptance condition, and keep the "
        "floor21 prediction.",
        f"- Short candidate lists: the topk stage recorded {fp['n_short_cands']:,} "
        f"affected lines with fewer than {TOPK_MARGIN} saved candidates, of which "
        f"{fp['n_inf_margin']:,} had fewer than 2 and are treated as having "
        "infinite margin (never moved), following the recorded margin-gate "
        "convention (analysis/margin_diagnostic.py's _gap()).",
        f"- All lines outside the moved set are bit-identical to pred_floor21.npy "
        f"({n_diff:,} lines differ, verified equal to n_moved). Output: "
        f"{out_pred}; metadata: {out_meta}.",
    ]
    out_md = f"outputs/tables/gate_{variant}_build.md"
    os.makedirs(os.path.dirname(out_md), exist_ok=True)
    with open(out_md, "w") as f:
        f.write("\n".join(lines_out) + "\n")
    print("\n".join(lines_out))
    print(f"\nWrote {out_pred}, {out_meta}, and {out_md}")
    return out_pred


def run(stage: str, variant: str | None = None) -> str:
    if stage == "topk":
        return _run_topk()
    if stage == "apply":
        if variant is None:
            raise ValueError("stage='apply' requires a variant name")
        return _run_apply(variant)
    raise ValueError(f"unknown stage {stage!r}; must be 'topk' or 'apply'")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("usage: python -m analysis.gate_variants topk | "
                         "python -m analysis.gate_variants apply VARIANT_NAME")
    stage_arg = sys.argv[1]
    if stage_arg == "topk":
        if len(sys.argv) != 2:
            raise SystemExit("usage: python -m analysis.gate_variants topk")
        run(stage_arg)
    elif stage_arg == "apply":
        if len(sys.argv) != 3:
            raise SystemExit("usage: python -m analysis.gate_variants apply "
                             "VARIANT_NAME")
        run(stage_arg, variant=sys.argv[2])
    else:
        raise SystemExit(f"unknown stage {stage_arg!r}; must be 'topk' or 'apply'")
