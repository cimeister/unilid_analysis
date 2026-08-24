"""E5: CommonLID for the camera-ready (EXPERIMENTS_PLAN.md, "Camera-ready
evaluation program", E5, pre-registered 2026-08-07). Evaluates the promoted
configuration (floor-21 plus the gate "gate_flat4_prox21") on CommonLID
(373,230 lines, 109 bare ISO-639-3 tags, web domain), following the
macrolanguage-aware scoring convention of analysis/commonlid_eval.py: a
prediction is correct if its bare ISO code equals the gold tag, or is a member
of the gold macrolanguage (SIL iso-639-3-macrolanguages.tab), plus the
documented tgl<->fil bridge. Metrics are accuracy and tag-level macro F1 via
analysis.metrics.compute_metrics after mapping predictions into tag space
(analysis/commonlid_eval.py's to_scored_tag convention).

Two stages, dispatched by --stage:

STAGE "score" (--stage score): loads the full UniLID model and must run on
SLURM (slurm_commonlid_calibrated.sh), never on a login node; this module
makes no host assertion, the requirement is documentation only, matching
analysis/external_bench_eval.py's own convention. Reads CommonLID via
analysis/commonlid_eval.py's own reader (_read_commonlid, imported, not
reimplemented), scores every text under the unmodified weight matrix W
(best_of_cached_weight_sets_batch, the baseline pass), builds the floor-21
matrix W_f21 = build_equalized_weights(W, FLOOR_TARGET) and sha256-verifies it
against fingerprint_floor21.json (both sha256_base_W and sha256_w21, the
E2-reviewed standard: analysis/external_bench_eval.py's run_score), then
scores under W_f21 with top_k_of_cached_weight_sets_batch, banking the top-5
ids and scores per line (float32 scores, -inf fill for short candidate lists,
plus a short-candidate counter: the same E2 convention,
analysis/external_bench_eval.py's _score_top5). The baseline and floor-21
scoring loops themselves (_score_baseline, _score_top5) are imported from
analysis/external_bench_eval.py rather than reimplemented, since that module
already carries the reviewed, self-checked implementation of exactly this
scoring pattern (SCORE_CHUNK batching, the EMPTY=-1 sentinel for rows empty
after preprocess).

A wiring gate compares this pass's baseline predictions and floor-21 top-1
predictions, row for row, against the persisted Exp 39 predictions in
outputs/diagnostic/commonlid_carried_preds.npz (analysis/commonlid_carried.py).
Agreement >= WIRING_GATE_HIGH passes silently; agreement in
[WIRING_GATE_LOW, WIRING_GATE_HIGH) prints a RECORDED NOTE and continues (this
band matches the recorded GlotLID re-scoring self-agreement, 0.9951,
EXPERIMENTS_RESULTS.md); agreement below WIRING_GATE_LOW aborts, naming both
artifacts. Persists gold tags, the baseline prediction, and the banked top-5
candidates to a scratch .npz plus a JSON provenance sidecar (git commit, tsv
sha256, matrix shas, model path, preprocessing description, counts).

STAGE "eval" (--stage eval): reads the .npz and sidecar, no model load.
Re-verifies the CommonLID tsv's sha256 against the sidecar (catches drift
since the score stage ran). Re-derives the promoted configuration's gate
(gate_flat4_prox21) from the banked top-5 candidates via
analysis/external_bench_eval.py's own _load_gate_thresholds and
_gate_walk_and_merge, IMPORTED and called directly, not reimplemented: those
two helpers already carry the group-A/group-B masking, threshold gate,
per-row replacement walk (analysis.gate_variants._walk_replacement), and merge
that reproduce the promoted configuration's re-examination logic bit-exactly,
and external_bench_eval.py passed its own bit-exact self-check
(--stage selfcheck) against analysis/gate_variants.py's production output.
Thresholds are unchanged (no refitting on this domain).

Before any new number is written to disk, four wiring gates must reproduce
the recorded Exp 12/39 values within EVAL_GATE_TOL: baseline macro-aware
accuracy (EXPECTED_BASELINE_ACC, imported from analysis.commonlid_carried),
baseline tag-level macro F1 (EXPECTED_BASELINE_TAG_F1, same module), floor-21
tag-level macro F1 (EVAL_FLOOR21_TAG_F1) and floor-21 macro-aware accuracy
(EVAL_FLOOR21_ACC, both this module, citing
outputs/tables/commonlid_carried.md's "floor21" row). Any failure aborts,
naming the recorded source.

THOSE FOUR GATES BIND THE RELEASED MODEL ONLY. All four constants are
measurements of the released weights, so holding another model to them would
compare two models and call the expected difference a reproduction failure,
and would make the corrected round's own CommonLID cells -- the whole
deliverable of such a run -- unreachable. For a non-default model the same four
comparisons are computed at the same EVAL_GATE_TOL, printed, and written into
the report labelled INFORMATIONAL, NOT A GATE; nothing is withheld and nothing
exits nonzero (analysis/paper_breakdowns.py's `gates_binding` rule, mirrored
here and in analysis/external_bench_eval.py).

FLOOR TARGET. The clamp constant both stages use is the one recorded in the
model's own <scratch-dir>/fingerprint_floor21.json, not the module constant
FLOOR_TARGET. That constant (-21.0, analysis.full_test_floor21) is the RELEASED
chain's guard-selected value; the floor target is a measured per-model selection
and the corrected base model's own round-grid sweep selected -17.0. Stage
"score" already read the fingerprint; stage "eval" checks the scored sidecar's
recorded floor_target against the SAME record via
analysis.external_bench_eval._check_sidecar_floor_target, imported rather than
reimplemented. For the default model the expected value IS the module constant
and the check and its message are exactly what they were. The `n_mod == n_lang`
assertion is likewise replaced by
analysis.floor_equalization.verify_one_sided_clamp: all 1,940 released rows
moving at c = -21 is an incidental fact of that model, not a property of the
clamp.

GATE THRESHOLDS AND THE SIBLING MODULE'S CONTEXT. The eval stage's
_load_gate_thresholds reads analysis/external_bench_eval.py's module-level tau
CSV paths, which default to the released tree. Stage "eval" therefore calls
analysis.external_bench_eval.configure() with this run's own (model, scratch
root, output root) before calling it, so a non-default model reads its own tau
CSVs; without that call a corrected model's banked candidates would silently be
gated by the released model's thresholds. For the default model configure()
resolves the same released pair the module already lazily resolves, so the
thresholds read are the same files.

Outputs: <out-dir>/tables/commonlid_calibrated.md (metrics table for baseline /
floor-21 / gated, the group A/B re-examination accounting, and a constants
block with tau CSV shas) and
<out-dir>/diagnostic/commonlid_calibrated_per_tag.csv
(tag, support, f1_baseline, f1_floor21, f1_gated). The default out-dir is the
released tree, `outputs`; a non-default model must pass its own
(analysis.model_context.resolve_out_root). The scored .npz and its sidecar go to
CL_DIR for the released model and to a per-model subdirectory of it for any
other, so a second model never overwrites the first's scored arrays.

Hard rules followed throughout: no silent fallback (every missing or
mismatched artifact aborts, naming the artifact); every numeric constant used
is imported from its origin module wherever one exists, never re-typed as a
literal (see "Constants imported, never redefined here" below).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter

import numpy as np
import pandas as pd

from analysis import external_bench_eval as _ebe
from analysis.commonlid_carried import (EXPECTED_BASELINE_ACC,
                                        EXPECTED_BASELINE_TAG_F1,
                                        N_SPECIAL_COLS, SCORE_CHUNK)
from analysis.commonlid_eval import (CL_DIR, CL_TSV, MACRO_TAB,
                                     _canonical_correct,
                                     _load_macro_member_to_macro, _read_commonlid)
from analysis.external_bench_eval import (_check_sidecar_floor_target,
                                          _gate_walk_and_merge, _git_commit,
                                          _load_gate_thresholds, _score_baseline,
                                          _score_top5, _sha256_file,
                                          configure as _configure_bench)
from analysis.floor_equalization import (build_equalized_weights,
                                         _special_columns,
                                         verify_one_sided_clamp)
from analysis.format_utils import to_markdown
from analysis.full_test_eval import EMPTY
from analysis.model_context import resolve, resolve_out_root
from analysis.full_test_floor21 import FLOOR_TARGET
from analysis.full_test_margin import HEAD_N
from analysis.gate_variants import D3_PROX
from analysis.hierarchical_pool import RES_CAP
from analysis.margin_diagnostic import PRF_CSV, TOPK_MARGIN
from analysis.metrics import compute_metrics, compute_per_class_metrics
from analysis.transfer_sweep import (UNILID_MODEL_PATH, _load_model_data,
                                     _load_unilid_model)

# Constants imported, never redefined here: EXPECTED_BASELINE_ACC,
# EXPECTED_BASELINE_TAG_F1, SCORE_CHUNK (analysis.commonlid_carried; Exp
# 12/39 for the first two; SCORE_CHUNK is the module _score_baseline/
# _score_top5 themselves use, imported here only so the meta JSON sidecar
# records the true chunk size rather than a re-typed literal -- note
# analysis.margin_diagnostic defines an UNRELATED SCORE_CHUNK=5_000 of its
# own for a different scoring loop; deliberately not imported from there);
# CL_DIR, CL_TSV, MACRO_TAB (analysis.commonlid_eval); FLOOR_TARGET
# (analysis.full_test_floor21); HEAD_N (analysis.full_test_margin); RES_CAP
# (analysis.hierarchical_pool); D3_PROX (analysis.gate_variants -- the two tau
# CSV paths are no longer imported from there directly: they are read from
# analysis.external_bench_eval's configured _TAU1_CSV/_TAU2_CSV, which are the
# released-tree literals for the released model and this run's own output root
# otherwise); PRF_CSV, TOPK_MARGIN (analysis.margin_diagnostic);
# EMPTY (analysis.full_test_eval); UNILID_MODEL_PATH (analysis.transfer_sweep).

# --- constants defined in this module ---

# Pre-registration (EXPERIMENTS_PLAN.md, "Camera-ready evaluation program", E5,
# 2026-08-07): CommonLID's row and tag counts. Independently re-verified
# against outputs/diagnostic/commonlid_carried_preds.npz's own shape and tag
# set (Exp 39, analysis/commonlid_carried.py).
EXPECTED_ROWS = 373_230
EXPECTED_TAGS = 109

SCORE_NPZ = os.path.join(CL_DIR, "scored_commonlid_calibrated.npz")
SCORE_META = os.path.join(CL_DIR, "scored_commonlid_calibrated_meta.json")

# The floor-21 fingerprint written by the model's own floor-21 stage, by name.
# Same file and field as analysis.external_bench_eval.FINGERPRINT_FLOOR21_NAME
# and analysis.commonlid_carried.FINGERPRINT_FLOOR21_NAME.
FINGERPRINT_FLOOR21_NAME = "fingerprint_floor21.json"

# The Exp 39 persisted per-line predictions this module's score stage is wired
# against (analysis/commonlid_carried.py's run()). EXPERIMENTS_PLAN.md E5
# pre-registration: "match the persisted per-line predictions in
# outputs/diagnostic/commonlid_carried_preds.npz for baseline and floor-21".
CARRIED_PREDS_NPZ = "outputs/diagnostic/commonlid_carried_preds.npz"

# Wiring-gate policy (revised after the Opus pre-run review, 2026-08-07).
# The BASELINE arm re-runs the identical deterministic code path that produced
# the persisted Exp 39 predictions (same model file, same preprocess, same row
# order), so the only acceptable outcome is exact equality; any difference
# means a real wiring change and aborts. The FLOOR-21 TOP-1 arm compares
# top_k's rank-1 against Exp 39's best_of-scored floor-21 predictions, whose
# tie-breaking can legitimately differ (the recorded top1_agree_overall on the
# internal pool is 0.99999955), so it keeps a band: >= WIRING_GATE_HIGH passes,
# [WIRING_GATE_LOW, WIRING_GATE_HIGH) prints a RECORDED NOTE and continues,
# below WIRING_GATE_LOW aborts. The eval stage's EVAL_GATE_TOL reproduction
# gates backstop this band: drift anywhere near WIRING_GATE_LOW would move the
# tag-level macro F1 far beyond 5e-4 and abort there.
WIRING_GATE_HIGH = 0.999
WIRING_GATE_LOW = 0.995

# Pre-registration (E5): tolerance for the three reproduction checks in the
# eval stage.
EVAL_GATE_TOL = 5e-4

# Exp 39 recorded floor-21 values (outputs/tables/commonlid_carried.md,
# "floor21" row; written by analysis/commonlid_carried.py's run()). The
# matching baseline constants (0.8452 / 0.7228, Exp 12/39) are imported above
# from analysis.commonlid_carried rather than retyped here; no equivalent
# module constants exist for the floor-21 numbers, so they are defined here
# with this citation.
EVAL_FLOOR21_TAG_F1 = 0.7181
EVAL_FLOOR21_ACC = 0.8491

OUT_MD = "outputs/tables/commonlid_calibrated.md"
OUT_PER_TAG_CSV = "outputs/diagnostic/commonlid_calibrated_per_tag.csv"


def _configure_paths(ctx, out_dir: str, purpose: str) -> str:
    """Point every path this module reads or writes at the resolved context.

    Five of them default to the RELEASED model's record and would be overwritten
    in place by a second model: OUT_MD is where tables/commonlid.tex's three rows
    come from, OUT_PER_TAG_CSV is the per-tag detail behind them,
    CARRIED_PREDS_NPZ is analysis/commonlid_carried.py's Exp 39 array (read here,
    and written by that module under the same rule), and SCORE_NPZ/SCORE_META are
    the scored arrays, whose names carry no model. resolve_out_root refuses a
    non-default model paired with the default root, with anything inside it, or
    with a store-backed root, exactly as in analysis/external_bench_eval.py's
    configure(); for the default model it returns the root and never raises, and
    every path below rebuilds to the string it already had.

    Returns the resolved repo-side output root."""
    global SCORE_NPZ, SCORE_META, CARRIED_PREDS_NPZ, OUT_MD, OUT_PER_TAG_CSV
    out_root = resolve_out_root(ctx, out_dir, purpose=purpose)
    if ctx.is_default_model:
        score_dir = CL_DIR
    else:
        stem = os.path.splitext(os.path.basename(ctx.model_path))[0]
        score_dir = os.path.join(CL_DIR, f"scored_{stem}")
    SCORE_NPZ = os.path.join(score_dir, "scored_commonlid_calibrated.npz")
    SCORE_META = os.path.join(score_dir, "scored_commonlid_calibrated_meta.json")
    CARRIED_PREDS_NPZ = os.path.join(out_root, "diagnostic",
                                     "commonlid_carried_preds.npz")
    OUT_MD = os.path.join(out_root, "tables", "commonlid_calibrated.md")
    OUT_PER_TAG_CSV = os.path.join(out_root, "diagnostic",
                                   "commonlid_calibrated_per_tag.csv")
    return out_root


# ---------------------------------------------------------------------------
# STAGE "score" (SLURM only: loads the full model)
# ---------------------------------------------------------------------------

def run_score(model_path: str = None, scratch_dir: str = None,
              out_dir: str = None) -> str:
    """STAGE "score". Must run on SLURM (loads the full ~1,940-language model);
    this function makes no host assertion, the requirement is documentation
    only (module docstring, and slurm_commonlid_calibrated.sh)."""
    # Resolved before the 373,230-line read so an unsafe (model, root) pair is
    # refused immediately rather than after decompressing the benchmark.
    ctx = resolve(model_path, scratch_dir,
                  purpose="CommonLID calibrated scoring")
    _configure_paths(ctx, out_dir, "CommonLID calibrated scoring")
    texts, tags = _read_commonlid()
    n_rows = len(texts)
    if n_rows != EXPECTED_ROWS:
        raise RuntimeError(
            f"{CL_TSV} has {n_rows:,} data rows, expected {EXPECTED_ROWS:,} "
            "(EXPERIMENTS_PLAN.md, \"Camera-ready evaluation program\", E5 "
            "pre-registration)")
    tags = np.array(tags)
    n_tags = len(set(tags.tolist()))
    if n_tags != EXPECTED_TAGS:
        raise RuntimeError(
            f"{CL_TSV} has {n_tags} distinct gold tags, expected "
            f"{EXPECTED_TAGS} (EXPERIMENTS_PLAN.md, \"Camera-ready evaluation "
            "program\", E5 pre-registration)")

    weights, langs, _unused_lang_to_idx = _load_model_data(ctx.model_path)
    W = np.array(weights, dtype=np.float32)
    del weights

    tsv_sha = _sha256_file(CL_TSV)
    macro_tab_sha = _sha256_file(MACRO_TAB)
    git_commit = _git_commit()

    print(f"Loading model ({ctx.model_path})...", flush=True)
    model = _load_unilid_model(ctx.model_path)
    if model.langs != langs:
        raise RuntimeError("_load_unilid_model's language list differs from "
                           "_load_model_data's; the two loaders read the "
                           "model file inconsistently")

    # Preprocessing convention reused from analysis/commonlid_carried.py and
    # analysis/external_bench_eval.py: apply the model's own preprocess()
    # (tokenizer normalizer, then pre_tokenizer), and treat a falsy result
    # (None or empty string) as "empty after preprocess".
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

    fp_path = os.path.join(ctx.scratch_dir, FINGERPRINT_FLOOR21_NAME)
    with open(fp_path) as f:
        fp = json.load(f)
    # The clamp comes from the model's own fingerprint, so the scored arrays
    # cannot be built at a different c than the full-pool predictions and the tau
    # CSVs they are gated by. A missing field aborts naming the artifact rather
    # than falling back to the released chain's FLOOR_TARGET.
    if "floor_target" not in fp:
        raise RuntimeError(
            f"{fp_path} records no floor_target field, so the clamp this stage "
            f"must build at cannot be read. The module constant FLOOR_TARGET "
            f"({FLOOR_TARGET}, analysis.full_test_floor21) is the RELEASED "
            f"chain's guard-selected value and must not stand in for it.")
    target = float(fp["floor_target"])
    if ctx.is_default_model and target != FLOOR_TARGET:
        raise RuntimeError(
            f"{fp_path} records floor_target {fp['floor_target']}, expected "
            f"FLOOR_TARGET {FLOOR_TARGET} (analysis.full_test_floor21); the "
            "fingerprint was built under a different floor than this run uses")
    print(f"building and verifying the clamped matrix at c = {target}...",
          flush=True)
    special = _special_columns(ctx.model_path)
    if len(special) != N_SPECIAL_COLS:
        raise RuntimeError(
            f"{ctx.model_path}: found {len(special)} special columns, expected "
            f"{N_SPECIAL_COLS} (analysis.commonlid_carried.N_SPECIAL_COLS). "
            f"analysis.floor_equalization._special_columns drops a special token "
            f"that is absent from the vocabulary without comment, and a short "
            f"list leaves a special column in the row minimum, so the clamp "
            f"would lower the wrong entries.")
    w21, n_mod = build_equalized_weights(W, target, special_idx=special)
    verify_one_sided_clamp(W, target, special, n_mod)
    sha_w = hashlib.sha256(W.tobytes()).hexdigest()
    if sha_w != fp["sha256_base_W"]:
        raise RuntimeError(
            f"loaded base weight matrix does not match the recorded "
            f"fingerprint: loaded sha256 {sha_w}, recorded sha256_base_W "
            f"{fp['sha256_base_W']} (from {fp_path}); this run's model file "
            "may differ from the one the internal full-test pipeline "
            "fingerprinted")
    sha_w21 = hashlib.sha256(w21.tobytes()).hexdigest()
    if sha_w21 != fp["sha256_w21"]:
        raise RuntimeError(
            f"rebuilt floor-21 matrix does not match the recorded "
            f"fingerprint: rebuilt sha256 {sha_w21}, recorded sha256_w21 "
            f"{fp['sha256_w21']} (from {fp_path})")
    model.model.set_weight_sets(w21.tolist())
    del W, w21

    print(f"floor-21 top-{TOPK_MARGIN} pass...", flush=True)
    top5_ids, top5_scores, n_short_cands = _score_top5(model, pre, vidx, n_rows)
    del model

    # --- WIRING GATE: this pass's baseline and floor-21 top-1 predictions vs
    # the persisted Exp 39 arrays (analysis/commonlid_carried.py) ---
    if not os.path.exists(CARRIED_PREDS_NPZ):
        raise FileNotFoundError(
            f"wiring-gate artifact missing: {CARRIED_PREDS_NPZ} "
            "(analysis/commonlid_carried.py's Exp 39 output)")
    with np.load(CARRIED_PREDS_NPZ) as cz:
        carried_keys = set(cz.files)
        required_keys = {"tags", "baseline", "floor21"}
        missing_keys = required_keys - carried_keys
        if missing_keys:
            raise RuntimeError(
                f"{CARRIED_PREDS_NPZ} is missing key(s) "
                f"{sorted(missing_keys)}; available keys: "
                f"{sorted(carried_keys)}")
        carried_tags = cz["tags"]
        carried_baseline = cz["baseline"].astype(np.int64)
        carried_floor21 = cz["floor21"].astype(np.int64)
    if carried_tags.shape[0] != n_rows:
        raise RuntimeError(
            f"{CARRIED_PREDS_NPZ}'s tags array has {carried_tags.shape[0]:,} "
            f"rows, this pass scored {n_rows:,}; the two artifacts are not "
            f"comparable row for row ({CL_TSV} vs {CARRIED_PREDS_NPZ})")
    if carried_baseline.shape[0] != n_rows:
        raise RuntimeError(
            f"{CARRIED_PREDS_NPZ} has {carried_baseline.shape[0]:,} rows, "
            f"this pass scored {n_rows:,}; the two artifacts are not "
            f"comparable row for row ({CL_TSV} vs {CARRIED_PREDS_NPZ})")
    if not np.array_equal(carried_tags, tags):
        n_tag_diff = int((carried_tags != tags).sum())
        raise RuntimeError(
            f"{CARRIED_PREDS_NPZ}'s gold tags differ from this pass's gold "
            f"tags on {n_tag_diff:,} of {n_rows:,} rows; the two artifacts "
            f"were built over different CommonLID row orderings ({CL_TSV} "
            f"vs {CARRIED_PREDS_NPZ})")

    baseline_arr = pred_baseline.astype(np.int64)
    baseline_agree = float((baseline_arr == carried_baseline).mean())
    floor21_agree = float((top5_ids[:, 0].astype(np.int64) == carried_floor21).mean())

    # Baseline arm: exact equality required (identical deterministic code path
    # as the Exp 39 pass; see the wiring-gate policy comment at the constants).
    if not np.array_equal(baseline_arr, carried_baseline):
        diff_idx = np.where(baseline_arr != carried_baseline)[0]
        raise RuntimeError(
            f"WIRING GATE (baseline) FAILED: {len(diff_idx):,} of {n_rows:,} "
            f"rows differ from {CARRIED_PREDS_NPZ}'s baseline predictions "
            f"(agreement {baseline_agree:.6f}); expected exact equality since "
            "this is the identical deterministic scoring path. First 10 "
            f"differing row indices: {diff_idx[:10].tolist()}.")
    print(f"WIRING GATE (baseline): exact equality with {CARRIED_PREDS_NPZ} "
          "-- OK", flush=True)

    # Floor-21 top-1 arm: banded (top_k rank-1 vs Exp 39's best_of argmax;
    # tie-breaking can differ legitimately).
    if floor21_agree >= WIRING_GATE_HIGH:
        print(f"WIRING GATE (floor21_top1): agreement {floor21_agree:.6f} >= "
              f"{WIRING_GATE_HIGH} -- OK", flush=True)
    elif floor21_agree >= WIRING_GATE_LOW:
        print(f"RECORDED NOTE: WIRING GATE (floor21_top1) agreement "
              f"{floor21_agree:.6f} is in the [{WIRING_GATE_LOW}, "
              f"{WIRING_GATE_HIGH}) band (top_k vs best_of tie-breaking); "
              "continuing; the eval stage's reproduction gates backstop this.",
              flush=True)
    else:
        raise RuntimeError(
            f"WIRING GATE (floor21_top1) agreement {floor21_agree:.6f} < "
            f"{WIRING_GATE_LOW}; aborting. Compared this pass's floor-21 "
            f"rank-1 (to be persisted to {SCORE_NPZ}) against "
            f"{CARRIED_PREDS_NPZ} (analysis/commonlid_carried.py, Exp 39).")

    meta = {
        "git_commit": git_commit,
        "tsv_path": CL_TSV,
        "tsv_sha256": tsv_sha,
        "macro_tab_path": MACRO_TAB,
        "macro_tab_sha256": macro_tab_sha,
        "n_rows": n_rows,
        "n_tags": n_tags,
        "n_empty": n_empty,
        "model_path": ctx.model_path,
        "model_sha256": ctx.sha256(),
        "is_default_model": ctx.is_default_model,
        "matrix_shas": {
            "sha256_W_loaded": sha_w,
            "sha256_w21_computed": sha_w21,
            "sha256_w21_fingerprint": fp["sha256_w21"],
            "sha256_base_W_fingerprint": fp["sha256_base_W"],
            "fingerprint_floor21_path": fp_path,
        },
        "preprocessing": (
            "model.preprocess(text): UnilidModel.preprocess (UNILID/unilid/"
            "model_io.py) applies the base tokenizer's normalizer, then its "
            "pre_tokenizer; a row is treated as empty when the result is "
            "None or an empty string (the same test analysis/"
            "commonlid_carried.py and analysis/external_bench_eval.py use, "
            "`if p:` on the preprocessed value). Empty rows are excluded "
            "from the batches sent to the scorer and keep the EMPTY (-1) "
            "sentinel (analysis.full_test_eval.EMPTY) as their prediction "
            "under both the baseline pass and the floor-21 top-k pass."
        ),
        "score_chunk": SCORE_CHUNK,
        "topk_margin": TOPK_MARGIN,
        "floor_target": target,
        "n_short_cands": n_short_cands,
        "wiring_gate": {
            "carried_preds_npz": CARRIED_PREDS_NPZ,
            "baseline_agreement": baseline_agree,
            "floor21_top1_agreement": floor21_agree,
            "gate_high": WIRING_GATE_HIGH,
            "gate_low": WIRING_GATE_LOW,
        },
    }

    os.makedirs(os.path.dirname(SCORE_NPZ), exist_ok=True)
    npz_tmp = SCORE_NPZ.replace(".npz", ".tmp.npz")
    np.savez_compressed(npz_tmp, tags=tags, pred_baseline=pred_baseline,
                        top5_ids=top5_ids, top5_scores=top5_scores,
                        n_empty=np.int64(n_empty))
    os.replace(npz_tmp, SCORE_NPZ)

    meta_tmp = SCORE_META + ".tmp"
    with open(meta_tmp, "w") as f:
        json.dump(meta, f, indent=2)
    os.replace(meta_tmp, SCORE_META)

    print(f"Wrote {SCORE_NPZ}, {SCORE_META}")
    print(f"SCORE OK. {n_rows:,} rows, {n_empty:,} empty, {n_tags} tags, "
          f"{n_short_cands:,} rows with fewer than {TOPK_MARGIN} saved "
          f"candidates; baseline wiring-gate agreement {baseline_agree:.6f}, "
          f"floor21_top1 wiring-gate agreement {floor21_agree:.6f}.")
    return SCORE_NPZ


def _cross_model_gate_message(model_path: str) -> str:
    """The non-default-model form of the reproduction-gate heading.

    EXPECTED_BASELINE_ACC, EXPECTED_BASELINE_TAG_F1, EVAL_FLOOR21_TAG_F1 and
    EVAL_FLOOR21_ACC are measurements of the RELEASED model. Comparing another
    model's recomputed values against them is a cross-model comparison: a
    difference is the expected outcome, not a reproduction failure. The
    comparisons are still computed at the same tolerance and reported in full;
    they withhold no number and fail no exit code. Wording and rule mirror
    analysis/external_bench_eval.py's `_cross_model_baseline_message`."""
    return (f"INFORMATIONAL, NOT A GATE: the four recorded values below are "
            f"measurements of the RELEASED model and this run scored "
            f"{model_path}, so a difference is an expected cross-model "
            f"difference, not a regression and not a reproduction failure. "
            f"Every configuration in the table above WAS computed and written, "
            f"with this run's own numbers.")


# ---------------------------------------------------------------------------
# STAGE "eval" (login node: no model load, reads the .npz)
# ---------------------------------------------------------------------------

def run_eval(model_path: str = None, scratch_dir: str = None,
             out_dir: str = None) -> str:
    """STAGE "eval". No model load: reads SCORE_NPZ/SCORE_META, re-derives the
    promoted configuration's gate via the imported E2-reviewed helpers, checks
    the four pre-registered reproduction gates, and writes the output table
    and per-tag CSV.

    --model/--scratch-dir locate the scored .npz (which lives in a per-model
    directory) and the model's own floor-21 fingerprint and tau CSVs; the model
    IDENTITY is still taken from the sidecar and the two are asserted equal
    below, so this stage cannot be pointed at one model's arrays and told they
    are another's."""
    ctx = resolve(model_path, scratch_dir,
                  purpose="CommonLID calibrated evaluation")
    _configure_paths(ctx, out_dir, "CommonLID calibrated evaluation")
    # The eval stage's gate thresholds come from analysis/external_bench_eval.py's
    # module-level tau CSV paths, which default to the released tree; configure()
    # repoints them at this run's own output root. Without it a non-default model
    # would be gated by the released model's thresholds with no error.
    _configure_bench(model_path, scratch_dir, out_dir)
    if not os.path.exists(SCORE_NPZ):
        raise FileNotFoundError(f"scored npz missing: {SCORE_NPZ}; run "
                                "--stage score first")
    if not os.path.exists(SCORE_META):
        raise FileNotFoundError(f"scored meta sidecar missing: {SCORE_META}")
    with open(SCORE_META) as f:
        meta = json.load(f)

    # The sidecar is the authority on which model produced these arrays; the
    # --model this stage was given only located them. Disagreement means the eval
    # is about to attribute one model's predictions to another.
    if os.path.abspath(meta["model_path"]) != ctx.model_path:
        raise RuntimeError(
            f"{SCORE_META} records model_path {meta['model_path']}, but this "
            f"eval resolved {ctx.model_path}; the scored arrays were produced "
            f"by a different model than the one this stage was pointed at")
    if meta.get("is_default_model", ctx.is_default_model) != ctx.is_default_model:
        raise RuntimeError(
            f"{SCORE_META} records is_default_model "
            f"{meta['is_default_model']}, this eval resolved "
            f"{ctx.is_default_model}")
    # The path check above does not survive a model file replaced in place. The
    # banked top-5 ids are language INDICES into the scored model's own language
    # list, and this stage reads that list from the file at the recorded path, so
    # a substituted file would relabel every candidate with no error.
    recorded_model_sha = meta.get("model_sha256")
    if recorded_model_sha is None:
        print(f"NOTE: {SCORE_META} records no model_sha256; it predates that "
              f"field, so the model behind these arrays is identified by path "
              f"only ({meta['model_path']}).", flush=True)
    else:
        current_model_sha = ctx.sha256()
        if current_model_sha != recorded_model_sha:
            raise RuntimeError(
                f"{SCORE_META} records model_sha256 {recorded_model_sha}, but "
                f"{ctx.model_path} now hashes to {current_model_sha}; the model "
                f"file was replaced since the score stage ran, so the banked "
                f"candidate language indices no longer mean what this stage "
                f"would read them as")

    current_tsv_sha = _sha256_file(CL_TSV)
    if current_tsv_sha != meta["tsv_sha256"]:
        raise RuntimeError(
            f"{CL_TSV} sha256 has drifted since the score stage ran: scored "
            f"under {meta['tsv_sha256']}, currently {current_tsv_sha}; rerun "
            "--stage score before evaluating")

    with np.load(SCORE_NPZ) as z:
        tags = z["tags"]
        pred_baseline = z["pred_baseline"].astype(np.int64)
        top5_ids = z["top5_ids"].astype(np.int64)
        # NOT upcast to float64: analysis.gate_variants stores and walks its
        # own candidate scores as float32 throughout, and this module's gate
        # must use the identical float32 arithmetic for the top1-minus-top2
        # margin and D3_PROX comparisons to mean the same thing
        # (analysis/external_bench_eval.py's run_eval, same reasoning).
        top5_scores = z["top5_scores"]
        n_empty = int(z["n_empty"])

    n_rows = len(tags)
    if n_rows != EXPECTED_ROWS:
        raise RuntimeError(f"{SCORE_NPZ} has {n_rows:,} rows, expected "
                           f"{EXPECTED_ROWS:,} (EXPERIMENTS_PLAN.md, "
                           "\"Camera-ready evaluation program\", E5 "
                           "pre-registration)")
    if top5_scores.dtype != np.float32:
        raise RuntimeError(
            f"{SCORE_NPZ}: top5_scores has dtype {top5_scores.dtype}, "
            "expected float32 (the score stage's own dtype, "
            "analysis.external_bench_eval._score_top5)")
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
            f"{SCORE_NPZ}: pred_baseline has "
            f"{int((pred_baseline == EMPTY).sum()):,} EMPTY entries, "
            f"recorded n_empty is {n_empty:,}")
    if int((top5_ids[:, 0] == EMPTY).sum()) != n_empty:
        raise RuntimeError(
            f"{SCORE_NPZ}: top5_ids[:, 0] has "
            f"{int((top5_ids[:, 0] == EMPTY).sum()):,} EMPTY entries, "
            f"recorded n_empty is {n_empty:,}")

    # --- meta sidecar vs. this run's own constants: catches an eval run
    # against a score-stage npz built under a different floor/top-k/row
    # configuration ---
    floor_target, floor_target_provenance = _check_sidecar_floor_target(
        meta["floor_target"], SCORE_META)
    if meta["topk_margin"] != TOPK_MARGIN:
        raise RuntimeError(
            f"{SCORE_META} records topk_margin {meta['topk_margin']}, "
            f"expected TOPK_MARGIN {TOPK_MARGIN} "
            "(analysis.margin_diagnostic)")
    if meta["n_rows"] != n_rows:
        raise RuntimeError(f"{SCORE_META} records n_rows {meta['n_rows']:,}, "
                           f"but {SCORE_NPZ} has {n_rows:,} rows")
    if meta["n_tags"] != EXPECTED_TAGS:
        raise RuntimeError(f"{SCORE_META} records n_tags {meta['n_tags']}, "
                           f"expected EXPECTED_TAGS {EXPECTED_TAGS}")
    macro_tab_sha_now = _sha256_file(MACRO_TAB)
    if meta.get("macro_tab_sha256") != macro_tab_sha_now:
        raise RuntimeError(
            f"{MACRO_TAB} changed since the score stage: sha256 now "
            f"{macro_tab_sha_now}, recorded "
            f"{meta.get('macro_tab_sha256')} in {SCORE_META}; the "
            "macrolanguage table is a load-bearing scoring input")
    # Rank-order guard: _gate_walk_and_merge's margins and walk assume each
    # row's saved scores are in descending rank order.
    valid_rows = top5_ids[:, 0] >= 0
    if not (np.diff(top5_scores[valid_rows], axis=1) <= 0).all():
        raise RuntimeError(f"{SCORE_NPZ}: top5_scores are not in descending "
                           "rank order on scored rows; the banked candidate "
                           "arrays are corrupt or mis-ordered")

    m2M = _load_macro_member_to_macro()

    # --- canonical language order + training-line counts N (a memmap header
    # read, not a heavy model load) ---
    weights, langs, _lang_to_idx = _load_model_data(ctx.model_path)
    del weights
    prf = pd.read_csv(PRF_CSV)
    if prf.lang.tolist() != langs:
        raise RuntimeError(f"{PRF_CSV} language order does not match "
                           "_load_model_data's canonical language list")
    N = prf.N.values.astype(np.int64)

    os.makedirs(os.path.dirname(OUT_MD), exist_ok=True)
    os.makedirs(os.path.dirname(OUT_PER_TAG_CSV), exist_ok=True)

    # --- gate: floor-21 top-1 is top5_ids[:, 0] by construction; re-derive
    # gate_flat4_prox21 via the E2-reviewed shared helpers (imported, not
    # reimplemented) ---
    base_pred = top5_ids[:, 0]
    thresholds = _load_gate_thresholds(langs, N)
    tau1, sha_tau1 = thresholds["tau1"], thresholds["sha_tau1"]
    tau2, sha_tau2 = thresholds["tau2"], thresholds["sha_tau2"]
    step2_idx = thresholds["step2_idx"]
    step1_langs = thresholds["step1_langs"]
    step2_langs = thresholds["step2_langs"]
    # agree_mask left None: base_pred IS top5_ids[:, 0] by construction, so
    # every row trivially agrees with its own top-1 (the same convention
    # analysis/external_bench_eval.py's run_eval documents for the external
    # benchmarks).
    pred_gated, stats_a, stats_b = _gate_walk_and_merge(
        base_pred, top5_ids, top5_scores, N, tau1, tau2, step2_idx)

    def _config_scores(pred_idx: np.ndarray):
        """Maps a language-index prediction array to (accuracy, tag-level
        macro F1, y_pred_tag), following analysis/commonlid_eval.py's own
        macro_aware_acc / to_scored_tag convention exactly. i < 0 (EMPTY) maps
        to the empty iso string, which is never canonically correct, so EMPTY
        rows always score as wrong."""
        pi = np.array([langs[i].split("_")[0] if i >= 0 else "" for i in pred_idx])
        correct = np.array([_canonical_correct(a, g, m2M) for a, g in zip(pi, tags)])
        accuracy = float(correct.mean())
        y_pred_tag = np.array([g if c else m2M.get(p, p)
                               for p, g, c in zip(pi, tags, correct)])
        tag_f1 = float(compute_metrics(tags, y_pred_tag)["macro_f1"])
        return accuracy, tag_f1, y_pred_tag

    acc_baseline, tagf1_baseline, ypt_baseline = _config_scores(pred_baseline)
    acc_floor21, tagf1_floor21, ypt_floor21 = _config_scores(base_pred)

    # --- WIRING GATES: must reproduce the Exp 12/39 recorded values before
    # the gated configuration's numbers are computed (the E2 sibling's
    # ordering standard) ---
    #
    # Do they BIND this run? All four recorded values came from the released
    # (pre-0.3.0) weights, so only the released model can be held to them. Under
    # any other model the difference is the point of the run: the four
    # comparisons are still computed at the same EVAL_GATE_TOL and written into
    # the report in full, labelled INFORMATIONAL, NOT A GATE, and nothing below
    # is withheld. Same rule and same test as
    # analysis/external_bench_eval.py's `gates_binding`.
    gates_binding = ctx.is_default_model
    if gates_binding and abs(acc_baseline - EXPECTED_BASELINE_ACC) > EVAL_GATE_TOL:
        raise RuntimeError(
            f"recomputed baseline macro-aware accuracy {acc_baseline:.4f} "
            f"does not reproduce the recorded value {EXPECTED_BASELINE_ACC} "
            "(analysis.commonlid_carried.EXPECTED_BASELINE_ACC, Exp 12/39, "
            f"outputs/tables/commonlid_carried.md) within tolerance "
            f"{EVAL_GATE_TOL}")
    if gates_binding and abs(tagf1_baseline - EXPECTED_BASELINE_TAG_F1) > EVAL_GATE_TOL:
        raise RuntimeError(
            f"recomputed baseline tag-level macro F1 {tagf1_baseline:.4f} "
            f"does not reproduce the recorded value "
            f"{EXPECTED_BASELINE_TAG_F1} "
            "(analysis.commonlid_carried.EXPECTED_BASELINE_TAG_F1, Exp "
            f"12/39, outputs/tables/commonlid_carried.md) within tolerance "
            f"{EVAL_GATE_TOL}")
    if gates_binding and abs(tagf1_floor21 - EVAL_FLOOR21_TAG_F1) > EVAL_GATE_TOL:
        raise RuntimeError(
            f"recomputed floor-21 tag-level macro F1 {tagf1_floor21:.4f} "
            f"does not reproduce the recorded value {EVAL_FLOOR21_TAG_F1} "
            "(Exp 39, outputs/tables/commonlid_carried.md, \"floor21\" row) "
            f"within tolerance {EVAL_GATE_TOL}")
    if gates_binding and abs(acc_floor21 - EVAL_FLOOR21_ACC) > EVAL_GATE_TOL:
        raise RuntimeError(
            f"recomputed floor-21 macro-aware accuracy {acc_floor21:.4f} "
            f"does not reproduce the recorded value {EVAL_FLOOR21_ACC} "
            "(Exp 39, outputs/tables/commonlid_carried.md, \"floor21\" row) "
            f"within tolerance {EVAL_GATE_TOL}")

    # The run proceeds: every reproduction gate passed, or none of them binds
    # this model. Either way all four comparisons are recorded below.
    gate_comparisons = (
        ("baseline macro-aware accuracy", acc_baseline, EXPECTED_BASELINE_ACC,
         "analysis.commonlid_carried.EXPECTED_BASELINE_ACC, Exp 12/39"),
        ("baseline tag-level macro F1", tagf1_baseline,
         EXPECTED_BASELINE_TAG_F1,
         "analysis.commonlid_carried.EXPECTED_BASELINE_TAG_F1, Exp 12/39"),
        ("floor-21 tag-level macro F1", tagf1_floor21, EVAL_FLOOR21_TAG_F1,
         "Exp 39, outputs/tables/commonlid_carried.md, \"floor21\" row"),
        ("floor-21 macro-aware accuracy", acc_floor21, EVAL_FLOOR21_ACC,
         "Exp 39, outputs/tables/commonlid_carried.md, \"floor21\" row"),
    )
    if gates_binding:
        gate_block = [
            "\n## Wiring gates (reproduced before any new number was written)\n",
            f"- baseline macro-aware accuracy: {acc_baseline:.4f} vs recorded "
            f"{EXPECTED_BASELINE_ACC} "
            "(analysis.commonlid_carried.EXPECTED_BASELINE_ACC, Exp 12/39), "
            f"within {EVAL_GATE_TOL}.",
            f"- baseline tag-level macro F1: {tagf1_baseline:.4f} vs recorded "
            f"{EXPECTED_BASELINE_TAG_F1} "
            "(analysis.commonlid_carried.EXPECTED_BASELINE_TAG_F1, Exp 12/39), "
            f"within {EVAL_GATE_TOL}.",
            f"- floor-21 tag-level macro F1: {tagf1_floor21:.4f} vs recorded "
            f"{EVAL_FLOOR21_TAG_F1} (Exp 39, outputs/tables/"
            f"commonlid_carried.md, \"floor21\" row), within {EVAL_GATE_TOL}.",
        ]
    else:
        gate_block = ["\n## Recorded-value comparisons (informational)\n",
                      _cross_model_gate_message(ctx.model_path)]
        for name, measured, recorded, cite in gate_comparisons:
            d = measured - recorded
            verdict = ("agrees with" if abs(d) <= EVAL_GATE_TOL
                       else "differs from")
            gate_block.append(
                f"- {name}: {measured:.4f} {verdict} the recorded {recorded} "
                f"({cite}), difference {d:+.4f}, EVAL_GATE_TOL "
                f"{EVAL_GATE_TOL}.")
        for line in gate_block[1:]:
            print(line, flush=True)

    acc_gated, tagf1_gated, ypt_gated = _config_scores(pred_gated)

    # --- per-tag F1 (baseline, floor21, gated), restricted to the 109 gold
    # tags (the domain compute_metrics' macro F1 averages over) ---
    gold_tags_sorted = sorted(set(tags.tolist()))
    if len(gold_tags_sorted) != EXPECTED_TAGS:
        raise RuntimeError(f"{SCORE_NPZ} carries {len(gold_tags_sorted)} "
                           f"distinct gold tags, expected EXPECTED_TAGS "
                           f"{EXPECTED_TAGS}")

    # Out-of-tag-set accounting: a prediction whose tag-space label is outside
    # the gold tag set adds no F1 term of its own (compute_metrics averages
    # over labels present in the gold tags) but still deflates its gold tag's
    # recall. The volume is large enough to state (over half of all errors),
    # so the convention and the counts are reported with the table.
    gold_tag_set = set(gold_tags_sorted)
    out_of_set = {}
    for cfg_name, ypt in (("baseline", ypt_baseline), ("floor21", ypt_floor21),
                          ("gated", ypt_gated)):
        outside = np.array([t not in gold_tag_set for t in ypt])
        out_of_set[cfg_name] = (int(outside.sum()),
                                len(set(ypt[outside].tolist())))
    pc_baseline = compute_per_class_metrics(tags, ypt_baseline)
    pc_floor21 = compute_per_class_metrics(tags, ypt_floor21)
    pc_gated = compute_per_class_metrics(tags, ypt_gated)
    for cfg_name, pc in (("baseline", pc_baseline), ("floor21", pc_floor21),
                        ("gated", pc_gated)):
        missing = [t for t in gold_tags_sorted if t not in pc]
        if missing:
            raise RuntimeError(
                f"{cfg_name} per-class metrics are missing gold tag(s) "
                f"{missing[:5]}; wiring error in the tag-space mapping")
    support = Counter(tags.tolist())
    per_tag_df = pd.DataFrame({
        "tag": gold_tags_sorted,
        "support": [support[t] for t in gold_tags_sorted],
        "f1_baseline": [pc_baseline[t]["f1"] for t in gold_tags_sorted],
        "f1_floor21": [pc_floor21[t]["f1"] for t in gold_tags_sorted],
        "f1_gated": [pc_gated[t]["f1"] for t in gold_tags_sorted],
    })
    per_tag_df.to_csv(OUT_PER_TAG_CSV, index=False)

    # --- outputs ---
    metrics_rows = [
        ["baseline", f"{acc_baseline:.4f}", f"{tagf1_baseline:.4f}"],
        ["floor21", f"{acc_floor21:.4f}", f"{tagf1_floor21:.4f}"],
        ["gated", f"{acc_gated:.4f}", f"{tagf1_gated:.4f}"],
    ]
    metrics_md = to_markdown(
        metrics_rows, ["config", "macro-aware accuracy", "tag-level macro F1"],
        caption=f"CommonLID, {n_rows:,} lines, {len(gold_tags_sorted)} tags, "
                "web domain; macrolanguage-aware mapping per the recorded "
                "convention")

    exam_rows = [
        ["A (N < HEAD_N)", f"{stats_a['n_examined']:,}", f"{stats_a['n_moved']:,}",
         f"{stats_a['n_blocked_by_proximity']:,}", f"{stats_a['n_no_cand']:,}"],
        ["B (flat-four)", f"{stats_b['n_examined']:,}", f"{stats_b['n_moved']:,}",
         f"{stats_b['n_blocked_by_proximity']:,}", f"{stats_b['n_no_cand']:,}"],
    ]
    exam_md = to_markdown(
        exam_rows,
        ["group", "examined", "moved", "blocked_by_proximity", "no_cand"],
        caption="Re-examination accounting (gate_flat4_prox21, the promoted "
                "configuration's own gate, thresholds unchanged)")

    # No fallback: a camera-ready artifact without a resolvable commit is a
    # provenance failure, so _git_commit's error propagates.
    git_commit = _git_commit()

    L = [
        "# E5: CommonLID for the camera-ready (baseline / floor-21 / gated)\n",
        "Pre-registration: EXPERIMENTS_PLAN.md \"Camera-ready evaluation "
        "program\" (2026-08-07), E5. Gate: gate_flat4_prox21 (the promoted "
        "configuration), thresholds unchanged, applied via the shared "
        "_gate_walk_and_merge (the E2-reviewed code path, "
        "analysis/external_bench_eval.py, imported here, not reimplemented).\n",
        f"{n_rows:,} rows, {n_empty:,} rows empty after preprocess (scored "
        "as wrong under every configuration, the repo's EMPTY convention); "
        f"{meta['n_short_cands']:,} rows with fewer than {TOPK_MARGIN} saved "
        "candidates.\n",
        "Out-of-tag-set convention: a prediction mapping to a label outside "
        "the gold tag set contributes no per-tag F1 term of its own and "
        "deflates its gold tag's recall. Rows with an out-of-set label "
        "(distinct labels in parentheses): "
        + ", ".join(f"{c} {n:,} ({d})" for c, (n, d) in out_of_set.items())
        + ".\n",
        metrics_md,
        exam_md,
        "## Constants used\n",
        f"- FLOOR_TARGET = {floor_target} ({floor_target_provenance})",
        f"- TOPK_MARGIN = {TOPK_MARGIN} (analysis.margin_diagnostic)",
        f"- HEAD_N = {HEAD_N:,} (analysis.full_test_margin)",
        f"- RES_CAP = {RES_CAP:,} (analysis.hierarchical_pool)",
        f"- D3_PROX = {D3_PROX} (analysis.gate_variants)",
        f"- EVAL_GATE_TOL = {EVAL_GATE_TOL} (this module; E5 pre-registration)",
        f"- {_ebe._TAU1_CSV} (group A thresholds, {len(step1_langs):,} "
        f"languages, sha256 {sha_tau1[:16]}...)",
        f"- {_ebe._TAU2_CSV} (group B thresholds, {len(step2_langs)} "
        f"languages, sha256 {sha_tau2[:16]}...)",
    ] + gate_block + [
        f"\nPer-tag detail: {OUT_PER_TAG_CSV}.",
        f"\nGit commit: {git_commit}.",
    ]
    with open(OUT_MD, "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\nWrote {OUT_MD}, {OUT_PER_TAG_CSV}")
    print(f"EVAL OK: baseline acc {acc_baseline:.4f} / tagF1 "
         f"{tagf1_baseline:.4f}; floor21 acc {acc_floor21:.4f} / tagF1 "
         f"{tagf1_floor21:.4f}; gated acc {acc_gated:.4f} / tagF1 "
         f"{tagf1_gated:.4f}; group A {stats_a['n_examined']:,} examined / "
         f"{stats_a['n_moved']:,} moved; group B {stats_b['n_examined']:,} "
         f"examined / {stats_b['n_moved']:,} moved.")
    return OUT_MD


def main() -> None:
    parser = argparse.ArgumentParser(
        description="E5 CommonLID camera-ready evaluation: stage 'score' "
                    "loads the full model and must run on SLURM; stage "
                    "'eval' reads the scored .npz and runs on the login node.")
    parser.add_argument("--stage", required=True, choices=["score", "eval"])
    parser.add_argument("--out-dir", default=None,
                        help="repo-side output root; required when --model is "
                             "not the released model")
    from analysis.model_context import add_arguments
    add_arguments(parser)
    args = parser.parse_args()
    if args.stage == "score":
        run_score(model_path=args.model_path, scratch_dir=args.scratch_dir,
                  out_dir=args.out_dir)
    else:
        run_eval(model_path=args.model_path, scratch_dir=args.scratch_dir,
                 out_dir=args.out_dir)


if __name__ == "__main__":
    main()
