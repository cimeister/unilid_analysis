#!/bin/bash
#SBATCH --job-name=unilid-nemo-stages-corr
#SBATCH --account=infra01
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=100G
#SBATCH --time=12:00:00
#SBATCH --output=/capstor/scratch/cscs/cmeister747/unilid_analysis/logs/nemo_stages_corrected_%j.out
#SBATCH --error=/capstor/scratch/cscs/cmeister747/unilid_analysis/logs/nemo_stages_corrected_%j.err
set -euo pipefail
#
# Mistral-Nemo variant, corrected weights: the five stages left after
# slurm_mistralnemo_baseline_corrected.sh (--stage baseline, already
# COMPLETED: full_test_eval_mistralnemo_corrected/{fingerprint_baseline.json,
# pred_nemo_baseline.npy,progress_baseline.json} all present).
#
# Run in ONE job, sequentially, matching the released chain's own stage
# order (analysis/mistralnemo_eval.py docstring): calibval, flatrule, tau,
# topk, eval. Bundled into one script rather than the released chain's split
# (slurm_mistralnemo_tau.sh / slurm_mistralnemo_topk.sh separate; flatrule
# and eval run by hand on the login node) because all five stages share an
# identical resource envelope (single node, 64 CPU, comfortably under 100G:
# the released chain's own baseline+calibval, tau, and topk jobs were ALL
# 100G/64CPU) -- no stage here is an order of magnitude heavier than
# another, so splitting them into separate sbatch submissions would only add
# manual hand-off steps without saving node time. flatrule and eval are
# "login node" stages by convention (cheap: in-memory over already-cached
# arrays, or bootstrap resampling) but running them inside this allocation
# is harmless -- a few minutes of idle-CPU cost on a node that is already
# up for the SLURM stages either side of them.
#
# Time budget derivation (EXPERIMENTS_CHRONOLOGICAL.md, "2026-08-08"/
# "2026-08-09" entries, released-model chain, same architecture/vocab size,
# so the corrected weights should cost the same order of magnitude):
#   calibval: folded into job 3032625 (baseline+calibval combined,
#     COMPLETED 2h14m for a FULL 45.6M-line baseline pass + calibval); the
#     baseline pass dominates that number and is NOT part of this job, so
#     calibval alone (250,000 lines) is budgeted 1h, generous.
#   flatrule: login node, in-memory; budgeted with eval below.
#   tau: job 3036829, matches slurm_mistralnemo_tau.sh's own budget, 4h.
#   topk: job 3038358, COMPLETED 3h13m against an 8h budget
#     (slurm_mistralnemo_topk.sh); kept at 8h here.
#   eval: login node; budgeted with flatrule, 1h combined for both.
#   Total: 1 + 4 + 8 + 1 = 14h; padded to 16h.
#
# --model/--scratch-dir/--base-scratch mirror
# slurm_mistralnemo_baseline_corrected.sh exactly: the corrected packed
# model, a fresh (non-store-backed) scratch root for this variant, and the
# corrected base model's scratch dir for the read-only y_true.npy reuse
# (verified bit-identical to the released model's over all 45,627,279
# entries, 2026-08-18).
#
# --out-dir and --floor-target are the two flags analysis/mistralnemo_eval.py
# gained on 2026-08-23, when the two structural defects this script's earlier
# revision could only abort on were fixed in that module itself:
#   --out-dir routes the nine repo-side artifacts (flat set CSV/MD, both tau
#     CSVs, the tau build note, the eval .md/.tex, the two per-language F1 CSVs)
#     through analysis.model_context.resolve_out_root, so this run writes beside
#     the corrected base chain's own tables instead of over the RELEASED model's
#     published E3 record. outputs_corrected_round is the root the corrected base
#     chain already used, and the Mistral-Nemo file names do not collide with
#     anything in it (every one is mistralnemo_-prefixed).
#   --floor-target replaces the module-level import of
#     analysis.full_test_floor21.FLOOR_TARGET. -17.0 is the corrected chain's own
#     round-grid sweep result, recorded in
#     ${BASE_SCRATCH}/fingerprint_floor21.json (2026-08-18); the module
#     cross-checks the flag against that file and aborts naming both numbers on
#     any disagreement, so the value below cannot silently drift from its record.
PYTHON="/users/cmeister747/.pyenv/versions/3.11.5/bin/python3"
export RAYON_NUM_THREADS=64
cd /users/cmeister747/unilid_analysis
SCR="/capstor/scratch/cscs/cmeister747/unilid_analysis"
MODEL="${SCR}/corrected/glotlid_mistralnemo_fp64_corrected.unilid"
SCRATCH_NEMO="${SCR}/full_test_eval_mistralnemo_corrected"
BASE_SCRATCH="${SCR}/full_test_eval_corrected"
OUT_DIR="outputs_corrected_round"
FLOOR_TARGET="-17.0"

echo "=== Mistral-Nemo corrected stages job at $(date) on $(hostname) ==="
echo "commit: $(git rev-parse HEAD)"
# The tree may be ahead of HEAD (uncommitted parametrization + clamp fixes,
# 2026-08-23); record the delta so the log's provenance is exact.
echo "dirty files: $(git status --short | tr '\n' ' ')"
echo "mistralnemo_eval.py sha256: $(sha256sum analysis/mistralnemo_eval.py | cut -d' ' -f1)"

# ---------------------------------------------------------------------------
# Preflight. Fails loudly, naming the exact missing/blocking artifact,
# before any stage runs -- SLURM node-hours are not spent on a run that
# cannot complete. Two classes of check:
#  (a) artifacts a stage reads that must already exist (chain-order gates
#      analysis/mistralnemo_eval.py itself also asserts at runtime -- these
#      preflight checks are redundant with those, deliberately, so a
#      misconfigured submission dies in seconds instead of after loading the
#      64-CPU model);
#  (b) the two hazards that used to be structural defects in
#      analysis/mistralnemo_eval.py (bare "outputs/..." literals, and a
#      hardcoded FLOOR_TARGET). Both are fixed in that module now and are
#      enforced there; the checks below are ordinary preflights that catch a
#      mis-edited submission in seconds rather than after the model load.
# ---------------------------------------------------------------------------

echo "--- preflight: shared inputs ---"
for p in "${MODEL}" "${BASE_SCRATCH}/y_true.npy" \
         "${SCRATCH_NEMO}/fingerprint_baseline.json" \
         "${SCRATCH_NEMO}/pred_nemo_baseline.npy"; do
    if [ ! -e "${p}" ]; then
        echo "ABORT: required artifact missing: ${p}" >&2
        exit 1
    fi
done
echo "OK: corrected model, base y_true.npy, and the completed baseline "
echo "    stage's outputs are all present."

# --- verified dependency scope (recorded here so a future reader does not
# have to re-derive it; re-checked 2026-08-23 after --out-dir/--floor-target
# were added): grepping analysis/mistralnemo_eval.py shows BASE_SCRATCH is read
# for exactly TWO files across all six stages --
#   y_true.npy (lines 718, 844, 1694 as of this writing), the label indices, and
#   fingerprint_floor21.json (line 471, _resolve_floor_target), opened for its
#     `floor_target` field alone and only to cross-check --floor-target. This is
#     NEW: the earlier revision of this comment correctly said that file was not
#     read, and the floor-target fix is what changed it.
# None of pred_baseline.npy, pred_floor21.npy, pred_floor21_gate.npy or
# gate_topk_*.npy under full_test_eval_corrected/ is read anywhere in this
# module, and neither is outputs_corrected_round/diagnostic/tau_flat4.csv or
# pred_gate_flat4_prox21.npy (job 3157817's own outputs, for the corrected
# GlotLID-C base model's OWN promoted-gate evaluation -- an unrelated
# pipeline: mistralnemo_eval.py builds and calibrates its own floor-21
# matrix and its own tau CSVs from scratch, it never reads the base model's
# gate artifacts). This job therefore does NOT wait on job 3157817, but it DOES
# now require full_test_eval_corrected/fingerprint_floor21.json to already
# exist, which it does (2026-08-18). If that premise changes (a future edit to
# mistralnemo_eval.py starts reading one of those files), this preflight will
# NOT catch it -- update it then.

echo "--- preflight: the corrected degeneracy scan the eval stage reads ---"
# analysis/mistralnemo_eval.py's eval stage reads
# <out-dir>/tables/degenerate_rows_mistralnemo.md for its degeneracy caveat.
# That file is a scan of the model's OWN weight rows (model-derived), so with
# --out-dir it is looked for under ${OUT_DIR} and is NEVER read from the released
# tree while corrected weights are scored -- the stage aborts naming the missing
# path instead. analysis/degeneracy_scan_mistralnemo.py takes --model/--out-md
# (added 2026-08-23) and the corrected scan was produced with them the same day
# (34 flagged rows under the real-column definition; see the chronology entry).
# Checked here so the four cheap stages are not spent before the fifth
# discovers it.
DEGEN_MD="${OUT_DIR}/tables/degenerate_rows_mistralnemo.md"
if [ ! -e "${DEGEN_MD}" ]; then
    echo "ABORT: ${DEGEN_MD} is missing." >&2
    echo "  The eval stage reads it for the degeneracy caveat and will not fall" >&2
    echo "  back to the released model's copy in outputs/tables/. Produce the" >&2
    echo "  corrected scan first (python3 -m analysis.degeneracy_scan_mistralnemo" >&2
    echo "  --model <corrected .unilid> --out-md ${DEGEN_MD}), or drop --stage eval" >&2
    echo "  from this job and run it separately once the scan exists." >&2
    exit 1
fi
echo "OK: ${DEGEN_MD} present."

echo "--- preflight: repo-side output root (was: outputs/ collision with the RELEASED record) ---"
# The nine repo-side artifacts of the flatrule/tau/eval stages used to be bare
# "outputs/..." literals in analysis/mistralnemo_eval.py, so this job would have
# overwritten the RELEASED model's own E3 record in place -- outputs/tables/
# mistralnemo_eval.md is cited directly in the paper (paper/tables/
# calibrated_nemo.tex, per EXPERIMENTS_CHRONOLOGICAL.md "2026-08-09").
#
# They now resolve under --out-dir through resolve_out_root, which REFUSES a
# non-default model paired with the default root, with anything inside it, or
# with a store-backed root: the corrected weights physically cannot write to the
# paths below any more. This check is therefore no longer the thing standing
# between the job and the published record; it stays as a cheap assertion that
# the released record is still where it belongs (a missing file here means
# someone moved or clobbered it outside this job) and that ${OUT_DIR} is not
# somehow the default root.
if [ "${OUT_DIR}" = "outputs" ]; then
    echo "ABORT: OUT_DIR is the default output root; that is the released" >&2
    echo "  model's published record. Point it at a separate tree." >&2
    exit 1
fi
OUTPUTS_COLLIDE=(
    "outputs/diagnostic/mistralnemo_flat_set.csv"
    "outputs/tables/mistralnemo_flat_set.md"
    "outputs/diagnostic/tau_mistralnemo_floor21_gate.csv"
    "outputs/diagnostic/tau_mistralnemo_flat.csv"
    "outputs/tables/mistralnemo_tau_build.md"
    "outputs/tables/mistralnemo_eval.md"
    "outputs/tables/mistralnemo_eval.tex"
    "outputs/diagnostic/mistralnemo_per_lang_f1_fullpool.csv"
    "outputs/diagnostic/mistralnemo_per_lang_f1_judge.csv"
)
missing=()
for p in "${OUTPUTS_COLLIDE[@]}"; do
    [ -e "${p}" ] || missing+=("${p}")
done
if [ "${#missing[@]}" -gt 0 ]; then
    echo "WARNING: ${#missing[@]} of the 9 RELEASED-model E3 artifacts are absent:" >&2
    for p in "${missing[@]}"; do echo "    ${p}" >&2; done
    echo "  This job does not touch them (its writes go to ${OUT_DIR}), so their" >&2
    echo "  absence was caused by something else. Not fatal here." >&2
fi
echo "OK: this job's repo-side writes go to ${OUT_DIR}, and the corrected model"
echo "    is structurally barred from the default output root by resolve_out_root."

echo "--- preflight: the floor-21 clamp target ---"
# analysis/mistralnemo_eval.py used to import analysis.full_test_floor21.
# FLOOR_TARGET (-21.0, the RELEASED chain's Exp 20 guard-selected constant) and
# use it unchanged in _build_and_fingerprint_floor21_nemo for the tau and topk
# stages, with no CLI override: a corrected run would have built and calibrated
# against the wrong clamp and produced floor-21/gated numbers indistinguishable
# in the tables from a correctly-targeted run. The module now takes
# --floor-target, REQUIRES it for any non-default --model, and cross-checks it
# against ${BASE_SCRATCH}/fingerprint_floor21.json before any matrix is built.
#
# This preflight reproduces that cross-check in shell so a wrong FLOOR_TARGET in
# this script dies before the queue, and so the value is visible in the job log.
BASE_FP="${BASE_SCRATCH}/fingerprint_floor21.json"
if [ ! -e "${BASE_FP}" ]; then
    echo "ABORT: ${BASE_FP} is missing; it is the record of the round-grid" >&2
    echo "  floor sweep that selected the corrected clamp, and the module will" >&2
    echo "  refuse to proceed on an unverified --floor-target." >&2
    exit 1
fi
RECORDED_FLOOR=$(${PYTHON} -c "import json,sys; print(float(json.load(open(sys.argv[1]))['floor_target']))" "${BASE_FP}")
if [ "${RECORDED_FLOOR}" != "${FLOOR_TARGET}" ]; then
    echo "ABORT: this script's FLOOR_TARGET=${FLOOR_TARGET} disagrees with the" >&2
    echo "  value recorded by the corrected base model's own floor sweep:" >&2
    echo "    ${BASE_FP} records floor_target=${RECORDED_FLOOR}" >&2
    echo "  One of the two is wrong. Do not proceed until that is settled." >&2
    exit 1
fi
echo "OK: floor target ${FLOOR_TARGET}, matching ${BASE_FP}."

# ---------------------------------------------------------------------------
# Stages. Every invocation carries the full five-flag context: a stage that
# dropped --out-dir or --floor-target would be refused by the module rather
# than falling back to the released chain's root or clamp.
# ---------------------------------------------------------------------------

echo "=== STAGE calibval at $(date) ==="
${PYTHON} -u -m analysis.mistralnemo_eval --stage calibval \
    --model "${MODEL}" --scratch-dir "${SCRATCH_NEMO}" --base-scratch "${BASE_SCRATCH}" \
    --out-dir "${OUT_DIR}" --floor-target "${FLOOR_TARGET}"

echo "=== STAGE flatrule at $(date) ==="
${PYTHON} -u -m analysis.mistralnemo_eval --stage flatrule \
    --model "${MODEL}" --scratch-dir "${SCRATCH_NEMO}" --base-scratch "${BASE_SCRATCH}" \
    --out-dir "${OUT_DIR}" --floor-target "${FLOOR_TARGET}"

echo "=== STAGE tau at $(date) ==="
${PYTHON} -u -m analysis.mistralnemo_eval --stage tau \
    --model "${MODEL}" --scratch-dir "${SCRATCH_NEMO}" --base-scratch "${BASE_SCRATCH}" \
    --out-dir "${OUT_DIR}" --floor-target "${FLOOR_TARGET}"

echo "=== STAGE topk at $(date) ==="
${PYTHON} -u -m analysis.mistralnemo_eval --stage topk \
    --model "${MODEL}" --scratch-dir "${SCRATCH_NEMO}" --base-scratch "${BASE_SCRATCH}" \
    --out-dir "${OUT_DIR}" --floor-target "${FLOOR_TARGET}"

echo "=== STAGE eval at $(date) ==="
${PYTHON} -u -m analysis.mistralnemo_eval --stage eval \
    --model "${MODEL}" --scratch-dir "${SCRATCH_NEMO}" --base-scratch "${BASE_SCRATCH}" \
    --out-dir "${OUT_DIR}" --floor-target "${FLOOR_TARGET}"

echo "=== Finished at $(date) ==="
