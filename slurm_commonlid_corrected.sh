#!/bin/bash
#SBATCH --job-name=unilid-commonlid-corr
#SBATCH --account=infra01
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=400G
#SBATCH --time=03:00:00
#SBATCH --output=/capstor/scratch/cscs/cmeister747/unilid_analysis/logs/commonlid_corrected_%j.out
#SBATCH --error=/capstor/scratch/cscs/cmeister747/unilid_analysis/logs/commonlid_corrected_%j.err
set -euo pipefail
#
# Both model-loading CommonLID stages for the corrected model, in one job.
#
# STAGE 1 analysis.commonlid_carried: baseline and floor-21 passes over the
#   373,230-line web-domain set, writing
#   outputs_corrected_round/diagnostic/commonlid_carried_preds.npz.
# STAGE 2 analysis.commonlid_calibrated --stage score: the same two passes
#   through the E2-reviewed scoring helpers, banking the top-5 candidates, with
#   a wiring gate that requires its baseline predictions to equal stage 1's
#   row for row and its floor-21 rank-1 to agree with stage 1's above the
#   module's band.
#
# One job rather than two with --dependency=afterok: stage 2 cannot start before
# stage 1's .npz exists, and on this queue a dependency-held job waits a second
# full cycle for its own allocation. Each stage ran 3m50s to 8m20s on this
# repo's record (jobs 2640066, 2731818, 2898246, 2903415, 3031609), so both fit
# one allocation with a wide margin.
#
# SLURM, not the login node: the released Exp 39 pass was OOM-killed at exit 137
# on a login node and resubmitted as job 2898246 (EXPERIMENTS_CHRONOLOGICAL.md,
# 2026-07-25), and a login-node attempt at the calibrated score stage was killed
# the same way while this script was being written.
#
# --configs baseline,floor21 for stage 1: the third released arm,
# gt_margin_adaptive, needs outputs/diagnostic/tau_gt_margin_adaptive.csv
# (thresholds fit on the released weights), outputs/diagnostic/gt_counts.csv and
# fingerprint_gt.json, none of which exists for this model, and
# analysis.full_test_gt.build_gt_weights finds the special columns by the
# SPECIAL_P = 0.2 probability the token defect produced. It feeds no paper cell.
# Naming it here would abort, by design.
#
# The clamp constant is read from fingerprint_floor21.json (c = -17) rather than
# the module default, so neither stage can be built at a different constant than
# the full-pool predictions and the tau CSVs they are gated by.
#
# Stage "eval" of analysis.commonlid_calibrated loads no model and runs on the
# login node afterwards, as it does for the released model.
#
# PARTITION. The header keeps --partition=normal, this repo's convention for the
# corrected round. The 2026-08-24 run was submitted with the override
#   sbatch --partition=debug --time=01:20:00 slurm_commonlid_corrected.sh
# because `normal` was 713 jobs deep that afternoon (sbatch --test-only put the
# start at 20:59 against 16:21 on `debug`, same node type, 288 cores either way)
# and both stages together need about 20 minutes, well inside debug's 1:30:00
# cap. Partition affects scheduling only; nothing it changes reaches a number.
PYTHON="/users/cmeister747/.pyenv/versions/3.11.5/bin/python3"
REPO_DIR="/users/cmeister747/unilid_analysis"
MODEL="/capstor/scratch/cscs/cmeister747/unilid_analysis/corrected/glotlidc_corrected.unilid"
SCRATCH="/capstor/scratch/cscs/cmeister747/unilid_analysis/full_test_eval_corrected"
OUT_DIR="outputs_corrected_round"
export RAYON_NUM_THREADS=64
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"
cd "${REPO_DIR}"
echo "=== corrected CommonLID, both scoring stages, at $(date) on $(hostname) ==="
echo "commit: $(git rev-parse HEAD)"

echo "--- stage 1/2: commonlid_carried (baseline, floor21) at $(date) ---"
${PYTHON} -u -m analysis.commonlid_carried \
    --model "${MODEL}" --scratch-dir "${SCRATCH}" --out-dir "${OUT_DIR}" \
    --configs baseline,floor21

echo "--- stage 2/2: commonlid_calibrated --stage score at $(date) ---"
${PYTHON} -u -m analysis.commonlid_calibrated --stage score \
    --model "${MODEL}" --scratch-dir "${SCRATCH}" --out-dir "${OUT_DIR}"

echo "=== Finished at $(date) ==="
