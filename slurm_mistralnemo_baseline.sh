#!/bin/bash
#SBATCH --job-name=unilid-mistralnemo-baseline
#SBATCH --account=infra01
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=100G
#SBATCH --time=09:00:00
#SBATCH --output=/capstor/scratch/cscs/cmeister747/unilid_analysis/logs/mistralnemo_baseline_%j.out
#SBATCH --error=/capstor/scratch/cscs/cmeister747/unilid_analysis/logs/mistralnemo_baseline_%j.err
set -euo pipefail
PYTHON="/users/cmeister747/.pyenv/versions/3.11.5/bin/python3"
export RAYON_NUM_THREADS=64
cd /users/cmeister747/unilid_analysis

# E3 pre-registration (EXPERIMENTS_PLAN.md, "Camera-ready evaluation program
# (2026-08-06)", E3). Runs STAGE baseline (one full-pool scoring pass of the
# Mistral-Nemo variant under its unmodified weight matrix) then STAGE
# calibval (scoring on the retired 250,000-line validation half only,
# feeding the flat-language rule's magnet_ratio) sequentially in one job, as
# pre-registered. Time budget: full_test_eval_131k_fp64.sh's precedent full-
# pool baseline pass (same vocab size, 131,072) measured 2h06m against an
# 8h budget (job 2911700); this job also runs calibval afterward, so the
# budget is padded to 9h.

echo "=== mistralnemo STAGE baseline at $(date) on $(hostname) ==="
${PYTHON} -m analysis.mistralnemo_eval --stage baseline

echo "=== mistralnemo STAGE calibval at $(date) ==="
${PYTHON} -m analysis.mistralnemo_eval --stage calibval

echo "=== Finished at $(date) ==="
