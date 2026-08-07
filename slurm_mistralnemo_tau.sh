#!/bin/bash
#SBATCH --job-name=unilid-mistralnemo-tau
#SBATCH --account=infra01
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=100G
#SBATCH --time=04:00:00
#SBATCH --output=/capstor/scratch/cscs/cmeister747/unilid_analysis/logs/mistralnemo_tau_%j.out
#SBATCH --error=/capstor/scratch/cscs/cmeister747/unilid_analysis/logs/mistralnemo_tau_%j.err
set -euo pipefail
PYTHON="/users/cmeister747/.pyenv/versions/3.11.5/bin/python3"
export RAYON_NUM_THREADS=64
cd /users/cmeister747/unilid_analysis

# E3 pre-registration (EXPERIMENTS_PLAN.md, "Camera-ready evaluation program
# (2026-08-06)", E3). Runs STAGE tau: builds the Mistral-Nemo variant's
# floor-21 matrix, writes fingerprint_floor21_mistralnemo.json, and
# recalibrates per-language tau under that matrix for both group A (N <
# HEAD_N, size-adaptive quantile) and group B (the variant's own flat set
# from STAGE flatrule, fixed 5th percentile). Requires STAGE flatrule to
# have already written outputs/diagnostic/mistralnemo_flat_set.csv (login
# node, not run here). Time budget matches the repo's existing per-language
# calibration jobs (analysis/solo_gates.py, analysis/gt_margin.py; e.g.
# slurm_gt_margin_adaptive.sh: 64 CPU, 100G, 4h).

echo "=== mistralnemo STAGE tau at $(date) on $(hostname) ==="
${PYTHON} -m analysis.mistralnemo_eval --stage tau

echo "=== Finished at $(date) ==="
