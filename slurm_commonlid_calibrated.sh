#!/bin/bash
#SBATCH --job-name=unilid-commonlid-calibrated
#SBATCH --account=infra01
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=100G
#SBATCH --time=01:00:00
#SBATCH --output=/capstor/scratch/cscs/cmeister747/unilid_analysis/logs/commonlid_calibrated_%j.out
#SBATCH --error=/capstor/scratch/cscs/cmeister747/unilid_analysis/logs/commonlid_calibrated_%j.err
# E5: CommonLID for the camera-ready (EXPERIMENTS_PLAN.md, "Camera-ready
# evaluation program", E5). Runs --stage score only (loads the full model);
# --stage eval runs on the login node separately, no model load, not launched
# here. No --uenv: this script does not use torch.distributed / NCCL, the
# same convention as slurm_external_bench.sh and slurm_commonlid_carried.sh.
# --time=01:00:00 is a wide margin over comparable CommonLID SLURM passes on
# this repo's own record (jobs 2640066, 2731818, 2898246, 2903415: 5m26s to
# 8m20s elapsed for baseline plus one or more floor-21-family passes over the
# same 373,230-line file).
set -euo pipefail
PYTHON="/users/cmeister747/.pyenv/versions/3.11.5/bin/python3"
REPO_DIR="/users/cmeister747/unilid_analysis"
export RAYON_NUM_THREADS=64
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"
cd "${REPO_DIR}"
echo "=== E5 CommonLID (camera-ready) scoring at $(date) on $(hostname) ==="
${PYTHON} -m analysis.commonlid_calibrated --stage score
echo "=== Finished at $(date) ==="
