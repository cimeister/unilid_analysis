#!/bin/bash
#SBATCH --job-name=unilid-bal-sweeps
#SBATCH --account=infra01
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=100G
#SBATCH --time=04:00:00
#SBATCH --output=/capstor/scratch/cscs/cmeister747/unilid_analysis/logs/bal_sweeps_%j.out
#SBATCH --error=/capstor/scratch/cscs/cmeister747/unilid_analysis/logs/bal_sweeps_%j.err
set -euo pipefail
PYTHON="/users/cmeister747/.pyenv/versions/3.11.5/bin/python3"
export RAYON_NUM_THREADS=64
cd /users/cmeister747/unilid_analysis
echo "=== balanced-protocol sweeps at $(date) on $(hostname) ==="
${PYTHON} -c "from analysis.balanced_sweeps import run_floor; run_floor()"
echo "=== punct prior at $(date) ==="
${PYTHON} -c "from analysis.balanced_sweeps import run_punct_prior; run_punct_prior()"
echo "=== bias refit at $(date) ==="
${PYTHON} -c "from analysis.balanced_sweeps import run_bias_refit; run_bias_refit()"
echo "=== Finished at $(date) ==="
