#!/bin/bash
#SBATCH --job-name=unilid-margin-diag
#SBATCH --account=infra01
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=100G
#SBATCH --time=02:00:00
#SBATCH --output=/capstor/scratch/cscs/cmeister747/unilid_analysis/logs/margin_diag_%j.out
#SBATCH --error=/capstor/scratch/cscs/cmeister747/unilid_analysis/logs/margin_diag_%j.err
set -euo pipefail
PYTHON="/users/cmeister747/.pyenv/versions/3.11.5/bin/python3"
export RAYON_NUM_THREADS=64
cd /users/cmeister747/unilid_analysis
echo "=== Margin diagnostic (plan B3) at $(date) on $(hostname) ==="
${PYTHON} -m analysis.margin_diagnostic
echo "=== Finished at $(date) ==="
