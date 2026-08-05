#!/bin/bash
#SBATCH --job-name=unilid-full-test-bgfloor
#SBATCH --account=infra01
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=100G
#SBATCH --time=08:00:00
#SBATCH --output=/capstor/scratch/cscs/cmeister747/unilid_analysis/logs/full_test_bgfloor_%j.out
#SBATCH --error=/capstor/scratch/cscs/cmeister747/unilid_analysis/logs/full_test_bgfloor_%j.err
set -euo pipefail
PYTHON="/users/cmeister747/.pyenv/versions/3.11.5/bin/python3"
export RAYON_NUM_THREADS=64
cd /users/cmeister747/unilid_analysis
echo "=== full-test bgfloor evaluation (Exp 50) at $(date) on $(hostname) ==="
${PYTHON} -c "from analysis.full_test_bgfloor import run; run()"
echo "=== Finished at $(date) ==="
