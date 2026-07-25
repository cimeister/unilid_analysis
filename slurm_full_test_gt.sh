#!/bin/bash
#SBATCH --job-name=unilid-full-test-gt
#SBATCH --account=infra01
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=100G
#SBATCH --time=06:00:00
#SBATCH --output=/capstor/scratch/cscs/cmeister747/unilid_analysis/logs/full_test_gt_%j.out
#SBATCH --error=/capstor/scratch/cscs/cmeister747/unilid_analysis/logs/full_test_gt_%j.err
set -euo pipefail
PYTHON="/users/cmeister747/.pyenv/versions/3.11.5/bin/python3"
export RAYON_NUM_THREADS=64
cd /users/cmeister747/unilid_analysis
echo "=== gt_min full-test pass (plan B4) at $(date) on $(hostname) ==="
${PYTHON} -m analysis.full_test_gt
echo "=== Finished at $(date) ==="
