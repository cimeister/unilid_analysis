#!/bin/bash
#SBATCH --job-name=unilid-learnprior
#SBATCH --account=infra01
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=400G
#SBATCH --time=04:00:00
#SBATCH --output=/capstor/scratch/cscs/cmeister747/unilid_analysis/logs/learnprior_%j.out
#SBATCH --error=/capstor/scratch/cscs/cmeister747/unilid_analysis/logs/learnprior_%j.err
set -euo pipefail
PYTHON="/users/cmeister747/.pyenv/versions/3.11.5/bin/python3"
export RAYON_NUM_THREADS=64
cd /users/cmeister747/unilid_analysis
echo "=== learned per-language bias at $(date) on $(hostname) ==="
${PYTHON} -c "from analysis.learned_prior import run; run(sample_size=500_000)"
echo "=== Finished at $(date) ==="
