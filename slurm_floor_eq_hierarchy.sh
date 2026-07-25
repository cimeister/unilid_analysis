#!/bin/bash
#SBATCH --job-name=unilid-flooreq-hier
#SBATCH --account=infra01
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=100G
#SBATCH --time=04:00:00
#SBATCH --output=/capstor/scratch/cscs/cmeister747/unilid_analysis/logs/flooreq_hier_%j.out
#SBATCH --error=/capstor/scratch/cscs/cmeister747/unilid_analysis/logs/flooreq_hier_%j.err
set -euo pipefail
PYTHON="/users/cmeister747/.pyenv/versions/3.11.5/bin/python3"
export RAYON_NUM_THREADS=64
cd /users/cmeister747/unilid_analysis
echo "=== downward floor equalization at $(date) on $(hostname) ==="
${PYTHON} -c "from analysis.floor_equalization import run; run(sample_size=500_000)"
echo "=== macrolanguage hierarchy at $(date) ==="
${PYTHON} -c "from analysis.macro_hierarchy import run; run(sample_size=500_000)"
echo "=== Finished at $(date) ==="
