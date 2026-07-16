#!/bin/bash
#SBATCH --job-name=unilid-commonlid
#SBATCH --account=infra01
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=400G
#SBATCH --time=03:00:00
#SBATCH --output=/capstor/scratch/cscs/cmeister747/unilid_analysis/logs/commonlid_%j.out
#SBATCH --error=/capstor/scratch/cscs/cmeister747/unilid_analysis/logs/commonlid_%j.err

set -euo pipefail
PYTHON="/users/cmeister747/.pyenv/versions/3.11.5/bin/python3"
WORKDIR="/users/cmeister747/unilid_analysis"
export RAYON_NUM_THREADS=64
cd "${WORKDIR}"

echo "=== CommonLID eval at $(date) on $(hostname) ==="
${PYTHON} -c "
from analysis.commonlid_eval import run
run()
"
echo "=== Finished at $(date) ==="
