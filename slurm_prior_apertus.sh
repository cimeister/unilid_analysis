#!/bin/bash
#SBATCH --job-name=unilid-prior-apertus
#SBATCH --account=infra01
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=400G
#SBATCH --time=04:00:00
#SBATCH --output=/capstor/scratch/cscs/cmeister747/unilid_analysis/logs/prior_apertus_%j.out
#SBATCH --error=/capstor/scratch/cscs/cmeister747/unilid_analysis/logs/prior_apertus_%j.err
set -euo pipefail
PYTHON="/users/cmeister747/.pyenv/versions/3.11.5/bin/python3"
export RAYON_NUM_THREADS=64
cd /users/cmeister747/unilid_analysis
echo "=== prior sweep on Apertus 200k model at $(date) ==="
${PYTHON} -c "
from analysis.prior_sweep import run
run(sample_size=500_000, model_path='/capstor/scratch/cscs/cmeister747/unilid_analysis/glotlid_apertus200k.unilid', tag='_apertus')
"
echo "=== Finished at $(date) ==="
