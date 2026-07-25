#!/bin/bash
#SBATCH --job-name=unilid-tying-dp
#SBATCH --account=infra01
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=100G
#SBATCH --time=04:00:00
#SBATCH --output=/capstor/scratch/cscs/cmeister747/unilid_analysis/logs/tying_dp_%j.out
#SBATCH --error=/capstor/scratch/cscs/cmeister747/unilid_analysis/logs/tying_dp_%j.err
set -euo pipefail
PYTHON="/users/cmeister747/.pyenv/versions/3.11.5/bin/python3"
export RAYON_NUM_THREADS=64
cd /users/cmeister747/unilid_analysis
echo "=== digit/neutral-punctuation tying at $(date) on $(hostname) ==="
${PYTHON} -c "from analysis.token_tying import run; run(sample_size=500_000, sweep=('dp_global', 'dp_script'), tag='_dp')"
echo "=== Finished at $(date) ==="
