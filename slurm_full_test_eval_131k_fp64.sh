#!/bin/bash
#SBATCH --job-name=unilid-ft-131k-fp64
#SBATCH --account=infra01
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=100G
#SBATCH --time=08:00:00
#SBATCH --output=/capstor/scratch/cscs/cmeister747/unilid_analysis/logs/ft_131k_fp64_%j.out
#SBATCH --error=/capstor/scratch/cscs/cmeister747/unilid_analysis/logs/ft_131k_fp64_%j.err
set -euo pipefail
PYTHON="/users/cmeister747/.pyenv/versions/3.11.5/bin/python3"
export RAYON_NUM_THREADS=64
SCR="/capstor/scratch/cscs/cmeister747/unilid_analysis"
cd /users/cmeister747/unilid_analysis
echo "=== 131k_fp64 full-test baseline (clean re-measurement) at $(date) on $(hostname) ==="
${PYTHON} -c "
from analysis.full_test_eval_131k import run
run(model_path='${SCR}/glotlid_apertus131k_fp64.unilid',
    scratch_dir='${SCR}/full_test_eval_131k_fp64',
    out_md='outputs/tables/full_test_eval_131k_fp64.md',
    out_csv='outputs/diagnostic/full_test_131k_fp64_per_lang_prf.csv',
    model_label='131k_fp64')
"
echo "=== Finished at $(date) ==="
