#!/bin/bash
#SBATCH --job-name=unilid-gt-margin-adapt
#SBATCH --account=infra01
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=100G
#SBATCH --time=04:00:00
#SBATCH --output=/capstor/scratch/cscs/cmeister747/unilid_analysis/logs/gt_margin_adapt_%j.out
#SBATCH --error=/capstor/scratch/cscs/cmeister747/unilid_analysis/logs/gt_margin_adapt_%j.err
set -euo pipefail
PYTHON="/users/cmeister747/.pyenv/versions/3.11.5/bin/python3"
export RAYON_NUM_THREADS=64
cd /users/cmeister747/unilid_analysis
echo "=== gt_margin_adaptive build (Exp 35 pre-registration) at $(date) on $(hostname) ==="
${PYTHON} -c "from analysis.gt_margin import run; run(gate='nonhead', target_n=100_000, adaptive_q=True)"
echo "=== Finished at $(date) ==="
