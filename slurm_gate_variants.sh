#!/bin/bash
#SBATCH --job-name=unilid-gate-variants
#SBATCH --account=infra01
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=100G
#SBATCH --time=03:00:00
#SBATCH --output=/capstor/scratch/cscs/cmeister747/unilid_analysis/logs/gate_variants_%j.out
#SBATCH --error=/capstor/scratch/cscs/cmeister747/unilid_analysis/logs/gate_variants_%j.err
set -euo pipefail
PYTHON="/users/cmeister747/.pyenv/versions/3.11.5/bin/python3"
export RAYON_NUM_THREADS=64
cd /users/cmeister747/unilid_analysis
echo "=== gate_variants stage topk at $(date) on $(hostname) ==="
${PYTHON} -m analysis.gate_variants topk
echo "=== gate_variants stage apply (shared9_bar18k) at $(date) ==="
${PYTHON} -m analysis.gate_variants apply shared9_bar18k
echo "=== Finished at $(date) ==="
