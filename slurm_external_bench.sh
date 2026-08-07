#!/bin/bash
#SBATCH --job-name=unilid-external-bench
#SBATCH --account=infra01
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=100G
#SBATCH --time=02:00:00
#SBATCH --output=/capstor/scratch/cscs/cmeister747/unilid_analysis/logs/external_bench_%j.out
#SBATCH --error=/capstor/scratch/cscs/cmeister747/unilid_analysis/logs/external_bench_%j.err
set -euo pipefail
PYTHON="/users/cmeister747/.pyenv/versions/3.11.5/bin/python3"
REPO_DIR="/users/cmeister747/unilid_analysis"
export RAYON_NUM_THREADS=64
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"
cd "${REPO_DIR}"
echo "=== E2 external-benchmark scoring (UDHR) at $(date) on $(hostname) ==="
${PYTHON} -m analysis.external_bench_eval --stage score --bench udhr
echo "=== E2 external-benchmark scoring (FLORES-200) at $(date) on $(hostname) ==="
${PYTHON} -m analysis.external_bench_eval --stage score --bench flores
echo "=== Finished at $(date) ==="
