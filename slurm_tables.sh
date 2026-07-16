#!/bin/bash
#SBATCH --job-name=unilid-tables
#SBATCH --account=a139
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/capstor/scratch/cscs/cmeister747/unilid_analysis/logs/tables_%j.out
#SBATCH --error=/capstor/scratch/cscs/cmeister747/unilid_analysis/logs/tables_%j.err

PYTHON="/users/cmeister747/.pyenv/versions/3.11.5/bin/python3"
WORKDIR="/users/cmeister747/unilid_analysis"

cd "${WORKDIR}"

echo "=== Starting tables job at $(date) ==="
echo "Node: $(hostname)"

${PYTHON} -m analysis.run_all \
    --sample-size 45627279 \
    --format both \
    --output-dir outputs

echo "=== Finished at $(date) ==="
