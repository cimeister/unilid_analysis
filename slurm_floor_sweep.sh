#!/bin/bash
#SBATCH --job-name=unilid-floor
#SBATCH --account=a139
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=400G
#SBATCH --time=02:00:00
#SBATCH --output=/capstor/scratch/cscs/cmeister747/unilid_analysis/logs/floor_%j.out
#SBATCH --error=/capstor/scratch/cscs/cmeister747/unilid_analysis/logs/floor_%j.err

PYTHON="/users/cmeister747/.pyenv/versions/3.11.5/bin/python3"
WORKDIR="/users/cmeister747/unilid_analysis"

cd "${WORKDIR}"

echo "=== Starting floor sweep at $(date) ==="
echo "Node: $(hostname)"

${PYTHON} -c "
from analysis.floor_sweep import generate_floor_sweep
generate_floor_sweep(sample_size=500_000, output_dir='outputs')
"

echo "=== Finished at $(date) ==="
