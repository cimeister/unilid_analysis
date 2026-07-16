#!/bin/bash
#SBATCH --job-name=unilid-norm
#SBATCH --account=a139
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=400G
#SBATCH --time=02:00:00
#SBATCH --output=/capstor/scratch/cscs/cmeister747/unilid_analysis/logs/norm_%j.out
#SBATCH --error=/capstor/scratch/cscs/cmeister747/unilid_analysis/logs/norm_%j.err

PYTHON="/users/cmeister747/.pyenv/versions/3.11.5/bin/python3"
WORKDIR="/users/cmeister747/unilid_analysis"

cd "${WORKDIR}"

echo "=== Starting normalized scoring job at $(date) ==="
echo "Node: $(hostname)"

${PYTHON} -c "
from analysis.normalized_predict import generate_normalized_analysis
generate_normalized_analysis(sample_size=500_000, output_dir='outputs')
"

echo "=== Finished at $(date) ==="
