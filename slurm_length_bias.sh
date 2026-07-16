#!/bin/bash
#SBATCH --job-name=unilid-lenbias
#SBATCH --account=a139
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=400G
#SBATCH --time=12:00:00
#SBATCH --output=/capstor/scratch/cscs/cmeister747/unilid_analysis/logs/lenbias_%j.out
#SBATCH --error=/capstor/scratch/cscs/cmeister747/unilid_analysis/logs/lenbias_%j.err

PYTHON="/users/cmeister747/.pyenv/versions/3.11.5/bin/python3"
WORKDIR="/users/cmeister747/unilid_analysis"

cd "${WORKDIR}"

echo "=== Starting length bias job at $(date) ==="
echo "Node: $(hostname)"

# Length bias streams misclassified texts directly from disk.
# Only runs the length bias analysis (skips tables which already completed).
${PYTHON} -c "
from analysis.length_bias import generate_length_bias_analysis
generate_length_bias_analysis(output_dir='outputs')
"

echo "=== Finished at $(date) ==="
