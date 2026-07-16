#!/bin/bash
#SBATCH --job-name=unilid-disc8a
#SBATCH --account=a139
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=400G
#SBATCH --time=02:00:00
#SBATCH --output=/capstor/scratch/cscs/cmeister747/unilid_analysis/logs/disc8a_%j.out
#SBATCH --error=/capstor/scratch/cscs/cmeister747/unilid_analysis/logs/disc8a_%j.err

PYTHON="/users/cmeister747/.pyenv/versions/3.11.5/bin/python3"
WORKDIR="/users/cmeister747/unilid_analysis"

cd "${WORKDIR}"

echo "=== Starting Exp 8a: heuristic discriminative weighting at $(date) ==="
echo "Node: $(hostname)"

${PYTHON} -c "
from analysis.discriminative_finetune import generate_heuristic_discriminative
generate_heuristic_discriminative(sample_size=500_000, output_dir='outputs')
"

echo "=== Finished at $(date) ==="
