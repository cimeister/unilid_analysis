#!/bin/bash
#SBATCH --job-name=unilid-cround-corr
#SBATCH --account=infra01
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=400G
#SBATCH --time=08:00:00
#SBATCH --output=/capstor/scratch/cscs/cmeister747/unilid_analysis/logs/cround_corrected_%j.out
#SBATCH --error=/capstor/scratch/cscs/cmeister747/unilid_analysis/logs/cround_corrected_%j.err
#
# Round-grid c sweep for the corrected model. Pre-registered in
# EXPERIMENTS_RESULTS.md ("Pre-registration: round-grid c sweep", 2026-08-18)
# BEFORE this job was submitted, including the predicted clamp counts.
#
# Grid {-15,-17,-19,-21} is chosen by the rule the published grid follows, not to
# move the answer: {-17,-19,-21,-23} put two values inside the released model's
# plateau range (-19.939 to -13.216) and two below all of it, and {-15,-17,-19,-21}
# does the same for the corrected model's range (-18.329 to -11.606).
#
# Selection procedure unchanged: all-strata guard on the validation half of the
# seed-42 500k draw, test half scored once.
#
# Writes to a SEPARATE output root so the shifted-grid sweep's table
# (outputs_corrected/) is not overwritten. Both are reported; neither is
# discarded.
set -euo pipefail
PYTHON="/users/cmeister747/.pyenv/versions/3.11.5/bin/python3"
export RAYON_NUM_THREADS=64
cd /users/cmeister747/unilid_analysis

MODEL="/capstor/scratch/cscs/cmeister747/unilid_analysis/corrected/glotlidc_corrected.unilid"

echo "=== corrected-model round-grid c sweep at $(date) on $(hostname) ==="
echo "model:  ${MODEL}"
echo "grid:   -15,-17,-19,-21"
echo "commit: $(git rev-parse HEAD)"
${PYTHON} -u -m analysis.floor_equalization \
    --model "${MODEL}" \
    --floors "-15,-17,-19,-21" \
    --out-dir outputs_corrected_round
echo "=== Finished at $(date) ==="
