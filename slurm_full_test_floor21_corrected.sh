#!/bin/bash
#SBATCH --job-name=unilid-floor21-corr
#SBATCH --account=infra01
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=400G
#SBATCH --time=12:00:00
#SBATCH --output=/capstor/scratch/cscs/cmeister747/unilid_analysis/logs/floor21_corrected_%j.out
#SBATCH --error=/capstor/scratch/cscs/cmeister747/unilid_analysis/logs/floor21_corrected_%j.err
#
# Full-pool scoring of the corrected model under its own selected unseen-token
# constant, c = -17.3906.
#
# That constant is what the published Exp 20 procedure selected when re-run on
# the corrected weights (job 3107082), not -21 + log 5 = -19.3906, which was the
# pre-registered expectation. Author decision 2026-08-18: ship what the protocol
# selects. Aligned by grid position the two sweeps are near-identical, and
# positions 2 and 3 are tied in both models (released picks -21 over -19 by
# 0.0001; corrected picks -17.3906 over -19.3906 by 0.0002), so this is a tie
# broken the other way rather than a constant that moved.
#
# Consequence, and it is a change from the released model: at c = -17.3906,
# 1,821 of 1,940 rows are clamped and 119 already sit at or below the target. At
# c = -21 on the released weights all 1,940 moved. The `n_mod == n_lang`
# assertion the chain used encoded that incidental fact; it is replaced by
# analysis.floor_equalization.verify_one_sided_clamp, which checks the property
# that actually has to hold: no row was skipped that should have been lowered.
set -euo pipefail
PYTHON="/users/cmeister747/.pyenv/versions/3.11.5/bin/python3"
export RAYON_NUM_THREADS=64
cd /users/cmeister747/unilid_analysis

MODEL="/capstor/scratch/cscs/cmeister747/unilid_analysis/corrected/glotlidc_corrected.unilid"
SCRATCH="/capstor/scratch/cscs/cmeister747/unilid_analysis/full_test_eval_corrected"
C="-17.3905620875659"

echo "=== corrected-model floor-c full-pool pass at $(date) on $(hostname) ==="
echo "model:  ${MODEL}"
echo "c:      ${C}"
echo "commit: $(git rev-parse HEAD)"
${PYTHON} -u -m analysis.full_test_floor21 \
    --model "${MODEL}" \
    --scratch-dir "${SCRATCH}" \
    --floor-target "${C}" \
    --out-dir outputs_corrected
echo "=== Finished at $(date) ==="
