#!/bin/bash
#SBATCH --job-name=unilid-fulltest-corr
#SBATCH --account=infra01
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=400G
#SBATCH --time=12:00:00
#SBATCH --output=/capstor/scratch/cscs/cmeister747/unilid_analysis/logs/fulltest_corrected_%j.out
#SBATCH --error=/capstor/scratch/cscs/cmeister747/unilid_analysis/logs/fulltest_corrected_%j.err
#
# Full-pool baseline scoring of the SPECIAL-TOKEN-CORRECTED model.
#
# Differs from slurm_full_test_eval.sh in three ways, all deliberate:
#  - --model points at the corrected weights;
#  - --scratch-dir is a fresh root, NOT the released model's directory, whose
#    entries are symlinks into the durable store (analysis/model_context.py
#    refuses the combination, so this is enforced rather than remembered);
#  - --configs baseline only. One full pool pass instead of three. Neither
#    freq_prior nor learned_bias appears in the paper, and learned_bias.npy was
#    fit against the released weights (author decision, 2026-08-17).
#
# Resumable: completed chunks are tracked in the scratch root's progress.json
# and the fingerprint covers the model's sha256, so a resume cannot mix chunks
# scored under two different models.
set -euo pipefail
PYTHON="/users/cmeister747/.pyenv/versions/3.11.5/bin/python3"
export RAYON_NUM_THREADS=64
cd /users/cmeister747/unilid_analysis

MODEL="/capstor/scratch/cscs/cmeister747/unilid_analysis/corrected/glotlidc_corrected.unilid"
SCRATCH="/capstor/scratch/cscs/cmeister747/unilid_analysis/full_test_eval_corrected"

echo "=== corrected-model full-pool baseline at $(date) on $(hostname) ==="
echo "model:   ${MODEL}"
echo "scratch: ${SCRATCH}"
echo "commit:  $(git rev-parse HEAD)"
${PYTHON} -u -m analysis.full_test_eval \
    --model "${MODEL}" \
    --scratch-dir "${SCRATCH}" \
    --configs baseline \
    --out-dir outputs_corrected
echo "=== Finished at $(date) ==="
