#!/bin/bash
#SBATCH --job-name=unilid-floorc-corr
#SBATCH --account=infra01
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=400G
#SBATCH --time=06:00:00
#SBATCH --output=/capstor/scratch/cscs/cmeister747/unilid_analysis/logs/floorc_corrected_%j.out
#SBATCH --error=/capstor/scratch/cscs/cmeister747/unilid_analysis/logs/floorc_corrected_%j.err
set -euo pipefail
#
# Full-pool scoring of the corrected model under the clamp, at whatever constant
# the round-grid sweep selected.
#
# Submitted with --dependency=afterok on that sweep, so it queues and accrues
# priority now instead of waiting for a human to read the sweep and submit. The
# constant is READ FROM THE SWEEP'S OUTPUT rather than typed in: typing it would
# mean either waiting or guessing, and a guess is exactly how the chain ends up
# built at a constant the predictions were never scored under.
#
# The fingerprint this writes records the constant, and analysis/gate_variants.py
# reads its clamp target from that fingerprint, so the rest of the chain inherits
# it and cannot diverge.
#
# Replaces job 3110918, which was cancelled while pending because it would have
# run at -17.3906 from the shifted-grid sweep, which the round-grid sweep
# supersedes.
PYTHON="/users/cmeister747/.pyenv/versions/3.11.5/bin/python3"
export RAYON_NUM_THREADS=64
cd /users/cmeister747/unilid_analysis

SWEEP="outputs_corrected_round/tables/floor_equalization.md"
MODEL="/capstor/scratch/cscs/cmeister747/unilid_analysis/corrected/glotlidc_corrected.unilid"
SCRATCH="/capstor/scratch/cscs/cmeister747/unilid_analysis/full_test_eval_corrected"

if [ ! -f "${SWEEP}" ]; then
  echo "FATAL: ${SWEEP} missing; the round-grid sweep did not produce its table" >&2
  exit 1
fi
C=$(${PYTHON} -m analysis.selected_floor_target "${SWEEP}")
echo "=== corrected floor-c full-pool pass at $(date) on $(hostname) ==="
echo "commit: $(git rev-parse HEAD)"
echo "constant read from ${SWEEP}: c = ${C}"

${PYTHON} -u -m analysis.full_test_floor21 \
    --model "${MODEL}" \
    --scratch-dir "${SCRATCH}" \
    --floor-target "${C}" \
    --out-dir outputs_corrected_round
echo "=== Finished at $(date) ==="
