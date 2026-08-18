#!/bin/bash
#SBATCH --job-name=unilid-nemo-base-corr
#SBATCH --account=infra01
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=400G
#SBATCH --time=12:00:00
#SBATCH --output=/capstor/scratch/cscs/cmeister747/unilid_analysis/logs/nemo_base_corrected_%j.out
#SBATCH --error=/capstor/scratch/cscs/cmeister747/unilid_analysis/logs/nemo_base_corrected_%j.err
set -euo pipefail
#
# Mistral-Nemo variant, corrected weights, stage "baseline" only.
#
# Submitted ahead of the rest of its chain because this stage is independent of
# every other queued job: it applies no clamp, reads no fingerprint_floor21.json,
# and touches FLOOR_TARGET nowhere, so it does not wait on the unseen-token
# constant. The later stages (calibval, flatrule, tau, topk, eval) DO clamp at
# the shared constant and must wait for the round-grid sweep, job 3111471.
#
# --base-scratch points at the corrected base model's root rather than the
# released one. The only thing read from there is y_true.npy, which was verified
# bit-identical between the two runs over all 45,627,279 entries on 2026-08-18,
# so this changes nothing numerically; it means the corrected chain reads nothing
# from the released model's directory.
#
# --scratch-dir is a fresh root. The chain's own default is a directory symlink
# into the durable store holding the E3 artifacts of record, and
# analysis/model_context.py refuses that combination rather than leaving it to be
# remembered.
PYTHON="/users/cmeister747/.pyenv/versions/3.11.5/bin/python3"
export RAYON_NUM_THREADS=64
cd /users/cmeister747/unilid_analysis
SCR="/capstor/scratch/cscs/cmeister747/unilid_analysis"

echo "=== Mistral-Nemo corrected baseline at $(date) on $(hostname) ==="
echo "commit: $(git rev-parse HEAD)"
${PYTHON} -u -m analysis.mistralnemo_eval --stage baseline \
    --model "${SCR}/corrected/glotlid_mistralnemo_fp64_corrected.unilid" \
    --scratch-dir "${SCR}/full_test_eval_mistralnemo_corrected" \
    --base-scratch "${SCR}/full_test_eval_corrected"
echo "=== Finished at $(date) ==="
