#!/bin/bash
#SBATCH --job-name=unilid-gatetopk-corr
#SBATCH --account=infra01
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=400G
#SBATCH --time=08:00:00
#SBATCH --output=/capstor/scratch/cscs/cmeister747/unilid_analysis/logs/gatetopk_corrected_%j.out
#SBATCH --error=/capstor/scratch/cscs/cmeister747/unilid_analysis/logs/gatetopk_corrected_%j.err
set -euo pipefail
#
# gate_variants stage "topk" on the corrected model: bank the top-5 candidates
# per affected line over the full pool.
#
# Submitted in PARALLEL with the group-A threshold job (3123324) rather than
# after it. This stage reads pred_floor21.npy and the fingerprint and scores
# candidates; it does not read either tau CSV, which only the "apply" stage
# needs. The recorded cost of the equivalent pass on the released model was
# 3h13m, so serializing it behind the thresholds would have cost that time for
# nothing.
#
# The clamp target is read from fingerprint_floor21.json (c = -17), so this
# cannot be built at a different constant than the predictions it compares
# against.
PYTHON="/users/cmeister747/.pyenv/versions/3.11.5/bin/python3"
export RAYON_NUM_THREADS=64
cd /users/cmeister747/unilid_analysis
SCR="/capstor/scratch/cscs/cmeister747/unilid_analysis"
echo "=== corrected gate topk at $(date) on $(hostname) ==="
echo "commit: $(git rev-parse HEAD)"
${PYTHON} -u -m analysis.gate_variants topk \
    --model "${SCR}/corrected/glotlidc_corrected.unilid" \
    --scratch-dir "${SCR}/full_test_eval_corrected" \
    --out-dir outputs_corrected_round
echo "=== Finished at $(date) ==="
