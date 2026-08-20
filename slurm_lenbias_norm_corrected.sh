#!/bin/bash
#SBATCH --job-name=unilid-lenbias-corr
#SBATCH --account=infra01
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=400G
#SBATCH --time=06:00:00
#SBATCH --output=/capstor/scratch/cscs/cmeister747/unilid_analysis/logs/lenbias_corrected_%j.out
#SBATCH --error=/capstor/scratch/cscs/cmeister747/unilid_analysis/logs/lenbias_corrected_%j.err
#
# tab:lenbias-norm on the corrected model: alpha = 0 (raw rescore) and alpha = 1
# (length-normalized) over the 500k sample. Base mode, so independent of the
# unseen-token constant.
#
# The Original column is omitted for a non-default model, since no recorded
# prediction column exists for it and filling it from the released model's
# predictions would compare two different models in one row.
set -euo pipefail
PYTHON="/users/cmeister747/.pyenv/versions/3.11.5/bin/python3"
export RAYON_NUM_THREADS=64
cd /users/cmeister747/unilid_analysis
echo "=== corrected-model lenbias-norm at $(date) on $(hostname) ==="
echo "commit: $(git rev-parse HEAD)"
${PYTHON} -u -m analysis.lenbias_norm_table \
    --model /capstor/scratch/cscs/cmeister747/unilid_analysis/corrected/glotlidc_corrected.unilid \
    --baseline-pred /capstor/scratch/cscs/cmeister747/unilid_analysis/full_test_eval_corrected/pred_baseline.npy \
    -o outputs_corrected_round/tables
echo "=== Finished at $(date) ==="
