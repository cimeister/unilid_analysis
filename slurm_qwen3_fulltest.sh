#!/bin/bash
#SBATCH --job-name=unilid-qwen3-fulltest
#SBATCH --account=infra01
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=400G
#SBATCH --time=06:00:00
#SBATCH --output=/capstor/scratch/cscs/cmeister747/unilid_analysis/logs/qwen3_fulltest_%j.out
#SBATCH --error=/capstor/scratch/cscs/cmeister747/unilid_analysis/logs/qwen3_fulltest_%j.err
set -euo pipefail
#
# Full-pool baseline scoring of the RETRAINED Qwen3-8B variant, for its
# tab:lid_main row. Chained with --dependency=afterok on the retrain (job 3112846)
# so it queues now rather than waiting for someone to notice the retrain landed.
#
# --configs baseline only: neither prior-side configuration appears in the paper.
#
# NOTE a convention change this forces, to carry into the table caption. The
# published DeepSeek3.2 and Qwen3 cells were computed on all 45,627,279 lines,
# while the UniLID and calibrated rows use the 45,377,279-line scored pool. This
# run puts the variants on the scored pool too, which makes tab:lid_main
# internally consistent for the first time and removes the caption's split.
PYTHON="/users/cmeister747/.pyenv/versions/3.11.5/bin/python3"
export RAYON_NUM_THREADS=64
cd /users/cmeister747/unilid_analysis
MODEL="/capstor/scratch/cscs/cmeister747/unilid_analysis/glotlid_qwen3_8b_fp64.unilid"
if [ ! -f "${MODEL}" ]; then
  echo "FATAL: ${MODEL} missing; the retrain did not pack a model" >&2
  exit 1
fi
echo "=== Qwen3-8B retrained full-pool baseline at $(date) on $(hostname) ==="
echo "commit: $(git rev-parse HEAD)"
${PYTHON} -u -m analysis.full_test_eval \
    --model "${MODEL}" \
    --scratch-dir "/capstor/scratch/cscs/cmeister747/unilid_analysis/full_test_eval_qwen3_fp64" \
    --configs baseline \
    --out-dir outputs_qwen3_fp64
echo "=== Finished at $(date) ==="
