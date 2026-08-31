#!/bin/bash
#SBATCH --job-name=unilid-lenbias-golden
#SBATCH --account=infra01
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=300G
#SBATCH --time=06:00:00
#SBATCH --output=/capstor/scratch/cscs/cmeister747/unilid_analysis/logs/lenbias_golden_%j.out
#SBATCH --error=/capstor/scratch/cscs/cmeister747/unilid_analysis/logs/lenbias_golden_%j.err
set -euo pipefail
#
# tab:lenbias-delta on the PD-4 basis (author ruling 2026-08-24): the golden
# subset, the 250,000-line test half of the seed-42 500,000-line draw, the same
# subset tab:lenbias-norm was rebuilt on (analysis/lenbias_norm_table.py).
#
# Two runs, in this order:
#   1. RELEASED model over the golden subset. This is the instrument gate: the
#      published table was computed over all 45,627,279 lines, so exact
#      reproduction is not expected, and the gate is directional -- same signs,
#      same ordering across length bins, magnitudes consistent with
#      paper/tables/lenbias-delta.tex. Writes outputs/tables/length_bias_golden.*
#      (suffixed, so the full-pool artifacts that back the published table are
#      untouched).
#   2. CORRECTED model over the same golden subset, predictions from its own
#      full_test_eval_corrected/pred_baseline.npy. Writes
#      outputs_corrected_round/tables/length_bias_golden.*
#
# Both runs resolve the model through analysis.model_context, which refuses to
# let the corrected model write under outputs/.
#
# HOW IT WAS ACTUALLY RUN (2026-08-25): directly on the login node, not through
# sbatch. The queue held 606 pending jobs and this account's own submissions were
# scheduled 21 hours out. Measured cost per arm: 729 per-language tokenizers and
# about 80 GB peak RSS, against 506 GB free on the login node; 20 minutes for the
# released arm and 22 for the corrected one. The two arms run in separate
# processes, so only one 80 GB peak is live at a time. This file is kept as the
# submission script for the same two runs; the body below is identical either way
# (bash it, or sbatch it).
#
# One caveat on the released arm: the released model file is a symlink into
# /capstor/store, and that mount was intermittently unavailable on 2026-08-25. One
# run stalled about seven minutes inside the tokenizer build and one refused to
# start at all ("model file does not exist"). Both recovered on a retry.
PYTHON="/users/cmeister747/.pyenv/versions/3.11.5/bin/python3"
export RAYON_NUM_THREADS=32
cd /users/cmeister747/unilid_analysis
SCR="/capstor/scratch/cscs/cmeister747/unilid_analysis"
echo "=== lenbias-delta golden subset at $(date) on $(hostname) ==="
echo "commit: $(git rev-parse HEAD)"

echo "--- 1/2 released model, golden subset ---"
"${PYTHON}" -u -m analysis.length_bias --subset golden

echo "--- 2/2 corrected model, golden subset ---"
"${PYTHON}" -u -m analysis.length_bias --subset golden \
    --model "${SCR}/corrected/glotlidc_corrected.unilid" \
    --scratch-dir "${SCR}/full_test_eval_corrected" \
    --out-dir outputs_corrected_round

echo "=== Finished at $(date) ==="
