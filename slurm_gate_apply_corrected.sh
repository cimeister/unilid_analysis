#!/bin/bash
#SBATCH --job-name=unilid-gateapply-corr
#SBATCH --account=infra01
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=400G
#SBATCH --time=04:00:00
#SBATCH --output=/capstor/scratch/cscs/cmeister747/unilid_analysis/logs/gateapply_corrected_%j.out
#SBATCH --error=/capstor/scratch/cscs/cmeister747/unilid_analysis/logs/gateapply_corrected_%j.err
set -euo pipefail
#
# gate_variants "apply" stages on the corrected model, in the mandatory order:
# flat4_tau5 first (its pred_gate_flat4_tau5.npy is the self-check reference
# that apply flat4_prox21 refuses to run without, gate_variants.py:1130-1133),
# then flat4_prox21, the promoted variant.
#
# Both stages only post-process the arrays banked by the topk stage (job
# 3127704); no test-pool scoring happens here. They read the flat_magnet
# categories from the RELEASED model's outputs/diagnostic/lang_diagnostic.csv
# (module-level DIAG_CSV constant) -- the same file the corrected topk stage
# validated its expanded-mask fingerprint against, so the gate structure stays
# the one banked; whether that structure must be re-derived on corrected
# weights is the open group-B question, tracked separately.
PYTHON="/users/cmeister747/.pyenv/versions/3.11.5/bin/python3"
export RAYON_NUM_THREADS=64
cd /users/cmeister747/unilid_analysis
SCR="/capstor/scratch/cscs/cmeister747/unilid_analysis"
echo "=== corrected gate apply at $(date) on $(hostname) ==="
echo "commit: $(git rev-parse HEAD)"
"${PYTHON}" -u -m analysis.gate_variants apply flat4_tau5 \
    --model "${SCR}/corrected/glotlidc_corrected.unilid" \
    --scratch-dir "${SCR}/full_test_eval_corrected" \
    --out-dir outputs_corrected_round
"${PYTHON}" -u -m analysis.gate_variants apply flat4_prox21 \
    --model "${SCR}/corrected/glotlidc_corrected.unilid" \
    --scratch-dir "${SCR}/full_test_eval_corrected" \
    --out-dir outputs_corrected_round
echo "=== Finished at $(date) ==="
