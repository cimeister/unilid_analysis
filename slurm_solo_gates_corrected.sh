#!/bin/bash
#SBATCH --job-name=unilid-sologates-corr
#SBATCH --account=infra01
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=400G
#SBATCH --time=08:00:00
#SBATCH --output=/capstor/scratch/cscs/cmeister747/unilid_analysis/logs/sologates_corrected_%j.out
#SBATCH --error=/capstor/scratch/cscs/cmeister747/unilid_analysis/logs/sologates_corrected_%j.err
set -euo pipefail
#
# Re-estimate all 1,084 group-A re-examination thresholds on the corrected model.
#
# The thresholds are percentiles of score margins measured on the CLAMPED matrix,
# so they depend on the unseen-token constant. The probe of 2026-08-17 measured
# them moving by -12.5%, -5.5%, -71.8% and +123.3% under the correction, in both
# directions and two orders of magnitude apart in relative size, so none can be
# carried or shifted; all of them are re-estimated from each language's own
# training lines.
#
# The clamp target is read from fingerprint_floor21.json, written by the floor-c
# pass (job 3117583) alongside the predictions this gate is built on, so it
# cannot diverge from them. That fingerprint records c = -17, the constant the
# round-grid sweep selected.
PYTHON="/users/cmeister747/.pyenv/versions/3.11.5/bin/python3"
export RAYON_NUM_THREADS=64
cd /users/cmeister747/unilid_analysis
SCR="/capstor/scratch/cscs/cmeister747/unilid_analysis"
echo "=== corrected solo gates (group A thresholds) at $(date) on $(hostname) ==="
echo "commit: $(git rev-parse HEAD)"
${PYTHON} -u -m analysis.solo_gates floor21 \
    --model "${SCR}/corrected/glotlidc_corrected.unilid" \
    --scratch-dir "${SCR}/full_test_eval_corrected" \
    --out-dir outputs_corrected_round
echo "=== Finished at $(date) ==="
