#!/bin/bash
#SBATCH --job-name=unilid-csweep-corr
#SBATCH --account=infra01
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=400G
#SBATCH --time=08:00:00
#SBATCH --output=/capstor/scratch/cscs/cmeister747/unilid_analysis/logs/csweep_corrected_%j.out
#SBATCH --error=/capstor/scratch/cscs/cmeister747/unilid_analysis/logs/csweep_corrected_%j.err
#
# Re-derive the unseen-token constant c for the SPECIAL-TOKEN-CORRECTED model.
#
# Protocol unchanged from Exp 20: sweep c, select on the validation half of the
# seed-42 500k draw under the all-strata guard, score the test half once.
#
# The grid is the published {-17,-19,-21,-23} shifted by log 5 = 1.6094
# (--corrected-grid, analysis/floor_equalization.FLOORS_CORRECTED). The clamp
# sets an ABSOLUTE target in log space and the correction raised every real
# token by exactly that amount, so the shifted grid asks the published question
# of a corrected model. Sweeping the unshifted grid would ask a different one:
# at c = -21 a corrected row's unseen tokens sit 1.609 nats further below its
# seen tokens than the released model's did.
#
# Expected landing point: near -19.3906 (= -21 + log 5), which would reproduce
# the released model's clamped structure up to the uniform shift. The 60,000-line
# validation probe put the corrected optimum at -17.5 against -19.5 released, a
# shift of +2.0 against log 5 = 1.609, on an optimum flat enough that the two
# were not distinguishable there. A result far from -19.3906 is a finding, not a
# tuning problem, and must be recorded as one.
set -euo pipefail
PYTHON="/users/cmeister747/.pyenv/versions/3.11.5/bin/python3"
export RAYON_NUM_THREADS=64
cd /users/cmeister747/unilid_analysis

MODEL="/capstor/scratch/cscs/cmeister747/unilid_analysis/corrected/glotlidc_corrected.unilid"

echo "=== corrected-model c sweep at $(date) on $(hostname) ==="
echo "model:  ${MODEL}"
echo "commit: $(git rev-parse HEAD)"
${PYTHON} -u -m analysis.floor_equalization \
    --model "${MODEL}" \
    --corrected-grid \
    --out-dir outputs_corrected
echo "=== Finished at $(date) ==="
