#!/bin/bash
#SBATCH --job-name=unilid-decoders-corr
#SBATCH --account=infra01
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=400G
#SBATCH --time=12:00:00
#SBATCH --output=/capstor/scratch/cscs/cmeister747/unilid_analysis/logs/decoders_corrected_%j.out
#SBATCH --error=/capstor/scratch/cscs/cmeister747/unilid_analysis/logs/decoders_corrected_%j.err
#
# tab:viterbi_vs_marginal on the corrected model. Independent of the
# unseen-token constant: both decoders run in base mode on the unclamped matrix,
# so this does not wait on the floor-c pass.
#
# Two full-pool passes per chunk, and marginalization is the more expensive
# decoder (the paper reports roughly 2x), so budget about three times a single
# baseline pass. Resumable per chunk.
set -euo pipefail
PYTHON="/users/cmeister747/.pyenv/versions/3.11.5/bin/python3"
export RAYON_NUM_THREADS=64
cd /users/cmeister747/unilid_analysis
echo "=== corrected-model decoder comparison at $(date) on $(hostname) ==="
echo "commit: $(git rev-parse HEAD)"
${PYTHON} -u -m analysis.viterbi_vs_marginal \
    --model /capstor/scratch/cscs/cmeister747/unilid_analysis/corrected/glotlidc_corrected.unilid \
    --scratch-dir /capstor/scratch/cscs/cmeister747/unilid_analysis/full_test_decoders_corrected \
    --out-dir outputs_corrected/tables
echo "=== Finished at $(date) ==="
