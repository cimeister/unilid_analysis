#!/bin/bash
#SBATCH --job-name=unilid-extudhr-corr
#SBATCH --account=infra01
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=400G
#SBATCH --time=02:00:00
#SBATCH --output=/capstor/scratch/cscs/cmeister747/unilid_analysis/logs/extudhr_corrected_%j.out
#SBATCH --error=/capstor/scratch/cscs/cmeister747/unilid_analysis/logs/extudhr_corrected_%j.err
set -euo pipefail
#
# UDHR / FLORES-200 scoring of the corrected model, stage "score".
#
# These benchmarks are a different TEST set, not a different training corpus, so
# the UniLID rows on them are GlotLID-C-trained and carry the correction. Both
# TSVs are already in this repository's scratch (udhr_eval.tsv 24,115 lines,
# flores200_eval.tsv 192,280 lines), so nothing here waits on the co-author.
#
# The clamp constant is read from fingerprint_floor21.json (c = -17) rather than
# the module default, so this cannot be built at a different constant than the
# full-pool predictions it is compared against.
PYTHON="/users/cmeister747/.pyenv/versions/3.11.5/bin/python3"
export RAYON_NUM_THREADS=64
cd /users/cmeister747/unilid_analysis
echo "=== corrected udhr scoring at $(date) on $(hostname) ==="
echo "commit: $(git rev-parse HEAD)"
${PYTHON} -u -m analysis.external_bench_eval --stage score --bench udhr \
    --model "/capstor/scratch/cscs/cmeister747/unilid_analysis/corrected/glotlidc_corrected.unilid" \
    --scratch-dir "/capstor/scratch/cscs/cmeister747/unilid_analysis/full_test_eval_corrected" \
    --out-dir outputs_corrected_round
echo "=== Finished at $(date) ==="
