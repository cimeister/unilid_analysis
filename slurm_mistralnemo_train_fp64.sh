#!/bin/bash
#SBATCH --job-name=unilid-mistralnemo-fp64
#SBATCH --account=infra01
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=400G
#SBATCH --time=12:00:00
#SBATCH --output=/capstor/scratch/cscs/cmeister747/unilid_analysis/logs/mistralnemo_fp64_%j.out
#SBATCH --error=/capstor/scratch/cscs/cmeister747/unilid_analysis/logs/mistralnemo_fp64_%j.err
set -euo pipefail

# E3 retrain (EXPERIMENTS_PLAN.md "Camera-ready evaluation program", E3).
# Mirrors the recorded Apertus fp64 retrain pipeline EXACTLY (jobs 2903767 and
# 2903768: same account/partition/shape/memory/walltime, the same single
# train.py invocation driving all 1,940 languages with its own per-language
# loop, then convert.py packing). Differences from slurm_apertus_train_131k_fp64.sh:
# the pinned Mistral-Nemo tokenizer (snapshot a4477a2f977929a969745b69bbd62e03043551a5,
# tokenizer.json sha256 e11c71726323d33da7b8d6f6f269f1988931c0a52b7122bcdd8c05042974e0db),
# the results dir, the packed-model name, and --fasttext omitted entirely
# (train.txt no longer exists on scratch; --corpus-dir with --reuse-corpus is a
# sufficient input per train.py's own argument logic, verified at lines 329/369).
# Run `python -m analysis.preflight_mistralnemo` by hand before submitting.
# The 131k Apertus job (same 131,072 vocab size) completed in 4h36m; 12h is
# headroom, not a measurement.

PYTHON="/users/cmeister747/.pyenv/versions/3.11.5/bin/python3"
export PATH="/users/cmeister747/.local/bin:${PATH}"   # patched fp64 spm_train (fork commits d0208d9+c5921a2)
SCR="/capstor/scratch/cscs/cmeister747/unilid_analysis"
TOKENIZER="/capstor/store/cscs/swissai/a0229/cmeister/huggingface/hub/models--mistralai--Mistral-Nemo-Base-2407/snapshots/a4477a2f977929a969745b69bbd62e03043551a5/tokenizer.json"
cd /users/cmeister747/unilid_analysis/UNILID

BASE_FLAG=""
if [ -f "${SCR}/results_mistralnemo/tokenizers/langspec_base_tokenizer.json" ]; then
  BASE_FLAG="--reuse-base"
fi

echo "=== Mistral-Nemo fp64 retrain at $(date) on $(hostname) ==="
spm_train --version 2>&1 | head -1 || true
${PYTHON} train.py \
  --initial-vocab "${TOKENIZER}" \
  --vocab-size 131072 --byte-level --per-lang-counts-method sp \
  --max-base-samples-per-lang 10000 \
  --lang-batch-size 20 \
  --results-dir "${SCR}/results_mistralnemo" \
  --corpus-dir "${SCR}/results_apertus200k/corpus" \
  --reuse-corpus --skip-existing-langs ${BASE_FLAG}

echo "=== packing at $(date) ==="
${PYTHON} convert.py "${SCR}/results_mistralnemo" -o "${SCR}/glotlid_mistralnemo_fp64.unilid"

echo "=== degeneracy scan at $(date) ==="
cd /users/cmeister747/unilid_analysis
${PYTHON} -m analysis.degeneracy_scan_mistralnemo

echo "=== Finished at $(date) ==="
