#!/bin/bash
#SBATCH --job-name=unilid-apertus131k-train
#SBATCH --account=infra01
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=400G
#SBATCH --time=12:00:00
#SBATCH --output=/capstor/scratch/cscs/cmeister747/unilid_analysis/logs/apertus131k_train_%j.out
#SBATCH --error=/capstor/scratch/cscs/cmeister747/unilid_analysis/logs/apertus131k_train_%j.err

set -euo pipefail
PYTHON="/users/cmeister747/.pyenv/versions/3.11.5/bin/python3"
export PATH="/users/cmeister747/.local/bin:${PATH}"   # forked spm_train (fixed-vocab EM)
SCR="/capstor/scratch/cscs/cmeister747/unilid_analysis"
TOKENIZER="/users/cmeister747/apertus-tokenizer-development/preliminary_mul/tokenizer.json"
cd /users/cmeister747/unilid_analysis/UNILID

# First run builds the base from --initial-vocab; a resume reuses the saved base.
# The per-language corpus split from the 200k run is reused read-only via --corpus-dir
# (raw text split by language label, tokenizer-independent; verified by preflight_131k).
BASE_FLAG=""
if [ -f "${SCR}/results_apertus131k/tokenizers/langspec_base_tokenizer.json" ]; then
  BASE_FLAG="--reuse-base"
  echo "=== Resume: existing base tokenizer found, passing --reuse-base ==="
fi

echo "=== Apertus 131k (preliminary_mul) retrain at $(date) on $(hostname) ==="
# NOTE: train.py's top-level "N existing lang tokenizers found / N remaining" line is
# wrong on resume (sp-vs-soft filename mismatch); the real per-language skip happens in
# language_specific_trainer.py. Progress = count of langspec_soft_*.tokenizer.json.
${PYTHON} train.py \
  --fasttext "${SCR}/train.txt" \
  --initial-vocab "${TOKENIZER}" \
  --vocab-size 131072 --byte-level --per-lang-counts-method sp \
  --max-base-samples-per-lang 10000 \
  --lang-batch-size 20 \
  --results-dir "${SCR}/results_apertus131k" \
  --corpus-dir "${SCR}/results_apertus200k/corpus" \
  --reuse-corpus --skip-existing-langs ${BASE_FLAG}

echo "=== train.py returned at $(date); packing .unilid ==="
${PYTHON} convert.py "${SCR}/results_apertus131k" -o "${SCR}/glotlid_apertus131k.unilid"
echo "=== Finished at $(date) ==="
