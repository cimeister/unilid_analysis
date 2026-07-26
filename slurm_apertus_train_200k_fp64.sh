#!/bin/bash
#SBATCH --job-name=unilid-apertus200k-fp64
#SBATCH --account=infra01
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=400G
#SBATCH --time=12:00:00
#SBATCH --output=/capstor/scratch/cscs/cmeister747/unilid_analysis/logs/apertus200k_fp64_%j.out
#SBATCH --error=/capstor/scratch/cscs/cmeister747/unilid_analysis/logs/apertus200k_fp64_%j.err
set -euo pipefail
PYTHON="/users/cmeister747/.pyenv/versions/3.11.5/bin/python3"
export PATH="/users/cmeister747/.local/bin:${PATH}"   # patched fp64 spm_train (fork commits d0208d9+c5921a2)
SCR="/capstor/scratch/cscs/cmeister747/unilid_analysis"
cd /users/cmeister747/unilid_analysis/UNILID
BASE_FLAG=""
if [ -f "${SCR}/results_apertus200k_fp64/tokenizers/langspec_base_tokenizer.json" ]; then
  BASE_FLAG="--reuse-base"
fi
echo "=== Apertus 200k retrain under the fp64 trainer at $(date) on $(hostname) ==="
spm_train --version 2>&1 | head -1 || true
${PYTHON} train.py \
  --fasttext "${SCR}/train.txt" \
  --initial-vocab "/capstor/scratch/cscs/cmeister747/unilid_analysis/apertus_v2_200k/tokenizer.json" \
  --vocab-size 200000 --byte-level --per-lang-counts-method sp \
  --max-base-samples-per-lang 10000 \
  --lang-batch-size 20 \
  --results-dir "${SCR}/results_apertus200k_fp64" \
  --corpus-dir "${SCR}/results_apertus200k/corpus" \
  --reuse-corpus --skip-existing-langs ${BASE_FLAG}
echo "=== packing at $(date) ==="
${PYTHON} convert.py "${SCR}/results_apertus200k_fp64" -o "${SCR}/glotlid_apertus200k_fp64.unilid"
echo "=== Finished at $(date) ==="
