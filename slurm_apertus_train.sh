#!/bin/bash
#SBATCH --job-name=unilid-apertus-train
#SBATCH --account=infra01
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=400G
#SBATCH --time=12:00:00
#SBATCH --output=/capstor/scratch/cscs/cmeister747/unilid_analysis/logs/apertus_train_%j.out
#SBATCH --error=/capstor/scratch/cscs/cmeister747/unilid_analysis/logs/apertus_train_%j.err

set -euo pipefail
PYTHON="/users/cmeister747/.pyenv/versions/3.11.5/bin/python3"
export PATH="/users/cmeister747/.local/bin:${PATH}"   # forked spm_train
SCR="/capstor/scratch/cscs/cmeister747/unilid_analysis"
cd /users/cmeister747/unilid_analysis/UNILID

echo "=== Apertus 200k retrain (standard setup) at $(date) on $(hostname) ==="
# Resumable: reuse the split corpus + base tokenizer, skip languages already done.
${PYTHON} train.py \
  --fasttext "${SCR}/train.txt" \
  --initial-vocab "${SCR}/apertus_v2_200k/tokenizer.json" \
  --vocab-size 200000 --byte-level --per-lang-counts-method sp \
  --max-base-samples-per-lang 10000 \
  --lang-batch-size 20 \
  --results-dir "${SCR}/results_apertus200k" \
  --reuse-corpus --reuse-base --skip-existing-langs

echo "=== train.py returned at $(date); packing .unilid ==="
${PYTHON} convert.py "${SCR}/results_apertus200k" -o "${SCR}/glotlid_apertus200k.unilid"
echo "=== Finished at $(date) ==="
