#!/bin/bash
#SBATCH --job-name=unilid-wili-fp64
#SBATCH --account=infra01
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=400G
#SBATCH --time=04:00:00
#SBATCH --output=/capstor/scratch/cscs/cmeister747/unilid_analysis/logs/wili_%x_%j.out
#SBATCH --error=/capstor/scratch/cscs/cmeister747/unilid_analysis/logs/wili_%x_%j.err
set -euo pipefail
#
# Retrain one WiLI model's per-language weights over its ORIGINAL vocabulary,
# with the patched fp64 spm_train and UNILID 0.3.0.
#
# Takes MODEL_NAME and VOCAB_SIZE from the environment (--export).
#
# Two changes from the published model, both recorded with the result:
#   1. 0.3.0 no longer gives the special tokens the base tokenizer's score-0
#      entries, so rows are born with real-token mass 1.0 rather than 0.2.
#   2. the patched fp64 spm_train (fork commits d0208d9 + c5921a2) replaces the
#      fp32 build. WiLI's longest training line is 40,578 bytes and 101 lines
#      exceed the 4,192-byte upstream cap, so the fp64 EM overflow is NOT
#      excluded here and this is not a cosmetic change.
#
# --results-dir and --base-tokenizer-path are BOTH mandatory. Without them
# train.py:450-452 defaults the base path to results_<vocab//1000>k/tokenizers/,
# the reuse test at train.py:455 fails, and train.py:465-492 silently trains a
# FRESH vocabulary and reports success, estimating every row over a different
# vocabulary than the model being replaced.
PYTHON="/users/cmeister747/.pyenv/versions/3.11.5/bin/python3"
export PATH="/users/cmeister747/.local/bin:${PATH}"   # patched fp64 spm_train
SCR="/capstor/scratch/cscs/cmeister747/unilid_analysis"
RES="${SCR}/results_${MODEL_NAME}_fp64"
BASE="${RES}/tokenizers/langspec_base_tokenizer.json"
CORPUS="${SCR}/wili_corpus_shared"
OUT="${SCR}/${MODEL_NAME}_fp64.unilid"

if [ ! -f "${BASE}" ]; then
  echo "FATAL: base tokenizer missing at ${BASE}" >&2; exit 1
fi
# skip-existing-langs defaults to true; that is safe only because this directory
# holds no per-language rows. A pre-0.3.0 row here would be loaded and packed
# uncorrected, since the loader validates token order but not real-token mass.
if compgen -G "${RES}/tokenizers/langspec_soft_*" > /dev/null || \
   compgen -G "${RES}/tokenizers/langspec_sp_*" > /dev/null; then
  echo "FATAL: ${RES}/tokenizers already holds per-language rows; refusing to" >&2
  echo "       resume, since --skip-existing-langs would reuse them unchecked." >&2
  exit 1
fi
if [ ! -d "${CORPUS}" ]; then echo "FATAL: corpus missing at ${CORPUS}" >&2; exit 1; fi

cd /users/cmeister747/unilid_analysis/UNILID
echo "=== WiLI retrain ${MODEL_NAME} (vocab ${VOCAB_SIZE}) at $(date) on $(hostname) ==="
echo "commit:    $(git -C /users/cmeister747/unilid_analysis rev-parse HEAD)"
echo "spm_train: $(command -v spm_train)"
spm_train --version 2>&1 | head -1 || true

${PYTHON} train.py \
  --corpus-dir "${CORPUS}" --reuse-corpus \
  --vocab-size "${VOCAB_SIZE}" --byte-level \
  --per-lang-counts-method sp \
  --results-dir "${RES}" \
  --base-tokenizer-path "${BASE}" --reuse-base \
  --lang-batch-size 20

echo "=== packing at $(date) ==="
${PYTHON} convert.py "${RES}" -o "${OUT}"

echo "=== post-training checks at $(date) ==="
cd /users/cmeister747/unilid_analysis
${PYTHON} -u -m analysis.inspect_variant_models "${OUT}" \
  -o "outputs/rerelease/wili_${MODEL_NAME}_fp64_inspect.json"
echo "=== Finished at $(date) ==="
