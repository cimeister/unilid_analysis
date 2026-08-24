#!/bin/bash
#SBATCH --job-name=w-mistralnemo_wili
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
# PHASE 2a of the WiLI regeneration plan: the \unilid-Mistral-Nemo row.
#
# Trains WiLI per-language weights over the Mistral-Nemo base vocabulary, which
# is EXTRACTED FROM THE GlotLID-C CONTAINER rather than re-converted from
# HuggingFace. The plan's Phase 2 table justifies that: "a model built on an LLM
# tokenizer carries that tokenizer unchanged, so it is the same across training
# corpora", measured byte-identical for the DeepSeek and Qwen pairs. The base
# vocabulary is also untouched by the special-token defect, which lives in the
# per-language path (language_specific_trainer.train_with_sentencepiece_direct).
#
# Source container: ${SCR}/glotlid_mistralnemo_fp64.unilid
# Extracted 2026-08-23 with `python -m analysis.extract_base_tokenizer`, which
# writes ONLY the base tokenizer; unpack_unilid would also write 1,940 defective
# langspec_sp_* rows beside the corrected langspec_soft_* rows this run produces,
# and convert.py picking the right set would then be naming coincidence
# (model_io.py:135-137). Record: outputs/rerelease/wili_mistralnemo_base_extract.json
#   vocab 131,072  Unigram  specials at [1, 2, 10, 0]
#   sha256 913ade8d70984a689ed4cf05efce731e9b03f0b674fcafda0626582ac2eaaa7b
#
# Mirrors slurm_wili_train_fp64.sh (the validated Phase 1 template) with the
# preflight moved into analysis/preflight_wili_base.py so it is testable.
# The same two changes from the published models are recorded with the result:
#   1. UNILID 0.3.0 no longer gives the special tokens the base tokenizer's
#      score-0 entries, so rows are born with real-token mass 1.0, not 0.2.
#   2. the patched fp64 spm_train (fork commits d0208d9 + c5921a2) replaces the
#      fp32 build. WiLI's longest training line is 40,578 bytes and 101 lines
#      exceed the 4,192-byte upstream cap, so the fp64 EM overflow is NOT
#      excluded here.
# NOTE for the results entry: unlike the three Phase 1 models, there is no
# published WiLI Mistral-Nemo container to compare against, so the transformation
# gate of the plan's Verification step 2 cannot be run on this model. Its only
# checks are real-token mass 1.0 and the label-set comparison (step 3).
#
# --results-dir and --base-tokenizer-path are BOTH mandatory. Without them
# train.py:450-452 defaults the base path to results_<vocab//1000>k/tokenizers/,
# the reuse test at train.py:455 fails, and train.py:465-492 silently trains a
# FRESH vocabulary and reports success, estimating every row over a different
# vocabulary than the model this row is meant to be.
#
# Phase 1's three retrains took 19 to 25 minutes each at vocabularies of 100,000
# to 151,670. 4h is the template's headroom, not a measurement.
PYTHON="/users/cmeister747/.pyenv/versions/3.11.5/bin/python3"
export PATH="/users/cmeister747/.local/bin:${PATH}"   # patched fp64 spm_train
REPO="/users/cmeister747/unilid_analysis"
SCR="/capstor/scratch/cscs/cmeister747/unilid_analysis"

MODEL_NAME="mistralnemo_wili"
VOCAB_SIZE=131072
BASE_SHA256="913ade8d70984a689ed4cf05efce731e9b03f0b674fcafda0626582ac2eaaa7b"
# sha256 of the patched fp64 spm_train installed 2026-07-27 (fork commits
# d0208d9 + c5921a2), recorded in slurm_wili_train_fp32null.sh. `spm_train
# --version` prints "sentencepiece 0.2.2" for both the patched and the unpatched
# build, so the hash is the only thing that discriminates them.
readonly PATCHED_SPM_SHA256="ff5744ae620074154e558dcf1d48d5f3e32934047b40e9f48f9333f0d7fc94d2"

RES="${SCR}/results_${MODEL_NAME}_fp64"
BASE="${RES}/tokenizers/langspec_base_tokenizer.json"
CORPUS="${SCR}/wili_corpus_shared"
OUT="${SCR}/${MODEL_NAME}_fp64.unilid"

echo "=== WiLI Mistral-Nemo retrain (vocab ${VOCAB_SIZE}) at $(date) on $(hostname) ==="
echo "commit: $(git -C ${REPO} rev-parse HEAD)"

RESOLVED_SPM="$(command -v spm_train || true)"
if [ -z "${RESOLVED_SPM}" ]; then
  echo "FATAL: spm_train not found on PATH" >&2; exit 1
fi
RESOLVED_SHA="$(sha256sum "${RESOLVED_SPM}" | awk '{print $1}')"
echo "spm_train path:   ${RESOLVED_SPM}"
echo "spm_train sha256: ${RESOLVED_SHA}"
if [ "${RESOLVED_SHA}" != "${PATCHED_SPM_SHA256}" ]; then
  echo "FATAL: resolved spm_train is not the patched fp64 build" >&2
  echo "       expected ${PATCHED_SPM_SHA256}" >&2
  exit 1
fi
echo "spm_train confirmed PATCHED (fp64, d0208d9 + c5921a2)."

cd "${REPO}"
${PYTHON} -m analysis.preflight_wili_base \
  --base "${BASE}" --expect-vocab "${VOCAB_SIZE}" --expect-sha256 "${BASE_SHA256}" \
  --results-dir "${RES}" --corpus "${CORPUS}" --output-container "${OUT}"

cd "${REPO}/UNILID"
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
cd "${REPO}"
${PYTHON} -u -m analysis.inspect_variant_models "${OUT}" \
  -o "outputs/rerelease/wili_${MODEL_NAME}_fp64_inspect.json"
echo "=== Finished at $(date) ==="
