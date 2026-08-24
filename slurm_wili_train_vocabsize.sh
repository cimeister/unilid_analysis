#!/bin/bash
#SBATCH --job-name=w-wili-vocabsize
#SBATCH --account=infra01
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=400G
#SBATCH --time=06:00:00
#SBATCH --output=/capstor/scratch/cscs/cmeister747/unilid_analysis/logs/wili_%x_%j.out
#SBATCH --error=/capstor/scratch/cscs/cmeister747/unilid_analysis/logs/wili_%x_%j.err
set -euo pipefail
#
# PHASE 3 of the WiLI regeneration plan: the four vocabulary sizes of
# tab:vocab_size_efficiency that have no container anywhere (10k / 20k / 50k /
# 200k). Each trains its own base vocabulary from the WiLI corpus with default
# settings, then its per-language weights over it.
#
# Takes VOCAB_SIZE from the environment (--export). One of 10000 20000 50000
# 200000; anything else aborts rather than being accepted as a typo.
# Submit with --job-name=w-wili-${K}k-defaults so logs are distinguishable.
#
# ###########################################################################
# # GATED. DO NOT SUBMIT BEFORE slurm_wili_train_100k_defaults.sh HAS RUN.  #
# ###########################################################################
# None of these four sizes has an artifact to verify against. The plan's Phase 3
# makes the 100k size the gating check — the one vocabulary size where a stored
# container exists to compare a freshly trained vocabulary against — and its
# outcome decides how these four are DESCRIBED in the paper:
#   match    -> they are the published models;
#   no match -> they are NEW models built by the published procedure.
# This script refuses to start until outputs/rerelease/wili_vocab_repro_check.json
# exists, and prints its verdict into this job's log so the result these four
# are reported under is recorded with them.
# ###########################################################################
#
# ###########################################################################
# # DELIBERATE EXCEPTION TO THE MANDATORY-FLAGS RULE. READ BEFORE EDITING.  #
# ###########################################################################
# --base-tokenizer-path is omitted and --no-reuse-base is passed ON PURPOSE:
# training a fresh base vocabulary is the entire point of Phase 3, which
# specifies "Same command as Phase 1 without --reuse-base and without
# --base-tokenizer-path, with --vocab-size set per row and an absolute
# --results-dir per row". Everywhere else in this repository the omission is the
# single most dangerous defect available, because train.py:450-452 defaults the
# base path, the reuse test at train.py:455 fails, and train.py:465-492 silently
# trains a fresh vocabulary and reports success. Here that is the intended
# product. --results-dir IS still passed and is absolute, and the preflight
# asserts no base tokenizer exists under it so nothing can be picked up by
# --reuse-base. The plan's Phase 3 is the authority for this exception.
# ###########################################################################
#
# "Default settings" is train.py's own default path: --base-training-method hf
# (HuggingFace's UnigramTrainer) over sample_corpus(..., 10,000 lines per
# language). WiLI holds 500 lines per language, so the cap never binds and the
# default trains on the entire training set. --byte-level is passed explicitly
# and is also the default.
#
# The containers are named wili_<K>k_defaults_fp64.unilid, NOT wili_<K>k_500, so
# a model trained here can never be mistaken on disk for a published one.
#
# 6h is headroom. Phase 1's per-language step took 19 to 25 minutes per model at
# 100,000 to 151,670 entries; the base-vocabulary training time on 64 MB of text
# has not been measured here, and 200,000 entries is the largest of the four.
PYTHON="/users/cmeister747/.pyenv/versions/3.11.5/bin/python3"
export PATH="/users/cmeister747/.local/bin:${PATH}"   # patched fp64 spm_train
REPO="/users/cmeister747/unilid_analysis"
SCR="/capstor/scratch/cscs/cmeister747/unilid_analysis"

# sha256 of the patched fp64 spm_train installed 2026-07-27 (fork commits
# d0208d9 + c5921a2), recorded in slurm_wili_train_fp32null.sh. `spm_train
# --version` prints "sentencepiece 0.2.2" for both builds, so the hash is the
# only thing that discriminates them.
readonly PATCHED_SPM_SHA256="ff5744ae620074154e558dcf1d48d5f3e32934047b40e9f48f9333f0d7fc94d2"
REPRO_RECORD="${REPO}/outputs/rerelease/wili_vocab_repro_check.json"

if [ -z "${VOCAB_SIZE:-}" ]; then
  echo "FATAL: VOCAB_SIZE unset. Submit with" >&2
  echo "       --export=ALL,VOCAB_SIZE=<10000|20000|50000|200000>" >&2
  exit 1
fi
case "${VOCAB_SIZE}" in
  10000)  K="10"  ;;
  20000)  K="20"  ;;
  50000)  K="50"  ;;
  200000) K="200" ;;
  *)
    echo "FATAL: VOCAB_SIZE '${VOCAB_SIZE}' is not one of the four sizes this" >&2
    echo "       plan calls for: 10000 20000 50000 200000." >&2
    exit 1
    ;;
esac

if [ ! -f "${REPRO_RECORD}" ]; then
  echo "FATAL: the Phase 3 gating check has not been run." >&2
  echo "       ${REPRO_RECORD} does not exist." >&2
  echo "       Submit slurm_wili_train_100k_defaults.sh first and read its" >&2
  echo "       verdict; it decides whether these four sizes are the published" >&2
  echo "       models or new models built by the published procedure." >&2
  exit 1
fi

MODEL_NAME="wili_${K}k_defaults"
RES="${SCR}/results_${MODEL_NAME}_fp64"
BASE="${RES}/tokenizers/langspec_base_tokenizer.json"   # MUST NOT EXIST; trained here
CORPUS="${SCR}/wili_corpus_shared"
OUT="${SCR}/${MODEL_NAME}_fp64.unilid"

echo "=== WiLI ${K}k default-vocabulary training at $(date) on $(hostname) ==="
echo "commit: $(git -C ${REPO} rev-parse HEAD)"
echo "--- Phase 3 gating check, as recorded ---"
${PYTHON} -c "
import json,sys
r=json.load(open('${REPRO_RECORD}'))
print('  ordered_token_list_match:', r['ordered_token_list_match'])
print('  first_divergence_index:  ', r['first_divergence_index'])
print('  overlap_tokens:          ', r['overlap_tokens'], 'of', r['container_vocab_size'])
print('  -> these four sizes are', 'THE PUBLISHED MODELS' if r['ordered_token_list_match']
      else 'NEW MODELS BUILT BY THE PUBLISHED PROCEDURE')
"

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
  --require-absent-base "${BASE}" \
  --results-dir "${RES}" --corpus "${CORPUS}" --output-container "${OUT}"

cd "${REPO}/UNILID"
${PYTHON} train.py \
  --corpus-dir "${CORPUS}" --reuse-corpus \
  --vocab-size "${VOCAB_SIZE}" --byte-level \
  --per-lang-counts-method sp \
  --results-dir "${RES}" \
  --no-reuse-base \
  --lang-batch-size 20

if [ ! -f "${BASE}" ]; then
  echo "FATAL: train.py did not write a base tokenizer to ${BASE}" >&2
  exit 1
fi
TRAINED_ENTRIES=$(${PYTHON} -c "
import json; print(len(json.load(open('${BASE}'))['model']['vocab']))")
echo "trained base vocabulary: ${TRAINED_ENTRIES} entries (requested ${VOCAB_SIZE})"
if [ "${TRAINED_ENTRIES}" != "${VOCAB_SIZE}" ]; then
  echo "FATAL: the trained base vocabulary has ${TRAINED_ENTRIES} entries, not" >&2
  echo "       the ${VOCAB_SIZE} this row is defined by. Do not pack this run." >&2
  exit 1
fi

echo "=== packing at $(date) ==="
${PYTHON} convert.py "${RES}" -o "${OUT}"

echo "=== post-training checks at $(date) ==="
cd "${REPO}"
${PYTHON} -u -m analysis.inspect_variant_models "${OUT}" \
  -o "outputs/rerelease/wili_${MODEL_NAME}_fp64_inspect.json"
echo "=== Finished at $(date) ==="
