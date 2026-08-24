#!/bin/bash
#SBATCH --job-name=w-wili100k-defaults
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
# PHASE 3 GATING RUN of the WiLI regeneration plan.
#
# Trains a 100,000-entry base vocabulary FROM THE WiLI CORPUS with default
# settings, then the per-language weights over it, and compares the base
# vocabulary against the one inside the published wili_100k_500.unilid. The
# four vocabulary sizes the plan needs (10k / 20k / 50k / 200k) have no
# container to lift a vocabulary from, so each must have one trained here and
# nothing verifies those four directly. 100k is the one size where a stored
# vocabulary exists to compare against, which makes this the only evidence
# available that the procedure reproduces:
#   Match    -> the four sizes are the published models.
#   No match -> the four are NEW models built by the published procedure, and
#               tab:vocab_size_efficiency has to say so.
# Recorded either way. slurm_wili_train_vocabsize.sh is GATED on this result
# and must not be submitted before it is read.
#
# ###########################################################################
# # DELIBERATE EXCEPTION TO THE MANDATORY-FLAGS RULE. READ BEFORE EDITING.  #
# ###########################################################################
# Every other WiLI job in this repository must pass BOTH --results-dir and
# --base-tokenizer-path, because train.py:450-452 otherwise defaults the base
# path to results_<vocab//1000>k/tokenizers/, the reuse test at train.py:455
# fails, and train.py:465-492 SILENTLY TRAINS A FRESH VOCABULARY and reports
# success — estimating every row over a different vocabulary than the model
# being replaced.
#
# THIS JOB OMITS --base-tokenizer-path AND PASSES --no-reuse-base ON PURPOSE:
# training a fresh base vocabulary is the entire point of the plan's Phase 3,
# which specifies "Same command as Phase 1 without --reuse-base and without
# --base-tokenizer-path, with --vocab-size set per row and an absolute
# --results-dir per row". The failure mode the rule guards against — a fresh
# vocabulary trained without anyone noticing — is here the intended product,
# and it is made visible three ways: --results-dir IS still passed and is
# absolute; the preflight asserts that no base tokenizer exists under it, so
# nothing can be picked up by --reuse-base; and the trained vocabulary is
# checked against the published one before anything downstream uses it.
# The plan's Phase 3 is the authority for this exception, nothing else.
# ###########################################################################
#
# "Default settings" means exactly what the author confirmed on 2026-08-21 and
# what train.py:465-492 does: --base-training-method hf (HuggingFace's
# UnigramTrainer, the default) over sample_corpus(..., 10,000 lines per
# language, the default). WiLI holds 500 lines per language, so that cap never
# binds and the default already trains on the entire training set.
# --byte-level is passed explicitly and is also the default (train.py:301-306).
#
# The corpus is the same wili_corpus_shared the Phase 1 retrains used, built
# once with train.prepare_corpus, which writes every line of the WiLI training
# split as {lang}_train.txt and performs no train/val split — so it is the same
# data --wili-dir would produce, byte-identically shared with Phase 1.
#
# Two things still differ from the published model and are recorded with the
# result: UNILID 0.3.0's special-token handling, and the patched fp64 spm_train
# (fork commits d0208d9 + c5921a2).
#
# Cost note: the gating check needs only the base vocabulary. The per-language
# step is run anyway because it is roughly 20 minutes on WiLI (Phase 1 measured
# 19 to 25 minutes per model) and produces a packed 100k-defaults container that
# can be compared against the Phase 1 retrain directly. 6h is headroom; the base
# training time at 100,000 entries on 64 MB of text has not been measured here.
PYTHON="/users/cmeister747/.pyenv/versions/3.11.5/bin/python3"
export PATH="/users/cmeister747/.local/bin:${PATH}"   # patched fp64 spm_train
REPO="/users/cmeister747/unilid_analysis"
SCR="/capstor/scratch/cscs/cmeister747/unilid_analysis"

MODEL_NAME="wili_100k_defaults"
VOCAB_SIZE=100000
# sha256 of the patched fp64 spm_train installed 2026-07-27 (fork commits
# d0208d9 + c5921a2), recorded in slurm_wili_train_fp32null.sh. `spm_train
# --version` prints "sentencepiece 0.2.2" for both the patched and unpatched
# builds, so the hash is the only thing that discriminates them.
readonly PATCHED_SPM_SHA256="ff5744ae620074154e558dcf1d48d5f3e32934047b40e9f48f9333f0d7fc94d2"

RES="${SCR}/results_${MODEL_NAME}_fp64"
BASE="${RES}/tokenizers/langspec_base_tokenizer.json"   # MUST NOT EXIST; trained here
CORPUS="${SCR}/wili_corpus_shared"
OUT="${SCR}/${MODEL_NAME}_fp64.unilid"
PUBLISHED="${SCR}/wili_assets/wili_100k_500.unilid"

echo "=== WiLI 100k default-vocabulary training at $(date) on $(hostname) ==="
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

if [ ! -f "${PUBLISHED}" ]; then
  echo "FATAL: published container missing at ${PUBLISHED}; the gating" >&2
  echo "       comparison has nothing to compare against." >&2
  exit 1
fi

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
  echo "FATAL: train.py did not write a base tokenizer to ${BASE}." >&2
  echo "       It defaults to <results-dir>/tokenizers/langspec_base_tokenizer.json" >&2
  echo "       (train.py:450-452); if it is not there, the vocabulary this run" >&2
  echo "       trained cannot be located and the gating check cannot be run." >&2
  exit 1
fi

echo "=== packing at $(date) ==="
${PYTHON} convert.py "${RES}" -o "${OUT}"

echo "=== PHASE 3 GATING CHECK at $(date) ==="
cd "${REPO}"
set +e
${PYTHON} -u -m analysis.wili_vocab_repro_check \
  --trained "${BASE}" --container "${PUBLISHED}" \
  -o "outputs/rerelease/wili_vocab_repro_check.json"
REPRO_RC=$?
set -e
case "${REPRO_RC}" in
  0) echo "GATE: MATCH. The four vocabulary sizes may be reported as the published models." ;;
  1) echo "GATE: NO MATCH. The four vocabulary sizes are NEW models built by the published procedure; the table must say so." ;;
  *) echo "FATAL: the reproducibility check aborted (exit ${REPRO_RC}) without deciding anything." >&2; exit "${REPRO_RC}" ;;
esac

echo "=== post-training checks at $(date) ==="
${PYTHON} -u -m analysis.inspect_variant_models "${OUT}" \
  -o "outputs/rerelease/wili_${MODEL_NAME}_fp64_inspect.json"
echo "=== Finished at $(date); repro check exit ${REPRO_RC} ==="
