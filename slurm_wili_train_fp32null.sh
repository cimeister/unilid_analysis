#!/bin/bash
#SBATCH --job-name=w-wili100k-fp32null
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
# NULL ARM for the wili_100k_500 fp64 retrain (see slurm_wili_train_fp64.sh).
#
# Retrains wili_100k_500's per-language weights over its ORIGINAL vocabulary
# with the SAME UNILID 0.3.0 trainer code as the fp64 arm, but with the
# UNPATCHED (fp32) spm_train instead of the patched fp64 build. This isolates
# whether the fp64-arm deltas vs. the stored released model (~102/235
# languages predicted to differ beyond threshold) are a build effect (fp32 vs
# fp64 spm_train) or a genuine defect in the stored model: if this fp32 retrain
# matches the stored model closely while the fp64 retrain does not, the fp64
# deltas are the build effect, not corruption.
#
# Exactly one change from slurm_wili_train_fp64.sh's wili_100k_500 invocation:
# spm_train is the unpatched build. Everything else (UNILID 0.3.0 code,
# corpus, vocab size, base tokenizer content, CLI flags) is identical.
#
# fp32 spm_train provenance: built 2026-08-23 from
# https://github.com/cimeister/sentencepiece.git commit
# 2b7ec9b8e86a61f7772236471f948850872d8918 ("Logging + num iteration ranges"),
# which is the PARENT of d0208d9 (the first fp64 patch commit, "Compute the
# trainer's forward-backward in double precision"). c5921a2 (the second fp64
# patch, "Fail loudly on non-finite expected counts instead of zeroing them")
# is also absent. Built with the exact recipe in SETUP.md #8
# (cmake -DCMAKE_BUILD_TYPE=Release -DSPM_ENABLE_SHARED=OFF; make spm_train)
# into an ISOLATED prefix that nothing else on the cluster depends on:
#   /capstor/scratch/cscs/cmeister747/unilid_analysis/sp_fp32_env/bin/spm_train
# The patched binary at ~/.local/bin/spm_train (used by every other job) was
# not touched, rebuilt, or reinstalled to produce this build.
#
# --results-dir and --base-tokenizer-path are BOTH mandatory. Without them
# train.py:450-452 defaults the base path to results_<vocab//1000>k/tokenizers/,
# the reuse test at train.py:455 fails, and train.py:465-492 silently trains a
# FRESH vocabulary and reports success, estimating every row over a different
# vocabulary than the model being replaced.
PYTHON="/users/cmeister747/.pyenv/versions/3.11.5/bin/python3"

# --- fp32 (unpatched) spm_train on PATH, ahead of everything else ---------
FP32_SPM_BIN_DIR="/capstor/scratch/cscs/cmeister747/unilid_analysis/sp_fp32_env/bin"
export PATH="${FP32_SPM_BIN_DIR}:${PATH}"

SCR="/capstor/scratch/cscs/cmeister747/unilid_analysis"
MODEL_NAME="wili_100k_500"
VOCAB_SIZE=100000
RES="${SCR}/results_${MODEL_NAME}_fp32null"
FP64_RES="${SCR}/results_${MODEL_NAME}_fp64"
FP64_BASE="${FP64_RES}/tokenizers/langspec_base_tokenizer.json"
BASE="${RES}/tokenizers/langspec_base_tokenizer.json"
CORPUS="${SCR}/wili_corpus_shared"
OUT="${SCR}/${MODEL_NAME}_fp32null.unilid"

# --- discriminator: hard-fail if the resolved spm_train is the patched one -
# sha256 of the two known builds, recorded at the time each was produced.
# PATCHED_SHA256: ~/.local/bin/spm_train, installed 2026-07-27, fork commits
#   d0208d9 + c5921a2 (measured on 2026-08-23; file mtime and hash unchanged
#   since the 2026-07-27 install, confirming this run did not touch it).
# UNPATCHED_FP32_SHA256: the build this script installs, from commit
#   2b7ec9b8e86a61f7772236471f948850872d8918 (measured immediately after
#   the 2026-08-23 build, before this job existed).
# `spm_train --version` prints "sentencepiece 0.2.2" for BOTH builds (the
# fork did not bump the version string across the patch), so version alone
# cannot discriminate; sha256 of the resolved binary is the only reliable
# check.
readonly PATCHED_SHA256="ff5744ae620074154e558dcf1d48d5f3e32934047b40e9f48f9333f0d7fc94d2"
readonly UNPATCHED_FP32_SHA256="4dd4a2e9e35b2731bca8b7124e101e5a3c65474cfb22f591f83b41135013217f"

echo "=== WiLI fp32-null retrain ${MODEL_NAME} (vocab ${VOCAB_SIZE}) at $(date) on $(hostname) ==="
echo "commit:    $(git -C /users/cmeister747/unilid_analysis rev-parse HEAD)"

RESOLVED_SPM="$(command -v spm_train || true)"
if [ -z "${RESOLVED_SPM}" ]; then
  echo "FATAL: spm_train not found on PATH (expected ${FP32_SPM_BIN_DIR}/spm_train)" >&2
  exit 1
fi
RESOLVED_SHA="$(sha256sum "${RESOLVED_SPM}" | awk '{print $1}')"
echo "spm_train path:    ${RESOLVED_SPM}"
echo "spm_train version: $(spm_train --version 2>&1 | head -1)"
echo "spm_train sha256:  ${RESOLVED_SHA}"

if [ "${RESOLVED_SHA}" = "${PATCHED_SHA256}" ]; then
  echo "FATAL: resolved spm_train matches the PATCHED fp64 build's sha256" >&2
  echo "       (${PATCHED_SHA256}). This job must run against the UNPATCHED" >&2
  echo "       fp32 build; PATH resolution picked up the wrong binary." >&2
  exit 1
fi
if [ "${RESOLVED_SHA}" != "${UNPATCHED_FP32_SHA256}" ]; then
  echo "FATAL: resolved spm_train sha256 (${RESOLVED_SHA}) matches NEITHER the" >&2
  echo "       known patched build nor the known unpatched fp32 build. Refusing" >&2
  echo "       to guess which trainer this is; investigate before rerunning." >&2
  exit 1
fi
echo "spm_train confirmed UNPATCHED (fp32, pre-d0208d9/c5921a2)."

if [ ! -f "${FP64_BASE}" ]; then
  echo "FATAL: base tokenizer missing at ${FP64_BASE}" >&2; exit 1
fi
mkdir -p "${RES}/tokenizers"
if [ ! -f "${BASE}" ]; then
  cp "${FP64_BASE}" "${BASE}"
  echo "Copied base tokenizer: ${FP64_BASE} -> ${BASE}"
  echo "  sha256: $(sha256sum "${BASE}" | awk '{print $1}')"
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
  -o "outputs/rerelease/wili_${MODEL_NAME}_fp32null_inspect.json"
echo "=== Finished at $(date) ==="
