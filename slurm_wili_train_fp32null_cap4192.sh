#!/bin/bash
#SBATCH --job-name=w-wili100k-fp32cap
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
# SECOND NULL ARM for the wili_100k_500 fp64 retrain: the SENTENCE-LENGTH-CAP
# arm (see slurm_wili_train_fp32null.sh for the first null arm, and
# slurm_wili_train_fp64.sh for the arm under test).
#
# HYPOTHESIS (outputs/rerelease/wili_fp32null_verdict.json,
# "upstream_sentence_cap_split"): the stored (published) wili_100k_500 was
# trained with sentencepiece's UPSTREAM DEFAULT max_sentence_length = 4192
# bytes, whereas this pipeline overrides it to 1,000,000. sentencepiece SKIPS
# an over-cap line outright rather than truncating it
# (sentencepiece src/trainer_interface.cc: `++too_long_lines; continue;`), so
# training at the default sees fewer tokens per language and lands on HIGHER
# floors -- which is the direction the stored model's floors actually sit. Of
# the 106 languages that fail the fp32-null comparison, 65 have at least one
# training line over 4192 raw bytes, and 0 of the 129 passing languages do.
# If this retrain collapses those failures against the transformed stored
# model, the cap explains most of the residual.
#
# WHAT DIFFERS FROM WHICH ARM -- this is the point of the design:
#   * vs. the fp64 arm (slurm_wili_train_fp64.sh): TWO things differ, the
#     unpatched fp32 spm_train AND the 4192-byte cap. This arm alone therefore
#     cannot separate build from cap.
#   * vs. the fp32-null arm (slurm_wili_train_fp32null.sh): exactly ONE thing
#     differs, the cap (1,000,000 -> 4192). Everything else -- UNILID 0.3.0
#     trainer code, the same unpatched fp32 spm_train binary, corpus, vocab
#     size, base tokenizer content, every other CLI flag -- is identical.
#     The cap's effect is read off THIS pair, not off the fp64 comparison.
#
# The cap is set through train.py's --max-sentence-length, added 2026-08-24 for
# this arm (patches/unilid_max_sentence_length.patch). It defaults to
# constants.SP_MAX_SENTENCE_LENGTH = 1,000,000, the value that was hardcoded in
# language_specific_trainer.py before, so every other job's spm_train argv is
# byte-identical to what it was. Nothing is sed-ed, copied, or shadowed: this
# job runs the same checkout every other job runs.
#
# NOTE ON UNITS: the cap applies to the BYTE-LEVEL ENCODED training file that
# write_hf_bytelevel_corpus writes, not to the raw corpus line. Non-ASCII bytes
# map to two-byte UTF-8 characters there, so a non-Latin-script line is dropped
# at roughly half the raw byte count and MORE lines will be skipped than the 367
# raw-byte-over-cap lines the verdict counted. That is the same arithmetic the
# published run would have been subject to, so it is the arm, not a defect --
# but it must be stated when the result is interpreted.
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
# train.py defaults the base path to results_<vocab//1000>k/tokenizers/, the
# reuse test fails, and train.py silently trains a FRESH vocabulary and reports
# success, estimating every row over a different vocabulary than the model
# being replaced.
PYTHON="/users/cmeister747/.pyenv/versions/3.11.5/bin/python3"

# --- fp32 (unpatched) spm_train on PATH, ahead of everything else ---------
FP32_SPM_BIN_DIR="/capstor/scratch/cscs/cmeister747/unilid_analysis/sp_fp32_env/bin"
export PATH="${FP32_SPM_BIN_DIR}:${PATH}"

SCR="/capstor/scratch/cscs/cmeister747/unilid_analysis"
REPO="/users/cmeister747/unilid_analysis"
MODEL_NAME="wili_100k_500"
VOCAB_SIZE=100000
# sentencepiece's upstream default (src/sentencepiece_model.proto:
# `optional int32 max_sentence_length = 18 [default = 4192];`). This is the
# whole experimental manipulation; it is written once, here.
MAX_SENTENCE_LENGTH=4192
RES="${SCR}/results_${MODEL_NAME}_fp32null_cap4192"
FP64_RES="${SCR}/results_${MODEL_NAME}_fp64"
FP64_BASE="${FP64_RES}/tokenizers/langspec_base_tokenizer.json"
BASE="${RES}/tokenizers/langspec_base_tokenizer.json"
CORPUS="${SCR}/wili_corpus_shared"
OUT="${SCR}/${MODEL_NAME}_fp32null_cap4192.unilid"

# --- discriminator: hard-fail if the resolved spm_train is the patched one -
# sha256 of the two known builds, recorded at the time each was produced.
# PATCHED_SHA256: ~/.local/bin/spm_train, installed 2026-07-27, fork commits
#   d0208d9 + c5921a2.
# UNPATCHED_FP32_SHA256: the build the first null arm installed, from commit
#   2b7ec9b8e86a61f7772236471f948850872d8918.
# `spm_train --version` prints "sentencepiece 0.2.2" for BOTH builds (the
# fork did not bump the version string across the patch), so version alone
# cannot discriminate; sha256 of the resolved binary is the only reliable
# check.
readonly PATCHED_SHA256="ff5744ae620074154e558dcf1d48d5f3e32934047b40e9f48f9333f0d7fc94d2"
readonly UNPATCHED_FP32_SHA256="4dd4a2e9e35b2731bca8b7124e101e5a3c65474cfb22f591f83b41135013217f"
# sha256 of the base tokenizer this model's rows must be estimated over,
# recorded in outputs/rerelease/wili_fp32null_verdict.json
# (fp32null_provenance.base_tokenizer) when the first null arm ran.
readonly EXPECTED_BASE_SHA256="5fa5342a011dca96b170785c220d21505ae91844b2050a5199959c62aeadfbe8"

echo "=== WiLI fp32-null CAP-${MAX_SENTENCE_LENGTH} retrain ${MODEL_NAME} (vocab ${VOCAB_SIZE}) at $(date) on $(hostname) ==="
echo "commit:        $(git -C "${REPO}" rev-parse HEAD)"
echo "UNILID commit: $(git -C "${REPO}/UNILID" rev-parse HEAD)"
echo "UNILID working-tree diff vs. that commit (the --max-sentence-length change):"
git -C "${REPO}/UNILID" diff --stat | sed 's/^/  /'
echo "UNILID file hashes:"
sha256sum "${REPO}/UNILID/train.py" \
          "${REPO}/UNILID/unilid/constants.py" \
          "${REPO}/UNILID/unilid/trainers/language_specific_trainer.py" | sed 's/^/  /'

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

# --- the cap must actually be reachable from the CLI ----------------------
# Without the --max-sentence-length change in the checkout, train.py would
# reject the flag and this job would die at argparse; check first so the
# failure names the cause instead of an argparse usage dump.
if ! ${PYTHON} "${REPO}/UNILID/train.py" --help 2>&1 | grep -q -- "--max-sentence-length"; then
  echo "FATAL: ${REPO}/UNILID/train.py does not expose --max-sentence-length." >&2
  echo "       This arm cannot set the cap without it; apply" >&2
  echo "       patches/unilid_max_sentence_length.patch and rerun." >&2
  exit 1
fi
echo "train.py exposes --max-sentence-length."

if [ ! -f "${FP64_BASE}" ]; then
  echo "FATAL: base tokenizer missing at ${FP64_BASE}" >&2; exit 1
fi
FP64_BASE_SHA="$(sha256sum "${FP64_BASE}" | awk '{print $1}')"
if [ "${FP64_BASE_SHA}" != "${EXPECTED_BASE_SHA256}" ]; then
  echo "FATAL: ${FP64_BASE} has sha256 ${FP64_BASE_SHA}, not the" >&2
  echo "       ${EXPECTED_BASE_SHA256} the first null arm was trained over." >&2
  echo "       The two arms would not share a vocabulary; refusing to run." >&2
  exit 1
fi
mkdir -p "${RES}/tokenizers"
if [ ! -f "${BASE}" ]; then
  cp "${FP64_BASE}" "${BASE}"
  echo "Copied base tokenizer: ${FP64_BASE} -> ${BASE}"
  echo "  sha256: $(sha256sum "${BASE}" | awk '{print $1}')"
fi
# skip-existing-langs defaults to true; that is safe only because this directory
# holds no per-language rows. A row trained at a DIFFERENT cap sitting here
# would be loaded and packed unchecked -- the loader validates token order, not
# which cap produced the counts.
if compgen -G "${RES}/tokenizers/langspec_soft_*" > /dev/null || \
   compgen -G "${RES}/tokenizers/langspec_sp_*" > /dev/null; then
  echo "FATAL: ${RES}/tokenizers already holds per-language rows; refusing to" >&2
  echo "       resume, since --skip-existing-langs would reuse them unchecked." >&2
  exit 1
fi
if [ ! -d "${CORPUS}" ]; then echo "FATAL: corpus missing at ${CORPUS}" >&2; exit 1; fi

cd "${REPO}/UNILID"

${PYTHON} train.py \
  --corpus-dir "${CORPUS}" --reuse-corpus \
  --vocab-size "${VOCAB_SIZE}" --byte-level \
  --per-lang-counts-method sp \
  --max-sentence-length "${MAX_SENTENCE_LENGTH}" \
  --results-dir "${RES}" \
  --base-tokenizer-path "${BASE}" --reuse-base \
  --lang-batch-size 20

echo "=== packing at $(date) ==="
${PYTHON} convert.py "${RES}" -o "${OUT}"

echo "=== post-training checks at $(date) ==="
cd "${REPO}"
${PYTHON} -u -m analysis.inspect_variant_models "${OUT}" \
  -o "outputs/rerelease/wili_${MODEL_NAME}_fp32null_cap4192_inspect.json"

# --- did the manipulation actually happen? --------------------------------
# Run LAST, after every artifact exists, so a negative answer marks the job
# FAILED without throwing away the run: an arm whose cap never engaged is an
# expensive duplicate of the fp32-null arm, and must not be read as one.
echo "--- cap engagement ---"
SUMMARY="${RES}/training_summary.json"
if [ ! -f "${SUMMARY}" ]; then
  echo "FATAL: ${SUMMARY} not written; cannot confirm the cap." >&2; exit 1
fi
RECORDED_CAP="$(${PYTHON} -c "import json,sys; print(json.load(open(sys.argv[1]))['method']['max_sentence_length'])" "${SUMMARY}")"
echo "training_summary.json method.max_sentence_length = ${RECORDED_CAP}"
if [ "${RECORDED_CAP}" != "${MAX_SENTENCE_LENGTH}" ]; then
  echo "FATAL: the run recorded cap ${RECORDED_CAP}, not ${MAX_SENTENCE_LENGTH}." >&2
  exit 1
fi
ERRLOG="${SCR}/logs/wili_${SLURM_JOB_NAME}_${SLURM_JOB_ID}.err"
if [ ! -f "${ERRLOG}" ]; then
  echo "FATAL: expected this job's stderr log at ${ERRLOG}; cannot confirm" >&2
  echo "       sentencepiece skipped any line. Do not use this run until the" >&2
  echo "       skip counts are checked by hand." >&2
  exit 1
fi
SKIP_LINES="$(grep -c 'too long sentences' "${ERRLOG}" || true)"
echo "spm_train 'Skipped N too long sentences.' lines in ${ERRLOG}: ${SKIP_LINES}"
grep -o 'Skipped [0-9]* too long sentences.' "${ERRLOG}" | sort | uniq -c | sed 's/^/  /' || true
if [ "${SKIP_LINES}" -eq 0 ]; then
  echo "FATAL: not one language skipped a line at cap ${MAX_SENTENCE_LENGTH}." >&2
  echo "       The manipulation did not engage; this run is a duplicate of the" >&2
  echo "       fp32-null arm and must not be reported as the cap arm." >&2
  exit 1
fi
echo "=== Finished at $(date) ==="
