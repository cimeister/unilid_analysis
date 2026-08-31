#!/bin/bash
#SBATCH --job-name=cld3-subset-train
#SBATCH --account=infra01
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=400G
#SBATCH --time=12:00:00
# 12 h is the partition maximum and is deliberate. A login-node probe of the
# same base fit measured 36.8 s on 4.04 MB and 87.8 s on 16.16 MB of this
# corpus, which fits a 0.63 exponent and extrapolates to ~6 min at the real
# 151.8 MB. That extrapolation runs 9.4x beyond its largest fitted point, and it
# is contradicted by the one large measurement of the SAME code path at the same
# vocabulary size: results_wili_100k_defaults_fp64 took 3,733.95 s on 64.1 MB,
# where the power law predicts 206 s. Two points spanning 4x cannot separate a
# fixed per-run cost from a size-dependent one, and an unmodelled fixed cost
# biases the exponent down. Scaling linearly from the WiLI measurement instead
# gives ~2.5 h for the base fit plus 0.2-0.6 h of row estimation. Both readings
# fit inside 12 h; only the optimistic one fits inside 4 h, so the walltime is
# set by the pessimistic one.
#SBATCH --output=/capstor/scratch/cscs/cmeister747/unilid_analysis/logs/cld3sub_%x_%j.out
#SBATCH --error=/capstor/scratch/cscs/cmeister747/unilid_analysis/logs/cld3sub_%x_%j.err
set -euo pipefail
#
# Train ONE subset-vocabulary UniLID model for the corrected generation's
# CLD3-subset columns of tab:lid_main.
#
# Takes SUBSET (83 | 80 | 77) and optionally RESUME=1 from the environment
# (--export). One model per benchmark subset:
#   83 -> GlotLID-C column, corpus cld3_corpus_83 (99 lang_Script corpora)
#   80 -> UDHR column,      corpus cld3_corpus_80 (94)
#   77 -> FLORES-200 column,corpus cld3_corpus_77 (93)
#
# WHY A FROM-SCRATCH BASE VOCABULARY IS THE POINT OF THIS JOB
# -----------------------------------------------------------
# The author answered on 2026-08-31 that "the base tokenizer used for the CLD3
# columns in the plain implementation of UniLID were trained on that subset of
# languages, so it is indeed a different vocabulary than the model used for the
# full dataset columns" (outputs/rerelease/cld_subset_gate_2026-08-31.md
# section 10.1). A restricted argmax over the full 1,940-label model is
# therefore NOT the computation behind those cells, and this job builds the
# model the answer describes: a base vocabulary fitted to the subset corpora
# alone, then one per-language row over it.
#
# The author ruled on 2026-08-31 that exact reproduction of the published cells
# is NOT the bar -- "Just approximately reproduce them. Again, there is no need
# to stick exactly to what was done in the first version." Every parameter below
# is therefore the stated default rather than a reconstruction, and each is
# recorded with its reason in outputs/rerelease/cld3_regenerated_2026-09-01.md.
#
# ###########################################################################
# # DELIBERATE EXCEPTION TO THE MANDATORY-FLAGS RULE. READ BEFORE EDITING.  #
# ###########################################################################
# Every job in this repository that retrains rows over an EXISTING vocabulary
# must pass BOTH --results-dir and --base-tokenizer-path. With NEITHER passed,
# train.py:393 defaults results_dir to results_<vocab//1000>k and train.py:473
# defaults the base path to <results-dir>/tokenizers/, so the reuse test at
# train.py:478 fails and train.py:488-515 SILENTLY TRAINS A FRESH VOCABULARY and
# reports success. (Passing --results-dir alone, as this job does, already pins
# the directory; the second flag is what pins the file inside it.)
#
# The FRESH branch of this job omits --base-tokenizer-path and passes
# --no-reuse-base on purpose, on the precedent of
# slurm_wili_train_100k_defaults.sh: a fresh subset vocabulary is the product,
# not the failure. It is made visible three ways -- --results-dir IS passed and
# is absolute; analysis/preflight_cld3_subset.py asserts that no base tokenizer
# and no per-language row exists under it, so nothing can be picked up; and the
# vocabulary that gets trained is written to a known path and checked before
# packing.
#
# The RESUME branch (RESUME=1, after a walltime kill) passes
# --base-tokenizer-path and --reuse-base and requires the base to be PRESENT, so
# the rows this run adds are estimated over the same vocabulary as the rows the
# killed run already wrote. Retraining the base under existing rows is the one
# corruption this job family can produce, and the preflight refuses both halves
# of it.
#
# The RESUME branch also DEPARTS from slurm_wili_train_fp64.sh:43-51, which
# refuses outright to run when per-language rows exist, on the ground that "the
# loader validates token order but not real-token mass" and a pre-0.3.0 row left
# in place would be packed uncorrected. That guard protects a job that REPLACES
# rows from a previous generation. Here every row is written by 0.3.0 in the
# first attempt of the same job, and the thing the template could not check --
# that the base under those rows is the right base -- is now checked directly by
# analysis/preflight_cld3_subset.py's check_base_provenance against the base
# tokenizer's own record of the corpora it was fitted on. What is still NOT
# checked on resume is each existing row's real-token mass; the packed
# container's mass is gated after convert.py instead, before anything evaluates
# it.
# ###########################################################################
#
# DEFAULTS, AND WHY
#   --vocab-size 100000        paper/submission.tex: "we use a vocabulary size
#                              of 100k" unless otherwise specified. Nothing in
#                              the record states a subset-model vocabulary, so
#                              the paper's general default is the stated one.
#                              The base sample (871k-927k lines, ~150 MB) is far
#                              above anything sentencepiece's or HF's own
#                              constraints would bind at 100,000 pieces.
#   --base-training-method hf  train.py's default (HuggingFace UnigramTrainer),
#                              the method behind every model in this repository.
#                              Passed implicitly; recorded in training_summary.
#   --per-lang-counts-method sp  train.py's default; the method every prior run
#                              here used, and the one the patched fp64 spm_train
#                              serves.
#   --byte-level               train.py's default.
#   --max-base-samples-per-lang  train.py's default 10,000 lines/language.
#   --max-sentence-length      NOT passed: the repo default
#                              constants.SP_MAX_SENTENCE_LENGTH = 1,000,000
#                              keeps every training line, which is what every
#                              GlotLID-C-scale run here has used. The effective
#                              value lands in training_summary.json.
#   --seed 42                  train.py's default.
#   --lang-batch-size 20       what every full-scale run in this repository uses
#                              (results_apertus200k, results_qwen3_8b_fp64, ...).
#                              Orchestration only; it changes no number.
#   corpus                     the shared draw at results_apertus200k/corpus,
#                              subsetted by analysis/build_cld3_subset_corpus.py.
#                              The monolithic train.txt the released model was
#                              trained from is gone and the author ruled a fresh
#                              draw acceptable.
PYTHON="/users/cmeister747/.pyenv/versions/3.11.5/bin/python3"
export PATH="/users/cmeister747/.local/bin:${PATH}"   # patched fp64 spm_train
REPO="/users/cmeister747/unilid_analysis"
SCR="/capstor/scratch/cscs/cmeister747/unilid_analysis"

# HuggingFace's UnigramTrainer (the base fit) sizes its rayon pool from the
# machine's visible core count, not from this job's cgroup: these nodes report
# 288 cores against the 64 CPUs allocated here, so the default oversubscribes by
# 4.5x. Pinned to the allocation, as slurm_cld_subset_glotlidc.sh already does
# for the scorer. The login-node probe this job's walltime was sized against ran
# unpinned on 48 visible cores, so leaving this unset would not reproduce it.
export RAYON_NUM_THREADS="${SLURM_CPUS_PER_TASK:-64}"

: "${SUBSET:?FATAL: SUBSET must be set to 83, 80 or 77 via --export}"
RESUME="${RESUME:-0}"
case "${SUBSET}" in
  83|80|77) ;;
  *) echo "FATAL: SUBSET=${SUBSET} is not one of 83, 80, 77" >&2; exit 1 ;;
esac
# Anything but 0 or 1 would otherwise fall into the FRESH branch silently, and
# "RESUME=true" is the obvious way to get that wrong.
case "${RESUME}" in
  0|1) ;;
  *) echo "FATAL: RESUME=${RESUME} is not 0 or 1" >&2; exit 1 ;;
esac

VOCAB_SIZE=100000
# sha256 of the patched fp64 spm_train installed 2026-07-27 (fork commits
# d0208d9 + c5921a2), carried from slurm_wili_train_fp32null.sh. `spm_train
# --version` prints "sentencepiece 0.2.2" for both the patched and unpatched
# builds, so the hash is the only thing that discriminates them.
readonly PATCHED_SPM_SHA256="ff5744ae620074154e558dcf1d48d5f3e32934047b40e9f48f9333f0d7fc94d2"

MODEL_NAME="cld3sub${SUBSET}"
RES="${SCR}/results_${MODEL_NAME}"
BASE="${RES}/tokenizers/langspec_base_tokenizer.json"
CORPUS="${SCR}/cld3_corpus_${SUBSET}"
MANIFEST="${REPO}/outputs/rerelease/cld3_subset_corpus_manifest_${SUBSET}.json"
OUT="${SCR}/${MODEL_NAME}.unilid"

echo "=== CLD3 subset-${SUBSET} training at $(date) on $(hostname) ==="
echo "commit:  $(git -C "${REPO}" rev-parse HEAD)"
echo "resume:  ${RESUME}"
echo "corpus:  ${CORPUS}"
echo "results: ${RES}"
echo "output:  ${OUT}"

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
if [ "${RESUME}" = "1" ]; then
  ${PYTHON} -m analysis.preflight_cld3_subset \
    --subset "${SUBSET}" --corpus "${CORPUS}" --manifest "${MANIFEST}" \
    --results-dir "${RES}" \
    --require-present-base "${BASE}" --expect-vocab "${VOCAB_SIZE}" \
    --output-container "${OUT}"
else
  ${PYTHON} -m analysis.preflight_cld3_subset \
    --subset "${SUBSET}" --corpus "${CORPUS}" --manifest "${MANIFEST}" \
    --results-dir "${RES}" \
    --require-absent-base "${BASE}" --output-container "${OUT}"
fi

cd "${REPO}/UNILID"
echo "=== train.py at $(date) ==="
if [ "${RESUME}" = "1" ]; then
  ${PYTHON} -u train.py \
    --corpus-dir "${CORPUS}" --reuse-corpus \
    --vocab-size "${VOCAB_SIZE}" --byte-level \
    --per-lang-counts-method sp \
    --results-dir "${RES}" \
    --base-tokenizer-path "${BASE}" --reuse-base \
    --lang-batch-size 20
else
  ${PYTHON} -u train.py \
    --corpus-dir "${CORPUS}" --reuse-corpus \
    --vocab-size "${VOCAB_SIZE}" --byte-level \
    --per-lang-counts-method sp \
    --results-dir "${RES}" \
    --no-reuse-base \
    --lang-batch-size 20
fi

if [ ! -f "${BASE}" ]; then
  echo "FATAL: train.py did not write a base tokenizer to ${BASE}." >&2
  echo "       It defaults to <results-dir>/tokenizers/langspec_base_tokenizer.json" >&2
  echo "       (train.py:473); if it is not there, the vocabulary this run" >&2
  echo "       trained cannot be located and the model cannot be packed." >&2
  exit 1
fi

# FAIL EARLY, BEFORE PACKING. convert.py packs whatever langspec_*.tokenizer.json
# files it finds (unilid/model_io.py:178-181) and counts nothing. A container one
# script variant short still covers every bare ISO code of the subset, so neither
# cld_subset_eval's subset check nor the evaluation job's row-count assertion
# would catch it -- the miss would surface only in the report, after the
# evaluation had spent its own allocation.
echo "=== row-count check at $(date) ==="
cd "${REPO}"
${PYTHON} - "${RES}" "${MANIFEST}" <<'PYEOF'
import glob, json, os, sys
res, manifest = sys.argv[1], sys.argv[2]
want = sorted(json.load(open(manifest))["labels"])
# sp per-language training writes langspec_soft_* (train.py:571 maps sp -> soft);
# both prefixes are accepted so a method change cannot silently pass this.
got = sorted(
    os.path.basename(p).split("_", 2)[2][: -len(".tokenizer.json")]
    for pat in ("langspec_sp_*.tokenizer.json", "langspec_soft_*.tokenizer.json")
    for p in glob.glob(os.path.join(res, "tokenizers", pat)))
if got != want:
    missing = sorted(set(want) - set(got))
    extra = sorted(set(got) - set(want))
    raise SystemExit(
        f"FATAL: {res}/tokenizers holds {len(got)} per-language rows, the "
        f"corpus manifest lists {len(want)} corpora. Missing: {missing[:10]}. "
        f"Unexpected: {extra[:10]}. Refusing to pack a container that does not "
        f"carry one row per corpus of this subset.")
print(f"OK  {len(got)} per-language rows, exactly the manifest's corpora")
PYEOF

echo "=== packing at $(date) ==="
cd "${REPO}/UNILID"
${PYTHON} convert.py "${RES}" -o "${OUT}"

echo "=== post-training checks at $(date) ==="
cd "${REPO}"
INSPECT="outputs/rerelease/cld3sub${SUBSET}_inspect.json"
${PYTHON} -u -m analysis.inspect_variant_models "${OUT}" -o "${INSPECT}"
# inspect_variant_models always exits 0 and only WRITES defect_present; without
# this the special-token defect would be caught in the report, after the
# evaluation. A defective container is not a corrected-generation model and must
# not reach the evaluation job at all.
${PYTHON} - "${INSPECT}" "${MANIFEST}" <<'PYEOF'
import json, sys
recs = json.load(open(sys.argv[1]))
if not isinstance(recs, list) or len(recs) != 1:
    raise SystemExit(f"FATAL: {sys.argv[1]} does not hold exactly one record")
r = recs[0]
want = len(json.load(open(sys.argv[2]))["labels"])
if r["defect_present"]:
    raise SystemExit(
        f"FATAL: {r['model']} carries the special-token defect (special mass "
        f"{r['special_mass_min']:.6f}-{r['special_mass_max']:.6f}). This is not "
        f"a corrected-generation model.")
if r["n_languages"] != want:
    raise SystemExit(
        f"FATAL: {r['model']} packed {r['n_languages']} rows, the manifest "
        f"lists {want} corpora")
print(f"OK  {r['n_languages']} rows, vocab {r['vocab_size']:,}, real-token mass "
      f"{r['real_mass_min']:.6f}-{r['real_mass_max']:.6f}, defect absent")
PYEOF
echo "=== Finished at $(date) ==="
