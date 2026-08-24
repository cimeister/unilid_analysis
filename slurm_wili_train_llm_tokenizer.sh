#!/bin/bash
#SBATCH --job-name=w-llmtok
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
# PHASE 2b of the WiLI regeneration plan: the \unilid-Mistral, \unilid-LLaMA3.2
# and \unilid-LLaMA2 rows, none of which has a container anywhere.
#
# Takes MODEL_NAME from the environment (--export). One of:
#   llama32_1b_wili       RUNNABLE
#   mistral7b_v02_wili    RUNNABLE since the 2026-08-23 dropped-entry decision
#   llama2_7b_wili        RUNNABLE since the 2026-08-23 dropped-entry decision
# Everything else about each model is in the case table below, so an unknown
# MODEL_NAME aborts rather than falling back to a default.
#
# THE DROPPED-ENTRY DECISION, AUTHOR, 2026-08-23. Measured 2026-08-23 by running
# this job's exact train.py argv on a two-language mini corpus: the Mistral and
# LLaMA-2 vocabularies are SentencePiece-style, not byte-level, and the
# per-language SentencePiece path aborted at UNILID/unilid/vocab_io.py:119-120 on
# vocabulary entries containing a raw carriage return (51 of 32,001 for Mistral,
# 24 of 32,001 for LLaMA-2). The author's decision is that those entries are
# DROPPED WHOLE from the converted base vocabulary: no character stripping, no
# byte-level re-encoding, no change of --per-lang-counts-method. It is applied by
# `analysis/convert_llm_tokenizer_base.py --drop-refused-entries`, which is OFF
# by default and inert on a vocabulary with no such entries (verified: with the
# flag on, the LLaMA-3.2 base comes out byte-identical, sha256 52773b24...).
# This is why the two VOCAB_SIZE values below are BELOW their source counts,
# where every other model in this repo is at or above.
#   Records: outputs/rerelease/wili_mistral7b_v02_base_convert.json and
#            outputs/rerelease/wili_llama2_7b_base_convert.json
#            (each lists every dropped token by name; the pre-drop conversions
#             are kept alongside as *_base_convert_predrop.json, and the pre-drop
#             base files under ${SCR}/superseded_bases_20260823/).
# LLaMA-3.2 and Mistral-Nemo are byte-level BPE with zero such entries; nothing
# was dropped from either and neither base changed.
#
# Submit with --job-name=w-${MODEL_NAME} so the log files are distinguishable.
#
# WHERE THE BASE VOCABULARY COMES FROM. These three rows have no container, so
# the base tokenizer is built from the author-designated HuggingFace tokenizer by
# UNILID's OWN converter, train.py:52 `_convert_to_unigram_base`, the same
# function train.py:458-462 runs when --initial-vocab names an HF tokenizer and
# no base tokenizer exists yet. It was run ahead of time by
# `python -m analysis.convert_llm_tokenizer_base`, which imports that function
# rather than reimplementing it, so the base vocabulary is an artifact that can be
# preflight-checked before hours of per-language training. This is the same
# mechanism that built the GlotLID-C Mistral-Nemo model
# (slurm_mistralnemo_train_fp64.sh:42 passes --initial-vocab and lets train.py
# convert in-job); only the timing differs.
#
# The converter takes the source vocabulary in id order, gives every non-special
# token one uniform log probability, appends any of UNILID's four special tokens
# the source lacks, and keeps the source's normalizer, pretokenizer and decoder.
# The per-language step then replaces those scores, so the conversion contributes
# the token inventory and nothing else. This is why the base entry count is one
# or four MORE than the source model.vocab count in the table below, except for
# the two rows where the 2026-08-23 dropped-entry decision applies and the count
# ends up BELOW it.
#
# UNCONFIRMED PROVENANCE, to be stated in the results entry: these three
# tokenizers are the AUTHOR'S DESIGNATION of what the published rows used, not a
# confirmation. The HuggingFace cache held dangling symlinks with no blobs, so
# the cache is not evidence either. Each was downloaded fresh on 2026-08-23 at
# the pinned revision recorded below, into ${SCR}/hf_tokenizers/<name>/.
# Manifest: outputs/rerelease/wili_phase2b_tokenizer_downloads.json
#
# --results-dir and --base-tokenizer-path are BOTH mandatory. Without them
# train.py:450-452 defaults the base path to results_<vocab//1000>k/tokenizers/,
# the reuse test at train.py:455 fails, and train.py:465-492 silently trains a
# FRESH vocabulary and reports success.
#
# Phase 1's three retrains took 19 to 25 minutes each at vocabularies of 100,000
# to 151,670. 4h is the template's headroom, not a measurement.
PYTHON="/users/cmeister747/.pyenv/versions/3.11.5/bin/python3"
export PATH="/users/cmeister747/.local/bin:${PATH}"   # patched fp64 spm_train
REPO="/users/cmeister747/unilid_analysis"
SCR="/capstor/scratch/cscs/cmeister747/unilid_analysis"

# sha256 of the patched fp64 spm_train installed 2026-07-27 (fork commits
# d0208d9 + c5921a2), recorded in slurm_wili_train_fp32null.sh. `spm_train
# --version` prints "sentencepiece 0.2.2" for both builds, so the hash is the
# only thing that discriminates them.
readonly PATCHED_SPM_SHA256="ff5744ae620074154e558dcf1d48d5f3e32934047b40e9f48f9333f0d7fc94d2"

if [ -z "${MODEL_NAME:-}" ]; then
  echo "FATAL: MODEL_NAME unset. Submit with --export=ALL,MODEL_NAME=<one of" >&2
  echo "       mistral7b_v02_wili llama32_1b_wili llama2_7b_wili>" >&2
  exit 1
fi

# ── per-model table: nothing here is inferred ───────────────────────────────
# VOCAB_SIZE is the CONVERTED base entry count and must be what train.py gets as
# --vocab-size; SOURCE_VOCAB is the source tokenizer's model.vocab count, kept
# here so the two are never confused.
case "${MODEL_NAME}" in
  mistral7b_v02_wili)
    HF_REPO="mistral-community/Mistral-7B-v0.2"
    HF_REVISION="80cf49b18de6354fd8a6d1e81c177dd830e05eea"
    SOURCE_VOCAB=32000
    # 32,000 source + <pad> (absent from the source) = 32,001 converted, MINUS
    # the 51 entries containing a raw carriage return, dropped under the
    # 2026-08-23 decision described in the header. The four UNILID specials are
    # never candidates for the drop (vocab_io.py:117-118 skips specials before
    # the refusal check) and the converter verified none of them was flagged.
    VOCAB_SIZE=31950
    BASE_SHA256="25030a1fa3ac71c9cd865618c1200b3ab52748ee519e1ae6d524fb92b84d73e9"
    # Pre-drop base was 32,001 entries, sha256 1e59ff82...; kept at
    # ${SCR}/superseded_bases_20260823/mistral7b_v02_langspec_base_tokenizer_predrop.json
    #
    # STILL OPEN, and INDEPENDENT of the dropped entries: the plan's Phase 2
    # table expects this tokenizer to have 32,768 entries and uses that number as
    # the discriminator against Mistral-Nemo's 131,072. It has 32,000. The
    # Mistral release with 32,768 is v0.3, a different repository. The plan's
    # CONCLUSION survives (32,000 is not 131,072, so this is not Mistral-Nemo),
    # but which repository the published \unilid-Mistral row used is unresolved.
    UNRESOLVED_NOTE="source vocabulary is ${SOURCE_VOCAB}, not the 32,768 the plan's Phase 2 table expects; the repository the published row used is unconfirmed"
    ;;
  llama32_1b_wili)
    HF_REPO="meta-llama/Llama-3.2-1B"
    HF_REVISION="4e20de362430cd3b72f300e6b0f18e50e7166e08"
    SOURCE_VOCAB=128000       # model.vocab; get_vocab() adds 256 added_tokens
    VOCAB_SIZE=128260         # 128,256 + all four UNILID specials, none present
    BASE_SHA256="52773b2470a130d50c4b7556d626005b531af47ed88a0578ad989485c5744703"
    ;;
  llama2_7b_wili)
    HF_REPO="meta-llama/Llama-2-7b-hf"
    HF_REVISION="01c7f73d771dfac7d292323805ebc428287df4f9"
    SOURCE_VOCAB=32000
    # 32,000 source + <pad> = 32,001 converted, MINUS the 24 entries containing a
    # raw carriage return, dropped under the 2026-08-23 decision in the header.
    # Same mechanism as mistral7b_v02_wili, a different count of entries.
    VOCAB_SIZE=31977
    BASE_SHA256="1cbf107fa57108ba9b74d69d6cea29f14adc5f20a7bde4b82bec571ac9cc4c75"
    # Pre-drop base was 32,001 entries, sha256 1854d6e5...; kept at
    # ${SCR}/superseded_bases_20260823/llama2_7b_langspec_base_tokenizer_predrop.json
    ;;
  *)
    echo "FATAL: unknown MODEL_NAME '${MODEL_NAME}'. Known: mistral7b_v02_wili" >&2
    echo "       llama32_1b_wili llama2_7b_wili" >&2
    exit 1
    ;;
esac

RES="${SCR}/results_${MODEL_NAME}_fp64"
BASE="${RES}/tokenizers/langspec_base_tokenizer.json"
CORPUS="${SCR}/wili_corpus_shared"
OUT="${SCR}/${MODEL_NAME}_fp64.unilid"

echo "=== WiLI ${MODEL_NAME} retrain at $(date) on $(hostname) ==="
echo "commit:       $(git -C ${REPO} rev-parse HEAD)"
echo "tokenizer:    ${HF_REPO} @ ${HF_REVISION} (UNCONFIRMED against the original)"
echo "source vocab: ${SOURCE_VOCAB}  ->  base vocab: ${VOCAB_SIZE}"
if [ "${VOCAB_SIZE}" -lt "${SOURCE_VOCAB}" ]; then
  echo "base vocab is BELOW the source count: entries refused by vocab_io.py:119"
  echo "were dropped under the author decision of 2026-08-23. Dropped tokens are"
  echo "listed by name in outputs/rerelease/wili_${MODEL_NAME%_wili}_base_convert.json"
fi
if [ -n "${UNRESOLVED_NOTE:-}" ]; then
  echo "UNRESOLVED (stated, not resolved, by this job): ${UNRESOLVED_NOTE}"
fi

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
