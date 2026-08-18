#!/bin/bash
#SBATCH --job-name=unilid-deepseek-fp64
#SBATCH --account=infra01
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=400G
#SBATCH --time=12:00:00
#SBATCH --output=/capstor/scratch/cscs/cmeister747/unilid_analysis/logs/deepseek_fp64_%j.out
#SBATCH --error=/capstor/scratch/cscs/cmeister747/unilid_analysis/logs/deepseek_fp64_%j.err
set -euo pipefail

# Retrain the DeepSeek3.2 tokenizer variant with the PATCHED (fp64) trainer.
#
# Why: the model in the co-author's Drive folder has a corrupted azj_Latn row.
# Retraining that language from its own corpus under the patched trainer and
# comparing against the stored row (analysis/gate_correction.py, 2026-08-18)
# gives a signed mean difference of +1.0002 nats at correlation 0.7057, against
# +6.7e-05 at correlation 1.0000 for zul_Latn, a language with the same
# N_L = 100,000 that acts as the control. Six size-spread languages all pass with
# correlation 1.00000000. Evidence:
# outputs/rerelease/gate_correction_deepseek.json.
#
# azj_Latn is the exact language EXPERIMENTAL_SETUP.md records as the fp64 EM
# bug's trigger (the 142,136-byte line, longest in all 1,940 corpora). Both LLM
# tokenizer variants were built on 2026-03-27, four months BEFORE that bug was
# fixed on 2026-07-27, while the base model and the Mistral-Nemo variant were
# retrained after it. So both variants are expected to carry it, and both do.
#
# Note the plateau diagnostic (analysis/variant_plateau_outliers.py) does NOT
# flag this row in the DeepSeek model: its plateau sits 1.7 sd below expectation,
# well inside the normal range, whereas the Qwen3 variant's equivalent row was
# driven to the hard training floor at 20.1 sd. The corruption here is milder in
# its effect on the plateau and severe in its effect on the row, which is why the
# retrain gate is the instrument that catches it and the plateau scan is not.
#
# This run fixes BOTH defects at once. The patched trainer removes the EM
# corruption, and UNILID 0.3.0's trainer no longer gives the special tokens the
# base tokenizer's score-0 entries, so the rows are born with their real tokens
# normalized to 1.0 rather than to 0.2. No separate correction pass is needed.
#
# Mirrors slurm_mistralnemo_train_fp64.sh, which itself mirrors the recorded
# Apertus fp64 retrains (jobs 2903767, 2903768). Differences:
#  - the base tokenizer is EXTRACTED FROM THE EXISTING CONTAINER rather than
#    re-converted from a HuggingFace tokenizer.json, so the vocabulary is
#    bit-identical to the model being replaced and the only things that change
#    are the trainer and the special-token handling. It is already written to
#    ${RES}/tokenizers/langspec_base_tokenizer.json, so --reuse-base loads it
#    directly and _convert_to_unigram_base never runs (train.py:455).
#  - --vocab-size 128819 passed explicitly, so --initial-vocab is unnecessary
#    (train.py:356). --fasttext is omitted; --corpus-dir with --reuse-corpus
#    satisfies the input validation on its own (train.py:349).
#
# CAVEAT to record with the result: the corpus is the shared Apertus draw, which
# is what every retrain in this project has used. For languages above the
# 100,000-line cap it is not necessarily the same sample the original Qwen3 run
# saw, so the retrained rows are not expected to reproduce the originals exactly
# even where the original was sound.
#
# The 131k-vocab Apertus retrain took 4h36m and the 200k took 7h28m; at 151,670
# expect roughly 5 to 6 hours. 12h is headroom, not a measurement.

PYTHON="/users/cmeister747/.pyenv/versions/3.11.5/bin/python3"
export PATH="/users/cmeister747/.local/bin:${PATH}"   # patched fp64 spm_train (fork commits d0208d9+c5921a2)
SCR="/capstor/scratch/cscs/cmeister747/unilid_analysis"
RES="${SCR}/results_deepseek_v32_fp64"
OUT="${SCR}/glotlid_deepseek_v32_fp64.unilid"

if [ ! -f "${RES}/tokenizers/langspec_base_tokenizer.json" ]; then
  echo "FATAL: base tokenizer missing at ${RES}/tokenizers/langspec_base_tokenizer.json" >&2
  echo "Extract it from the source container before submitting." >&2
  exit 1
fi

cd /users/cmeister747/unilid_analysis/UNILID
echo "=== DeepSeek3.2 fp64 retrain at $(date) on $(hostname) ==="
echo "commit: $(git -C /users/cmeister747/unilid_analysis rev-parse HEAD)"
echo "spm_train: $(command -v spm_train)"
spm_train --version 2>&1 | head -1 || true

${PYTHON} train.py \
  --vocab-size 128819 --byte-level --per-lang-counts-method sp \
  --max-base-samples-per-lang 10000 \
  --lang-batch-size 20 \
  --results-dir "${RES}" \
  --corpus-dir "${SCR}/results_apertus200k/corpus" \
  --reuse-corpus --skip-existing-langs --reuse-base

echo "=== packing at $(date) ==="
${PYTHON} convert.py "${RES}" -o "${OUT}"

echo "=== post-training gates at $(date) ==="
cd /users/cmeister747/unilid_analysis
# The defect signature: real-token mass must now be 1.0 per row, not 0.2.
${PYTHON} -u -m analysis.inspect_variant_models "${OUT}" \
  -o outputs/rerelease/deepseek_fp64_inspect.json
# The diagnostic that found the corruption, re-run as the gate. azj_Latn must no
# longer appear, and nothing new may appear that is not a shared minority-script
# coverage effect.
${PYTHON} -u -m analysis.variant_plateau_outliers "${OUT}" \
  -o outputs/rerelease/deepseek_fp64_plateau_outliers.json

echo "=== Finished at $(date) ==="
