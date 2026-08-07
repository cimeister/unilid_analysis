"""Shared constants for the E3 Mistral-Nemo retrain (Camera-ready evaluation
program, `EXPERIMENTS_PLAN.md` "E3: Mistral-Nemo variant").

Single source of truth for the scripts in this family so the snapshot hash,
vocab size, and directory layout cannot drift between them:
  preflight_mistralnemo.py    -- validates the pinned tokenizer + corpus + trainer
  mistralnemo_export_vocab.py -- converts the tokenizer into the fixed-vocabulary
                                  EM trainer's Unigram base format
  mistralnemo_train_sweep.py  -- resumable per-language training driver
  mistralnemo_pack.py         -- packs the per-language tokenizers into .unilid
  degeneracy_scan_mistralnemo.py -- post-pack degeneracy gate

Every new constant here is new as of the E3 preparation work (2026-08-07); none
of it existed in the repo before this file. Flagged explicitly per the project's
magic-number rule.
"""
from __future__ import annotations

import os

# ---------------------------------------------------------------------------
# Pinned HuggingFace tokenizer snapshot (task 1)
# ---------------------------------------------------------------------------
# The commit hash of the local HF snapshot, i.e. the name of the
# snapshots/<hash>/ directory. Read from refs/main on 2026-08-07 and verified
# against the snapshot directory listing; preflight re-checks this on every
# run so a later `huggingface-cli download` that moves refs/main is caught
# instead of silently retraining against a different tokenizer revision.
HF_HUB_DIR = (
    "/capstor/store/cscs/swissai/a0229/cmeister/huggingface/hub/"
    "models--mistralai--Mistral-Nemo-Base-2407"
)
SNAPSHOT_HASH = "a4477a2f977929a969745b69bbd62e03043551a5"
SNAPSHOT_DIR = os.path.join(HF_HUB_DIR, "snapshots", SNAPSHOT_HASH)
REFS_MAIN_PATH = os.path.join(HF_HUB_DIR, "refs", "main")
TOKENIZER_JSON = os.path.join(SNAPSHOT_DIR, "tokenizer.json")

# sha256 of tokenizer.json's resolved blob content, recorded 2026-08-07.
# Read-only inspection at pin time; see EXPECTED_VOCAB check for the same
# unk-id-0 silent-fallback guard EXPERIMENTAL_SETUP.md describes for the
# 131k Apertus retrain (train.py::_convert_to_unigram_base resolves
# vocab.get("<unk>", 0), which would silently pick token id 0 if the
# literal "<unk>" string were absent).
EXPECTED_TOKENIZER_SHA256 = (
    "e11c71726323d33da7b8d6f6f269f1988931c0a52b7122bcdd8c05042974e0db"
)
EXPECTED_VOCAB = 131_072  # mistralai/Mistral-Nemo-Base-2407 tokenizer.json
                          # model.vocab length, verified 2026-08-07. Coincides
                          # numerically with the unrelated Apertus 131k
                          # tokenizer's vocab size -- different tokenizer,
                          # same round vocab-size choice.

# ---------------------------------------------------------------------------
# Trainer and corpus (task 2)
# ---------------------------------------------------------------------------
SPM_TRAIN = "/users/cmeister747/.local/bin/spm_train"  # forked fixed-vocab-em
                                                        # binary, fp64 E-step
                                                        # (commits d0208d9,
                                                        # c5921a2)
UNILID_DIR = "/users/cmeister747/unilid_analysis/UNILID"
PYTHON = "/users/cmeister747/.pyenv/versions/3.11.5/bin/python3"

SCR = "/capstor/scratch/cscs/cmeister747/unilid_analysis"
# The 1,940-file per-language corpus split is tokenizer-independent and is
# reused unmodified across retrains (EXPERIMENTAL_SETUP.md "Per-language
# training pipeline"). Read-only for this project: this path resolves through
# the results_apertus200k scratch symlink into
# /capstor/store/cscs/swissai/a0229/cmeister/unilid_analysis/results_apertus200k,
# and nothing in this file family writes into that store path.
CORPUS_DIR = os.path.join(SCR, "results_apertus200k", "corpus")
EXPECTED_LANGS = 1_940

# On-disk filename tag for per-language tokenizer files. train.py sets this to
# "soft" whenever --per-lang-counts-method is "sp" (see
# UNILID/train.py: `lang_em_mode = args.per_lang_counts_method if
# args.per_lang_counts_method != "sp" else "soft"`), even though the actual
# per-language estimation runs through SentencePiece
# (train_with_sentencepiece_direct), not the custom soft-EM path. Verified
# against real files on disk under
# results_apertus200k/tokenizers/langspec_soft_*.tokenizer.json and against
# UNILID/unilid/model_io.py::save_unilid, which globs langspec_soft_*.json
# first. Do not change this to "sp" -- that would silently write files the
# packer will not find.
PER_LANG_TOKENIZER_TAG = "soft"

# ---------------------------------------------------------------------------
# E3 output locations -- all new, all on scratch, none under
# /capstor/store/... (constraint: nothing in this family writes into the
# store directly).
# ---------------------------------------------------------------------------
RESULTS_DIR = os.path.join(SCR, "results_mistralnemo")
TOKENIZERS_DIR = os.path.join(RESULTS_DIR, "tokenizers")
LOG_DIR = os.path.join(RESULTS_DIR, "logs")
BASE_TOKENIZER_PATH = os.path.join(TOKENIZERS_DIR, "langspec_base_tokenizer.json")
PROVENANCE_PATH = os.path.join(RESULTS_DIR, "mistralnemo_tokenizer_provenance.json")
SWEEP_SUMMARY_PATH = os.path.join(RESULTS_DIR, "training_sweep_summary.json")
FAILED_LANGS_PATH = os.path.join(RESULTS_DIR, "failed_languages.txt")
# Flat under scratch, matching glotlid_apertus{131k,200k}_fp64.unilid (decision
# 2026-08-07: keep the recorded Apertus placement convention).
PACKED_MODEL_PATH = os.path.join(SCR, "glotlid_mistralnemo_fp64.unilid")
PACKED_META_PATH = os.path.join(SCR, "glotlid_mistralnemo_fp64.meta.json")
DEGENERACY_OUT_MD = "outputs/tables/degenerate_rows_mistralnemo.md"


def ensure_results_dirs() -> None:
    """Create RESULTS_DIR/TOKENIZERS_DIR/LOG_DIR if absent. Never touches
    /capstor/store/... ; every path created here is under RESULTS_DIR."""
    os.makedirs(TOKENIZERS_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
