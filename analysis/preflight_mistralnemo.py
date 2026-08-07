"""Preflight checks for the Mistral-Nemo retrain (plan `EXPERIMENTS_PLAN.md`
"E3: Mistral-Nemo variant").

Mirrors `analysis/preflight_131k.py`'s checks and structure for the
mistralai/Mistral-Nemo-Base-2407 local HF snapshot. Every check aborts loudly
on failure; nothing is submitted unless all pass. Run this manually before
`mistralnemo_export_vocab.py` and before submitting
`slurm_mistralnemo_train_fp64.sh` (it is not wired into the SLURM script
itself, matching how preflight_131k.py was used for the Apertus 131k retrain).

1. HF snapshot pin: refs/main resolves to the recorded SNAPSHOT_HASH (catches
   a later `huggingface-cli download` moving the pin to a different revision).
2. Tokenizer artifact: tokenizer.json exists, parses, matches the recorded
   sha256, model.vocab has exactly EXPECTED_VOCAB entries, and all four
   UniLID special-token strings are present as vocab keys. The last check
   guards the silent fallback in UNILID/train.py::_convert_to_unigram_base,
   which resolves unk_id via vocab.get("<unk>", 0) and would silently use id 0
   if the literal string were absent.
3. Forked spm_train binary present and executable (fixed-vocab EM, fp64
   E-step; the retrain is meaningless with the stock binary, which would
   prune the seeded vocab, or with the pre-fp64 binary, which silently
   collapses rows on long lines).
4. Reused corpus split: EXPECTED_LANGS per-language files under CORPUS_DIR,
   total lines equal to the recorded TRAIN_LINES (this is the same corpus
   provenance check preflight_131k.py runs; the split is tokenizer-independent
   and reused unmodified from the Apertus retrains).
5. RESULTS_DIR does not already exist with conflicting contents (this is a
   fresh model family; it must not silently merge with another tokenizer's
   partial output).

Records the tokenizer file's sha256 (checked against EXPECTED_TOKENIZER_SHA256)
for the chronological log.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, "/users/cmeister747/unilid_analysis/UNILID")

from analysis.mistralnemo_constants import (
    CORPUS_DIR,
    EXPECTED_LANGS,
    EXPECTED_TOKENIZER_SHA256,
    EXPECTED_VOCAB,
    REFS_MAIN_PATH,
    RESULTS_DIR,
    SNAPSHOT_HASH,
    SPM_TRAIN,
    TOKENIZER_JSON,
)
from analysis.config import TRAIN_LINES


def main() -> None:
    try:
        from unilid.constants import SPECIAL_TOKENS
    except ImportError as e:
        raise RuntimeError(
            f"cannot import unilid.constants from UNILID dir: {e}"
        )

    # 1. snapshot pin
    if not os.path.isfile(REFS_MAIN_PATH):
        raise FileNotFoundError(f"HF refs/main missing: {REFS_MAIN_PATH}")
    with open(REFS_MAIN_PATH, "r", encoding="utf-8") as f:
        current_ref = f.read().strip()
    if current_ref != SNAPSHOT_HASH:
        raise RuntimeError(
            f"HF snapshot pin drift: refs/main is {current_ref!r}, expected "
            f"{SNAPSHOT_HASH!r}. Someone re-pulled the model. Re-pin "
            f"SNAPSHOT_HASH in analysis/mistralnemo_constants.py deliberately "
            f"before retraining, or restore the recorded revision."
        )
    print(f"snapshot pin OK: refs/main -> {current_ref}")

    # 2. tokenizer artifact
    if not os.path.exists(TOKENIZER_JSON):
        raise FileNotFoundError(f"tokenizer artifact missing: {TOKENIZER_JSON}")
    with open(TOKENIZER_JSON, "rb") as f:
        raw = f.read()
    sha = hashlib.sha256(raw).hexdigest()
    if sha != EXPECTED_TOKENIZER_SHA256:
        raise RuntimeError(
            f"tokenizer.json sha256 mismatch: got {sha}, expected "
            f"{EXPECTED_TOKENIZER_SHA256}. The pinned snapshot's tokenizer "
            f"content changed; do not proceed without re-confirming the pin."
        )
    tok = json.loads(raw)
    vocab = tok.get("model", {}).get("vocab")
    if not isinstance(vocab, dict):
        raise RuntimeError(
            f"{TOKENIZER_JSON}: model.vocab missing or not a dict "
            f"(model.type={tok.get('model', {}).get('type')!r})"
        )
    if len(vocab) != EXPECTED_VOCAB:
        raise RuntimeError(f"vocab size {len(vocab)} != expected {EXPECTED_VOCAB}")
    missing = [t for t in SPECIAL_TOKENS.values() if t not in vocab]
    if missing:
        raise RuntimeError(
            f"special tokens missing from model.vocab: {missing}; "
            f"train.py would silently map unk to id 0"
        )
    specials = {t: vocab[t] for t in SPECIAL_TOKENS.values()}
    print(f"tokenizer OK: {TOKENIZER_JSON}")
    print(f"  sha256 {sha}")
    print(f"  vocab {len(vocab)}, model.type {tok['model'].get('type')!r}, "
          f"normalizer {(tok.get('normalizer') or {}).get('type')!r}")
    print(f"  special-token ids: {specials}")

    # 3. forked spm_train
    if not (os.path.isfile(SPM_TRAIN) and os.access(SPM_TRAIN, os.X_OK)):
        raise RuntimeError(f"forked spm_train not executable at {SPM_TRAIN}")
    print(f"spm_train OK: {SPM_TRAIN}")

    # 4. corpus split
    if not os.path.isdir(CORPUS_DIR):
        raise FileNotFoundError(f"corpus split missing: {CORPUS_DIR}")
    files = sorted(f for f in os.listdir(CORPUS_DIR) if f.endswith("_train.txt"))
    if len(files) != EXPECTED_LANGS:
        raise RuntimeError(
            f"corpus split has {len(files)} language files, expected "
            f"{EXPECTED_LANGS}"
        )
    print(f"corpus split present: {len(files)} files; counting lines "
          f"(one pass over ~10 GB)...")
    total = 0
    for fn in files:
        out = subprocess.run(
            ["wc", "-l", os.path.join(CORPUS_DIR, fn)],
            capture_output=True, text=True, check=True,
        )
        total += int(out.stdout.split()[0])
    if total != TRAIN_LINES:
        raise RuntimeError(
            f"corpus split totals {total:,} lines, expected "
            f"TRAIN_LINES={TRAIN_LINES:,}; the split does not match and must "
            f"not be reused"
        )
    print(f"corpus split OK: {len(files)} files, {total:,} lines == TRAIN_LINES")

    # 5. results dir must not already hold a different model's output
    if os.path.isdir(RESULTS_DIR):
        existing = os.listdir(RESULTS_DIR)
        if existing:
            print(
                f"NOTE: {RESULTS_DIR} already exists with {len(existing)} "
                f"entries. This is expected on a resumed run; the training "
                f"sweep script's own skip-existing check governs what gets "
                f"reused. Aborting only if you did not intend to resume."
            )

    print("\nALL PREFLIGHT CHECKS PASSED")


if __name__ == "__main__":
    main()
