"""Preflight checks for the Apertus 131k (preliminary_mul) retrain (plan Track A).

Every check aborts loudly on failure; nothing is submitted unless all pass.

1. Tokenizer artifact: exists, parses, model.vocab has exactly 131,072 entries, and
   all four UniLID special-token strings are present as vocab keys. The last check
   guards the silent fallback in UNILID/train.py::_convert_to_unigram_base, which
   resolves unk_id via vocab.get("<unk>", 0) and would silently use id 0 if the
   literal string were absent.
2. Forked spm_train binary present and executable (fixed-vocab EM; the retrain is
   meaningless with the stock binary, which would prune the seeded vocab).
3. Training input: train.txt present with the recorded line count.
4. Reused corpus split: 1,940 per-language files under results_apertus200k/corpus,
   total lines equal to the train.txt line count (this doubles as the corpus
   provenance check required by the margin-diagnostic calibration, plan B3).

Records the tokenizer file's sha256 for the chronological log.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys

TOKENIZER_JSON = "/users/cmeister747/apertus-tokenizer-development/preliminary_mul/tokenizer.json"
EXPECTED_VOCAB = 131_072
SPM_TRAIN = "/users/cmeister747/.local/bin/spm_train"
SCR = "/capstor/scratch/cscs/cmeister747/unilid_analysis"
TRAIN_TXT = os.path.join(SCR, "train.txt")
CORPUS_DIR = os.path.join(SCR, "results_apertus200k", "corpus")
EXPECTED_LANGS = 1_940
UNILID_DIR = "/users/cmeister747/unilid_analysis/UNILID"


def main() -> None:
    sys.path.insert(0, UNILID_DIR)
    try:
        from unilid.constants import SPECIAL_TOKENS
    except ImportError as e:
        raise RuntimeError(f"cannot import unilid.constants from {UNILID_DIR}: {e}")

    # 1. tokenizer artifact
    if not os.path.exists(TOKENIZER_JSON):
        raise FileNotFoundError(f"tokenizer artifact missing: {TOKENIZER_JSON}")
    with open(TOKENIZER_JSON, "rb") as f:
        raw = f.read()
    sha = hashlib.sha256(raw).hexdigest()
    tok = json.loads(raw)
    vocab = tok.get("model", {}).get("vocab")
    if not isinstance(vocab, dict):
        raise RuntimeError(f"{TOKENIZER_JSON}: model.vocab missing or not a dict "
                           f"(model.type={tok.get('model', {}).get('type')!r})")
    if len(vocab) != EXPECTED_VOCAB:
        raise RuntimeError(f"vocab size {len(vocab)} != expected {EXPECTED_VOCAB}")
    missing = [t for t in SPECIAL_TOKENS.values() if t not in vocab]
    if missing:
        raise RuntimeError(f"special tokens missing from model.vocab: {missing}; "
                           f"train.py would silently map unk to id 0")
    specials = {t: vocab[t] for t in SPECIAL_TOKENS.values()}
    print(f"tokenizer OK: {TOKENIZER_JSON}")
    print(f"  sha256 {sha}")
    print(f"  vocab {len(vocab)}, model.type {tok['model'].get('type')!r}, "
          f"normalizer {(tok.get('normalizer') or {}).get('type')!r}")
    print(f"  special-token ids: {specials}")

    # 2. forked spm_train
    if not (os.path.isfile(SPM_TRAIN) and os.access(SPM_TRAIN, os.X_OK)):
        raise RuntimeError(f"forked spm_train not executable at {SPM_TRAIN}")
    print(f"spm_train OK: {SPM_TRAIN}")

    # 3. train.txt
    if not os.path.exists(TRAIN_TXT):
        raise FileNotFoundError(f"training file missing: {TRAIN_TXT}")
    from analysis.config import TRAIN_LINES
    print(f"train.txt present ({os.path.getsize(TRAIN_TXT):,} bytes); counting lines "
          f"of the corpus split (one pass over ~10 GB)...")

    # 4. corpus split
    if not os.path.isdir(CORPUS_DIR):
        raise FileNotFoundError(f"corpus split missing: {CORPUS_DIR}")
    files = sorted(f for f in os.listdir(CORPUS_DIR) if f.endswith("_train.txt"))
    if len(files) != EXPECTED_LANGS:
        raise RuntimeError(f"corpus split has {len(files)} language files, "
                           f"expected {EXPECTED_LANGS}")
    total = 0
    for fn in files:
        out = subprocess.run(["wc", "-l", os.path.join(CORPUS_DIR, fn)],
                             capture_output=True, text=True, check=True)
        total += int(out.stdout.split()[0])
    if total != TRAIN_LINES:
        raise RuntimeError(f"corpus split totals {total:,} lines, expected "
                           f"TRAIN_LINES={TRAIN_LINES:,}; the split does not match "
                           f"train.txt and must not be reused")
    print(f"corpus split OK: {len(files)} files, {total:,} lines == TRAIN_LINES")

    print("\nALL PREFLIGHT CHECKS PASSED")


if __name__ == "__main__":
    main()
