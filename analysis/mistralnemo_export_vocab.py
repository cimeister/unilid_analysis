"""Vocab export step for the Mistral-Nemo retrain (task 3, "vocab export").

The fixed-vocabulary EM trainer (UNILID/train.py plus the forked spm_train
binary) does not train a base tokenizer from scratch for this family: it takes
an existing tokenizer's token inventory as a fixed vocabulary and later
estimates per-language probabilities over it. That conversion step --
building a Unigram base tokenizer with uniform log-probabilities over the
source vocabulary, preserving the source's pretokenizer/normalizer/decoder --
lives in UNILID/train.py::_convert_to_unigram_base and is normally triggered
implicitly, inline, the first time train.py is invoked with
--initial-vocab pointing at an HF tokenizer.json (see
EXPERIMENTAL_SETUP.md "Per-language training pipeline and the trainer fix").

For the Apertus 131k/200k retrains this happened inside the same `train.py`
process as the first batch of per-language training (slurm_apertus_train_
{131k,200k}_fp64.sh call train.py once, with no separate export step). This
script makes that step explicit and standalone instead, so it can be run,
inspected, and logged once before the per-language sweep starts, and so the
sweep driver (mistralnemo_train_sweep.py) can assume the base tokenizer
already exists and abort loudly if it does not, rather than silently building
it on whichever language happens to run first.

Idempotent: if BASE_TOKENIZER_PATH already exists, this step is skipped (logged,
not re-run) so re-invoking it during a resumed sweep is safe.

Writes:
  - BASE_TOKENIZER_PATH (the Unigram base tokenizer used by every per-language run)
  - PROVENANCE_PATH (JSON: pinned snapshot hash + tokenizer sha256 + output
    path + git commit + timestamp; every later meta file in this family
    copies these same three fields forward per the task's requirement that
    the tokenizer snapshot be recorded in every output meta file)
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, "/users/cmeister747/unilid_analysis/UNILID")

from analysis import preflight_mistralnemo
from analysis.mistralnemo_constants import (
    BASE_TOKENIZER_PATH,
    EXPECTED_TOKENIZER_SHA256,
    EXPECTED_VOCAB,
    PROVENANCE_PATH,
    RESULTS_DIR,
    SNAPSHOT_HASH,
    TOKENIZER_JSON,
    ensure_results_dirs,
)


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        ).stdout.strip()
    except (subprocess.CalledProcessError, OSError) as e:
        raise RuntimeError(f"git rev-parse HEAD failed: {e}") from e


def main() -> None:
    print("=== running preflight checks before vocab export ===")
    preflight_mistralnemo.main()

    ensure_results_dirs()
    if RESULTS_DIR.startswith("/capstor/store/"):
        raise RuntimeError(
            f"RESULTS_DIR resolves under /capstor/store/, refusing to write: "
            f"{RESULTS_DIR}"
        )

    if os.path.exists(BASE_TOKENIZER_PATH):
        print(f"base tokenizer already exported, skipping: {BASE_TOKENIZER_PATH}")
    else:
        # train._convert_to_unigram_base is the exact function the Apertus
        # retrains ran implicitly inside train.py's first invocation.
        import train as unilid_train_script  # UNILID/train.py

        print(f"exporting vocab: {TOKENIZER_JSON} -> {BASE_TOKENIZER_PATH}")
        unilid_train_script._convert_to_unigram_base(TOKENIZER_JSON, BASE_TOKENIZER_PATH)
        if not os.path.exists(BASE_TOKENIZER_PATH):
            raise RuntimeError(
                f"_convert_to_unigram_base returned without writing "
                f"{BASE_TOKENIZER_PATH}"
            )
        print(f"wrote base tokenizer: {BASE_TOKENIZER_PATH}")

    # Verify the exported base tokenizer's vocab size before recording success.
    from tokenizers import Tokenizer

    base_tok = Tokenizer.from_file(BASE_TOKENIZER_PATH)
    base_vocab_size = len(base_tok.get_vocab())
    if base_vocab_size != EXPECTED_VOCAB:
        raise RuntimeError(
            f"exported base tokenizer vocab size {base_vocab_size} != expected "
            f"{EXPECTED_VOCAB}: {BASE_TOKENIZER_PATH}"
        )

    provenance = {
        "step": "mistralnemo_export_vocab",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "source_tokenizer": {
            "hf_snapshot_hash": SNAPSHOT_HASH,
            "tokenizer_json_path": TOKENIZER_JSON,
            "tokenizer_json_sha256": EXPECTED_TOKENIZER_SHA256,
            "vocab_size": EXPECTED_VOCAB,
        },
        "base_tokenizer_path": BASE_TOKENIZER_PATH,
        "base_tokenizer_vocab_size": base_vocab_size,
    }
    with open(PROVENANCE_PATH, "w", encoding="utf-8") as f:
        json.dump(provenance, f, indent=2)
    print(f"wrote provenance: {PROVENANCE_PATH}")
    print("\nVOCAB EXPORT COMPLETE")


if __name__ == "__main__":
    main()
