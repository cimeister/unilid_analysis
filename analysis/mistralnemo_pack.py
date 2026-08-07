"""Packing step for the Mistral-Nemo retrain (task 3, "packing step").

Reuses UNILID/unilid/model_io.py::save_unilid unmodified -- the same function
UNILID/convert.py calls, and the same one the Apertus 131k/200k fp64 retrains
used (`python convert.py results_apertus{131k,200k}_fp64 -o
glotlid_apertus{131k,200k}_fp64.unilid`). No new packing logic is introduced;
this script only adds the pre-pack completeness gate and the post-pack sidecar
metadata file convert.py does not write on its own.

Pre-pack gate: refuses to pack unless all EXPECTED_LANGS per-language
tokenizer files are present and FAILED_LANGS_PATH is absent or empty. Packing
a model missing languages would silently produce a evaluable-looking .unilid
file that is missing rows; that must be a deliberate, visible decision, not a
side effect of running this script early.

Post-pack gate: reloads the packed weights via the same reader
analysis/transfer_sweep.py::_load_model_data uses (the one degeneracy_scan.py
and fp64_retrain_check.py already rely on) and checks the language count and
vocabulary width match what was expected before declaring success.

Writes:
  - PACKED_MODEL_PATH (glotlid_mistralnemo_fp64.unilid)
  - PACKED_META_PATH (JSON sidecar: source tokenizer snapshot hash + sha256,
    language/vocab counts, packed file sha256, git commit, timestamp)
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, "/users/cmeister747/unilid_analysis/UNILID")

from analysis.mistralnemo_constants import (
    EXPECTED_LANGS,
    EXPECTED_TOKENIZER_SHA256,
    EXPECTED_VOCAB,
    FAILED_LANGS_PATH,
    PACKED_META_PATH,
    PACKED_MODEL_PATH,
    PER_LANG_TOKENIZER_TAG,
    RESULTS_DIR,
    SNAPSHOT_HASH,
    TOKENIZER_JSON,
    TOKENIZERS_DIR,
)
from unilid.model_io import save_unilid


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        ).stdout.strip()
    except (subprocess.CalledProcessError, OSError) as e:
        raise RuntimeError(f"git rev-parse HEAD failed: {e}") from e


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    if os.path.isfile(FAILED_LANGS_PATH) and os.path.getsize(FAILED_LANGS_PATH) > 0:
        raise RuntimeError(
            f"refusing to pack: {FAILED_LANGS_PATH} is non-empty. Resolve the "
            f"failed languages (retrain, or a deliberate documented exclusion) "
            f"before packing an incomplete model."
        )

    lang_files = sorted(Path(TOKENIZERS_DIR).glob(f"langspec_{PER_LANG_TOKENIZER_TAG}_*.tokenizer.json"))
    lang_files = [p for p in lang_files if not p.name.endswith(".metadata.json")]
    if len(lang_files) != EXPECTED_LANGS:
        raise RuntimeError(
            f"found {len(lang_files)} per-language tokenizer files under "
            f"{TOKENIZERS_DIR}, expected {EXPECTED_LANGS}. Run "
            f"mistralnemo_train_sweep.py to completion first."
        )

    if os.path.exists(PACKED_MODEL_PATH):
        raise RuntimeError(
            f"refusing to overwrite existing packed model: {PACKED_MODEL_PATH}. "
            f"Move or delete it explicitly before re-packing."
        )

    print(f"packing {len(lang_files)} per-language tokenizers from "
          f"{RESULTS_DIR} -> {PACKED_MODEL_PATH}")
    save_unilid(Path(RESULTS_DIR), Path(PACKED_MODEL_PATH))

    if not os.path.exists(PACKED_MODEL_PATH):
        raise RuntimeError(f"save_unilid returned without writing {PACKED_MODEL_PATH}")

    # Post-pack shape gate, using the same reader degeneracy_scan.py and
    # fp64_retrain_check.py already rely on.
    from analysis.transfer_sweep import _load_model_data

    W, langs, _ = _load_model_data(str(PACKED_MODEL_PATH))
    if len(langs) != EXPECTED_LANGS:
        raise RuntimeError(
            f"packed model has {len(langs)} languages, expected {EXPECTED_LANGS}"
        )
    if W.shape[1] != EXPECTED_VOCAB:
        raise RuntimeError(
            f"packed model vocab width {W.shape[1]} != expected {EXPECTED_VOCAB}"
        )
    print(f"post-pack shape OK: {len(langs)} languages x {W.shape[1]} vocab")

    packed_sha256 = _sha256_file(PACKED_MODEL_PATH)
    packed_size = os.path.getsize(PACKED_MODEL_PATH)

    meta = {
        "step": "mistralnemo_pack",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "source_tokenizer": {
            "hf_snapshot_hash": SNAPSHOT_HASH,
            "tokenizer_json_path": TOKENIZER_JSON,
            "tokenizer_json_sha256": EXPECTED_TOKENIZER_SHA256,
        },
        "results_dir": RESULTS_DIR,
        "num_languages": len(langs),
        "vocab_size": int(W.shape[1]),
        "packed_model_path": PACKED_MODEL_PATH,
        "packed_model_bytes": packed_size,
        "packed_model_sha256": packed_sha256,
    }
    with open(PACKED_META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"wrote packed model: {PACKED_MODEL_PATH} ({packed_size:,} bytes)")
    print(f"wrote packed meta: {PACKED_META_PATH}")
    print("\nPACKING COMPLETE")


if __name__ == "__main__":
    main()
