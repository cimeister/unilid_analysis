"""Extract only the base tokenizer from a .unilid container, with a preflight.

Used to retrain a model's per-language weights over its ORIGINAL vocabulary. The
extracted file goes to <results-dir>/tokenizers/langspec_base_tokenizer.json,
which train.py loads when given --reuse-base and --base-tokenizer-path, so
_convert_to_unigram_base never runs (train.py:455).

Only the base tokenizer is written. `unilid.model_io.unpack_unilid` would also
write one langspec_sp_<lang>.tokenizer.json per language, and those rows carry the
special-token defect. A retrain writes langspec_soft_<lang>.tokenizer.json, and
convert.py globs langspec_soft_* before langspec_sp_* (model_io.py:135-137), so
the pack would pick the corrected set by naming coincidence rather than by
construction. Leaving the defective files out removes the coincidence.

  python -m analysis.extract_base_tokenizer MODEL.unilid --results-dir DIR
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "UNILID"))

from unilid.constants import SPECIAL_TOKENS  # noqa: E402
from unilid.model_io import load_unilid_raw  # noqa: E402

STORE_ROOT = "/capstor/store/cscs/swissai"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model")
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("-o", "--output", default=None, help="preflight record JSON")
    a = ap.parse_args(argv)

    res = os.path.abspath(a.results_dir)
    # train.py:372 makedirs the results dir with no guard, and several
    # results_* names in the scratch root are symlinks into the durable store.
    if os.path.realpath(res).startswith(STORE_ROOT):
        raise SystemExit(
            f"refusing to use {res}: it resolves into {STORE_ROOT}, the durable "
            f"store. Training there would write over published artifacts.")

    tok_json, weights, langs = load_unilid_raw(a.model)
    del weights
    text = tok_json if isinstance(tok_json, str) else tok_json.decode("utf-8")
    d = json.loads(text)
    vocab = [t for t, _ in d["model"]["vocab"]]

    # preflight, all aborting
    if d["model"].get("type") != "Unigram":
        raise SystemExit(f"base tokenizer type is {d['model'].get('type')!r}, "
                         f"expected Unigram")
    missing = [t for t in SPECIAL_TOKENS.values() if t not in vocab]
    if missing:
        raise SystemExit(f"special tokens absent from the base vocabulary: {missing}")
    special_idx = [vocab.index(t) for t in SPECIAL_TOKENS.values()]

    out_dir = os.path.join(res, "tokenizers")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "langspec_base_tokenizer.json")
    Path(out_path).write_text(text, encoding="utf-8")

    rec = {"source_model": os.path.abspath(a.model), "results_dir": res,
           "base_tokenizer": out_path,
           "vocab_size": len(vocab), "n_languages": len(langs),
           "model_type": d["model"]["type"],
           "special_token_indices": special_idx,
           "base_tokenizer_sha256": hashlib.sha256(text.encode()).hexdigest()}
    print(f"wrote {out_path}")
    print(f"  vocab {rec['vocab_size']:,}  languages {rec['n_languages']:,}  "
          f"specials at {special_idx}")
    print(f"  sha256 {rec['base_tokenizer_sha256'][:32]}")
    print(f"  --vocab-size {rec['vocab_size']} must be passed to train.py")
    if a.output:
        Path(a.output).write_text(json.dumps(rec, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
