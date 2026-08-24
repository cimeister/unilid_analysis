"""Aborting preflight for a WiLI retrain, run inside the SLURM job before train.py.

Every check raises SystemExit with the offending artifact named. Nothing is
substituted, defaulted or repaired.

Checks, in order:
 1. the base tokenizer file exists and parses;
 2. its model type is Unigram (train.py loads it as the base for a Unigram
    per-language step; a BPE file here would mean the conversion never ran);
 3. its entry count equals --expect-vocab, which is the number the job passes to
    train.py as --vocab-size;
 4. all four UNILID special tokens are present, and their indices are reported
    (they are NOT the same across models: 1/2/10/0 for Mistral-Nemo, 1/2/31949/0
    for the Mistral conversion and 1/2/31976/0 for the LLaMA-2 one -- <pad> is
    appended last by the converter, so dropping entries under the 2026-08-23
    decision moves it down);
 5. its sha256 equals --expect-sha256, recorded when the base was extracted or
    converted. This is what catches a base file rebuilt from a different source
    between preparation and submission;
 6. --results-dir does not resolve into the durable store (train.py:372 calls
    os.makedirs with no guard, and several results_* names in the scratch root
    are symlinks into the store);
 7. --results-dir/tokenizers holds no per-language rows. train.py's
    --skip-existing-langs defaults to true and the loader validates token order
    but not real-token mass, so a pre-0.3.0 row left in place would be loaded
    and packed uncorrected;
 8. the corpus directory holds exactly WILI_LANG_COUNT *_train.txt files
    totalling WILI_TRAIN_LINES lines;
 9. --output-container, if given, does not already exist.

In the from-scratch mode of the plan's Phase 3 there is deliberately NO base
tokenizer, because training the base vocabulary is the point. Pass
--require-absent-base instead of --base: checks 1 to 5 are replaced by the single
check that the file is absent, so a leftover base from an earlier attempt cannot
be picked up by --reuse-base and silently reused. Checks 6 to 9 still run.

  python -m analysis.preflight_wili_base --base B.json --expect-vocab 131072 \
      --expect-sha256 <64 hex> --results-dir DIR --corpus DIR \
      --output-container OUT.unilid
  python -m analysis.preflight_wili_base --require-absent-base B.json \
      --results-dir DIR --corpus DIR --output-container OUT.unilid
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "UNILID"))

from unilid.constants import SPECIAL_TOKENS  # noqa: E402

STORE_ROOT = "/capstor/store/cscs/swissai"
# --- constants defined by this script -------------------------------------
# The WiLI-2018 training split as drawn from the GitHub release and written to
# wili_corpus_shared by train.prepare_corpus: 235 languages, 500 lines each.
# Measured on the shared corpus 2026-08-23 and stated in the plan's Context.
WILI_LANG_COUNT = 235
WILI_TRAIN_LINES = 117_500
# Chunk size for streaming a file into sha256.
HASH_BLOCK = 1 << 20


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(HASH_BLOCK)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", default=None)
    ap.add_argument("--expect-vocab", type=int, default=None)
    ap.add_argument("--expect-sha256", default=None)
    ap.add_argument("--require-absent-base", default=None,
                    help="from-scratch mode: this path must NOT exist")
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--output-container", default=None)
    a = ap.parse_args(argv)

    if bool(a.base) == bool(a.require_absent_base):
        raise SystemExit("PREFLIGHT FAIL: pass exactly one of --base (reuse an "
                         "existing base vocabulary) or --require-absent-base "
                         "(train a fresh one)")
    if a.require_absent_base:
        if a.expect_vocab is not None or a.expect_sha256 is not None:
            raise SystemExit("PREFLIGHT FAIL: --expect-vocab / --expect-sha256 "
                             "describe an existing base and cannot be combined "
                             "with --require-absent-base")
        if os.path.exists(a.require_absent_base):
            raise SystemExit(
                f"PREFLIGHT FAIL: a base tokenizer already exists at "
                f"{a.require_absent_base}. This job trains its base vocabulary "
                f"from the corpus; --reuse-base defaults to true and train.py:455 "
                f"would reuse this file instead. Move it aside deliberately.")
        return _tail_checks(a, base_report=None)

    if a.expect_vocab is None or a.expect_sha256 is None:
        raise SystemExit("PREFLIGHT FAIL: --base requires both --expect-vocab "
                         "and --expect-sha256")
    base = Path(a.base)
    if not base.is_file():
        raise SystemExit(f"PREFLIGHT FAIL: base tokenizer missing: {base}")
    try:
        d = json.loads(base.read_text(encoding="utf-8"))
    except Exception as e:
        raise SystemExit(f"PREFLIGHT FAIL: {base} does not parse as JSON: {e}")
    if "model" not in d or "vocab" not in d.get("model", {}):
        raise SystemExit(f"PREFLIGHT FAIL: {base} has no model.vocab")
    mtype = d["model"].get("type")
    if mtype != "Unigram":
        raise SystemExit(f"PREFLIGHT FAIL: {base} model type is {mtype!r}, "
                         f"expected 'Unigram'")
    vocab = [t for t, _ in d["model"]["vocab"]]
    if len(vocab) != a.expect_vocab:
        raise SystemExit(
            f"PREFLIGHT FAIL: {base} has {len(vocab):,} entries, expected "
            f"{a.expect_vocab:,}. --vocab-size and the base file disagree.")
    missing = [t for t in SPECIAL_TOKENS.values() if t not in vocab]
    if missing:
        raise SystemExit(f"PREFLIGHT FAIL: special tokens absent from {base}: "
                         f"{missing}")
    special_idx = [vocab.index(t) for t in SPECIAL_TOKENS.values()]
    got = _sha256(base)
    if got != a.expect_sha256:
        raise SystemExit(
            f"PREFLIGHT FAIL: {base} sha256 is {got}, expected "
            f"{a.expect_sha256}. This base tokenizer is not the one this job "
            f"was prepared against.")

    return _tail_checks(a, base_report=(base, len(vocab), special_idx, got))


def _tail_checks(a, base_report):
    res = os.path.abspath(a.results_dir)
    if os.path.realpath(res).startswith(STORE_ROOT):
        raise SystemExit(
            f"PREFLIGHT FAIL: {res} resolves into {STORE_ROOT}, the durable "
            f"store. Training there would write over published artifacts.")
    rows = (glob.glob(os.path.join(res, "tokenizers", "langspec_sp_*"))
            + glob.glob(os.path.join(res, "tokenizers", "langspec_soft_*")))
    if rows:
        raise SystemExit(
            f"PREFLIGHT FAIL: {res}/tokenizers already holds {len(rows)} "
            f"per-language rows (e.g. {os.path.basename(rows[0])}). Refusing "
            f"to resume: --skip-existing-langs defaults to true and would "
            f"reuse them without checking their real-token mass.")

    corpus = Path(a.corpus)
    if not corpus.is_dir():
        raise SystemExit(f"PREFLIGHT FAIL: corpus missing: {corpus}")
    files = sorted(corpus.glob("*_train.txt"))
    if len(files) != WILI_LANG_COUNT:
        raise SystemExit(
            f"PREFLIGHT FAIL: {corpus} holds {len(files)} *_train.txt files, "
            f"expected {WILI_LANG_COUNT}")
    total = 0
    for p in files:
        with open(p, "rb") as f:
            total += sum(1 for _ in f)
    if total != WILI_TRAIN_LINES:
        raise SystemExit(
            f"PREFLIGHT FAIL: {corpus} holds {total:,} lines, expected "
            f"{WILI_TRAIN_LINES:,}")

    if a.output_container and os.path.exists(a.output_container):
        raise SystemExit(
            f"PREFLIGHT FAIL: output container already exists: "
            f"{a.output_container}. Move it aside deliberately.")

    if base_report is None:
        print("PREFLIGHT OK  no base tokenizer, as required "
              f"({a.require_absent_base} is absent; it will be trained)")
    else:
        base, n, special_idx, got = base_report
        print(f"PREFLIGHT OK  base={base}")
        print(f"  Unigram, {n:,} entries, specials at {special_idx}")
        print(f"  sha256 {got}")
    print(f"  results-dir {res} (no per-language rows, not store-backed)")
    print(f"  corpus {corpus}: {len(files)} languages, {total:,} lines")
    if a.output_container:
        print(f"  output container {a.output_container} does not yet exist")
    return 0


if __name__ == "__main__":
    sys.exit(main())
