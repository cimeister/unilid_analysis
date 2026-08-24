"""Convert an LLM tokenizer.json into a UNILID base tokenizer, with preflights.

This is the Phase 2b counterpart of `analysis/extract_base_tokenizer.py`. Where
that script lifts an existing base vocabulary out of a `.unilid` container, this
one builds a base vocabulary from a HuggingFace tokenizer that has no container,
which is the case for the `\\unilid-Mistral`, `\\unilid-LLaMA3.2` and
`\\unilid-LLaMA2` rows.

It does NOT reimplement the conversion. It calls `UNILID/train.py`'s own
`_convert_to_unigram_base` (train.py:52), the same function train.py runs at
train.py:458-462 when `--initial-vocab` names an HF tokenizer and no base
tokenizer exists yet. Running it here rather than inside the SLURM job makes the
base vocabulary an artifact that can be preflight-checked before hours of
per-language training, and lets every Phase 2 job script take the same shape as
the Phase 1 template (`--base-tokenizer-path` + `--reuse-base`).

What the conversion does, from the code: it takes the source's vocabulary in id
order, gives every non-special token the same uniform log probability and every
special token 0.0, appends any of UNILID's four special tokens the source lacks,
and keeps the source's normalizer, pretokenizer and decoder. The per-language
step then replaces those scores, so the conversion supplies the token inventory
and nothing else.

  python -m analysis.convert_llm_tokenizer_base SOURCE_TOKENIZER.json \
      --results-dir DIR --expect-source-vocab N [-o record.json] \
      [--drop-refused-entries]

`--expect-source-vocab` is mandatory and aborting: the entry count is the only
thing that distinguishes these tokenizers from one another, and a silently
different source tokenizer would produce a model that is not the row it claims
to be.

--drop-refused-entries (OFF by default)
---------------------------------------
SentencePiece-style sources (Mistral-7B-v0.2, Llama-2-7b-hf) carry vocabulary
entries containing a raw carriage return, tab or newline. The per-language
SentencePiece path refuses to write such an entry into its seed vocabulary file
(`_write_sp_seed_vocab_file`, UNILID/unilid/vocab_io.py:119-120, "Token contains
tab/newline"), so the whole retrain aborts minutes in.

AUTHOR DECISION, 2026-08-23: those entries are DROPPED from the converted base
vocabulary. Whole entries are dropped; nothing is sanitized, stripped or
re-encoded. `--drop-refused-entries` is what applies that decision, and it is
off by default: without it this script's behaviour is exactly what it was, and a
source with refused entries still converts and still fails downstream as before.

With the flag, the drop happens BEFORE the base is finalized, so the uniform log
probability the base carries is the uniform probability over the entries that
survive, which is what `_convert_to_unigram_base` would have computed had the
source never contained them. Every dropped token is recorded by name in the
output JSON. If the vocabulary has no refused entries the flag changes nothing:
the converter's own output file is kept untouched.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "UNILID"))

from unilid.constants import SPECIAL_TOKENS  # noqa: E402
from unilid.vocab_io import _write_sp_seed_vocab_file  # noqa: E402

STORE_ROOT = "/capstor/store/cscs/swissai"
TRAIN_PY = REPO / "UNILID" / "train.py"
VOCAB_IO_PY = REPO / "UNILID" / "unilid" / "vocab_io.py"

# --- constants defined by this script -------------------------------------
# The three characters the per-language SentencePiece seed-vocab writer refuses.
# Mirror of the predicate at UNILID/unilid/vocab_io.py:119
#     if ("\t" in t) or ("\n" in t) or ("\r" in t):
#         raise ValueError(f"Token contains tab/newline: {repr(t)}")
# It is a mirror and not the authority: _refused_by_sp_seed_writer() below is
# checked against the real writer, imported above, in BOTH directions before any
# entry is dropped (_verify_filter_against_writer).
SP_SEED_REFUSED_CHARS = ("\t", "\n", "\r")
# vocab_io.py:113/117-118 skips the special tokens before that check, so a
# special could never be refused. `unk_token` there is whatever the trainer
# passes; the per-language trainer passes SPECIAL_TOKENS["unk_token"]
# (language_specific_trainer.py:75), so the skipped set is the four specials.
SP_SEED_UNK_TOKEN = SPECIAL_TOKENS["unk_token"]
# The author decision this flag implements, stamped into every record it writes.
DROP_DECISION_DATE = "2026-08-23"
# Substring of the writer's own message, used to confirm that a probe failed for
# the refusal reason and not some unrelated one.
REFUSAL_MESSAGE = "Token contains tab/newline"


def _load_train_module():
    """Import UNILID/train.py as a module so its own converter is the one used."""
    spec = importlib.util.spec_from_file_location("unilid_train_script", TRAIN_PY)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot import {TRAIN_PY}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "_convert_to_unigram_base"):
        raise SystemExit(
            f"{TRAIN_PY} has no _convert_to_unigram_base; the conversion path "
            f"this script depends on has moved or been removed.")
    return mod


def _sp_seed_specials() -> set:
    """The set vocab_io.py:113 builds and skips before the refusal check."""
    return set(SPECIAL_TOKENS.values()) | {SP_SEED_UNK_TOKEN}


def _refused_by_sp_seed_writer(token: str) -> bool:
    """True iff _write_sp_seed_vocab_file would raise on this token.

    Mirrors UNILID/unilid/vocab_io.py:119. Callers must apply the specials skip
    of vocab_io.py:117-118 themselves.
    """
    return any(ch in token for ch in SP_SEED_REFUSED_CHARS)


def _verify_filter_against_writer(dropped: list, kept: list, probe_dir: str):
    """Check the mirror against the real writer, in both directions.

    Direction 1, nothing over-dropped: every token this script drops must make
    the real `_write_sp_seed_vocab_file` raise its refusal.
    Direction 2, nothing under-dropped: the surviving vocabulary must pass
    through the real writer in one call without raising.
    """
    probe = os.path.join(probe_dir, "seed_probe.tsv")
    for t in dropped:
        try:
            _write_sp_seed_vocab_file([(t, 0.0)], probe, SP_SEED_UNK_TOKEN)
        except ValueError as e:
            if REFUSAL_MESSAGE not in str(e):
                raise SystemExit(
                    f"{VOCAB_IO_PY} rejected {t!r} for an unexpected reason: "
                    f"{e}. The drop filter mirrors only the tab/newline "
                    f"refusal; refusing to guess at another one.")
        else:
            raise SystemExit(
                f"drop filter flagged {t!r} but {VOCAB_IO_PY}'s "
                f"_write_sp_seed_vocab_file accepts it. The filter is no longer "
                f"a mirror of the writer; refusing to drop entries the writer "
                f"would have kept.")
    try:
        n = _write_sp_seed_vocab_file([(t, 0.0) for t in kept], probe,
                                      SP_SEED_UNK_TOKEN)
    except ValueError as e:
        raise SystemExit(
            f"the surviving vocabulary is still refused by "
            f"{VOCAB_IO_PY}'s _write_sp_seed_vocab_file: {e}. The drop filter "
            f"is not a complete mirror of the writer's refusal.")
    expect = len([t for t in kept if t not in _sp_seed_specials()])
    if n != expect:
        raise SystemExit(
            f"seed-vocab probe wrote {n} lines, expected {expect} "
            f"(surviving entries minus the specials the writer skips)")
    os.remove(probe)


def _rebuild_without(conv_path: str, drop: set, out_path: str) -> dict:
    """Rebuild the converted base with `drop` removed, before it is finalized.

    Uses the same builder `_convert_to_unigram_base` uses
    (unilid.tokenizer_builder._build_unigramlm_hf_tokenizer_from_lprobs) and the
    same score convention (train.py:70-79): 0.0 for every special, one uniform
    log probability for every other SURVIVING entry.
    """
    from tokenizers import Tokenizer  # noqa: E402
    from unilid.tokenizer_builder import (  # noqa: E402
        _build_unigramlm_hf_tokenizer_from_lprobs)
    from unilid.constants import MIN_TOKEN_LOG_PROB  # noqa: E402

    conv = Tokenizer.from_file(conv_path)
    d = json.loads(Path(conv_path).read_text(encoding="utf-8"))
    special_set = set(SPECIAL_TOKENS.values())
    kept = [t for t, _ in d["model"]["vocab"] if t not in drop]
    non_special = [t for t in kept if t not in special_set]
    uniform_lp = np.log(1.0 / len(non_special)) if non_special else MIN_TOKEN_LOG_PROB
    tuples = [(t, 0.0 if t in special_set else uniform_lp) for t in kept]
    if SPECIAL_TOKENS["unk_token"] not in kept:
        raise SystemExit(f"{SPECIAL_TOKENS['unk_token']!r} absent from the "
                         f"surviving vocabulary; cannot set unk_id")
    unk_id = kept.index(SPECIAL_TOKENS["unk_token"])
    tok = _build_unigramlm_hf_tokenizer_from_lprobs(
        tuples, unk_id, pretokenizer=conv.pre_tokenizer, decoder=conv.decoder)
    if conv.normalizer:
        tok.normalizer = conv.normalizer
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    tok.save(out_path)
    return {"kept": len(kept), "unk_id": unk_id, "uniform_log_prob": float(uniform_lp)}


def _assert_rebuild_is_faithful(conv_path: str, tmpdir: str) -> bool:
    """Rebuild with an EMPTY drop set and require the converter's own output back.

    This is what makes the drop the only difference between the converter's base
    and the base this script writes. Returns True iff the bytes also matched.
    """
    probe = os.path.join(tmpdir, "rebuild_identity.json")
    _rebuild_without(conv_path, set(), probe)
    a = Path(conv_path).read_bytes()
    b = Path(probe).read_bytes()
    if a != b and json.loads(a) != json.loads(b):
        raise SystemExit(
            f"rebuilding the converted base with nothing dropped did not "
            f"reproduce it ({conv_path} vs {probe}). The rebuild path is not "
            f"equivalent to train.py:_convert_to_unigram_base; refusing to use "
            f"it to drop entries.")
    os.remove(probe)
    return a == b


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("source", help="HuggingFace tokenizer.json to convert")
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--expect-source-vocab", type=int, required=True,
                    help="model.vocab entry count the source MUST have")
    ap.add_argument("-o", "--output", default=None, help="record JSON")
    ap.add_argument("--drop-refused-entries", action="store_true",
                    help="AUTHOR DECISION " + DROP_DECISION_DATE + ": drop the "
                         "vocabulary entries the per-language SentencePiece "
                         "seed-vocab writer refuses (vocab_io.py:119, raw tab / "
                         "newline / carriage return). Whole entries only. Off by "
                         "default; without it such a source converts as before "
                         "and still fails in training.")
    a = ap.parse_args(argv)

    src = os.path.abspath(a.source)
    if not os.path.isfile(src):
        raise SystemExit(f"source tokenizer missing: {src}")

    res = os.path.abspath(a.results_dir)
    # train.py:372 makedirs the results dir with no guard, and several
    # results_* names in the scratch root are symlinks into the durable store.
    if os.path.realpath(res).startswith(STORE_ROOT):
        raise SystemExit(
            f"refusing to use {res}: it resolves into {STORE_ROOT}, the durable "
            f"store. Training there would write over published artifacts.")

    raw = Path(src).read_bytes()
    src_sha = hashlib.sha256(raw).hexdigest()
    sd = json.loads(raw)
    if "model" not in sd or "vocab" not in sd.get("model", {}):
        raise SystemExit(f"{src} is not an HF tokenizer JSON (no model.vocab); "
                         f"train.py:_is_hf_tokenizer would reject it too")
    src_vocab = sd["model"]["vocab"]
    n_src = len(src_vocab)
    if n_src != a.expect_source_vocab:
        raise SystemExit(
            f"source vocabulary size mismatch: {src} has {n_src} model.vocab "
            f"entries, --expect-source-vocab says {a.expect_source_vocab}. "
            f"Refusing to convert a tokenizer that is not the declared one.")

    out_dir = os.path.join(res, "tokenizers")
    out_path = os.path.join(out_dir, "langspec_base_tokenizer.json")
    if os.path.exists(out_path):
        raise SystemExit(
            f"refusing to overwrite an existing base tokenizer at {out_path}. "
            f"Delete it deliberately if it is meant to be rebuilt.")
    os.makedirs(out_dir, exist_ok=True)

    train_mod = _load_train_module()

    drop_rec = {"flag_given": bool(a.drop_refused_entries),
                "refusal_site": f"{VOCAB_IO_PY}:119",
                "refused_chars": [repr(c) for c in SP_SEED_REFUSED_CHARS]}

    with tempfile.TemporaryDirectory(prefix="unilid_convert_") as tmpdir:
        conv_path = os.path.join(tmpdir, "converted_base.json")
        train_mod._convert_to_unigram_base(src, conv_path)

        cd = json.loads(Path(conv_path).read_text(encoding="utf-8"))
        conv_vocab = [t for t, _ in cd["model"]["vocab"]]
        specials_skipped = _sp_seed_specials()
        refused = [t for t in conv_vocab
                   if t not in specials_skipped and _refused_by_sp_seed_writer(t)]
        # A special could never reach the refusal check (vocab_io.py:117-118
        # skips them), so dropping one would be over-dropping. Named separately
        # so this can never be silently folded into `refused`.
        refused_specials = [t for t in conv_vocab
                            if t in specials_skipped and _refused_by_sp_seed_writer(t)]
        drop_rec["converted_entries_before_drop"] = len(conv_vocab)
        drop_rec["refused_entries_found"] = len(refused)

        if refused_specials:
            raise SystemExit(
                f"{len(refused_specials)} UNILID SPECIAL token(s) contain a "
                f"refused character: {[repr(t) for t in refused_specials]}. "
                f"{VOCAB_IO_PY}:117-118 skips specials before the check, so "
                f"these would never be refused and must not be dropped. This "
                f"needs the author.")

        if not a.drop_refused_entries:
            if refused:
                print(f"WARNING: {len(refused)} of {len(conv_vocab)} converted "
                      f"entries contain a raw tab/newline/carriage return and "
                      f"will be REFUSED by {VOCAB_IO_PY}:119 during per-language "
                      f"training. --drop-refused-entries was NOT given, so they "
                      f"are kept and this base will fail downstream, as before.")
            shutil.copyfile(conv_path, out_path)
            drop_rec["applied"] = False
            drop_rec["dropped_tokens"] = []
            drop_rec["dropped_count"] = 0
        elif not refused:
            # Inert by construction: the converter's own file is what is kept.
            print(f"--drop-refused-entries: 0 refused entries in "
                  f"{len(conv_vocab)} - the converter's output is used "
                  f"unchanged.")
            shutil.copyfile(conv_path, out_path)
            drop_rec["applied"] = False
            drop_rec["dropped_tokens"] = []
            drop_rec["dropped_count"] = 0
            drop_rec["decision_date"] = DROP_DECISION_DATE
        else:
            refused_set = set(refused)
            _verify_filter_against_writer(
                refused, [t for t in conv_vocab if t not in refused_set], tmpdir)
            byte_identical = _assert_rebuild_is_faithful(conv_path, tmpdir)
            info = _rebuild_without(conv_path, refused_set, out_path)
            drop_rec.update({
                "applied": True,
                "decision_date": DROP_DECISION_DATE,
                "decision": "entries the SentencePiece seed-vocab writer refuses "
                            "are dropped whole from the converted base; nothing "
                            "is stripped, sanitized or re-encoded",
                "dropped_count": len(refused),
                "dropped_tokens": [repr(t) for t in refused],
                "entries_after_drop": info["kept"],
                "unk_id_after_drop": info["unk_id"],
                "uniform_log_prob_after_drop": info["uniform_log_prob"],
                "verified_against_writer": True,
                "identity_rebuild_byte_identical": byte_identical,
            })
            print("=" * 72)
            print(f"DROPPED {len(refused)} REFUSED VOCABULARY ENTRIES "
                  f"(author decision {DROP_DECISION_DATE})")
            print(f"  refusal: {VOCAB_IO_PY}:119, characters "
                  f"{[repr(c) for c in SP_SEED_REFUSED_CHARS]}")
            print(f"  {len(conv_vocab):,} converted entries -> "
                  f"{info['kept']:,} after the drop")
            for t in refused:
                print(f"    dropped {repr(t)}")
            print(f"  each dropped token confirmed refused by the real writer; "
                  f"the surviving vocabulary confirmed accepted by it")
            print(f"  identity rebuild reproduced the converter's base "
                  f"({'byte-identical' if byte_identical else 'JSON-equal'})")
            print("=" * 72)

    text = Path(out_path).read_text(encoding="utf-8")
    d = json.loads(text)
    # preflights, all aborting, mirroring analysis/extract_base_tokenizer.py
    if d["model"].get("type") != "Unigram":
        raise SystemExit(f"converted base type is {d['model'].get('type')!r}, "
                         f"expected Unigram")
    vocab = [t for t, _ in d["model"]["vocab"]]
    missing = [t for t in SPECIAL_TOKENS.values() if t not in vocab]
    if missing:
        raise SystemExit(f"special tokens absent after conversion: {missing}")
    special_idx = [vocab.index(t) for t in SPECIAL_TOKENS.values()]
    added = [t for t in SPECIAL_TOKENS.values() if t not in
             (set(src_vocab.keys()) if isinstance(src_vocab, dict)
              else {tok for tok, _ in src_vocab})]
    still_refused = [t for t in vocab
                     if t not in _sp_seed_specials() and _refused_by_sp_seed_writer(t)]
    if a.drop_refused_entries and still_refused:
        raise SystemExit(
            f"{len(still_refused)} refused entries survive in {out_path} after "
            f"--drop-refused-entries; the drop did not take effect")

    rec = {"source_tokenizer": src, "source_sha256": src_sha,
           "source_model_vocab_entries": n_src,
           "source_model_type": sd["model"].get("type"),
           "results_dir": res, "base_tokenizer": out_path,
           "vocab_size": len(vocab), "model_type": d["model"]["type"],
           "special_token_indices": special_idx,
           "specials_absent_from_source_model_vocab": added,
           "refused_entries_remaining": len(still_refused),
           "sp_seed_vocab_drop": drop_rec,
           "base_tokenizer_sha256": hashlib.sha256(text.encode()).hexdigest()}
    print(f"wrote {out_path}")
    print(f"  source {n_src:,} {sd['model'].get('type')} entries -> base "
          f"{rec['vocab_size']:,} Unigram entries")
    print(f"  specials at {special_idx}; absent from source model.vocab: {added}")
    print(f"  sha256 {rec['base_tokenizer_sha256'][:32]}")
    print(f"  --vocab-size {rec['vocab_size']} must be passed to train.py")
    if a.output:
        Path(a.output).write_text(json.dumps(rec, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
