"""Aborting preflight for a CLD3-subset training, run inside the SLURM job.

Sibling of `analysis/preflight_wili_base.py`, which bakes in WiLI's 235
languages and 117,500 lines and so cannot check a subset corpus. Kept separate
rather than generalising that module, because its hardcoded counts are the
guarantee the WiLI jobs rely on and several of those jobs are already recorded
against it.

Every check raises SystemExit with the offending artifact named. Nothing is
substituted, defaulted or repaired.

Checks, in order:
 1. the corpus directory matches the manifest `analysis/build_cld3_subset_corpus`
    wrote for this subset -- the same `*_train.txt` label set, the same
    per-label line counts, and the same total. This is what catches a corpus
    directory rebuilt for a different subset, or a shared-draw file that changed
    under the symlinks between preparation and submission;
 2. --results-dir does not resolve into the durable store (train.py:394 calls
    os.makedirs with no guard, and several results_* names in the scratch root
    are symlinks into the store);
 3. base-vocabulary discipline, in exactly one of two modes:
      --require-absent-base PATH  the fresh mode. PATH must NOT exist and
          --results-dir/tokenizers must hold no per-language rows, so nothing an
          earlier attempt left behind can be picked up by --reuse-base (which
          defaults to true, train.py:478) or by --skip-existing-langs (also true,
          train.py:353). The whole model is built in one job or not at all.
      --require-present-base PATH the resume mode. PATH must exist and parse as
          a Unigram tokenizer with --expect-vocab entries and all four UNILID
          special tokens. Per-language rows MAY exist and will be reused: they
          were estimated over this same file, which is why the fresh mode
          forbids the base from being retrained under them.
 4. --output-container, if given, does not already exist.

  python -m analysis.preflight_cld3_subset --subset 83 --manifest M.json \\
      --results-dir DIR --require-absent-base B.json --output-container OUT.unilid
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "UNILID"))

from unilid.constants import SPECIAL_TOKENS  # noqa: E402

from analysis.build_cld3_subset_corpus import SUBSETS, CORPUS_SUFFIX  # noqa: E402

STORE_ROOT = "/capstor/store/cscs/swissai"


def _check_corpus(subset: str, corpus: str, manifest_path: str) -> tuple[int, int]:
    if not os.path.isfile(manifest_path):
        raise SystemExit(f"PREFLIGHT FAIL: corpus manifest missing: {manifest_path}")
    man = json.load(open(manifest_path))
    if man["subset"] != subset:
        raise SystemExit(
            f"PREFLIGHT FAIL: {manifest_path} describes subset {man['subset']}, "
            f"this job is subset {subset}")
    if not os.path.isdir(corpus):
        raise SystemExit(f"PREFLIGHT FAIL: corpus directory missing: {corpus}")
    if os.path.abspath(corpus) != man["out_dir"]:
        raise SystemExit(
            f"PREFLIGHT FAIL: corpus {os.path.abspath(corpus)} is not the "
            f"directory the manifest was written for ({man['out_dir']})")

    _def, n_codes, n_corpora = SUBSETS[subset]
    if man["n_codes"] != n_codes or man["n_corpora"] != n_corpora:
        raise SystemExit(
            f"PREFLIGHT FAIL: manifest says {man['n_codes']} codes / "
            f"{man['n_corpora']} corpora, SUBSETS[{subset!r}] says "
            f"{n_codes} / {n_corpora}")

    on_disk = sorted(os.path.basename(p)[: -len(CORPUS_SUFFIX)]
                     for p in glob.glob(os.path.join(corpus, "*" + CORPUS_SUFFIX)))
    expected = sorted(man["labels"])
    if on_disk != expected:
        extra = sorted(set(on_disk) - set(expected))
        gone = sorted(set(expected) - set(on_disk))
        raise SystemExit(
            f"PREFLIGHT FAIL: {corpus} holds {len(on_disk)} corpora, the "
            f"manifest lists {len(expected)}. Unexpected: {extra[:10]}. "
            f"Missing: {gone[:10]}")

    total = 0
    for label in expected:
        p = os.path.join(corpus, label + CORPUS_SUFFIX)
        n = 0
        with open(p, "rb") as f:
            for _ in f:
                n += 1
        if n != man["lines_per_label"][label]:
            raise SystemExit(
                f"PREFLIGHT FAIL: {p} holds {n:,} lines, the manifest recorded "
                f"{man['lines_per_label'][label]:,}. The shared draw changed "
                f"under the symlink.")
        total += n
    if total != man["total_lines"]:
        raise SystemExit(
            f"PREFLIGHT FAIL: {corpus} holds {total:,} lines, the manifest "
            f"recorded {man['total_lines']:,}")
    return len(expected), total


# train.py's sample_corpus writes the base-fit sample as
# "<lang>_train.sampled.txt" under <results-dir>/corpus_base_sampled, and
# StandardUnigramLMTokenizer records the exact file list it was fitted on in the
# tokenizer's sidecar metadata: _collect_training_metadata builds it
# (unilid/trainers/pruning_strategy.py, from line 167) and
# _add_metadata_to_saved_tokenizer (line 156) writes it beside the tokenizer.
SAMPLED_SUFFIX = "_train.sampled.txt"


def base_metadata_path(base_tokenizer_path: str) -> str:
    """unilid/metadata.py:16-18: <name>.json -> <name>.metadata.json"""
    p = str(base_tokenizer_path)
    return (p[:-5] + ".metadata.json") if p.endswith(".json") \
        else p + ".metadata.json"


def check_base_provenance(base_tokenizer_path: str, expected_labels: list[str],
                          expect_vocab: int) -> dict:
    """Assert the base vocabulary was fitted on THIS subset's corpora.

    This is the claim the whole record rests on -- "the base tokenizer was
    trained on that subset of languages" -- so it is checked against the
    tokenizer's own record of the files it consumed, not against a per-run flag.
    `training_summary.json` only says whether the LAST run refitted the base, so
    a run resumed after a walltime kill reports `base_tokenizer_reused: true`
    for a base that was in fact fitted from scratch on the right corpora; that
    flag cannot answer this question and is not used for it.
    """
    meta_path = base_metadata_path(base_tokenizer_path)
    if not os.path.isfile(meta_path):
        raise SystemExit(
            f"FATAL: base tokenizer metadata missing at {meta_path}. Without it "
            f"there is no record of which corpora the vocabulary was fitted on, "
            f"which is the one property that makes this a subset model.")
    meta = json.load(open(meta_path))
    cfg = meta.get("training_config") or {}
    if cfg.get("vocab_size") != expect_vocab:
        raise SystemExit(
            f"FATAL: {meta_path} says the base was fitted at vocab_size "
            f"{cfg.get('vocab_size')}, expected {expect_vocab}")
    if cfg.get("em_mode") != "hf":
        raise SystemExit(
            f"FATAL: {meta_path} says em_mode={cfg.get('em_mode')!r}, expected "
            f"'hf' (train.py's default base training method)")
    if cfg.get("byte_level") is not True:
        raise SystemExit(
            f"FATAL: {meta_path} says byte_level={cfg.get('byte_level')!r}, "
            f"expected True")
    files = ((meta.get("corpus_info") or {}).get("training_files")) or []
    if not files:
        raise SystemExit(f"FATAL: {meta_path} records no training files")
    got = sorted(os.path.basename(f)[: -len(SAMPLED_SUFFIX)] for f in files
                 if f.endswith(SAMPLED_SUFFIX))
    if len(got) != len(files):
        raise SystemExit(
            f"FATAL: {meta_path} lists {len(files)} training files but only "
            f"{len(got)} are base samples named *{SAMPLED_SUFFIX}; the base was "
            f"not fitted on a train.py base sample.")
    want = sorted(expected_labels)
    if got != want:
        extra = sorted(set(got) - set(want))
        gone = sorted(set(want) - set(got))
        raise SystemExit(
            f"FATAL: the base vocabulary at {base_tokenizer_path} was fitted on "
            f"{len(got)} corpora, this subset has {len(want)}. Fitted but not "
            f"in the subset: {extra[:10]}. In the subset but not fitted: "
            f"{gone[:10]}. This vocabulary is not this subset's vocabulary.")
    sizes = (meta.get("corpus_info") or {}).get("corpus_sizes") or {}
    if not sizes:
        raise SystemExit(
            f"FATAL: {meta_path} records no corpus_sizes. How many lines the "
            f"base vocabulary was fitted on is part of the provenance this "
            f"record publishes, and it cannot be recovered from anywhere else.")
    return {
        "metadata_path": meta_path,
        "n_files": len(files),
        "vocab_size": cfg.get("vocab_size"),
        "em_mode": cfg.get("em_mode"),
        "byte_level": cfg.get("byte_level"),
        "base_fit_lines": sum(sizes.values()),
        "created_at": meta.get("created_at"),
    }


def _check_base_present(base: Path, expect_vocab: int) -> list[int]:
    if not base.is_file():
        raise SystemExit(f"PREFLIGHT FAIL: base tokenizer missing: {base}")
    try:
        d = json.loads(base.read_text(encoding="utf-8"))
    except Exception as e:
        raise SystemExit(f"PREFLIGHT FAIL: {base} does not parse as JSON: {e}")
    if "model" not in d or "vocab" not in d.get("model", {}):
        raise SystemExit(f"PREFLIGHT FAIL: {base} has no model.vocab")
    if d["model"].get("type") != "Unigram":
        raise SystemExit(f"PREFLIGHT FAIL: {base} model type is "
                         f"{d['model'].get('type')!r}, expected 'Unigram'")
    vocab = [t for t, _ in d["model"]["vocab"]]
    if len(vocab) != expect_vocab:
        raise SystemExit(
            f"PREFLIGHT FAIL: {base} has {len(vocab):,} entries, expected "
            f"{expect_vocab:,}. --vocab-size and the base file disagree.")
    missing = [t for t in SPECIAL_TOKENS.values() if t not in vocab]
    if missing:
        raise SystemExit(f"PREFLIGHT FAIL: special tokens absent from {base}: "
                         f"{missing}")
    return [vocab.index(t) for t in SPECIAL_TOKENS.values()]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--subset", required=True, choices=sorted(SUBSETS))
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--require-absent-base", default=None)
    ap.add_argument("--require-present-base", default=None)
    ap.add_argument("--expect-vocab", type=int, default=None)
    ap.add_argument("--output-container", default=None)
    a = ap.parse_args(argv)

    if bool(a.require_absent_base) == bool(a.require_present_base):
        raise SystemExit("PREFLIGHT FAIL: pass exactly one of "
                         "--require-absent-base (fresh training) or "
                         "--require-present-base (resume over the same base)")

    n_corpora, total_lines = _check_corpus(a.subset, a.corpus, a.manifest)
    manifest_labels = json.load(open(a.manifest))["labels"]

    res = os.path.abspath(a.results_dir)
    if os.path.realpath(res).startswith(STORE_ROOT):
        raise SystemExit(
            f"PREFLIGHT FAIL: {res} resolves into {STORE_ROOT}, the durable "
            f"store. Training there would write over published artifacts.")

    # Both prefixes, and the ".tokenizer.json" suffix. The prefix train.py's own
    # skip test builds is "langspec_<per-lang-method>_" (train.py:518-522), i.e.
    # "langspec_sp_" for this job family, but the per-language trainer maps sp ->
    # soft (train.py:571) and writes "langspec_soft_<lang>.tokenizer.json"
    # (language_specific_trainer.py:302-305), so the real files carry the other
    # prefix. Without the suffix these globs also match each row's
    # ".metadata.json" sidecar and report twice the true count.
    rows = (glob.glob(os.path.join(res, "tokenizers",
                                   "langspec_sp_*.tokenizer.json"))
            + glob.glob(os.path.join(res, "tokenizers",
                                     "langspec_soft_*.tokenizer.json")))

    if a.require_absent_base:
        if a.expect_vocab is not None:
            raise SystemExit("PREFLIGHT FAIL: --expect-vocab describes an "
                             "existing base and cannot be combined with "
                             "--require-absent-base")
        if os.path.exists(a.require_absent_base):
            raise SystemExit(
                f"PREFLIGHT FAIL: a base tokenizer already exists at "
                f"{a.require_absent_base}. This job trains its base vocabulary "
                f"from the subset corpus; --reuse-base defaults to true "
                f"(train.py:350-351) and the reuse test at train.py:478 "
                f"would reuse this file instead. Move it aside "
                f"deliberately, or resume with --require-present-base.")
        if rows:
            raise SystemExit(
                f"PREFLIGHT FAIL: {res}/tokenizers already holds {len(rows)} "
                f"per-language rows (e.g. {os.path.basename(rows[0])}) but no "
                f"base tokenizer. Those rows were estimated over a vocabulary "
                f"this job is about to replace; --skip-existing-langs defaults "
                f"to true and would pack them against the new one.")
        print(f"PREFLIGHT OK  fresh base: {a.require_absent_base} is absent, "
              f"{res}/tokenizers holds no rows")
    else:
        if a.expect_vocab is None:
            raise SystemExit("PREFLIGHT FAIL: --require-present-base requires "
                             "--expect-vocab")
        idx = _check_base_present(Path(a.require_present_base), a.expect_vocab)
        prov = check_base_provenance(a.require_present_base, manifest_labels,
                                     a.expect_vocab)
        # Every check that can still fail runs BEFORE the OK line. Printing
        # "PREFLIGHT OK" and then dying leaves a log whose last positive
        # statement is false.
        if len(rows) > n_corpora:
            raise SystemExit(
                f"PREFLIGHT FAIL: {res}/tokenizers holds {len(rows)} "
                f"per-language rows but this subset has only {n_corpora} "
                f"corpora. Rows for languages outside the subset would be "
                f"packed into the container by convert.py.")
        print(f"PREFLIGHT OK  resume over {a.require_present_base}")
        print(f"  Unigram, {a.expect_vocab:,} entries, specials at {idx}")
        print(f"  fitted on this subset's {prov['n_files']} corpora "
              f"({prov['base_fit_lines']:,} lines) at {prov['created_at']}")
        print(f"  {len(rows)} of {n_corpora} per-language rows already present; "
              f"they will be reused over this same base")

    if a.output_container and os.path.exists(a.output_container):
        raise SystemExit(
            f"PREFLIGHT FAIL: output container already exists: "
            f"{a.output_container}. Move it aside deliberately.")

    print(f"  corpus {a.corpus}: {n_corpora} lang_Script corpora, "
          f"{total_lines:,} lines (matches {a.manifest})")
    print(f"  results-dir {res} (not store-backed)")
    if a.output_container:
        print(f"  output container {a.output_container} does not yet exist")
    return 0


if __name__ == "__main__":
    sys.exit(main())
