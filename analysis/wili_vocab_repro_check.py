"""Is WiLI base-vocabulary training reproducible? Phase 3's gating check.

The plan's Phase 3 needs four models (10k / 20k / 50k / 200k) whose base
vocabularies have no container to be lifted from, so each must be trained here.
Nothing verifies those four directly. The one piece of evidence available is the
100k size, where a stored vocabulary DOES exist: train a 100k base vocabulary
from the WiLI corpus with default settings and compare it against the base
vocabulary inside `wili_assets/wili_100k_500.unilid`.

  Match    -> base-vocabulary training is reproducible, and the four sizes are
              the published models.
  No match -> the four are NEW models built by the published procedure, not the
              published models, and the table has to say so.

Recorded either way; never silently substituted.

**Token lists are compared, not serialisations.** A sha256 over the JSON would
also capture `tokenizers` library formatting, which has nothing to do with
whether the same vocabulary was learned. Scores are excluded for the same
reason: the base scores are replaced by the per-language step, so the token
inventory in its trained order is the thing that has to reproduce.

Exit codes: 0 match, 1 no match (a recorded outcome, not a failure of this
script), 2 aborted before deciding anything.

  python -m analysis.wili_vocab_repro_check --trained RES/tokenizers/langspec_base_tokenizer.json \
      --container /capstor/.../wili_assets/wili_100k_500.unilid -o out.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "UNILID"))

from unilid.model_io import load_unilid_raw  # noqa: E402

# --- constants defined by this script -------------------------------------
EXIT_MATCH = 0
EXIT_NO_MATCH = 1
EXIT_ABORT = 2
# How many tokens to show around the first divergence.
CONTEXT = 5


def _abort(msg: str):
    print(f"ABORT: {msg}", file=sys.stderr)
    sys.exit(EXIT_ABORT)


def _tokens_from_tokenizer_json(text: str, where: str):
    try:
        d = json.loads(text)
    except Exception as e:
        _abort(f"{where} does not parse as JSON: {e}")
    model = d.get("model")
    if not isinstance(model, dict) or "vocab" not in model:
        _abort(f"{where} has no model.vocab")
    if model.get("type") != "Unigram":
        _abort(f"{where} model type is {model.get('type')!r}, expected 'Unigram'")
    vocab = model["vocab"]
    if not vocab:
        _abort(f"{where} has an empty vocabulary")
    try:
        return [t for t, _ in vocab]
    except Exception as e:
        _abort(f"{where} model.vocab is not a list of (token, score) pairs: {e}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trained", required=True,
                    help="freshly trained base tokenizer JSON")
    ap.add_argument("--container", required=True,
                    help=".unilid container holding the stored base vocabulary")
    ap.add_argument("-o", "--output", default=None)
    a = ap.parse_args(argv)

    tp = Path(a.trained)
    if not tp.is_file():
        _abort(f"trained base tokenizer missing: {tp}")
    cp = Path(a.container)
    if not cp.is_file():
        _abort(f"container missing: {cp}")

    trained_text = tp.read_text(encoding="utf-8")
    fresh = _tokens_from_tokenizer_json(trained_text, str(tp))

    tok_json, _weights, langs = load_unilid_raw(str(cp))
    stored_text = tok_json if isinstance(tok_json, str) else tok_json.decode("utf-8")
    stored = _tokens_from_tokenizer_json(stored_text, f"{cp} (base tokenizer)")

    first_div = None
    for i in range(min(len(fresh), len(stored))):
        if fresh[i] != stored[i]:
            first_div = i
            break
    if first_div is None and len(fresh) != len(stored):
        first_div = min(len(fresh), len(stored))
    ordered_match = first_div is None and len(fresh) == len(stored)
    overlap = len(set(fresh) & set(stored))

    rec = {
        "trained_base": str(tp.resolve()),
        "trained_base_sha256": hashlib.sha256(trained_text.encode()).hexdigest(),
        "trained_vocab_size": len(fresh),
        "container": str(cp.resolve()),
        "container_base_sha256": hashlib.sha256(stored_text.encode()).hexdigest(),
        "container_vocab_size": len(stored),
        "container_n_languages": len(langs),
        "ordered_token_list_match": ordered_match,
        "first_divergence_index": first_div,
        "overlap_tokens": overlap,
        "overlap_fraction_of_stored": overlap / len(stored),
    }
    if first_div is not None:
        lo = max(0, first_div - CONTEXT)
        rec["trained_context"] = fresh[lo:first_div + CONTEXT]
        rec["stored_context"] = stored[lo:first_div + CONTEXT]

    print(f"trained  {len(fresh):,} tokens  {tp}")
    print(f"stored   {len(stored):,} tokens  {cp}")
    print(f"overlap  {overlap:,} tokens "
          f"({overlap / len(stored):.4%} of the stored vocabulary)")
    if ordered_match:
        print("VERDICT: MATCH — the ordered token lists are identical. "
              "Base-vocabulary training is reproducible and the four vocabulary "
              "sizes are the published models.")
    else:
        print(f"VERDICT: NO MATCH — first divergence at index {first_div}.")
        if first_div is not None and "trained_context" in rec:
            print(f"  trained[{max(0, first_div - CONTEXT)}:]: {rec['trained_context']}")
            print(f"  stored [{max(0, first_div - CONTEXT)}:]: {rec['stored_context']}")
        print("  The four vocabulary sizes are NEW models built by the published "
              "procedure, not the published models, and the table must say so.")
    if a.output:
        Path(a.output).write_text(json.dumps(rec, indent=2))
        print(f"wrote {a.output}")
    return EXIT_MATCH if ordered_match else EXIT_NO_MATCH


if __name__ == "__main__":
    sys.exit(main())
