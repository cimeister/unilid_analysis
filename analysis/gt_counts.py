"""Per-language token counts for the Good-Turing unseen-mass candidate (plan B4).

For each of the 1,940 languages: rebuild the language's standalone tokenizer from the
shared base tokenizer plus its weight row (the unpack pattern from
UNILID/unilid/model_io.py::unpack_unilid), Viterbi-encode its per-language training
corpus file with the ordinary Tokenizer.encode_batch (unmodified upstream code path,
the tokenizer's own normalizer/pretokenizer applied), and count token-id frequencies.

Output row per language (outputs/diagnostic/gt_counts.csv):
- T: total non-special tokens over the corpus
- n1: non-special vocab types occurring exactly once
- n_types: distinct non-special types with count > 0
- plateau_size, plateau_mass, floor: the row's exact-minimum plateau (the emergent
  unseen-token block, Exp 10) and its total probability mass
- plateau_types_used: plateau types that Viterbi nevertheless selected (diagnostic;
  expected near zero)
- corpus_lines, empty_lines

Resumable per language (rows already in the CSV are skipped; a torn trailing row from
a mid-write kill is dropped and recomputed). A missing corpus file aborts. Global
gates before any counting: exactly 4 special columns at p=SPECIAL_P, 1,940 corpus
files present, first-language probe encodes non-empty ids.

Segmentation path, deliberate choice: encode_batch runs Unigram Viterbi per ByteLevel
pre-token piece, while the deployment scorer runs Viterbi over the joined preprocessed
string; they differ only when a vocab token would span a ByteLevel boundary (rare).
The GT estimate is intended to reflect these encode_batch statistics.

The GT candidate itself (built later by full_test_gt.py) uses the one-sided-min rule
fixed by the user 2026-07-23: target plateau mass M' = min(M_L, NONSPECIAL_BUDGET *
n1/T); this script only measures T, n1, M_L.
"""
from __future__ import annotations

import json
import os
import struct

import numpy as np

from analysis.transfer_sweep import _load_model_data, UNILID_MODEL_PATH
from analysis.floor_equalization import SPECIAL_P

CORPUS_DIR = "/capstor/scratch/cscs/cmeister747/unilid_analysis/results_apertus200k/corpus"
OUT_CSV = "outputs/diagnostic/gt_counts.csv"
ENCODE_CHUNK = 20_000
HEADER = ("lang,T,n1,n_types,plateau_size,plateau_mass,floor,plateau_types_used,"
          "corpus_lines,empty_lines")


def run(out_csv: str = OUT_CSV) -> str:
    from tokenizers import Tokenizer

    weights, langs, _m = _load_model_data()
    W = np.asarray(weights)
    n_lang, vocab_size = W.shape

    special_cols = np.where(np.all(np.abs(np.exp(W.astype(np.float64))
                                          - SPECIAL_P) < 1e-4, axis=0))[0]
    if len(special_cols) != 4:
        raise RuntimeError(f"expected 4 special columns at p={SPECIAL_P}, "
                           f"found {len(special_cols)}")
    special_set = set(int(c) for c in special_cols)

    missing = [l for l in langs
               if not os.path.exists(os.path.join(CORPUS_DIR, f"{l}_train.txt"))]
    if missing:
        raise FileNotFoundError(f"{len(missing)} corpus files missing, first: "
                                f"{missing[0]}")

    # base tokenizer JSON read straight from the .unilid header (same format as
    # transfer_sweep._load_model_data); the Rust weight cache is not needed here
    header_fmt = "<8sIIIII4x"
    with open(UNILID_MODEL_PATH, "rb") as f:
        header = f.read(struct.calcsize(header_fmt))
        _, _, h_langs, h_vocab, base_tok_len, _langs_len = struct.unpack(header_fmt,
                                                                         header)
        base_str = f.read(base_tok_len).decode("utf-8")
    if (h_langs, h_vocab) != (n_lang, vocab_size):
        raise RuntimeError(f"header ({h_langs},{h_vocab}) != weights "
                           f"({n_lang},{vocab_size})")
    base_tok = Tokenizer.from_str(base_str)
    ref_vocab = base_tok.get_vocab()
    if len(ref_vocab) != vocab_size:
        raise RuntimeError(f"base vocab {len(ref_vocab)} != weight width {vocab_size}")
    vocab_list = sorted(ref_vocab.items(), key=lambda x: x[1])
    base_state = json.loads(base_tok.model.__getstate__().decode("utf-8"))
    base_state.pop("type", None)

    n_cols = HEADER.count(",") + 1
    done = set()
    if os.path.exists(out_csv):
        with open(out_csv) as f:
            first = f.readline().rstrip("\n")
            if first != HEADER:
                raise RuntimeError(f"{out_csv} has unexpected header {first!r}")
            rows = [line.rstrip("\n") for line in f if line.strip()]
        complete = [r for r in rows if r.count(",") == n_cols - 1]
        if len(complete) != len(rows):
            # a mid-write kill leaves a torn final line; drop it and recompute that
            # language rather than trusting a prefix (review fix, 2026-07-23)
            print(f"dropping {len(rows) - len(complete)} torn row(s) on resume")
            with open(out_csv + ".tmp", "w") as f:
                f.write(HEADER + "\n" + "".join(r + "\n" for r in complete))
            os.replace(out_csv + ".tmp", out_csv)
        done = {r.split(",", 1)[0] for r in complete}
        print(f"resuming: {len(done)} languages already counted")
    else:
        os.makedirs(os.path.dirname(out_csv), exist_ok=True)
        with open(out_csv, "w") as f:
            f.write(HEADER + "\n")

    out_f = open(out_csv, "a", buffering=1)
    for li, lang in enumerate(langs):
        if lang in done:
            continue
        row = W[li].astype(np.float64)
        floor = float(row.min())
        plateau = row == floor
        plateau_size = int(plateau.sum())
        plateau_mass = float(np.exp(row[plateau]).sum())

        state = dict(base_state)
        state["vocab"] = [(tok, float(W[li, tid])) for tok, tid in vocab_list]
        lang_tok = Tokenizer.from_str(base_str)
        lang_tok.model = lang_tok.model.__class__(**state)

        counts = np.zeros(vocab_size, dtype=np.int64)
        corpus_lines = empty_lines = 0
        chunk = []
        with open(os.path.join(CORPUS_DIR, f"{lang}_train.txt")) as fh:
            for line in fh:
                text = line.rstrip("\n")
                corpus_lines += 1
                if not text:
                    empty_lines += 1
                    continue
                chunk.append(text)
                if len(chunk) >= ENCODE_CHUNK:
                    for enc in lang_tok.encode_batch(chunk):
                        np.add.at(counts, enc.ids, 1)
                    chunk = []
        if chunk:
            for enc in lang_tok.encode_batch(chunk):
                np.add.at(counts, enc.ids, 1)
        if li == 0 and counts.sum() == 0:
            raise RuntimeError(f"probe language {lang} produced zero tokens; "
                               "tokenizer reconstruction broken")

        ns = np.ones(vocab_size, bool)
        ns[list(special_set)] = False
        c = counts[ns]
        T = int(c.sum())
        n1 = int((c == 1).sum())
        n_types = int((c > 0).sum())
        plateau_used = int(((counts > 0) & plateau & ns).sum())
        if T == 0:
            raise RuntimeError(f"{lang}: zero non-special tokens over "
                               f"{corpus_lines} corpus lines")
        out_f.write(f"{lang},{T},{n1},{n_types},{plateau_size},{plateau_mass:.8e},"
                    f"{floor:.6f},{plateau_used},{corpus_lines},{empty_lines}\n")
        if (li + 1) % 50 == 0:
            print(f"{li + 1}/{n_lang} languages counted (latest {lang}: T={T:,}, "
                  f"n1={n1:,}, n1/T={n1 / T:.5f})", flush=True)
    out_f.close()

    with open(out_csv) as f:
        n_rows = sum(1 for _ in f) - 1
    if n_rows != n_lang:
        raise RuntimeError(f"{out_csv} has {n_rows} rows, expected {n_lang}")
    print(f"Wrote {out_csv} ({n_rows} languages)")
    return out_csv


if __name__ == "__main__":
    run()
