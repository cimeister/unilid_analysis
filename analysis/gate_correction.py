"""Gate the special-token correction: does it reproduce a retrain?

The correction claims that a corrected row equals what UNILID 0.3.0 produces
when it trains the same corpus against the same base tokenizer. This retrains a
spread of languages with the current code and compares, so the claim is measured
rather than argued from the arithmetic.

  python -m analysis.gate_correction CORRECTED.unilid --n-langs 8 -o out.json
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "UNILID"))

from unilid.constants import SPECIAL_TOKENS  # noqa: E402
from unilid.model_io import _get_vocab_with_scores, load_unilid_raw  # noqa: E402

STORE = Path("/capstor/store/cscs/swissai/a0229/cmeister/unilid_analysis")
CORPUS_DIR = STORE / "results_apertus200k/corpus"
TRAIN_COUNTS = STORE / "glotlid_unilid/glotlid_train_counts.json"

# What this gate can and cannot establish.
#
# A retrain cannot be expected to reproduce a stored row exactly. Languages whose
# available data exceeded the 100,000-line cap were subsampled, and the corpus
# directory here is the one the Apertus run drew, not necessarily the draw the
# released model saw. Measured on zul_Latn (N_L = 100,000): two retrains of the
# same corpus are bit-identical (max difference 0.0), while the retrain differs
# from the stored row with signed mean -1.0e-3 and a maximum of 2.04 confined to
# tokens below p = 1e-5, which together carry 1.6% of the mass. That is the
# signature of a different sample of the same language.
#
# What the gate must catch is a wrong transformation, which would appear as a
# systematic offset: getting the constant wrong shifts every real token by the
# same amount, so the SIGNED mean deviation is the discriminating statistic and
# it is bounded tightly. Divergence is then measured where it affects scores, by
# weighting each token's deviation by its probability, rather than by a median
# over 100,000 tokens that are almost all negligible.
MAX_ABS_SIGNED_MEAN = 0.01      # a wrong constant would be O(1), e.g. log 5
MAX_MASS_WEIGHTED_DIFF = 0.02   # sum of p * |delta| over the real tokens
MIN_CORRELATION = 0.9999


def pick_languages(langs, counts, n):
    """Spread over the training-size range, deterministically."""
    have = [l for l in langs
            if l in counts and (CORPUS_DIR / f"{l}_train.txt").is_file()]
    have.sort(key=lambda l: (counts[l], l))
    if n >= len(have):
        return have
    idx = np.linspace(0, len(have) - 1, n).round().astype(int)
    return [have[i] for i in dict.fromkeys(idx.tolist())]


def retrain_row(base_tok_json: str, lang: str, corpus: Path, vocab, workdir: Path):
    from unilid.trainers.language_specific_trainer import (
        LanguageSpecificUnigramLMTokenizer,
    )
    from tokenizers import Tokenizer

    workdir.mkdir(parents=True, exist_ok=True)
    base_path = workdir / "base.json"
    base_path.write_text(base_tok_json, encoding="utf-8")
    trainer = LanguageSpecificUnigramLMTokenizer(num_iterations=20)
    trainer.train(corpus_info={lang: str(corpus)},
                  base_tokenizer_path=str(base_path), output_dir=str(workdir),
                  load_base_if_exists=True, load_tokenizers_if_exists=False,
                  use_sentencepiece=True, num_reestimation_iterations=20)
    written = sorted(workdir.glob(f"langspec_*_{lang}.tokenizer.json"))
    if len(written) != 1:
        raise RuntimeError(f"expected one trained tokenizer for {lang}, "
                           f"got {[p.name for p in written]}")
    scores = {t: s for t, s in _get_vocab_with_scores(Tokenizer.from_file(str(written[0])))}
    return np.array([scores[t] for t in vocab], dtype=np.float64)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("corrected")
    ap.add_argument("--n-langs", type=int, default=8)
    ap.add_argument("--langs", default=None,
                    help="comma-separated languages to gate IN ADDITION to the "
                         "size-spread selection. Use it to target a row that is "
                         "suspect for another reason, e.g. azj_Latn, which "
                         "carries the fp64 EM bug in the Qwen3 variant.")
    ap.add_argument("-o", "--output", default=None)
    args = ap.parse_args(argv)

    tok_json, weights, langs = load_unilid_raw(args.corrected)
    tok_text = tok_json if isinstance(tok_json, str) else tok_json.decode("utf-8")
    vocab = [t for t, _ in json.loads(tok_text)["model"]["vocab"]]
    spec = {v for v in SPECIAL_TOKENS.values()}
    real_mask = np.array([t not in spec for t in vocab])
    counts = json.loads(TRAIN_COUNTS.read_text())

    chosen = pick_languages(langs, counts, args.n_langs)
    if args.langs:
        extra = [l.strip() for l in args.langs.split(",") if l.strip()]
        missing = [l for l in extra
                   if l not in counts or not (CORPUS_DIR / f"{l}_train.txt").is_file()]
        if missing:
            raise SystemExit(f"requested language(s) with no corpus or count: {missing}")
        chosen = chosen + [l for l in extra if l not in chosen]
        print(f"  targeted additions: {extra}", flush=True)
    print(f"gating {len(chosen)} languages spanning N_L "
          f"{min(counts[l] for l in chosen):,} to "
          f"{max(counts[l] for l in chosen):,}", flush=True)

    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for k, lang in enumerate(chosen, 1):
            corrected = np.asarray(weights[langs.index(lang)]).astype(np.float64)
            t0 = time.time()
            got = retrain_row(tok_text, lang, CORPUS_DIR / f"{lang}_train.txt",
                              vocab, Path(tmp) / lang)
            delta = got[real_mask] - corrected[real_mask]
            p_stored = np.exp(corrected[real_mask])
            corr = float(np.corrcoef(got[real_mask], corrected[real_mask])[0, 1])
            row = {"lang": lang, "n_l": counts[lang],
                   "signed_mean": float(delta.mean()),
                   "mass_weighted_diff": float((p_stored * np.abs(delta)).sum()),
                   "median_abs_diff": float(np.median(np.abs(delta))),
                   "max_abs_diff": float(np.abs(delta).max()),
                   "correlation": corr,
                   "seconds": round(time.time() - t0, 1)}
            row["passed"] = (abs(row["signed_mean"]) <= MAX_ABS_SIGNED_MEAN
                             and row["mass_weighted_diff"] <= MAX_MASS_WEIGHTED_DIFF
                             and corr >= MIN_CORRELATION)
            rows.append(row)
            print(f"  [{k}/{len(chosen)}] {lang:12} N={counts[lang]:>6,}  "
                  f"signed mean {row['signed_mean']:+.2e}  "
                  f"mass-weighted {row['mass_weighted_diff']:.2e}  "
                  f"corr {corr:.6f}  {'PASS' if row['passed'] else 'FAIL'}  "
                  f"({row['seconds']}s)", flush=True)

    passed = all(r["passed"] for r in rows)
    result = {"corrected": args.corrected, "languages": rows, "passed": passed,
              "thresholds": {"max_abs_signed_mean": MAX_ABS_SIGNED_MEAN,
                             "max_mass_weighted_diff": MAX_MASS_WEIGHTED_DIFF,
                             "min_correlation": MIN_CORRELATION}}
    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=2))
    print(f"\ncorrection gate: {'PASS' if passed else 'FAIL'} "
          f"({sum(r['passed'] for r in rows)}/{len(rows)} languages)")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
