"""Does corpus size alone set the unseen-token plateau, or is it language identity?

`analysis/plateau_reference_fit.py` measures the plateau falling by about 2.04
nats per decade of training tokens across 1,940 languages. That relation is
confounded: each point is a different language, so corpus size and language
identity vary together and neither can be credited on its own.

This holds language identity fixed. One language's corpus is shuffled once and
nested prefixes of it are retrained against the same unmodified base tokenizer,
so the only quantity that changes between runs is how much text the estimator
saw. If the within-language slope matches the cross-language slope, corpus size
alone accounts for the relation and the paper's appendix can say so. If it does
not, the relation is partly a property of which languages happen to be small, and
the appendix must state the correlation without the causal claim.

Also records `real_missing`, the number of base-vocabulary tokens absent from the
SentencePiece model, for which the trainer falls back to the base tokenizer's
log-prob. That fallback is the one mechanism that could manufacture a plateau
without the fit doing it, so it has to be shown to be near zero. The trainer only
logs it, so it is captured here from the log record rather than by changing the
shipped package.

Note on the offset: rows retrained under 0.3.0 are normalized over their real
tokens, whereas the released model's rows carry the special-token defect and sum
to 0.2. Retrained plateaus therefore sit log 5 = 1.6094 nats above the reference
fit's intercept. The SLOPE, which is the quantity under test, is unaffected.

  python -m analysis.plateau_vs_corpus_size -o outputs/rerelease/plateau_vs_corpus_size.json
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "UNILID"))

from analysis.gate_correction import CORPUS_DIR, TRAIN_COUNTS, retrain_row  # noqa: E402
from analysis.plateau_reference_fit import fit  # noqa: E402

from unilid.constants import SPECIAL_TOKENS  # noqa: E402
from unilid.model_io import load_unilid_raw  # noqa: E402

RELEASED = ("/capstor/store/cscs/swissai/a0229/cmeister/unilid_analysis/"
            "glotlidc.unilid")
REFERENCE_FIT = REPO / "outputs/rerelease/plateau_reference_fit.json"

# Nested prefixes of one shuffle, spanning two decades up to the corpus cap.
SIZES = [1_000, 3_000, 10_000, 30_000, 100_000]
SHUFFLE_SEED = 20260817
N_LANGS = 3
LOG5 = float(np.log(5.0))
# The within-language slope counts as matching the cross-language slope if it
# lands inside this band around it. Set from the cross-language fit's own spread:
# the token-based and line-based fits differ by 0.035 nats/decade, and the
# residual scatter is R^2 0.985, so a factor-of-two disagreement is the smallest
# difference worth calling a mismatch.
SLOPE_MATCH_TOLERANCE = 0.5      # fraction of the reference slope
MAX_REAL_MISSING_FRACTION = 0.01  # of the vocabulary


class _CaptureMissing(logging.Handler):
    """Record the trainer's `N tokens absent from the sentencepiece model`
    warning, which is the only place that count is reported."""

    PATTERN = re.compile(r"^(\d+) tokens absent from the sentencepiece model")

    def __init__(self):
        super().__init__(level=logging.WARNING)
        self.counts = []

    def emit(self, record):
        m = self.PATTERN.match(record.getMessage())
        if m:
            self.counts.append(int(m.group(1)))


def subsample(path: Path, n: int, seed: int) -> list[str]:
    """A nested prefix of one fixed shuffle, so the smaller corpora are subsets
    of the larger ones and only the amount of text varies."""
    lines = [l for l in path.read_text(encoding="utf-8", errors="replace")
             .splitlines() if l.strip()]
    order = np.random.default_rng(seed).permutation(len(lines))
    if n > len(lines):
        raise ValueError(f"{path.name} has {len(lines):,} usable lines, "
                         f"cannot take {n:,}")
    return [lines[i] for i in sorted(order[:n].tolist())]


def plateau_of(row: np.ndarray, real_mask: np.ndarray) -> tuple[float, int]:
    """The row's unseen-token plateau: the exact minimum over REAL tokens, and
    how many entries sit on it. Specials are excluded because 0.3.0 parks them at
    the training floor, below every real token."""
    real = row[real_mask]
    value = float(real.min())
    return value, int((real == real.min()).sum())


def count_tokens(tok_path: Path, lines: list[str]) -> int:
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(tok_path))
    return sum(len(e.tokens) for e in tok.encode_batch(lines))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--output", default=None)
    ap.add_argument("--n-langs", type=int, default=N_LANGS)
    args = ap.parse_args(argv)

    reference = json.loads(REFERENCE_FIT.read_text())
    ref_lines = reference["vs_log10_lines"]["slope"]
    ref_tokens = reference["vs_log10_tokens"]["slope"]
    print(f"reference (across 1,940 languages): {ref_lines:+.3f} nats per decade "
          f"of lines, {ref_tokens:+.3f} per decade of tokens")

    tok_json, _weights, langs = load_unilid_raw(RELEASED)
    tok_text = tok_json if isinstance(tok_json, str) else tok_json.decode("utf-8")
    vocab = [t for t, _ in json.loads(tok_text)["model"]["vocab"]]
    spec = set(SPECIAL_TOKENS.values())
    real_mask = np.array([t not in spec for t in vocab])
    counts = json.loads(Path(TRAIN_COUNTS).read_text())

    # Deterministic spread over the languages whose corpus reaches the largest
    # size, so two decades are available without cherry-picking one language.
    at_cap = sorted(l for l in langs
                    if counts.get(l, 0) >= max(SIZES)
                    and (CORPUS_DIR / f"{l}_train.txt").is_file())
    idx = np.linspace(0, len(at_cap) - 1, args.n_langs).round().astype(int)
    chosen = [at_cap[i] for i in dict.fromkeys(idx.tolist())]
    print(f"{len(at_cap):,} languages reach {max(SIZES):,} lines; "
          f"retraining {chosen}\n")

    handler = _CaptureMissing()
    trainer_log = logging.getLogger(
        "unilid.trainers.language_specific_trainer")
    trainer_log.addHandler(handler)

    results = {}
    with tempfile.TemporaryDirectory() as tmp:
        for lang in chosen:
            corpus = CORPUS_DIR / f"{lang}_train.txt"
            rows = []
            for n in SIZES:
                work = Path(tmp) / f"{lang}_{n}"
                work.mkdir(parents=True, exist_ok=True)
                lines = subsample(corpus, n, SHUFFLE_SEED)
                sub_path = work / "corpus.txt"
                sub_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

                handler.counts.clear()
                t0 = time.time()
                row = retrain_row(tok_text, lang, sub_path, vocab, work)
                secs = round(time.time() - t0, 1)
                value, size = plateau_of(row, real_mask)
                written = sorted(work.glob(f"langspec_*_{lang}.tokenizer.json"))
                n_tokens = count_tokens(written[0], lines)
                missing = sum(handler.counts)

                rec = {"lines": n, "tokens": n_tokens, "plateau": value,
                       "plateau_pre_0_3_0_scale": value - LOG5,
                       "plateau_size": size, "real_missing": missing,
                       "seconds": secs}
                rows.append(rec)
                print(f"  {lang:12} lines {n:>7,}  tokens {n_tokens:>9,}  "
                      f"plateau {value:>8.3f}  block {size:>6,}  "
                      f"missing {missing:>4}  ({secs}s)", flush=True)

            x_lines = np.log10([r["lines"] for r in rows])
            x_tokens = np.log10([r["tokens"] for r in rows])
            y = np.array([r["plateau"] for r in rows])
            results[lang] = {
                "n_l": counts[lang], "points": rows,
                "vs_log10_lines": fit(x_lines, y),
                "vs_log10_tokens": fit(x_tokens, y)}
            f = results[lang]["vs_log10_lines"]
            g = results[lang]["vs_log10_tokens"]
            print(f"  {lang:12} within-language slope {f['slope']:+.3f}/decade of "
                  f"lines (R^2 {f['r_squared']:.3f}), {g['slope']:+.3f}/decade of "
                  f"tokens (R^2 {g['r_squared']:.3f})\n", flush=True)

    trainer_log.removeHandler(handler)

    print("=== within-language against cross-language ===")
    verdicts = []
    for lang, r in results.items():
        s_lines = r["vs_log10_lines"]["slope"]
        s_tokens = r["vs_log10_tokens"]["slope"]
        ok_lines = abs(s_lines - ref_lines) <= SLOPE_MATCH_TOLERANCE * abs(ref_lines)
        ok_tokens = abs(s_tokens - ref_tokens) <= SLOPE_MATCH_TOLERANCE * abs(ref_tokens)
        worst_missing = max(p["real_missing"] for p in r["points"])
        ok_missing = worst_missing <= MAX_REAL_MISSING_FRACTION * len(vocab)
        verdict = bool(ok_lines and ok_tokens and ok_missing)
        verdicts.append(verdict)
        r["matches_reference"] = verdict
        r["max_real_missing"] = worst_missing
        print(f"  {lang:12} lines {s_lines:+.3f} vs {ref_lines:+.3f} "
              f"{'ok' if ok_lines else 'MISMATCH'}   "
              f"tokens {s_tokens:+.3f} vs {ref_tokens:+.3f} "
              f"{'ok' if ok_tokens else 'MISMATCH'}   "
              f"max real_missing {worst_missing} "
              f"{'ok' if ok_missing else 'TOO HIGH'}")

    # Put both fits on the released model's scale and compare their predictions
    # across the observed range of T. The within-language fits come from rows
    # normalized over real tokens, which sit log 5 above the released rows.
    mean_slope = float(np.mean([r["vs_log10_tokens"]["slope"]
                                for r in results.values()]))
    mean_intercept = float(np.mean([r["vs_log10_tokens"]["intercept"]
                                    for r in results.values()]) - LOG5)
    print(f"\n=== both fits on the released model's scale ===")
    print(f"  within-language  plateau = {mean_intercept:.3f} "
          f"{mean_slope:+.3f} * log10 T")
    print(f"  cross-language   plateau = {reference['vs_log10_tokens']['intercept']:.3f} "
          f"{ref_tokens:+.3f} * log10 T")
    agreement = []
    for lt in (4.0, 5.0, 6.0, 6.5, 7.0):
        within = mean_intercept + mean_slope * lt
        cross = (reference["vs_log10_tokens"]["intercept"] + ref_tokens * lt)
        agreement.append({"log10_tokens": lt, "within": within, "cross": cross,
                          "difference": within - cross})
        print(f"  log10 T={lt:<4} within {within:8.3f}  cross {cross:8.3f}  "
              f"difference {within - cross:+.3f}")
    exponent = mean_slope / float(np.log(10))
    print(f"\n  the plateau probability scales as T^{exponent:.2f}, that is, "
          f"approximately as one count in T")

    passed = all(verdicts)
    print(f"\nB0: {'PASS' if passed else 'FAIL'} "
          f"({sum(verdicts)}/{len(verdicts)} languages). "
          + ("Corpus size alone reproduces the cross-language slope with language "
             "identity held fixed."
             if passed else
             "The within-language slope does not match; corpus size does not by "
             "itself account for the cross-language relation."))

    out = {"reference": reference, "sizes": SIZES, "shuffle_seed": SHUFFLE_SEED,
           "log5": LOG5, "languages": results, "passed": passed,
           "within_language_on_released_scale": {
               "slope": mean_slope, "intercept": mean_intercept,
               "exponent_on_tokens": exponent},
           "agreement_across_range": agreement,
           "thresholds": {"slope_match_tolerance": SLOPE_MATCH_TOLERANCE,
                          "max_real_missing_fraction": MAX_REAL_MISSING_FRACTION}}
    if args.output:
        Path(args.output).write_text(json.dumps(out, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
