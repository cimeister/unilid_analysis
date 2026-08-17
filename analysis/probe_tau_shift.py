"""How far do the per-language thresholds move under the correction?

tau_L is a percentile of the score margins (s1 - s2) on the language's own
training lines, computed on the clamped matrix. A per-token shift of log 5 does
not cancel in a margin, because the two languages being differenced can segment a
line into different numbers of tokens, so the thresholds have to be re-estimated
rather than carried. This measures how far they actually move on a handful of
languages before committing to all 1,084.

The released model is clamped at its own c = -21; the corrected model at
-21 + log 5 = -19.3906, which reproduces the old clamped matrix up to the uniform
shift and is therefore the like-for-like comparison.

  python -m analysis.probe_tau_shift -o outputs/rerelease/probe_tau.json
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "UNILID"))

from analysis.floor_equalization import build_equalized_weights  # noqa: E402

STORE = Path("/capstor/store/cscs/swissai/a0229/cmeister/unilid_analysis")
RELEASED = STORE / "glotlidc.unilid"
CORRECTED = Path("/capstor/scratch/cscs/cmeister747/unilid_analysis/corrected/"
                 "glotlidc_corrected.unilid")
CORPUS_DIR = STORE / "results_apertus200k/corpus"
TRAIN_COUNTS = STORE / "glotlid_unilid/glotlid_train_counts.json"

C_RELEASED = -21.0
C_CORRECTED = -21.0 + float(np.log(5.0))
# The recipe of record's constants.
HEAD_N = 18000
MARGIN_Q = 5.0
CALIB_MAX = 2000
MIN_CALIB_LINES = 200
CALIB_SEED = 0
TOPK = 5


def q_for(n_l: int) -> float:
    return MARGIN_Q * (1.0 - min(n_l, HEAD_N) / HEAD_N)


def calibration_lines(lang: str, n_l: int) -> list[str]:
    path = CORPUS_DIR / f"{lang}_train.txt"
    lines = [l for l in path.read_text(encoding="utf-8", errors="replace")
             .splitlines() if l.strip()]
    if len(lines) > CALIB_MAX:
        rng = np.random.default_rng(CALIB_SEED)
        idx = np.sort(rng.choice(len(lines), CALIB_MAX, replace=False))
        lines = [lines[i] for i in idx]
    return lines


def tau_for(model, langs, lang: str, lines: list[str], n_l: int) -> dict:
    """The recipe of record: top-k over the language's own lines, keep the ones
    it wins, take the size-adaptive percentile of the winning margins."""
    idx_of = {l: i for i, l in enumerate(langs)}
    want = idx_of[lang]
    gaps = []
    n_won = 0
    for start in range(0, len(lines), 2000):
        batch = lines[start:start + 2000]
        for row in model.model.top_k_of_cached_weight_sets_batch(batch, TOPK):
            if not row or row[0][0] != want:
                continue
            n_won += 1
            if len(row) > 1:
                gap = float(row[0][1]) - float(row[1][1])
                if np.isfinite(gap):
                    gaps.append(gap)
    q = q_for(n_l)
    out = {"lang": lang, "n_l": n_l, "q": q, "n_scoreable": len(lines),
           "n_self_won": n_won, "n_finite_gaps": len(gaps)}
    if len(gaps) < MIN_CALIB_LINES:
        out["tau"] = None
        out["cause"] = "low_calibration"
    elif q <= 0:
        out["tau"] = None
        out["cause"] = "zero_strength"
    else:
        out["tau"] = float(np.percentile(np.array(gaps), q))
        out["cause"] = ""
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--output", default=None)
    ap.add_argument("--n-langs", type=int, default=6)
    args = ap.parse_args(argv)

    from unilid.constants import SPECIAL_TOKENS
    from unilid.model_io import UnilidModel, load_unilid_raw

    counts = json.loads(TRAIN_COUNTS.read_text())
    _tok, _w, langs = load_unilid_raw(RELEASED)
    # Group A is the languages the thresholds exist for: N_L below head_n.
    group_a = sorted((l for l in langs
                      if counts.get(l, 0) < HEAD_N
                      and (CORPUS_DIR / f"{l}_train.txt").is_file()),
                     key=lambda l: (counts[l], l))
    pick = [group_a[i] for i in
            np.linspace(0, len(group_a) - 1, args.n_langs).round().astype(int)]
    print(f"probing {len(pick)} of {len(group_a):,} group-A languages, N_L "
          f"{counts[pick[0]]:,} to {counts[pick[-1]]:,}", flush=True)

    lines_by_lang = {l: calibration_lines(l, counts[l]) for l in pick}
    results = {}
    for tag, path, c in (("released", RELEASED, C_RELEASED),
                         ("corrected", CORRECTED, C_CORRECTED)):
        tok_json, weights, langs_m = load_unilid_raw(path)
        tok_text = tok_json if isinstance(tok_json, str) else tok_json.decode("utf-8")
        vocab = [t for t, _ in json.loads(tok_text)["model"]["vocab"]]
        special_idx = [vocab.index(t) for t in SPECIAL_TOKENS.values() if t in vocab]
        W = np.asarray(weights).astype(np.float32)
        del weights
        w_cal, n_mod = build_equalized_weights(W, c, special_idx)
        del W
        model = UnilidModel(path, calibrated=False)
        model.model.set_weight_sets_numpy(w_cal)
        del w_cal
        gc.collect()
        print(f"\n=== {tag}: clamped at c={c:.4f} ({n_mod} rows modified)", flush=True)
        rows = []
        for lang in pick:
            r = tau_for(model, langs_m, lang, lines_by_lang[lang], counts[lang])
            rows.append(r)
            tau = "excluded" if r["tau"] is None else f"{r['tau']:.4f}"
            print(f"  {lang:12} N={r['n_l']:>6,} q={r['q']:.3f}  "
                  f"own-won {r['n_self_won']:>5}/{r['n_scoreable']:<5} tau={tau}",
                  flush=True)
        results[tag] = rows
        del model
        gc.collect()

    print("\n=== tau, released against corrected ===")
    deltas = []
    for a, b in zip(results["released"], results["corrected"]):
        if a["tau"] is None or b["tau"] is None:
            print(f"  {a['lang']:12} excluded in at least one model "
                  f"({a['cause'] or 'ok'} / {b['cause'] or 'ok'})")
            continue
        d = b["tau"] - a["tau"]
        deltas.append(d)
        print(f"  {a['lang']:12} {a['tau']:>9.4f} -> {b['tau']:>9.4f}   "
              f"delta {d:+.4f}  ({d / a['tau'] * 100:+.1f}%)")
    if deltas:
        print(f"  mean delta {np.mean(deltas):+.4f}, median {np.median(deltas):+.4f}, "
              f"max |delta| {max(abs(d) for d in deltas):.4f}")

    if args.output:
        Path(args.output).write_text(json.dumps(
            {"c_released": C_RELEASED, "c_corrected": C_CORRECTED,
             "results": results}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
