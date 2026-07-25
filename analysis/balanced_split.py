"""Language-balanced validation protocol (plan item 10, the split redesign).

Why: the original protocol drew val uniformly from the head-dominated test set, so val
contains zero examples for 65 of 96 tail languages and at most 2 for all of them, and
median zero for magnets. Three guard-passed operating points (Apertus gamma=3.0, the
frequency prior's apparent tail safety, floor-21) were overturned at full-test scale
because selection could not see the strata it was supposed to protect.

Protocol:
- Pool: the 45,377,279 kept lines of the Exp 16 full-test evaluation (the test set
  minus the RETIRED original 250k val lines; those were tuning data for Exp 13-20
  selections and are never reused for anything).
- Draw r with seed SEEDS[r]: for each language, sample
  min(VAL_PER_LANG_K, floor(VAL_FRACTION * n_L)) of its pool lines without replacement
  (n_L = the language's pool count). Languages with n_L == 1 contribute nothing to val
  and keep their single example in test.
- The working val is draw 0; the remaining draws exist to check that a selected
  operating point is stable across val draws (split-selection variance).
- Test for a draw = pool minus that draw's val. Final full-test numbers exclude the
  union of the draw's val lines and the retired original val lines.
- Macro-F1 is a per-language average, so a language-balanced val estimates the
  selection objective with real sensitivity on every stratum; it is deliberately NOT
  distribution-matched (selection does not need distribution match; final test numbers
  keep the natural distribution).

Constants (documented here, referenced from EXPERIMENTAL_SETUP.md):
- VAL_PER_LANG_K = 100: per-language val cap; the full-test tail averages ~80 examples
  per language, so most tail languages contribute 30-50 under the fraction rule.
- VAL_FRACTION = 0.5: a scarce language never gives more than half its examples to
  val, so test coverage survives for every language with >= 2 examples.
- SEEDS = [101, 102, 103, 104, 105]: five draws; 101 is the working val.

Artifacts: outputs/diagnostic/balanced_val/{manifest.json, val_lines_seed{S}.npy};
seed-101 text cache on scratch (texts + labels + line indices); a re-baseline report
scoring the four SAVED full-test prediction sets (baseline, freq prior gamma=0.5,
learned bias reg=5.0, floor-21) under the new protocol, including the guard verdicts
the balanced val would have produced.
"""
from __future__ import annotations

import json
import os
import pickle

import numpy as np

from analysis.config import TEST_FILE, TOTAL_LINES
from analysis.metrics import compute_metrics
from analysis.transfer_sweep import _load_model_data, _load_train_counts
from analysis.hierarchical_pool import passes_guard, GUARD_STRATA, GUARD_TOL, DIAG_CSV
from analysis.full_test_eval import SCRATCH_DIR, _parse_line

VAL_PER_LANG_K = 100
VAL_FRACTION = 0.5
SEEDS = [101, 102, 103, 104, 105]
# Balanced TEST draw (2026-07-23, adoption-rule protocol): disjoint from the working
# val (draw 101). Deviation from the first plan wording (union-of-all-draws exclusion),
# recorded in EXPERIMENTAL_SETUP.md: excluding the union of five independent half-draws
# exhausts small pools (a 66-line tail pool keeps ~2 lines), so the test draw excludes
# draw 101 only and the stability draws 102-105 are REGENERATED to exclude the test
# draw (they had no consumers before 2026-07-23; verified by grep before regeneration).
TEST_SEED = 201
OUT_DIR = "outputs/diagnostic/balanced_val"
TEXT_CACHE = os.path.join(SCRATCH_DIR, "balanced_val_texts_seed101.pkl")
SAVED_PREDS = ["pred_baseline.npy", "pred_freq_prior.npy", "pred_learned_bias.npy",
               "pred_floor21.npy"]


def _load_y() -> np.ndarray:
    y = np.asarray(np.lib.format.open_memmap(
        os.path.join(SCRATCH_DIR, "y_true.npy"), mode="r"))
    if y.shape != (TOTAL_LINES,):
        raise RuntimeError(f"y_true memmap has shape {y.shape}")
    if int((y == -3).sum()) != 0:
        raise RuntimeError("y_true memmap incomplete (UNSEEN present)")
    return y


def _per_language_pool(y: np.ndarray, n_lang: int) -> list:
    """List of sorted line-index arrays, one per language, over the kept pool."""
    order = np.argsort(y, kind="stable")
    ys = y[order]
    starts = np.searchsorted(ys, np.arange(n_lang), side="left")
    ends = np.searchsorted(ys, np.arange(n_lang), side="right")
    return [np.sort(order[s:e]) for s, e in zip(starts, ends)]


def build_draws(out_dir: str = OUT_DIR) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    _w, langs, _m = _load_model_data()
    n_lang = len(langs)
    y = _load_y()
    pools = _per_language_pool(y, n_lang)
    pool_sizes = np.array([len(p) for p in pools])
    if int(pool_sizes.sum()) != int((y >= 0).sum()):
        raise RuntimeError("per-language pools do not partition the kept lines")

    manifest = {"K": VAL_PER_LANG_K, "fraction": VAL_FRACTION, "seeds": SEEDS,
                "pool_lines": int(pool_sizes.sum()), "draws": {}}
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        picked = []
        n_with_val = 0
        for li in range(n_lang):
            k = min(VAL_PER_LANG_K, int(VAL_FRACTION * pool_sizes[li]))
            if k <= 0:
                continue
            picked.append(rng.choice(pools[li], size=k, replace=False))
            n_with_val += 1
        val_lines = np.sort(np.concatenate(picked))
        if len(set(val_lines.tolist())) != len(val_lines):
            raise RuntimeError("duplicate val lines within a draw")
        path = os.path.join(out_dir, f"val_lines_seed{seed}.npy")
        np.save(path, val_lines)
        manifest["draws"][str(seed)] = {"n_lines": int(len(val_lines)),
                                        "languages_with_val": n_with_val}
        print(f"seed {seed}: {len(val_lines):,} val lines, "
              f"{n_with_val}/{n_lang} languages represented")

    train_counts = _load_train_counts()
    N = np.array([train_counts.get(l, 0) for l in langs])
    v0 = np.load(os.path.join(out_dir, f"val_lines_seed{SEEDS[0]}.npy"))
    vy = y[v0]
    sup = np.bincount(vy, minlength=n_lang)
    for name, mask in [("tail (N<1k)", N < 1000), ("all", np.ones(n_lang, bool))]:
        langs_m = int(mask.sum())
        ge10 = int(((sup >= 10) & mask).sum())
        ge30 = int(((sup >= 30) & mask).sum())
        med = int(np.median(sup[mask]))
        print(f"draw {SEEDS[0]} coverage, {name}: {langs_m} languages, median val "
              f"support {med}, >=10 examples {ge10}, >=30 examples {ge30}")
        manifest.setdefault("coverage_seed0", {})[name] = {
            "median_support": med, "ge10": ge10, "ge30": ge30}
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    return manifest


def _load_manifest(out_dir: str) -> dict:
    path = os.path.join(out_dir, "manifest.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"manifest missing: {path}; run build_draws first")
    with open(path) as f:
        return json.load(f)


def _save_manifest(out_dir: str, manifest: dict) -> None:
    path = os.path.join(out_dir, "manifest.json")
    with open(path + ".tmp", "w") as f:
        json.dump(manifest, f, indent=1)
    os.replace(path + ".tmp", path)


def _draw_excluding(pools, exclude_mask, seed, k_rule):
    """One balanced draw over pool lines not in exclude_mask.

    k_rule="remaining": k = min(K, floor(fraction * remaining)). Used for the TEST
      draw so a natural remainder survives beyond val + test for every language.
    k_rule="pool": k = min(K, floor(fraction * full pool), remaining). Used for the
      regenerated stability draws so their size matches the working val's k rule
      (exchangeable alternative vals); languages capped by the exclusion are counted
      and reported, never silently resized below that cap.
    """
    rng = np.random.default_rng(seed)
    picked = []
    n_with = 0
    n_shrunk = 0
    for p in pools:
        rem = p[~exclude_mask[p]]
        k_pool = min(VAL_PER_LANG_K, int(VAL_FRACTION * len(p)))
        if k_rule == "remaining":
            k = min(VAL_PER_LANG_K, int(VAL_FRACTION * len(rem)))
        elif k_rule == "pool":
            k = min(k_pool, len(rem))
        else:
            raise ValueError(f"unknown k_rule {k_rule!r}")
        if k <= 0:
            continue
        if k < k_pool:
            n_shrunk += 1
        picked.append(rng.choice(rem, size=k, replace=False))
        n_with += 1
    lines = np.sort(np.concatenate(picked))
    if len(set(lines.tolist())) != len(lines):
        raise RuntimeError("duplicate lines within a draw")
    if exclude_mask[lines].any():
        raise RuntimeError("draw overlaps its exclusion set")
    return lines, n_with, n_shrunk


def build_test_draw(out_dir: str = OUT_DIR) -> str:
    """Balanced TEST draw, seed TEST_SEED, disjoint from the working val (draw 101).
    Headline instrument of the precision-primary adoption rule; closes protocol
    caveat 5 in EXPERIMENTAL_SETUP.md."""
    _w, langs, _m = _load_model_data()
    n_lang = len(langs)
    y = _load_y()
    pools = _per_language_pool(y, n_lang)
    val101 = np.load(os.path.join(out_dir, f"val_lines_seed{SEEDS[0]}.npy"))
    excl = np.zeros(TOTAL_LINES, bool)
    excl[val101] = True
    lines, n_with, _ = _draw_excluding(pools, excl, TEST_SEED, k_rule="remaining")

    train_counts = _load_train_counts()
    N = np.array([train_counts.get(l, 0) for l in langs])
    sup = np.bincount(y[lines], minlength=n_lang)
    tail = N < 1000
    cov = {"n_lines": int(len(lines)), "languages_with_test": n_with,
           "tail_median_support": int(np.median(sup[tail])),
           "tail_ge10": int(((sup >= 10) & tail).sum()),
           "excludes": f"val draw seed {SEEDS[0]} only (see TEST_SEED comment)"}
    path = os.path.join(out_dir, f"val_lines_seed{TEST_SEED}.npy")
    np.save(path, lines)
    manifest = _load_manifest(out_dir)
    manifest["test_draw"] = {"seed": TEST_SEED, **cov}
    _save_manifest(out_dir, manifest)
    print(f"test draw seed {TEST_SEED}: {len(lines):,} lines, {n_with}/{n_lang} "
          f"languages, tail median support {cov['tail_median_support']}, "
          f"tail >=10 examples {cov['tail_ge10']}/96")
    return path


def rebuild_stability_draws(out_dir: str = OUT_DIR) -> None:
    """Regenerate draws 102-105 to exclude the seed-201 test draw. The original
    102-105 draws were never consumed by any fit, report, or cache (verified
    2026-07-23 before regeneration); they remain alternative val draws and may
    overlap draw 101, but must not overlap the balanced test draw, which future
    refit-per-draw stability checks would otherwise contaminate."""
    _w, langs, _m = _load_model_data()
    n_lang = len(langs)
    y = _load_y()
    pools = _per_language_pool(y, n_lang)
    test_path = os.path.join(out_dir, f"val_lines_seed{TEST_SEED}.npy")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"{test_path} missing; run build_test_draw first")
    excl = np.zeros(TOTAL_LINES, bool)
    excl[np.load(test_path)] = True
    manifest = _load_manifest(out_dir)
    for seed in SEEDS[1:]:
        lines, n_with, n_shrunk = _draw_excluding(pools, excl, seed, k_rule="pool")
        np.save(os.path.join(out_dir, f"val_lines_seed{seed}.npy"), lines)
        manifest["draws"][str(seed)] = {
            "n_lines": int(len(lines)), "languages_with_val": n_with,
            "regenerated": "2026-07-23 excluding the seed-201 test draw",
            "languages_with_reduced_k": n_shrunk}
        print(f"seed {seed} regenerated: {len(lines):,} lines, {n_with}/{n_lang} "
              f"languages, {n_shrunk} languages at reduced k due to the exclusion")
    _save_manifest(out_dir, manifest)


def extract_val_texts(seed: int = SEEDS[0], cache_path: str = TEXT_CACHE) -> str:
    """One streaming pass over the test file collecting the draw's texts; labels are
    validated against the y_true memmap line by line."""
    _w, langs, lang_to_idx = _load_model_data()
    y = _load_y()
    val_lines = np.load(os.path.join(OUT_DIR, f"val_lines_seed{seed}.npy"))
    want = set(val_lines.tolist())
    texts, labels, idxs = [], [], []
    with open(TEST_FILE) as fh:
        for i in range(TOTAL_LINES):
            line = fh.readline()
            if i not in want:
                continue
            label, text = _parse_line(line)
            if lang_to_idx.get(label) != int(y[i]):
                raise RuntimeError(f"line {i}: label {label!r} does not match y_true "
                                   f"{int(y[i])}")
            texts.append(text)
            labels.append(label)
            idxs.append(i)
    if len(texts) != len(val_lines):
        raise RuntimeError(f"collected {len(texts)} texts for {len(val_lines)} lines")
    with open(cache_path + ".tmp", "wb") as f:
        pickle.dump({"seed": seed, "line_idx": np.array(idxs), "y_true": labels,
                     "texts": texts}, f)
    os.replace(cache_path + ".tmp", cache_path)
    print(f"cached {len(texts):,} val texts to {cache_path}")
    return cache_path


def rebaseline_report(out_dir: str = "outputs") -> str:
    """Score the four SAVED full-test prediction sets under the new protocol: guard
    verdicts on the balanced val (what selection WOULD have decided), and final
    stratified deltas on pool-minus-val. Uses only the saved memmaps; no scoring."""
    import pandas as pd
    _w, langs, _m = _load_model_data()
    n_lang = len(langs)
    train_counts = _load_train_counts()
    N = np.array([train_counts.get(l, 0) for l in langs], dtype=np.float64)
    y = _load_y()
    diag = pd.read_csv(DIAG_CSV)
    cat = dict(zip(diag["lang"], diag["category"]))
    lang_flags = {
        "overall": np.ones(n_lang, bool),
        "tail": N < 1_000,
        "magnets": np.array([cat.get(l) == "flat_magnet" for l in langs]),
        "twins": np.array([cat.get(l) == "twin" for l in langs]),
        "head": N >= 18_000,
    }
    val_lines = np.load(os.path.join(OUT_DIR, f"val_lines_seed{SEEDS[0]}.npy"))
    val_mask = np.zeros(TOTAL_LINES, bool)
    val_mask[val_lines] = True
    kept = y >= 0
    test_mask = kept & ~val_mask

    preds = {}
    for fn in SAVED_PREDS:
        arr = np.asarray(np.lib.format.open_memmap(os.path.join(SCRATCH_DIR, fn),
                                                   mode="r"))
        if int((arr[kept] == -3).sum()) != 0:
            raise RuntimeError(f"{fn} has unscored kept lines")
        preds[fn.removeprefix("pred_").removesuffix(".npy")] = arr

    def strat_rows(mask):
        ym = y[mask]
        rows = {}
        for name, p in preds.items():
            pm = p[mask]
            rows[name] = {st: compute_metrics(ym[flags[ym]], pm[flags[ym]])["macro_f1"]
                          for st, flags in [(s, lang_flags[s]) for s in lang_flags]}
        return rows

    val_rows = strat_rows(val_mask)
    test_rows = strat_rows(test_mask)

    lines = ["# Balanced-split re-baseline of the saved full-test configurations\n",
             f"Balanced val: draw seed {SEEDS[0]}, {int(val_mask.sum()):,} lines; new "
             f"test = pool minus val ({int(test_mask.sum()):,} lines). No new scoring: "
             f"all numbers from the saved Exp 16 / floor-21 prediction memmaps.",
             f"Guard: val overall must improve and no stratum "
             f"({'/'.join(GUARD_STRATA)}) may drop more than {GUARD_TOL}.\n",
             "## Balanced-val stratified macro-F1 (selection view) and guard verdicts\n",
             "| config | overall | tail | magnets | twins | head | guard |",
             "|---|---|---|---|---|---|---|"]
    base_v = val_rows["baseline"]
    for name, r in val_rows.items():
        if name == "baseline":
            verdict = ""
        else:
            verdict = "PASS" if passes_guard(r, base_v) else "FAIL"
        lines.append("| " + name + " | " +
                     " | ".join(f"{r[st]:.4f}" for st in
                                ["overall", "tail", "magnets", "twins", "head"]) +
                     f" | {verdict} |")
    lines.append("\n## New-test stratified macro-F1 (final view)\n")
    lines.append("| config | overall | tail | magnets | twins | head |")
    lines.append("|---|---|---|---|---|---|")
    for name, r in test_rows.items():
        lines.append("| " + name + " | " +
                     " | ".join(f"{r[st]:.4f}" for st in
                                ["overall", "tail", "magnets", "twins", "head"]) + " |")
    out_path = os.path.join(out_dir, "tables", "balanced_split_rebaseline.md")
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))
    print(f"\nWrote {out_path}")
    return out_path


if __name__ == "__main__":
    # Deliberately not a pipeline: build_draws() would resample 102-105 WITHOUT the
    # seed-201 exclusion and silently undo rebuild_stability_draws (review flag,
    # 2026-07-23). Call the stage you need explicitly:
    #   build_draws / extract_val_texts / rebaseline_report   (initial protocol)
    #   build_test_draw -> rebuild_stability_draws            (adoption-rule protocol)
    raise SystemExit("balanced_split has no default pipeline; call a specific "
                     "function (see the __main__ comment).")
