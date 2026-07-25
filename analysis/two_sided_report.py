"""Two-sided selection/confirmation report (plan B1).

Reporting side of the precision-primary adoption rule (EXPERIMENTAL_SETUP.md
"Precision-primary adoption rule"). For each configuration with a saved full-pool
prediction memmap this shows, side by side:
1. the balanced-val (draw 101) within-stratum macro-F1 row (selection view);
2. the veto-instrument view: global per-language F1/precision/recall for tail and
   magnets on the pool minus the SELECTION draw (101) and the HEADLINE draw (201);
3. the `passes_two_sided` verdict with explicit failure reasons.

Veto-instrument definition (amended 2026-07-23, first run of this module): the first
formulation excluded the union of all six balanced draws, which removes nearly every
true tail line (six half-draws over a 66-line tail pool leave ~1 line; measured veto
tail recall 0.22), so per-language F1 was recall-broken exactly where the veto needs
it. The veto now excludes only the draws with tuning or headline roles. The stability
draws 102-105 have no consumers; if a future candidate is FIT on them (refit-per-draw
checks), its veto must additionally exclude the draws it was fit on: this conditional
is part of the rule, recorded in EXPERIMENTAL_SETUP.md.

Ranking among rule-eligible configurations uses the selection instrument
(balanced-val overall macro-F1), consistent with the guard's original selection
principle; the veto is pass/fail only and the headline (balanced test draw, seed 201)
is reported for the baseline and the selected configuration only.

Wiring gate: before any exclusion is applied, global per-language F1 is recomputed on
the FULL kept pool and must match outputs/diagnostic/full_test_per_lang_prf.csv
(the Exp 24 artifact) exactly for every configuration; abort otherwise.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from analysis.config import TOTAL_LINES
from analysis.metrics import compute_metrics
from analysis.metric_decomposition import _per_lang_stats
from analysis.balanced_split import OUT_DIR as DRAW_DIR, SEEDS, TEST_SEED
from analysis.full_test_eval import SCRATCH_DIR
from analysis.hierarchical_pool import (passes_two_sided, passes_uniform,
                                        GUARD_STRATA, GUARD_TOL,
                                        TAIL_RECALL_TOL, PREC_TOL,
                                        LANG_COLLAPSE_BOUND, MIN_COLLAPSE_SUPPORT,
                                        MAX_LANG_COLLAPSE_OUTLIERS, NEAR_TIE_BAND)

PRF_CSV = "outputs/diagnostic/full_test_per_lang_prf.csv"
OUT_MD = "outputs/tables/two_sided_selection.md"
CONFIGS = ["baseline", "freq_prior", "learned_bias", "floor21", "margin_q5",
           "margin_q5_head", "gt_min", "gt_margin", "gt_margin_all",
           "gt_margin_all_100k", "gt_margin_adaptive"]
# Configs with columns in the Exp 24 PRF CSV; the wiring gate covers exactly these.
# margin_q5 postdates Exp 24 (its memmap is built by full_test_margin.py) and is
# instead anchored by that script's own gates (bit-identity to baseline off the
# gated lines, top-1 agreement, tau provenance).
GATED_CONFIGS = ["baseline", "freq_prior", "learned_bias", "floor21"]
STRATA_ORDER = ["overall", "tail", "magnets", "twins", "head"]
# Draw seeds each configuration was FIT on. The four Exp 16/20 configs predate the
# balanced protocol; margin_q5's tau values are fit on TRAINING data only (Exp 26),
# so none was fit on any draw. Any future candidate (e.g. a refit-per-draw bias)
# MUST declare its fit draws here; run() refuses to score a config whose fit draws
# are not excluded from the veto (enforced conditional, 2026-07-23 delta review
# flag (a)).
CONFIG_FIT_DRAWS = {c: [] for c in CONFIGS}


def _load_draw(seed: int) -> np.ndarray:
    path = os.path.join(DRAW_DIR, f"val_lines_seed{seed}.npy")
    if not os.path.exists(path):
        raise FileNotFoundError(f"draw file missing: {path}")
    return np.load(path)


def _within_stratum_row(y, pred, mask, lang_flags):
    ym, pm = y[mask], pred[mask]
    return {st: compute_metrics(ym[flags[ym]], pm[flags[ym]])["macro_f1"]
            for st, flags in lang_flags.items()}


def run(out_md: str = OUT_MD) -> str:
    prf = pd.read_csv(PRF_CSV)
    langs = prf.lang.tolist()
    n_lang = len(langs)
    N = prf.N.values
    category = prf.category.values
    lang_flags = {
        "overall": np.ones(n_lang, bool),
        "tail": N < 1_000,
        "magnets": category == "flat_magnet",
        "twins": category == "twin",
        "head": N >= 18_000,
    }
    strata_masks = {"tail": lang_flags["tail"], "magnets": lang_flags["magnets"]}

    y = np.asarray(np.lib.format.open_memmap(
        os.path.join(SCRATCH_DIR, "y_true.npy"), mode="r"))
    if y.shape != (TOTAL_LINES,):
        raise RuntimeError(f"y_true memmap has shape {y.shape}")
    kept = y >= 0

    preds = {}
    for c in CONFIGS:
        p = os.path.join(SCRATCH_DIR, f"pred_{c}.npy")
        if not os.path.exists(p):
            raise FileNotFoundError(f"prediction memmap missing: {p}")
        preds[c] = np.asarray(np.lib.format.open_memmap(p, mode="r"))

    # --- wiring gate: full-kept global per-language F1 must equal the Exp 24 CSV ---
    yk = y[kept]
    for c in GATED_CONFIGS:
        stats = _per_lang_stats(preds[c][kept], yk, n_lang)
        for col, idx in [("f1", 2), ("prec", 0), ("rec", 1)]:
            diff = np.abs(stats[idx] - prf[f"{col}_{c}"].values).max()
            if diff > 1e-9:
                raise RuntimeError(f"wiring gate failed: {col}_{c} max |diff| "
                                   f"{diff:.2e} vs {PRF_CSV}")
    print("Wiring gate passed: full-pool per-language stats reproduce the Exp 24 CSV.")

    # --- draw masks ---
    val101 = _load_draw(SEEDS[0])
    test201 = _load_draw(TEST_SEED)
    if np.intersect1d(val101, test201).size:
        raise RuntimeError("test draw overlaps the working val draw")
    excl = np.zeros(TOTAL_LINES, bool)
    excl[val101] = True
    excl[test201] = True
    val_mask = np.zeros(TOTAL_LINES, bool)
    val_mask[val101] = True
    test_mask = np.zeros(TOTAL_LINES, bool)
    test_mask[test201] = True
    veto_mask = kept & ~excl
    if not (veto_mask.sum() < kept.sum() and (kept & excl).sum() == excl.sum()):
        raise RuntimeError("draw lines fall outside the kept pool")
    veto_excluded_seeds = {SEEDS[0], TEST_SEED}
    for c, fit_seeds in CONFIG_FIT_DRAWS.items():
        not_excluded = [s for s in fit_seeds if s not in veto_excluded_seeds]
        if not_excluded:
            raise RuntimeError(
                f"config {c} declares fit draws {not_excluded} that are not excluded "
                "from the veto instrument; extend the exclusion for this candidate "
                "before scoring it (see the CONFIG_FIT_DRAWS comment)")
    # veto support transparency: the collapse clause (C) is judged per language, so
    # the report must show how much true support each stratum retains here
    veto_sup = np.bincount(y[veto_mask], minlength=n_lang)
    tail_sup = veto_sup[N < 1_000]
    if int(np.median(tail_sup)) < 10:
        raise RuntimeError(f"veto instrument retains median {int(np.median(tail_sup))} "
                           "true lines per tail language; instrument definition broken")

    # --- the three views ---
    yv = y[veto_mask]
    veto_stats = {c: _per_lang_stats(preds[c][veto_mask], yv, n_lang)
                  for c in CONFIGS}
    val_rows = {c: _within_stratum_row(y, preds[c], val_mask, lang_flags)
                for c in CONFIGS}

    verdicts = {}
    for c in CONFIGS[1:]:
        verdicts[c] = passes_two_sided(val_rows[c], val_rows["baseline"],
                                       veto_stats[c][2], veto_stats["baseline"][2],
                                       strata_masks, veto_sup)
    eligible = [c for c in CONFIGS[1:] if verdicts[c][0]]
    selected = (max(eligible, key=lambda c: val_rows[c]["overall"])
                if eligible else "baseline")
    co_selected = ([c for c in eligible
                    if val_rows[selected]["overall"] - val_rows[c]["overall"]
                    <= NEAR_TIE_BAND]
                   if eligible else ["baseline"])
    flagged = {c: v[2] for c, v in verdicts.items() if v[0] and v[2]}

    L = [
        "# Two-sided selection report (precision-primary adoption rule)\n",
        f"Instruments: selection = balanced-val draw {SEEDS[0]} "
        f"({int(val_mask.sum()):,} lines); veto = pool minus the selection and "
        f"headline draws ({int(veto_mask.sum()):,} lines, from the saved full-pool "
        f"memmaps; retains median {int(np.median(tail_sup))} true lines per tail "
        f"language, minimum {int(tail_sup.min())}); "
        f"headline = balanced test draw seed {TEST_SEED} "
        f"({int(test_mask.sum()):,} lines), reported for baseline and the selected "
        "configuration only.",
        f"Rule constants: GUARD_TOL={GUARD_TOL}, TAIL_RECALL_TOL={TAIL_RECALL_TOL} "
        f"(tail and magnets, symmetric widening), PREC_TOL={PREC_TOL}, per-language "
        f"collapse bound {LANG_COLLAPSE_BOUND}. Ranking among eligible "
        "configurations: highest balanced-val overall.\n",
        "## Selection view: balanced-val within-stratum macro-F1\n",
        "| config | " + " | ".join(STRATA_ORDER) + " |",
        "|" + "---|" * (len(STRATA_ORDER) + 1),
    ]
    for c in CONFIGS:
        L.append(f"| {c} | " + " | ".join(f"{val_rows[c][st]:.4f}"
                                          for st in STRATA_ORDER) + " |")

    L += ["\n## Veto view: global per-language stats, pool minus the selection and "
          "headline draws\n",
          "Levels here are not comparable to the full-pool Exp 24 numbers: the "
          "excluded draws hold roughly half of each tail language's true lines while "
          "every false positive remains, so absolute precision/F1 sit lower. The rule "
          "uses gains and drops, which are unaffected.\n",
          "| config | overall F1 | tail F1 | tail prec | tail rec | magnets F1 | "
          "magnets prec | magnets rec | FPs into tail |",
          "|" + "---|" * 9]
    for c in CONFIGS:
        prec, rec, f1, tp, fp, fn = veto_stats[c]
        t, m = lang_flags["tail"], lang_flags["magnets"]
        L.append(f"| {c} | {f1.mean():.4f} | {f1[t].mean():.4f} | "
                 f"{prec[t].mean():.4f} | {rec[t].mean():.4f} | {f1[m].mean():.4f} | "
                 f"{prec[m].mean():.4f} | {rec[m].mean():.4f} | "
                 f"{fp[t].sum():,.0f} |")

    L += ["\n## Verdicts\n",
          f"Collapse clause: more than {MAX_LANG_COLLAPSE_OUTLIERS} supported "
          "outliers rejects (class-level pattern); one or two flag a required "
          "dig-in without blocking (user decision 2026-07-24).\n"]
    for c in CONFIGS[1:]:
        ok, reasons, outliers = verdicts[c]
        if ok and outliers:
            names = ", ".join(f"{langs[i]}" for i in outliers)
            L.append(f"- **{c}: ELIGIBLE, flagged** (outlier collapse(s) requiring "
                     f"dig-in: {names})")
        elif ok:
            L.append(f"- **{c}: ELIGIBLE**")
        else:
            L.append(f"- **{c}: REJECTED** ({'; '.join(reasons)})")
    if len(co_selected) > 1:
        L.append(f"\nCarried-forward set (all eligible within {NEAR_TIE_BAND} of "
                 f"the top balanced-val overall; user decision 2026-07-25: near-tied "
                 f"methods are kept and explored, not narrowed to one): **"
                 + ", ".join(co_selected) + "**. Top-ranked: **" + selected + "**. "
                 "Balanced-test rows remain restricted to the top-ranked candidate "
                 "per track to preserve the confirmation discipline.")
    else:
        L.append(f"\nSelected configuration: **{selected}**"
                 + ("" if selected == "baseline"
                    else " (highest balanced-val overall among eligible)"))

    # per-language drops near or beyond the collapse bound, for the record
    base_f1 = veto_stats["baseline"][2]
    for c in CONFIGS[1:]:
        drops = base_f1 - veto_stats[c][2]
        idx = np.argsort(-drops)[:5]
        worst = [(langs[i], drops[i], int(veto_sup[i])) for i in idx
                 if drops[i] > 0.05]
        if worst:
            L.append(f"\nLargest per-language global-F1 drops under {c} "
                     "(with veto true-line support): "
                     + ", ".join(f"{l} -{d:.3f} (n={s})" for l, d, s in worst))
        sub = np.where((drops > LANG_COLLAPSE_BOUND)
                       & (veto_sup < MIN_COLLAPSE_SUPPORT))[0]
        if len(sub):
            L.append(f"Informational, exempt from clause (C) under {c} (support < "
                     f"{MIN_COLLAPSE_SUPPORT}): "
                     + ", ".join(f"{langs[i]} -{drops[i]:.3f} (n={int(veto_sup[i])})"
                                 for i in sub))

    # --- uniform-prior track (added 2026-07-24, gating reconsideration) ---
    uni_pass = [c for c in CONFIGS[1:] if passes_uniform(val_rows[c],
                                                         val_rows["baseline"])]
    uni_selected = (max(uni_pass, key=lambda c: val_rows[c]["overall"])
                    if uni_pass else "baseline")
    L += ["\n## Uniform-prior track\n",
          f"Track passers on balanced val (`passes_uniform`): "
          + (", ".join(uni_pass) if uni_pass else "none")
          + f". Track-selected: **{uni_selected}**."]
    if uni_selected != "baseline":
        yt = y[test_mask]
        t_base = _per_lang_stats(preds["baseline"][test_mask], yt, n_lang)
        t_cand = _per_lang_stats(preds[uni_selected][test_mask], yt, n_lang)
        t_sup = np.bincount(yt, minlength=n_lang)
        t_drops = t_base[2] - t_cand[2]
        judged = t_sup >= MIN_COLLAPSE_SUPPORT
        col_idx = np.where(judged & (t_drops > LANG_COLLAPSE_BOUND))[0]
        if len(col_idx) > MAX_LANG_COLLAPSE_OUTLIERS:
            worst_i = np.where(judged)[0][np.argmax(t_drops[judged])]
            L.append(f"Balanced-test collapse check: FAIL, {len(col_idx)} "
                     f"language(s) with support >= {MIN_COLLAPSE_SUPPORT} lose more "
                     f"than {LANG_COLLAPSE_BOUND} (worst {langs[worst_i]} "
                     f"-{t_drops[worst_i]:.3f}); class-level pattern, track "
                     "adoption blocked.")
        elif len(col_idx):
            names = ", ".join(f"{langs[i]} -{t_drops[i]:.3f} (n={int(t_sup[i])})"
                              for i in col_idx)
            L.append(f"Balanced-test collapse check: PASS with flagged outlier(s) "
                     f"requiring dig-in: {names}. **{uni_selected} is the "
                     "uniform-prior track champion, flagged.**")
        else:
            L.append(f"Balanced-test collapse check: PASS (no language with "
                     f"support >= {MIN_COLLAPSE_SUPPORT} loses more than "
                     f"{LANG_COLLAPSE_BOUND}). **{uni_selected} is the uniform-prior "
                     "track champion.**")
        sub = np.where((t_drops > LANG_COLLAPSE_BOUND)
                       & (t_sup < MIN_COLLAPSE_SUPPORT))[0]
        if len(sub):
            L.append(f"Informational, below the support floor on the test draw: "
                     + ", ".join(f"{langs[i]} -{t_drops[i]:.3f} (n={int(t_sup[i])})"
                                 for i in sub))

    L += ["\n## Headline: balanced test draw, within-stratum macro-F1\n",
          "Rows for the baseline and each track-selected configuration only.\n",
          "| config | " + " | ".join(STRATA_ORDER) + " |",
          "|" + "---|" * (len(STRATA_ORDER) + 1)]
    show = ["baseline"] + [c for c in dict.fromkeys([selected, uni_selected])
                           if c != "baseline"]
    for c in show:
        row = _within_stratum_row(y, preds[c], test_mask, lang_flags)
        L.append(f"| {c} | " + " | ".join(f"{row[st]:.4f}"
                                          for st in STRATA_ORDER) + " |")

    os.makedirs(os.path.dirname(out_md), exist_ok=True)
    with open(out_md, "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\nWrote {out_md}")
    return out_md


if __name__ == "__main__":
    run()
