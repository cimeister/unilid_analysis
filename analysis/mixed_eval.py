"""Judge-part evaluation for the per-language combined-method program.

Pre-registration: EXPERIMENTS_PLAN.md, "Plan: per-language combined method",
2026-07-29 amendments. Evaluates candidate configurations on the judge part
of the seed-301 split (analysis/combined_evidence.py derives and records
that split; this script only reads it), the confirmation instrument for
candidates that came to attention through the derivation part. Also reports
the balanced-val draw-101 selection view and both track verdicts
(`passes_two_sided`, `passes_uniform`), and a paired bootstrap of each
requested configuration against the two fixed anchors (gt_margin_adaptive,
the configuration to beat; floor21_gate, one of the two solo-gate reference
builds) on the judge part.

This script writes its own outputs (a judge-part per-language F1 CSV and a
report) and never writes to outputs/tables/two_sided_selection.md, the
recorded selection-and-veto report for ordinary candidates whose parameters
are not fit on remainder data (its final section is the balanced-test
headline, not the whole report).
"""
from __future__ import annotations

import hashlib
import os
import time

import numpy as np
import pandas as pd

from analysis.config import TOTAL_LINES
from analysis.metrics import compute_metrics
from analysis.metric_decomposition import _per_lang_stats
from analysis.margin_diagnostic import PRF_CSV
from analysis.balanced_split import OUT_DIR as DRAW_DIR, SEEDS, TEST_SEED
from analysis.full_test_eval import SCRATCH_DIR
from analysis.carried_set_comparison import (CARRIED, EXPECTED_REMAINDER,
                                             OUT_CSV as CARRIED_CSV)
from analysis.hierarchical_pool import (passes_two_sided, passes_uniform,
                                        MIN_COLLAPSE_SUPPORT, LANG_COLLAPSE_BOUND,
                                        MAX_LANG_COLLAPSE_OUTLIERS, PREC_TOL,
                                        GUARD_TOL, TAIL_RECALL_TOL)
from analysis.combined_evidence import (RULE_SPLIT_SEED, RULE_SPLIT_FRACTION,
                                        EXPECTED_DERIVATION, EXPECTED_JUDGE,
                                        SPLIT_PATH, BOOT_B, BOOT_SEED,
                                        GT_MIN_RECORDED_TAIL_FP)

# Fixed comparator set (plan step 5): every requested configuration is judged
# against these four regardless of which configs are requested.
COMPARATORS = ("baseline", "floor21", "gt_min", "gt_margin_adaptive")

# The two fixed anchors for the judge-part paired bootstrap (plan step 9):
# gt_margin_adaptive is the configuration to beat (the primary quantity's
# leader on the full remainder, EXPERIMENTS_RESULTS.md); floor21_gate is one
# of the two solo-gate reference builds and a default requested config here.
# Both need a judge-part F1 vector even on a call whose `configs` excludes
# one of them (e.g. a later configs=("mixed",) call), so both always get a
# memmap loaded regardless of what is requested.
ANCHORS = ("gt_margin_adaptive", "floor21_gate")

# Reference points recorded on the FULL held-out remainder, not the judge
# part: outputs/tables/two_sided_selection.md, "Veto view" table (read
# 2026-07-30). Used only for the non-comparable-levels note in the FP table.
# baseline's 22,404 is local to that table (no other module records it);
# gt_min's is imported from analysis.combined_evidence, which anchors it to
# its own wiring gate 2 against the same recorded table.
RECORDED_REMAINDER_FP_TAIL = {"baseline": 22_404, "gt_min": GT_MIN_RECORDED_TAIL_FP}


def _load_draw(seed: int) -> np.ndarray:
    """Mirrors analysis.combined_evidence._load_draw /
    analysis.two_sided_report._load_draw."""
    path = os.path.join(DRAW_DIR, f"val_lines_seed{seed}.npy")
    if not os.path.exists(path):
        raise FileNotFoundError(f"draw file missing: {path}")
    return np.load(path)


def _within_stratum_row(y, pred, mask, lang_flags):
    """Mirrors analysis.two_sided_report._within_stratum_row exactly (that
    module's lines 74-77): within-stratum macro-F1 restricted to lines whose
    TRUE label falls in the stratum, for every stratum in lang_flags."""
    ym, pm = y[mask], pred[mask]
    return {st: compute_metrics(ym[flags[ym]], pm[flags[ym]])["macro_f1"]
            for st, flags in lang_flags.items()}


def run(configs=("floor21_gate", "unmod_gate"), out_md: str = None,
        out_csv: str = None) -> str:
    configs = tuple(dict.fromkeys(configs))
    # Default paths are derived from the requested config set so that calls
    # with different `configs` do not overwrite each other's outputs; pass
    # out_md/out_csv explicitly to override.
    if out_md is None:
        out_md = f"outputs/tables/mixed_eval_{'_'.join(configs)}.md"
    if out_csv is None:
        out_csv = f"outputs/diagnostic/mixed_eval_judge_f1_{'_'.join(configs)}.csv"

    # --- step 1: languages, N, category from the Exp 24 PRF CSV; check the
    # Exp 38 carried CSV has the same language order ---
    prf = pd.read_csv(PRF_CSV)
    langs = prf.lang.tolist()
    n_lang = len(langs)
    N = prf.N.values
    category = prf.category.values

    carried_df = pd.read_csv(CARRIED_CSV)
    if carried_df.lang.tolist() != langs:
        raise RuntimeError(f"{CARRIED_CSV} language order does not match {PRF_CSV}")

    # --- step 2: y_true, draws 101 and 201, remainder mask ---
    y = np.asarray(np.lib.format.open_memmap(
        os.path.join(SCRATCH_DIR, "y_true.npy"), mode="r"))
    if y.shape != (TOTAL_LINES,):
        raise RuntimeError(f"y_true memmap has shape {y.shape}, expected "
                           f"({TOTAL_LINES},)")
    kept = y >= 0

    val101 = _load_draw(SEEDS[0])
    test201 = _load_draw(TEST_SEED)
    if np.intersect1d(val101, test201).size:
        raise RuntimeError("test draw overlaps the working val draw")
    excl = np.zeros(TOTAL_LINES, bool)
    excl[val101] = True
    excl[test201] = True
    if int((kept & excl).sum()) != int(excl.sum()):
        raise RuntimeError("draw lines fall outside the kept pool")
    remainder_mask = kept & ~excl
    if int(remainder_mask.sum()) != EXPECTED_REMAINDER:
        raise RuntimeError(f"remainder {int(remainder_mask.sum()):,} != "
                           f"{EXPECTED_REMAINDER:,}")
    yr = y[remainder_mask]

    # --- step 3: rebuild the seed-301 split exactly as combined_evidence
    # does; size gates; the split record must already exist (this script
    # never creates it, that is combined_evidence's job) ---
    remainder_idx = np.where(remainder_mask)[0]
    u = np.random.default_rng(RULE_SPLIT_SEED).random(remainder_idx.size)
    derive_idx = remainder_idx[u < RULE_SPLIT_FRACTION]
    judge_idx = remainder_idx[u >= RULE_SPLIT_FRACTION]
    if len(derive_idx) != EXPECTED_DERIVATION:
        raise RuntimeError(f"derivation part {len(derive_idx):,} != "
                           f"EXPECTED_DERIVATION {EXPECTED_DERIVATION:,}")
    if len(judge_idx) != EXPECTED_JUDGE:
        raise RuntimeError(f"judge part {len(judge_idx):,} != "
                           f"EXPECTED_JUDGE {EXPECTED_JUDGE:,}")
    if len(derive_idx) + len(judge_idx) != EXPECTED_REMAINDER:
        raise RuntimeError(f"split sizes {len(derive_idx):,} + "
                           f"{len(judge_idx):,} != remainder "
                           f"{EXPECTED_REMAINDER:,}")

    if not os.path.exists(SPLIT_PATH):
        raise FileNotFoundError(
            f"split record missing at {SPLIT_PATH}; this script never "
            "creates the seed-301 split, only analysis.combined_evidence "
            "does (run analysis.combined_evidence.run() first)")
    with np.load(SPLIT_PATH) as stored:
        if not (np.array_equal(stored["derive_idx"], derive_idx)
                and np.array_equal(stored["judge_idx"], judge_idx)):
            raise RuntimeError(f"split stored at {SPLIT_PATH} does not match "
                               "the recomputed seed-301 split")
    judge_sha = hashlib.sha256(judge_idx.tobytes()).hexdigest()

    # --- step 4/5: memmaps + provenance for every config this run needs:
    # the seven Exp 38 CARRIED configs (wiring gate only), the union of the
    # requested configs and the fixed comparator set (judge-part evaluation),
    # and the two bootstrap anchors (needed even if not otherwise requested) ---
    eval_configs = list(dict.fromkeys(list(configs) + list(COMPARATORS)))
    boot_configs = list(dict.fromkeys(eval_configs + list(ANCHORS)))
    all_pred_configs = list(dict.fromkeys(list(CARRIED) + boot_configs))

    preds = {}
    prov = {}
    for c in all_pred_configs:
        p = os.path.join(SCRATCH_DIR, f"pred_{c}.npy")
        if not os.path.exists(p):
            raise FileNotFoundError(f"prediction memmap missing: {p}")
        preds[c] = np.asarray(np.lib.format.open_memmap(p, mode="r"))
        st = os.stat(p)
        with open(p, "rb") as fh:
            file_sha = hashlib.sha256(fh.read()).hexdigest()
        prov[c] = {"path": p, "bytes": st.st_size, "sha256": file_sha,
                   "mtime": time.strftime("%Y-%m-%d %H:%M:%S",
                                          time.localtime(st.st_mtime))}

    # --- wiring gate: recompute per-language F1 on the FULL remainder for
    # the seven CARRIED configs and match carried_set_per_lang_f1.csv to
    # 1e-9 (same gate as analysis.combined_evidence's wiring gate 1). Proves
    # the code path before any judge-part number is trusted. ---
    gate_diffs = {}
    for c in CARRIED:
        f1_here = _per_lang_stats(preds[c][remainder_mask], yr, n_lang)[2]
        diff = float(np.abs(f1_here - carried_df[f"f1_{c}"].values).max())
        gate_diffs[c] = diff
        if diff > 1e-9:
            raise RuntimeError(f"wiring gate failed for {c}: max |diff| "
                               f"{diff:.2e} vs {CARRIED_CSV}")
    print("Wiring gate passed: remainder per-language F1 reproduces "
          f"{CARRIED_CSV} for all {len(CARRIED)} Exp 38 configs.")

    # --- step 6: judge-part statistics ---
    yj = y[judge_idx]
    support_judge = np.bincount(yj, minlength=n_lang)
    tail_mask = N < 1_000
    magnet_mask = category == "flat_magnet"

    # Pre-registered veto-instrument support gate (mirrors
    # analysis.two_sided_report.run, lines 144-148, and EXPERIMENTAL_SETUP.md,
    # "Precision-primary adoption rule", line 296): the judge part stands in
    # here for that module's veto instrument, so it must retain at least as
    # much tail support or per-language F1 there cannot be trusted.
    tail_support_judge = support_judge[tail_mask]
    median_tail_support_judge = float(np.median(tail_support_judge))
    min_tail_support_judge = int(tail_support_judge.min())
    if median_tail_support_judge < MIN_COLLAPSE_SUPPORT:
        raise RuntimeError(
            f"judge-part veto instrument retains median "
            f"{median_tail_support_judge:.1f} true lines per tail language, "
            f"below MIN_COLLAPSE_SUPPORT ({MIN_COLLAPSE_SUPPORT}); "
            "instrument definition broken")

    stats_judge = {c: _per_lang_stats(preds[c][judge_idx], yj, n_lang)
                   for c in boot_configs}
    f1_judge = {c: stats_judge[c][2] for c in boot_configs}

    groups = {
        f"all {n_lang:,}": np.ones(n_lang, bool),
        "tail (N<1k)": N < 1_000,
        "lowmid (1k-18k)": (N >= 1_000) & (N < 18_000),
        "head (N>=18k)": N >= 18_000,
        "flat_magnet": category == "flat_magnet",
        "twin": category == "twin",
    }

    fp_tail_judge = {c: int(stats_judge[c][4][tail_mask].sum()) for c in eval_configs}
    fp_magnet_judge = {c: int(stats_judge[c][4][magnet_mask].sum()) for c in eval_configs}

    # --- step 7: clause (B) / (C) for each REQUESTED config against
    # baseline, on the judge part ---
    base_f1_j = f1_judge["baseline"]
    clause_bc = {}
    for c in configs:
        cand_f1 = f1_judge[c]
        gain_overall = float(cand_f1.mean() - base_f1_j.mean())
        gain_tail = float(cand_f1[tail_mask].mean() - base_f1_j[tail_mask].mean())
        gain_magnet = float(cand_f1[magnet_mask].mean() - base_f1_j[magnet_mask].mean())
        clauseB_pass = (gain_overall >= -GUARD_TOL and gain_tail >= -PREC_TOL
                        and gain_magnet >= -PREC_TOL)

        drops = base_f1_j - cand_f1
        judged = support_judge >= MIN_COLLAPSE_SUPPORT
        outlier_idx = np.where(judged & (drops > LANG_COLLAPSE_BOUND))[0]
        sub_idx = np.where((drops > LANG_COLLAPSE_BOUND)
                           & (support_judge < MIN_COLLAPSE_SUPPORT))[0]
        n_collapse = len(outlier_idx)
        if n_collapse == 0:
            clauseC_verdict = "clean"
        elif n_collapse <= MAX_LANG_COLLAPSE_OUTLIERS:
            clauseC_verdict = "flagged (required dig-in)"
        else:
            clauseC_verdict = "class-level fail"

        clause_bc[c] = {
            "gain_overall": gain_overall, "gain_tail": gain_tail,
            "gain_magnet": gain_magnet, "clauseB_pass": clauseB_pass,
            "outliers": [(langs[i], float(cand_f1[i]), float(base_f1_j[i]),
                         int(support_judge[i])) for i in outlier_idx],
            "n_collapse": n_collapse, "clauseC_verdict": clauseC_verdict,
            "sub_support": [(langs[i], float(cand_f1[i]), float(base_f1_j[i]),
                            int(support_judge[i])) for i in sub_idx],
        }

    # --- step 8: draw-101 selection view, mirroring
    # analysis.two_sided_report.run exactly: stratum masks and STRATA_ORDER
    # (that module's lines 86-92, 57), val_mask construction (lines 127-128),
    # val_rows via _within_stratum_row (line 154), then the passes_two_sided
    # / passes_uniform verdicts (lines 158-161, 256-257) with judge-part
    # global stats standing in for that module's veto-instrument stats ---
    lang_flags = {
        "overall": np.ones(n_lang, bool),
        "tail": tail_mask,
        "magnets": magnet_mask,
        "twins": category == "twin",
        "head": N >= 18_000,
    }
    strata_masks = {"tail": lang_flags["tail"], "magnets": lang_flags["magnets"]}
    STRATA_ORDER = ["overall", "tail", "magnets", "twins", "head"]

    val_mask = np.zeros(TOTAL_LINES, bool)
    val_mask[val101] = True

    val_rows = {c: _within_stratum_row(y, preds[c], val_mask, lang_flags)
                for c in eval_configs}

    verdicts = {}
    uniform_pass = {}
    for c in eval_configs:
        if c == "baseline":
            continue
        verdicts[c] = passes_two_sided(val_rows[c], val_rows["baseline"],
                                       f1_judge[c], f1_judge["baseline"],
                                       strata_masks, support_judge)
        uniform_pass[c] = passes_uniform(val_rows[c], val_rows["baseline"])

    # --- step 9: paired bootstrap, judge part, all languages, one
    # shared resample matrix reused across configs and both anchors ---
    rng = np.random.default_rng(BOOT_SEED)
    resample = rng.integers(0, n_lang, size=(BOOT_B, n_lang))
    boots = {c: f1_judge[c][resample].mean(axis=1) for c in boot_configs}

    bootstrap_rows = []
    for c in configs:
        for anchor in ANCHORS:
            if c == anchor:
                continue
            d = boots[c] - boots[anchor]
            point = float(f1_judge[c].mean() - f1_judge[anchor].mean())
            ci_lo, ci_hi = np.percentile(d, [2.5, 97.5])
            bootstrap_rows.append({"config": c, "anchor": anchor,
                                   "mean_diff": point, "ci_lo": float(ci_lo),
                                   "ci_hi": float(ci_hi)})

    # --- step 10: outputs ---
    out = {"lang": langs, "N": N.astype(int), "category": category,
           "support_judge": support_judge}
    for c in boot_configs:
        out[f"f1_{c}"] = f1_judge[c]
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    pd.DataFrame(out).to_csv(out_csv, index=False)

    L = [
        "# Judge-part evaluation for the per-language combined method\n",
        "Pre-registration: EXPERIMENTS_PLAN.md, \"Plan: per-language combined "
        "method\", 2026-07-29 amendments. Confirmation instrument for "
        "configurations whose treatment came to attention through the "
        "derivation part of the seed-301 split; analysis/combined_evidence.py "
        "derives and records that split, this script only reads it. "
        f"Requested configs: {', '.join(configs)}. Fixed comparator set: "
        f"{', '.join(COMPARATORS)}.\n",
        "## Gates passed\n",
    ]
    L.append(f"- y_true memmap shape: {tuple(int(x) for x in y.shape)}, matches "
             f"TOTAL_LINES ({TOTAL_LINES:,}).")
    L.append(f"- Draws {SEEDS[0]} and {TEST_SEED} loaded from {DRAW_DIR}, "
             f"disjoint ({len(val101):,} and {len(test201):,} lines, zero "
             "overlap).")
    L.append(f"- Full remainder (kept minus draws 101 and 201): "
             f"{int(remainder_mask.sum()):,} lines, matches EXPECTED_REMAINDER "
             f"({EXPECTED_REMAINDER:,}) from analysis.carried_set_comparison.")
    L.append(f"- Seed-{RULE_SPLIT_SEED} split recomputed with fraction "
             f"{RULE_SPLIT_FRACTION}: derivation part {len(derive_idx):,} "
             f"lines matches EXPECTED_DERIVATION ({EXPECTED_DERIVATION:,}); "
             f"judge part {len(judge_idx):,} lines matches EXPECTED_JUDGE "
             f"({EXPECTED_JUDGE:,}); the two sizes sum to the full remainder "
             f"{EXPECTED_REMAINDER:,}.")
    L.append(f"- Split record at {SPLIT_PATH} exists and matches the "
             "recomputed split exactly (this script never writes that file). "
             f"sha256 of judge_idx.tobytes(): {judge_sha}.")
    L.append(f"- Wiring gate, full remainder: recomputed per-language F1 for "
             f"all {len(CARRIED)} Exp 38 carried configs matches {CARRIED_CSV} "
             f"within max absolute difference {max(gate_diffs.values()):.2e} "
             "(limit 1e-9). Per-config max difference: "
             + ", ".join(f"{c} {d:.2e}" for c, d in gate_diffs.items()) + ".")
    L.append("- Veto-instrument support gate, judge part: median tail true-line "
             f"support {median_tail_support_judge:.1f}, minimum "
             f"{min_tail_support_judge}, both required to be at least "
             f"MIN_COLLAPSE_SUPPORT ({MIN_COLLAPSE_SUPPORT}) for the median.")

    L += ["\n## Input prediction memmaps\n"]
    L.append("Every number below derives from these files as they existed at "
             "run time.\n")
    L.append("| config | sha256 (first 16) | bytes | mtime |")
    L.append("|---|---|---|---|")
    for c in all_pred_configs:
        L.append(f"| {c} | {prov[c]['sha256'][:16]} | {prov[c]['bytes']:,} | "
                 f"{prov[c]['mtime']} |")
    carried_only = [c for c in CARRIED if c not in boot_configs]
    if carried_only:
        L.append("\nLoaded only for the wiring gate above (full remainder, "
                 "not judge-part evaluated in this report): "
                 + ", ".join(carried_only) + ".")
    anchor_only = [c for c in ANCHORS if c not in eval_configs]
    if anchor_only:
        L.append("Loaded only as bootstrap anchors below (judge-part F1 "
                 "computed for the contrasts, but not part of the requested "
                 "plus comparator config set): " + ", ".join(anchor_only) + ".")

    L += ["\n## Judge-part group mean F1\n"]
    L.append("Judge part of the seed-301 split "
             f"({len(judge_idx):,} lines; verified against SPLIT_PATH "
             "above). Six groups, same names and masks as "
             "analysis.combined_evidence.\n")
    L.append("| group | n langs | " + " | ".join(eval_configs) + " |")
    L.append("|" + "---|" * (len(eval_configs) + 2))
    for gname, gmask in groups.items():
        gidx = np.where(gmask)[0]
        means = " | ".join(f"{f1_judge[c][gidx].mean():.4f}" for c in eval_configs)
        L.append(f"| {gname} | {int(gmask.sum())} | {means} |")

    L += ["\n## FPs into tail labels and flat_magnet labels, judge part of "
          "the seed-301 split\n"]
    L.append("| config | FPs into tail labels | FPs into flat_magnet labels |")
    L.append("|---|---|---|")
    for c in eval_configs:
        L.append(f"| {c} | {fp_tail_judge[c]:,} | {fp_magnet_judge[c]:,} |")
    L.append("\nFor reference, the FPs-into-tail-labels numbers recorded on "
             "the FULL held-out remainder (outputs/tables/two_sided_selection.md, "
             "veto view table): baseline "
             f"{RECORDED_REMAINDER_FP_TAIL['baseline']:,}, gt_min "
             f"{RECORDED_REMAINDER_FP_TAIL['gt_min']:,}. The judge part "
             f"above is {len(judge_idx):,} of the remainder's "
             f"{EXPECTED_REMAINDER:,} lines (about 60%), so these levels are "
             "not directly comparable to the table above; the comparison of "
             "interest is each configuration against baseline on the SAME "
             "judge part.")

    L += ["\n## Clause (B) and (C), judge part of the seed-301 split, vs "
          "baseline\n"]
    L.append(f"Clause (B): overall global mean F1 on the judge part may not "
             f"drop below baseline's by more than GUARD_TOL ({GUARD_TOL}), "
             f"and tail and flat_magnet global mean F1 may not drop below "
             f"baseline's by more than PREC_TOL ({PREC_TOL}). Clause (C): "
             "languages with judge-part true "
             f"support >= MIN_COLLAPSE_SUPPORT ({MIN_COLLAPSE_SUPPORT}) whose "
             f"F1 drops more than LANG_COLLAPSE_BOUND ({LANG_COLLAPSE_BOUND}) "
             "below baseline's; verdict by outlier count: 0 clean, 1 to "
             f"MAX_LANG_COLLAPSE_OUTLIERS ({MAX_LANG_COLLAPSE_OUTLIERS}) "
             "flagged with a required dig-in, more than "
             f"{MAX_LANG_COLLAPSE_OUTLIERS} a class-level fail. Sub-support "
             "languages beyond the bound are reported informationally, never "
             "silently dropped.\n")
    for c in configs:
        row = clause_bc[c]
        L.append(f"### {c}\n")
        L.append(f"- Clause (B): overall global mean F1 gain "
                 f"{row['gain_overall']:+.4f} (must be >= -GUARD_TOL = "
                 f"{-GUARD_TOL:+.4f}), tail global mean F1 gain "
                 f"{row['gain_tail']:+.4f}, flat_magnet global mean F1 gain "
                 f"{row['gain_magnet']:+.4f} (each must be >= -PREC_TOL = "
                 f"{-PREC_TOL:+.4f}) "
                 f"({'PASS' if row['clauseB_pass'] else 'FAIL'}).")
        if row["outliers"]:
            names = "; ".join(f"{l} (F1 {f1c:.4f} vs baseline {f1b:.4f}, "
                              f"support {s})"
                              for l, f1c, f1b, s in row["outliers"])
            L.append(f"- Clause (C): {row['n_collapse']} supported "
                     f"collapse(s), verdict **{row['clauseC_verdict']}**: "
                     f"{names}")
        else:
            L.append("- Clause (C): 0 supported collapses, verdict "
                     f"**{row['clauseC_verdict']}**.")
        if row["sub_support"]:
            names = "; ".join(f"{l} (F1 {f1c:.4f} vs baseline {f1b:.4f}, "
                              f"support {s})"
                              for l, f1c, f1b, s in row["sub_support"])
            L.append(f"- Informational, below the support floor, exempt "
                     f"from clause (C): {names}")

    L += ["\n## Draw-101 selection view (balanced-val within-stratum "
          "macro-F1)\n"]
    L.append("Mirrors analysis.two_sided_report's selection view exactly "
             "(via the _within_stratum_row helper above, copied from that "
             f"module's lines 74-77): {int(val_mask.sum()):,} lines.\n")
    L.append("| config | " + " | ".join(STRATA_ORDER) + " |")
    L.append("|" + "---|" * (len(STRATA_ORDER) + 1))
    for c in eval_configs:
        L.append(f"| {c} | " + " | ".join(f"{val_rows[c][st]:.4f}"
                                          for st in STRATA_ORDER) + " |")

    L += ["\n## Track verdicts: draw-101 selection view plus judge-part "
          "veto\n"]
    L.append("passes_two_sided(val_rows[c], val_rows['baseline'], "
             "f1_judge[c], f1_judge['baseline'], strata_masks, "
             "support_judge): veto computed on the judge part of the "
             "seed-301 split (not analysis.two_sided_report's own veto "
             "instrument, which is the full remainder minus draws 101 and "
             "201). support_judge is the per-language true-line count on "
             f"the judge part ({len(judge_idx):,} lines). Clauses: (A) "
             f"balanced-val overall/twins/head bounded by GUARD_TOL "
             f"({GUARD_TOL}), tail/magnets by the symmetric widening rule "
             f"(TAIL_RECALL_TOL={TAIL_RECALL_TOL} when the veto-instrument "
             "gain for that stratum exceeds its within-stratum loss, else "
             f"GUARD_TOL); (B) judge-part veto overall bounded by GUARD_TOL, "
             f"tail/magnet global mean F1 by PREC_TOL ({PREC_TOL}); (C) the "
             "collapse clause above, computed identically inside "
             "passes_two_sided. passes_uniform(val_rows[c], "
             "val_rows['baseline']): the uniform-prior track, balanced-val "
             "only.\n")
    for c in eval_configs:
        if c == "baseline":
            continue
        ok, reasons, verdict_outliers = verdicts[c]
        if ok and verdict_outliers:
            names = ", ".join(langs[i] for i in verdict_outliers)
            L.append(f"- **{c}: two-sided ELIGIBLE, flagged** (dig-in: "
                     f"{names})")
        elif ok:
            L.append(f"- **{c}: two-sided ELIGIBLE**")
        else:
            L.append(f"- **{c}: two-sided REJECTED** ({'; '.join(reasons)})")
        L.append("  uniform-prior track (passes_uniform): "
                 f"{'PASS' if uniform_pass[c] else 'FAIL'}")

    L += ["\n## Paired bootstrap, judge part of the seed-301 split\n"]
    L.append(f"B={BOOT_B:,}, seed={BOOT_SEED}, percentile 95% interval, one "
             f"resample index matrix over all {n_lang:,} language positions "
             "(paired), shared across every contrast below. Point estimate "
             "= (config judge-part mean F1) minus (anchor judge-part mean "
             "F1); interval from the bootstrap distribution of the same "
             "difference. The self-contrast against an anchor equal to the "
             "requested config is skipped (trivially zero).\n")
    L.append("| config | anchor | mean diff | 95% CI |")
    L.append("|---|---|---|---|")
    for row in bootstrap_rows:
        L.append(f"| {row['config']} | {row['anchor']} | "
                 f"{row['mean_diff']:+.4f} | [{row['ci_lo']:+.4f}, "
                 f"{row['ci_hi']:+.4f}] |")

    os.makedirs(os.path.dirname(out_md), exist_ok=True)
    with open(out_md, "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\nWrote {out_csv} and {out_md}")
    return out_md


if __name__ == "__main__":
    run()
