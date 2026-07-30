"""Step-1b evidence base for the per-language combined method.

Pre-registration: EXPERIMENTS_PLAN.md, "Plan: per-language combined method",
with its 2026-07-29 amendments. This script builds the seed-301 rule split of
the held-out remainder, gates itself by reproducing recorded numbers from the
Exp 38 carried-set comparison (analysis/carried_set_comparison.py) and the
two-sided selection report (outputs/tables/two_sided_selection.md), and
computes the derivation-part evidence for the combined method: per-group
means, the per-group leader table, and the codomain oracle.

Judge-part predictions are never read here. The judge part of the split is
used only to record its true-line support counts, for the support report.
Judge-part evaluation of the combined method happens in the later mixed_eval
step, after the mixed matrix (analysis/mixed_matrix.py) has been scored.
"""
from __future__ import annotations

import hashlib
import os
import time

import numpy as np
import pandas as pd

from analysis.config import TOTAL_LINES
from analysis.metric_decomposition import _per_lang_stats
from analysis.margin_diagnostic import PRF_CSV
from analysis.balanced_split import OUT_DIR as DRAW_DIR, SEEDS, TEST_SEED
from analysis.full_test_eval import SCRATCH_DIR
from analysis.carried_set_comparison import (CARRIED, EXPECTED_REMAINDER,
                                             OUT_CSV as CARRIED_CSV)
from analysis.hierarchical_pool import MIN_COLLAPSE_SUPPORT

# Pre-registered 2026-07-29 (EXPERIMENTS_PLAN.md, "Plan: per-language combined
# method", amendment 1). The assignment rule is derived on a seeded 40% part
# of the held-out remainder and judged on the other 60%, replacing the
# earlier draw-101 derivation. Measured reason for the amendment: on draw 101
# the per-group leader disagrees with the held-out-remainder leader in all
# six groups (EXPERIMENTS_RESULTS.md, "Current state (2026-07-29)", point 3a).
RULE_SPLIT_SEED = 301
RULE_SPLIT_FRACTION = 0.40

# Pre-registered 2026-07-29 (amendment 2). Paired bootstrap over languages,
# used in the pre-registration to decide promotion of the combined method
# against gt_margin_adaptive on the judge part. The same constants are reused
# here for the derivation-part leader intervals, so the two analyses use one
# bootstrap convention.
BOOT_B = 10_000
BOOT_SEED = 0

# Expected split sizes, measured in the 2026-07-29 review (EXPERIMENTS_RESULTS.md,
# "Current state (2026-07-29)", point 3a: "derivation part about 18.0M lines,
# judge part about 27.0M lines"). The gates in run() abort on mismatch.
EXPECTED_DERIVATION = 18_001_573
EXPECTED_JUDGE = 27_002_441

# Scratch, deliberately not the repo: this is a derived index array (two
# arrays of tens of millions of int64 line indices), not a source artifact.
SPLIT_PATH = os.path.join(SCRATCH_DIR, "rule_split_seed301.npz")

# The combined method's treatment codomain (pre-registered 2026-07-29,
# EXPERIMENTS_PLAN.md "Treatment set (modularity-preserving only)"): the four
# solo configurations plus the two gated variants. pred_unmod_gate.npy and
# pred_floor21_gate.npy are built by analysis/solo_gates.py before this
# script runs; they are not produced here.
CODOMAIN = ["baseline", "unmod_gate", "floor21", "floor21_gate", "gt_min",
            "gt_margin_adaptive"]

# Reference configurations: the remainder of the Exp 38 carried set, excluded
# from the combined method's codomain because they need global data to fit
# for a new language (freq_prior, learned_bias) or because their margin gate
# is folded into the gated codomain variants above (margin_q5,
# margin_q5_head). Their per-language F1 is written to OUT_CSV for context
# only; the group, leader, oracle, and unique-best analysis below is
# restricted to CODOMAIN, the modular treatment set.
REFERENCE = ["freq_prior", "learned_bias", "margin_q5", "margin_q5_head"]

# gt_min's recorded veto-instrument numbers, outputs/tables/two_sided_selection.md,
# "Veto view: global per-language stats" table, gt_min row (read 2026-07-29):
# overall F1 0.9114, tail F1 0.2950, tail prec 0.2205, tail rec 0.9675,
# magnets F1 0.2533, magnets prec 0.1732, magnets rec 0.9565, FPs into tail
# 79,113. Only the tail F1 and the FP count anchor wiring gate 2 below. The
# veto mask in that report is the full kept pool minus draws 101 and 201,
# the same set as remainder_mask computed in run(), so the two numbers should
# match closely.
GT_MIN_RECORDED_TAIL_F1 = 0.2950
GT_MIN_RECORDED_TAIL_FP = 79_113
GT_MIN_GATE_TOL = 6e-5   # 4-decimal rounding tolerance, precedent metric_decomposition.py

OUT_MD = "outputs/tables/combined_evidence.md"
OUT_CSV = "outputs/diagnostic/combined_evidence_derivation_f1.csv"


def _load_draw(seed: int) -> np.ndarray:
    """Mirrors analysis.two_sided_report._load_draw."""
    path = os.path.join(DRAW_DIR, f"val_lines_seed{seed}.npy")
    if not os.path.exists(path):
        raise FileNotFoundError(f"draw file missing: {path}")
    return np.load(path)


def run(out_md: str = OUT_MD, out_csv: str = OUT_CSV) -> str:
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
    remainder_mask = kept & ~excl
    if int(remainder_mask.sum()) != EXPECTED_REMAINDER:
        raise RuntimeError(f"remainder {int(remainder_mask.sum()):,} != "
                           f"{EXPECTED_REMAINDER:,}")
    yr = y[remainder_mask]

    # --- step 3: prediction memmaps for CODOMAIN + REFERENCE plus the
    # Exp 38 gate set (CARRIED); the union is CODOMAIN + REFERENCE since
    # every CARRIED config already belongs to one of the two lists ---
    missing_from_lists = [c for c in CARRIED if c not in CODOMAIN + REFERENCE]
    if missing_from_lists:
        raise RuntimeError(f"CARRIED configs absent from CODOMAIN + REFERENCE "
                           f"(their F1 would silently miss the output CSV): "
                           f"{missing_from_lists}")
    all_pred_configs = list(dict.fromkeys(CODOMAIN + REFERENCE + CARRIED))
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

    # --- step 4: wiring gate 1, reproduce the Exp 38 carried-set F1 exactly
    # on the full remainder ---
    gate1_diffs = {}
    for c in CARRIED:
        f1_here = _per_lang_stats(preds[c][remainder_mask], yr, n_lang)[2]
        diff = float(np.abs(f1_here - carried_df[f"f1_{c}"].values).max())
        gate1_diffs[c] = diff
        if diff > 1e-9:
            raise RuntimeError(f"wiring gate 1 failed for {c}: max |diff| "
                               f"{diff:.2e} vs {CARRIED_CSV}")
    print("Wiring gate 1 passed: remainder per-language F1 reproduces "
          f"{CARRIED_CSV} for all {len(CARRIED)} Exp 38 configs.")

    # --- step 5: wiring gate 2, gt_min veto-instrument anchor on the full
    # remainder ---
    tail_mask = N < 1_000
    gm_prec, gm_rec, gm_f1, gm_tp, gm_fp, gm_fn = _per_lang_stats(
        preds["gt_min"][remainder_mask], yr, n_lang)
    gt_min_tail_f1 = float(gm_f1[tail_mask].mean())
    gt_min_tail_fp = int(gm_fp[tail_mask].sum())
    f1_diff = abs(gt_min_tail_f1 - GT_MIN_RECORDED_TAIL_F1)
    if f1_diff > GT_MIN_GATE_TOL:
        raise RuntimeError(f"wiring gate 2 failed: gt_min tail global mean F1 "
                           f"{gt_min_tail_f1:.6f} vs recorded "
                           f"{GT_MIN_RECORDED_TAIL_F1} (diff {f1_diff:.2e}, "
                           f"limit {GT_MIN_GATE_TOL:.0e})")
    if gt_min_tail_fp != GT_MIN_RECORDED_TAIL_FP:
        raise RuntimeError(f"wiring gate 2 failed: gt_min FPs into tail "
                           f"labels {gt_min_tail_fp:,} != recorded "
                           f"{GT_MIN_RECORDED_TAIL_FP:,}")
    print(f"Wiring gate 2 passed: gt_min tail global mean F1 {gt_min_tail_f1:.4f} "
          f"(recorded {GT_MIN_RECORDED_TAIL_F1}), FPs into tail labels "
          f"{gt_min_tail_fp:,} (recorded {GT_MIN_RECORDED_TAIL_FP:,}).")

    # --- step 6: seed-301 rule split of the remainder ---
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

    if os.path.exists(SPLIT_PATH):
        stored = np.load(SPLIT_PATH)
        if not (np.array_equal(stored["derive_idx"], derive_idx)
                and np.array_equal(stored["judge_idx"], judge_idx)):
            raise RuntimeError(f"split stored at {SPLIT_PATH} does not match "
                               "the recomputed seed-301 split")
        print(f"Split record at {SPLIT_PATH} matches the recomputed split.")
    else:
        os.makedirs(os.path.dirname(SPLIT_PATH), exist_ok=True)
        np.savez(SPLIT_PATH, derive_idx=derive_idx, judge_idx=judge_idx)
        print(f"Wrote the seed-301 split to {SPLIT_PATH}.")
    derive_sha = hashlib.sha256(derive_idx.tobytes()).hexdigest()

    # --- step 7: support report, true-line counts on each part ---
    support_derive = np.bincount(y[derive_idx], minlength=n_lang)
    support_judge = np.bincount(y[judge_idx], minlength=n_lang)
    magnet_mask = category == "flat_magnet"

    tail_derive = support_derive[tail_mask]
    tail_judge = support_judge[tail_mask]
    tail_derive_ge_floor = int((tail_derive >= MIN_COLLAPSE_SUPPORT).sum())
    tail_judge_ge_floor = int((tail_judge >= MIN_COLLAPSE_SUPPORT).sum())
    magnet_derive_ge_floor = int(
        (support_derive[magnet_mask] >= MIN_COLLAPSE_SUPPORT).sum())
    below_floor_idx = np.where(
        tail_mask & (support_derive < MIN_COLLAPSE_SUPPORT))[0]

    # --- step 8: derivation-part per-language F1 for all ten configs ---
    yd = y[derive_idx]
    f1s_derive = {}
    for c in CODOMAIN + REFERENCE:
        f1s_derive[c] = _per_lang_stats(preds[c][derive_idx], yd, n_lang)[2]

    groups = {
        "all 1,940": np.ones(n_lang, bool),
        "tail (N<1k)": N < 1_000,
        "lowmid (1k-18k)": (N >= 1_000) & (N < 18_000),
        "head (N>=18k)": N >= 18_000,
        "flat_magnet": category == "flat_magnet",
        "twin": category == "twin",
    }

    # --- step 9: per-group means, paired bootstrap leader intervals, and
    # fixed contrasts against gt_margin_adaptive (the configuration to beat;
    # the anchor is chosen in advance, so these contrasts are not
    # post-selection, unlike the leader-vs-runner-up interval) ---
    ANCHOR = "gt_margin_adaptive"
    rng = np.random.default_rng(BOOT_SEED)
    group_rows = []
    for gname, gmask in groups.items():
        gidx = np.where(gmask)[0]
        gsize = int(gidx.size)
        means = {c: float(f1s_derive[c][gidx].mean()) for c in CODOMAIN}
        order = sorted(CODOMAIN, key=lambda c: -means[c])
        leader, runner_up = order[0], order[1]
        resample = rng.integers(0, gsize, size=(BOOT_B, gsize))
        boots = {c: f1s_derive[c][gidx][resample].mean(axis=1) for c in CODOMAIN}
        delta_boot = boots[leader] - boots[runner_up]
        ci_lo, ci_hi = np.percentile(delta_boot, [2.5, 97.5])
        contrasts = {}
        for c in CODOMAIN:
            if c == ANCHOR:
                continue
            d = boots[c] - boots[ANCHOR]
            contrasts[c] = (means[c] - means[ANCHOR],
                            float(np.percentile(d, 2.5)),
                            float(np.percentile(d, 97.5)))
        group_rows.append({
            "group": gname, "size": gsize, "means": means,
            "leader": leader, "leader_mean": means[leader],
            "runner_up": runner_up, "runner_up_mean": means[runner_up],
            "delta": means[leader] - means[runner_up],
            "ci_lo": float(ci_lo), "ci_hi": float(ci_hi),
            "contrasts": contrasts,
        })

    # --- step 10: oracles on the derivation part: the six-config CODOMAIN
    # oracle (the design's honest ceiling) and the seven-config full
    # carried-set oracle asked for by the committed step 1(b) text ---
    stack = np.stack([f1s_derive[c] for c in CODOMAIN])
    mx = stack.max(axis=0)     # per-language oracle F1 over CODOMAIN
    stack_carried = np.stack([f1s_derive[c] for c in CARRIED])
    mx_carried = stack_carried.max(axis=0)
    oracle_rows = []
    for gname, gmask in groups.items():
        gidx = np.where(gmask)[0]
        oracle_mean = float(mx[gidx].mean())
        single_means = {c: float(f1s_derive[c][gidx].mean()) for c in CODOMAIN}
        best_c = max(single_means, key=lambda c: single_means[c])
        carried_means = {c: float(f1s_derive[c][gidx].mean()) for c in CARRIED}
        best_carried = max(carried_means, key=lambda c: carried_means[c])
        oracle_rows.append({
            "group": gname, "oracle_mean": oracle_mean, "best_config": best_c,
            "best_mean": single_means[best_c],
            "gain": oracle_mean - single_means[best_c],
            "carried_oracle_mean": float(mx_carried[gidx].mean()),
            "carried_best_config": best_carried,
            "carried_best_mean": carried_means[best_carried],
            "carried_gain": (float(mx_carried[gidx].mean())
                             - carried_means[best_carried]),
        })

    # --- step 11: descriptive unique-best counts over CODOMAIN, all 1,940 ---
    is_max = stack == mx[None, :]
    n_tied = int((is_max.sum(axis=0) > 1).sum())
    unique_counts = {CODOMAIN[i]: int((is_max[i] & (is_max.sum(axis=0) == 1)).sum())
                     for i in range(len(CODOMAIN))}

    # --- step 12: outputs ---
    out = {"lang": langs, "N": N.astype(int), "category": category,
           "support_derive": support_derive, "support_judge": support_judge}
    for c in CODOMAIN + REFERENCE:
        out[f"f1_{c}"] = f1s_derive[c]
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    pd.DataFrame(out).to_csv(out_csv, index=False)

    L = [
        "# Step-1b evidence base for the per-language combined method\n",
        "Pre-registration: EXPERIMENTS_PLAN.md, \"Plan: per-language combined "
        "method\", 2026-07-29 amendments. This report covers the derivation "
        "part of the seed-301 split. Judge-part predictions are not read by "
        "this script; judge-part evaluation of the combined method happens "
        "in the later mixed_eval step. The feature-provenance audit (plan "
        "step 0) is at outputs/tables/combined_feature_provenance.md.\n",
        "## Gates passed\n",
    ]
    L.append(f"- y_true memmap shape: {tuple(int(x) for x in y.shape)}, matches "
             f"TOTAL_LINES ({TOTAL_LINES:,}).")
    L.append(f"- Draws {SEEDS[0]} and {TEST_SEED} loaded from {DRAW_DIR}, "
             f"disjoint ({len(val101):,} and {len(test201):,} lines, zero "
             f"overlap).")
    L.append(f"- Full remainder (kept minus draws 101 and 201): "
             f"{int(remainder_mask.sum()):,} lines, matches EXPECTED_REMAINDER "
             f"({EXPECTED_REMAINDER:,}) from analysis.carried_set_comparison.")
    L.append(f"- Wiring gate 1, full remainder: recomputed per-language F1 for "
             f"all {len(CARRIED)} Exp 38 carried configs matches {CARRIED_CSV} "
             f"within max absolute difference {max(gate1_diffs.values()):.2e} "
             "(limit 1e-9). Per-config max difference: "
             + ", ".join(f"{c} {d:.2e}" for c, d in gate1_diffs.items()) + ".")
    L.append(f"- Wiring gate 2, gt_min veto-instrument anchor, full remainder: "
             "recorded in outputs/tables/two_sided_selection.md, gt_min row of "
             f"the veto view table, tail global mean F1 {GT_MIN_RECORDED_TAIL_F1} "
             f"and FPs into tail labels {GT_MIN_RECORDED_TAIL_FP:,}. Recomputed "
             f"here: tail global mean F1 {gt_min_tail_f1:.4f} (difference "
             f"{f1_diff:.2e}, limit {GT_MIN_GATE_TOL:.0e}), FPs into tail "
             f"labels {gt_min_tail_fp:,} (exact match required).")
    L.append(f"- Split size gates: derivation part {len(derive_idx):,} lines "
             f"matches EXPECTED_DERIVATION ({EXPECTED_DERIVATION:,}); judge "
             f"part {len(judge_idx):,} lines matches EXPECTED_JUDGE "
             f"({EXPECTED_JUDGE:,}); the two sizes sum to the full remainder "
             f"{EXPECTED_REMAINDER:,}.")

    L += ["\n## Input prediction memmaps\n"]
    L.append("Every number below derives from these files as they existed at "
             "run time. pred_unmod_gate.npy and pred_floor21_gate.npy are "
             "built by analysis/solo_gates.py; their build reports are "
             "outputs/tables/unmod_gate_build.md and "
             "outputs/tables/floor21_gate_build.md.\n")
    L.append("| config | sha256 (first 16) | bytes | mtime |")
    L.append("|---|---|---|---|")
    for c in all_pred_configs:
        L.append(f"| {c} | {prov[c]['sha256'][:16]} | {prov[c]['bytes']:,} | "
                 f"{prov[c]['mtime']} |")

    L += ["\n## Split record\n"]
    L.append(f"Seed {RULE_SPLIT_SEED}, fraction {RULE_SPLIT_FRACTION} of the "
             "full remainder assigned to the derivation part. Derivation "
             f"part: {len(derive_idx):,} lines. Judge part: "
             f"{len(judge_idx):,} lines. sha256 of derive_idx.tobytes(): "
             f"{derive_sha}. Stored at {SPLIT_PATH} (scratch, not the repo).")

    n_tail = int(tail_mask.sum())
    n_magnet = int(magnet_mask.sum())
    L += ["\n## Support report\n"]
    L.append("Per-language true-line counts (bincount of y_true), tail "
             f"languages only (N < 1,000, {n_tail} languages).")
    L.append(f"- Derivation part of the seed-301 split ({len(derive_idx):,} "
             f"lines): median tail support {np.median(tail_derive):.1f}, "
             f"minimum {int(tail_derive.min())}, {tail_derive_ge_floor} of "
             f"{n_tail} tail languages at support >= {MIN_COLLAPSE_SUPPORT}.")
    L.append(f"- Judge part of the seed-301 split ({len(judge_idx):,} lines): "
             f"median tail support {np.median(tail_judge):.1f}, minimum "
             f"{int(tail_judge.min())}, {tail_judge_ge_floor} of {n_tail} "
             f"tail languages at support >= {MIN_COLLAPSE_SUPPORT}.")
    L.append(f"- flat_magnet languages ({n_magnet} total), derivation part of "
             f"the seed-301 split, at support >= {MIN_COLLAPSE_SUPPORT}: "
             f"{magnet_derive_ge_floor}.")
    if len(below_floor_idx):
        names = ", ".join(f"{langs[i]} (n={int(support_derive[i])})"
                          for i in below_floor_idx)
        n_zero = int((support_derive[below_floor_idx] == 0).sum())
        L.append(f"\nTail languages with derivation-part support below "
                 f"{MIN_COLLAPSE_SUPPORT}, {len(below_floor_idx)} of "
                 f"{n_tail}: each of their per-language F1 values on the "
                 f"derivation part rests on at most {MIN_COLLAPSE_SUPPORT - 1} "
                 f"true lines ({n_zero} of them have support 0, where F1 is 0 "
                 "for every configuration). They still enter the group means, "
                 "the oracle, and the unique-best counts above. Per the "
                 "pre-registration, their per-language values carry no leader "
                 "evidence for the rule; their treatment assignment comes "
                 f"from the training-side assignment rule alone. {names}")
    else:
        L.append(f"\nNo tail language falls below MIN_COLLAPSE_SUPPORT "
                 f"({MIN_COLLAPSE_SUPPORT}) on the derivation part of the "
                 "seed-301 split.")

    L += ["\n## Per-group mean F1, CODOMAIN configs, derivation part of the "
          "seed-301 split\n"]
    L.append("| group | n langs | " + " | ".join(CODOMAIN)
             + " | leader minus runner-up 95% CI |")
    L.append("|" + "---|" * (len(CODOMAIN) + 3))
    for row in group_rows:
        means_str = " | ".join(f"{row['means'][c]:.4f}" for c in CODOMAIN)
        L.append(f"| {row['group']} | {row['size']} | {means_str} | "
                 f"[{row['ci_lo']:.4f}, {row['ci_hi']:.4f}] |")
    L.append(f"\nBootstrap: paired resampling of the group's languages with "
             f"replacement, {BOOT_B:,} resamples, seed {BOOT_SEED}, one "
             "shared resample index matrix per group reused across all "
             "configurations, 95% percentile interval of (leader mean F1 "
             "minus runner-up mean F1), computed on the derivation part of "
             "the seed-301 split. The leader and runner-up were chosen by "
             "the point estimates on the same data, so this interval is "
             "biased away from zero and is descriptive, not a test; the "
             "fixed contrasts in the next section do not have this problem.")

    L += ["\n## Leader table, derivation part of the seed-301 split\n"]
    L.append("| group | leader | leader mean F1 | runner-up | runner-up mean "
             "F1 | delta | 95% CI |")
    L.append("|---|---|---|---|---|---|---|")
    for row in group_rows:
        L.append(f"| {row['group']} | {row['leader']} | "
                 f"{row['leader_mean']:.4f} | {row['runner_up']} | "
                 f"{row['runner_up_mean']:.4f} | {row['delta']:.4f} | "
                 f"[{row['ci_lo']:.4f}, {row['ci_hi']:.4f}] |")

    non_anchor = [c for c in CODOMAIN if c != ANCHOR]
    L += ["\n## Fixed contrasts against gt_margin_adaptive, derivation part "
          "of the seed-301 split\n"]
    L.append("The anchor configuration gt_margin_adaptive is fixed in "
             "advance (the configuration to beat on the primary quantity, "
             "0.9334 on the full remainder), so these intervals are not "
             "post-selection. Each cell is (config mean F1 minus "
             "gt_margin_adaptive mean F1) with its paired 95% interval from "
             "the same per-group resample matrix. Positive means the "
             "configuration is ahead of gt_margin_adaptive on the "
             "derivation part.\n")
    L.append("| group | " + " | ".join(non_anchor) + " |")
    L.append("|" + "---|" * (len(non_anchor) + 1))
    for row in group_rows:
        cells = " | ".join(
            f"{row['contrasts'][c][0]:+.4f} [{row['contrasts'][c][1]:+.4f}, "
            f"{row['contrasts'][c][2]:+.4f}]" for c in non_anchor)
        L.append(f"| {row['group']} | {cells} |")

    L += ["\n## Codomain oracle, derivation part of the seed-301 split\n"]
    L.append("Per-language maximum F1 over the six CODOMAIN configurations, "
             "then averaged unweighted over each group.\n")
    L.append("| group | oracle mean F1 | best single config | best single "
             "mean F1 | oracle gain |")
    L.append("|---|---|---|---|---|")
    for row in oracle_rows:
        L.append(f"| {row['group']} | {row['oracle_mean']:.4f} | "
                 f"{row['best_config']} | {row['best_mean']:.4f} | "
                 f"{row['gain']:.4f} |")

    L += ["\n## Full carried-set oracle, derivation part of the seed-301 "
          "split\n"]
    L.append("Per-language maximum F1 over the seven Exp 38 carried "
             "configurations (the committed step 1(b) asked for this oracle "
             "alongside the modular one). It includes freq_prior and "
             "learned_bias, which the combined method's codomain excludes, "
             "so it is an upper reference, not the design's ceiling.\n")
    L.append("| group | carried oracle mean F1 | best carried config | best "
             "carried mean F1 | oracle gain |")
    L.append("|---|---|---|---|---|")
    for row in oracle_rows:
        L.append(f"| {row['group']} | {row['carried_oracle_mean']:.4f} | "
                 f"{row['carried_best_config']} | "
                 f"{row['carried_best_mean']:.4f} | "
                 f"{row['carried_gain']:.4f} |")

    L += ["\n## Descriptive unique-best counts over CODOMAIN, derivation "
          "part of the seed-301 split\n"]
    L.append("Not a rule input. For each of the 1,940 languages, this counts "
             "which CODOMAIN configuration reaches the highest per-language "
             f"F1 on the derivation part. {n_tied} of {n_lang} languages "
             "have ties at the maximum; the strict counts below exclude "
             "tied languages, counting a configuration only where it is the "
             "unique best.\n")
    L.append(", ".join(f"{c}: {unique_counts[c]} strict" for c in CODOMAIN))

    L += ["\n## REFERENCE configurations\n"]
    L.append("Per-language F1 for the four REFERENCE configurations "
             "(freq_prior, learned_bias, margin_q5, margin_q5_head), "
             f"derivation part of the seed-301 split, is in {out_csv} for "
             "context. They are excluded from the group, leader, oracle, "
             "and unique-best sections above, which are restricted to "
             "CODOMAIN, the combined method's modular treatment set "
             "(EXPERIMENTS_PLAN.md, \"Treatment set (modularity-preserving "
             "only)\").")

    os.makedirs(os.path.dirname(out_md), exist_ok=True)
    with open(out_md, "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\nWrote {out_csv} and {out_md}")
    return out_md


if __name__ == "__main__":
    run()
