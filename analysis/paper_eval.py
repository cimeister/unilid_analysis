"""Camera-ready E1: common reporting set (EXPERIMENTS_PLAN.md, "Camera-ready
evaluation program (2026-08-06)"; conventions fixed in EXPERIMENTAL_SETUP.md,
"Camera-ready reporting conventions").

Three configurations: baseline (pred_baseline.npy), the promoted gate
gate_flat4_prox21 (pred_gate_flat4_prox21.npy, promoted 2026-08-06), and the
imported external fastText prediction file (pred_fasttext.npy, built by
analysis/import_external_pred.py). No new scoring happens here; every number is
recomputed from saved prediction memmaps under analysis.full_test_eval.SCRATCH_DIR.

Two reporting instruments, named at every table and gate below (no delta pairs terms
from different line sets, Exp 16 conclusion 3):
- full kept pool: the 45,377,279 kept lines (y_true >= 0), used for the camera-ready
  Table 1 cells.
- judge part of the seed-301 rule split: 27,002,441 lines, the appendix held-out
  comparison. analysis/combined_evidence.py derives and records this split; this
  script only re-derives it from the seed to verify the stored record, exactly as
  analysis/mixed_eval.py does.

Gates, in order, each aborting on failure (no silent fallback):
1. Language order: _load_model_data's canonical 1,940-language list must match the
   lang column of outputs/diagnostic/full_test_per_lang_prf.csv (the repo's standard
   order gate; every downstream per-language array is positionally aligned to this
   order).
2. Seed-301 rule split: re-derived from RULE_SPLIT_SEED/RULE_SPLIT_FRACTION over the
   held-out remainder (kept minus draws 101 and 201) exactly as
   analysis/mixed_eval.py does, and required to match the split already recorded at
   SPLIT_PATH (this script never creates that record).
3. Wiring gate A: recomputed full-remainder per-language F1 for the seven Exp 38
   carried configurations must match outputs/diagnostic/carried_set_per_lang_f1.csv
   to 1e-9 (same gate as analysis/mixed_eval.py and analysis/combined_evidence.py).
4. Wiring gate B: recomputed judge-part mean per-language F1 for baseline and
   gate_flat4_prox21 must match the corresponding column means of
   outputs/diagnostic/mixed_eval_judge_f1_gate_flat4_prox21.csv (computed fresh from
   that CSV at run time, not hardcoded) to 1e-6.
5. Wiring gate C: full-pool baseline macro F1 must reproduce the Exp 16 recorded
   0.9292 within 5e-5, and full-pool baseline macro FPR must reproduce the recorded
   2.0263e-5 within 1e-8. Macro FPR: per language, FPR_l = FP_l / (FP_l + TN_l) with
   TN_l = n_kept_in_instrument - support_l - FP_l, averaged unweighted over the 1,940
   languages, counts from analysis.metric_decomposition._per_lang_stats.

After the gates: per-config per-instrument per-language F1/FP tables, macro F1, macro
FPR; a paired bootstrap (B=10,000, seed 0, percentile 95%) on the judge part for
(gate_flat4_prox21 - baseline) and (gate_flat4_prox21 - fastText) over the 1,940
languages.

Outputs: outputs/tables/paper_eval.md, outputs/tables/paper_eval_table1_row.tex
(full-pool instrument), outputs/tables/paper_eval_appendix.tex (judge-part instrument
plus the bootstrap contrasts), outputs/diagnostic/paper_eval_per_lang_f1_fullpool.csv
and _judge.csv.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import subprocess

import numpy as np
import pandas as pd

from analysis.balanced_split import OUT_DIR as DRAW_DIR, SEEDS, TEST_SEED
from analysis.carried_set_comparison import CARRIED, EXPECTED_REMAINDER
from analysis.carried_set_comparison import OUT_CSV as CARRIED_CSV
from analysis.combined_evidence import (BOOT_B, BOOT_SEED, EXPECTED_DERIVATION,
                                        EXPECTED_JUDGE, RULE_SPLIT_FRACTION,
                                        RULE_SPLIT_SEED, SPLIT_PATH)
from analysis.config import TOTAL_LINES
from analysis.format_utils import to_latex, to_markdown
from analysis.full_test_eval import SCRATCH_DIR
from analysis.margin_diagnostic import PRF_CSV
from analysis.metric_decomposition import EXPECTED_KEPT, _per_lang_stats
from analysis.transfer_sweep import _load_model_data

# The camera-ready E1 reporting set. gate_flat4_prox21 is the promoted configuration
# (user decision 2026-08-06); fasttext is the imported external file built by
# analysis/import_external_pred.py.
CONFIGS = ("baseline", "gate_flat4_prox21", "fasttext")

# Wiring gate B is a fixed correctness check on these two configs regardless of what
# CONFIGS is overridden to; their memmaps are always loaded so the gate can run.
GATE_B_CONFIGS = ("baseline", "gate_flat4_prox21")

GATE_A_TOL = 1e-9    # precedent: analysis/mixed_eval.py, analysis/combined_evidence.py
GATE_B_TOL = 1e-9    # per-language comparison against the CSV, same standard as gate A
GATE_C_F1_TOL = 5e-5     # 4-decimal rounding tolerance, precedent metric_decomposition.py
GATE_C_FPR_TOL = 1e-8

# Pre-registered judge-part anchors (EXPERIMENTS_PLAN.md E1: "judge 0.9117/0.9498").
# Gate B checks the per-language vectors against the CSV at GATE_B_TOL AND the
# recomputed means against these anchors, so a silently regenerated CSV cannot
# redefine the gate's target.
GATE_B_ANCHORS = {"baseline": 0.9117, "gate_flat4_prox21": 0.9498}
GATE_B_ANCHOR_TOL = 5e-5

# Display names for LaTeX output: to_latex escapes only percent signs, so raw config
# names with underscores would break compilation; these names are also what the
# camera-ready fragments show.
DISPLAY = {"baseline": "baseline", "gate_flat4_prox21": "gate-flat4-prox21",
           "fasttext": "fastText"}

# All FPR values in tables are reported multiplied by 1e5, the convention of the
# submission's tables (header "Ma-FPR (x1e5)", analysis/tables.py).
FPR_SCALE = 1e5
FPR_HEADER = "Ma-FPR (x1e5)"

# Exp 16 recorded full-pool baseline values (metric_decomposition.py RECORDED
# ["baseline"]["overall"] and EXPERIMENTAL_SETUP.md "Camera-ready reporting
# conventions"), reproduced here as the wiring gate C reference.
GATE_C_F1_RECORDED = 0.9292
GATE_C_FPR_RECORDED = 2.0263e-5

MIXED_EVAL_JUDGE_CSV = "outputs/diagnostic/mixed_eval_judge_f1_gate_flat4_prox21.csv"

OUT_MD = "outputs/tables/paper_eval.md"
OUT_TEX_TABLE1 = "outputs/tables/paper_eval_table1_row.tex"
OUT_TEX_APPENDIX = "outputs/tables/paper_eval_appendix.tex"
OUT_CSV_FULLPOOL = "outputs/diagnostic/paper_eval_per_lang_f1_fullpool.csv"
OUT_CSV_JUDGE = "outputs/diagnostic/paper_eval_per_lang_f1_judge.csv"

FULLPOOL_INSTRUMENT = f"full kept pool, {EXPECTED_KEPT:,} lines"
JUDGE_INSTRUMENT = f"judge part, {EXPECTED_JUDGE:,} lines"


def _load_draw(seed: int) -> np.ndarray:
    """Mirrors analysis.combined_evidence._load_draw / analysis.mixed_eval._load_draw."""
    path = os.path.join(DRAW_DIR, f"val_lines_seed{seed}.npy")
    if not os.path.exists(path):
        raise FileNotFoundError(f"draw file missing: {path}")
    return np.load(path)


def _macro_fpr(fp: np.ndarray, support: np.ndarray, n_instrument: int) -> np.ndarray:
    """Per-language FPR_l = FP_l / (FP_l + TN_l), TN_l = n_instrument - support_l -
    FP_l. Returns the per-language array; the caller takes the unweighted mean."""
    tn = n_instrument - support - fp
    denom = fp + tn
    return np.divide(fp, denom, out=np.zeros_like(fp, dtype=float), where=denom > 0)


def _require_file(path: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(f"required artifact missing: {path}")


def run(configs=CONFIGS, out_md: str = OUT_MD) -> str:
    configs = tuple(dict.fromkeys(configs))

    # A --configs override must not overwrite the canonical camera-ready artifacts:
    # every output path gets a config-derived suffix unless the full default set runs.
    if configs == CONFIGS:
        out_tex_table1, out_tex_appendix = OUT_TEX_TABLE1, OUT_TEX_APPENDIX
        out_csv_fullpool, out_csv_judge = OUT_CSV_FULLPOOL, OUT_CSV_JUDGE
    else:
        suffix = "_" + "-".join(configs)
        if out_md == OUT_MD:
            out_md = OUT_MD.replace(".md", f"{suffix}.md")
        out_tex_table1 = OUT_TEX_TABLE1.replace(".tex", f"{suffix}.tex")
        out_tex_appendix = OUT_TEX_APPENDIX.replace(".tex", f"{suffix}.tex")
        out_csv_fullpool = OUT_CSV_FULLPOOL.replace(".csv", f"{suffix}.csv")
        out_csv_judge = OUT_CSV_JUDGE.replace(".csv", f"{suffix}.csv")

    # --- gate 1: canonical language order ---
    weights, langs, _m = _load_model_data()
    del weights
    n_lang = len(langs)
    _require_file(PRF_CSV)
    prf = pd.read_csv(PRF_CSV)
    if prf.lang.tolist() != langs:
        raise RuntimeError(f"language order gate failed: {PRF_CSV} lang column does "
                           "not match _load_model_data's canonical language list")
    N = prf.N.values

    y_path = os.path.join(SCRATCH_DIR, "y_true.npy")
    _require_file(y_path)
    y = np.asarray(np.lib.format.open_memmap(y_path, mode="r"))
    if y.shape != (TOTAL_LINES,):
        raise RuntimeError(f"y_true memmap has shape {y.shape}, expected "
                           f"({TOTAL_LINES},)")
    kept = y >= 0
    n_kept = int(kept.sum())
    if n_kept != EXPECTED_KEPT:
        raise RuntimeError(f"full kept pool {n_kept:,} != EXPECTED_KEPT "
                           f"({EXPECTED_KEPT:,})")
    yk = y[kept]

    # --- gate 2: seed-301 rule split, re-derived exactly as mixed_eval.py does ---
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

    remainder_idx = np.where(remainder_mask)[0]
    u = np.random.default_rng(RULE_SPLIT_SEED).random(remainder_idx.size)
    derive_idx = remainder_idx[u < RULE_SPLIT_FRACTION]
    judge_idx = remainder_idx[u >= RULE_SPLIT_FRACTION]
    if len(derive_idx) != EXPECTED_DERIVATION:
        raise RuntimeError(f"derivation part {len(derive_idx):,} != "
                           f"EXPECTED_DERIVATION {EXPECTED_DERIVATION:,}")
    if len(judge_idx) != EXPECTED_JUDGE:
        raise RuntimeError(f"judge part {len(judge_idx):,} != EXPECTED_JUDGE "
                           f"{EXPECTED_JUDGE:,}")
    if len(derive_idx) + len(judge_idx) != EXPECTED_REMAINDER:
        raise RuntimeError(f"split sizes {len(derive_idx):,} + {len(judge_idx):,} != "
                           f"remainder {EXPECTED_REMAINDER:,}")

    _require_file(SPLIT_PATH)
    with np.load(SPLIT_PATH) as stored:
        if not (np.array_equal(stored["derive_idx"], derive_idx)
                and np.array_equal(stored["judge_idx"], judge_idx)):
            raise RuntimeError(f"split stored at {SPLIT_PATH} does not match the "
                               "recomputed seed-301 split")
    yj = y[judge_idx]
    n_judge = len(judge_idx)
    print(f"Gates 1-2 passed: language order matches {PRF_CSV}; seed-{RULE_SPLIT_SEED} "
          f"split reproduced ({len(derive_idx):,} derivation / {len(judge_idx):,} "
          f"judge) and matches {SPLIT_PATH}.")

    # --- load prediction memmaps: CARRIED (gate A) + requested configs + the fixed
    # gate B configs, union, so every gate can run regardless of a --configs override ---
    all_pred_configs = list(dict.fromkeys(
        list(CARRIED) + list(configs) + list(GATE_B_CONFIGS)))
    preds = {}
    prov = {}
    empty_counts = {}
    for c in all_pred_configs:
        p = os.path.join(SCRATCH_DIR, f"pred_{c}.npy")
        _require_file(p)
        preds[c] = np.asarray(np.lib.format.open_memmap(p, mode="r"))
        st = os.stat(p)
        with open(p, "rb") as fh:
            file_sha = hashlib.sha256(fh.read()).hexdigest()
        prov[c] = {"bytes": st.st_size, "sha256": file_sha}
        # Sentinel guard: a partially imported or aborted memmap leaves UNSEEN (-3)
        # or EXCLUDED (-2) values, which _per_lang_stats would silently score as
        # false negatives with no false positive, deflating F1 and inflating FPR
        # plausibly. Only EMPTY (-1) is a legitimate value on kept lines.
        n_bad = int((preds[c][kept] < -1).sum())
        if n_bad:
            raise RuntimeError(
                f"prediction memmap {p} has {n_bad:,} sentinel values below -1 on "
                "the kept pool (UNSEEN or EXCLUDED); the import or scoring pass "
                "that produced it is incomplete")
        empty_counts[c] = int((preds[c][kept] == -1).sum())

    # --- gate 3: wiring gate A, carried set on the full remainder ---
    _require_file(CARRIED_CSV)
    carried_df = pd.read_csv(CARRIED_CSV)
    if carried_df.lang.tolist() != langs:
        raise RuntimeError(f"{CARRIED_CSV} language order does not match the "
                           "canonical language list")
    gate_a_diffs = {}
    for c in CARRIED:
        f1_here = _per_lang_stats(preds[c][remainder_mask], yr, n_lang)[2]
        diff = float(np.abs(f1_here - carried_df[f"f1_{c}"].values).max())
        gate_a_diffs[c] = diff
        if diff > GATE_A_TOL:
            raise RuntimeError(f"wiring gate A failed for {c}: max |diff| {diff:.2e} "
                               f"vs {CARRIED_CSV} (tol {GATE_A_TOL:.0e})")
    print(f"Wiring gate A passed: full-remainder per-language F1 reproduces "
          f"{CARRIED_CSV} for all {len(CARRIED)} carried configs (max diff "
          f"{max(gate_a_diffs.values()):.2e}).")

    # --- gate 4: wiring gate B, judge-part mean F1 for baseline and gate_flat4_prox21 ---
    _require_file(MIXED_EVAL_JUDGE_CSV)
    judge_ref = pd.read_csv(MIXED_EVAL_JUDGE_CSV)
    if judge_ref.lang.tolist() != langs:
        raise RuntimeError(f"{MIXED_EVAL_JUDGE_CSV} language order does not match "
                           "the canonical language list")
    stats_judge_gate = {c: _per_lang_stats(preds[c][judge_idx], yj, n_lang)
                        for c in GATE_B_CONFIGS}
    gate_b_diffs = {}
    for c in GATE_B_CONFIGS:
        f1_vec = stats_judge_gate[c][2]
        vec_diff = float(np.abs(f1_vec - judge_ref[f"f1_{c}"].values).max())
        if vec_diff > GATE_B_TOL:
            raise RuntimeError(f"wiring gate B failed for {c}: max per-language "
                               f"|diff| {vec_diff:.2e} vs {MIXED_EVAL_JUDGE_CSV} "
                               f"(tol {GATE_B_TOL:.0e})")
        recomputed_mean = float(f1_vec.mean())
        anchor = GATE_B_ANCHORS[c]
        anchor_diff = abs(recomputed_mean - anchor)
        if anchor_diff > GATE_B_ANCHOR_TOL:
            raise RuntimeError(f"wiring gate B failed for {c}: recomputed judge-part "
                               f"mean F1 {recomputed_mean:.6f} vs pre-registered "
                               f"anchor {anchor} (diff {anchor_diff:.2e}, tol "
                               f"{GATE_B_ANCHOR_TOL:.0e})")
        gate_b_diffs[c] = (recomputed_mean, anchor, vec_diff)
    print("Wiring gate B passed: judge-part per-language F1 matches "
          f"{MIXED_EVAL_JUDGE_CSV} at {GATE_B_TOL:.0e}, and means match the "
          "pre-registered anchors: "
          + ", ".join(f"{c} {r:.6f} (anchor {a})"
                       for c, (r, a, v) in gate_b_diffs.items()) + ".")

    # --- gate 5: wiring gate C, full-pool baseline reproduces the Exp 16 numbers ---
    prec_b, rec_b, f1_b, tp_b, fp_b, fn_b = _per_lang_stats(
        preds["baseline"][kept], yk, n_lang)
    fullpool_baseline_f1 = float(f1_b.mean())
    diff_f1 = abs(fullpool_baseline_f1 - GATE_C_F1_RECORDED)
    if diff_f1 > GATE_C_F1_TOL:
        raise RuntimeError(f"wiring gate C failed: full-pool baseline macro F1 "
                           f"{fullpool_baseline_f1:.6f} vs recorded "
                           f"{GATE_C_F1_RECORDED} (diff {diff_f1:.2e}, tol "
                           f"{GATE_C_F1_TOL:.0e})")
    support_fullpool = np.bincount(yk, minlength=n_lang).astype(float)
    fpr_b = _macro_fpr(fp_b, support_fullpool, n_kept)
    fullpool_baseline_fpr = float(fpr_b.mean())
    diff_fpr = abs(fullpool_baseline_fpr - GATE_C_FPR_RECORDED)
    if diff_fpr > GATE_C_FPR_TOL:
        raise RuntimeError(f"wiring gate C failed: full-pool baseline macro FPR "
                           f"{fullpool_baseline_fpr:.8f} vs recorded "
                           f"{GATE_C_FPR_RECORDED} (diff {diff_fpr:.2e}, tol "
                           f"{GATE_C_FPR_TOL:.0e})")
    print(f"Wiring gate C passed: full-pool baseline macro F1 "
          f"{fullpool_baseline_f1:.6f} (recorded {GATE_C_F1_RECORDED}), macro FPR "
          f"{fullpool_baseline_fpr:.8f} (recorded {GATE_C_FPR_RECORDED}).")

    # --- per-config per-instrument per-language stats ---
    stats_fullpool = {c: _per_lang_stats(preds[c][kept], yk, n_lang) for c in configs}
    stats_judge = {c: _per_lang_stats(preds[c][judge_idx], yj, n_lang) for c in configs}
    support_judge = np.bincount(yj, minlength=n_lang).astype(int)
    support_fullpool_int = support_fullpool.astype(int)

    macro_f1_fullpool = {c: float(stats_fullpool[c][2].mean()) for c in configs}
    macro_f1_judge = {c: float(stats_judge[c][2].mean()) for c in configs}
    macro_fpr_fullpool = {
        c: float(_macro_fpr(stats_fullpool[c][4], support_fullpool, n_kept).mean())
        for c in configs}
    macro_fpr_judge = {
        c: float(_macro_fpr(stats_judge[c][4], support_judge.astype(float),
                            n_judge).mean())
        for c in configs}

    # git commit is resolved before the first write so a failure cannot leave a
    # mixed-provenance output set (fresh CSVs next to stale reports).
    try:
        git_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, OSError) as e:
        raise RuntimeError(f"git rev-parse HEAD failed: {e}") from e

    def _write_per_lang_csv(path, stats, support):
        out = {"lang": langs, "N": N.astype(int)}
        for c in configs:
            out[f"f1_{c}"] = stats[c][2]
            out[f"fp_{c}"] = stats[c][4].astype(int)
        out["support"] = support
        os.makedirs(os.path.dirname(path), exist_ok=True)
        pd.DataFrame(out).to_csv(path, index=False)

    _write_per_lang_csv(out_csv_fullpool, stats_fullpool, support_fullpool_int)
    _write_per_lang_csv(out_csv_judge, stats_judge, support_judge)

    # --- paired bootstrap, judge part, over the 1,940 languages ---
    f1_judge_vec = {c: stats_judge[c][2] for c in configs}
    bootstrap_results = {}
    if "gate_flat4_prox21" in configs:
        rng = np.random.default_rng(BOOT_SEED)
        resample = rng.integers(0, n_lang, size=(BOOT_B, n_lang))
        boots = {c: f1_judge_vec[c][resample].mean(axis=1) for c in configs}
        for other in ("baseline", "fasttext"):
            if other not in configs:
                continue
            d = boots["gate_flat4_prox21"] - boots[other]
            point = float(f1_judge_vec["gate_flat4_prox21"].mean()
                          - f1_judge_vec[other].mean())
            lo, hi = np.percentile(d, [2.5, 97.5])
            bootstrap_results[other] = {"mean_diff": point, "ci_lo": float(lo),
                                        "ci_hi": float(hi)}

    # --- markdown report ---
    L = [
        "# Camera-ready E1: common reporting set\n",
        "Pre-registration: EXPERIMENTS_PLAN.md, \"Camera-ready evaluation program "
        "(2026-08-06)\", E1. Configurations: " + ", ".join(configs) + ". Instruments "
        "follow EXPERIMENTAL_SETUP.md, \"Camera-ready reporting conventions\": Table "
        f"1 cells use the {FULLPOOL_INSTRUMENT}; the appendix comparison uses the "
        f"{JUDGE_INSTRUMENT}. No delta pairs terms from different line sets.\n",
        "## Gates passed\n",
    ]
    L.append(f"- Language order: _load_model_data's {n_lang:,}-language list matches "
             f"the lang column of {PRF_CSV}.")
    L.append(f"- y_true memmap shape {tuple(int(x) for x in y.shape)} matches "
             f"TOTAL_LINES ({TOTAL_LINES:,}); {FULLPOOL_INSTRUMENT} matches "
             f"EXPECTED_KEPT.")
    L.append(f"- Full remainder (kept minus draws {SEEDS[0]} and {TEST_SEED}): "
             f"{int(remainder_mask.sum()):,} lines, matches EXPECTED_REMAINDER "
             f"({EXPECTED_REMAINDER:,}).")
    L.append(f"- Seed-{RULE_SPLIT_SEED} split recomputed with fraction "
             f"{RULE_SPLIT_FRACTION}: derivation part {len(derive_idx):,} lines "
             f"(EXPECTED_DERIVATION {EXPECTED_DERIVATION:,}), judge part "
             f"{len(judge_idx):,} lines (EXPECTED_JUDGE {EXPECTED_JUDGE:,}); matches "
             f"the split recorded at {SPLIT_PATH}.")
    L.append(f"- Wiring gate A (full-remainder per-language F1 vs {CARRIED_CSV}): "
             f"max absolute difference {max(gate_a_diffs.values()):.2e} over "
             f"{len(CARRIED)} carried configs (tol {GATE_A_TOL:.0e}). Per-config: "
             + ", ".join(f"{c} {d:.2e}" for c, d in gate_a_diffs.items()) + ".")
    L.append(f"- Wiring gate B (judge-part mean F1 vs {MIXED_EVAL_JUDGE_CSV}, tol "
             f"{GATE_B_TOL:.0e}): "
             + "; ".join(f"{c} recomputed {r:.6f} vs csv {m:.6f} (diff {d:.2e})"
                         for c, (r, m, d) in gate_b_diffs.items()) + ".")
    L.append(f"- Wiring gate C (full-pool baseline reproduces the Exp 16 recorded "
             f"values): macro F1 {fullpool_baseline_f1:.6f} vs recorded "
             f"{GATE_C_F1_RECORDED} (diff {diff_f1:.2e}, tol {GATE_C_F1_TOL:.0e}); "
             f"macro FPR {fullpool_baseline_fpr:.8f} vs recorded "
             f"{GATE_C_FPR_RECORDED} (diff {diff_fpr:.2e}, tol "
             f"{GATE_C_FPR_TOL:.0e}).")

    L.append("\nSentinel guard: zero UNSEEN/EXCLUDED values on the kept pool for "
             "every loaded memmap. EMPTY (-1) counts on the kept pool: "
             + ", ".join(f"{c} {empty_counts[c]:,}" for c in all_pred_configs) + ".")
    L.append("\nNote for the camera-ready table: the submission's Table 1 rows state "
             f"N = {TOTAL_LINES:,}; the cells here use the {FULLPOOL_INSTRUMENT}, so "
             "every restated row in the camera-ready table must carry N = "
             f"{EXPECTED_KEPT:,} and the N column must be updated for all rows, not "
             "only the new one.")

    rows_fullpool = [[c, f"{macro_f1_fullpool[c]:.4f}",
                      f"{macro_fpr_fullpool[c] * FPR_SCALE:.4f}"]
                     for c in configs]
    L.append("\n" + to_markdown(rows_fullpool, ["config", "macro F1", FPR_HEADER],
                                caption=f"Macro F1 and macro FPR (x1e5), "
                                        f"{FULLPOOL_INSTRUMENT}"))

    rows_judge = [[c, f"{macro_f1_judge[c]:.4f}",
                   f"{macro_fpr_judge[c] * FPR_SCALE:.4f}"]
                 for c in configs]
    L.append(to_markdown(rows_judge, ["config", "macro F1", FPR_HEADER],
                         caption=f"Macro F1 and macro FPR (x1e5), {JUDGE_INSTRUMENT}"))

    L.append(f"## Paired bootstrap, {JUDGE_INSTRUMENT}\n")
    L.append(f"B={BOOT_B:,}, seed={BOOT_SEED}, percentile 95% interval, paired "
             f"resample over the {n_lang:,} language positions, one shared resample "
             "matrix reused across contrasts. Point estimate = (gate_flat4_prox21 "
             "judge-part mean F1) minus (comparator judge-part mean F1).\n")
    if bootstrap_results:
        rows_boot = [[other, f"{r['mean_diff']:+.4f}",
                      f"[{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}]"]
                     for other, r in bootstrap_results.items()]
        L.append(to_markdown(rows_boot, ["comparator", "mean diff", "95% CI"],
                             caption="gate_flat4_prox21 minus comparator, "
                                     f"{JUDGE_INSTRUMENT}"))
    else:
        L.append("gate_flat4_prox21 not in the requested config set; no bootstrap "
                 "contrasts computed.\n")

    L.append("## Input prediction memmaps\n")
    L.append("| config | sha256 (first 16) | bytes |")
    L.append("|---|---|---|")
    for c in all_pred_configs:
        L.append(f"| {c} | {prov[c]['sha256'][:16]} | {prov[c]['bytes']:,} |")
    carried_only = [c for c in all_pred_configs if c not in configs]
    if carried_only:
        L.append("\nLoaded only for wiring gates A/B above (not part of the "
                 "requested config set): " + ", ".join(carried_only) + ".")

    L.append(f"\nPer-language detail: {out_csv_fullpool} ({FULLPOOL_INSTRUMENT}), "
             f"{out_csv_judge} ({JUDGE_INSTRUMENT}).")
    L.append(f"\nGit commit: {git_commit}.")

    os.makedirs(os.path.dirname(out_md), exist_ok=True)
    with open(out_md, "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))

    # --- LaTeX outputs (display names avoid raw underscores, which to_latex does
    # not escape; FPR pre-formatted at the x1e5 scale so _fmt_fpr's 1-decimal
    # rounding never applies) ---
    def _disp(c):
        return DISPLAY.get(c, c.replace("_", "-"))

    tex_table1_rows = [[_disp(c), macro_f1_fullpool[c],
                        f"{macro_fpr_fullpool[c] * FPR_SCALE:.4f}"]
                       for c in configs]
    tex_table1 = to_latex(
        tex_table1_rows, ["Method", "Macro F1", FPR_HEADER],
        caption=f"Camera-ready Table 1 cells, {FULLPOOL_INSTRUMENT}. FPR values are "
                "multiplied by 1e5.",
        label="tab:paper_eval_table1", col_formats=["str", "metric", "str"])
    with open(out_tex_table1, "w") as f:
        f.write(tex_table1 + "\n")

    tex_appendix_rows = [[_disp(c), macro_f1_judge[c],
                          f"{macro_fpr_judge[c] * FPR_SCALE:.4f}"] for c in configs]
    tex_appendix = to_latex(
        tex_appendix_rows, ["Method", "Macro F1", FPR_HEADER],
        caption=f"Appendix held-out comparison, {JUDGE_INSTRUMENT}. FPR values are "
                "multiplied by 1e5.",
        label="tab:paper_eval_appendix", col_formats=["str", "metric", "str"])
    appendix_parts = [tex_appendix]
    if bootstrap_results:
        tex_boot_rows = [[_disp(other), r["mean_diff"], r["ci_lo"], r["ci_hi"]]
                         for other, r in bootstrap_results.items()]
        tex_boot = to_latex(
            tex_boot_rows, ["Comparator", "Mean diff", "CI low", "CI high"],
            caption=f"Paired bootstrap, {_disp('gate_flat4_prox21')} minus "
                    f"comparator, {JUDGE_INSTRUMENT}.",
            label="tab:paper_eval_bootstrap",
            col_formats=["str", "metric", "metric", "metric"])
        appendix_parts.append(tex_boot)
    with open(out_tex_appendix, "w") as f:
        f.write("\n\n".join(appendix_parts) + "\n")

    print(f"\nWrote {out_md}, {out_tex_table1}, {out_tex_appendix}, "
          f"{out_csv_fullpool}, {out_csv_judge}")
    print(f"\nKey numbers ({FULLPOOL_INSTRUMENT}): "
          + ", ".join(f"{c} macro F1 {macro_f1_fullpool[c]:.4f} / FPR "
                       f"{macro_fpr_fullpool[c] * FPR_SCALE:.4f}e-5" for c in configs))
    print(f"Key numbers ({JUDGE_INSTRUMENT}): "
          + ", ".join(f"{c} macro F1 {macro_f1_judge[c]:.4f} / FPR "
                       f"{macro_fpr_judge[c] * FPR_SCALE:.4f}e-5" for c in configs))
    for other, r in bootstrap_results.items():
        print(f"Bootstrap gate_flat4_prox21 - {other}: {r['mean_diff']:+.4f} "
              f"[{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}]")
    return out_md


def main():
    parser = argparse.ArgumentParser(
        description="Camera-ready E1 evaluation: baseline, gate_flat4_prox21, "
                    "fastText on the full kept pool and the judge part.")
    parser.add_argument("--configs", nargs="+", default=list(CONFIGS),
                        help="override the evaluated config set (default: all "
                             f"three E1 configs: {', '.join(CONFIGS)})")
    args = parser.parse_args()
    run(tuple(args.configs))


if __name__ == "__main__":
    main()
