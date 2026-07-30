"""Step-2 assignment rule for the per-language combined method (EXPERIMENTS_PLAN.md,
"Plan: per-language combined method", 2026-07-29 amendments).

Rule version 1, drafted 2026-07-30 from the derivation-part evidence in
outputs/tables/combined_evidence.md: floor21_gate leads every group except head,
where unmodified and gt_min lead it by at most 0.0010. The rule is a two-clause
decision list with a single carve-out for the head stratum:

- Clause 1 (head carve-out): if a language's training-corpus line count N is at
  least HEAD_N (18,000, analysis.full_test_margin.HEAD_N), assign row_treatment
  "unmod" (the unmodified weight row), gate_on False, rule_clause "head_unmod".
- Terminal clause: every other language gets row_treatment "floor21" (the row
  clamped to FLOOR_TARGET, analysis.full_test_floor21.FLOOR_TARGET = -21.0),
  gate_on True, rule_clause "else_floor21_gate".

Head is assigned unmod rather than gt_min despite gt_min's slightly higher
derivation-part head mean (0.9596 gt_min vs 0.9594 unmod,
outputs/tables/combined_evidence.md, "Per-group mean F1" table, head row):
the 0.0002-0.0004 difference is negligible against keeping production head
rows bit-identical to the unmodified model and avoiding a gt_counts.csv
dependency in the head clause. gt_min for head is left as the noted
alternative for a later rule version.

N is the only feature the rule reads. It is training-side provenance (see
outputs/tables/combined_feature_provenance.md) and is present for all 1,940
languages; the script asserts this rather than assuming it.

This script only materializes the rule into a CSV. It does not score, gate, or
build any prediction matrix, and the rule itself is pending user sign-off before
step 3 (analysis/mixed_matrix.py) may use it.

RULE_VERSION and EXPECTED_N_LANG are the two new constants this module
defines. RULE_VERSION is a version tag for the decision list above, not a
numeric threshold, and exists so a later revision of the rule can be told
apart from this one in any output or log that carries it. EXPECTED_N_LANG is
the language-count gate value (1,940, the model's own language count),
asserted against the loaded PRF_CSV in step 1.

Output: outputs/diagnostic/mixed_assignments.csv, one row per language in the
model's own language order (1,940 rows), columns lang, N, row_treatment,
gate_on, rule_clause, floor_unmod, floor_assigned, seen_rescale, target_lang,
floor_gap_shift.

row_treatment is the authoritative instruction for the matrix build (step 3,
analysis/mixed_matrix.py); floor_assigned is diagnostic only, not an
instruction to apply on its own.

floor_unmod is the recorded floor column from outputs/diagnostic/gt_counts.csv
(the row's minimum log-weight before any treatment). floor_assigned is
FLOOR_TARGET where row_treatment is "floor21". For unmod rows it is used
internally (to compute floor_gap_shift below) as floor_unmod, the row's own
six-decimal-rounded recorded floor; that value is not a clamp target, so the
output CSV column instead leaves floor_assigned empty (NaN) for unmod rows,
where a reader could otherwise mistake it for one. seen_rescale is
0.0 for the unmod and floor21 treatments; it is defined generally from the
gt_counts.csv columns as the Good-Turing seen-token rescale constant
log((NONSPECIAL_BUDGET - target) / (NONSPECIAL_BUDGET - plateau_mass)), with
target = min(plateau_mass, NONSPECIAL_BUDGET * n1 / T), so the column stays
correct if a later rule version assigns the "gt" treatment. No language
receives that treatment under rule version 1, so seen_rescale is 0.0 for every
row in this version's output. NONSPECIAL_BUDGET is imported from
analysis.full_test_gt rather than restated here.

floor_gap_shift is the pre-registered mechanism quantity for step 2: for a
language L with a non-null higher-resource same-script confuser (target_lang in
lang_diagnostic.csv, null for 302 of the 1,940 languages), let c be that
confuser's index. gap_mixed is floor_assigned[L] minus floor_assigned[c] under
the mixed assignment. gap_solo is the same difference computed as if c also
received L's own treatment: 0.0 if L is assigned floor21 (both rows sit at
FLOOR_TARGET), or floor_unmod[L] minus floor_unmod[c] if L is assigned unmod.
floor_gap_shift is gap_mixed minus gap_solo, left null for languages without a
confuser.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from analysis.full_test_floor21 import FLOOR_TARGET
from analysis.full_test_gt import NONSPECIAL_BUDGET
from analysis.full_test_margin import HEAD_N
from analysis.margin_diagnostic import PRF_CSV
from analysis.transfer_sweep import _load_model_data

RULE_VERSION = 1   # decision-list version tag; see module docstring

# The full row_treatment codomain across rule versions, not just the two
# values rule version 1 assigns (unmod, floor21). "gt" is included so the
# assignment gate below does not need updating when a later rule version
# starts assigning it; gap_solo is undefined for "gt" and that is enforced
# separately, further down, where it correctly aborts if "gt" ever appears.
TREATMENTS = ("unmod", "floor21", "gt")

LANG_DIAG_CSV = "outputs/diagnostic/lang_diagnostic.csv"
GT_COUNTS_CSV = "outputs/diagnostic/gt_counts.csv"
OUT_CSV = "outputs/diagnostic/mixed_assignments.csv"

EXPECTED_N_LANG = 1_940

# Tolerance for the gt_counts.csv floor cross-check against the recomputed
# per-row weight minima (step 1): both are float32-derived minima over the
# same rows, so agreement should be near machine precision; 1e-5 catches a
# stale or mismatched gt_counts.csv without over-tripping on float rounding.
FLOOR_CROSSCHECK_TOL = 1e-5


def run(out_csv: str = OUT_CSV) -> str:
    # --- step 1: load the four sources and gate their language order ---
    prf = pd.read_csv(PRF_CSV)
    langs = prf.lang.tolist()
    n_lang = len(langs)
    if n_lang != EXPECTED_N_LANG:
        raise RuntimeError(f"{PRF_CSV} has {n_lang} languages, expected "
                           f"{EXPECTED_N_LANG}")
    if pd.isna(prf["N"]).any():
        raise RuntimeError(f"NaN found in the N column of {PRF_CSV}; the plan "
                           "states N is present for all 1,940 languages")
    N = prf["N"].values
    category = prf["category"].values   # used only for the summary's group
                                         # membership breakdown, step 6

    diag = pd.read_csv(LANG_DIAG_CSV)
    required_diag_cols = {"lang", "d_up", "target_lang"}
    missing_diag_cols = required_diag_cols - set(diag.columns)
    if missing_diag_cols:
        raise RuntimeError(f"{LANG_DIAG_CSV} missing required columns: "
                           f"{sorted(missing_diag_cols)}")

    gt = pd.read_csv(GT_COUNTS_CSV)
    required_gt_cols = {"lang", "floor", "plateau_size", "plateau_mass", "T", "n1"}
    missing_gt_cols = required_gt_cols - set(gt.columns)
    if missing_gt_cols:
        raise RuntimeError(f"{GT_COUNTS_CSV} missing required columns: "
                           f"{sorted(missing_gt_cols)}")
    for col in ("floor", "plateau_mass", "n1"):
        if gt[col].isna().any():
            raise RuntimeError(f"NaN found in {GT_COUNTS_CSV} column {col!r}")
    if gt["T"].isna().any():
        raise RuntimeError(f"NaN found in {GT_COUNTS_CSV} column 'T'")
    if (gt["T"].values <= 0).any():
        raise RuntimeError(f"{GT_COUNTS_CSV} contains a non-positive T value; "
                           "the seen_rescale formula divides by T")

    weights, model_langs, _lang_to_idx = _load_model_data()
    for name, other_langs in ((PRF_CSV, langs), (LANG_DIAG_CSV, diag.lang.tolist()),
                              (GT_COUNTS_CSV, gt.lang.tolist())):
        if other_langs != model_langs:
            raise RuntimeError(f"{name} language order does not match the model "
                               "language list from _load_model_data()")

    # Cross-check gt_counts.csv's floor column against the recomputed per-row
    # weight minima before the weight matrix is discarded: floor is defined
    # as each row's own minimum log-weight, so a mismatch here means
    # gt_counts.csv is stale or was built from a different model file than
    # the one _load_model_data() just loaded.
    floor_unmod = gt["floor"].values.astype(np.float64)
    row_min = np.asarray(weights).min(axis=1).astype(np.float64)
    floor_diff = float(np.abs(row_min - floor_unmod).max())
    if floor_diff > FLOOR_CROSSCHECK_TOL:
        raise RuntimeError(f"{GT_COUNTS_CSV} floor column does not match the "
                           f"recomputed per-row weight minima: max |diff| "
                           f"{floor_diff:.2e} (limit {FLOOR_CROSSCHECK_TOL:.0e})")
    del weights

    # --- step 2: the two-clause decision list ---
    is_head = N >= HEAD_N
    row_treatment = np.where(is_head, "unmod", "floor21")
    gate_on = np.where(is_head, False, True)
    rule_clause = np.where(is_head, "head_unmod", "else_floor21_gate")

    assigned = np.isin(row_treatment, TREATMENTS)
    if int(assigned.sum()) != n_lang:
        raise RuntimeError("assignment gate failed: not every language received "
                           "a treatment in TREATMENTS")
    bad_gate = gate_on & (N >= HEAD_N)
    if bad_gate.any():
        raise RuntimeError(f"{int(bad_gate.sum())} gated language(s) have "
                           f"N >= HEAD_N ({HEAD_N}), violating the head carve-out")

    n_head = int((rule_clause == "head_unmod").sum())
    n_else = int((rule_clause == "else_floor21_gate").sum())
    if n_head + n_else != n_lang:
        raise RuntimeError("clause counts do not sum to the full language count")

    # --- step 3: floor columns ---
    # floor_unmod was already computed in step 1 (used there for the
    # gt_counts.csv cross-check); reused here unchanged.
    is_floor21 = row_treatment == "floor21"
    floor_assigned = np.where(is_floor21, FLOOR_TARGET, floor_unmod)

    plateau_mass = gt["plateau_mass"].values.astype(np.float64)
    n1 = gt["n1"].values.astype(np.float64)
    T_col = gt["T"].values.astype(np.float64)   # bracket access: gt["T"], never gt.T
    gt_target = np.minimum(plateau_mass, NONSPECIAL_BUDGET * n1 / T_col)
    gt_seen_rescale = np.log((NONSPECIAL_BUDGET - gt_target)
                             / (NONSPECIAL_BUDGET - plateau_mass))
    is_gt = row_treatment == "gt"   # always False under rule version 1
    seen_rescale = np.where(is_gt, gt_seen_rescale, 0.0)

    # --- step 4: floor-gap shift against the solo reference ---
    lang_to_idx = {l: i for i, l in enumerate(langs)}
    target_lang_list = diag["target_lang"].tolist()
    has_confuser = np.array([not pd.isna(t) for t in target_lang_list])
    c_idx = np.full(n_lang, -1, dtype=np.int64)
    for i in range(n_lang):
        if not has_confuser[i]:
            continue
        conf = str(target_lang_list[i])
        if conf not in lang_to_idx:
            raise RuntimeError(f"target_lang {conf!r} for {langs[i]} is not in "
                               "the model language list")
        c_idx[i] = lang_to_idx[conf]

    gap_mixed = np.full(n_lang, np.nan)
    gap_solo = np.full(n_lang, np.nan)
    gap_mixed[has_confuser] = (floor_assigned[has_confuser]
                               - floor_assigned[c_idx[has_confuser]])

    is_floor21_conf = has_confuser & is_floor21
    is_unmod_conf = has_confuser & (row_treatment == "unmod")
    unexpected = has_confuser & ~is_floor21_conf & ~is_unmod_conf
    if unexpected.any():
        raise RuntimeError(f"{int(unexpected.sum())} language(s) with a confuser "
                           "have a row_treatment other than unmod/floor21; rule "
                           f"version {RULE_VERSION} does not define gap_solo for it")
    gap_solo[is_floor21_conf] = 0.0
    gap_solo[is_unmod_conf] = (floor_unmod[is_unmod_conf]
                               - floor_unmod[c_idx[is_unmod_conf]])
    floor_gap_shift = gap_mixed - gap_solo   # NaN for languages without a confuser

    # --- step 5: output CSV ---
    # floor_assigned is diagnostic only (see module docstring); for unmod
    # rows it is left empty (NaN) in the CSV so it cannot be misread as a
    # clamp instruction (unmod rows are not clamped to floor_unmod, they
    # carry their own unmodified weight row unchanged). The numeric
    # floor_unmod-valued floor_assigned above is used only internally, to
    # compute gap_mixed/floor_gap_shift.
    floor_assigned_csv = np.where(row_treatment == "unmod", np.nan, floor_assigned)

    out = pd.DataFrame({
        "lang": langs,
        "N": N,
        "row_treatment": row_treatment,
        "gate_on": gate_on,
        "rule_clause": rule_clause,
        "floor_unmod": floor_unmod,
        "floor_assigned": floor_assigned_csv,
        "seen_rescale": seen_rescale,
        "target_lang": target_lang_list,
        "floor_gap_shift": floor_gap_shift,
    })
    if len(out) != EXPECTED_N_LANG:
        raise RuntimeError(f"output has {len(out)} rows, expected "
                           f"{EXPECTED_N_LANG}")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    out.to_csv(out_csv, index=False)

    # --- step 6: summary ---
    n_confuser = int(has_confuser.sum())
    n_null_confuser = n_lang - n_confuser
    confuser_idx = np.where(has_confuser)[0]
    shift_valid = floor_gap_shift[has_confuser]
    median_shift = float(np.median(shift_valid))
    p5_shift = float(np.percentile(shift_valid, 5))
    n_large_shift = int((np.abs(shift_valid) > 1).sum())

    # The median above is 0.0 whenever a majority of confuser-having
    # languages have floor_gap_shift exactly 0.0 (both L and its confuser
    # assigned the same treatment, so the mixed and solo gaps coincide),
    # which hides the sign and concentration of the shift for the languages
    # where it is non-zero. Report that subset's distribution and which
    # groups it falls in.
    nonzero_mask = shift_valid != 0.0
    shift_nonzero = shift_valid[nonzero_mask]
    nonzero_idx = confuser_idx[nonzero_mask]
    n_nonzero = int(nonzero_mask.sum())

    print(f"RULE_VERSION {RULE_VERSION}: {n_head} of {n_lang} languages take "
          "clause head_unmod (row_treatment unmod, gate off); "
          f"{n_else} take clause else_floor21_gate (row_treatment floor21, "
          "gate on).")
    print(f"floor_gap_shift among the {n_confuser} languages with a same-script "
          f"higher-resource confuser: median {median_shift:.4f}, 5th percentile "
          f"{p5_shift:.4f}, {n_large_shift} language(s) with "
          "|floor_gap_shift| > 1.")
    if n_nonzero:
        q1, q2, q3 = np.percentile(shift_nonzero, [25, 50, 75])
        print(f"floor_gap_shift, non-zero subset only ({n_nonzero} of "
              f"{n_confuser} confuser-having languages; the other "
              f"{n_confuser - n_nonzero} are exactly 0.0): min "
              f"{float(shift_nonzero.min()):.4f}, 25th percentile {q1:.4f}, "
              f"median {q2:.4f}, 75th percentile {q3:.4f}, max "
              f"{float(shift_nonzero.max()):.4f}.")
        tail_ct = int((N[nonzero_idx] < 1_000).sum())
        lowmid_ct = int(((N[nonzero_idx] >= 1_000) & (N[nonzero_idx] < HEAD_N)).sum())
        magnet_ct = int((category[nonzero_idx] == "flat_magnet").sum())
        twin_ct = int((category[nonzero_idx] == "twin").sum())
        print("Group membership among the non-zero subset (not a partition; "
              f"a language may count in more than one group): lowmid "
              f"{lowmid_ct}, flat_magnet {magnet_ct}, twin {twin_ct}, tail "
              f"{tail_ct}.")
    else:
        print("floor_gap_shift: every confuser-having language has a shift "
              "of exactly 0.0.")
    print(f"{n_null_confuser} of {n_lang} languages have no confuser "
          "(target_lang null); floor_gap_shift is left empty for them.")
    print(f"Wrote {out_csv}")
    return out_csv


if __name__ == "__main__":
    run()
