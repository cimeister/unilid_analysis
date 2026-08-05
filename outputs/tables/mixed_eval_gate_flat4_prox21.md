# Judge-part evaluation for the per-language combined method

Pre-registration: EXPERIMENTS_PLAN.md, "Plan: per-language combined method", 2026-07-29 amendments. Confirmation instrument for configurations whose treatment came to attention through the derivation part of the seed-301 split; analysis/combined_evidence.py derives and records that split, this script only reads it. Requested configs: gate_flat4_prox21. Fixed comparator set: baseline, floor21, gt_min, gt_margin_adaptive.

## Gates passed

- y_true memmap shape: (45627279,), matches TOTAL_LINES (45,627,279).
- Draws 101 and 201 loaded from outputs/diagnostic/balanced_val, disjoint (188,061 and 185,204 lines, zero overlap).
- Full remainder (kept minus draws 101 and 201): 45,004,014 lines, matches EXPECTED_REMAINDER (45,004,014) from analysis.carried_set_comparison.
- Seed-301 split recomputed with fraction 0.4: derivation part 18,001,573 lines matches EXPECTED_DERIVATION (18,001,573); judge part 27,002,441 lines matches EXPECTED_JUDGE (27,002,441); the two sizes sum to the full remainder 45,004,014.
- Split record at /capstor/scratch/cscs/cmeister747/unilid_analysis/full_test_eval/rule_split_seed301.npz exists and matches the recomputed split exactly (this script never writes that file). sha256 of judge_idx.tobytes(): 5fbfd80b4a171d0d2db5407ba3e13703dfbbee9048eccbf08f999c356ed0e393.
- Wiring gate, full remainder: recomputed per-language F1 for all 7 Exp 38 carried configs matches outputs/diagnostic/carried_set_per_lang_f1.csv within max absolute difference 1.11e-16 (limit 1e-9). Per-config max difference: baseline 1.11e-16, freq_prior 1.11e-16, learned_bias 1.11e-16, floor21 1.11e-16, margin_q5 1.11e-16, margin_q5_head 1.11e-16, gt_margin_adaptive 1.11e-16.
- Veto-instrument support gate, judge part: median tail true-line support 11.0, minimum 2, both required to be at least MIN_COLLAPSE_SUPPORT (10) for the median.

## Input prediction memmaps

Every number below derives from these files as they existed at run time.

| config | sha256 (first 16) | bytes | mtime |
|---|---|---|---|
| baseline | 235380aa759b35fc | 91,254,686 | 2026-07-29 21:23:39 |
| freq_prior | 5b6e503210032d10 | 91,254,686 | 2026-07-29 21:23:39 |
| learned_bias | 74b80c8fb5be92e4 | 91,254,686 | 2026-07-29 21:23:39 |
| floor21 | 1922f9e73d9da3a2 | 91,254,686 | 2026-07-29 21:23:39 |
| margin_q5 | f2a42b91ea6942fe | 91,254,686 | 2026-07-29 21:23:39 |
| margin_q5_head | 74619bac8810f2ee | 91,254,686 | 2026-07-29 21:23:39 |
| gt_margin_adaptive | 2591a01cb8729336 | 91,254,686 | 2026-07-29 21:23:39 |
| gate_flat4_prox21 | 9b0ad2ccb670d836 | 91,254,686 | 2026-08-06 01:34:59 |
| gt_min | ea692557281d4044 | 91,254,686 | 2026-07-29 21:23:39 |
| floor21_gate | 76694dc34ddf7414 | 91,254,686 | 2026-07-30 09:06:39 |

Loaded only for the wiring gate above (full remainder, not judge-part evaluated in this report): freq_prior, learned_bias, margin_q5, margin_q5_head.
Loaded only as bootstrap anchors below (judge-part F1 computed for the contrasts, but not part of the requested plus comparator config set): floor21_gate.

## Judge-part group mean F1

Judge part of the seed-301 split (27,002,441 lines; verified against SPLIT_PATH above). Six groups, same names and masks as analysis.combined_evidence.

| group | n langs | gate_flat4_prox21 | baseline | floor21 | gt_min | gt_margin_adaptive |
|---|---|---|---|---|---|---|
| all 1,940 | 1940 | 0.9498 | 0.9117 | 0.9300 | 0.9111 | 0.9329 |
| tail (N<1k) | 96 | 0.7322 | 0.3319 | 0.6161 | 0.2883 | 0.4523 |
| lowmid (1k-18k) | 984 | 0.9620 | 0.9267 | 0.9352 | 0.9294 | 0.9568 |
| head (N>=18k) | 860 | 0.9601 | 0.9593 | 0.9590 | 0.9596 | 0.9592 |
| flat_magnet | 118 | 0.6521 | 0.2845 | 0.5197 | 0.2476 | 0.4128 |
| twin | 77 | 0.8912 | 0.8858 | 0.8850 | 0.8866 | 0.8909 |

## FPs into tail labels and flat_magnet labels, judge part of the seed-301 split

| config | FPs into tail labels | FPs into flat_magnet labels |
|---|---|---|
| gate_flat4_prox21 | 1,985 | 68,975 |
| baseline | 13,483 | 217,419 |
| floor21 | 5,425 | 188,903 |
| gt_min | 47,526 | 268,419 |
| gt_margin_adaptive | 17,332 | 153,894 |

For reference, the FPs-into-tail-labels numbers recorded on the FULL held-out remainder (outputs/tables/two_sided_selection.md, veto view table): baseline 22,404, gt_min 79,113. The judge part above is 27,002,441 of the remainder's 45,004,014 lines (about 60%), so these levels are not directly comparable to the table above; the comparison of interest is each configuration against baseline on the SAME judge part.

## Clause (B) and (C), judge part of the seed-301 split, vs baseline

Clause (B): overall global mean F1 on the judge part may not drop below baseline's by more than GUARD_TOL (0.01), and tail and flat_magnet global mean F1 may not drop below baseline's by more than PREC_TOL (0.0). Clause (C): languages with judge-part true support >= MIN_COLLAPSE_SUPPORT (10) whose F1 drops more than LANG_COLLAPSE_BOUND (0.1) below baseline's; verdict by outlier count: 0 clean, 1 to MAX_LANG_COLLAPSE_OUTLIERS (2) flagged with a required dig-in, more than 2 a class-level fail. Sub-support languages beyond the bound are reported informationally, never silently dropped.

### gate_flat4_prox21

- Clause (B): overall global mean F1 gain +0.0380 (must be >= -GUARD_TOL = -0.0100), tail global mean F1 gain +0.4002, flat_magnet global mean F1 gain +0.3675 (each must be >= -PREC_TOL = -0.0000) (PASS).
- Clause (C): 0 supported collapses, verdict **clean**.

## Draw-101 selection view (balanced-val within-stratum macro-F1)

Mirrors analysis.two_sided_report's selection view exactly (via the _within_stratum_row helper above, copied from that module's lines 74-77): 188,061 lines.

| config | overall | tail | magnets | twins | head |
|---|---|---|---|---|---|
| gate_flat4_prox21 | 0.9778 | 0.8813 | 0.8836 | 0.9414 | 0.9810 |
| baseline | 0.9811 | 0.9170 | 0.9174 | 0.9406 | 0.9814 |
| floor21 | 0.9800 | 0.8942 | 0.8978 | 0.9406 | 0.9813 |
| gt_min | 0.9841 | 0.9769 | 0.9688 | 0.9409 | 0.9811 |
| gt_margin_adaptive | 0.9798 | 0.9580 | 0.9503 | 0.9408 | 0.9808 |

## Track verdicts: draw-101 selection view plus judge-part veto

passes_two_sided(val_rows[c], val_rows['baseline'], f1_judge[c], f1_judge['baseline'], strata_masks, support_judge): veto computed on the judge part of the seed-301 split (not analysis.two_sided_report's own veto instrument, which is the full remainder minus draws 101 and 201). support_judge is the per-language true-line count on the judge part (27,002,441 lines). Clauses: (A) balanced-val overall/twins/head bounded by GUARD_TOL (0.01), tail/magnets by the symmetric widening rule (TAIL_RECALL_TOL=0.03 when the veto-instrument gain for that stratum exceeds its within-stratum loss, else GUARD_TOL); (B) judge-part veto overall bounded by GUARD_TOL, tail/magnet global mean F1 by PREC_TOL (0.0); (C) the collapse clause above, computed identically inside passes_two_sided. passes_uniform(val_rows[c], val_rows['baseline']): the uniform-prior track, balanced-val only.

- **gate_flat4_prox21: two-sided ELIGIBLE**
  uniform-prior track (passes_uniform): FAIL
- **floor21: two-sided ELIGIBLE**
  uniform-prior track (passes_uniform): FAIL
- **gt_min: two-sided REJECTED** (veto tail global mean F1 drops 0.0437; veto magnets global mean F1 drops 0.0370; 11 language(s) with support >= 10 lose more than 0.1 global F1 (worst 0.2036); more than 2 outliers is a class-level pattern)
  uniform-prior track (passes_uniform): PASS
- **gt_margin_adaptive: two-sided ELIGIBLE, flagged** (dig-in: ota_Arab)
  uniform-prior track (passes_uniform): FAIL

## Paired bootstrap, judge part of the seed-301 split

B=10,000, seed=0, percentile 95% interval, one resample index matrix over all 1,940 language positions (paired), shared across every contrast below. Point estimate = (config judge-part mean F1) minus (anchor judge-part mean F1); interval from the bootstrap distribution of the same difference. The self-contrast against an anchor equal to the requested config is skipped (trivially zero).

| config | anchor | mean diff | 95% CI |
|---|---|---|---|
| gate_flat4_prox21 | gt_margin_adaptive | +0.0168 | [+0.0129, +0.0209] |
| gate_flat4_prox21 | floor21_gate | +0.0018 | [+0.0010, +0.0026] |
