# Step-1b evidence base for the per-language combined method

Pre-registration: EXPERIMENTS_PLAN.md, "Plan: per-language combined method", 2026-07-29 amendments. This report covers the derivation part of the seed-301 split. Judge-part predictions are not read by this script; judge-part evaluation of the combined method happens in the later mixed_eval step. The feature-provenance audit (plan step 0) is at outputs/tables/combined_feature_provenance.md.

## Gates passed

- y_true memmap shape: (45627279,), matches TOTAL_LINES (45,627,279).
- Draws 101 and 201 loaded from outputs/diagnostic/balanced_val, disjoint (188,061 and 185,204 lines, zero overlap).
- Full remainder (kept minus draws 101 and 201): 45,004,014 lines, matches EXPECTED_REMAINDER (45,004,014) from analysis.carried_set_comparison.
- Wiring gate 1, full remainder: recomputed per-language F1 for all 7 Exp 38 carried configs matches outputs/diagnostic/carried_set_per_lang_f1.csv within max absolute difference 1.11e-16 (limit 1e-9). Per-config max difference: baseline 1.11e-16, freq_prior 1.11e-16, learned_bias 1.11e-16, floor21 1.11e-16, margin_q5 1.11e-16, margin_q5_head 1.11e-16, gt_margin_adaptive 1.11e-16.
- Wiring gate 2, gt_min veto-instrument anchor, full remainder: recorded in outputs/tables/two_sided_selection.md, gt_min row of the veto view table, tail global mean F1 0.295 and FPs into tail labels 79,113. Recomputed here: tail global mean F1 0.2950 (difference 3.23e-05, limit 6e-05), FPs into tail labels 79,113 (exact match required).
- Split size gates: derivation part 18,001,573 lines matches EXPECTED_DERIVATION (18,001,573); judge part 27,002,441 lines matches EXPECTED_JUDGE (27,002,441); the two sizes sum to the full remainder 45,004,014.

## Input prediction memmaps

Every number below derives from these files as they existed at run time. pred_unmod_gate.npy and pred_floor21_gate.npy are built by analysis/solo_gates.py; their build reports are outputs/tables/unmod_gate_build.md and outputs/tables/floor21_gate_build.md.

| config | sha256 (first 16) | bytes | mtime |
|---|---|---|---|
| baseline | 235380aa759b35fc | 91,254,686 | 2026-07-29 21:23:39 |
| unmod_gate | 5ec9e200a19c6948 | 91,254,686 | 2026-07-30 09:08:14 |
| floor21 | 1922f9e73d9da3a2 | 91,254,686 | 2026-07-29 21:23:39 |
| floor21_gate | 76694dc34ddf7414 | 91,254,686 | 2026-07-30 09:06:39 |
| gt_min | ea692557281d4044 | 91,254,686 | 2026-07-29 21:23:39 |
| gt_margin_adaptive | 2591a01cb8729336 | 91,254,686 | 2026-07-29 21:23:39 |
| freq_prior | 5b6e503210032d10 | 91,254,686 | 2026-07-29 21:23:39 |
| learned_bias | 74b80c8fb5be92e4 | 91,254,686 | 2026-07-29 21:23:39 |
| margin_q5 | f2a42b91ea6942fe | 91,254,686 | 2026-07-29 21:23:39 |
| margin_q5_head | 74619bac8810f2ee | 91,254,686 | 2026-07-29 21:23:39 |

## Split record

Seed 301, fraction 0.4 of the full remainder assigned to the derivation part. Derivation part: 18,001,573 lines. Judge part: 27,002,441 lines. sha256 of derive_idx.tobytes(): 95db5ceb106087dc89e134a89c11b41fd73dfe1d28c690e2ef432061adc51e25. Stored at /capstor/scratch/cscs/cmeister747/unilid_analysis/full_test_eval/rule_split_seed301.npz (scratch, not the repo).

## Support report

Per-language true-line counts (bincount of y_true), tail languages only (N < 1,000, 96 languages).
- Derivation part of the seed-301 split (18,001,573 lines): median tail support 8.0, minimum 0, 37 of 96 tail languages at support >= 10.
- Judge part of the seed-301 split (27,002,441 lines): median tail support 11.0, minimum 2, 52 of 96 tail languages at support >= 10.
- flat_magnet languages (118 total), derivation part of the seed-301 split, at support >= 10: 60.

Tail languages with derivation-part support below 10, 59 of 96: each of their per-language F1 values on the derivation part rests on at most 9 true lines (1 of them have support 0, where F1 is 0 for every configuration). They still enter the group means, the oracle, and the unique-best counts above. Per the pre-registration, their per-language values carry no leader evidence for the rule; their treatment assignment comes from the training-side assignment rule alone. abq_Cyrl (n=1), aqz_Latn (n=6), arc_Syrc (n=7), asm_Latn (n=5), bor_Latn (n=3), brx_Latn (n=8), bwi_Latn (n=6), csw_Cans (n=2), csw_Latn (n=4), cwd_Cans (n=2), diu_Latn (n=6), drg_Latn (n=5), eto_Latn (n=3), goh_Latn (n=3), hoc_Latn (n=9), kak_Latn (n=6), kap_Cyrl (n=8), kas_Latn (n=9), kdr_Latn (n=1), kei_Latn (n=4), knx_Latn (n=5), ktz_Latn (n=6), lad_Hebr (n=4), lbe_Cyrl (n=0), ldn_Latn (n=3), lez_Cyrl (n=7), lki_Arab (n=5), lrc_Arab (n=3), max_Latn (n=8), mev_Latn (n=8), mgm_Latn (n=3), mhw_Latn (n=7), mnw_Mymr (n=5), mrv_Latn (n=2), mus_Latn (n=5), mwf_Latn (n=6), nio_Cyrl (n=1), nla_Latn (n=6), non_Latn (n=9), npi_Latn (n=8), otw_Latn (n=1), pmq_Latn (n=7), rme_Latn (n=8), sbs_Latn (n=4), sby_Latn (n=4), sdc_Latn (n=8), sdo_Latn (n=9), sel_Cyrl (n=1), sju_Latn (n=3), thv_Latn (n=2), tpn_Latn (n=9), tpw_Latn (n=5), tuv_Latn (n=5), tzl_Latn (n=6), vot_Latn (n=3), xum_Latn (n=2), yup_Latn (n=5), zpj_Latn (n=1), zsm_Arab (n=6)

## Per-group mean F1, CODOMAIN configs, derivation part of the seed-301 split

| group | n langs | baseline | unmod_gate | floor21 | floor21_gate | gt_min | gt_margin_adaptive | leader minus runner-up 95% CI |
|---|---|---|---|---|---|---|---|---|
| all 1,940 | 1940 | 0.9115 | 0.9327 | 0.9303 | 0.9478 | 0.9118 | 0.9339 | [0.0100, 0.0178] |
| tail (N<1k) | 96 | 0.3278 | 0.4605 | 0.6230 | 0.7330 | 0.3023 | 0.4753 | [0.0734, 0.1480] |
| lowmid (1k-18k) | 984 | 0.9265 | 0.9558 | 0.9351 | 0.9593 | 0.9294 | 0.9566 | [0.0014, 0.0042] |
| head (N>=18k) | 860 | 0.9594 | 0.9590 | 0.9590 | 0.9586 | 0.9596 | 0.9593 | [0.0000, 0.0005] |
| flat_magnet | 118 | 0.2799 | 0.4239 | 0.5270 | 0.6434 | 0.2597 | 0.4310 | [0.0864, 0.1488] |
| twin | 77 | 0.8855 | 0.8915 | 0.8860 | 0.8921 | 0.8884 | 0.8929 | [-0.0003, 0.0021] |

Bootstrap: paired resampling of the group's languages with replacement, 10,000 resamples, seed 0, one shared resample index matrix per group reused across all configurations, 95% percentile interval of (leader mean F1 minus runner-up mean F1), computed on the derivation part of the seed-301 split. The leader and runner-up were chosen by the point estimates on the same data, so this interval is biased away from zero and is descriptive, not a test; the fixed contrasts in the next section do not have this problem.

## Leader table, derivation part of the seed-301 split

| group | leader | leader mean F1 | runner-up | runner-up mean F1 | delta | 95% CI |
|---|---|---|---|---|---|---|
| all 1,940 | floor21_gate | 0.9478 | gt_margin_adaptive | 0.9339 | 0.0139 | [0.0100, 0.0178] |
| tail (N<1k) | floor21_gate | 0.7330 | floor21 | 0.6230 | 0.1100 | [0.0734, 0.1480] |
| lowmid (1k-18k) | floor21_gate | 0.9593 | gt_margin_adaptive | 0.9566 | 0.0027 | [0.0014, 0.0042] |
| head (N>=18k) | gt_min | 0.9596 | baseline | 0.9594 | 0.0003 | [0.0000, 0.0005] |
| flat_magnet | floor21_gate | 0.6434 | floor21 | 0.5270 | 0.1164 | [0.0864, 0.1488] |
| twin | gt_margin_adaptive | 0.8929 | floor21_gate | 0.8921 | 0.0008 | [-0.0003, 0.0021] |

## Fixed contrasts against gt_margin_adaptive, derivation part of the seed-301 split

The anchor configuration gt_margin_adaptive is fixed in advance (the configuration to beat on the primary quantity, 0.9334 on the full remainder), so these intervals are not post-selection. Each cell is (config mean F1 minus gt_margin_adaptive mean F1) with its paired 95% interval from the same per-group resample matrix. Positive means the configuration is ahead of gt_margin_adaptive on the derivation part.

| group | baseline | unmod_gate | floor21 | floor21_gate | gt_min |
|---|---|---|---|---|---|
| all 1,940 | -0.0225 [-0.0265, -0.0186] | -0.0012 [-0.0033, +0.0008] | -0.0037 [-0.0079, +0.0006] | +0.0139 [+0.0100, +0.0178] | -0.0222 [-0.0258, -0.0186] |
| tail (N<1k) | -0.1475 [-0.1978, -0.0976] | -0.0148 [-0.0554, +0.0229] | +0.1478 [+0.0820, +0.2164] | +0.2577 [+0.1973, +0.3175] | -0.1730 [-0.2115, -0.1363] |
| lowmid (1k-18k) | -0.0300 [-0.0354, -0.0250] | -0.0007 [-0.0017, +0.0003] | -0.0214 [-0.0260, -0.0172] | +0.0027 [+0.0014, +0.0042] | -0.0271 [-0.0322, -0.0223] |
| head (N>=18k) | +0.0001 [-0.0003, +0.0004] | -0.0003 [-0.0005, -0.0000] | -0.0002 [-0.0006, +0.0001] | -0.0007 [-0.0010, -0.0004] | +0.0004 [+0.0001, +0.0006] |
| flat_magnet | -0.1511 [-0.1950, -0.1076] | -0.0071 [-0.0402, +0.0228] | +0.0960 [+0.0362, +0.1565] | +0.2124 [+0.1608, +0.2650] | -0.1713 [-0.2045, -0.1389] |
| twin | -0.0074 [-0.0167, -0.0004] | -0.0014 [-0.0041, +0.0006] | -0.0069 [-0.0156, -0.0007] | -0.0008 [-0.0021, +0.0003] | -0.0045 [-0.0110, +0.0003] |

## Codomain oracle, derivation part of the seed-301 split

Per-language maximum F1 over the six CODOMAIN configurations, then averaged unweighted over each group.

| group | oracle mean F1 | best single config | best single mean F1 | oracle gain |
|---|---|---|---|---|
| all 1,940 | 0.9514 | floor21_gate | 0.9478 | 0.0036 |
| tail (N<1k) | 0.7566 | floor21_gate | 0.7330 | 0.0236 |
| lowmid (1k-18k) | 0.9624 | floor21_gate | 0.9593 | 0.0031 |
| head (N>=18k) | 0.9605 | gt_min | 0.9596 | 0.0009 |
| flat_magnet | 0.6617 | floor21_gate | 0.6434 | 0.0183 |
| twin | 0.8953 | gt_margin_adaptive | 0.8929 | 0.0024 |

## Full carried-set oracle, derivation part of the seed-301 split

Per-language maximum F1 over the seven Exp 38 carried configurations (the committed step 1(b) asked for this oracle alongside the modular one). It includes freq_prior and learned_bias, which the combined method's codomain excludes, so it is an upper reference, not the design's ceiling.

| group | carried oracle mean F1 | best carried config | best carried mean F1 | oracle gain |
|---|---|---|---|---|
| all 1,940 | 0.9526 | gt_margin_adaptive | 0.9339 | 0.0187 |
| tail (N<1k) | 0.7073 | floor21 | 0.6230 | 0.0843 |
| lowmid (1k-18k) | 0.9610 | gt_margin_adaptive | 0.9566 | 0.0045 |
| head (N>=18k) | 0.9705 | learned_bias | 0.9696 | 0.0009 |
| flat_magnet | 0.6328 | floor21 | 0.5270 | 0.1058 |
| twin | 0.9128 | learned_bias | 0.9081 | 0.0047 |

## Descriptive unique-best counts over CODOMAIN, derivation part of the seed-301 split

Not a rule input. For each of the 1,940 languages, this counts which CODOMAIN configuration reaches the highest per-language F1 on the derivation part. 1135 of 1940 languages have ties at the maximum; the strict counts below exclude tied languages, counting a configuration only where it is the unique best.

baseline: 113 strict, unmod_gate: 216 strict, floor21: 77 strict, floor21_gate: 226 strict, gt_min: 67 strict, gt_margin_adaptive: 106 strict

## REFERENCE configurations

Per-language F1 for the four REFERENCE configurations (freq_prior, learned_bias, margin_q5, margin_q5_head), derivation part of the seed-301 split, is in outputs/diagnostic/combined_evidence_derivation_f1.csv for context. They are excluded from the group, leader, oracle, and unique-best sections above, which are restricted to CODOMAIN, the combined method's modular treatment set (EXPERIMENTS_PLAN.md, "Treatment set (modularity-preserving only)").
