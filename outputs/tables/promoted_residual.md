# Camera-ready E4: promoted-configuration residual re-measurement

Pre-registration: EXPERIMENTS_PLAN.md, "Camera-ready evaluation program (2026-08-06)", E4 bullet. Instrument: judge part, 27,002,441 lines (analysis.combined_evidence's seed-301 rule split, /capstor/scratch/cscs/cmeister747/unilid_analysis/full_test_eval/rule_split_seed301.npz). HEAD_N = 18,000 training lines (analysis.full_test_margin). A prediction is "wrong" when it differs from the gold label; EMPTY (pred = -1, no specific predicted language) counts as wrong and is reported separately.

## n_wrong / head-true / head-head, per configuration

### Residual summary, judge part, 27,002,441 lines

|            config | n_wrong | n_empty (of n_wrong) | head-true share | head-head share (of head-true) |
|-------------------|---------|----------------------|-----------------|--------------------------------|
| gate_flat4_prox21 | 926,299 |                    0 |          0.9915 |                         0.8864 |
|      floor21_gate | 962,633 |                    0 |          0.9873 |                         0.8929 |


## floor21_gate vs the EXPERIMENTS_RESULTS.md recorded figures

EXPERIMENTS_RESULTS.md, "Current state (2026-08-06)", open item 3, measured 2026-07-30 (for the record; not a pass/fail gate, printed for comparison as pre-registered).

### floor21_gate recomputed vs recorded, judge part, 27,002,441 lines

|        quantity | recomputed | recorded |    diff |
|-----------------|------------|----------|---------|
|         n_wrong |    962,633 |  962,633 |      +0 |
| head-true share |     0.9873 |    0.987 | +0.0003 |
| head-head share |     0.8929 |    0.882 | +0.0109 |


## Top 20 confused (true, predicted) pairs, gate_flat4_prox21 (the promoted configuration)

### Top 20 confused pairs, gate_flat4_prox21, judge part, 27,002,441 lines

| true_lang | pred_lang | n_lines |  N_true |  N_pred |
|-----------|-----------|---------|---------|---------|
|  ind_Latn |  zsm_Latn |  31,113 | 100,000 | 100,000 |
|  arb_Arab |  ars_Arab |  23,608 | 100,000 |  18,539 |
|  hbs_Latn |  srp_Latn |  21,395 | 100,000 |  79,008 |
|  kin_Latn |  run_Latn |  20,167 | 100,000 | 100,000 |
|  fas_Arab |  glk_Arab |  19,447 | 100,000 |  22,263 |
|  arb_Arab |  arz_Arab |  19,212 | 100,000 | 100,000 |
|  dan_Latn |  nob_Latn |  18,632 | 100,000 | 100,000 |
|  eng_Latn |  enm_Latn |  16,367 | 100,000 |  34,177 |
|  cmn_Hani |  wuu_Hani |  15,651 | 100,000 |  74,364 |
|  nob_Latn |  nno_Latn |  13,261 | 100,000 | 100,000 |
|  hin_Deva |  anp_Deva |  12,680 | 100,000 |   4,499 |
|  fas_Arab |  mzn_Arab |  12,304 | 100,000 |  25,413 |
|  zsm_Latn |  ind_Latn |  11,603 | 100,000 | 100,000 |
|  spa_Latn |  ast_Latn |  10,940 | 100,000 | 100,000 |
|  por_Latn |  glg_Latn |  10,658 | 100,000 | 100,000 |
|  run_Latn |  kin_Latn |  10,154 | 100,000 | 100,000 |
|  rus_Cyrl |  bul_Cyrl |   9,286 | 100,000 | 100,000 |
|  arb_Arab |  ary_Arab |   9,028 | 100,000 | 100,000 |
|  rus_Cyrl |  ukr_Cyrl |   8,947 | 100,000 | 100,000 |
|  slv_Latn |  hbs_Latn |   7,932 | 100,000 | 100,000 |


## Top 20 confused (true, predicted) pairs, floor21_gate (for comparison)

### Top 20 confused pairs, floor21_gate, judge part, 27,002,441 lines

| true_lang | pred_lang | n_lines |  N_true |  N_pred |
|-----------|-----------|---------|---------|---------|
|  ind_Latn |  zsm_Latn |  30,076 | 100,000 | 100,000 |
|  eng_Latn |  sco_Latn |  29,779 | 100,000 |  87,458 |
|  arb_Arab |  ars_Arab |  23,608 | 100,000 |  18,539 |
|  hbs_Latn |  srp_Latn |  21,395 | 100,000 |  79,008 |
|  kin_Latn |  run_Latn |  20,166 | 100,000 | 100,000 |
|  fas_Arab |  glk_Arab |  19,447 | 100,000 |  22,263 |
|  arb_Arab |  arz_Arab |  19,255 | 100,000 | 100,000 |
|  dan_Latn |  nob_Latn |  18,607 | 100,000 | 100,000 |
|  eng_Latn |  enm_Latn |  16,367 | 100,000 |  34,177 |
|  cmn_Hani |  wuu_Hani |  15,651 | 100,000 |  74,364 |
|  nob_Latn |  nno_Latn |  13,227 | 100,000 | 100,000 |
|  hin_Deva |  anp_Deva |  12,678 | 100,000 |   4,499 |
|  fas_Arab |  mzn_Arab |  12,304 | 100,000 |  25,413 |
|  zsm_Latn |  ind_Latn |  11,433 | 100,000 | 100,000 |
|  por_Latn |  glg_Latn |  10,470 | 100,000 | 100,000 |
|  spa_Latn |  ast_Latn |  10,294 | 100,000 | 100,000 |
|  run_Latn |  kin_Latn |  10,153 | 100,000 | 100,000 |
|  rus_Cyrl |  bul_Cyrl |   9,288 | 100,000 | 100,000 |
|  arb_Arab |  ary_Arab |   9,028 | 100,000 | 100,000 |
|  rus_Cyrl |  ukr_Cyrl |   8,948 | 100,000 | 100,000 |


Git commit: 9b1ed200c433589ede4b982ac76b0cd6e8b07157.
