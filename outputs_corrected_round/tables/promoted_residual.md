# Camera-ready E4: promoted-configuration residual re-measurement

**NON-DEFAULT MODEL RUN.** Every number below was computed from `/capstor/scratch/cscs/cmeister747/unilid_analysis/corrected/glotlidc_corrected.unilid` and its predictions under `/capstor/scratch/cscs/cmeister747/unilid_analysis/full_test_eval_corrected`, not from the released model (`/capstor/scratch/cscs/cmeister747/unilid_analysis/glotlidc.unilid`), and must not be read as a restatement of the released model's tables. The "recorded" column below (EXPERIMENTS_RESULTS.md, measured 2026-07-30) is a RELEASED-model measurement, printed for comparison only; it gates nothing, and a difference against it here is a cross-model difference.

Pre-registration: EXPERIMENTS_PLAN.md, "Camera-ready evaluation program (2026-08-06)", E4 bullet. Instrument: judge part, 27,002,441 lines (analysis.combined_evidence's seed-301 rule split, /capstor/scratch/cscs/cmeister747/unilid_analysis/full_test_eval/rule_split_seed301.npz). HEAD_N = 18,000 training lines (analysis.full_test_margin). A prediction is "wrong" when it differs from the gold label; EMPTY (pred = -1, no specific predicted language) counts as wrong and is reported separately.

## n_wrong / head-true / head-head, per configuration

### Residual summary, judge part, 27,002,441 lines

|            config | n_wrong | n_empty (of n_wrong) | head-true share | head-head share (of head-true) |
|-------------------|---------|----------------------|-----------------|--------------------------------|
| gate_flat4_prox21 | 930,576 |                    0 |          0.9914 |                         0.8855 |
|      floor21_gate | 963,563 |                    0 |          0.9879 |                         0.8917 |


## floor21_gate vs the EXPERIMENTS_RESULTS.md recorded figures

EXPERIMENTS_RESULTS.md, "Current state (2026-08-06)", open item 3, measured 2026-07-30 (for the record; not a pass/fail gate, printed for comparison as pre-registered).

### floor21_gate recomputed vs recorded, judge part, 27,002,441 lines

|        quantity | recomputed | recorded |    diff |
|-----------------|------------|----------|---------|
|         n_wrong |    963,563 |  962,633 |    +930 |
| head-true share |     0.9879 |    0.987 | +0.0009 |
| head-head share |     0.8917 |    0.882 | +0.0097 |


## Top 20 confused (true, predicted) pairs, gate_flat4_prox21 (the promoted configuration)

### Top 20 confused pairs, gate_flat4_prox21, judge part, 27,002,441 lines

| true_lang | pred_lang | n_lines |  N_true |  N_pred |
|-----------|-----------|---------|---------|---------|
|  ind_Latn |  zsm_Latn |  31,105 | 100,000 | 100,000 |
|  arb_Arab |  ars_Arab |  22,184 | 100,000 |  18,539 |
|  hbs_Latn |  srp_Latn |  21,955 | 100,000 |  79,008 |
|  kin_Latn |  run_Latn |  19,850 | 100,000 | 100,000 |
|  arb_Arab |  arz_Arab |  18,911 | 100,000 | 100,000 |
|  eng_Latn |  enm_Latn |  18,430 | 100,000 |  34,177 |
|  dan_Latn |  nob_Latn |  18,349 | 100,000 | 100,000 |
|  fas_Arab |  glk_Arab |  16,741 | 100,000 |  22,263 |
|  cmn_Hani |  wuu_Hani |  15,976 | 100,000 |  74,364 |
|  hin_Deva |  anp_Deva |  12,887 | 100,000 |   4,499 |
|  nob_Latn |  nno_Latn |  12,675 | 100,000 | 100,000 |
|  fas_Arab |  mzn_Arab |  12,410 | 100,000 |  25,413 |
|  zsm_Latn |  ind_Latn |  11,768 | 100,000 | 100,000 |
|  spa_Latn |  ast_Latn |  11,335 | 100,000 | 100,000 |
|  rus_Cyrl |  ukr_Cyrl |  10,477 | 100,000 | 100,000 |
|  por_Latn |  glg_Latn |  10,444 | 100,000 | 100,000 |
|  run_Latn |  kin_Latn |  10,165 | 100,000 | 100,000 |
|  rus_Cyrl |  bul_Cyrl |   9,588 | 100,000 | 100,000 |
|  arb_Arab |  ary_Arab |   8,433 | 100,000 | 100,000 |
|  slv_Latn |  hbs_Latn |   7,986 | 100,000 | 100,000 |


## Top 20 confused (true, predicted) pairs, floor21_gate (for comparison)

### Top 20 confused pairs, floor21_gate, judge part, 27,002,441 lines

| true_lang | pred_lang | n_lines |  N_true |  N_pred |
|-----------|-----------|---------|---------|---------|
|  ind_Latn |  zsm_Latn |  30,106 | 100,000 | 100,000 |
|  eng_Latn |  sco_Latn |  26,411 | 100,000 |  87,458 |
|  arb_Arab |  ars_Arab |  22,184 | 100,000 |  18,539 |
|  hbs_Latn |  srp_Latn |  21,955 | 100,000 |  79,008 |
|  kin_Latn |  run_Latn |  19,848 | 100,000 | 100,000 |
|  arb_Arab |  arz_Arab |  18,950 | 100,000 | 100,000 |
|  eng_Latn |  enm_Latn |  18,430 | 100,000 |  34,177 |
|  dan_Latn |  nob_Latn |  18,323 | 100,000 | 100,000 |
|  fas_Arab |  glk_Arab |  16,741 | 100,000 |  22,263 |
|  cmn_Hani |  wuu_Hani |  15,976 | 100,000 |  74,364 |
|  hin_Deva |  anp_Deva |  12,887 | 100,000 |   4,499 |
|  nob_Latn |  nno_Latn |  12,637 | 100,000 | 100,000 |
|  fas_Arab |  mzn_Arab |  12,410 | 100,000 |  25,413 |
|  zsm_Latn |  ind_Latn |  11,626 | 100,000 | 100,000 |
|  spa_Latn |  ast_Latn |  10,637 | 100,000 | 100,000 |
|  rus_Cyrl |  ukr_Cyrl |  10,478 | 100,000 | 100,000 |
|  por_Latn |  glg_Latn |  10,242 | 100,000 | 100,000 |
|  run_Latn |  kin_Latn |  10,164 | 100,000 | 100,000 |
|  rus_Cyrl |  bul_Cyrl |   9,589 | 100,000 | 100,000 |
|  arb_Arab |  ary_Arab |   8,433 | 100,000 | 100,000 |


Git commit: dd9c570ad761ceba672aea5fa9ed57f492144c07.
