# Camera-ready E1: common reporting set

Pre-registration: EXPERIMENTS_PLAN.md, "Camera-ready evaluation program (2026-08-06)", E1. Configurations: baseline, gate_flat4_prox21, fasttext. Instruments follow EXPERIMENTAL_SETUP.md, "Camera-ready reporting conventions": Table 1 cells use the full kept pool, 45,377,279 lines; the appendix comparison uses the judge part, 27,002,441 lines. No delta pairs terms from different line sets.

## Gates passed

- Language order: _load_model_data's 1,940-language list matches the lang column of outputs/diagnostic/full_test_per_lang_prf.csv.
- y_true memmap shape (45627279,) matches TOTAL_LINES (45,627,279); full kept pool, 45,377,279 lines matches EXPECTED_KEPT.
- Full remainder (kept minus draws 101 and 201): 45,004,014 lines, matches EXPECTED_REMAINDER (45,004,014).
- Seed-301 split recomputed with fraction 0.4: derivation part 18,001,573 lines (EXPECTED_DERIVATION 18,001,573), judge part 27,002,441 lines (EXPECTED_JUDGE 27,002,441); matches the split recorded at /capstor/scratch/cscs/cmeister747/unilid_analysis/full_test_eval/rule_split_seed301.npz.
- Wiring gate A (full-remainder per-language F1 vs outputs/diagnostic/carried_set_per_lang_f1.csv): max absolute difference 1.11e-16 over 7 carried configs (tol 1e-09). Per-config: baseline 1.11e-16, freq_prior 1.11e-16, learned_bias 1.11e-16, floor21 1.11e-16, margin_q5 1.11e-16, margin_q5_head 1.11e-16, gt_margin_adaptive 1.11e-16.
- Wiring gate B (judge-part mean F1 vs outputs/diagnostic/mixed_eval_judge_f1_gate_flat4_prox21.csv, tol 1e-09): baseline recomputed 0.911731 vs csv 0.911700 (diff 1.11e-16); gate_flat4_prox21 recomputed 0.949763 vs csv 0.949800 (diff 1.11e-16).
- Wiring gate C (full-pool baseline reproduces the Exp 16 recorded values): macro F1 0.929190 vs recorded 0.9292 (diff 9.95e-06, tol 5e-05); macro FPR 0.00002026 vs recorded 2.0263e-05 (diff 1.62e-10, tol 1e-08).

Sentinel guard: zero UNSEEN/EXCLUDED values on the kept pool for every loaded memmap. EMPTY (-1) counts on the kept pool: baseline 0, freq_prior 0, learned_bias 0, floor21 0, margin_q5 0, margin_q5_head 0, gt_margin_adaptive 0, gate_flat4_prox21 0, fasttext 0.

Note for the camera-ready table: the submission's Table 1 rows state N = 45,627,279; the cells here use the full kept pool, 45,377,279 lines, so every restated row in the camera-ready table must carry N = 45,377,279 and the N column must be updated for all rows, not only the new one.

### Macro F1 and macro FPR (x1e5), full kept pool, 45,377,279 lines

|            config | macro F1 | Ma-FPR (x1e5) |
|-------------------|----------|---------------|
|          baseline |   0.9292 |        2.0263 |
| gate_flat4_prox21 |   0.9569 |        1.7665 |
|          fasttext |   0.9443 |        2.7063 |

### Macro F1 and macro FPR (x1e5), judge part, 27,002,441 lines

|            config | macro F1 | Ma-FPR (x1e5) |
|-------------------|----------|---------------|
|          baseline |   0.9117 |        2.0373 |
| gate_flat4_prox21 |   0.9498 |        1.7743 |
|          fasttext |   0.9332 |        2.7165 |

## Paired bootstrap, judge part, 27,002,441 lines

B=10,000, seed=0, percentile 95% interval, paired resample over the 1,940 language positions, one shared resample matrix reused across contrasts. Point estimate = (gate_flat4_prox21 judge-part mean F1) minus (comparator judge-part mean F1).

### gate_flat4_prox21 minus comparator, judge part, 27,002,441 lines

| comparator | mean diff |             95% CI |
|------------|-----------|--------------------|
|   baseline |   +0.0380 | [+0.0328, +0.0434] |
|   fasttext |   +0.0166 | [+0.0112, +0.0223] |

## Input prediction memmaps

| config | sha256 (first 16) | bytes |
|---|---|---|
| baseline | 235380aa759b35fc | 91,254,686 |
| freq_prior | 5b6e503210032d10 | 91,254,686 |
| learned_bias | 74b80c8fb5be92e4 | 91,254,686 |
| floor21 | 1922f9e73d9da3a2 | 91,254,686 |
| margin_q5 | f2a42b91ea6942fe | 91,254,686 |
| margin_q5_head | 74619bac8810f2ee | 91,254,686 |
| gt_margin_adaptive | 2591a01cb8729336 | 91,254,686 |
| gate_flat4_prox21 | 9b0ad2ccb670d836 | 91,254,686 |
| fasttext | 4ff74fb55ce5668b | 91,254,686 |

Loaded only for wiring gates A/B above (not part of the requested config set): freq_prior, learned_bias, floor21, margin_q5, margin_q5_head, gt_margin_adaptive.

Per-language detail: outputs/diagnostic/paper_eval_per_lang_f1_fullpool.csv (full kept pool, 45,377,279 lines), outputs/diagnostic/paper_eval_per_lang_f1_judge.csv (judge part, 27,002,441 lines).

Git commit: 02a346e74b1a907cd30eeb0deb8d185190bc4be4.
