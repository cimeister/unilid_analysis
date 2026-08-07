# E5: CommonLID for the camera-ready (baseline / floor-21 / gated)

Pre-registration: EXPERIMENTS_PLAN.md "Camera-ready evaluation program" (2026-08-07), E5. Gate: gate_flat4_prox21 (the promoted configuration), thresholds unchanged, applied via the shared _gate_walk_and_merge (the E2-reviewed code path, analysis/external_bench_eval.py, imported here, not reimplemented).

373,230 rows, 0 rows empty after preprocess (scored as wrong under every configuration, the repo's EMPTY convention); 0 rows with fewer than 5 saved candidates.

Out-of-tag-set convention: a prediction mapping to a label outside the gold tag set contributes no per-tag F1 term of its own and deflates its gold tag's recall. Rows with an out-of-set label (distinct labels in parentheses): baseline 32,901 (1089), floor21 31,280 (1033), gated 25,884 (782).

### CommonLID, 373,230 lines, 109 tags, web domain; macrolanguage-aware mapping per the recorded convention

|   config | macro-aware accuracy | tag-level macro F1 |
|----------|----------------------|--------------------|
| baseline |               0.8452 |             0.7228 |
|  floor21 |               0.8491 |             0.7181 |
|    gated |               0.8604 |             0.7149 |

### Re-examination accounting (gate_flat4_prox21, the promoted configuration's own gate, thresholds unchanged)

|          group | examined | moved | blocked_by_proximity | no_cand |
|----------------|----------|-------|----------------------|---------|
| A (N < HEAD_N) |    9,086 | 7,844 |                  171 |   1,071 |
|  B (flat-four) |    3,971 | 3,844 |                   12 |     115 |

## Constants used

- FLOOR_TARGET = -21.0 (analysis.full_test_floor21)
- TOPK_MARGIN = 5 (analysis.margin_diagnostic)
- HEAD_N = 18,000 (analysis.full_test_margin)
- RES_CAP = 100,000 (analysis.hierarchical_pool)
- D3_PROX = 21.0 (analysis.gate_variants)
- EVAL_GATE_TOL = 0.0005 (this module; E5 pre-registration)
- outputs/diagnostic/tau_floor21_gate.csv (group A thresholds, 1,080 languages, sha256 8ad290e36409c085...)
- outputs/diagnostic/tau_flat4.csv (group B thresholds, 4 languages, sha256 9f85dd5f2bb5f9db...)

## Wiring gates (reproduced before any new number was written)

- baseline macro-aware accuracy: 0.8452 vs recorded 0.8452 (analysis.commonlid_carried.EXPECTED_BASELINE_ACC, Exp 12/39), within 0.0005.
- baseline tag-level macro F1: 0.7228 vs recorded 0.7228 (analysis.commonlid_carried.EXPECTED_BASELINE_TAG_F1, Exp 12/39), within 0.0005.
- floor-21 tag-level macro F1: 0.7181 vs recorded 0.7181 (Exp 39, outputs/tables/commonlid_carried.md, "floor21" row), within 0.0005.

Per-tag detail: outputs/diagnostic/commonlid_calibrated_per_tag.csv.

Git commit: 453acf7688b7b8c7673cf723199641e0909b0ed7.
