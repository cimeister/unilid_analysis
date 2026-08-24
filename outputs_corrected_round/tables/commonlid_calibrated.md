# E5: CommonLID for the camera-ready (baseline / floor-21 / gated)

Pre-registration: EXPERIMENTS_PLAN.md "Camera-ready evaluation program" (2026-08-07), E5. Gate: gate_flat4_prox21 (the promoted configuration), thresholds unchanged, applied via the shared _gate_walk_and_merge (the E2-reviewed code path, analysis/external_bench_eval.py, imported here, not reimplemented).

373,230 rows, 0 rows empty after preprocess (scored as wrong under every configuration, the repo's EMPTY convention); 0 rows with fewer than 5 saved candidates.

Out-of-tag-set convention: a prediction mapping to a label outside the gold tag set contributes no per-tag F1 term of its own and deflates its gold tag's recall. Rows with an out-of-set label (distinct labels in parentheses): baseline 32,525 (1095), floor21 31,309 (1065), gated 25,994 (810).

### CommonLID, 373,230 lines, 109 tags, web domain; macrolanguage-aware mapping per the recorded convention

|   config | macro-aware accuracy | tag-level macro F1 |
|----------|----------------------|--------------------|
| baseline |               0.8476 |             0.7218 |
|  floor21 |               0.8512 |             0.7203 |
|    gated |               0.8624 |             0.7171 |

### Re-examination accounting (gate_flat4_prox21, the promoted configuration's own gate, thresholds unchanged)

|          group | examined | moved | blocked_by_proximity | no_cand |
|----------------|----------|-------|----------------------|---------|
| A (N < HEAD_N) |    9,094 | 7,776 |                  149 |   1,169 |
|  B (flat-four) |    3,999 | 3,862 |                   11 |     126 |

## Constants used

- FLOOR_TARGET = -17.0 (/capstor/scratch/cscs/cmeister747/unilid_analysis/full_test_eval_corrected/fingerprint_floor21.json)
- TOPK_MARGIN = 5 (analysis.margin_diagnostic)
- HEAD_N = 18,000 (analysis.full_test_margin)
- RES_CAP = 100,000 (analysis.hierarchical_pool)
- D3_PROX = 21.0 (analysis.gate_variants)
- EVAL_GATE_TOL = 0.0005 (this module; E5 pre-registration)
- outputs_corrected_round/diagnostic/tau_floor21_gate.csv (group A thresholds, 1,080 languages, sha256 1139e7b1706160b7...)
- outputs_corrected_round/diagnostic/tau_flat4.csv (group B thresholds, 4 languages, sha256 0cd41df29ed6b2ae...)

## Recorded-value comparisons (informational)

INFORMATIONAL, NOT A GATE: the four recorded values below are measurements of the RELEASED model and this run scored /capstor/scratch/cscs/cmeister747/unilid_analysis/corrected/glotlidc_corrected.unilid, so a difference is an expected cross-model difference, not a regression and not a reproduction failure. Every configuration in the table above WAS computed and written, with this run's own numbers.
- baseline macro-aware accuracy: 0.8476 differs from the recorded 0.8452 (analysis.commonlid_carried.EXPECTED_BASELINE_ACC, Exp 12/39), difference +0.0024, EVAL_GATE_TOL 0.0005.
- baseline tag-level macro F1: 0.7218 differs from the recorded 0.7228 (analysis.commonlid_carried.EXPECTED_BASELINE_TAG_F1, Exp 12/39), difference -0.0010, EVAL_GATE_TOL 0.0005.
- floor-21 tag-level macro F1: 0.7203 differs from the recorded 0.7181 (Exp 39, outputs/tables/commonlid_carried.md, "floor21" row), difference +0.0022, EVAL_GATE_TOL 0.0005.
- floor-21 macro-aware accuracy: 0.8512 differs from the recorded 0.8491 (Exp 39, outputs/tables/commonlid_carried.md, "floor21" row), difference +0.0021, EVAL_GATE_TOL 0.0005.

Per-tag detail: outputs_corrected_round/diagnostic/commonlid_calibrated_per_tag.csv.

Git commit: dd9c570ad761ceba672aea5fa9ed57f492144c07.
