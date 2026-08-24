# External benchmark E2 (udhr): baseline / floor-21 / gated

Pre-registration: EXPERIMENTS_PLAN.md "Camera-ready evaluation program (2026-08-06)", E2; EXPERIMENTAL_SETUP.md "Camera-ready reporting conventions" (no-refitting transfer test: every constant below is the promoted configuration's own training-derived value, applied unchanged to this benchmark).

Benchmark: udhr, 24,115 rows, 366 labels (paper: 366). 0 rows empty after preprocess (scored as wrong under every configuration, the repo's EMPTY convention).

INFORMATIONAL, NOT A GATE: the recomputed udhr baseline macro F1 0.856046 agrees with the paper's published UniLID baseline cell 0.859 (absolute difference 0.002954, ACCEPTANCE_TOL 0.005). That published cell is a measurement of the RELEASED model and this run scored /capstor/scratch/cscs/cmeister747/unilid_analysis/corrected/glotlidc_corrected.unilid, so a difference here is an expected cross-model difference, not a regression and not a reproduction failure. Every configuration below WAS computed and written, with this run's own numbers.

The floor-21 prediction reported here is the top-k pass's own rank-1 candidate under the floor-21 matrix (the same matrix the margins are scored from); analysis/gate_variants.py's own agree_mask carve-out is vacuous under that definition, and there is no separate best_of-scored floor-21 pass for this benchmark.

### udhr: macro F1 and macro FPR over the 366 benchmark labels, 24,115 rows

|   config | macro F1 | Ma-FPR (x1e5) | n_out_of_set |
|----------|----------|---------------|--------------|
| baseline |   0.8560 |       15.2027 |        2,589 |
|  floor21 |   0.8512 |       16.8837 |        2,482 |
|    gated |   0.8419 |       20.2808 |        2,302 |

### Re-examination accounting (gated configuration)

|          group | examined | moved | blocked_by_proximity | no_cand |
|----------------|----------|-------|----------------------|---------|
| A (N < HEAD_N) |    1,169 |   454 |                  263 |     452 |
|  B (flat-four) |        7 |     7 |                    0 |       0 |

## Constants used

- ACCEPTANCE_TOL = 0.005 (this module; pre-registration)
- HEAD_N = 18,000 (analysis.full_test_margin)
- RES_CAP = 100,000 (analysis.hierarchical_pool)
- D3_PROX = 21.0 (analysis.gate_variants)
- FLOOR_TARGET = -17.0 (/capstor/scratch/cscs/cmeister747/unilid_analysis/full_test_eval_corrected/fingerprint_floor21.json)
- TOPK_MARGIN = 5 (analysis.margin_diagnostic)
- outputs_corrected_round/diagnostic/tau_floor21_gate.csv (group A thresholds, 1,080 languages, sha256 1139e7b1706160b7...)
- outputs_corrected_round/diagnostic/tau_flat4.csv (group B thresholds, 4 languages, sha256 0cd41df29ed6b2ae...)

Per-label detail: outputs_corrected_round/diagnostic/external_bench/udhr_per_label.csv.

Git commit: dd9c570ad761ceba672aea5fa9ed57f492144c07.
