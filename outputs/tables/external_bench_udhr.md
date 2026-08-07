# External benchmark E2 (udhr): baseline / floor-21 / gated

Pre-registration: EXPERIMENTS_PLAN.md "Camera-ready evaluation program (2026-08-06)", E2; EXPERIMENTAL_SETUP.md "Camera-ready reporting conventions" (no-refitting transfer test: every constant below is the promoted configuration's own training-derived value, applied unchanged to this benchmark).

Benchmark: udhr, 24,115 rows, 366 labels (paper: 366). 0 rows empty after preprocess (scored as wrong under every configuration, the repo's EMPTY convention).

Acceptance gate: baseline macro F1 0.858977 vs paper 0.859 (diff 0.000023, tolerance 0.005) -- PASSED.

The floor-21 prediction reported here is the top-k pass's own rank-1 candidate under the floor-21 matrix (the same matrix the margins are scored from); analysis/gate_variants.py's own agree_mask carve-out is vacuous under that definition, and there is no separate best_of-scored floor-21 pass for this benchmark.

### udhr: macro F1 and macro FPR over the 366 benchmark labels, 24,115 rows

|   config | macro F1 | Ma-FPR (x1e5) | n_out_of_set |
|----------|----------|---------------|--------------|
| baseline |   0.8590 |       14.2934 |        2,624 |
|  floor21 |   0.8474 |       17.5195 |        2,480 |
|    gated |   0.8383 |       20.7576 |        2,322 |

### Re-examination accounting (gated configuration)

|          group | examined | moved | blocked_by_proximity | no_cand |
|----------------|----------|-------|----------------------|---------|
| A (N < HEAD_N) |    1,117 |   425 |                  322 |     370 |
|  B (flat-four) |       10 |    10 |                    0 |       0 |

## Constants used

- ACCEPTANCE_TOL = 0.005 (this module; pre-registration)
- HEAD_N = 18,000 (analysis.full_test_margin)
- RES_CAP = 100,000 (analysis.hierarchical_pool)
- D3_PROX = 21.0 (analysis.gate_variants)
- FLOOR_TARGET = -21.0 (analysis.full_test_floor21)
- TOPK_MARGIN = 5 (analysis.margin_diagnostic)
- outputs/diagnostic/tau_floor21_gate.csv (group A thresholds, 1,080 languages, sha256 8ad290e36409c085...)
- outputs/diagnostic/tau_flat4.csv (group B thresholds, 4 languages, sha256 9f85dd5f2bb5f9db...)

Per-label detail: outputs/diagnostic/external_bench/udhr_per_label.csv.

Git commit: a952831b28658ba566634c9481f08e4e4efc6d97.
