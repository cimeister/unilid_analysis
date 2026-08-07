# External benchmark E2 (flores): baseline / floor-21 / gated

Pre-registration: EXPERIMENTS_PLAN.md "Camera-ready evaluation program (2026-08-06)", E2; EXPERIMENTAL_SETUP.md "Camera-ready reporting conventions" (no-refitting transfer test: every constant below is the promoted configuration's own training-derived value, applied unchanged to this benchmark).

Benchmark: flores, 192,280 rows, 190 labels (paper: 190). 0 rows empty after preprocess (scored as wrong under every configuration, the repo's EMPTY convention).

Acceptance gate: baseline macro F1 0.931741 vs paper 0.932 (diff 0.000259, tolerance 0.005) -- PASSED.

The floor-21 prediction reported here is the top-k pass's own rank-1 candidate under the floor-21 matrix (the same matrix the margins are scored from); analysis/gate_variants.py's own agree_mask carve-out is vacuous under that definition, and there is no separate best_of-scored floor-21 pass for this benchmark.

### flores: macro F1 and macro FPR over the 190 benchmark labels, 192,280 rows

|   config | macro F1 | Ma-FPR (x1e5) | n_out_of_set |
|----------|----------|---------------|--------------|
| baseline |   0.9317 |       27.7538 |        3,615 |
|  floor21 |   0.9323 |       28.5243 |        3,028 |
|    gated |   0.9326 |       29.0747 |        2,610 |

### Re-examination accounting (gated configuration)

|          group | examined | moved | blocked_by_proximity | no_cand |
|----------------|----------|-------|----------------------|---------|
| A (N < HEAD_N) |    1,313 |   636 |                  394 |     283 |
|  B (flat-four) |      171 |   163 |                    2 |       6 |

## Constants used

- ACCEPTANCE_TOL = 0.005 (this module; pre-registration)
- HEAD_N = 18,000 (analysis.full_test_margin)
- RES_CAP = 100,000 (analysis.hierarchical_pool)
- D3_PROX = 21.0 (analysis.gate_variants)
- FLOOR_TARGET = -21.0 (analysis.full_test_floor21)
- TOPK_MARGIN = 5 (analysis.margin_diagnostic)
- outputs/diagnostic/tau_floor21_gate.csv (group A thresholds, 1,080 languages, sha256 8ad290e36409c085...)
- outputs/diagnostic/tau_flat4.csv (group B thresholds, 4 languages, sha256 9f85dd5f2bb5f9db...)

Per-label detail: outputs/diagnostic/external_bench/flores_per_label.csv.

Git commit: a952831b28658ba566634c9481f08e4e4efc6d97.
