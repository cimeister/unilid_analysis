# E3: Mistral-Nemo variant evaluation

Pre-registration: EXPERIMENTS_PLAN.md, "Camera-ready evaluation program (2026-08-06)", E3. Model: `glotlid_mistralnemo_fp64_corrected.unilid` (/capstor/scratch/cscs/cmeister747/unilid_analysis/corrected/glotlid_mistralnemo_fp64_corrected.unilid). Configurations: nemo_baseline (unmodified matrix), nemo_floor21 (the variant's own floor-21 matrix, no gate), nemo_gated (floor-21 plus the promoted configuration's two-step re-examination, both tau sets recalibrated for this matrix, D3_PROX/RES_CAP/HEAD_N unchanged).

## Gates passed

- Language order: the variant's 1,940-language list matches the canonical order.
- y_true.npy reuse: shape (45,627,279,), no UNSEEN, 250,000 EXCLUDED lines (== EXPECTED_VAL_LINES 250,000).
- Full kept pool: 45,377,279 lines (== EXPECTED_KEPT 45,377,279).
- Seed-301 judge split: 18,001,573 derivation / 27,002,441 judge (== EXPECTED_DERIVATION 18,001,573 / EXPECTED_JUDGE 27,002,441); matches the stored record at /capstor/scratch/cscs/cmeister747/unilid_analysis/full_test_eval/rule_split_seed301.npz.
- Sentinel guard: no value < -1 on the kept pool for nemo_baseline, nemo_floor21, or nemo_gated.
- Banked-array identity: gate_topk_ids[:,0] == pred_nemo_floor21 at every banked line (asserted at both the topk and eval stages).
- Weight-matrix sha: /capstor/scratch/cscs/cmeister747/unilid_analysis/full_test_eval_mistralnemo_corrected/fingerprint_baseline.json's weight_matrix_sha256 matches /capstor/scratch/cscs/cmeister747/unilid_analysis/full_test_eval_mistralnemo_corrected/fingerprint_floor21_mistralnemo.json's sha256_base_W.
- /capstor/scratch/cscs/cmeister747/unilid_analysis/full_test_eval_mistralnemo_corrected/gate_topk_fingerprint_nemo.json: flat_set/tau CSV shas, langs_sha256, head_n, topk_margin, n_affected, and gate_ids/gate_scores shape all match the current state.
- Gate-group membership: every banked line's floor-21 base prediction falls in exactly one of group A (N < HEAD_N) or group B (the flat set).

### full kept pool, 45,377,279 lines

|        config | macro F1 | Ma-FPR (x1e5) |
|---------------|----------|---------------|
| nemo_baseline |   0.9119 |        1.8583 |
|  nemo_floor21 |   0.9350 |        1.7897 |
|    nemo_gated |   0.9504 |        1.6247 |

### judge part, 27,002,441 lines

|        config | macro F1 | Ma-FPR (x1e5) |
|---------------|----------|---------------|
| nemo_baseline |   0.8951 |        1.8653 |
|  nemo_floor21 |   0.9232 |        1.7960 |
|    nemo_gated |   0.9440 |        1.6283 |

- EMPTY (-1) predictions on the full kept pool, 45,377,279 lines: nemo_baseline 0, nemo_floor21 0, nemo_gated 0.

## Re-examination accounting

### Re-examination accounting (gated configuration)

|          group | examined |   moved | blocked_by_proximity | no_cand |
|----------------|----------|---------|----------------------|---------|
| A (N < HEAD_N) |  201,516 | 170,333 |                8,672 |  22,511 |
|   B (flat set) |   58,165 |  57,514 |                   35 |     616 |


Group A tau CSV (outputs_corrected_round/diagnostic/tau_mistralnemo_floor21_gate.csv): 33 of 1,080 languages excluded (low_calibration or zero_strength). Group B tau CSV (outputs_corrected_round/diagnostic/tau_mistralnemo_flat.csv): 0 of 3 languages excluded.

227,847 of 45,377,279 kept lines have nemo_gated != nemo_floor21 (the lines the two-step re-examination actually moved).

Topk stage candidate-list shortfall (persisted in /capstor/scratch/cscs/cmeister747/unilid_analysis/full_test_eval_mistralnemo_corrected/gate_topk_fingerprint_nemo.json): 0 of 2,129,946 affected lines returned fewer than 5 saved candidates; 0 of those returned fewer than 2 and are treated as having infinite margin (never moved), following the recorded margin-gate convention (analysis.margin_diagnostic's _gap()).

## Paired bootstrap, judge part, 27,002,441 lines

B=10,000, seed=0, percentile 95% interval, paired resample over the 1,940 language positions. (nemo_gated - nemo_baseline) judge-part mean F1: +0.0489 [+0.0424, +0.0555].

## Comparability to the paper's own Mistral-Nemo row (recorded measurement, not a gate)

The paper's UniLID-Mistral-Nemo full-pool cell (paper/tables/lid_main.tex) is F1 0.912, FPR 1.84e-05 (raw scientific notation, not the x1e5-scaled convention used in the table above), computed over the paper team's own N = 45,627,279 lines. This module's nemo_baseline full-pool cell is computed over the full kept pool, 45,377,279 lines: the two instruments differ by the 250,000 retired validation lines (== EXPECTED_VAL_LINES 250,000), so that difference is part of any gap below, not attributable to the retrain alone. That row is the paper team's own training run of the variant; this module evaluates an independent retrain from the same recipe, so rough proximity is expected, not equality. Measured nemo_baseline full-pool: F1 0.9119 (diff -0.0001), FPR 1.8583e-05 (diff +1.8261e-07).

## Degeneracy caveat

34 of 1,940 rows are flagged degenerate (fewer than 100 estimated tokens; outputs_corrected_round/tables/degenerate_rows_mistralnemo.md), adjudicated as an accepted model property (base-vocab script coverage in minority scripts: Ethiopic, Canadian syllabics, Syriac, Tibetan, and similar; EXPERIMENTS_CHRONOLOGICAL.md, 2026-08-07), not gated on here. Per-language F1 for these rows is carried in the `degenerate` column of outputs_corrected_round/diagnostic/mistralnemo_per_lang_f1_fullpool.csv and outputs_corrected_round/diagnostic/mistralnemo_per_lang_f1_judge.csv for the reader to inspect directly.

## Constants used

- HEAD_N = 18,000 (analysis.full_test_margin)
- RES_CAP = 100,000 (analysis.hierarchical_pool)
- D3_PROX = 21.0 (analysis.gate_variants)
- FLOOR_TARGET = -17.0 (--floor-target override)
- TOPK_MARGIN = 5 (analysis.margin_diagnostic)
- outputs_corrected_round/diagnostic/tau_mistralnemo_floor21_gate.csv (group A, 1080 languages, sha256 258b25dc829616f7...)
- outputs_corrected_round/diagnostic/tau_mistralnemo_flat.csv (group B, 3 languages, sha256 4bdcfd27e137e14e...)

Per-language detail: outputs_corrected_round/diagnostic/mistralnemo_per_lang_f1_fullpool.csv (full kept pool, 45,377,279 lines), outputs_corrected_round/diagnostic/mistralnemo_per_lang_f1_judge.csv (judge part, 27,002,441 lines).

Git commit: dd9c570ad761ceba672aea5fa9ed57f492144c07.
