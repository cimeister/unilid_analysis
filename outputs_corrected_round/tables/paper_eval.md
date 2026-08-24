# Camera-ready E1: common reporting set

**NON-DEFAULT MODEL RUN.** Every number below was computed from `/capstor/scratch/cscs/cmeister747/unilid_analysis/corrected/glotlidc_corrected.unilid`, scored into `/capstor/scratch/cscs/cmeister747/unilid_analysis/full_test_eval_corrected`, and written under `outputs_corrected_round`. It is NOT the released model (`/capstor/scratch/cscs/cmeister747/unilid_analysis/glotlidc.unilid`) and these cells must not be compared term-by-term with the released model's tables, which come from a different scoring run. Wiring gates A, B and C were waived (--waive-released-model-gates): their reference CSVs and their pre-registered anchors are released-model measurements that cannot be regenerated for this model. Gates 1 and 2 (canonical language order, seed-301 rule split re-derived from this run's own y_true) did run.

Pre-registration: EXPERIMENTS_PLAN.md, "Camera-ready evaluation program (2026-08-06)", E1. Configurations: baseline, gate_flat4_prox21, fasttext. Instruments follow EXPERIMENTAL_SETUP.md, "Camera-ready reporting conventions": Table 1 cells use the full kept pool, 45,377,279 lines; the appendix comparison uses the judge part, 27,002,441 lines. No delta pairs terms from different line sets.

## Gates passed

- Language order: _load_model_data's 1,940-language list matches the lang column of outputs/diagnostic/full_test_per_lang_prf.csv.
- y_true memmap shape (45627279,) matches TOTAL_LINES (45,627,279); full kept pool, 45,377,279 lines matches EXPECTED_KEPT.
- Full remainder (kept minus draws 101 and 201): 45,004,014 lines, matches EXPECTED_REMAINDER (45,004,014).
- Seed-301 split recomputed with fraction 0.4: derivation part 18,001,573 lines (EXPECTED_DERIVATION 18,001,573), judge part 27,002,441 lines (EXPECTED_JUDGE 27,002,441); matches the split recorded at /capstor/scratch/cscs/cmeister747/unilid_analysis/full_test_eval/rule_split_seed301.npz.
- Wiring gates A, B and C: WAIVED. Gate A's reference (outputs_corrected_round/diagnostic/carried_set_per_lang_f1.csv), gate B's reference (outputs_corrected_round/diagnostic/mixed_eval_judge_f1_gate_flat4_prox21.csv) and its anchors {'baseline': 0.9117, 'gate_flat4_prox21': 0.9498}, and gate C's recorded 0.9292 / 2.0263e-05 are all measurements of /capstor/scratch/cscs/cmeister747/unilid_analysis/glotlidc.unilid; none of them was read or compared against while scoring /capstor/scratch/cscs/cmeister747/unilid_analysis/corrected/glotlidc_corrected.unilid. The seven CARRIED prediction memmaps gate A needs were likewise not loaded.

Sentinel guard: zero UNSEEN/EXCLUDED values on the kept pool for every loaded memmap. EMPTY (-1) counts on the kept pool: baseline 0, gate_flat4_prox21 0, fasttext 0.

Note for the camera-ready table: the submission's Table 1 rows state N = 45,627,279; the cells here use the full kept pool, 45,377,279 lines, so every restated row in the camera-ready table must carry N = 45,377,279 and the N column must be updated for all rows, not only the new one.

### Macro F1 and macro FPR (x1e5), full kept pool, 45,377,279 lines

|            config | macro F1 | Ma-FPR (x1e5) |
|-------------------|----------|---------------|
|          baseline |   0.9327 |        2.0187 |
| gate_flat4_prox21 |   0.9564 |        1.7745 |
|          fasttext |   0.9443 |        2.7063 |

### Macro F1 and macro FPR (x1e5), judge part, 27,002,441 lines

|            config | macro F1 | Ma-FPR (x1e5) |
|-------------------|----------|---------------|
|          baseline |   0.9159 |        2.0301 |
| gate_flat4_prox21 |   0.9495 |        1.7825 |
|          fasttext |   0.9332 |        2.7165 |

## Paired bootstrap, judge part, 27,002,441 lines

B=10,000, seed=0, percentile 95% interval, paired resample over the 1,940 language positions, one shared resample matrix reused across contrasts. Point estimate = (gate_flat4_prox21 judge-part mean F1) minus (comparator judge-part mean F1).

### gate_flat4_prox21 minus comparator, judge part, 27,002,441 lines

| comparator | mean diff |             95% CI |
|------------|-----------|--------------------|
|   baseline |   +0.0336 | [+0.0290, +0.0383] |
|   fasttext |   +0.0163 | [+0.0108, +0.0221] |

## Input prediction memmaps

| config | sha256 (first 16) | bytes |
|---|---|---|
| baseline | a89c1448214a0f7e | 91,254,686 |
| gate_flat4_prox21 | d2b948d36f967794 | 91,254,686 |
| fasttext | 4ff74fb55ce5668b | 91,254,686 |

Per-language detail: outputs_corrected_round/diagnostic/paper_eval_per_lang_f1_fullpool.csv (full kept pool, 45,377,279 lines), outputs_corrected_round/diagnostic/paper_eval_per_lang_f1_judge.csv (judge part, 27,002,441 lines).

Git commit: dd9c570ad761ceba672aea5fa9ed57f492144c07.
