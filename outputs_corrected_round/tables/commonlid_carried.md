# CommonLID (web-domain) check of the top carried configurations (Exp 39)

373,230 lines; macro-aware accuracy convention of Exp 12 (prediction correct if its iso code, or its macrolanguage, matches the gold tag).

| config | macro-aware accuracy | delta | tag-level macro-F1 | delta |
|---|---|---|---|---|
| baseline | 0.8476 |  | 0.7218 |  |
| floor21 | 0.8512 | +0.0036 | 0.7203 | -0.0015 |

Configurations run: 2 of the 3 configurations (baseline, floor21, gt_margin_adaptive): baseline, floor21. gt_margin_adaptive was not requested: it needs released-model thresholds and the Exp 27 Good-Turing record, and feeds no paper cell.

Model: /capstor/scratch/cscs/cmeister747/unilid_analysis/corrected/glotlidc_corrected.unilid
Clamp: c = -17.0 (/capstor/scratch/cscs/cmeister747/unilid_analysis/full_test_eval_corrected/fingerprint_floor21.json)

INFORMATIONAL, NOT A GATE: the two recorded CommonLID baseline values below are measurements of the RELEASED model and this run scored /capstor/scratch/cscs/cmeister747/unilid_analysis/corrected/glotlidc_corrected.unilid, so a difference is an expected cross-model difference, not a regression and not a reproduction failure. Every configuration in the table above WAS computed and written, with this run's own numbers.
- baseline macro-aware accuracy: 0.8476 differs from the recorded 0.8452 (Exp 12/39, outputs/tables/commonlid_carried.md), difference +0.0024.
- baseline tag-level macro-F1: 0.7218 differs from the recorded 0.7228 (Exp 12/39, outputs/tables/commonlid_carried.md), difference -0.0010.
