# shared9_bar18k candidate build (Experiment 47)

Direction 1 (Experiment 47): one shared re-examination threshold of 9.0 nats in place of the promoted configuration's 1,080 per-language calibrated thresholds, and a replacement-candidate minimum lowered from RES_CAP (100,000 training lines, the promoted configuration's own bar) to HEAD_N (18,000 training lines).

- Constants: HEAD_N = 18,000 training lines (the topk-stage low-N criterion); flat_magnet category from outputs/diagnostic/lang_diagnostic.csv (the topk-stage flatness criterion; ZH_MAGNET = 1.5 is one of several inputs to that classification in analysis/diagnostic.py, not applied directly here); TOPK_MARGIN = 5 saved candidates per affected line; this variant's re-examination threshold is 9.0 nats; RES_CAP = 100,000 training lines is the promoted configuration's own replacement-candidate bar, not used by this variant.
- Re-examined set for this variant: languages with N < HEAD_N (18,000 training lines) only; the flat-distribution languages added to the topk-stage candidate universe by the category == 'flat_magnet' criterion are excluded from this variant's re-examined set.
- Affected: 2,236,864 lines carry a saved floor21-prediction candidate list (the topk-stage expanded label set).
- In-set: 2,103,258 of 2,236,864 affected lines have a floor21 prediction in this variant's re-examined set.
- Top1-disagreements: 1 in-set lines whose saved top-1 candidate disagrees with pred_floor21.npy; left unchanged.
- Of the remaining 2,103,257 in-set, agreeing lines: 1,712,981 have a top1-minus-top2 score gap at or above 9.0 nats and are kept unchanged; 390,276 fall below 9.0 nats and are re-examined.
- Moved: 387,039 of the 390,276 re-examined lines move to a replacement candidate ranked 2 to 5 with N >= 18,000 training lines.
- Moved-to-true: 213,146 of the 387,039 moved lines land on the true label recorded in y_true.npy.
- Kept-no-candidate: 3,237 re-examined lines have no candidate ranked 2 to 5 meeting the acceptance condition, and keep the floor21 prediction.
- Short candidate lists: the topk stage recorded 0 affected lines with fewer than 5 saved candidates, of which 0 had fewer than 2 and are treated as having infinite margin (never moved), following the recorded margin-gate convention (analysis/margin_diagnostic.py's _gap()).
- All lines outside the moved set are bit-identical to pred_floor21.npy (387,039 lines differ, verified equal to n_moved). Output: /capstor/scratch/cscs/cmeister747/unilid_analysis/full_test_eval/pred_gate_shared9_bar18k.npy; metadata: /capstor/scratch/cscs/cmeister747/unilid_analysis/full_test_eval/pred_gate_shared9_bar18k_meta.json.
