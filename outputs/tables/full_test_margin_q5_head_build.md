# margin_q5_head candidate build (plan B3 follow-up, Exp 26 rule, target=head)

- Tail-predicted lines scored: 29,606 (top-1 agreement 1.0000; 0 disagreeing lines left ungated at the baseline prediction).
- Reassigned: 16,239 of 29,606 gated lines; 6,858 reassignments land on the true label. Below-tau lines kept at baseline for lack of a head candidate in the top-5: 1,534.
- All other lines are bit-identical to pred_baseline.npy. Output: /capstor/scratch/cscs/cmeister747/unilid_analysis/full_test_eval/pred_margin_q5_head.npy.
- tau source: outputs/diagnostic/tau_per_lang.csv (Exp 26, job 2883715); excluded languages have tau = -inf and are never gated.
