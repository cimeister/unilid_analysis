# gt_margin candidate build (pre-registered composition, Exp 28 + Exp 26)

- gt_min matrix rebuilt and fingerprint-verified (sha 26ba5f7e303e289b...).
- tau recalibrated under gt_min: 22 of 96 tail languages excluded (< 200 self-won train lines); values in outputs/diagnostic/tau_per_lang_gtmin.csv.
- Tail gt_min-predicted lines scored: 86,924 (top-1 agreement 1.0000; 0 disagreeing lines left ungated).
- Reassigned to a head candidate: 60,320 of 86,924 gated lines; 28,533 land on the true label; 3,741 below-tau lines kept for lack of a head candidate in the top-5.
- All other lines bit-identical to pred_gt_min.npy. Output: /capstor/scratch/cscs/cmeister747/unilid_analysis/full_test_eval/pred_gt_margin.npy.
