# gt_margin_all_100k candidate build (pre-registered composition; gate=nonhead)

- gt_min matrix rebuilt and fingerprint-verified (sha 26ba5f7e303e289b...).
- tau recalibrated under gt_min: 22 of 1080 gated languages excluded (< 200 self-won train lines); values in outputs/diagnostic/tau_gt_margin_all_100k.csv.
- Gated gt_min-predicted lines scored: 2,265,417 (top-1 agreement 1.0000; 1 disagreeing lines left ungated).
- Reassigned to a head candidate: 407,562 of 2,265,416 gated lines; 251,419 land on the true label; 94,462 below-tau lines kept for lack of a head candidate in the top-5.
- All other lines bit-identical to pred_gt_min.npy. Output: /capstor/scratch/cscs/cmeister747/unilid_analysis/full_test_eval/pred_gt_margin_all_100k.npy.
