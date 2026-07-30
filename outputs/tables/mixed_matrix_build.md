# Mixed-matrix build report (per-language combined method, rule v1)

## Stage B (tau recalibrated under the mixed matrix)

- mixed matrix rebuilt and fingerprint-verified against /capstor/scratch/cscs/cmeister747/unilid_analysis/full_test_eval/fingerprint_mixed.json (sha 0c31f143e86da4f5...); assignments CSV sha256-verified.
- gate_on set: 1080 of 1940 languages.
- tau recalibrated under the mixed matrix: 26 of 1080 gated languages excluded (not_excluded: 1054, low_calibration: 26).
- Gate_on mixed-nogate-predicted lines scored: 2,086,908 (top-1 agreement 1.0000; 1 disagreeing lines left ungated).
- Of 2,086,907 agreeing gated lines: reassigned to a head candidate 235,309; 137,536 land on the true label; 24,303 below-tau lines kept for lack of a head candidate in the top-5.
- All other lines bit-identical to pred_mixed_nogate.npy. Output: /capstor/scratch/cscs/cmeister747/unilid_analysis/full_test_eval/pred_mixed.npy.

## Stage B_solotau (tau read from the floor21_gate solo reference)

- mixed matrix rebuilt and fingerprint-verified against /capstor/scratch/cscs/cmeister747/unilid_analysis/full_test_eval/fingerprint_mixed.json (sha 0c31f143e86da4f5...); assignments CSV sha256-verified.
- gate_on set: 1080 of 1940 languages.
- tau read from outputs/diagnostic/tau_floor21_gate.csv (rule v1's gate_on solo reference is floor21_gate), no recalibration: 26 of 1080 gated languages excluded (not_excluded: 1054, low_calibration: 26).
- Gate_on mixed-nogate-predicted lines scored: 2,086,908 (top-1 agreement 1.0000; 1 disagreeing lines left ungated).
- Of 2,086,907 agreeing gated lines: reassigned to a head candidate 237,244; 137,792 land on the true label; 25,634 below-tau lines kept for lack of a head candidate in the top-5.
- All other lines bit-identical to pred_mixed_nogate.npy. Output: /capstor/scratch/cscs/cmeister747/unilid_analysis/full_test_eval/pred_mixed_solotau.npy.
