# Mistral-Nemo variant: tau recalibration under the floor-21 matrix (E3 pre-registration)

floor-21 matrix: FLOOR_TARGET=-17.0 (--floor-target override), fingerprint at /capstor/scratch/cscs/cmeister747/unilid_analysis/full_test_eval_mistralnemo_corrected/fingerprint_floor21_mistralnemo.json.

Group A (N < HEAD_N=18,000): 1080 languages, size-adaptive quantile q_L = MARGIN_Q * (1 - min(N,HEAD_N)/HEAD_N), MARGIN_Q=5 (analysis.margin_diagnostic); exclude-and-log on calibration shortfall (mirrors analysis/solo_gates.py's run("floor21")). Output: outputs_corrected_round/diagnostic/tau_mistralnemo_floor21_gate.csv.

Group B (the variant's flat set, N >= HEAD_N): 3 languages, fixed 5th percentile; abort (not exclude) on calibration shortfall (mirrors analysis.gate_variants._calibrate_flat4_tau5). Output: outputs_corrected_round/diagnostic/tau_mistralnemo_flat.csv.

Calibration constants (analysis.margin_diagnostic, unchanged): CALIB_MAX=2,000, CALIB_SEED=0, MIN_CALIB_LINES=200, TOPK_MARGIN=5, corpus files at /capstor/scratch/cscs/cmeister747/unilid_analysis/results_apertus200k/corpus/{lang}_train.txt.
