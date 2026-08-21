# floor21 solo-gate build (combined-method plan amendment 3 reference build)

- floor-21 matrix rebuilt and fingerprint-verified (sha 0f1812e0c73d4e2b...).
- tau recalibrated under the floor21 matrix: 26 of 1080 gated languages excluded (cause recorded per language); values in outputs_corrected_round/diagnostic/tau_floor21_gate.csv.
- Gated floor21-predicted lines scored: 2,101,845 (top-1 agreement 1.0000; 0 disagreeing lines left ungated).
- Reassigned to a head candidate: 238,618 of 2,101,845 gated lines; 136,878 land on the true label; 26,903 below-tau lines kept for lack of a head candidate in the top-5.
- All other lines bit-identical to pred_floor21.npy. Output: /capstor/scratch/cscs/cmeister747/unilid_analysis/full_test_eval_corrected/pred_floor21_gate.npy.
