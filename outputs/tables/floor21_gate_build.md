# floor21 solo-gate build (combined-method plan amendment 3 reference build)

- floor-21 matrix rebuilt and fingerprint-verified (sha 70ad322b5f53662d...).
- tau recalibrated under the floor21 matrix: 26 of 1080 gated languages excluded (cause recorded per language); values in outputs/diagnostic/tau_floor21_gate.csv.
- Gated floor21-predicted lines scored: 2,103,258 (top-1 agreement 1.0000; 1 disagreeing lines left ungated).
- Reassigned to a head candidate: 243,652 of 2,103,257 gated lines; 141,667 land on the true label; 24,029 below-tau lines kept for lack of a head candidate in the top-5.
- All other lines bit-identical to pred_floor21.npy. Output: /capstor/scratch/cscs/cmeister747/unilid_analysis/full_test_eval/pred_floor21_gate.npy.
