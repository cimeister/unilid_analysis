# unmod solo-gate build (combined-method plan amendment 3 reference build)

- unmodified weight matrix, sha256-verified against sha256_base_W (sha 18e8d53637018cb4...).
- tau recalibrated under the unmod matrix: 26 of 1080 gated languages excluded (cause recorded per language); values in outputs/diagnostic/tau_unmod_gate.csv.
- Gated unmod-predicted lines scored: 2,175,310 (top-1 agreement 1.0000; 0 disagreeing lines left ungated).
- Reassigned to a head candidate: 270,704 of 2,175,310 gated lines; 164,986 land on the true label; 42,758 below-tau lines kept for lack of a head candidate in the top-5.
- All other lines bit-identical to pred_baseline.npy. Output: /capstor/scratch/cscs/cmeister747/unilid_analysis/full_test_eval/pred_unmod_gate.npy.
