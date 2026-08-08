# Mistral-Nemo variant: flat-language rule (E3 pre-registration)

zH (within-script entropy z-score) and magnet_ratio recomputed for the variant's own weight matrix, reusing analysis.diagnostic._probs_and_logprobs / ._empirical_magnet by import (entropy H computed directly as -(P * logP).sum(axis=1), verified bit-identical to _sym_kl_matrix's own entropy return, without that function's unused symKL matmul). Rule (imported constants, unchanged): is_magnet = (zH > ZH_MAGNET=1.5 and magnet_ratio > MAGNET_RATIO_MIN=2.0) or zH > ZH_EXTREME=5.0. magnet_ratio, support_val, and fp_val come from /capstor/scratch/cscs/cmeister747/unilid_analysis/full_test_eval_mistralnemo/pred_nemo_calibval.npy (the retired 250,000-line validation half), never the test pool.

- Languages flagged is_magnet by this rule: 90 of 1,940.
- Flat set (is_magnet AND N >= HEAD_N=18,000), the languages the gate's step 2 re-examines: 3.

| lang | N | zH | support_val | fp_val | magnet_ratio |
|---|---|---|---|---|---|
| bjn_Latn | 27655 | 2.4973 | 20 | 48 | 2.2857 |
| sco_Latn | 87458 | 2.5611 | 87 | 219 | 2.4886 |
| srp_Latn | 79008 | 1.9673 | 76 | 237 | 3.0779 |
