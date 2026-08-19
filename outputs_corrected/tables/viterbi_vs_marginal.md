# tab:viterbi_vs_marginal: Viterbi decoding against exact marginalization

Model: `/capstor/scratch/cscs/cmeister747/unilid_analysis/corrected/glotlidc_corrected.unilid`. Pool: 45,377,279 lines (full test set minus the 250,000 validation lines).

| Decoding | Accuracy | Macro F1 |
|---|---|---|
| UniLID (Viterbi) | 0.961 | 0.933 |
| UniLID (Marginalization) | 0.961 | 0.935 |

Marginalization changes macro F1 by +0.0023 and accuracy by +0.0004.
