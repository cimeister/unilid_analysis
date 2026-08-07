# GlotLID-C CLD3-subset macro F1 (camera-ready right-side Table 1 cells)

Subset: the 83 bare ISO codes in unilid_resources/glotlidc_cld3subset_83.txt,
each mapped to its largest-training-corpus lang_Script variant (convention
verified: reproduces the paper's printed baseline cell .971). Instrument: kept
test lines whose true label is one of the 83 (23,293,775 lines); per-language
F1 with the full confusion on those lines, averaged over the 83.
FPR cells are NOT computed here: neither the restricted-pool nor the
global-pool convention reproduces the paper's printed 1.63e-4 (measured
9.71e-5 and 7.77e-5); the paper team's eval script is needed first.
Source commit 123cdf50437ffb0ff3ccb457a0b95a6d6e7d5972.

- baseline: 0.9719
- gate_flat4_prox21: 0.9751
- fasttext: 0.9767

## Follow-up measurements (same session)

Global-view alternative (full-pool per-language F1 averaged over the same 83
labels): baseline 0.9680, gate_flat4_prox21 0.9706, fasttext 0.9719; the paper
team's own fastText per-language JSON gives 0.9719 over the same 83, exactly
matching our fastText values. None of the tested conventions reproduces the
paper's printed fastText subset cell (.990), while the printed UniLID cell
(.971) is reproduced by the restricted-lines open-set convention above. The
submission's right-side columns therefore appear to mix conventions per
system; resolving which convention the .990 used needs the paper team's eval
script, and the new row's subset cells should be computed under the same
convention as the existing UniLID rows (the reproduced one) with the
discrepancy flagged to the authors.
