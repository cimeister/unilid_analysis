# resource-tier F1 and FPR on the scored pool (45,377,279 lines), within-stratum view

Sources: outputs/diagnostic/paper_eval_per_lang_f1_fullpool.csv (label set, training counts, alignment gate); /capstor/scratch/cscs/cmeister747/unilid_analysis/full_test_eval/y_true.npy, /capstor/scratch/cscs/cmeister747/unilid_analysis/full_test_eval/pred_baseline.npy, /capstor/scratch/cscs/cmeister747/unilid_analysis/full_test_eval/pred_fasttext.npy.

View: examples restricted to true labels inside the tier (cross-tier false positives excluded). F1 is the mean per-language F1 over the tier's languages; FPR is the mean per-language FP_L / (N_test - support_L) over the same languages.

| tier | n_lang | N_test | UniLID F1 | UniLID FPR | fastText F1 | fastText FPR |
|---|---|---|---|---|---|---|
| <500 | 56 | 2,513 | 0.8709 | 7.2268e-05 | 0.9150 | 1.1561e-04 |
| 500-1k | 40 | 5,222 | 0.9748 | 1.4778e-05 | 0.9641 | 1.9575e-05 |
| 1k-12k | 458 | 552,346 | 0.9897 | 8.1738e-06 | 0.9787 | 7.7514e-06 |
| 12k-18k | 526 | 1,151,363 | 0.9972 | 1.9252e-06 | 0.9864 | 1.0150e-05 |
| 18k-35k | 398 | 1,125,105 | 0.9919 | 6.7179e-06 | 0.9808 | 1.5690e-05 |
| 35k+ | 462 | 42,540,730 | 0.9583 | 5.2873e-05 | 0.9421 | 9.1020e-05 |

## Micro form, for the record (not the reported cells)

Summed false positives over summed negatives, same restricted lines.

| tier | UniLID FPR (micro) | fastText FPR (micro) |
|---|---|---|
| <500 | 7.2351e-05 | 1.1576e-04 |
| 500-1k | 1.4731e-05 | 1.9641e-05 |
| 1k-12k | 8.1768e-06 | 7.7529e-06 |
| 12k-18k | 1.9257e-06 | 1.0153e-05 |
| 18k-35k | 6.7119e-06 | 1.5683e-05 |
| 35k+ | 5.2761e-05 | 9.0916e-05 |

## Reproduction gate against paper/submission.tex, tab:resource-tier

Tolerances: F1 0.005 absolute, FPR 5% relative (the published FPR cells are printed at one or two significant figures).

| tier | cell | ours | published | diff | status |
|---|---|---|---|---|---|
| <500 | UniLID F1 | 0.8709 | 0.871 | -0.0001 | OK |
| <500 | UniLID FPR | 7.2268e-05 | 7.2e-05 | +0.4% | OK |
| <500 | fastText F1 | 0.9150 | 0.915 | +0.0000 | OK |
| <500 | fastText FPR | 1.1561e-04 | 0.000115 | +0.5% | OK |
| 500-1k | UniLID F1 | 0.9748 | 0.975 | -0.0002 | OK |
| 500-1k | UniLID FPR | 1.4778e-05 | 1.5e-05 | -1.5% | OK |
| 500-1k | fastText F1 | 0.9641 | 0.964 | +0.0001 | OK |
| 500-1k | fastText FPR | 1.9575e-05 | 1.9e-05 | +3.0% | OK |
| 1k-12k | UniLID F1 | 0.9897 | 0.990 | -0.0003 | OK |
| 1k-12k | UniLID FPR | 8.1738e-06 | 8e-06 | +2.2% | OK |
| 1k-12k | fastText F1 | 0.9787 | 0.979 | -0.0003 | OK |
| 1k-12k | fastText FPR | 7.7514e-06 | 8e-06 | -3.1% | OK |
| 12k-18k | UniLID F1 | 0.9972 | 0.997 | +0.0002 | OK |
| 12k-18k | UniLID FPR | 1.9252e-06 | 2e-06 | -3.7% | OK |
| 12k-18k | fastText F1 | 0.9864 | 0.986 | +0.0004 | OK |
| 12k-18k | fastText FPR | 1.0150e-05 | 1e-05 | +1.5% | OK |
| 18k-35k | UniLID F1 | 0.9919 | 0.992 | -0.0001 | OK |
| 18k-35k | UniLID FPR | 6.7179e-06 | 7e-06 | -4.0% | OK |
| 18k-35k | fastText F1 | 0.9808 | 0.981 | -0.0002 | OK |
| 18k-35k | fastText FPR | 1.5690e-05 | 1.6e-05 | -1.9% | OK |
| 35k+ | UniLID F1 | 0.9583 | 0.958 | +0.0003 | OK |
| 35k+ | UniLID FPR | 5.2873e-05 | 5.3e-05 | -0.2% | OK |
| 35k+ | fastText F1 | 0.9421 | 0.942 | +0.0001 | OK |
| 35k+ | fastText FPR | 9.1020e-05 | 9.1e-05 | +0.0% | OK |

Reproduction gate: PASSED.
