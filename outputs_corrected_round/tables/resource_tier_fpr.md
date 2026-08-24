# resource-tier F1 and FPR on the scored pool (45,377,279 lines), within-stratum view

**NON-DEFAULT MODEL RUN.** Every number below was computed from `/capstor/scratch/cscs/cmeister747/unilid_analysis/corrected/glotlidc_corrected.unilid` and its predictions under `/capstor/scratch/cscs/cmeister747/unilid_analysis/full_test_eval_corrected`, not from the released model (`/capstor/scratch/cscs/cmeister747/unilid_analysis/glotlidc.unilid`), and must not be read as a restatement of the released model's table.

Sources: outputs_corrected_round/diagnostic/paper_eval_per_lang_f1_fullpool.csv (label set, training counts, alignment gate); /capstor/scratch/cscs/cmeister747/unilid_analysis/full_test_eval_corrected/y_true.npy, /capstor/scratch/cscs/cmeister747/unilid_analysis/full_test_eval_corrected/pred_baseline.npy, /capstor/scratch/cscs/cmeister747/unilid_analysis/full_test_eval_corrected/pred_fasttext.npy.

View: examples restricted to true labels inside the tier (cross-tier false positives excluded). F1 is the mean per-language F1 over the tier's languages; FPR is the mean per-language FP_L / (N_test - support_L) over the same languages.

| tier | n_lang | N_test | UniLID F1 | UniLID FPR | fastText F1 | fastText FPR |
|---|---|---|---|---|---|---|
| <500 | 56 | 2,513 | 0.8572 | 6.5053e-05 | 0.9150 | 1.1561e-04 |
| 500-1k | 40 | 5,222 | 0.9731 | 9.8602e-06 | 0.9641 | 1.9575e-05 |
| 1k-12k | 458 | 552,346 | 0.9895 | 8.0751e-06 | 0.9787 | 7.7514e-06 |
| 12k-18k | 526 | 1,151,363 | 0.9971 | 1.9765e-06 | 0.9864 | 1.0150e-05 |
| 18k-35k | 398 | 1,125,105 | 0.9918 | 6.7805e-06 | 0.9808 | 1.5690e-05 |
| 35k+ | 462 | 42,540,730 | 0.9576 | 5.3737e-05 | 0.9421 | 9.1020e-05 |

## Micro form, for the record (not the reported cells)

Summed false positives over summed negatives, same restricted lines.

| tier | UniLID FPR (micro) | fastText FPR (micro) |
|---|---|---|
| <500 | 6.5116e-05 | 1.1576e-04 |
| 500-1k | 9.8204e-06 | 1.9641e-05 |
| 1k-12k | 8.0777e-06 | 7.7529e-06 |
| 12k-18k | 1.9770e-06 | 1.0153e-05 |
| 18k-35k | 6.7746e-06 | 1.5683e-05 |
| 35k+ | 5.3622e-05 | 9.0916e-05 |

## INFORMATIONAL comparison against paper/submission.tex, tab:resource-tier (NOT a gate)

Those published cells are measurements of the RELEASED model and this run scored `/capstor/scratch/cscs/cmeister747/unilid_analysis/corrected/glotlidc_corrected.unilid`, so a difference here is an expected cross-model difference, not a regression and not a reproduction failure. The comparison is computed and reported in full, it withholds nothing, and it does not set the exit code.

Tolerances: F1 0.005 absolute, FPR 5% relative (the published FPR cells are printed at one or two significant figures).

| tier | cell | ours | published | diff | status |
|---|---|---|---|---|---|
| <500 | UniLID F1 | 0.8572 | 0.871 | -0.0138 | MISMATCH |
| <500 | UniLID FPR | 6.5053e-05 | 7.2e-05 | -9.6% | MISMATCH |
| <500 | fastText F1 | 0.9150 | 0.915 | +0.0000 | OK |
| <500 | fastText FPR | 1.1561e-04 | 0.000115 | +0.5% | OK |
| 500-1k | UniLID F1 | 0.9731 | 0.975 | -0.0019 | OK |
| 500-1k | UniLID FPR | 9.8602e-06 | 1.5e-05 | -34.3% | MISMATCH |
| 500-1k | fastText F1 | 0.9641 | 0.964 | +0.0001 | OK |
| 500-1k | fastText FPR | 1.9575e-05 | 1.9e-05 | +3.0% | OK |
| 1k-12k | UniLID F1 | 0.9895 | 0.990 | -0.0005 | OK |
| 1k-12k | UniLID FPR | 8.0751e-06 | 8e-06 | +0.9% | OK |
| 1k-12k | fastText F1 | 0.9787 | 0.979 | -0.0003 | OK |
| 1k-12k | fastText FPR | 7.7514e-06 | 8e-06 | -3.1% | OK |
| 12k-18k | UniLID F1 | 0.9971 | 0.997 | +0.0001 | OK |
| 12k-18k | UniLID FPR | 1.9765e-06 | 2e-06 | -1.2% | OK |
| 12k-18k | fastText F1 | 0.9864 | 0.986 | +0.0004 | OK |
| 12k-18k | fastText FPR | 1.0150e-05 | 1e-05 | +1.5% | OK |
| 18k-35k | UniLID F1 | 0.9918 | 0.992 | -0.0002 | OK |
| 18k-35k | UniLID FPR | 6.7805e-06 | 7e-06 | -3.1% | OK |
| 18k-35k | fastText F1 | 0.9808 | 0.981 | -0.0002 | OK |
| 18k-35k | fastText FPR | 1.5690e-05 | 1.6e-05 | -1.9% | OK |
| 35k+ | UniLID F1 | 0.9576 | 0.958 | -0.0004 | OK |
| 35k+ | UniLID FPR | 5.3737e-05 | 5.3e-05 | +1.4% | OK |
| 35k+ | fastText F1 | 0.9421 | 0.942 | +0.0001 | OK |
| 35k+ | fastText FPR | 9.1020e-05 | 9.1e-05 | +0.0% | OK |

Comparison outcome (informational): cells ['<500/UniLID F1', '<500/UniLID FPR', '500-1k/UniLID FPR'] differ from the published table, as expected for a different model.
