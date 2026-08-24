# Camera-ready E4: script and resource-tier breakdowns

**NON-DEFAULT MODEL RUN.** Every number below was computed from `/capstor/scratch/cscs/cmeister747/unilid_analysis/corrected/glotlidc_corrected.unilid` and its predictions under `/capstor/scratch/cscs/cmeister747/unilid_analysis/full_test_eval_corrected`, not from the released model (`/capstor/scratch/cscs/cmeister747/unilid_analysis/glotlidc.unilid`), and must not be read as a restatement of the released model's tables. The comparisons below against paper/submission.tex are INFORMATIONAL for this run, not gates: those published cells were produced by the RELEASED model, so a difference against them is the expected outcome of a cross-model comparison, not a regression. They are computed and reported in full, and they withhold nothing: the .tex fragments below carry THIS run's regenerated numbers whether or not they match the published cells, and the script does not exit nonzero for such a difference. Under the released model the same comparisons are binding gates and a MISMATCH still withholds the affected .tex and exits 1.

Pre-registration: EXPERIMENTS_PLAN.md, "Camera-ready evaluation program (2026-08-06)", E4 bullet. Conventions: EXPERIMENTAL_SETUP.md, "Camera-ready reporting conventions". Input: outputs_corrected_round/diagnostic/paper_eval_per_lang_f1_fullpool.csv (analysis/paper_eval.py) for the global view; /capstor/scratch/cscs/cmeister747/unilid_analysis/full_test_eval_corrected prediction memmaps for the within-stratum view.

Both views below: global = per-language F1 with all false positives counted; within-stratum = examples restricted to true labels in the group, cross-group false positives excluded; full kept pool, 45,377,279 lines.

### Script breakdown, both views (global = per-language F1 with all false positives counted; within-stratum = examples restricted to true labels in the group, cross-group false positives excluded; full kept pool, 45,377,279 lines). Uses the full 1,940-language basis (all 84 Other languages); the paper's original basis excluded jpn_Jpan and kor_Hang from Other.

| Script | # Langs | baseline (global) | gate-flat4-prox21 (global) | fastText (global) | baseline (within-stratum) | gate-flat4-prox21 (within-stratum) | fastText (within-stratum) |
|--------|---------|-------------------|----------------------------|-------------------|---------------------------|------------------------------------|---------------------------|
|   Latn |    1700 |            0.9440 |                     0.9657 |            0.9464 |                    0.9443 |                             0.9657 |                    0.9465 |
|   Cyrl |      70 |            0.8741 |                     0.9219 |            0.9684 |                    0.8801 |                             0.9220 |                    0.9697 |
|   Arab |      38 |            0.6914 |                     0.7447 |            0.7472 |                    0.6926 |                             0.7447 |                    0.7474 |
|   Deva |      32 |            0.8109 |                     0.8706 |            0.9311 |                    0.8109 |                             0.8707 |                    0.9317 |
|   Beng |       6 |            0.8790 |                     0.9466 |            0.9844 |                    0.8790 |                             0.9466 |                    0.9846 |
|   Grek |       4 |            0.6741 |                     0.7466 |            0.9252 |                    0.6750 |                             0.7469 |                    0.9253 |
|   Hebr |       4 |            0.6943 |                     0.7705 |            0.9665 |                    0.7376 |                             0.7705 |                    0.9666 |
|   Armn |       2 |            0.9720 |                     0.9720 |            0.9855 |                    0.9721 |                             0.9721 |                    0.9856 |
|  Other |      84 |            0.9354 |                     0.9460 |            0.9721 |                    0.9384 |                             0.9462 |                    0.9727 |

Script-table published-cell comparison (within-stratum): INFORMATIONAL, NOT A GATE: script-table rows ['Beng'] differ from paper/submission.tex. Those published cells are measurements of the RELEASED model and this run scored /capstor/scratch/cscs/cmeister747/unilid_analysis/corrected/glotlidc_corrected.unilid, so a difference here is an expected cross-model difference, not a regression and not a reproduction failure. The .tex fragment WAS written, with this run's regenerated numbers. See outputs_corrected_round/tables/paper_breakdowns_gate.md for the full comparison.

### Resource-tier breakdown, both views (global = per-language F1 with all false positives counted; within-stratum = examples restricted to true labels in the group, cross-group false positives excluded; full kept pool, 45,377,279 lines)

| Resource | # Langs | baseline (global) | gate-flat4-prox21 (global) | fastText (global) | baseline (within-stratum) | gate-flat4-prox21 (within-stratum) | fastText (within-stratum) |
|----------|---------|-------------------|----------------------------|-------------------|---------------------------|------------------------------------|---------------------------|
|     <500 |      56 |            0.5957 |                     0.7806 |            0.7497 |                    0.8572 |                             0.8199 |                    0.9150 |
|  500--1k |      40 |            0.6762 |                     0.8932 |            0.8614 |                    0.9731 |                             0.9537 |                    0.9641 |
|  1k--12k |     458 |            0.8936 |                     0.9437 |            0.9414 |                    0.9895 |                             0.9864 |                    0.9787 |
| 12k--18k |     526 |            0.9786 |                     0.9848 |            0.9708 |                    0.9971 |                             0.9965 |                    0.9864 |
| 18k--35k |     398 |            0.9622 |                     0.9647 |            0.9526 |                    0.9918 |                             0.9916 |                    0.9808 |
|     35k+ |     462 |            0.9568 |                     0.9564 |            0.9408 |                    0.9576 |                             0.9573 |                    0.9421 |

Resource-tier published-cell comparison (within-stratum): INFORMATIONAL, NOT A GATE: resource-tier rows ['<500'] differ from paper/submission.tex. Those published cells are measurements of the RELEASED model and this run scored /capstor/scratch/cscs/cmeister747/unilid_analysis/corrected/glotlidc_corrected.unilid, so a difference here is an expected cross-model difference, not a regression and not a reproduction failure. The .tex fragment WAS written, with this run's regenerated numbers. See outputs_corrected_round/tables/paper_breakdowns_gate.md for the full comparison.


Git commit: dd9c570ad761ceba672aea5fa9ed57f492144c07.
