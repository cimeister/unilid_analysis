# Camera-ready E4: script and resource-tier breakdowns

Pre-registration: EXPERIMENTS_PLAN.md, "Camera-ready evaluation program (2026-08-06)", E4 bullet. Conventions: EXPERIMENTAL_SETUP.md, "Camera-ready reporting conventions". Input: outputs/diagnostic/paper_eval_per_lang_f1_fullpool.csv (analysis/paper_eval.py) for the global view; /capstor/scratch/cscs/cmeister747/unilid_analysis/full_test_eval prediction memmaps for the within-stratum view.

Both views below: global = per-language F1 with all false positives counted; within-stratum = examples restricted to true labels in the group, cross-group false positives excluded; full kept pool, 45,377,279 lines.

### Script breakdown, both views (global = per-language F1 with all false positives counted; within-stratum = examples restricted to true labels in the group, cross-group false positives excluded; full kept pool, 45,377,279 lines). Uses the full 1,940-language basis (all 84 Other languages); the paper's original basis excluded jpn_Jpan and kor_Hang from Other.

| Script | # Langs | baseline (global) | gate-flat4-prox21 (global) | fastText (global) | baseline (within-stratum) | gate-flat4-prox21 (within-stratum) | fastText (within-stratum) |
|--------|---------|-------------------|----------------------------|-------------------|---------------------------|------------------------------------|---------------------------|
|   Latn |    1700 |            0.9401 |                     0.9662 |            0.9464 |                    0.9404 |                             0.9662 |                    0.9465 |
|   Cyrl |      70 |            0.8712 |                     0.9223 |            0.9684 |                    0.8774 |                             0.9224 |                    0.9697 |
|   Arab |      38 |            0.6899 |                     0.7432 |            0.7472 |                    0.6911 |                             0.7432 |                    0.7474 |
|   Deva |      32 |            0.8109 |                     0.8718 |            0.9311 |                    0.8109 |                             0.8718 |                    0.9317 |
|   Beng |       6 |            0.8857 |                     0.9462 |            0.9844 |                    0.8858 |                             0.9462 |                    0.9846 |
|   Grek |       4 |            0.6762 |                     0.7344 |            0.9252 |                    0.6769 |                             0.7345 |                    0.9253 |
|   Hebr |       4 |            0.6966 |                     0.7787 |            0.9665 |                    0.7401 |                             0.7787 |                    0.9666 |
|   Armn |       2 |            0.9742 |                     0.9741 |            0.9855 |                    0.9742 |                             0.9741 |                    0.9856 |
|  Other |      84 |            0.9348 |                     0.9464 |            0.9721 |                    0.9384 |                             0.9466 |                    0.9727 |

Script-table reproduction gate (within-stratum): PASSED.

### Resource-tier breakdown, both views (global = per-language F1 with all false positives counted; within-stratum = examples restricted to true labels in the group, cross-group false positives excluded; full kept pool, 45,377,279 lines)

| Resource | # Langs | baseline (global) | gate-flat4-prox21 (global) | fastText (global) | baseline (within-stratum) | gate-flat4-prox21 (within-stratum) | fastText (within-stratum) |
|----------|---------|-------------------|----------------------------|-------------------|---------------------------|------------------------------------|---------------------------|
|     <500 |      56 |            0.5145 |                     0.7796 |            0.7497 |                    0.8709 |                             0.8272 |                    0.9150 |
|  500--1k |      40 |            0.6280 |                     0.8924 |            0.8614 |                    0.9748 |                             0.9553 |                    0.9641 |
|  1k--12k |     458 |            0.8912 |                     0.9449 |            0.9414 |                    0.9897 |                             0.9866 |                    0.9787 |
| 12k--18k |     526 |            0.9790 |                     0.9850 |            0.9708 |                    0.9972 |                             0.9966 |                    0.9864 |
| 18k--35k |     398 |            0.9628 |                     0.9651 |            0.9526 |                    0.9919 |                             0.9918 |                    0.9808 |
|     35k+ |     462 |            0.9575 |                     0.9568 |            0.9408 |                    0.9583 |                             0.9577 |                    0.9421 |

Resource-tier reproduction gate (within-stratum): PASSED.


Git commit: 9b1ed200c433589ede4b982ac76b0cd6e8b07157.
