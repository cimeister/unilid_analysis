# Camera-ready E4: script and resource-tier reproduction gates

Full comparison tables (every group, OK or MISMATCH), tolerance 0.005 on F1 and exact match on language count. Non-fatal: a MISMATCH blocks only the affected .tex output, not this report. The gates below compare the WITHIN-STRATUM view (EXPERIMENTS_CHRONOLOGICAL.md, "provenance of the paper's appendix breakdown tables resolved", 2026-08-07): both paper appendix tables turned out to be within-stratum, not the global per-language view this script used to gate against previously.

Our columns below are the full kept pool (45,377,279 lines), in the view stated per table (global or within-stratum). The paper team's own computation used a different basis: their own metrics JSON (outputs/diagnostic/paper_team/fasttext_folder/glotlid_fasttext_e100_sanity_metrics.json) states total_samples 45,627,279 (the full raw test-file line count, not restricted to our kept pool).

## Script-table reproduction gate (within-stratum)

Our recomputed within-stratum baseline column vs paper/submission.tex, tab:script-breakdown (UniLID column). Other's row uses the paper's own basis: our Other group minus jpn_Jpan and kor_Hang.

### Script-table reproduction gate (within-stratum)

| group | our # langs | paper # langs | our F1 | paper F1 |    diff | status |
|-------|-------------|---------------|--------|----------|---------|--------|
|  Latn |        1700 |          1700 | 0.9404 |   0.9400 | +0.0004 |     OK |
|  Cyrl |          70 |            70 | 0.8774 |   0.8770 | +0.0004 |     OK |
|  Arab |          38 |            38 | 0.6911 |   0.6910 | +0.0001 |     OK |
|  Deva |          32 |            32 | 0.8109 |   0.8110 | -0.0001 |     OK |
|  Beng |           6 |             6 | 0.8858 |   0.8850 | +0.0008 |     OK |
|  Grek |           4 |             4 | 0.6769 |   0.6770 | -0.0001 |     OK |
|  Hebr |           4 |             4 | 0.7401 |   0.7400 | +0.0001 |     OK |
|  Armn |           2 |             2 | 0.9742 |   0.9740 | +0.0002 |     OK |
| Other |          82 |            82 | 0.9374 |   0.9370 | +0.0004 |     OK |


## RECORDED: global-view comparison against the paper's script table (expected-mismatch cross-view comparison, NOT a gate)

### Global baseline F1 (global per-language F1, all false positives counted, averaged over languages in the group) vs paper/submission.tex, tab:script-breakdown; expected mismatch, the paper is within-stratum

| group | our # langs | paper # langs | our F1 | paper F1 |    diff |   status |
|-------|-------------|---------------|--------|----------|---------|----------|
|  Latn |        1700 |          1700 | 0.9401 |   0.9400 | +0.0001 |       OK |
|  Cyrl |          70 |            70 | 0.8712 |   0.8770 | -0.0058 | MISMATCH |
|  Arab |          38 |            38 | 0.6899 |   0.6910 | -0.0011 |       OK |
|  Deva |          32 |            32 | 0.8109 |   0.8110 | -0.0001 |       OK |
|  Beng |           6 |             6 | 0.8857 |   0.8850 | +0.0007 |       OK |
|  Grek |           4 |             4 | 0.6762 |   0.6770 | -0.0008 |       OK |
|  Hebr |           4 |             4 | 0.6966 |   0.7400 | -0.0434 | MISMATCH |
|  Armn |           2 |             2 | 0.9742 |   0.9740 | +0.0002 |       OK |
| Other |          84 |            82 | 0.9348 |   0.9370 | -0.0022 | MISMATCH |


## Resource-tier reproduction gate (within-stratum)

Our recomputed within-stratum baseline column vs paper/submission.tex, tab:resource-tier (UniLID F1 column).

### Resource-tier reproduction gate (within-stratum)

|    group | our # langs | paper # langs | our F1 | paper F1 |    diff | status |
|----------|-------------|---------------|--------|----------|---------|--------|
|     <500 |          56 |            56 | 0.8709 |   0.8710 | -0.0001 |     OK |
|  500--1k |          40 |            40 | 0.9748 |   0.9750 | -0.0002 |     OK |
|  1k--12k |         458 |           458 | 0.9897 |   0.9900 | -0.0003 |     OK |
| 12k--18k |         526 |           526 | 0.9972 |   0.9970 | +0.0002 |     OK |
| 18k--35k |         398 |           398 | 0.9919 |   0.9920 | -0.0001 |     OK |
|     35k+ |         462 |           462 | 0.9583 |   0.9580 | +0.0003 |     OK |


## RECORDED: global-view comparison against the paper's resource-tier table (expected-mismatch cross-view comparison, NOT a gate)

### Global baseline F1 (global per-language F1, all false positives counted, averaged over languages in the group) vs paper/submission.tex, tab:resource-tier; expected mismatch, the paper is within-stratum

|    group | our # langs | paper # langs | our F1 | paper F1 |    diff |   status |
|----------|-------------|---------------|--------|----------|---------|----------|
|     <500 |          56 |            56 | 0.5145 |   0.8710 | -0.3565 | MISMATCH |
|  500--1k |          40 |            40 | 0.6280 |   0.9750 | -0.3470 | MISMATCH |
|  1k--12k |         458 |           458 | 0.8912 |   0.9900 | -0.0988 | MISMATCH |
| 12k--18k |         526 |           526 | 0.9790 |   0.9970 | -0.0180 | MISMATCH |
| 18k--35k |         398 |           398 | 0.9628 |   0.9920 | -0.0292 | MISMATCH |
|     35k+ |         462 |           462 | 0.9575 |   0.9580 | -0.0005 |       OK |

