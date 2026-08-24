# Camera-ready E4: script and resource-tier reproduction gates

**NON-DEFAULT MODEL RUN.** Every number below was computed from `/capstor/scratch/cscs/cmeister747/unilid_analysis/corrected/glotlidc_corrected.unilid` and its predictions under `/capstor/scratch/cscs/cmeister747/unilid_analysis/full_test_eval_corrected`, not from the released model (`/capstor/scratch/cscs/cmeister747/unilid_analysis/glotlidc.unilid`), and must not be read as a restatement of the released model's tables. The comparisons below against paper/submission.tex are INFORMATIONAL for this run, not gates: those published cells were produced by the RELEASED model, so a difference against them is the expected outcome of a cross-model comparison, not a regression. They are computed and reported in full, and they withhold nothing: the .tex fragments below carry THIS run's regenerated numbers whether or not they match the published cells, and the script does not exit nonzero for such a difference. Under the released model the same comparisons are binding gates and a MISMATCH still withholds the affected .tex and exits 1.

Full comparison tables (every group, OK or MISMATCH), tolerance 0.005 on F1 and exact match on language count. Non-fatal: a MISMATCH blocks only the affected .tex output, not this report, and under this run's non-default model it blocks nothing at all (see the banner above). The gates below compare the WITHIN-STRATUM view (EXPERIMENTS_CHRONOLOGICAL.md, "provenance of the paper's appendix breakdown tables resolved", 2026-08-07): both paper appendix tables turned out to be within-stratum, not the global per-language view this script used to gate against previously.

Our columns below are the full kept pool (45,377,279 lines), in the view stated per table (global or within-stratum). The paper team's own computation used a different basis: their own metrics JSON (outputs/diagnostic/paper_team/fasttext_folder/glotlid_fasttext_e100_sanity_metrics.json) states total_samples 45,627,279 (the full raw test-file line count, not restricted to our kept pool).

## Script-table reproduction gate (within-stratum)

Our recomputed within-stratum baseline column vs paper/submission.tex, tab:script-breakdown (UniLID column). Other's row uses the paper's own basis: our Other group minus jpn_Jpan and kor_Hang.

### Script-table reproduction gate (within-stratum)

| group | our # langs | paper # langs | our F1 | paper F1 |    diff |   status |
|-------|-------------|---------------|--------|----------|---------|----------|
|  Latn |        1700 |          1700 | 0.9443 |   0.9400 | +0.0043 |       OK |
|  Cyrl |          70 |            70 | 0.8801 |   0.8770 | +0.0031 |       OK |
|  Arab |          38 |            38 | 0.6926 |   0.6910 | +0.0016 |       OK |
|  Deva |          32 |            32 | 0.8109 |   0.8110 | -0.0001 |       OK |
|  Beng |           6 |             6 | 0.8790 |   0.8850 | -0.0060 | MISMATCH |
|  Grek |           4 |             4 | 0.6750 |   0.6770 | -0.0020 |       OK |
|  Hebr |           4 |             4 | 0.7376 |   0.7400 | -0.0024 |       OK |
|  Armn |           2 |             2 | 0.9721 |   0.9740 | -0.0019 |       OK |
| Other |          82 |            82 | 0.9374 |   0.9370 | +0.0004 |       OK |


## RECORDED: global-view comparison against the paper's script table (expected-mismatch cross-view comparison, NOT a gate)

### Global baseline F1 (global per-language F1, all false positives counted, averaged over languages in the group) vs paper/submission.tex, tab:script-breakdown; expected mismatch, the paper is within-stratum

| group | our # langs | paper # langs | our F1 | paper F1 |    diff |   status |
|-------|-------------|---------------|--------|----------|---------|----------|
|  Latn |        1700 |          1700 | 0.9440 |   0.9400 | +0.0040 |       OK |
|  Cyrl |          70 |            70 | 0.8741 |   0.8770 | -0.0029 |       OK |
|  Arab |          38 |            38 | 0.6914 |   0.6910 | +0.0004 |       OK |
|  Deva |          32 |            32 | 0.8109 |   0.8110 | -0.0001 |       OK |
|  Beng |           6 |             6 | 0.8790 |   0.8850 | -0.0060 | MISMATCH |
|  Grek |           4 |             4 | 0.6741 |   0.6770 | -0.0029 |       OK |
|  Hebr |           4 |             4 | 0.6943 |   0.7400 | -0.0457 | MISMATCH |
|  Armn |           2 |             2 | 0.9720 |   0.9740 | -0.0020 |       OK |
| Other |          84 |            82 | 0.9354 |   0.9370 | -0.0016 | MISMATCH |


## Resource-tier reproduction gate (within-stratum)

Our recomputed within-stratum baseline column vs paper/submission.tex, tab:resource-tier (UniLID F1 column).

### Resource-tier reproduction gate (within-stratum)

|    group | our # langs | paper # langs | our F1 | paper F1 |    diff |   status |
|----------|-------------|---------------|--------|----------|---------|----------|
|     <500 |          56 |            56 | 0.8572 |   0.8710 | -0.0138 | MISMATCH |
|  500--1k |          40 |            40 | 0.9731 |   0.9750 | -0.0019 |       OK |
|  1k--12k |         458 |           458 | 0.9895 |   0.9900 | -0.0005 |       OK |
| 12k--18k |         526 |           526 | 0.9971 |   0.9970 | +0.0001 |       OK |
| 18k--35k |         398 |           398 | 0.9918 |   0.9920 | -0.0002 |       OK |
|     35k+ |         462 |           462 | 0.9576 |   0.9580 | -0.0004 |       OK |


## RECORDED: global-view comparison against the paper's resource-tier table (expected-mismatch cross-view comparison, NOT a gate)

### Global baseline F1 (global per-language F1, all false positives counted, averaged over languages in the group) vs paper/submission.tex, tab:resource-tier; expected mismatch, the paper is within-stratum

|    group | our # langs | paper # langs | our F1 | paper F1 |    diff |   status |
|----------|-------------|---------------|--------|----------|---------|----------|
|     <500 |          56 |            56 | 0.5957 |   0.8710 | -0.2753 | MISMATCH |
|  500--1k |          40 |            40 | 0.6762 |   0.9750 | -0.2988 | MISMATCH |
|  1k--12k |         458 |           458 | 0.8936 |   0.9900 | -0.0964 | MISMATCH |
| 12k--18k |         526 |           526 | 0.9786 |   0.9970 | -0.0184 | MISMATCH |
| 18k--35k |         398 |           398 | 0.9622 |   0.9920 | -0.0298 | MISMATCH |
|     35k+ |         462 |           462 | 0.9568 |   0.9580 | -0.0012 |       OK |

