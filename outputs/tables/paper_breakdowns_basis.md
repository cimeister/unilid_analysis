# Camera-ready E4: label-basis diagnostic

Pre-registration: EXPERIMENTS_PLAN.md, "Camera-ready evaluation program (2026-08-06)", E4 bullet. Inspects outputs/diagnostic/paper_team/fasttext_folder/glotlid_fasttext_e100_sanity_per_language.json (the paper team's own per-language fastText results) to see whether it explains the paper's stated 1,938-language script-table basis against our 1,940 languages and the Hebr-row mismatch.

## Label inventory

- outputs/diagnostic/paper_team/fasttext_folder/glotlid_fasttext_e100_sanity_per_language.json: 1,940 labels.
- Our canonical language list: 1,940 labels.
- Absent from the JSON, present in ours: 0.
- Present in the JSON, absent from ours: 0.

## Self-check

Unweighted mean of the JSON's own 1,940 f1 values: 0.9443269.
- vs sibling outputs/diagnostic/paper_team/fasttext_folder/glotlid_fasttext_e100_sanity_metrics.json macro_f1 0.9443269 (diff 2.78e-15).
- vs recorded reference 0.9443269 (diff 2.56e-08, tol 1e-06): OK.

## Hypothesis test: does the JSON explain the 1,938-vs-1,940 basis?

NOT CONFIRMED: the JSON has 1,940 labels, not the hypothesized 1,938. 0 of our 1,940 labels are missing from it and 0 labels in it are not in ours. This diagnostic does not resolve the 1,938-vs-1,940 basis or the Hebr-row discrepancy; both remain open (ask list item 3, ~/.claude/plans/steady-finding-abelson.md, "What to ask Ahmetcan for").

## JSON's own script-group means vs the paper's fastText column (all 1,940 JSON labels, no exclusion)

### JSON per-language f1, grouped by script, vs paper/submission.tex fastText column

| group | our # langs | paper # langs | our F1 | paper F1 |    diff |   status |
|-------|-------------|---------------|--------|----------|---------|----------|
|  Latn |        1700 |          1700 | 0.9464 |   0.9460 | +0.0004 |       OK |
|  Cyrl |          70 |            70 | 0.9684 |   0.9700 | -0.0016 |       OK |
|  Arab |          38 |            38 | 0.7473 |   0.7470 | +0.0003 |       OK |
|  Deva |          32 |            32 | 0.9311 |   0.9320 | -0.0009 |       OK |
|  Beng |           6 |             6 | 0.9843 |   0.9850 | -0.0007 |       OK |
|  Grek |           4 |             4 | 0.9252 |   0.9250 | +0.0002 |       OK |
|  Hebr |           4 |             4 | 0.9665 |   0.9670 | -0.0005 |       OK |
|  Armn |           2 |             2 | 0.9855 |   0.9860 | -0.0005 |       OK |
| Other |          84 |            82 | 0.9721 |   0.9730 | -0.0009 | MISMATCH |


Sum of the JSON's own support values: 45,627,279 (total_samples in outputs/diagnostic/paper_team/fasttext_folder/glotlid_fasttext_e100_sanity_metrics.json: 45,627,279). Note this may differ from our full kept pool (45,377,279 lines) if the JSON's run scored a different line set.
