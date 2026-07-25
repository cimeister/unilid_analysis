# Full-test-set evaluation of the Good-Turing unseen-mass candidate (gt_min)

One new scoring pass under the gt_min matrix (45,377,279 lines); baseline predictions and y_true reused from the Exp 16 run. Single pre-registered candidate, no sweep; adoption judged by analysis/two_sided_report.py.
Bootstrap CIs (B=1000) for strata with at most 3,000,000 examples; larger strata report the point delta only.

| stratum | base macroF1 | gt_min macroF1 | delta | 95% CI |
|---|---|---|---|---|
| overall (45,377,279) | 0.9292 | 0.9256 | -0.0035 | point only (n > cap) |
| tail (7,735) | 0.9132 | 0.9788 | +0.0656 | [+0.0603, +0.0729] |
| magnets (64,657) | 0.9138 | 0.9667 | +0.0528 | [+0.0478, +0.0588] |
| twins (9,156,023) | 0.9167 | 0.9166 | -0.0001 | point only (n > cap) |
| head (43,665,835) | 0.9602 | 0.9604 | +0.0002 | point only (n > cap) |

Overall accuracy: 0.9608 -> 0.9592 (-0.0016).