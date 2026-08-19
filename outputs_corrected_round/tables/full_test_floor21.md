# Full-test-set evaluation of floor equalization (F=-21)

One new scoring pass under the floor-21 matrix (45,377,279 lines); baseline predictions and y_true reused from the Exp 16 run (job 2784115). Config fixed on val by the Exp 20 guard; no selection here.
Bootstrap CIs (B=1000) for strata with at most 3,000,000 examples; larger strata report the point delta only.

| stratum | base macroF1 | floor-21 macroF1 | delta | 95% CI |
|---|---|---|---|---|
| overall (45,377,279) | 0.9327 | 0.9419 | +0.0092 | point only (n > cap) |
| tail (7,735) | 0.9045 | 0.8875 | -0.0171 | [-0.0222, -0.0127] |
| magnets (64,657) | 0.9067 | 0.8928 | -0.0139 | [-0.0181, -0.0105] |
| twins (9,156,023) | 0.9164 | 0.9166 | +0.0002 | point only (n > cap) |
| head (43,665,835) | 0.9596 | 0.9595 | -0.0001 | point only (n > cap) |

Overall accuracy: 0.9609 -> 0.9617 (+0.0008).