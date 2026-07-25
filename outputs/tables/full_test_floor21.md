# Full-test-set evaluation of floor equalization (F=-21)

One new scoring pass under the floor-21 matrix (45,377,279 lines); baseline predictions and y_true reused from the Exp 16 run (job 2784115). Config fixed on val by the Exp 20 guard; no selection here.
Bootstrap CIs (B=1000) for strata with at most 3,000,000 examples; larger strata report the point delta only.

| stratum | base macroF1 | floor-21 macroF1 | delta | 95% CI |
|---|---|---|---|---|
| overall (45,377,279) | 0.9292 | 0.9421 | +0.0129 | point only (n > cap) |
| tail (7,735) | 0.9132 | 0.8928 | -0.0204 | [-0.0257, -0.0161] |
| magnets (64,657) | 0.9138 | 0.8974 | -0.0164 | [-0.0210, -0.0129] |
| twins (9,156,023) | 0.9167 | 0.9166 | -0.0001 | point only (n > cap) |
| head (43,665,835) | 0.9602 | 0.9599 | -0.0003 | point only (n > cap) |

Overall accuracy: 0.9608 -> 0.9616 (+0.0009).