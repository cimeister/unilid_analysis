# Full-test-set evaluation of the pooled-token-frequency unseen-token rule (bgfloor, Exp 50)

One new scoring pass under the bgfloor matrix (45,377,279 lines); baseline predictions and y_true reused from the Exp 16 run (job 2784115). c=-8.4740 (-21.0 minus the mean of log_p_base over the non-special vocabulary); no per-language fitting; no selection here.
Bootstrap CIs (B=1000) for strata with at most 3,000,000 examples; larger strata report the point delta only.

| stratum | base macroF1 | bgfloor macroF1 | delta | 95% CI |
|---|---|---|---|---|
| overall (45,377,279) | 0.9292 | 0.9424 | +0.0132 | point only (n > cap) |
| tail (7,735) | 0.9132 | 0.8929 | -0.0203 | [-0.0250, -0.0159] |
| magnets (64,657) | 0.9138 | 0.8975 | -0.0163 | [-0.0206, -0.0130] |
| twins (9,156,023) | 0.9167 | 0.9170 | +0.0004 | point only (n > cap) |
| head (43,665,835) | 0.9602 | 0.9602 | +0.0000 | point only (n > cap) |

Overall accuracy: 0.9608 -> 0.9620 (+0.0012).

The computed constant is c = -8.4740. The standard deviation of log_p_base over the non-special vocabulary is 0.9824 nats. The p1-to-p99 spread of log_p_base over the non-special vocabulary is 4.9010 nats. The minimum assigned plateau value is -27.6066. The maximum assigned plateau value is -12.3143.