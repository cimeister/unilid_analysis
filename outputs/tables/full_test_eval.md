# Full-test-set evaluation (test set minus the 250k val lines)

Lines evaluated: 45,377,279 of 45,627,279. Configurations fixed on val: frequency prior gamma=0.5, learned bias = reg=5.0 fit (job 2731802). Empty-preprocess lines are scored as wrong, matching the 500k-sample sweeps.
Bootstrap CIs (B=1000) are computed for strata with at most 3,000,000 examples; for larger strata the item-level CI half-width is below ~0.001 and only the point delta is reported.

## overall: 45,377,279 examples, 1940/1940 languages with test support

| config | macroF1 | delta | 95% CI | accuracy |
|---|---|---|---|---|
| baseline | 0.9292 | | | 0.9608 |
| freq_prior | 0.9408 | +0.0116 | point only (n > cap) | 0.9638 |
| learned_bias | 0.9421 | +0.0129 | point only (n > cap) | 0.9751 |

## tail: 7,735 examples, 96/96 languages with test support

| config | macroF1 | delta | 95% CI | accuracy |
|---|---|---|---|---|
| baseline | 0.9132 | | | 0.9158 |
| freq_prior | 0.8950 | -0.0182 | [-0.0225, -0.0146] | 0.8998 |
| learned_bias | 0.9114 | -0.0018 | [-0.0035, -0.0001] | 0.9131 |

## magnets: 64,657 examples, 118/118 languages with test support

| config | macroF1 | delta | 95% CI | accuracy |
|---|---|---|---|---|
| baseline | 0.9138 | | | 0.9451 |
| freq_prior | 0.8965 | -0.0173 | [-0.0207, -0.0141] | 0.9354 |
| learned_bias | 0.9056 | -0.0082 | [-0.0099, -0.0066] | 0.9091 |

## twins: 9,156,023 examples, 77/77 languages with test support

| config | macroF1 | delta | 95% CI | accuracy |
|---|---|---|---|---|
| baseline | 0.9167 | | | 0.9251 |
| freq_prior | 0.9178 | +0.0011 | point only (n > cap) | 0.9275 |
| learned_bias | 0.9283 | +0.0116 | point only (n > cap) | 0.9513 |

## head: 43,665,835 examples, 860/860 languages with test support

| config | macroF1 | delta | 95% CI | accuracy |
|---|---|---|---|---|
| baseline | 0.9602 | | | 0.9595 |
| freq_prior | 0.9613 | +0.0011 | point only (n > cap) | 0.9626 |
| learned_bias | 0.9703 | +0.0101 | point only (n > cap) | 0.9744 |
