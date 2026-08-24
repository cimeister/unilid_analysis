# Full-test-set evaluation (test set minus the 250k val lines)

Lines evaluated: 45,377,279 of 45,627,279. Configurations fixed on val: frequency prior gamma=0.5, learned bias = reg=5.0 fit (job 2731802). Empty-preprocess lines are scored as wrong, matching the 500k-sample sweeps.
Bootstrap CIs (B=1000) are computed for strata with at most 3,000,000 examples; for larger strata the item-level CI half-width is below ~0.001 and only the point delta is reported.

## overall: 45,377,279 examples, 1940/1940 languages with test support

| config | macroF1 | delta | 95% CI | accuracy |
|---|---|---|---|---|
| baseline | 0.9089 | | | 0.9617 |

## tail: 7,735 examples, 96/96 languages with test support

| config | macroF1 | delta | 95% CI | accuracy |
|---|---|---|---|---|
| baseline | 0.8229 | | | 0.8421 |

## magnets: 64,657 examples, 118/118 languages with test support

| config | macroF1 | delta | 95% CI | accuracy |
|---|---|---|---|---|
| baseline | 0.8337 | | | 0.9324 |

## twins: 9,156,023 examples, 77/77 languages with test support

| config | macroF1 | delta | 95% CI | accuracy |
|---|---|---|---|---|
| baseline | 0.8997 | | | 0.9298 |

## head: 43,665,835 examples, 860/860 languages with test support

| config | macroF1 | delta | 95% CI | accuracy |
|---|---|---|---|---|
| baseline | 0.9537 | | | 0.9606 |
