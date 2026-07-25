| config         |   n_tied |   val_overall |   val_tail |   val_head |   val_twins |   val_magnets |
|:---------------|---------:|--------------:|-----------:|-----------:|------------:|--------------:|
| baseline       |        0 |        0.9451 |     0.871  |     0.9601 |      0.9246 |        0.8797 |
| digits_ws      |      298 |        0.9441 |     0.871  |     0.9604 |      0.9233 |        0.8762 |
| nonalpha_ascii |      479 |        0.9388 |     0.871  |     0.9577 |      0.9106 |        0.871  |
| nonalpha_all   |     1291 |        0.9373 |     0.8602 |     0.9566 |      0.9073 |        0.8629 |

# Non-content token tying — TEST evaluation

Best config selected on val: **baseline** (baseline means nothing passed the guard). Baseline agreement 0.9951.
Selection guard: val overall must improve and no stratum (tail/magnets/twins/head) may drop more than 0.01 vs baseline.

| stratum | base macroF1 | tied macroF1 | delta | 95% CI |
|---|---|---|---|---|
| overall | 0.9454 | 0.9454 | +0.0000 | [+0.0000, +0.0000] |
| magnets | 0.8832 | 0.8832 | +0.0000 | [+0.0000, +0.0000] |
| tail | 0.9310 | 0.9310 | +0.0000 | [+0.0000, +0.0000] |
| twins | 0.9224 | 0.9224 | +0.0000 | [+0.0000, +0.0000] |
| head | 0.9603 | 0.9603 | +0.0000 | [+0.0000, +0.0000] |

Overall accuracy: 0.9603 -> 0.9603 (+0.0000).