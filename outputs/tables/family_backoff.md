| config      |   val_overall |   val_tail |   val_head |   val_twins |   val_magnets |
|:------------|--------------:|-----------:|-----------:|------------:|--------------:|
| baseline    |        0.9451 |     0.871  |     0.9601 |      0.9246 |        0.8797 |
| lift_a300   |        0.9422 |     0.871  |     0.9604 |      0.9252 |        0.8797 |
| lift_a3000  |        0.9297 |     0.8387 |     0.9591 |      0.9216 |        0.8609 |
| lift_a30000 |        0.9162 |     0.8387 |     0.9545 |      0.9177 |        0.8626 |
| full_a300   |        0.9423 |     0.871  |     0.9602 |      0.9249 |        0.8797 |
| full_a3000  |        0.93   |     0.8387 |     0.9592 |      0.9216 |        0.8609 |
| full_a30000 |        0.9168 |     0.8387 |     0.9545 |      0.9178 |        0.8626 |

# Script-mean back-off at floor positions — TEST evaluation

Best config selected on val: **baseline** (baseline means nothing passed the guard). Baseline agreement 0.9951. 33 languages have no backbone and stay unmodified.
lam_L = alpha/(N_L+alpha); modes: lift = raise floor entries only, full = replace floor entries in both directions.
Selection guard: val overall must improve and no stratum (tail/magnets/twins/head) may drop more than 0.01 vs baseline.

| stratum | base macroF1 | backoff macroF1 | delta | 95% CI |
|---|---|---|---|---|
| overall | 0.9454 | 0.9454 | +0.0000 | [+0.0000, +0.0000] |
| magnets | 0.8832 | 0.8832 | +0.0000 | [+0.0000, +0.0000] |
| tail | 0.9310 | 0.9310 | +0.0000 | [+0.0000, +0.0000] |
| twins | 0.9224 | 0.9224 | +0.0000 | [+0.0000, +0.0000] |
| head | 0.9603 | 0.9603 | +0.0000 | [+0.0000, +0.0000] |

Overall accuracy: 0.9603 -> 0.9603 (+0.0000).