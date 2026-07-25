| config   |   n_modified |   val_overall |   val_tail |   val_head |   val_twins |   val_magnets |
|:---------|-------------:|--------------:|-----------:|-----------:|------------:|--------------:|
| baseline |            0 |        0.9451 |      0.871 |     0.9601 |      0.9246 |        0.8797 |
| floor-17 |          452 |        0.9475 |      0.871 |     0.9597 |      0.9251 |        0.8797 |
| floor-19 |         1821 |        0.9488 |      0.871 |     0.9598 |      0.9256 |        0.8797 |
| floor-21 |         1940 |        0.9489 |      0.871 |     0.9596 |      0.9251 |        0.8797 |
| floor-23 |         1940 |        0.9486 |      0.871 |     0.9591 |      0.9248 |        0.8797 |

# Downward floor equalization — TEST evaluation

Best config selected on val: **floor-21** (baseline means nothing passed the guard). Baseline agreement 0.9951.
Floor plateaus clamped to min(floor_L, F); observed tokens and specials bit-identical; no entry is ever raised.
Selection guard: val overall must improve and no stratum (tail/magnets/twins/head) may drop more than 0.01 vs baseline.

| stratum | base macroF1 | equalized macroF1 | delta | 95% CI |
|---|---|---|---|---|
| overall | 0.9454 | 0.9494 | +0.0030 | [+0.0016, +0.0044] |
| magnets | 0.8832 | 0.8630 | -0.0108 | [-0.0429, +0.0295] |
| tail | 0.9310 | 0.8621 | -0.0623 | [-0.1111, +0.0000] |
| twins | 0.9224 | 0.9228 | +0.0003 | [-0.0004, +0.0014] |
| head | 0.9603 | 0.9600 | -0.0003 | [-0.0009, +0.0003] |

Overall accuracy: 0.9603 -> 0.9612 (+0.0009).