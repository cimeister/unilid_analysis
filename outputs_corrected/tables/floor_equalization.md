| config        |   n_modified |   val_overall |   val_tail |   val_head |   val_twins |   val_magnets |
|:--------------|-------------:|--------------:|-----------:|-----------:|------------:|--------------:|
| baseline      |            0 |        0.9453 |      0.871 |     0.9598 |      0.924  |        0.8811 |
| floor-15.3906 |          452 |        0.9474 |      0.871 |     0.9595 |      0.9244 |        0.8811 |
| floor-17.3906 |         1821 |        0.9486 |      0.871 |     0.9595 |      0.9245 |        0.8811 |
| floor-19.3906 |         1940 |        0.9484 |      0.871 |     0.9594 |      0.924  |        0.8811 |
| floor-21.3906 |         1940 |        0.9481 |      0.871 |     0.9588 |      0.9228 |        0.8811 |

# Downward floor equalization — TEST evaluation

Best config selected on val: **floor-17.3906** (baseline means nothing passed the guard). Baseline agreement 0.9916.
Floor plateaus clamped to min(floor_L, F); observed tokens and specials bit-identical; no entry is ever raised.
Selection guard: val overall must improve and no stratum (tail/magnets/twins/head) may drop more than 0.01 vs baseline.

| stratum | base macroF1 | equalized macroF1 | delta | 95% CI |
|---|---|---|---|---|
| overall | 0.9460 | 0.9488 | +0.0019 | [+0.0006, +0.0033] |
| magnets | 0.8968 | 0.8654 | -0.0218 | [-0.0455, +0.0114] |
| tail | 0.9310 | 0.8621 | -0.0623 | [-0.1111, +0.0000] |
| twins | 0.9213 | 0.9217 | +0.0003 | [-0.0002, +0.0012] |
| head | 0.9595 | 0.9594 | -0.0001 | [-0.0003, +0.0002] |

Overall accuracy: 0.9604 -> 0.9611 (+0.0008).