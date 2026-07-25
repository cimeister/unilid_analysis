| config    |   n_tied |   val_overall |   val_tail |   val_head |   val_twins |   val_magnets |
|:----------|---------:|--------------:|-----------:|-----------:|------------:|--------------:|
| baseline  |        0 |        0.9451 |      0.871 |     0.9601 |      0.9246 |        0.8797 |
| dp_global |      212 |        0.9437 |      0.871 |     0.96   |      0.9186 |        0.879  |
| dp_script |      212 |        0.9435 |      0.871 |     0.9599 |      0.9143 |        0.879  |

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