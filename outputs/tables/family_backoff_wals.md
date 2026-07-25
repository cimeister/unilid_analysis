| config           |   val_overall |   val_tail |   val_head |   val_twins |   val_magnets |
|:-----------------|--------------:|-----------:|-----------:|------------:|--------------:|
| baseline         |        0.9451 |     0.871  |     0.9601 |      0.9246 |        0.8797 |
| wals_lift_a300   |        0.9415 |     0.871  |     0.9603 |      0.9251 |        0.8797 |
| wals_lift_a3000  |        0.9291 |     0.8387 |     0.959  |      0.9207 |        0.861  |
| wals_lift_a30000 |        0.9147 |     0.8387 |     0.955  |      0.9147 |        0.8627 |
| wals_full_a300   |        0.942  |     0.871  |     0.9595 |      0.9235 |        0.8797 |
| wals_full_a3000  |        0.9296 |     0.8387 |     0.9587 |      0.9193 |        0.861  |
| wals_full_a30000 |        0.9147 |     0.8387 |     0.9549 |      0.9138 |        0.8627 |

# Group-mean back-off at floor positions — TEST evaluation

Best config selected on val: **baseline** (baseline means nothing passed the guard). Baseline agreement 0.9951.
grouping=wals: tiers {'script': 1012, 'family_script': 360, 'genus_script': 535, 'none': 33}; 33 languages unmodified
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