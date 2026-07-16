|   gamma |   val_overall |   val_tail |   val_head |   val_twins |   val_magnets |
|--------:|--------------:|-----------:|-----------:|------------:|--------------:|
|    0    |        0.9451 |     0.871  |     0.9601 |      0.9246 |        0.8797 |
|    0.25 |        0.948  |     0.871  |     0.9605 |      0.9252 |        0.8758 |
|    0.5  |        0.9512 |     0.871  |     0.9612 |      0.9262 |        0.8716 |
|    1    |        0.956  |     0.871  |     0.9625 |      0.9124 |        0.8595 |
|    1.5  |        0.9597 |     0.871  |     0.9636 |      0.9129 |        0.8548 |
|    2    |        0.962  |     0.8387 |     0.9645 |      0.9128 |        0.8242 |
|    3    |        0.9631 |     0.6989 |     0.9658 |      0.9122 |        0.7313 |
|    5    |        0.9639 |     0.6129 |     0.9686 |      0.9108 |        0.6392 |

# Per-language frequency prior (b_L = gamma*log N_L) — TEST evaluation

Best gamma on val: **0.5** (gamma=0 means no eligible operating point).
Selection guard: val overall must improve and no stratum (tail/magnets/twins/head) may drop more than 0.01 vs gamma=0.

| stratum | base macroF1 | prior macroF1 | delta | 95% CI |
|---|---|---|---|---|
| overall | 0.9454 | 0.9524 | +0.0058 | [+0.0048, +0.0069] |
| magnets | 0.8832 | 0.8811 | -0.0026 | [-0.0130, +0.0035] |
| tail | 0.9310 | 0.9310 | +0.0000 | [+0.0000, +0.0000] |
| twins | 0.9224 | 0.9243 | +0.0019 | [+0.0004, +0.0042] |
| head | 0.9603 | 0.9615 | +0.0012 | [+0.0008, +0.0017] |

Overall accuracy: 0.9603 -> 0.9634 (+0.0032).