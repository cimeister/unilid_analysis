|   gamma |   val_overall |   val_tail |   val_head |   val_twins |   val_magnets |
|--------:|--------------:|-----------:|-----------:|------------:|--------------:|
|    0    |        0.9415 |     0.8387 |     0.9607 |      0.9239 |        0.8693 |
|    0.25 |        0.9453 |     0.8065 |     0.9611 |      0.925  |        0.8451 |
|    0.5  |        0.9499 |     0.8065 |     0.9616 |      0.9248 |        0.8462 |
|    1    |        0.9554 |     0.7957 |     0.9633 |      0.9248 |        0.8162 |
|    1.5  |        0.959  |     0.7957 |     0.9644 |      0.9248 |        0.7924 |
|    2    |        0.9612 |     0.7634 |     0.9651 |      0.9267 |        0.7664 |
|    3    |        0.9638 |     0.6989 |     0.9671 |      0.9263 |        0.7217 |
|    5    |        0.9626 |     0.5806 |     0.9689 |      0.9119 |        0.6188 |

# Per-language frequency prior (b_L = gamma*log N_L) — TEST evaluation

Best gamma on val: **0.0** (gamma=0 means no eligible operating point).
Selection guard: val overall must improve and no stratum (tail/magnets/twins/head) may drop more than 0.01 vs gamma=0.

| stratum | base macroF1 | prior macroF1 | delta | 95% CI |
|---|---|---|---|---|
| overall | 0.9447 | 0.9447 | +0.0000 | [+0.0000, +0.0000] |
| magnets | 0.8999 | 0.8999 | +0.0000 | [+0.0000, +0.0000] |
| tail | 0.8966 | 0.8966 | +0.0000 | [+0.0000, +0.0000] |
| twins | 0.9219 | 0.9219 | +0.0000 | [+0.0000, +0.0000] |
| head | 0.9608 | 0.9608 | +0.0000 | [+0.0000, +0.0000] |

Overall accuracy: 0.9644 -> 0.9644 (+0.0000).