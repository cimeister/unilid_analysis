|   reg |   val_recall@k |   val_overall |   val_tail |   val_head |   val_twins |   val_magnets |
|------:|---------------:|--------------:|-----------:|-----------:|------------:|--------------:|
|   0   |         0.9971 |        0.9451 |      0.871 |     0.9601 |      0.9246 |        0.8797 |
|   0.3 |         0.9971 |        0.9631 |      0.871 |     0.9736 |      0.9365 |        0.8479 |
|   1   |         0.9971 |        0.9609 |      0.871 |     0.9728 |      0.9379 |        0.8592 |
|   3   |         0.9971 |        0.9578 |      0.871 |     0.9711 |      0.936  |        0.8642 |
|   5   |         0.9971 |        0.9563 |      0.871 |     0.9699 |      0.9354 |        0.8722 |
|   7   |         0.9971 |        0.9551 |      0.871 |     0.9692 |      0.9355 |        0.8722 |
|  10   |         0.9971 |        0.9537 |      0.871 |     0.9685 |      0.9345 |        0.8787 |

# Learned per-language bias — TEST evaluation

Top-20 val recall of true label: 0.9971. Selected reg=5.0 (None means no reg passed the guard; baseline b=0 evaluated).
Selection guard: val overall must improve and no stratum (tail/magnets/twins/head) may drop more than 0.01 vs baseline.
Compare to the gamma=0.5 frequency prior (Exp 14): overall macro-F1 +0.0058.

| stratum | base macroF1 | learned macroF1 | delta | 95% CI |
|---|---|---|---|---|
| overall | 0.9454 | 0.9567 | +0.0112 | [+0.0099, +0.0124] |
| magnets | 0.8832 | 0.8862 | +0.0051 | [-0.0302, +0.0463] |
| tail | 0.9310 | 0.8966 | -0.0320 | [-0.0588, +0.0000] |
| twins | 0.9224 | 0.9358 | +0.0135 | [+0.0101, +0.0170] |
| head | 0.9603 | 0.9696 | +0.0094 | [+0.0086, +0.0101] |

Overall accuracy: 0.9603 -> 0.9749 (+0.0147).