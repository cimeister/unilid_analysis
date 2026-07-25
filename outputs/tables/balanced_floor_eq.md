# Floor equalization under the balanced protocol — balanced-val selection (no test scoring; a passing config needs its own full-test pass)

Best config (guarded): **baseline** (baseline means nothing passed).
Guard: val overall must improve and no stratum (tail/magnets/twins/head) may drop more than 0.01.

| config   |   val_overall |   val_tail |   val_magnets |   val_twins |   val_head |
|:---------|--------------:|-----------:|--------------:|------------:|-----------:|
| baseline |        0.9811 |     0.917  |        0.9174 |      0.9406 |     0.9814 |
| floor-17 |        0.9803 |     0.8993 |        0.9022 |      0.9408 |     0.9814 |
| floor-19 |        0.9802 |     0.8959 |        0.899  |      0.941  |     0.9814 |
| floor-21 |        0.98   |     0.8942 |        0.8978 |      0.9406 |     0.9813 |
| floor-23 |        0.9796 |     0.8901 |        0.8942 |      0.9405 |     0.9812 |