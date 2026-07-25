# Punctuation partial pooling (script-level hierarchical prior) — balanced-val selection (no test scoring; a passing config needs its own full-test pass)

Best config (guarded): **punct_a300** (baseline means nothing passed).
Guard: val overall must improve and no stratum (tail/magnets/twins/head) may drop more than 0.01.

| config       |   val_overall |   val_tail |   val_magnets |   val_twins |   val_head |
|:-------------|--------------:|-----------:|--------------:|------------:|-----------:|
| baseline     |        0.9811 |     0.917  |        0.9174 |      0.9406 |     0.9814 |
| punct_a300   |        0.9812 |     0.9174 |        0.9177 |      0.9407 |     0.9814 |
| punct_a3000  |        0.981  |     0.9168 |        0.917  |      0.9398 |     0.9813 |
| punct_a30000 |        0.9809 |     0.9171 |        0.9168 |      0.9375 |     0.9812 |