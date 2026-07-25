| config         |   val_overall |   val_tail |   val_head |   val_twins |   val_magnets |
|:---------------|--------------:|-----------:|-----------:|------------:|--------------:|
| baseline       |        0.9451 |      0.871 |     0.9601 |      0.9246 |        0.8797 |
| macro_marginal |        0.9452 |      0.871 |     0.96   |      0.9247 |        0.8797 |

# Macrolanguage-hierarchical decision — TEST evaluation

score(G) = logsumexp over members (top-50 candidates); prediction = best member of the argmax group. Parameter-free (no tuning), so the table reports the hierarchical-vs-baseline deltas unconditionally; the guard verdict on val (PASS) is the accept/reject criterion for adopting the rule. Top-1 agreement 0.9951.
Selection guard: val overall must improve and no stratum (tail/magnets/twins/head) may drop more than 0.01 vs baseline.

| stratum | base macroF1 | hierarchical macroF1 | delta | 95% CI |
|---|---|---|---|---|
| overall | 0.9454 | 0.9454 | -0.0000 | [-0.0001, +0.0000] |
| magnets | 0.8832 | 0.8832 | +0.0000 | [+0.0000, +0.0000] |
| tail | 0.9310 | 0.9310 | +0.0000 | [+0.0000, +0.0000] |
| twins | 0.9224 | 0.9224 | -0.0000 | [-0.0001, +0.0000] |
| head | 0.9603 | 0.9603 | -0.0000 | [-0.0001, +0.0000] |

Overall exact accuracy: 0.9603 -> 0.9603 (+0.0000).
Macro-aware accuracy (test half): baseline 0.9680 -> hierarchical 0.9680.