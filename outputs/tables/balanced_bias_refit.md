|   reg |   sel_overall |   sel_tail |   sel_magnets |   sel_twins |   sel_head |
|------:|--------------:|-----------:|--------------:|------------:|-----------:|
|   0   |        0.9818 |     0.9205 |        0.9217 |      0.9395 |     0.9819 |
|   0.3 |        0.9834 |     0.9504 |        0.9469 |      0.9379 |     0.982  |
|   1   |        0.9829 |     0.9366 |        0.9358 |      0.939  |     0.9824 |
|   3   |        0.9825 |     0.929  |        0.9295 |      0.9388 |     0.9823 |
|   5   |        0.9824 |     0.927  |        0.9278 |      0.9394 |     0.9821 |
|   7   |        0.9822 |     0.9245 |        0.9259 |      0.9399 |     0.9821 |
|  10   |        0.9821 |     0.9241 |        0.9253 |      0.94   |     0.982  |

# Learned bias refit on balanced data — selection-half results

Fit on the language-balanced fit half (top-20 recall 0.9988); plain L2 (uniform-prior objective); corrected NLL gradient.
Guard: selection-half overall must improve and no stratum (tail/magnets/twins/head) may drop more than 0.01.

Selected reg=0.3. ||b||_inf=11.302; nonzero mass concentrates on:
| lang | b | category |
|---|---|---|
| nya_Latn | -11.302 | head |
| por_Latn | -8.889 | twin |
| npl_Latn | -8.295 | head |
| crn_Latn | -7.977 | head |
| heb_Hebr | -7.205 | head |
| las_Latn | -6.674 | mid |
| syc_Syrc | -6.575 | mid |
| dip_Latn | -6.182 | mid |
| swc_Latn | -5.854 | twin |
| ttq_Latn | -5.575 | mid |

A guard-passing bias still needs a full-test pass (exact biased scorer) before adoption.