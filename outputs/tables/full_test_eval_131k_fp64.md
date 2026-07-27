# Apertus 131k_fp64 baseline vs the 100k baseline

Model: `glotlid_apertus131k_fp64.unilid`. One scoring pass, b = 0, 45,377,279 lines; the 100k baseline predictions and y_true are reused read-only from the Exp 16 memmaps.

## Within-stratum macro-F1 (full test)

| stratum | 100k | 131k_fp64 | delta | 95% CI |
|---|---|---|---|---|
| overall (45,377,279) | 0.9292 | 0.9202 | -0.0090 | point only (n > cap) |
| tail (7,735) | 0.9132 | 0.8737 | -0.0395 | [-0.0476, -0.0322] |
| magnets (64,657) | 0.9138 | 0.8820 | -0.0318 | [-0.0386, -0.0251] |
| twins (9,156,023) | 0.9167 | 0.9122 | -0.0044 | point only (n > cap) |
| head (43,665,835) | 0.9602 | 0.9596 | -0.0006 | point only (n > cap) |

Overall accuracy: 0.9608 -> 0.9663 (+0.0055).

## Global per-language mean F1 by group (full pool)

| model | tail (N<1k) | lowmid (1k<=N<18k) | head (N>=18k) | flat_magnet | twin | all 1,940 |
|---|---|---|---|---|---|---|
| 100k | 0.5618 | 0.9381 | 0.9600 | 0.4716 | 0.8887 | 0.9292 |
| 131k_fp64 | 0.4269 | 0.9342 | 0.9593 | 0.3782 | 0.8844 | 0.9202 |

FPs into tail labels: 100k 22,522 -> 131k_fp64 32,211.

## Balanced-val draw 101, within-stratum macro-F1 (selection view)

| model | overall | tail | magnets | twins | head |
|---|---|---|---|---|---|
| 100k | 0.9811 | 0.9170 | 0.9174 | 0.9406 | 0.9814 |
| 131k_fp64 | 0.9776 | 0.8722 | 0.8803 | 0.9363 | 0.9802 |
