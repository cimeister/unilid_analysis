# Apertus 131k (preliminary_mul) baseline vs the 100k baseline

One scoring pass, b = 0, 45,377,279 lines; 100k baseline and y_true reused read-only from the Exp 16 memmaps. Branch-decision numbers for plan Track A.

## Within-stratum macro-F1 (full test)

| stratum | 100k | 131k | delta | 95% CI |
|---|---|---|---|---|
| overall (45,377,279) | 0.9292 | 0.9179 | -0.0113 | point only (n > cap) |
| tail (7,735) | 0.9132 | 0.8695 | -0.0437 | [-0.0515, -0.0371] |
| magnets (64,657) | 0.9138 | 0.8786 | -0.0352 | [-0.0417, -0.0289] |
| twins (9,156,023) | 0.9167 | 0.9122 | -0.0044 | point only (n > cap) |
| head (43,665,835) | 0.9602 | 0.9583 | -0.0019 | point only (n > cap) |

Overall accuracy: 0.9608 -> 0.9612 (+0.0004).

## Global per-language mean F1 by group (full pool)

| model | tail (N<1k) | lowmid (1k<=N<18k) | head (N>=18k) | flat_magnet | twin | all 1,940 |
|---|---|---|---|---|---|---|
| 100k | 0.5618 | 0.9381 | 0.9600 | 0.4716 | 0.8887 | 0.9292 |
| 131k | 0.4046 | 0.9331 | 0.9579 | 0.3551 | 0.8844 | 0.9179 |

FPs into tail labels: 100k 22,522 -> 131k 51,926.

## Balanced-val draw 101, within-stratum macro-F1 (selection view)

| model | overall | tail | magnets | twins | head |
|---|---|---|---|---|---|
| 100k | 0.9811 | 0.9170 | 0.9174 | 0.9406 | 0.9814 |
| 131k | 0.9766 | 0.8679 | 0.8769 | 0.9363 | 0.9790 |
