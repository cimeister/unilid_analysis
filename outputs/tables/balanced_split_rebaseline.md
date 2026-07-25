# Balanced-split re-baseline of the saved full-test configurations

Balanced val: draw seed 101, 188,061 lines; new test = pool minus val (45,189,218 lines). No new scoring: all numbers from the saved Exp 16 / floor-21 prediction memmaps.
Guard: val overall must improve and no stratum (tail/magnets/twins/head) may drop more than 0.01.

## Balanced-val stratified macro-F1 (selection view) and guard verdicts

| config | overall | tail | magnets | twins | head | guard |
|---|---|---|---|---|---|---|
| baseline | 0.9811 | 0.9170 | 0.9174 | 0.9406 | 0.9814 |  |
| freq_prior | 0.9798 | 0.8975 | 0.8977 | 0.9414 | 0.9814 | FAIL |
| learned_bias | 0.9799 | 0.9143 | 0.9089 | 0.9362 | 0.9797 | FAIL |
| floor21 | 0.9800 | 0.8942 | 0.8978 | 0.9406 | 0.9813 | FAIL |

## New-test stratified macro-F1 (final view)

| config | overall | tail | magnets | twins | head |
|---|---|---|---|---|---|
| baseline | 0.9210 | 0.9069 | 0.9077 | 0.9166 | 0.9599 |
| freq_prior | 0.9343 | 0.8908 | 0.8928 | 0.9178 | 0.9611 |
| learned_bias | 0.9342 | 0.9062 | 0.8999 | 0.9283 | 0.9701 |
| floor21 | 0.9372 | 0.8883 | 0.8936 | 0.9166 | 0.9596 |