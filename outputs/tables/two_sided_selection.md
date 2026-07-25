# Two-sided selection report (precision-primary adoption rule)

Instruments: selection = balanced-val draw 101 (188,061 lines); veto = pool minus the selection and headline draws (45,004,014 lines, from the saved full-pool memmaps; retains median 17 true lines per tail language, minimum 4); headline = balanced test draw seed 201 (185,204 lines), reported for baseline and the selected configuration only.
Rule constants: GUARD_TOL=0.01, TAIL_RECALL_TOL=0.03 (tail and magnets, symmetric widening), PREC_TOL=0.0, per-language collapse bound 0.1. Ranking among eligible configurations: highest balanced-val overall.

## Selection view: balanced-val within-stratum macro-F1

| config | overall | tail | magnets | twins | head |
|---|---|---|---|---|---|
| baseline | 0.9811 | 0.9170 | 0.9174 | 0.9406 | 0.9814 |
| freq_prior | 0.9798 | 0.8975 | 0.8977 | 0.9414 | 0.9814 |
| learned_bias | 0.9799 | 0.9143 | 0.9089 | 0.9362 | 0.9797 |
| floor21 | 0.9800 | 0.8942 | 0.8978 | 0.9406 | 0.9813 |
| margin_q5 | 0.9794 | 0.8889 | 0.8976 | 0.9407 | 0.9814 |
| margin_q5_head | 0.9799 | 0.8981 | 0.9036 | 0.9406 | 0.9814 |
| gt_min | 0.9841 | 0.9769 | 0.9688 | 0.9409 | 0.9811 |
| gt_margin | 0.9827 | 0.9507 | 0.9493 | 0.9410 | 0.9811 |
| gt_margin_all | 0.9675 | 0.9507 | 0.9418 | 0.9380 | 0.9817 |

## Veto view: global per-language stats, pool minus the selection and headline draws

Levels here are not comparable to the full-pool Exp 24 numbers: the excluded draws hold roughly half of each tail language's true lines while every false positive remains, so absolute precision/F1 sit lower. The rule uses gains and drops, which are unaffected.

| config | overall F1 | tail F1 | tail prec | tail rec | magnets F1 | magnets prec | magnets rec | FPs into tail |
|---|---|---|---|---|---|---|---|---|
| baseline | 0.9121 | 0.3382 | 0.2449 | 0.8664 | 0.2890 | 0.1927 | 0.8705 | 22,404 |
| freq_prior | 0.9264 | 0.4816 | 0.3873 | 0.8463 | 0.4168 | 0.3176 | 0.8489 | 12,302 |
| learned_bias | 0.9254 | 0.3736 | 0.2750 | 0.8666 | 0.3562 | 0.2508 | 0.8606 | 17,375 |
| floor21 | 0.9309 | 0.6337 | 0.6008 | 0.8316 | 0.5345 | 0.4876 | 0.8405 | 9,055 |
| margin_q5 | 0.9201 | 0.5125 | 0.4284 | 0.8133 | 0.4038 | 0.3116 | 0.8358 | 6,594 |
| margin_q5_head | 0.9215 | 0.5321 | 0.4445 | 0.8271 | 0.4185 | 0.3250 | 0.8428 | 6,560 |
| gt_min | 0.9114 | 0.2950 | 0.2205 | 0.9675 | 0.2533 | 0.1732 | 0.9565 | 79,113 |
| gt_margin | 0.9230 | 0.5373 | 0.4774 | 0.9124 | 0.4135 | 0.3400 | 0.9159 | 19,390 |
| gt_margin_all | 0.9331 | 0.5373 | 0.4774 | 0.9124 | 0.4942 | 0.4281 | 0.9002 | 19,390 |

## Verdicts

Collapse clause: more than 2 supported outliers rejects (class-level pattern); one or two flag a required dig-in without blocking (user decision 2026-07-24).

- **freq_prior: ELIGIBLE**
- **learned_bias: ELIGIBLE, flagged** (outlier collapse(s) requiring dig-in: llb_Latn)
- **floor21: ELIGIBLE**
- **margin_q5: ELIGIBLE, flagged** (outlier collapse(s) requiring dig-in: szy_Latn)
- **margin_q5_head: ELIGIBLE**
- **gt_min: REJECTED** (veto tail global mean F1 drops 0.0432; veto magnets global mean F1 drops 0.0358; 12 language(s) with support >= 10 lose more than 0.1 global F1 (worst 0.2123); more than 2 outliers is a class-level pattern)
- **gt_margin: REJECTED** (4 language(s) with support >= 10 lose more than 0.1 global F1 (worst 0.2060); more than 2 outliers is a class-level pattern)
- **gt_margin_all: REJECTED** (val overall drops more than 0.01; 4 language(s) with support >= 10 lose more than 0.1 global F1 (worst 0.3211); more than 2 outliers is a class-level pattern)

Selected configuration: **floor21** (highest balanced-val overall among eligible)

Largest per-language global-F1 drops under freq_prior (with veto true-line support): zsm_Arab -0.085 (n=13)

Largest per-language global-F1 drops under learned_bias (with veto true-line support): llb_Latn -0.113 (n=4181), shu_Arab -0.100 (n=554), led_Latn -0.079 (n=448), lea_Latn -0.058 (n=360), ngb_Latn -0.054 (n=380)

Largest per-language global-F1 drops under margin_q5 (with veto true-line support): szy_Latn -0.107 (n=175), lbe_Cyrl -0.100 (n=7), lea_Latn -0.087 (n=360), ldn_Latn -0.078 (n=5), mgm_Latn -0.078 (n=8)
Informational, exempt from clause (C) under margin_q5 (support < 10): lbe_Cyrl -0.100 (n=7)

Largest per-language global-F1 drops under gt_min (with veto true-line support): lbe_Cyrl -0.566 (n=7), ldn_Latn -0.444 (n=5), abq_Cyrl -0.398 (n=7), sel_Cyrl -0.353 (n=5), bor_Latn -0.296 (n=7)
Informational, exempt from clause (C) under gt_min (support < 10): abq_Cyrl -0.398 (n=7), bor_Latn -0.296 (n=7), drg_Latn -0.180 (n=9), eto_Latn -0.240 (n=7), lbe_Cyrl -0.566 (n=7), ldn_Latn -0.444 (n=5), mgm_Latn -0.266 (n=8), mrv_Latn -0.215 (n=7), otw_Latn -0.148 (n=4), sby_Latn -0.135 (n=8), sel_Cyrl -0.353 (n=5), tzl_Latn -0.274 (n=9), vot_Latn -0.162 (n=6), xum_Latn -0.237 (n=4), zpj_Latn -0.202 (n=5)

Largest per-language global-F1 drops under gt_margin (with veto true-line support): lbe_Cyrl -0.566 (n=7), ldn_Latn -0.444 (n=5), abq_Cyrl -0.398 (n=7), sel_Cyrl -0.353 (n=5), bor_Latn -0.296 (n=7)
Informational, exempt from clause (C) under gt_margin (support < 10): abq_Cyrl -0.398 (n=7), bor_Latn -0.296 (n=7), eto_Latn -0.240 (n=7), lbe_Cyrl -0.566 (n=7), ldn_Latn -0.444 (n=5), mgm_Latn -0.266 (n=8), mrv_Latn -0.215 (n=7), otw_Latn -0.148 (n=4), sby_Latn -0.135 (n=8), sel_Cyrl -0.353 (n=5), tzl_Latn -0.274 (n=9), vot_Latn -0.162 (n=6), xum_Latn -0.237 (n=4), zpj_Latn -0.202 (n=5)

Largest per-language global-F1 drops under gt_margin_all (with veto true-line support): lbe_Cyrl -0.566 (n=7), ldn_Latn -0.444 (n=5), abq_Cyrl -0.398 (n=7), sel_Cyrl -0.353 (n=5), llb_Latn -0.321 (n=4181)
Informational, exempt from clause (C) under gt_margin_all (support < 10): abq_Cyrl -0.398 (n=7), bor_Latn -0.296 (n=7), eto_Latn -0.240 (n=7), lbe_Cyrl -0.566 (n=7), ldn_Latn -0.444 (n=5), mgm_Latn -0.266 (n=8), mrv_Latn -0.215 (n=7), otw_Latn -0.148 (n=4), sby_Latn -0.135 (n=8), sel_Cyrl -0.353 (n=5), tzl_Latn -0.274 (n=9), vot_Latn -0.162 (n=6), xum_Latn -0.237 (n=4), zpj_Latn -0.202 (n=5)

## Uniform-prior track

Track passers on balanced val (`passes_uniform`): gt_min, gt_margin. Track-selected: **gt_min**.
Balanced-test collapse check: PASS with flagged outlier(s) requiring dig-in: mev_Latn -0.172 (n=12), sbs_Latn -0.182 (n=12). **gt_min is the uniform-prior track champion, flagged.**
Informational, below the support floor on the test draw: csw_Cans -0.167 (n=5), mgm_Latn -0.151 (n=8), zpj_Latn -0.200 (n=4)

## Headline: balanced test draw, within-stratum macro-F1

Rows for the baseline and each track-selected configuration only.

| config | overall | tail | magnets | twins | head |
|---|---|---|---|---|---|
| baseline | 0.9809 | 0.9086 | 0.9121 | 0.9435 | 0.9817 |
| floor21 | 0.9804 | 0.8924 | 0.8984 | 0.9433 | 0.9817 |
| gt_min | 0.9839 | 0.9809 | 0.9719 | 0.9426 | 0.9813 |
