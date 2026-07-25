# Metric decomposition of the full-test predictions (Exp 24)

Inputs: prediction memmaps under /capstor/scratch/cscs/cmeister747/unilid_analysis/full_test_eval (Exp 16 job 2784115; floor-21 job 2791722), 45,377,279 kept lines. Analysis only, no new scoring. All within-stratum values below reproduce the recorded tables (gate tolerance 6e-5); global per-language F1 reproduces full_test_per_lang_f1.csv exactly.

Definitions. Within-stratum macro-F1 restricts both truth and predictions to examples whose true label is in the stratum, so false positives arriving from outside the stratum are excluded; this is the view every stratum row and the selection guard use. Global per-language F1 counts the full confusion row and column for each language; the overall rows already use it.

## Tail (96 languages), two views

| config | within-stratum F1 | global mean F1 | global mean precision | global mean recall | FPs into tail labels |
|---|---|---|---|---|---|
| baseline | 0.9132 | 0.5618 | 0.4590 | 0.8741 | 22,522 |
| freq_prior | 0.8950 | 0.6800 | 0.6164 | 0.8505 | 12,381 |
| learned_bias | 0.9114 | 0.6003 | 0.5024 | 0.8714 | 17,496 |
| floor21 | 0.8928 | 0.7655 | 0.7634 | 0.8416 | 9,103 |

Baseline counterfactual: with precision fixed at 1.0 the tail global mean F1 would be 0.9154, against the within-stratum 0.9132; the two views differ almost entirely by the false positives. Baseline tail languages with precision below 0.5: 57/96; with recall below 0.5: 7/96.

## Global per-language mean F1 by group

| config | tail (N<1k) | lowmid (1k<=N<18k) | head (N>=18k) | flat_magnet | twin | isolated_tail | tight_lowres | all 1,940 |
|---|---|---|---|---|---|---|---|---|
| baseline | 0.5618 | 0.9381 | 0.9600 | 0.4716 | 0.8887 | 0.7588 | 0.3322 | 0.9292 |
| freq_prior | 0.6800 | 0.9485 | 0.9611 | 0.5782 | 0.8919 | 0.8441 | 0.4309 | 0.9408 |
| learned_bias | 0.6003 | 0.9510 | 0.9700 | 0.5431 | 0.9103 | 0.8030 | 0.4866 | 0.9421 |
| floor21 | 0.7655 | 0.9440 | 0.9596 | 0.6402 | 0.8886 | 0.8470 | 0.3539 | 0.9421 |

Group sizes: tail (N<1k) = 96, lowmid (1k<=N<18k) = 984, head (N>=18k) = 860, flat_magnet = 118, twin = 77, isolated_tail = 14, tight_lowres = 5, all 1,940 = 1940.

## Overall delta decomposition by category (vs baseline)

Overall global macro-F1 delta as the sum of per-language deltas divided by 1,940, split by diagnostic category.

| config | total | flat_magnet | twin | isolated_tail | tight_lowres | mid | head |
|---|---|---|---|---|---|---|---|
| freq_prior | +0.0116 | +0.0065 | +0.0001 | +0.0006 | +0.0003 | +0.0037 | +0.0004 |
| learned_bias | +0.0129 | +0.0043 | +0.0009 | +0.0003 | +0.0004 | +0.0039 | +0.0031 |
| floor21 | +0.0129 | +0.0103 | -0.0000 | +0.0006 | +0.0001 | +0.0021 | -0.0001 |

## False-positive flow into tail labels (baseline)

- Total FPs into tail labels: 22,522 against 7,735 true tail examples.
- Source-language resources: median N = 100,000; 98.9% of FPs come from head sources (N>=18k), 0.1% from other tail languages.
- Concentration across receiving tail languages: top 5 hold 52.5%, top 10 64.9%, top 20 79.5% of all FPs into tail.
- Tail recall misses: 651 of 7,735 (8.4%); 52.2% of the misrouted lines go to head languages, 4.5% to other tail languages.

## The 15 tail languages receiving the most false positives, baseline vs floor-21

| lang | N | prec base -> floor21 | rec base -> floor21 | F1 base -> floor21 | FP base -> floor21 | top baseline FP sources |
|---|---|---|---|---|---|---|
| sbs_Latn | 271 | 0.007 -> 0.009 | 0.830 -> 0.532 | 0.014 -> 0.018 | 5,476 -> 2,638 | loz_Latn (452), bem_Latn (336), kng_Latn (294) |
| pnt_Grek | 846 | 0.038 -> 0.051 | 0.947 -> 0.947 | 0.073 -> 0.096 | 3,592 -> 2,660 | ell_Grek (3426), bul_Cyrl (17), grc_Grek (11) |
| mrq_Latn | 827 | 0.115 -> 0.209 | 1.000 -> 0.973 | 0.207 -> 0.343 | 1,119 -> 539 | tah_Latn (497), rar_Latn (346), niu_Latn (89) |
| ang_Latn | 490 | 0.087 -> 0.305 | 0.897 -> 0.897 | 0.159 -> 0.455 | 816 -> 178 | eng_Latn (248), dan_Latn (44), lvs_Latn (37) |
| pwn_Latn | 561 | 0.098 -> 0.220 | 0.889 -> 0.889 | 0.176 -> 0.353 | 811 -> 312 | ind_Latn (67), hun_Latn (43), zsm_Latn (43) |
| arb_Latn | 847 | 0.166 -> 0.351 | 1.000 -> 1.000 | 0.285 -> 0.519 | 747 -> 276 | uzn_Latn (53), dan_Latn (50), mlt_Latn (50) |
| tat_Latn | 885 | 0.175 -> 0.325 | 0.942 -> 0.942 | 0.295 -> 0.483 | 694 -> 306 | tur_Latn (359), nst_Latn (54), uzn_Latn (44) |
| rme_Latn | 366 | 0.106 -> 0.240 | 0.969 -> 0.969 | 0.191 -> 0.384 | 531 -> 200 | eng_Latn (435), slk_Latn (10), uzn_Latn (7) |
| lud_Latn | 554 | 0.186 -> 0.560 | 0.979 -> 0.969 | 0.312 -> 0.710 | 412 -> 73 | ekk_Latn (71), fin_Latn (46), hun_Latn (37) |
| goh_Latn | 306 | 0.079 -> 0.554 | 0.648 -> 0.574 | 0.140 -> 0.564 | 410 -> 25 | hun_Latn (35), kin_Latn (34), uzn_Latn (28) |
| asm_Latn | 455 | 0.154 -> 0.512 | 0.877 -> 0.778 | 0.262 -> 0.618 | 390 -> 60 | uzn_Latn (55), hun_Latn (28), lit_Latn (20) |
| min_Arab | 847 | 0.258 -> 0.325 | 0.900 -> 0.893 | 0.401 -> 0.477 | 388 -> 278 | fas_Arab (264), arb_Arab (84), arz_Arab (10) |
| mns_Cyrl | 971 | 0.312 -> 0.479 | 1.000 -> 1.000 | 0.476 -> 0.648 | 379 -> 187 | rus_Cyrl (245), bul_Cyrl (21), azj_Latn (11) |
| kas_Latn | 405 | 0.162 -> 0.467 | 0.958 -> 0.889 | 0.277 -> 0.612 | 357 -> 73 | uzn_Latn (74), eng_Latn (33), ekk_Latn (16) |
| prg_Latn | 891 | 0.324 -> 0.784 | 0.975 -> 0.968 | 0.486 -> 0.866 | 319 -> 42 | lvs_Latn (82), lat_Latn (65), lit_Latn (31) |

## Residual FPs into tail labels under floor-21

9,103 residual FPs; 100.0% are same-script. Top directed pairs (pred <- true):

- pnt_Grek <- ell_Grek: 2,644
- sbs_Latn <- loz_Latn: 354
- mrq_Latn <- tah_Latn: 316
- tat_Latn <- tur_Latn: 209
- sbs_Latn <- bem_Latn: 207
- min_Arab <- fas_Arab: 199
- rme_Latn <- eng_Latn: 193
- sbs_Latn <- kng_Latn: 169
- mrq_Latn <- rar_Latn: 164
- sbs_Latn <- toi_Latn: 162
- mns_Cyrl <- rus_Cyrl: 158
- sbs_Latn <- kqn_Latn: 150
