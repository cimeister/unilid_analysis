# fp64 retrain check: corrupted vs corrected trainer (Exp 42)

Row-level comparison of each Apertus model against its retrain under the double-precision E-step. All numbers read from the packed weight matrices; no scoring. A row is degenerate when fewer than 100 entries lie above the row minimum.

## 131k (131,072 vocabulary entries, 1,940 languages)

- Degenerate rows: 18 before, 17 after.
  - Repaired by the fix: azj_Latn.
  - Still degenerate (expected: the vocabulary-coverage class, unrelated to the bug): aii_Syrc, arc_Syrc, chr_Cher, cop_Copt, crj_Cans, crk_Cans, crl_Cans, crm_Cans, csw_Cans, cwd_Cans, got_Goth, kyu_Kali, lif_Limb, lis_Lisu, mni_Mtei, sjo_Mong, syc_Syrc.
  - Newly degenerate: none.
- Rows changed by more than 1.0 nat at their largest entry: 20 of 1,940; by more than 5.0 nats: 5.
- Row entropy: mean 2.7466 -> 2.7472 nats.

Largest entropy movements:

| lang | entropy before | entropy after | max |delta| (nats) |
|---|---|---|---|
| azj_Latn | 1.609 | 3.025 | 22.50 |
| csw_Cans | 2.879 | 2.776 | 0.33 |
| pcm_Latn | 2.997 | 2.990 | 3.36 |
| bod_Tibt | 2.814 | 2.809 | 2.37 |
| quc_Latn | 2.763 | 2.759 | 9.43 |
| ded_Latn | 2.708 | 2.705 | 2.68 |
| got_Goth | 2.090 | 2.093 | 2.82 |
| hoc_Wara | 2.454 | 2.452 | 0.06 |

Diagnosed victims and controls:

| lang | above-min before | above-min after | entropy before | entropy after |
|---|---|---|---|---|
| azj_Latn | 7 | 22,704 | 1.609 | 3.025 |
| tat_Latn | 4,532 | 4,523 | 3.376 | 3.376 |
| quc_Latn | 13,382 | 13,373 | 2.763 | 2.759 |
| fra_Latn | 26,864 | 26,811 | 3.025 | 3.025 |
| csw_Cans | 96 | 96 | 2.879 | 2.776 |
| cwd_Cans | 89 | 89 | 2.859 | 2.859 |
| crj_Cans | 59 | 59 | 2.220 | 2.220 |
| crk_Cans | 81 | 81 | 2.228 | 2.228 |
| crl_Cans | 70 | 70 | 2.207 | 2.207 |
| crm_Cans | 68 | 68 | 2.221 | 2.221 |

## 200k (200,000 vocabulary entries, 1,940 languages)

- Degenerate rows: 17 before, 17 after.
  - Repaired by the fix: none.
  - Still degenerate (expected: the vocabulary-coverage class, unrelated to the bug): aii_Syrc, arc_Syrc, chr_Cher, cop_Copt, crj_Cans, crk_Cans, crl_Cans, crm_Cans, csw_Cans, cwd_Cans, got_Goth, kyu_Kali, lif_Limb, lis_Lisu, mni_Mtei, sjo_Mong, syc_Syrc.
  - Newly degenerate: none.
- Rows changed by more than 1.0 nat at their largest entry: 18 of 1,940; by more than 5.0 nats: 7.
- Row entropy: mean 2.7828 -> 2.7831 nats.

Largest entropy movements:

| lang | entropy before | entropy after | max |delta| (nats) |
|---|---|---|---|
| azj_Latn | 2.473 | 3.000 | 21.57 |
| csw_Cans | 2.874 | 2.975 | 0.20 |
| quc_Latn | 2.774 | 2.768 | 8.70 |
| pcm_Latn | 3.002 | 2.999 | 7.32 |
| fas_Arab | 2.884 | 2.881 | 5.53 |
| hoc_Wara | 2.608 | 2.605 | 0.05 |
| bod_Tibt | 2.738 | 2.737 | 1.81 |
| mam_Latn | 2.717 | 2.716 | 8.73 |

Diagnosed victims and controls:

| lang | above-min before | above-min after | entropy before | entropy after |
|---|---|---|---|---|
| azj_Latn | 1,798 | 31,210 | 2.473 | 3.000 |
| tat_Latn | 5,053 | 5,039 | 3.540 | 3.540 |
| quc_Latn | 17,333 | 17,275 | 2.774 | 2.768 |
| fra_Latn | 43,575 | 43,411 | 3.070 | 3.070 |
| csw_Cans | 95 | 95 | 2.874 | 2.975 |
| cwd_Cans | 89 | 89 | 3.070 | 3.071 |
| crj_Cans | 59 | 59 | 2.235 | 2.235 |
| crk_Cans | 81 | 81 | 2.247 | 2.247 |
| crl_Cans | 70 | 70 | 2.221 | 2.221 |
| crm_Cans | 68 | 68 | 2.229 | 2.229 |

