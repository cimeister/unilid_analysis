# Victim dig-ins (Exp 32): flagged outliers and the 131k pathology

## 1. llb_Latn under learned_bias; the gt_margin lowmid class (veto instrument)

| victim | config | true n | new FPs (top sources) | FPs removed | recall lost (top destinations) |
|---|---|---|---|---|---|
| llb_Latn | learned_bias | 4,181 | 1,356 (fij_Latn (56), ndo_Latn (54), kua_Latn (52), ven_Latn (50), luo_Latn (48)) | 98 | 3 (mck_Latn (1), tke_Latn (1), mny_Latn (1)) |
| llb_Latn | gt_margin | 4,181 | 2,695 (ndo_Latn (137), kua_Latn (100), bem_Latn (97), nya_Latn (91), zul_Latn (86)) | 170 | 9 (kzn_Latn (9)) |
| arq_Arab | gt_margin | 271 | 765 (ary_Arab (204), arb_Arab (191), fas_Arab (169), arz_Arab (70), aeb_Arab (67)) | 79 | 0 (none) |
| skr_Arab | gt_margin | 157 | 636 (pnb_Arab (488), urd_Arab (140), bal_Arab (4), fas_Arab (4)) | 63 | 0 (none) |
| vmk_Latn | gt_margin | 93 | 463 (vmw_Latn (346), ngl_Latn (97), mgh_Latn (11), twi_Latn (2), xho_Latn (2)) | 57 | 0 (none) |

## 2. sbs_Latn and mev_Latn under gt_min (balanced test draw)

| victim | true n | new FPs (top sources) | FPs removed | recall lost (top destinations) |
|---|---|---|---|---|
| sbs_Latn | 12 | 42 (kmb_Latn (2), phm_Latn (2), ach_Latn (1), alz_Latn (1), apr_Latn (1)) | 0 | 0 (none) |
| mev_Latn | 12 | 5 (bci_Latn (1), dyu_Latn (1), ewe_Latn (1), gaa_Latn (1), ogo_Latn (1)) | 0 | 0 (none) |

## 3. The 131k azj/tat pathology

| lang | model | entropy (nats) | floor | plateau size | plateau mass |
|---|---|---|---|---|---|
| tat_Latn | 100k | 3.283 | -14.39 | 94,840 | 5.3436e-02 |
| tat_Latn | 131k | 3.376 | -14.47 | 126,540 | 6.6017e-02 |
| azj_Latn | 100k | 2.922 | -19.13 | 81,470 | 4.0251e-04 |
| azj_Latn | 131k | 1.609 | -27.63 | 131,065 | 1.3106e-07 |

azj_Latn test lines (229,211) under 100k: recall 0.9904; top destinations azj_Latn (227,010), tly_Latn (794), tur_Latn (607), crh_Latn (477).

azj_Latn test lines (229,211) under 131k: recall 0.0000; top destinations tly_Latn (161,886), crh_Latn (29,433), tat_Latn (17,603), tur_Latn (14,498).
