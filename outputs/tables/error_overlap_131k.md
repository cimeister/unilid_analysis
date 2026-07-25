# Error overlap: 131k baseline vs 100k baseline (Exp 30)

45,377,279 kept lines; accuracies 100k 0.9608, 131k 0.9612 (gates passed).

## Error-set overlap

- 100k errors: 1,779,499; 131k errors: 1,759,424.
- Shared (both wrong): 1,026,036 = 57.7% of 100k errors; of the shared errors, 679,002 (66.2%) pick the SAME wrong label.
- Fixed by 131k (100k wrong, 131k right): 753,463 (42.3% of 100k errors).
- Introduced by 131k (100k right, 131k wrong): 733,388.
- Net: -20,075 errors.

## Where the fixed and introduced errors live (true language)

| set | tail | lowmid | head | Indic scripts | Han/CJK |
|---|---|---|---|---|---|
| fixed | 95 | 3,314 | 750,054 | 27,241 | 19,224 |
| introduced | 339 | 4,789 | 728,260 | 23,778 | 20,245 |
| shared | 556 | 7,525 | 1,017,955 | 50,612 | 27,984 |

Indic-script lines total 2,022,732; 131k fixes 35.0% of the 100k errors there and introduces new errors on 1.22% of previously-correct lines. Han/CJK lines total 1,031,284.

## Largest per-language global-F1 changes under 131k

Improvements: syl_Latn +0.381, nhk_Latn +0.291, pwn_Latn +0.258, tig_Ethi +0.248, lad_Hebr +0.241, dip_Latn +0.235, qus_Latn +0.217, cnk_Latn +0.207, tcy_Knda +0.198, hlt_Latn +0.185, ina_Latn +0.165, fro_Latn +0.150
Regressions: azj_Latn -0.993, csw_Cans -0.643, ldn_Latn -0.595, nla_Latn -0.537, hoc_Latn -0.462, mgm_Latn -0.430, aqz_Latn -0.424, otw_Latn -0.415, diu_Latn -0.413, mrv_Latn -0.412, drg_Latn -0.395, kei_Latn -0.392
Languages improved by more than 0.01: 190; regressed by more than 0.01: 403.

## FPs into tail labels under 131k (51,926 vs 22,522 under 100k)

Top (pred <- true) pairs:
- tat_Latn <- azj_Latn: 17,603
- pnt_Grek <- ell_Grek: 2,810
- rme_Latn <- eng_Latn: 624
- mrq_Latn <- tah_Latn: 534
- sbs_Latn <- loz_Latn: 434
- sbs_Latn <- bem_Latn: 419
- ang_Latn <- eng_Latn: 413
- mev_Latn <- gaa_Latn: 391
- mrq_Latn <- rar_Latn: 334
- sbs_Latn <- ekk_Latn: 326
- sbs_Latn <- fin_Latn: 311
- sbs_Latn <- xho_Latn: 278
