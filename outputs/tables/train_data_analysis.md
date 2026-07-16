### Overall domain distribution

|    Domain |    Samples | % of total |
|-----------|------------|------------|
| religious |  1,149,136 |        1.9 |
| wikipedia |      1,032 |        0.0 |
|     other | 59,532,983 |       98.1 |


### Domain distribution by resource level

| Resource bin | Religious % | Wikipedia % | Other % |  N samples |
|--------------|-------------|-------------|---------|------------|
|         <500 |         2.5 |         0.0 |    97.5 |     14,307 |
|      500--1k |         1.9 |         0.0 |    98.1 |     30,169 |
|      1k--12k |         0.5 |         0.0 |    99.5 |  3,334,989 |
|     12k--18k |         0.3 |         0.0 |    99.7 |  8,602,771 |
|     18k--35k |         0.6 |         0.0 |    99.4 | 10,272,541 |
|         35k+ |         2.7 |         0.0 |    97.3 | 38,428,374 |


### Domain distribution for confusion cluster languages

| Language |       N | Religious % | Wikipedia % | Other % |
|----------|---------|-------------|-------------|---------|
| arb_Arab | 100,000 |         0.0 |         0.0 |   100.0 |
| arz_Arab | 100,000 |         0.0 |         0.0 |    99.9 |
| ary_Arab | 100,000 |         0.0 |         0.0 |   100.0 |
| ars_Arab |  18,539 |         0.1 |         0.0 |    99.9 |
| apc_Arab |  79,058 |         0.0 |         0.0 |   100.0 |
| acm_Arab |   4,566 |         0.2 |         0.0 |    99.8 |
| cmn_Hani | 100,000 |         0.1 |         0.0 |    99.9 |
| wuu_Hani |  74,364 |         0.1 |         0.0 |    99.9 |
| yue_Hani | 100,000 |         0.0 |         0.0 |   100.0 |
| lzh_Hani |  97,818 |         0.0 |         0.0 |   100.0 |
| hak_Hani |  29,565 |         0.0 |         0.0 |   100.0 |
| hin_Deva | 100,000 |         0.2 |         0.0 |    99.8 |
| anp_Deva |   4,499 |         0.2 |         0.4 |    99.4 |
| bho_Deva |  55,844 |         0.1 |         0.1 |    99.8 |
| mai_Deva |  43,579 |         0.1 |         0.0 |    99.9 |
| mag_Deva |  12,847 |         0.0 |         0.0 |   100.0 |
| hne_Deva |  50,507 |         0.0 |         0.0 |   100.0 |
| kas_Deva |   7,226 |         0.2 |         0.0 |    99.8 |
| doi_Deva |   3,674 |         0.0 |         0.0 |   100.0 |
| ind_Latn | 100,000 |         2.2 |         0.0 |    97.8 |
| zsm_Latn | 100,000 |         0.2 |         0.0 |    99.8 |
| bjn_Latn |  27,655 |         0.2 |         0.0 |    99.8 |
| dan_Latn | 100,000 |         0.3 |         0.0 |    99.7 |
| nob_Latn | 100,000 |         6.2 |         0.0 |    93.8 |
| nno_Latn | 100,000 |         0.3 |         0.0 |    99.7 |
| swe_Latn | 100,000 |         0.4 |         0.0 |    99.6 |
| heb_Hebr | 100,000 |         0.2 |         0.0 |    99.8 |
| hbo_Hebr | 100,000 |         0.0 |         0.0 |   100.0 |
| fas_Arab | 100,000 |         0.2 |         0.0 |    99.8 |
| glk_Arab |  22,263 |         0.5 |         0.0 |    99.5 |
| mzn_Arab |  25,413 |         0.0 |         0.0 |   100.0 |
| sdh_Arab | 100,000 |         0.3 |         0.0 |    99.7 |


### Corpus quality by resource level

| Resource bin | # Langs | Mean text len | Mean char entropy | Mean vocab size |
|--------------|---------|---------------|-------------------|-----------------|
|         <500 |      56 |          83.8 |              4.47 |           1,447 |
|      500--1k |      40 |          94.7 |              4.51 |           3,670 |
|      1k--12k |     458 |         148.0 |              4.39 |          15,718 |
|     12k--18k |     526 |         166.7 |              4.34 |          19,597 |
|     18k--35k |     398 |         164.1 |              4.28 |          29,619 |
|         35k+ |     462 |         104.6 |              4.55 |         102,776 |


### Script verification: 20 languages with >5% unexpected script (showing top 50)

| Language |       N |     Expected script | Expected % | Dominant script | Dominant % |
|----------|---------|---------------------|------------|-----------------|------------|
| crk_Cans |   4,454 | Canadian_Aboriginal |        0.0 |           Latin |      100.0 |
| crm_Cans |  11,058 | Canadian_Aboriginal |        0.0 |           Latin |      100.0 |
| csw_Cans |     124 | Canadian_Aboriginal |        0.0 |           Latin |      100.0 |
| cwd_Cans |     231 | Canadian_Aboriginal |        0.0 |           Latin |      100.0 |
| ike_Cans |  30,725 | Canadian_Aboriginal |        0.0 |           Latin |      100.0 |
| ojb_Cans |  33,721 | Canadian_Aboriginal |        0.0 |           Latin |      100.0 |
| jpn_Jpan | 100,000 |                 Han |       35.4 |        Hiragana |       52.3 |
| shn_Mymr |  32,679 |             Myanmar |       91.6 |         Myanmar |       91.6 |
| kca_Cyrl |     951 |            Cyrillic |       91.7 |        Cyrillic |       91.7 |
| cdo_Latn |   9,577 |               Latin |       91.9 |           Latin |       91.9 |
| urk_Thai |  13,466 |                Thai |       92.0 |            Thai |       92.0 |
| kpv_Cyrl |  37,519 |            Cyrillic |       92.4 |        Cyrillic |       92.4 |
| wuu_Hani |  74,364 |                 Han |       92.9 |             Han |       92.9 |
| chv_Cyrl |  47,817 |            Cyrillic |       93.8 |        Cyrillic |       93.8 |
| abi_Latn |   6,735 |               Latin |       93.8 |           Latin |       93.8 |
| koi_Cyrl |  13,537 |            Cyrillic |       94.0 |        Cyrillic |       94.0 |
| cmn_Hani | 100,000 |                 Han |       94.6 |             Han |       94.6 |
| kxm_Thai |  13,492 |                Thai |       94.6 |            Thai |       94.6 |
| itl_Cyrl |     978 |            Cyrillic |       94.6 |        Cyrillic |       94.6 |
| mns_Cyrl |     971 |            Cyrillic |       94.9 |        Cyrillic |       94.9 |
