# Per-language diagnostic

Languages: 1940. Magnet: zH>1.5 & magnet_ratio>2.0 (or zH>5.0). Twin/tight: dnn_lo=0.239, dup_lo=0.310, tau(kclose radius)=0.997.

## Category counts

- mid: 925
- head: 801
- flat_magnet: 118
- twin: 77
- isolated_tail: 14
- tight_lowres: 5

## Top 25 flat magnets by val false-positives

| lang     |     N |    zH |   k_close |   support_val |   fp_val |   magnet_ratio |
|:---------|------:|------:|----------:|--------------:|---------:|---------------:|
| sco_Latn | 87458 | 2.472 |       276 |            87 |      398 |          4.523 |
| tly_Latn |  3892 | 2.333 |       221 |             4 |      182 |         36.4   |
| fit_Latn |  3881 | 3.399 |       329 |             3 |      174 |         43.5   |
| kzn_Latn |  2125 | 5.037 |       472 |             1 |      134 |         67     |
| zea_Latn |  8505 | 2.756 |       314 |             7 |      122 |         15.25  |
| frp_Latn |  5176 | 2.931 |       294 |             6 |       75 |         10.714 |
| bjn_Latn | 27655 | 3.228 |       398 |            20 |       74 |          3.524 |
| arg_Latn | 25914 | 2.552 |       295 |            27 |       67 |          2.393 |
| lad_Latn | 10070 | 3.282 |       447 |             7 |       56 |          7     |
| vls_Latn | 25500 | 1.567 |       264 |            23 |       56 |          2.333 |
| vep_Latn |  9947 | 2.418 |       297 |            10 |       48 |          4.364 |
| acm_Arab |  4566 | 2.721 |        84 |             7 |       46 |          5.75  |
| nov_Latn |  1813 | 4.526 |       572 |             2 |       45 |         15     |
| ltg_Latn | 14559 | 1.739 |       234 |             8 |       43 |          4.778 |
| abs_Latn |  1335 | 5.203 |       639 |             2 |       43 |         14.333 |
| qus_Latn |  5732 | 2.077 |       190 |             7 |       42 |          5.25  |
| fro_Latn | 15794 | 1.678 |       275 |            13 |       40 |          2.857 |
| pcd_Latn |  2213 | 5.199 |       434 |             2 |       37 |         12.333 |
| nrf_Latn |  6460 | 2.329 |       255 |             5 |       36 |          6     |
| ile_Latn | 14414 | 2.722 |       317 |            13 |       33 |          2.357 |
| fkv_Latn |  1884 | 5.094 |       422 |             3 |       33 |          8.25  |
| ksh_Latn | 11430 | 2.1   |       295 |            10 |       31 |          2.818 |
| crh_Latn | 17238 | 2.145 |       234 |            12 |       29 |          2.231 |
| tig_Ethi |  3064 | 2.072 |        59 |             5 |       26 |          4.333 |
| ext_Latn |  8692 | 3.23  |       350 |             7 |       24 |          3     |

## Sample tight_lowres (shrink toward target)

| lang     |    N |   d_up | target_lang   |   up_resratio |
|:---------|-----:|-------:|:--------------|--------------:|
| arq_Arab | 2712 | 0.2507 | ary_Arab      |        0.0271 |
| gom_Deva | 3496 | 0.2153 | mar_Deva      |        0.035  |
| syl_Beng | 4163 | 0.2999 | ben_Beng      |        0.0416 |
| inh_Cyrl | 2521 | 0.2953 | che_Cyrl      |        0.0448 |
| anp_Deva | 4499 | 0.0848 | hin_Deva      |        0.045  |

## Sample twins (do NOT pool)

| lang     |      N |   d_nn | nn_lang   |
|:---------|-------:|-------:|:----------|
| wuu_Hani |  74364 | 0.0286 | cmn_Hani  |
| cmn_Hani | 100000 | 0.0286 | wuu_Hani  |
| zgh_Tfng |   2897 | 0.0505 | tzm_Tfng  |
| tzm_Tfng |   1590 | 0.0505 | zgh_Tfng  |
| yue_Hani | 100000 | 0.062  | cmn_Hani  |
| ind_Latn | 100000 | 0.0644 | zsm_Latn  |
| zsm_Latn | 100000 | 0.0644 | ind_Latn  |
| zom_Latn |  48432 | 0.0758 | ctd_Latn  |
| ctd_Latn |  66425 | 0.0758 | zom_Latn  |
| run_Latn | 100000 | 0.0826 | kin_Latn  |
| kin_Latn | 100000 | 0.0826 | run_Latn  |
| lzh_Hani |  97818 | 0.0856 | cmn_Hani  |
| ace_Arab |   6107 | 0.1031 | bjn_Arab  |
| bjn_Arab |   6110 | 0.1031 | ace_Arab  |
| hyw_Armn | 100000 | 0.1085 | hye_Armn  |
| hye_Armn | 100000 | 0.1085 | hyw_Armn  |
| bps_Latn |  17697 | 0.1149 | bpr_Latn  |
| bpr_Latn |  17724 | 0.1149 | bps_Latn  |
| nde_Latn | 100000 | 0.1198 | nbl_Latn  |
| nbl_Latn | 100000 | 0.1198 | nde_Latn  |