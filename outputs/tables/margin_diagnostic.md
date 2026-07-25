# Margin diagnostic (per-language decision threshold, plan B3)

Margin = top1 minus top2 of 5-best scores. tau_L = 5th percentile of L's own train-line margins (lines L wins); calibration sample <= 2000 lines, seed 0; languages with < 200 scoreable self-won train lines are excluded from gating (tau = -inf).
Excluded languages (26): abq_Cyrl, arc_Syrc, bor_Latn, csw_Cans, diu_Latn, drg_Latn, eto_Latn, kak_Latn, kdr_Latn, lad_Hebr, lbe_Cyrl, ldn_Latn, lki_Arab, lrc_Arab, mgm_Latn, mrv_Latn, nio_Cyrl, otw_Latn, sby_Latn, sel_Cyrl, sju_Latn, tzl_Latn, vot_Latn, xum_Latn, zpj_Latn, zsm_Arab.

## Aggregate (all 96 tail languages)

- FP-into-tail lines scored: 22,522; caught at tau: 17,299 (76.8%); caught lines whose runner-up is the true label: 5,413 (31.3% of caught).
- Genuine true-tail test lines won by their own language: 7,084; suppressed at tau: 474 (6.7%); of the suppressed, 53 have another tail language as runner-up (cascade check).

## Per-language table (sorted by FP count; NaN = no FPs or no wins)

| lang     |   n_fp |       tau | excluded   |   fp_catch |   recover |      auc |   n_genuine_test_won |   test_suppressed |   supp_to_tail |
|:---------|-------:|----------:|:-----------|-----------:|----------:|---------:|---------------------:|------------------:|---------------:|
| sbs_Latn |   5476 |    5.1405 | False      |     0.5884 |    0.0919 |   0.9022 |                   39 |            0.2564 |              0 |
| pnt_Grek |   3592 |    4.9454 | False      |     0.738  |    0.7405 |   0.9409 |                  142 |            0.0282 |              0 |
| mrq_Latn |   1119 |   16.7334 | False      |     0.8811 |    0.1075 |   0.9763 |                  146 |            0.0753 |              0 |
| ang_Latn |    816 |    4.5326 | False      |     0.8235 |    0.3571 |   0.9669 |                   78 |            0.0128 |              0 |
| pwn_Latn |    811 |   90.1036 | False      |     0.9975 |    0.1273 |   0.999  |                   88 |            0.1023 |              0 |
| arb_Latn |    747 |   52.9786 | False      |     1      |    0.2182 |   0.9998 |                  149 |            0.0805 |              0 |
| tat_Latn |    694 |    8.6833 | False      |     0.9438 |    0.5496 |   0.9831 |                  147 |            0.068  |              0 |
| rme_Latn |    531 |   18.4641 | False      |     0.9944 |    0.6742 |   0.9974 |                   63 |            0.0635 |              0 |
| lud_Latn |    412 |   13.4092 | False      |     0.9976 |    0.2579 |   0.9965 |                   94 |            0.1277 |              1 |
| goh_Latn |    410 |    7.2808 | False      |     0.8512 |    0.2407 |   0.9704 |                   35 |            0.0571 |              1 |
| asm_Latn |    390 |    5.0777 | False      |     0.9436 |    0.1793 |   0.9857 |                   71 |            0.169  |              0 |
| min_Arab |    388 |    3.7522 | False      |     0.9149 |    0.1549 |   0.982  |                  135 |            0.0519 |              0 |
| mns_Cyrl |    379 |   25.2977 | False      |     0.9921 |    0.2367 |   0.9974 |                  172 |            0.0465 |              5 |
| kas_Latn |    357 |   14.8894 | False      |     1      |    0.2633 |   0.999  |                   69 |            0.1304 |              1 |
| prg_Latn |    319 |    8.1135 | False      |     0.9655 |    0.3539 |   0.9915 |                  153 |            0.085  |              3 |
| lez_Cyrl |    316 |    5.541  | False      |     0.7025 |    0.1892 |   0.9258 |                   58 |            0.1034 |              0 |
| non_Latn |    310 |    3.2299 | False      |     0.7774 |    0.3485 |   0.9363 |                   70 |            0.0857 |              0 |
| mni_Latn |    298 |   23.1706 | False      |     1      |    0.198  |   0.9983 |                  155 |            0.071  |              0 |
| lld_Latn |    281 |    8.8996 | False      |     0.9609 |    0.2704 |   0.9876 |                  129 |            0.0388 |              1 |
| liv_Latn |    256 |   21.9218 | False      |     0.957  |    0.249  |   0.9897 |                  111 |            0.0721 |              0 |
| sdc_Latn |    255 |   11.9469 | False      |     0.9922 |    0.3755 |   0.9954 |                  119 |            0.1933 |              0 |
| tkl_Latn |    231 |    2.973  | False      |     0.619  |    0.6503 |   0.9482 |                   82 |            0.1098 |              0 |
| kca_Cyrl |    209 |   35.988  | False      |     1      |    0.2344 |   0.9996 |                  165 |            0.0909 |              8 |
| mev_Latn |    198 |   16.08   | False      |     0.9848 |    0.2462 |   0.9881 |                   47 |            0.234  |              0 |
| brx_Latn |    196 |   38.9965 | False      |     1      |    0.199  |   0.9999 |                   64 |            0.0469 |              2 |
| ory_Latn |    195 |    8.1066 | False      |     0.9949 |    0.1856 |   0.9955 |                   69 |            0.1594 |              1 |
| ota_Arab |    147 |    1.1635 | False      |     0.3333 |    0.102  |   0.8117 |                   87 |            0.046  |              0 |
| say_Latn |    139 |   15.1851 | False      |     0.9712 |    0.1556 |   0.9918 |                  103 |            0.0097 |              0 |
| tay_Latn |    132 |   57.2086 | False      |     1      |    0.2197 |   1      |                  129 |            0.1085 |              0 |
| kex_Deva |    128 |    1.3035 | False      |     0.4219 |    0.5    |   0.8516 |                   96 |            0.0729 |              0 |
| kak_Latn |    127 | -inf      | True       |     0      |  nan      |   0.9665 |                   28 |            0      |              0 |
| mus_Latn |    125 |   10.8071 | False      |     1      |    0.184  |   0.9979 |                   70 |            0.0714 |              0 |
| npi_Latn |    124 |   13.1777 | False      |     0.9919 |    0.1545 |   0.9947 |                   60 |            0.25   |              0 |
| tuv_Latn |    119 |   12.4545 | False      |     0.9748 |    0.319  |   0.9925 |                   56 |            0.0893 |              1 |
| tpn_Latn |    113 |    6.109  | False      |     0.7522 |    0.2588 |   0.9692 |                   83 |            0.0964 |              0 |
| mlu_Latn |    112 |   20.2917 | False      |     0.9286 |    0.3269 |   0.9847 |                  137 |            0.1022 |              2 |
| thv_Latn |    109 |    0.5257 | False      |     0.3578 |    0.3077 |   0.8935 |                   29 |            0.0345 |              0 |
| lki_Arab |     99 | -inf      | True       |     0      |  nan      |   0.9009 |                   14 |            0      |              0 |
| tpw_Latn |     93 |    2.9507 | False      |     0.8172 |    0.2368 |   0.9765 |                   57 |            0.0702 |              0 |
| kdr_Latn |     83 | -inf      | True       |     0      |  nan      |   0.9999 |                   16 |            0      |              0 |
| max_Latn |     76 |    0.6975 | False      |     0.4605 |    0.3714 |   0.9083 |                   34 |            0.0882 |              0 |
| zpg_Latn |     75 |   40.1204 | False      |     1      |    0.2133 |   0.9994 |                  137 |            0.0292 |              0 |
| lrc_Arab |     72 | -inf      | True       |     0      |  nan      |   0.7851 |                    6 |            0      |              0 |
| wew_Latn |     64 |   20.3241 | False      |     1      |    0.375  |   0.9983 |                  124 |            0.0645 |              0 |
| ndh_Latn |     64 |   10.3159 | False      |     0.9688 |    0.3548 |   0.9922 |                  104 |            0.0192 |              0 |
| arr_Latn |     60 |    2.9904 | False      |     0.7167 |    0.093  |   0.9593 |                   88 |            0.0455 |              3 |
| sju_Latn |     57 | -inf      | True       |     0      |  nan      |   0.9793 |                   12 |            0      |              0 |
| whg_Latn |     53 |   25.4137 | False      |     1      |    0.2642 |   0.9995 |                  135 |            0.0519 |              0 |
| krx_Latn |     51 |   41.3921 | False      |     1      |    0.3333 |   0.9998 |                  171 |            0.0643 |              1 |
| knx_Latn |     50 |   18.3055 | False      |     1      |    0.26   |   0.9975 |                   60 |            0.15   |              2 |
| drg_Latn |     50 | -inf      | True       |     0      |  nan      |   0.9785 |                   29 |            0      |              0 |
| mhw_Latn |     48 |   14.8486 | False      |     1      |    0.3958 |   0.9976 |                   93 |            0.0645 |              0 |
| lad_Hebr |     48 | -inf      | True       |     0      |  nan      |   0.528  |                   14 |            0      |              0 |
| hoc_Latn |     47 |    5.2174 | False      |     0.9574 |    0.2889 |   0.9904 |                   65 |            0.1385 |              2 |
| kei_Latn |     46 |   13.9888 | False      |     1      |    0.1739 |   0.9979 |                   36 |            0.1111 |              0 |
| yup_Latn |     45 |   26.7866 | False      |     1      |    0.2444 |   0.9995 |                   45 |            0.1111 |              2 |
| tzl_Latn |     43 | -inf      | True       |     0      |  nan      |   0.9587 |                   30 |            0      |              0 |
| csw_Latn |     42 |   18.6091 | False      |     1      |    0.2381 |   0.9981 |                   40 |            0.15   |              0 |
| sgh_Cyrl |     40 |   14.4981 | False      |     0.925  |    0.3514 |   0.9882 |                   97 |            0.1031 |              0 |
| nio_Cyrl |     38 | -inf      | True       |     0      |  nan      |   0.998  |                   17 |            0      |              0 |
| soe_Latn |     38 |   13.1403 | False      |     1      |    0.4211 |   0.9973 |                  161 |            0.0497 |              0 |
| ktz_Latn |     38 |   66.0326 | False      |     0.9737 |    0.2973 |   0.9955 |                   63 |            0.0159 |              1 |
| zpj_Latn |     36 | -inf      | True       |     0      |  nan      |   0.9997 |                   17 |            0      |              0 |
| sdo_Latn |     35 |   20.2974 | False      |     1      |    0.4857 |   0.9995 |                  102 |            0.0196 |              0 |
| mgm_Latn |     34 | -inf      | True       |     0      |  nan      |   0.9831 |                   27 |            0      |              0 |
| zsm_Arab |     32 | -inf      | True       |     0      |  nan      |   0.6956 |                    6 |            0      |              0 |
| eme_Latn |     32 |    2.1805 | False      |     0.5312 |    0.1176 |   0.9374 |                   96 |            0.0312 |              2 |
| fmu_Deva |     25 |   48.046  | False      |     1      |    0.32   |   1      |                  172 |            0.0233 |              0 |
| cwd_Cans |     25 |   13.6214 | False      |     0.72   |    0.6111 |   0.928  |                   38 |            0.0526 |              1 |
| mrv_Latn |     25 | -inf      | True       |     0      |  nan      |   0.9763 |                   17 |            0      |              0 |
| sby_Latn |     24 | -inf      | True       |     0      |  nan      |   0.9732 |                   17 |            0      |              0 |
| diu_Latn |     23 | -inf      | True       |     0      |  nan      |   0.9883 |                   22 |            0      |              0 |
| bwi_Latn |     22 |   34.5538 | False      |     1      |    0.3182 |   0.9998 |                   38 |            0.0526 |              0 |
| eto_Latn |     22 | -inf      | True       |     0      |  nan      |   0.9889 |                   26 |            0      |              0 |
| otw_Latn |     21 | -inf      | True       |     0      |  nan      |   0.9512 |                   14 |            0      |              0 |
| pmq_Latn |     20 |   78.4823 | False      |     1      |    0.05   |   1      |                   56 |            0.0714 |              2 |
| nla_Latn |     20 |   23.9325 | False      |     1      |    0.4    |   0.9985 |                   40 |            0.025  |              0 |
| mnw_Mymr |     19 |   75.7578 | False      |     1      |    0.5263 |   0.9918 |                   51 |            0.0784 |              0 |
| hoc_Wara |     18 |   73.3244 | False      |     1      |    0.1667 |   0.9981 |                   88 |            0.0227 |              1 |
| vot_Latn |     14 | -inf      | True       |     0      |  nan      |   0.9526 |                    9 |            0      |              0 |
| sel_Cyrl |     14 | -inf      | True       |     0      |  nan      |   1      |                   20 |            0      |              0 |
| tkr_Cyrl |      9 |   39.1278 | False      |     1      |    0.2222 |   1      |                  136 |            0.0809 |              0 |
| itl_Cyrl |      9 |   61.6675 | False      |     1      |    0      |   0.9995 |                  172 |            0.0174 |              1 |
| abq_Cyrl |      8 | -inf      | True       |     0      |  nan      |   0.9855 |                   25 |            0      |              0 |
| dng_Cyrl |      7 |   47.4887 | False      |     1      |    0.2857 |   0.9982 |                  141 |            0.0638 |              1 |
| xum_Latn |      7 | -inf      | True       |     0      |  nan      |   0.9505 |                   14 |            0      |              0 |
| aqz_Latn |      6 |    0.7729 | False      |     0      |  nan      |   0.8654 |                   30 |            0      |              0 |
| ldn_Latn |      5 | -inf      | True       |     0      |  nan      |   0.998  |                   14 |            0      |              0 |
| lbe_Cyrl |      5 | -inf      | True       |     0      |  nan      |   0.9978 |                   28 |            0      |              0 |
| csw_Cans |      5 | -inf      | True       |     0      |  nan      |   0.783  |                   15 |            0      |              0 |
| bor_Latn |      5 | -inf      | True       |     0      |  nan      |   0.9585 |                   15 |            0      |              0 |
| kap_Cyrl |      4 |   54.7199 | False      |     1      |    0.25   |   1      |                  136 |            0.0441 |              0 |
| kqa_Latn |      3 |   72.4122 | False      |     1      |    0.3333 |   1      |                  156 |            0.0449 |              2 |
| mwf_Latn |      3 |  103.13   | False      |     1      |    0      |   1      |                   48 |            0.0625 |              2 |
| arc_Syrc |      3 | -inf      | True       |     0      |  nan      | nan      |                    0 |          nan      |              0 |
| sjo_Mong |      1 |   74.2139 | False      |     1      |    0      |   1      |                  122 |            0.0328 |              4 |
