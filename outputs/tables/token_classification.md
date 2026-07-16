### Aggregate token classification across all pairs

|         Category | # Tokens | % Tokens | Mean |delta| | Sum |KL| | % of KL |
|------------------|----------|----------|--------------|----------|---------|
|    Function word |       33 |     11.0 |         5.02 |   0.1674 |    15.7 |
|     Content word |       80 |     26.7 |         4.90 |   0.2431 |    22.8 |
|     Morph. affix |       92 |     30.7 |         5.33 |   0.3473 |    32.6 |
| Domain/religious |        2 |      0.7 |         8.82 |   0.0033 |     0.3 |
|      Punctuation |       24 |      8.0 |         5.91 |   0.1115 |    10.5 |
|  Multi-word unit |       20 |      6.7 |         7.40 |   0.0840 |     7.9 |
| Char/phonotactic |       49 |     16.3 |         2.81 |   0.1088 |    10.2 |

### High-resource pairs (both >50k) (13 pairs: Indonesian / Malay, Danish / Bokmal, English / Scots, MSA / Egyptian Arabic, MSA / Moroccan Arabic (+8 more))

|         Category | # Tokens | % Tokens |
|------------------|----------|----------|
|    Function word |       33 |     12.7 |
|     Content word |       51 |     19.6 |
|     Morph. affix |       87 |     33.5 |
| Domain/religious |        2 |      0.8 |
|      Punctuation |       19 |      7.3 |
|  Multi-word unit |       20 |      7.7 |
| Char/phonotactic |       48 |     18.5 |

### Mixed-resource pairs (one <10k) (1 pairs: Hindi / Angika)

|     Category | # Tokens | % Tokens |
|--------------|----------|----------|
| Content word |       15 |     75.0 |
| Morph. affix |        2 |     10.0 |
|  Punctuation |        3 |     15.0 |

### Pairs with >5% of KL from domain/religious tokens

|               Pair |   Lang A |   Lang B | % KL from domain terms |
|--------------------|----------|----------|------------------------|
| Indonesian / Malay | ind_Latn | zsm_Latn |                    7.3 |

### Classified tokens: Indonesian / Malay (ind_Latn vs zsm_Latn)

|     Token |         Category | Score (ind_Latn) | Score (zsm_Latn) | Delta | KL contrib |
|-----------|------------------|------------------|------------------|-------|------------|
| 'Ġkarena' |    Function word |            -8.09 |           -14.37 | +6.27 |     0.0019 |
|  'Ġbahwa' |    Function word |            -7.86 |           -12.21 | +4.36 |     0.0017 |
| 'ĠYehuwa' | Domain/religious |            -9.05 |           -19.02 | +9.97 |     0.0012 |
|   'Ġbisa' |     Content word |            -8.49 |           -14.03 | +5.54 |     0.0011 |
| 'Ġbagian' |     Content word |            -9.23 |           -18.95 | +9.72 |     0.0009 |
|      'Ġ-' |      Punctuation |            -7.75 |            -9.91 | +2.17 |     0.0009 |
|      'si' | Char/phonotactic |            -6.76 |            -7.52 | +0.76 |     0.0009 |
|     'nya' |     Morph. affix |            -6.07 |            -6.43 | +0.35 |     0.0008 |
|      'an' | Char/phonotactic |            -5.89 |            -5.62 | -0.27 |    -0.0007 |
|     ',"Ġ' |      Punctuation |            -8.56 |           -12.04 | +3.47 |     0.0007 |
|   'Ġsaya' |    Function word |            -8.04 |           -10.07 | +2.03 |     0.0007 |
|    'Ġtim' |     Content word |            -9.67 |           -18.54 | +8.87 |     0.0006 |
|    'beda' |     Morph. affix |            -9.60 |           -17.78 | +8.18 |     0.0006 |
|       'h' | Char/phonotactic |            -7.03 |            -6.41 | -0.62 |    -0.0006 |
|   'Ġdari' |    Function word |            -6.96 |            -7.52 | +0.56 |     0.0005 |
|   'takan' |     Morph. affix |            -8.07 |            -9.74 | +1.67 |     0.0005 |
|   'Ġyang' |    Function word |            -6.39 |            -6.68 | +0.29 |     0.0005 |
|       'i' | Char/phonotactic |            -6.41 |            -6.12 | -0.29 |    -0.0005 |
|   'ĠAnda' |     Content word |            -8.97 |           -12.54 | +3.58 |     0.0005 |
|      'tu' | Char/phonotactic |            -7.58 |            -8.39 | +0.82 |     0.0004 |

### Classified tokens: Hindi / Angika (hin_Deva vs anp_Deva)

|           Token |     Category | Score (hin_Deva) | Score (anp_Deva) | Delta | KL contrib |
|-----------------|--------------|------------------|------------------|-------|------------|
|          'Ġà¥¤' |  Punctuation |            -6.38 |            -9.47 | +3.09 |     0.0052 |
|            'Ġ,' |  Punctuation |            -8.41 |           -16.35 | +7.95 |     0.0018 |
|    'Ġà¤¤à¥ģà¤®' | Content word |            -8.30 |           -12.97 | +4.67 |     0.0012 |
|       'Ġà¤¹à¥Ī' | Content word |            -6.45 |            -7.18 | +0.72 |     0.0011 |
|           'à¥ĩ' | Morph. affix |            -6.58 |            -7.32 | +0.74 |     0.0010 |
|       'Ġà¤ķà¤¿' | Content word |            -6.73 |            -7.52 | +0.78 |     0.0009 |
|        'à¤¨à¥ĩ' | Content word |            -6.71 |            -7.47 | +0.76 |     0.0009 |
|       'Ġà¤ķà¥Ģ' | Content word |            -6.15 |            -6.57 | +0.42 |     0.0009 |
|       'Ġà¤īà¤¸' | Content word |            -7.09 |            -8.15 | +1.06 |     0.0009 |
|       'Ġà¤Ķà¤°' | Content word |            -6.12 |            -6.51 | +0.39 |     0.0009 |
|       'Ġà¤¸à¥ĩ' | Content word |            -6.28 |            -6.72 | +0.44 |     0.0008 |
|       'Ġà¤ķà¥ĩ' | Content word |            -5.88 |            -5.58 | -0.29 |    -0.0008 |
|       'Ġà¤ķà¥ĭ' | Content word |            -6.57 |            -7.16 | +0.59 |     0.0008 |
|           'à¥į' | Morph. affix |            -7.55 |            -9.05 | +1.50 |     0.0008 |
|    'Ġà¤®à¥ĩà¤Ĥ' | Content word |            -5.95 |            -6.21 | +0.27 |     0.0007 |
|             '?' |  Punctuation |            -9.28 |           -16.72 | +7.44 |     0.0007 |
|          'Ġà¤ķ' | Content word |            -7.25 |            -6.30 | -0.94 |    -0.0007 |
| 'Ġà¤¨à¤¹à¥Ģà¤Ĥ' | Content word |            -7.53 |            -8.72 | +1.19 |     0.0006 |
|       'Ġà¤ĩà¤¸' | Content word |            -6.87 |            -7.47 | +0.60 |     0.0006 |
|       'Ġà¤ķà¤°' | Content word |            -6.89 |            -7.48 | +0.60 |     0.0006 |

### Classified tokens: Danish / Bokmal (dan_Latn vs nob_Latn)

|      Token |         Category | Score (dan_Latn) | Score (nob_Latn) |  Delta | KL contrib |
|------------|------------------|------------------|------------------|--------|------------|
|      'Ġaf' |    Function word |            -5.81 |           -11.57 |  +5.77 |     0.0174 |
|      'Ġud' | Char/phonotactic |            -7.07 |           -18.82 | +11.76 |     0.0100 |
|     'Ġind' |     Content word |            -7.49 |           -18.74 | +11.25 |     0.0063 |
|      'hed' |     Morph. affix |            -7.12 |           -13.24 |  +6.13 |     0.0050 |
|   ',ĠderĠ' |  Multi-word unit |            -7.27 |           -12.08 |  +4.81 |     0.0034 |
|      'rÃ¦' |     Morph. affix |            -7.47 |           -12.86 |  +5.39 |     0.0031 |
| 'Ġartikel' |     Content word |            -8.19 |           -19.03 | +10.84 |     0.0030 |
|     'Ġgen' |     Content word |            -8.17 |           -18.39 | +10.22 |     0.0029 |
|       'er' | Char/phonotactic |            -5.48 |            -6.11 |  +0.63 |     0.0027 |
|      'Ġop' | Char/phonotactic |            -7.33 |           -11.18 |  +3.85 |     0.0025 |
|      'ede' |     Morph. affix |            -8.31 |           -18.32 | +10.01 |     0.0025 |
|        'Ġ' |      Punctuation |            -5.50 |            -6.10 |  +0.60 |     0.0024 |
|      'tag' |     Morph. affix |            -7.91 |           -14.37 |  +6.46 |     0.0024 |
|      'bej' |     Morph. affix |            -8.48 |           -19.03 | +10.56 |     0.0022 |
|     'ligt' |     Morph. affix |            -8.53 |           -19.02 | +10.49 |     0.0021 |
|      'Ã¦l' |     Morph. affix |            -7.74 |           -12.38 |  +4.64 |     0.0020 |
|     ',Ġat' |     Morph. affix |            -7.63 |           -11.63 |  +4.00 |     0.0019 |
|      'Ã¦n' |     Morph. affix |            -8.49 |           -17.76 |  +9.27 |     0.0019 |
|     'tion' |     Morph. affix |            -7.91 |           -12.39 |  +4.48 |     0.0016 |
|      'gÃ¸' |     Morph. affix |            -8.72 |           -18.82 | +10.09 |     0.0016 |

### Classified tokens: English / Scots (eng_Latn vs sco_Latn)

|     Token |        Category | Score (eng_Latn) | Score (sco_Latn) |  Delta | KL contrib |
|-----------|-----------------|------------------|------------------|--------|------------|
|     'Ġto' |   Function word |            -6.05 |            -9.74 |  +3.70 |     0.0088 |
|    'Ġand' |   Function word |            -6.32 |           -10.08 |  +3.76 |     0.0068 |
|     'Ġof' |   Function word |            -6.25 |            -8.90 |  +2.65 |     0.0051 |
|     'ing' |    Morph. affix |            -6.40 |            -9.27 |  +2.87 |     0.0048 |
|   'Ġhave' |   Function word |            -7.48 |           -12.92 |  +5.45 |     0.0031 |
|   'Ġwith' |   Function word |            -7.19 |           -11.02 |  +3.83 |     0.0029 |
|   'Ġsaid' |   Function word |            -7.25 |           -11.10 |  +3.86 |     0.0027 |
|    'Ġnot' |   Function word |            -7.58 |           -12.49 |  +4.91 |     0.0025 |
|    'Ġone' |   Function word |            -8.41 |           -18.73 | +10.31 |     0.0023 |
|    'Ġhas' |   Function word |            -7.62 |           -12.26 |  +4.64 |     0.0023 |
|     'ted' |    Morph. affix |            -7.54 |           -11.51 |  +3.97 |     0.0021 |
|    'Ġwas' |   Function word |            -6.98 |            -9.22 |  +2.24 |     0.0021 |
| 'ĠofĠthe' | Multi-word unit |            -7.25 |           -10.19 |  +2.93 |     0.0021 |
|    'Ġyou' |    Content word |            -7.28 |           -10.24 |  +2.95 |     0.0020 |
|   'Ġfrom' |   Function word |            -7.59 |           -11.60 |  +4.01 |     0.0020 |
|      "'t" |    Morph. affix |            -8.43 |           -17.61 |  +9.18 |     0.0020 |
|   ',Ġand' |    Morph. affix |            -7.69 |           -12.09 |  +4.40 |     0.0020 |
|    'Ġhad' |   Function word |            -7.89 |           -12.60 |  +4.71 |     0.0018 |
|    'Ġals' |    Content word |            -8.38 |           -15.93 |  +7.55 |     0.0017 |
|   'Ġover' |   Function word |            -8.55 |           -17.36 |  +8.80 |     0.0017 |

### Classified tokens: MSA / Egyptian Arabic (arb_Arab vs arz_Arab)

|        Token |         Category | Score (arb_Arab) | Score (arz_Arab) |  Delta | KL contrib |
|--------------|------------------|------------------|------------------|--------|------------|
|         'ï»' |     Morph. affix |            -6.65 |           -19.25 | +12.60 |     0.0162 |
|        'Ġ/Ġ' |  Multi-word unit |            -7.29 |           -19.25 | +11.96 |     0.0082 |
|      'ĠØ§ÙĦ' |     Content word |            -5.40 |            -6.42 |  +1.02 |     0.0046 |
|       'ÙĬØ©' |     Morph. affix |            -6.23 |            -8.22 |  +1.99 |     0.0039 |
|          '»' |      Punctuation |            -7.34 |           -12.74 |  +5.40 |     0.0035 |
|         'ØĽ' | Char/phonotactic |            -8.14 |           -19.25 | +11.11 |     0.0032 |
|       'ĠØĮĠ' |  Multi-word unit |            -6.82 |            -9.34 |  +2.53 |     0.0028 |
|          '·' |      Punctuation |            -7.70 |           -13.70 |  +6.00 |     0.0027 |
|    'ĠØ§ÙĦØ£' |     Content word |            -6.64 |            -8.53 |  +1.89 |     0.0025 |
|      'ĠÙģÙĬ' |     Content word |            -6.44 |            -7.93 |  +1.49 |     0.0024 |
|    'ĠØ§ÙĦÙħ' |     Content word |            -6.12 |            -7.19 |  +1.07 |     0.0024 |
|    'Ø©ĠØ§ÙĦ' |     Content word |            -6.61 |            -8.32 |  +1.72 |     0.0023 |
|       'Ø§Øª' |     Morph. affix |            -6.42 |            -7.75 |  +1.34 |     0.0022 |
|      'ĠÙĪØª' |     Content word |            -8.20 |           -16.12 |  +7.91 |     0.0022 |
| 'ĠØ§ÙĦØªÙĬĠ' |  Multi-word unit |            -7.64 |           -12.13 |  +4.49 |     0.0022 |
|      'ĠØ£ÙĨ' |     Content word |            -7.13 |            -9.69 |  +2.55 |     0.0020 |
|   'ĠØ¥ÙĦÙīĠ' |  Multi-word unit |            -7.53 |           -11.34 |  +3.80 |     0.0020 |
|       'Ø£ÙĨ' |     Morph. affix |            -7.42 |           -10.69 |  +3.27 |     0.0020 |
|         'ÙĬ' | Char/phonotactic |            -5.69 |            -6.26 |  +0.57 |     0.0019 |
|        'Ø©Ġ' |     Morph. affix |            -6.74 |            -8.39 |  +1.64 |     0.0019 |

### Classified tokens: MSA / Moroccan Arabic (arb_Arab vs ary_Arab)

|        Token |         Category | Score (arb_Arab) | Score (ary_Arab) |  Delta | KL contrib |
|--------------|------------------|------------------|------------------|--------|------------|
|         'ï»' |     Morph. affix |            -6.65 |           -19.09 | +12.44 |     0.0160 |
|        'Ġ/Ġ' |  Multi-word unit |            -7.29 |           -19.09 | +11.80 |     0.0081 |
|          '»' |      Punctuation |            -7.34 |           -12.98 |  +5.63 |     0.0036 |
|     'ĠØĮĠÙĪ' |  Multi-word unit |            -7.10 |           -11.16 |  +4.06 |     0.0034 |
|        'Ġ-Ġ' |  Multi-word unit |            -8.05 |           -18.51 | +10.46 |     0.0033 |
|        'ĠØĮ' | Char/phonotactic |            -6.94 |           -10.34 |  +3.40 |     0.0033 |
|          '·' |      Punctuation |            -7.70 |           -14.83 |  +7.13 |     0.0032 |
|         'ØĽ' | Char/phonotactic |            -8.14 |           -19.09 | +10.95 |     0.0032 |
|       'ÙħÙħ' |     Morph. affix |            -8.10 |           -18.19 | +10.08 |     0.0030 |
|       'ĠØĮĠ' |  Multi-word unit |            -6.82 |            -9.44 |  +2.63 |     0.0029 |
|        'ĠØ§' |     Content word |            -7.14 |            -9.33 |  +2.19 |     0.0017 |
|         ')Ġ' |     Morph. affix |            -8.67 |           -18.28 |  +9.61 |     0.0016 |
|    'ĠØ§ÙĦÙħ' |     Content word |            -6.12 |            -6.83 |  +0.71 |     0.0016 |
|         'Ø£' |     Morph. affix |            -6.40 |            -7.33 |  +0.93 |     0.0015 |
|         'Ġ/' |      Punctuation |            -8.82 |           -19.09 | +10.27 |     0.0015 |
|          '¹' |      Punctuation |            -8.43 |           -14.83 |  +6.39 |     0.0014 |
|      'ĠØ£ÙĨ' |     Content word |            -7.13 |            -8.85 |  +1.71 |     0.0014 |
|       'Ø£ÙĨ' |     Morph. affix |            -7.42 |            -9.61 |  +2.19 |     0.0013 |
|    'ĠØ§ÙĦØ£' |     Content word |            -6.64 |            -7.65 |  +1.00 |     0.0013 |
| 'ĠØ§ÙĦØªÙĬĠ' |  Multi-word unit |            -7.64 |           -10.28 |  +2.65 |     0.0013 |

### Classified tokens: Persian / Gilaki (fas_Arab vs glk_Arab)

|      Token |         Category | Score (fas_Arab) | Score (glk_Arab) | Delta | KL contrib |
|------------|------------------|------------------|------------------|-------|------------|
| 'ĠÙħÛĮâĢĮ' |     Content word |            -6.75 |           -11.19 | +4.45 |     0.0052 |
|    'ĠØ¯Ø±' |     Content word |            -5.68 |            -7.04 | +1.36 |     0.0046 |
|    'ĠØ±Ø§' |     Content word |            -6.28 |            -8.69 | +2.41 |     0.0045 |
|  'ĠØ¨ÙĪØ¯' |     Content word |            -6.88 |           -10.75 | +3.87 |     0.0040 |
|    'ĠØ¢ÙĨ' |     Content word |            -6.84 |           -10.56 | +3.72 |     0.0040 |
|    'ĠØ´Ø¯' |     Content word |            -7.29 |           -12.75 | +5.47 |     0.0037 |
|    'ĠØ¨Ùĩ' |     Content word |            -5.90 |            -7.23 | +1.33 |     0.0036 |
|     'ÙĨØ¯' |     Morph. affix |            -6.21 |            -7.90 | +1.69 |     0.0034 |
|        '.' |      Punctuation |            -5.35 |            -6.03 | +0.68 |     0.0033 |
|  'âĢĮÙĩØ§' |     Content word |            -7.11 |           -10.99 | +3.88 |     0.0032 |
|     'ÙĩØ§' |     Morph. affix |            -7.83 |           -15.71 | +7.88 |     0.0031 |
|  'ĠÚ©Ø±Ø¯' |     Content word |            -7.15 |           -10.48 | +3.33 |     0.0026 |
|  'ĠØ§Ø³Øª' |     Content word |            -6.70 |            -8.77 | +2.07 |     0.0026 |
|  'ĠØ§ÛĮÙĨ' |     Content word |            -6.58 |            -8.37 | +1.79 |     0.0025 |
|    'ĠØ§Ø²' |     Content word |            -6.47 |            -7.61 | +1.14 |     0.0018 |
|        'Ġ' |      Punctuation |            -5.55 |            -5.10 | -0.44 |    -0.0017 |
|   'Ø§Ø³Øª' |     Content word |            -7.70 |           -10.82 | +3.12 |     0.0014 |
|       'Ùħ' | Char/phonotactic |            -5.94 |            -5.41 | -0.53 |    -0.0014 |
|    'ÙĩâĢĮ' |     Morph. affix |            -7.28 |            -9.30 | +2.02 |     0.0014 |
|  'ĠØ´Ø¯Ùĩ' |     Content word |            -7.92 |           -11.55 | +3.62 |     0.0013 |

### Classified tokens: Mandarin / Wu (cmn_Hani vs wuu_Hani)

|     Token |         Category | Score (cmn_Hani) | Score (wuu_Hani) |  Delta | KL contrib |
|-----------|------------------|------------------|------------------|--------|------------|
|     'çļĦ' |     Morph. affix |            -5.85 |            -9.64 |  +3.79 |     0.0109 |
|       'Ġ' |      Punctuation |            -5.27 |            -6.10 |  +0.84 |     0.0043 |
|     'ä½ł' |     Morph. affix |            -8.23 |           -19.04 | +10.82 |     0.0029 |
|    'ĠãĢĤ' |     Content word |            -8.68 |           -18.47 |  +9.80 |     0.0017 |
|     'åľ¨' |     Morph. affix |            -7.04 |            -8.66 |  +1.62 |     0.0014 |
|   'Ġï¼ĮĠ' |  Multi-word unit |            -8.33 |           -14.01 |  +5.68 |     0.0014 |
|     'ä¸į' |     Morph. affix |            -7.62 |            -9.91 |  +2.29 |     0.0011 |
|     'çĤº' |     Morph. affix |            -7.75 |           -10.29 |  +2.54 |     0.0011 |
|      'ä»' |     Morph. affix |            -6.60 |            -7.35 |  +0.75 |     0.0010 |
|     'Ġci' | Char/phonotactic |            -8.60 |           -13.87 |  +5.27 |     0.0010 |
| 'tationĠ' |     Content word |            -8.49 |           -13.13 |  +4.64 |     0.0010 |
|      'ĴĮ' | Char/phonotactic |            -7.63 |            -9.46 |  +1.83 |     0.0009 |
|       'ī' | Char/phonotactic |            -6.46 |            -6.00 |  -0.46 |    -0.0007 |
|      'ĸĠ' | Char/phonotactic |            -9.53 |           -19.27 |  +9.73 |     0.0007 |
|     'æĪĳ' |     Morph. affix |            -8.07 |           -10.30 |  +2.23 |     0.0007 |
|       'Ń' | Char/phonotactic |            -6.86 |            -6.20 |  -0.66 |    -0.0007 |
|       'æ' | Char/phonotactic |            -5.09 |            -4.99 |  -0.10 |    -0.0006 |
|       'Ĳ' | Char/phonotactic |            -6.85 |            -6.28 |  -0.57 |    -0.0006 |
|       'º' | Char/phonotactic |            -6.31 |            -5.98 |  -0.33 |    -0.0006 |
|      'Ġå' | Char/phonotactic |            -7.53 |            -8.60 |  +1.07 |     0.0006 |

### Classified tokens: Mandarin / Cantonese (cmn_Hani vs yue_Hani)

|     Token |         Category | Score (cmn_Hani) | Score (yue_Hani) |  Delta | KL contrib |
|-----------|------------------|------------------|------------------|--------|------------|
|     'çļĦ' |     Morph. affix |            -5.85 |           -10.25 |  +4.40 |     0.0127 |
|       'å' | Char/phonotactic |            -5.18 |            -4.46 |  -0.72 |    -0.0041 |
|     'åľ¨' |     Morph. affix |            -7.04 |           -10.30 |  +3.27 |     0.0029 |
|       'Ġ' |      Punctuation |            -5.27 |            -5.72 |  +0.45 |     0.0023 |
| 'tationĠ' |     Content word |            -8.49 |           -19.39 | +10.89 |     0.0022 |
|     'æĺ¯' |     Morph. affix |            -7.24 |           -10.33 |  +3.09 |     0.0022 |
|      'ä¸' |     Morph. affix |            -6.68 |            -8.01 |  +1.33 |     0.0017 |
|     'Ġci' | Char/phonotactic |            -8.60 |           -17.01 |  +8.41 |     0.0015 |
|      'äº' | Char/phonotactic |            -6.50 |            -7.50 |  +1.00 |     0.0015 |
|      'ĴĮ' | Char/phonotactic |            -7.63 |           -10.30 |  +2.67 |     0.0013 |
|     'ä¸ª' |     Morph. affix |            -8.99 |           -19.19 | +10.20 |     0.0013 |
|     'ï¼Į' |     Morph. affix |            -5.63 |            -5.94 |  +0.31 |     0.0011 |
|     'ä¹Ł' |     Morph. affix |            -8.42 |           -12.77 |  +4.34 |     0.0010 |
|      'ä»' |     Morph. affix |            -6.60 |            -7.22 |  +0.61 |     0.0008 |
|     'âĢĿ' |     Morph. affix |            -8.59 |           -12.80 |  +4.21 |     0.0008 |
|       'Ĩ' | Char/phonotactic |            -6.34 |            -6.77 |  +0.43 |     0.0008 |
|       'é' | Char/phonotactic |            -5.51 |            -5.33 |  -0.19 |    -0.0008 |
|       'Ł' | Char/phonotactic |            -6.31 |            -5.91 |  -0.41 |    -0.0007 |
|      'åĢ' | Char/phonotactic |            -8.30 |           -11.22 |  +2.92 |     0.0007 |
|      'åĽ' | Char/phonotactic |            -7.32 |            -8.40 |  +1.08 |     0.0007 |

### Classified tokens: Kinyarwanda / Kirundi (kin_Latn vs run_Latn)

|    Token |         Category | Score (kin_Latn) | Score (run_Latn) | Delta | KL contrib |
|----------|------------------|------------------|------------------|-------|------------|
|   'Ġcya' |     Content word |            -6.85 |            -9.77 | +2.92 |     0.0031 |
|    'byo' |     Morph. affix |            -7.33 |           -11.03 | +3.70 |     0.0024 |
|   'raga' |     Morph. affix |            -7.64 |           -10.87 | +3.24 |     0.0016 |
|   'Ġbya' |     Content word |            -7.52 |           -10.29 | +2.77 |     0.0015 |
|     'za' | Char/phonotactic |            -5.88 |            -6.35 | +0.46 |     0.0013 |
|    'Ġic' | Char/phonotactic |            -7.87 |           -10.79 | +2.92 |     0.0011 |
|    'byi' |     Morph. affix |            -8.25 |           -12.30 | +4.05 |     0.0011 |
|     'nj' | Char/phonotactic |            -8.35 |           -12.70 | +4.35 |     0.0010 |
|    'bye' |     Morph. affix |            -8.20 |           -11.86 | +3.66 |     0.0010 |
|     'sh' | Char/phonotactic |            -7.69 |            -9.88 | +2.19 |     0.0010 |
|   'Ġbyo' |     Content word |            -8.08 |           -11.17 | +3.09 |     0.0010 |
|   'reba' |     Morph. affix |            -9.16 |           -18.23 | +9.07 |     0.0010 |
|    'bya' |     Morph. affix |            -8.18 |           -11.43 | +3.25 |     0.0009 |
|      'j' | Char/phonotactic |            -8.05 |           -10.85 | +2.80 |     0.0009 |
| 'kristo' |     Content word |            -8.74 |           -14.25 | +5.51 |     0.0009 |
|     'ya' | Char/phonotactic |            -7.02 |            -7.97 | +0.96 |     0.0009 |
|    'Ġby' |    Function word |            -8.08 |           -10.79 | +2.71 |     0.0008 |
|     'ra' | Char/phonotactic |            -5.83 |            -5.59 | -0.24 |    -0.0007 |
|   'ngwa' |     Morph. affix |            -7.82 |            -9.58 | +1.76 |     0.0007 |
|   'Ġcyo' |     Content word |            -8.38 |           -11.30 | +2.92 |     0.0007 |

### Classified tokens: Bokmal / Nynorsk (nob_Latn vs nno_Latn)

|     Token |         Category | Score (nob_Latn) | Score (nno_Latn) |  Delta | KL contrib |
|-----------|------------------|------------------|------------------|--------|------------|
|  'ĠikkeĠ' |  Multi-word unit |            -7.35 |           -17.14 |  +9.80 |     0.0063 |
|     'Ġen' |    Function word |            -6.43 |            -9.58 |  +3.14 |     0.0051 |
|     'het' |     Morph. affix |            -7.37 |           -13.40 |  +6.04 |     0.0038 |
|    'Ġble' |     Content word |            -7.23 |           -12.38 |  +5.15 |     0.0037 |
|     'Ġet' |    Function word |            -7.06 |           -11.14 |  +4.08 |     0.0035 |
|     'lig' |     Morph. affix |            -6.98 |           -10.75 |  +3.77 |     0.0035 |
|     'Ġje' |    Function word |            -8.08 |           -18.53 | +10.44 |     0.0032 |
|      'er' | Char/phonotactic |            -6.11 |            -7.26 |  +1.15 |     0.0025 |
|    'Ġfra' |     Content word |            -7.61 |           -12.24 |  +4.63 |     0.0023 |
| 'ĠJehova' | Domain/religious |            -8.17 |           -15.83 |  +7.66 |     0.0022 |
|      'Ġh' | Char/phonotactic |            -6.92 |            -8.71 |  +1.79 |     0.0018 |
|      'se' | Char/phonotactic |            -6.86 |            -8.48 |  +1.62 |     0.0017 |
|       't' | Char/phonotactic |            -5.36 |            -5.05 |  -0.31 |    -0.0014 |
|    'sker' |     Morph. affix |            -8.82 |           -18.28 |  +9.46 |     0.0014 |
|    'lige' |     Morph. affix |            -7.72 |           -10.85 |  +3.13 |     0.0014 |
|       's' | Char/phonotactic |            -5.40 |            -5.70 |  +0.30 |     0.0014 |
|     'Ġde' |    Function word |            -6.86 |            -8.06 |  +1.20 |     0.0013 |
|    'Ġfor' |    Function word |            -6.06 |            -6.59 |  +0.52 |     0.0012 |
|     'ĠÃ¥' |     Content word |            -6.23 |            -6.84 |  +0.61 |     0.0012 |
|     'vor' |     Morph. affix |            -7.89 |           -11.08 |  +3.20 |     0.0012 |

### Classified tokens: Hebrew / Biblical Hebrew (heb_Hebr vs hbo_Hebr)

|     Token |        Category | Score (heb_Hebr) | Score (hbo_Hebr) |  Delta | KL contrib |
|-----------|-----------------|------------------|------------------|--------|------------|
|       '.' |     Punctuation |            -5.75 |           -19.65 | +13.89 |     0.0441 |
|      ',Ġ' |    Morph. affix |            -6.05 |           -19.65 | +13.59 |     0.0320 |
|      '×ķ' |    Morph. affix |            -4.71 |            -5.75 |  +1.04 |     0.0094 |
| 'Ġ×©×ľĠ×' | Multi-word unit |            -7.26 |           -19.65 | +12.38 |     0.0087 |
|     'Ġ×Ķ' |    Content word |            -5.41 |            -6.87 |  +1.45 |     0.0065 |
|       'Ġ' |     Punctuation |            -4.70 |            -5.33 |  +0.62 |     0.0056 |
|      '×Ķ' |    Morph. affix |            -5.03 |            -5.89 |  +0.86 |     0.0056 |
|    '×ķ×ª' |    Morph. affix |            -5.92 |            -7.92 |  +2.00 |     0.0054 |
|      '×ª' |    Morph. affix |            -5.08 |            -5.95 |  +0.86 |     0.0053 |
|      'Ġ"' |     Punctuation |            -7.74 |           -19.65 | +11.91 |     0.0052 |
|       '"' |     Punctuation |            -7.98 |           -19.65 | +11.67 |     0.0040 |
|       "'" |     Punctuation |            -8.02 |           -19.65 | +11.63 |     0.0038 |
|     'Ġ×©' |     Punctuation |            -6.07 |            -7.51 |  +1.44 |     0.0033 |
|       ',' |     Punctuation |            -8.16 |           -19.65 | +11.49 |     0.0033 |
|    '×Ļ×Ŀ' |    Morph. affix |            -6.25 |            -7.95 |  +1.69 |     0.0033 |
|      '×ŀ' |    Morph. affix |            -5.49 |            -6.25 |  +0.76 |     0.0032 |
|      '×Ĺ' |    Morph. affix |            -5.44 |            -6.15 |  +0.71 |     0.0031 |
|      '×Ĵ' |    Morph. affix |            -6.19 |            -7.53 |  +1.34 |     0.0027 |
|    '×Ļ×Ļ' |    Morph. affix |            -7.36 |           -11.67 |  +4.31 |     0.0027 |
|       '?' |     Punctuation |            -8.34 |           -19.65 | +11.31 |     0.0027 |

### Classified tokens: Russian / Ukrainian (rus_Cyrl vs ukr_Cyrl)

|         Token |         Category | Score (rus_Cyrl) | Score (ukr_Cyrl) |  Delta | KL contrib |
|---------------|------------------|------------------|------------------|--------|------------|
|         'ĠÐ¸' |     Content word |            -6.46 |           -19.11 | +12.65 |     0.0197 |
|        'Ð¾Ð¹' |     Morph. affix |            -7.01 |           -18.42 | +11.41 |     0.0103 |
|    'ÐµÐ½Ð¸Ñı' |     Content word |            -7.07 |           -18.68 | +11.61 |     0.0098 |
|        'Ð½Ñĭ' |     Morph. affix |            -7.15 |           -19.18 | +12.02 |     0.0094 |
|          'Ñĭ' | Char/phonotactic |            -7.18 |           -19.18 | +12.00 |     0.0092 |
|      'Ð½ÑĭÑħ' |     Content word |            -7.36 |           -19.18 | +11.82 |     0.0076 |
|        'Ð¶Ð´' |     Morph. affix |            -7.46 |           -18.71 | +11.25 |     0.0065 |
|          'Ð¸' |     Morph. affix |            -5.80 |            -7.79 |  +1.99 |     0.0060 |
|        'Ð¸Ð¸' |     Morph. affix |            -7.57 |           -19.16 | +11.60 |     0.0060 |
|        'ĠÐ¸Ġ' |  Multi-word unit |            -7.58 |           -19.10 | +11.51 |     0.0059 |
|       'ĠÑįÑĤ' |     Content word |            -7.60 |           -19.18 | +11.57 |     0.0058 |
|        'Ð¸Ñı' |     Morph. affix |            -7.60 |           -18.78 | +11.18 |     0.0056 |
|       'ĠÐ²Ñĭ' |     Content word |            -7.67 |           -19.18 | +11.51 |     0.0054 |
|    'Ð°Ð½Ð¸Ñı' |     Content word |            -7.67 |           -19.16 | +11.49 |     0.0053 |
|          'Ðµ' | Char/phonotactic |            -5.82 |            -7.54 |  +1.72 |     0.0051 |
|      'Ð½ÑĭÐ¼' |     Content word |            -7.74 |           -19.18 | +11.44 |     0.0050 |
|    ',ĠÑĩÑĤÐ¾' |     Content word |            -7.52 |           -16.36 |  +8.84 |     0.0048 |
|        'ÑĭÑħ' |     Morph. affix |            -7.81 |           -19.18 | +11.37 |     0.0046 |
| 'ĠÐºÐ¾ÑĤÐ¾ÑĢ' |     Content word |            -7.75 |           -18.29 | +10.53 |     0.0045 |
|    'ÐµÐ½Ð¸Ðµ' |     Content word |            -7.60 |           -16.28 |  +8.68 |     0.0043 |

### Classified tokens: Spanish / Portuguese (spa_Latn vs por_Latn)

|    Token |         Category | Score (spa_Latn) | Score (por_Latn) |  Delta | KL contrib |
|----------|------------------|------------------|------------------|--------|------------|
|    'Ġel' |    Function word |            -6.08 |           -19.01 | +12.93 |     0.0296 |
|   'Ġlos' |    Function word |            -6.53 |           -18.76 | +12.23 |     0.0178 |
|     'Ġy' | Char/phonotactic |            -6.14 |           -13.81 |  +7.67 |     0.0165 |
|   'Ġdel' |    Function word |            -6.83 |           -18.77 | +11.94 |     0.0129 |
|   'Ġlas' |    Function word |            -7.37 |           -19.03 | +11.66 |     0.0073 |
|    'Ġla' |    Function word |            -6.42 |           -10.42 |  +4.00 |     0.0065 |
|  'ciÃ³n' |     Morph. affix |            -7.09 |           -14.46 |  +7.37 |     0.0061 |
|    ',Ġy' |     Morph. affix |            -7.72 |           -19.25 | +11.53 |     0.0051 |
|   'Ġ,Ġy' |  Multi-word unit |            -7.88 |           -19.25 | +11.37 |     0.0043 |
|    'Ġen' |    Function word |            -6.19 |            -8.16 |  +1.97 |     0.0041 |
|    'Ġes' |    Function word |            -6.57 |            -9.33 |  +2.75 |     0.0038 |
| 'ĠenĠel' |  Multi-word unit |            -7.79 |           -17.06 |  +9.26 |     0.0038 |
|    'ÃŃa' |     Morph. affix |            -7.22 |           -12.39 |  +5.16 |     0.0038 |
|    'Ġun' |    Function word |            -7.03 |           -11.11 |  +4.08 |     0.0036 |
| 'ĠenĠla' |  Multi-word unit |            -7.86 |           -17.04 |  +9.19 |     0.0036 |
|   'Ġuna' |     Content word |            -7.22 |           -11.98 |  +4.76 |     0.0035 |
|    'Ã©n' |     Morph. affix |            -8.00 |           -18.22 | +10.22 |     0.0034 |
|    'dad' |     Morph. affix |            -7.55 |           -14.09 |  +6.54 |     0.0034 |
| 'ĠdeĠla' |  Multi-word unit |            -7.30 |           -12.28 |  +4.98 |     0.0034 |
|    'Ġha' | Char/phonotactic |            -7.28 |           -11.88 |  +4.60 |     0.0032 |

### Classified tokens: Slovak / Czech (slk_Latn vs ces_Latn)

|     Token |         Category | Score (slk_Latn) | Score (ces_Latn) |  Delta | KL contrib |
|-----------|------------------|------------------|------------------|--------|------------|
|   'ĠprÃŃ' |     Content word |            -7.20 |           -19.19 | +11.99 |     0.0089 |
|  ',Ġktor' |     Content word |            -7.24 |           -19.19 | +11.95 |     0.0086 |
|      'Ä¾' |     Morph. affix |            -6.86 |           -15.04 |  +8.18 |     0.0085 |
| 'ĠaleboĠ' |  Multi-word unit |            -7.39 |           -19.19 | +11.80 |     0.0073 |
|     'Ġsa' | Char/phonotactic |            -6.69 |           -12.20 |  +5.51 |     0.0068 |
|     'jÃº' |     Morph. affix |            -6.94 |           -12.81 |  +5.87 |     0.0057 |
|     'nie' |     Morph. affix |            -6.74 |           -11.46 |  +4.72 |     0.0056 |
|    'ĠsÃº' |     Content word |            -7.30 |           -15.42 |  +8.13 |     0.0055 |
|     'nia' |     Morph. affix |            -7.18 |           -13.69 |  +6.51 |     0.0050 |
|   'Ġpred' |     Content word |            -7.70 |           -18.47 | +10.77 |     0.0049 |
|      'Å¥' |     Morph. affix |            -6.20 |            -8.54 |  +2.34 |     0.0048 |
|    'Ġpre' |     Content word |            -7.03 |           -11.37 |  +4.34 |     0.0038 |
|     'Ä¾a' |     Morph. affix |            -7.65 |           -15.54 |  +7.89 |     0.0038 |
|    'Ġako' |     Content word |            -8.10 |           -19.09 | +10.99 |     0.0033 |
|     'nÃº' |     Morph. affix |            -7.94 |           -17.22 |  +9.28 |     0.0033 |
|  'Ġnaria' |     Content word |            -8.14 |           -19.19 | +11.05 |     0.0032 |
|    'polo' |     Morph. affix |            -8.17 |           -18.40 | +10.23 |     0.0029 |
|     'nej' |     Morph. affix |            -7.36 |           -11.50 |  +4.13 |     0.0026 |
|      'om' | Char/phonotactic |            -7.31 |           -11.19 |  +3.88 |     0.0026 |
|    'Ġpri' |     Content word |            -7.47 |           -11.94 |  +4.46 |     0.0025 |
