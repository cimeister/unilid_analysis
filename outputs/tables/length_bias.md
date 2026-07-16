### Token count delta (pred - true) for UniLID misclassifications

|          Category |         N | Mean delta | Median delta | % fewer | % same | % more |
|-------------------|-----------|------------|--------------|---------|--------|--------|
| All misclassified | 1,789,423 |      -0.17 |         0.00 |   24.94 |  61.08 |  13.99 |
|               <30 |   515,094 |      -0.11 |         0.00 |   17.69 |  74.62 |   7.69 |
|            30--75 |   771,812 |      -0.15 |         0.00 |   24.28 |  62.76 |  12.96 |
|           75--150 |   392,549 |      -0.21 |         0.00 |   32.10 |  47.71 |  20.19 |
|          150--300 |   102,497 |      -0.24 |         0.00 |   36.80 |  34.85 |  28.34 |
|              300+ |     7,471 |      -2.71 |        -1.00 |   53.09 |  15.11 |  31.80 |

### Token delta by confusion cluster

|                Category |       N | Mean delta | Median delta | % fewer | % same | % more |
|-------------------------|---------|------------|--------------|---------|--------|--------|
|         Arabic Dialects | 116,877 |      -0.20 |         0.00 |   30.66 |  50.29 |  19.05 |
|       Chinese Varieties |  42,389 |      -0.01 |         0.00 |    3.81 |  93.10 |   3.09 |
| Hindi Belt (Devanagari) |  60,220 |      -0.06 |         0.00 |   11.78 |  81.60 |   6.62 |
|       Malay--Indonesian |  87,514 |      -0.11 |         0.00 |   20.28 |  68.46 |  11.27 |
|            Scandinavian | 129,158 |      -0.13 |         0.00 |   24.28 |  59.99 |  15.73 |
|                  Hebrew |  19,666 |      -0.18 |         0.00 |   38.94 |  28.56 |  32.50 |
|        Persian--Iranian |  66,958 |      -0.22 |         0.00 |   28.70 |  61.59 |   9.72 |

### Statistical tests for systematic bias

|                                   Test |        Statistic |  p-value |
|----------------------------------------|------------------|----------|
| One-sample t-test (H0: mean delta = 0) |        -122.4824 | 0.00e+00 |
|     Wilcoxon signed-rank (excl. zeros) | 84666726461.0000 | 0.00e+00 |
|                              Cohen's d |          -0.0916 |          |

### Length normalization counterfactual: misclassifications corrected by normalizing scores by token count

|          Category |         N | N corrected | % correctable |
|-------------------|-----------|-------------|---------------|
| All misclassified | 1,789,423 |     332,034 |          18.6 |
|               <30 |   515,094 |      78,796 |          15.3 |
|            30--75 |   771,812 |     147,501 |          19.1 |
|           75--150 |   392,549 |      84,024 |          21.4 |
|          150--300 |   102,497 |      19,845 |          19.4 |
|              300+ |     7,471 |       1,868 |          25.0 |

### Length normalization counterfactual by confusion cluster

|                Category |       N | N corrected | % correctable |
|-------------------------|---------|-------------|---------------|
|         Arabic Dialects | 116,877 |      21,849 |          18.7 |
|       Chinese Varieties |  42,389 |       1,359 |           3.2 |
| Hindi Belt (Devanagari) |  60,220 |       6,193 |          10.3 |
|       Malay--Indonesian |  87,514 |      16,603 |          19.0 |
|            Scandinavian | 129,158 |      25,171 |          19.5 |
|                  Hebrew |  19,666 |         806 |           4.1 |
|        Persian--Iranian |  66,958 |      17,067 |          25.5 |

### Length normalization counterfactual by token delta

| Category |         N | N corrected | % correctable |
|----------|-----------|-------------|---------------|
|      <-2 |   112,762 |      82,765 |          73.4 |
| -2 to -1 |   333,452 |     249,267 |          74.8 |
|       -1 | 1,092,902 |           2 |           0.0 |
|        0 | 1,092,902 |           2 |           0.0 |
|       +1 |   197,984 |           0 |           0.0 |
| +1 to +2 |    37,879 |           0 |           0.0 |
|      >+2 |    14,444 |           0 |           0.0 |
