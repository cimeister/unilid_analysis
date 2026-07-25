# Carried-forward set under the primary quantity (Exp 38)

Per-language F1 on the held-out remainder (45,004,014 lines, natural distribution, all false positives counted), averaged unweighted over languages.

| config | all 1,940 | tail (N<1k) | lowmid (1k-18k) | head (N>=18k) | flat_magnet | twin |
|---|---|---|---|---|---|---|
| gt_margin_adaptive | 0.9334 | 0.4620 | 0.9567 | 0.9593 | 0.4206 | 0.8917 |
| floor21 | 0.9309 | 0.6337 | 0.9352 | 0.9590 | 0.5345 | 0.8854 |
| freq_prior | 0.9264 | 0.4816 | 0.9399 | 0.9605 | 0.4168 | 0.8892 |
| learned_bias | 0.9254 | 0.3736 | 0.9405 | 0.9696 | 0.3562 | 0.9079 |
| margin_q5_head | 0.9215 | 0.5321 | 0.9267 | 0.9590 | 0.4185 | 0.8856 |
| margin_q5 | 0.9201 | 0.5125 | 0.9256 | 0.9592 | 0.4038 | 0.8847 |
| baseline | 0.9121 | 0.3382 | 0.9267 | 0.9593 | 0.2890 | 0.8856 |

Per-language leaders. 326 of 1940 languages have ties at the maximum; strict counts exclude them (a config is counted only where it is the UNIQUE best):
baseline: 10 strict, freq_prior: 211 strict, learned_bias: 602 strict, floor21: 173 strict, margin_q5: 10 strict, margin_q5_head: 21 strict, gt_margin_adaptive: 587 strict
