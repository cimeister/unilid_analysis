# CommonLID (web-domain) check of the top carried configurations (Exp 39)

373,230 lines; macro-aware accuracy convention of Exp 12 (prediction correct if its iso code, or its macrolanguage, matches the gold tag). The adaptive gate runs with its training-calibrated tau values unchanged (no refitting on this domain).

| config | macro-aware accuracy | delta | tag-level macro-F1 | delta |
|---|---|---|---|---|
| baseline | 0.8452 |  | 0.7228 |  |
| floor21 | 0.8491 | +0.0040 | 0.7181 | -0.0046 |
| gt_margin_adaptive | 0.8521 | +0.0070 | 0.7167 | -0.0061 |

Reference points from Exp 12: frequency prior +0.0067, learned bias +0.0427.
Adaptive-gate activity on this domain: 9,886 reassignments; 2,268 below-tau lines kept for lack of a top-resource candidate in the top-5.
