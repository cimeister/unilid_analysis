# Session Status

## Ongoing experiments
- E3: topk RUNNING as job 3038358 (the banked floor-21 pass; watcher was killed, re-check with squeue). When it completes: `python -m analysis.mistralnemo_eval --stage eval` on the login node produces the variant's baseline/floor-21/gated numbers (full pool + judge part + bootstrap). Chain so far: training done (3028465), baseline+calibval done (3032625), flat set = bjn/sco/srp (recorded), tau done (3037165: 1,080 + 3 thresholds), three startup failures each diagnosed as base-model-invariant gates misfiring on variant properties (floors below -21: khm/ory; special ids 0,1,2,10; json tuple round-trip) and fixed at commits 6b27f23 / a25cdb3 / the fingerprint-list fix.
- Everything else (E1, E2, E4, E5) finished and recorded.

## Paper state (all committed through c7ef48e)
- Calibrated UniLID fully integrated: abstract + intro sentence, sec:calibration paragraph (accuracy-reviewed after the user's edits; 5 introduced errors corrected), section-5 hyperparameters, Effect of Calibration in Results, Table 1 row (left cells + subset F1 cells .975/.986/.992; subset FPR dashed), app:protocol, tab:commonlid, both-views tables, trimming items 2-5.
- The paper fits the page limit exactly (user-compiled); no further trimming planned.

## Open decisions
- Mistral-Nemo variant presentation once E3 eval lands: recommendation = appendix comparison (our retrain, clearly marked), existing Table 1 row untouched; awaiting user confirmation.
- Subset FPR cells + the printed fastText GlotLID-C subset cell (.990, reproduces under no tested convention): needs Ahmetcan's eval script or an author decision.
- UDHR regression framing and the "calibrated UniLID" name: co-author read.
