# Session Status

## Ongoing experiments
- Exp 46 (mixed matrix, rule v1) running as job 2932154: no-op scorer gate, then full-pool scoring, gate stage, and solotau variant, fail-fast; ~2.5 h once scheduled; monitor armed
- On completion: `analysis/mixed_eval.py` with configs=("mixed",) delivers the judge-part verdict (anchor floor21_gate, gt_margin_adaptive alongside), the mixed-minus-solo decomposition, and the floor-gap-shift correlation test
- floor21_gate stands as the strongest configuration on the judging instrument (judge part 0.9480, +0.0151 [+0.0112, +0.0191] over gt_margin_adaptive), in the pool, not promotable under the current clause (A)

## Open decisions
- Clause-(A) promotion cap (user deferred 2026-07-30): decide with the Exp 46 results in hand
