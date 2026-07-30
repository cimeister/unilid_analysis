# Session Status

## Ongoing experiments
- floor21_gate judge-part evaluation done (`outputs/tables/mixed_eval_floor21_gate_unmod_gate.md`): +0.0151 [+0.0112, +0.0191] over gt_margin_adaptive, tail 0.7306, flat_magnet 0.6435, FPs into tail labels 1,912 against baseline 13,483, clause (C) clean; clause (A) fails (balanced-val tail drop 0.0426 and magnets 0.0383 against the 0.03 cap), so under amendment 7 it is in the pool, not promotable in current form
- Rule v1 materialized (`outputs/diagnostic/mixed_assignments.csv`: 860 head languages unmodified ungated, 1,080 on floor-21 plus gate; 555 non-zero floor-gap shifts, all negative, 549 in lowmid); mixed-matrix implementation and scoring wait on the rule sign-off

## Open decisions
- Step-2 checkpoint (asked): rule v1 sign-off and whether to score the mixed matrix; how promotion clause (A) should treat floor21_gate's recall-view cost; amendment-scope confirmation (judge part as confirmation instrument for derivation-informed candidates)
