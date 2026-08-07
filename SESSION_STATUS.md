# Session Status

## Ongoing experiments
- E3 training RUNNING as job 3028465 (pending in queue; watcher active). After completion: degeneracy scan (in-job), variant baseline scoring, floor-21 matrix, tau recalibration, flat set via the recorded rule, gate, eval.
- E5 (CommonLID for the camera-ready, user request 2026-08-07): pre-registered; implementation agent writing analysis/commonlid_calibrated.py + SLURM script (score under W_f21 banking top-5, gate via the self-checked E2 helpers, macrolanguage-aware metrics; wiring gates on the recorded 0.8452/0.7228/0.7181). Then Opus review, SLURM run, record, paper integration. CommonLID data re-touched against the purge.
- Paper: Table 1 calibrated row + items 2-5 integrated; right-side subset cells dashed pending convention alignment.

## Open decisions
- Right-side Table 1 columns: submission's fastText subset cell (.990) and subset FPR cells reproduce under no tested convention. Needs Ahmetcan's eval script or an author decision.
- UDHR regression framing: worth a co-author read.
- E5 paper placement: a small table in the appendix protocol section plus one main-text sentence is the default; no fastText row is possible (no fastText model binary for CommonLID).
