# Session Status

## Ongoing experiments
- E5 FINISHED (job 3031609; gated accuracy 0.8604 vs baseline 0.8452, tag F1 0.7149 vs 0.7228; all gates exact; integrated into the paper as tab:commonlid). E3 training FINISHED (job 3028465, 4h33m; degeneracy scan: 32 minority-script rows, adjudicated as base-vocab script coverage, same pattern as the Apertus models; not a trainer failure). E3 next stages pending implementation: variant baseline full-pool scoring, floor-21 matrix + fingerprint, tau recalibration, flat set via the recorded rule (250k-val pass), floor-21 pass with top-5 banking, gate, eval.
- Paper-trimming agent resumed after the session restart; ranked report (no implementation) pending.

## Open decisions
- Right-side Table 1 columns: submission's fastText subset cell (.990) and subset FPR cells reproduce under no tested convention. Needs Ahmetcan's eval script or an author decision.
- UDHR regression framing: worth a co-author read.
