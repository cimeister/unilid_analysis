# Session Status

## Ongoing experiments
- E3: training done (job 3028465, degeneracy adjudicated); the six-stage evaluation pipeline (analysis/mistralnemo_eval.py + three SLURM scripts) is implemented. The 16-item Opus review-fix checklist is now fully applied to mistralnemo_eval.py (py_compile, import, and em-dash checks pass); not yet re-reviewed or run. Next: re-review, then submit slurm_mistralnemo_baseline.sh (baseline + calibval), then flatrule (login), tau, topk, eval. The flat set is derived by the recorded rule on the variant's own rows and calibval predictions.
- E1, E2, E4, E5 finished and integrated into the paper (calibrated Table 1 row, sec:calibration paragraph, app:protocol, tab:commonlid).

## Open decisions
- Paper trimming: ranked report delivered in chat (recommended items 1-8, ~112 column-lines; reserves 9-12; caveats: verify the ICML impact-statement placement rule; estimates uncompiled). Awaiting the user's selection; nothing implemented.
- Right-side Table 1 columns: submission's fastText subset cell (.990) and subset FPR cells reproduce under no tested convention. Needs Ahmetcan's eval script or an author decision.
- UDHR regression framing: worth a co-author read.
