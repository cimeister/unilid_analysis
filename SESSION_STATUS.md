# Session Status

## Ongoing experiments
- E3: baseline+calibval RUNNING as job 3032625 (watcher active). Pipeline committed after the full 16-fix verification. Remaining chain on completion: flatrule (login), tau (SLURM), topk (SLURM), eval (login), each fingerprint-gated.
- E1, E2, E4, E5 finished; paper carries the calibrated row, sec:calibration, app:protocol, tab:commonlid.

## Open decisions
- Paper trimming items 2-5: IMPLEMENTED, UNCOMMITTED, awaiting the user's diff review (git diff paper/submission.tex; ~290 words + two float relocations; protected regions byte-identical to HEAD). Items 9-12 in reserve if the compiled page count still runs over.
- Right-side Table 1 columns: fastText subset cell (.990) and subset FPR cells reproduce under no tested convention. Needs Ahmetcan's eval script or an author decision.
- UDHR regression framing: worth a co-author read.
