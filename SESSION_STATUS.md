# Session Status

## Ongoing experiments
- Camera-ready program: E1, E2, E4 FINISHED. E3 training RUNNING as job 3028465 (recorded Apertus pipeline reused exactly; Mistral-Nemo tokenizer pinned at snapshot a4477a2; expected ~4.6 h + queue; watcher active). After it completes: degeneracy scan (in-job), then the variant's baseline scoring, floor-21 matrix, tau recalibration, flat set via the recorded rule (250k-val pass), gate, eval.
- Paper: Table 1 calibrated row + abstract + items 2-5 integrated; right-side subset cells dashed pending convention alignment.

## Open decisions
- Right-side Table 1 columns: submission's fastText subset cell (.990) and subset FPR cells reproduce under no tested convention (UniLID cells do). Needs Ahmetcan's eval script or an author decision.
- UDHR regression framing (0.859 -> 0.838 for the calibrated row): stated in subsection/appendix/caption; worth a co-author read.
- degeneracy_scan.py's stale MODELS dict (pre-fp64 paths): separate fix, not blocking.
