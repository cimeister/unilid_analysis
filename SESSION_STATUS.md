# Session Status

## Ongoing experiments
- None running. E1 through E5 all FINISHED and recorded (EXPERIMENTS_RESULTS.md: "Camera-ready E1" through "Camera-ready E5", including E3's transfer result: nemo_baseline 0.9132 -> nemo_gated 0.9538 full pool, judge bootstrap +0.0504 [+0.0438, +0.0573]).

## Pending actions
- Store migration of the E3 artifacts of record (model, memmaps, banked arrays, per-language tokenizers) from scratch to /capstor/store/cscs/swissai/a0229/cmeister with scratch symlinks; purge deadline about 2026-08-22; login-node chunked, no SLURM for transfers.

## Open decisions
- Mistral-Nemo variant in the paper: recommendation = appendix comparison (our retrain, clearly marked; existing Table 1 row untouched); numbers now available; awaiting user confirmation before any tex edit.
- Subset FPR cells + the printed fastText GlotLID-C subset cell (.990): needs Ahmetcan's eval script or an author decision.
- UDHR regression framing and the "calibrated UniLID" name: co-author read.
