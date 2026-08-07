# Session Status

## Ongoing experiments
- Camera-ready program: E1 and E4 FINISHED. E2 scoring RUNNING as job 3028291 (watcher active; eval + acceptance gates follow; label sets independently confirmed by Ahmetcan's lists). E3 preparation running (agent: pin tokenizer snapshot, map retrain pipeline, write drivers; no launch before review). Paper items 2, 3, 5 + appendix integrated into submission.tex; Table 1 row waits on E2 cells; CLD3-subset F1 cells computed under the reproduced convention (baseline 0.9719 = printed .971; calibrated 0.9751; fastText 0.9767).

## Open decisions
- CLD3-subset column inconsistency in the submission: the printed fastText cell (.990) and the FPR cells are reproduced by no tested convention, while the UniLID cells are (restricted-lines open-set). Needs the paper team's eval script or an author decision on which convention the camera-ready right-side columns use.
- Remaining Ahmetcan asks: the eval script (conventions) and the Mistral-Nemo model (E3 retrain proceeds meanwhile, pre-authorized).
