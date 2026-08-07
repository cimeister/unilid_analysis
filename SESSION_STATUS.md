# Session Status

## Ongoing experiments
- Camera-ready program: E1 FINISHED (full pool: baseline 0.9292, promoted 0.9569, fastText 0.9443; judge bootstrap vs fastText +0.0166 [+0.0112, +0.0223]). E4 FINISHED (both-views breakdown tables; reproduction gates pass under the within-stratum view; residual of record 926,299 / 99.15% / 88.64%, eng->sco pair eliminated). E2 machinery verified by self-check (bit-exact replay of the promoted configuration); scoring awaits mapping approval, SLURM script ready, not submitted. E3 not started (sequenced last).

## Open decisions
- E2 mapping approval: outputs/tables/external_bench_mapping.md (+ addendum). Recommendation: score udhr_eval.tsv (366 labels) and flores200_eval.tsv (190 labels), both exact-match, no remapping decisions needed.
- Paper prose sample (paper/draft_new_sections.tex): awaiting user OK before any edit to submission.tex; working name "calibrated UniLID" proposed there.
- Ahmetcan ask list: item 3's script-table-basis question is now resolved internally (within-stratum view; Other excludes jpn_Jpan/kor_Hang) and can be dropped from the message if not yet sent.
