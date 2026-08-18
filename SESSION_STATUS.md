# Session Status

Snapshot as of 2026-08-17. The paper is blocked on regenerating every UniLID
number against the corrected weights; the open-source package fix is done and in
PR #3.

## Where the authoritative records are

- `RERELEASE_PLAN.md`: the execution plan for the re-release, revised after an
  adversarial review. Author decisions, gates, dependency order.
- `EXPERIMENTS_PLAN.md`, section "Special-token correction and re-release
  (2026-08-17)": status of record for each work item, in the plan file's own
  status vocabulary.
- `EXPERIMENTS_RESULTS.md`: the 2026-08-17 entries at the top, plus the two
  subsections of "Invalidated / superseded results" added the same day.
- `EXPERIMENTAL_SETUP.md`: the defect, the corrected artifact and its gate, the
  clamp regression, the unseen-token mechanism, the probe protocols, and the
  release packaging step.
- `OPEN_SOURCE_STATUS.md`: package state, PR #3, the Hub and polybox plan.

## Ongoing experiments

All three queued as of 2026-08-18, none started (reason `Priority`; congestion is
normal here, do not cancel and resubmit).

- **Job 3110918, corrected floor-c full-pool pass** (`slurm_full_test_floor21_corrected.sh`):
  at the selected c = -17.3906. Gates the group-A thresholds and everything after
  them.
- **Job 3110925, corrected decoder comparison** (`slurm_viterbi_vs_marginal_corrected.sh`):
  `tab:viterbi_vs_marginal`. Base mode, so independent of c. Two full-pool passes
  per chunk; budget about three times a single baseline pass.
- **Job 3110926, corrected lenbias-norm** (`slurm_lenbias_norm_corrected.sh`):
  `tab:lenbias-norm`, alpha 0 and 1 over the 500k sample. Base mode, independent
  of c.

## Finished 2026-08-18

- **Job 3107045**, corrected full-pool baseline: overall macro F1 0.9292 to
  0.9327 (+0.0035), accuracy +0.0001, **tail -0.0087, magnets -0.0071**. The
  earlier "essentially a wash" was the golden-subset measurement and stays true
  there; it must not be quoted as covering the full pool.
- **Job 3107082**, c sweep: selected **c = -17.3906**. Aligned by grid position
  the released and corrected sweeps clamp identical row counts and every step is
  exactly log 5; positions 2 and 3 are tied in both models (0.0001 released,
  0.0002 corrected). The constant did not move, a tie broke the other way.

## Next work, in order

- **Blocked on job 3110918**: `solo_gates.py floor21` for all 1,084 group-A
  thresholds, then `gate_variants.py` (group B, then the gated predictions), then
  `build_release_calibration.py`, then fresh gate references, then `paper_eval.py`
  and the breakdowns.
- **The Mistral-Nemo chain** is independent and not started. Its `configure()`
  now takes `--model`, `--scratch-dir` and `--base-scratch`, but its stages after
  `baseline` still read `FLOOR_TARGET` as a module constant, so it needs the same
  `--floor-target` treatment before its clamped stages can run at the corrected
  model's own c.
- **Still importing `FLOOR_TARGET` as a module constant**, and needing the
  `--floor-target` treatment before a corrected-weights run: `gate_variants.py`,
  `commonlid_calibrated.py`, `external_bench_eval.py`, `mistralnemo_eval.py`,
  `mixed_assign.py`, `mixed_matrix.py`, `full_test_bgfloor.py`.
- **Paper**: `paper/appendix_revision_draft_2026-08-17.md` now has four items
  ready for sign-off (1, 1b, 1c, 1d) and two blocked. Nothing has been edited into
  `submission.tex`.

## Open decisions

- RESOLVED 2026-08-17: the `learned_bias` and `freq_prior` configs are not run
  for the corrected model. Neither appears in `submission.tex`, and one full pool
  pass each is a 3x cost. The Exp 14/16 records stay marked superseded pending
  regeneration.
- Whether the Apertus 200k and 131k variants are published or only corrected
  locally. They appear in no paper table.
- Whether the package offers users a migration for their own pre-0.3.0 models.
- Whether the Mistral-Nemo variant ships in a v1.1 release (carried over,
  unresolved; artifacts ready on store).
- Camera-ready items below, all carried over unresolved.

## Camera-ready items carried over

- Edit pass applied 2026-08-09, every text edit wrapped in `\camrev{}`. User to
  review the red text, especially the two published-row unbolds in Table 1, the
  Table 1 caption editorial note, and the C5 fixed-constants provenance sentence.
  Dispositions in `paper/review_notes_2026-08-09.md`.
- Ahmetcan ask list: subset-evaluation script/command (standing); UDHR-subset
  UniLID FPR 1.06e-5 confirmation; fastText WiLI config; DSL-ML competitor-score
  source and split; CommonLID citation; latency run configs.
- The user compiles the PDF (no icml2026.sty in this repo).
- Uncommitted in the working tree and not mine to touch: an edit to
  `paper/submission.tex` removing the abstract's calibration sentence, and an
  untracked `paper/initial_version.tex`.
