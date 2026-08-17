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

- None running. All background jobs from 2026-08-17 completed; measurements are
  in `outputs/rerelease/`.

## Next work, in order

- B0: `finished` 2026-08-17, PASS 3/3. The unseen-token plateau is set by corpus
  size, scaling as `T^-0.95`, and the paper's appendix sentence can now be
  rewritten with a causal statement.
- B1 (`ongoing`, gates all scoring): eight scripts still need a model parameter
  and a fresh output root, and two paper tables (`viterbi_vs_marginal`,
  `lenbias-norm`) have no reproducible generator at all. Table in
  `RERELEASE_PLAN.md` step 2.
- B2, B3, B4 (`waiting on dependency`): re-derive c, all 1,084 thresholds and the
  high-entropy group; regenerate the paper numbers; ship.

## Open decisions

- Whether the `learned_bias` config is still reported: its `learned_bias.npy` was
  fit to the old model, and keeping it means paying for a refit and a third full
  pass per `full_test_eval.py` invocation.
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
