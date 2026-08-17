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

- **Job 3107045, corrected-model full-pool baseline** (`slurm_full_test_eval_corrected.sh`):
  queued. `--configs baseline` only, fresh scratch root
  `full_test_eval_corrected/`, tables to `outputs_corrected/`. 3 of 92 chunks
  already banked, resumable. This is B3 step 1 and everything downstream needs it.
- **Job 3107082, corrected-model c sweep** (`slurm_floor_sweep_corrected.sh`):
  queued. Published Exp 20 protocol on the grid shifted by log 5. Expected to
  land near -19.3906; **a result far from that is a finding, not a tuning
  problem.** This is B2's first step and gates the floor-21 pass.

## Next work, in order

- B0: `finished` 2026-08-17, PASS 3/3. The unseen-token plateau is set by corpus
  size, scaling as `T^-0.95`, and the paper's appendix sentence can now be
  rewritten with a causal statement.
- B1: `finished` 2026-08-17. Single resolver (`analysis/model_context.py`), eleven
  scripts wired, two missing generators written, 13/13 guard cases verified by
  triggering.
- B2 remainder (`waiting on` job 3107082): all 1,084 group-A thresholds via
  `solo_gates.py floor21`, then the high-entropy group re-identified.
- B3 remainder (`waiting on` job 3107045): `full_test_floor21.py`, then
  `solo_gates.py`, then `gate_variants.py`, then `build_release_calibration.py`;
  fresh gate references; `paper_eval.py` and the breakdowns; the Mistral-Nemo
  chain, which is independent and can run in parallel.
- B4 (`waiting on dependency`): ship, retire polybox.

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
