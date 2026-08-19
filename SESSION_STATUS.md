# Session Status

Snapshot, 2026-08-19. The re-release of the special-token-corrected model is in
execution. The paper is blocked on regenerating its UniLID numbers; the
open-source package fix is done and sits in PR #3.

## Where the authoritative records are

- `paper/PAPER_EDITS_pending.md`: the concrete, itemized paper edit list. Read
  this first for anything paper-related.
- `paper/appendix_revision_draft_2026-08-17.md`: proposed wording for the edits
  that are prose rewrites rather than number substitutions. Awaiting sign-off.
- `RERELEASE_PLAN.md`: execution plan, gates, dependency order.
- `EXPERIMENTS_PLAN.md`, "Special-token correction and re-release": status per
  work item, in the file's own status vocabulary.
- `EXPERIMENTS_RESULTS.md`: the 2026-08-17 to 2026-08-19 entries at the top.
- `CODE_CHANGES_2026-08-17.md`: every code change, for review in one place.
- `OPEN_SOURCE_STATUS.md`: package state, PR #3, the Hub and polybox plan.

## Running

| job | what | note |
|---|---|---|
| 3123324 | group-A thresholds, all 1,084 (`solo_gates floor21`) | reads c from the fingerprint |
| 3127704 | `gate_variants topk`, full-pool candidate banking | parallel with 3123324, not after: it needs no tau CSV |
| 3112846 | Qwen3-8B retrain, patched fp64 trainer | ~5-6h once started |
| 3112879 | DeepSeek3.2 retrain, patched fp64 trainer | ~5-6h once started |
| 3117576 | Qwen3 full-pool eval | chained `afterok:3112846` |
| 3117575 | DeepSeek3.2 full-pool eval | chained `afterok:3112879` |

## Settled

- **c = -17** for the corrected model (job 3117581), round grid {-15,-17,-19,-21}
  chosen by the rule the published grid follows. The pre-registration was exact:
  predicted clamp counts 317 / 1,655 / 1,940 / 1,940 and the selection of -17, all
  correct. 1,655 rows clamped, 285 already below.
- **Corrected base, full pool**: macro F1 0.9292 to 0.9327, macro FPR 2.03e-5 to
  2.02e-5.
- **Corrected + clamp, full pool**: overall 0.9419, against the released model's
  0.9421 at c = -21. The two clamped models land 0.0002 apart.
- **Tail, both views** (96 languages, N_L < 1,000): within-stratum 0.8875 against
  the released clamped 0.8928; global per-language F1 **0.7743 against 0.7655**,
  with false positives into tail labels down to 8,727 from 22,522 at the released
  baseline. The views disagree by construction (Exp 24). **Priorities unchanged
  by author decision 2026-08-19: c stays selected on validation overall macro F1
  under the all-strata guard.**
- **Three tables regenerated**: `viterbi_vs_marginal` (.961/.933 against
  .961/.935; the paper's "+0.002 from marginalization" survives),
  `lenbias-norm` (0.961 to 0.838 under normalization, a larger drop than the
  published 0.960 to 0.885), Mistral-Nemo GlotLID-C cells (0.9119 / 1.858e-5,
  unchanged to three decimals).
- **B0**: the unseen-token plateau is set by corpus size, scaling as `T^-0.95`.
  Independently reproduced on the DeepSeek and Qwen vocabularies (slopes -2.016
  and -2.010 against -2.068 for the base model).
- **B1**: the analysis chain is safe to point at a second model.
  `analysis/model_context.py` is the single resolution point; 13/13 guard cases
  verified by triggering.

## Open decisions

- **The two variant models carry a second, pre-existing defect.** Both were built
  2026-03-27, four months before the fp64 EM bug was fixed, and both have a
  corrupted `azj_Latn` row (Qwen3: plateau at the training floor, 20.1 sd below
  expectation; DeepSeek3.2: retrain differs by +1.0002 nats at correlation 0.7057
  against a zul_Latn control at 1.00000000). Both are being retrained. **`mya_Mymr`
  in the Qwen3 model is unresolved**: Burmese script coverage against a second EM
  casualty. The retrain settles it, since coverage survives a retrain and
  corruption does not.
- **The base model's `azj_Latn`, `bod_Tibt` and `mya_Mymr` exceed the correction
  gate's thresholds**, but by 93x less than the variants and only 1.08x the
  threshold for `azj_Latn`. Read as threshold calibration on hard languages
  (capped corpora, minority scripts) rather than corruption, consistent with zero
  plateau outliers. Worth a closer look; not the same phenomenon.
- **`tab:lenbias-delta` instrument**: the corrected predictions exclude the
  250,000 validation lines, while the published table used all 45,627,279. Small
  change of instrument, needs an author call.
- **Group B (high-entropy) membership** is read from
  `outputs/diagnostic/lang_diagnostic.csv`, computed on the released model.
  Re-identifying it on corrected predictions needs that diagnostic regenerated;
  `build_release_calibration.py` asserts the current four and will abort until
  updated.
- **WiLI and DSL-ML artifacts** (six tables) are not on this machine and were not
  in the co-author's Drive folder.
- **Latency discrepancy to raise with the co-author**: the Drive `full_prob` run
  reports 1,075 samples/s against `tab:latency_glotlid`'s 3,253 for the same
  45,627,279 samples. Probably a full-probability-output run; the latency table's
  configuration remains an open ask.
- Whether the Apertus 200k and 131k variants are published or only corrected
  locally. They appear in no paper table.
- Whether the package offers users a migration for their own pre-0.3.0 models.

## Blocked, in dependency order

`gate_variants apply` (needs 3123324 and 3127704) then
`build_release_calibration.py`, fresh gate references, `release_gates.py`,
`paper_eval.py` and the breakdowns, the CommonLID chain, and Mistral-Nemo's
stages after `baseline` (all of which clamp, so they need the constant threaded
the way `solo_gates` and `gate_variants` now do it).

## Carried over: camera-ready items

- Edit pass applied 2026-08-09, every text edit wrapped in `\camrev{}`. User to
  review the red text; dispositions in `paper/review_notes_2026-08-09.md`.
- Ahmetcan ask list: subset-evaluation script/command; UDHR-subset UniLID FPR
  1.06e-5 confirmation; fastText WiLI config; DSL-ML competitor-score source and
  split; CommonLID citation; latency run configs.
- The user compiles the PDF (no icml2026.sty in this repo).
- Uncommitted in the working tree and not mine to touch: an edit to
  `paper/submission.tex` removing the abstract's calibration sentence, and an
  untracked `paper/initial_version.tex`.
