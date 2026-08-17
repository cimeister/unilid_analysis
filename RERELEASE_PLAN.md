# Re-releasing the 1,940-language model with the special-token correction

Written 2026-08-17, after the special-token defect was found and fixed in the
package (UNILID 0.3.0, folded into PR #3). Revised the same day after an
adversarial review that returned five blocking findings, two of which would have
produced numbers that looked valid and answered a different question. Findings
marked [R] are that review's, each verified here before being folded in.

Status: APPROVED and IN EXECUTION.

## Decisions taken by the author (2026-08-17)

1. Re-release goes ahead. Every released model's weights are corrected and
   re-released, not just the calibrated one.
2. The paper is updated IN PLACE, not issued as an erratum.
3. The experiments that can be run on this machine are re-run here. The WiLI and
   DSL-ML evaluations stay with the co-author.
4. The polybox mirror of the original uncalibrated model is taken down.
   HuggingFace Hub carries both the calibrated and the uncalibrated model.

## What is wrong with the released weights

Per-language training gave each of the four special tokens a pre-normalization
log-probability of 0.0, i.e. probability 1.0. After normalization each holds
exactly 1/5, so every real token is a factor of five too small. All four stored
GlotLID-C-scale models carry 0.800000 on the special tokens, measured over every
row: `glotlidc.unilid`, `glotlid_apertus200k_fp64.unilid`,
`glotlid_apertus131k_fp64.unilid`, `glotlid_mistralnemo_fp64.unilid`.

The mass is unusable. No special token's stored weight is read when scoring: the
scorer takes its unknown-token score from a model-wide constant
(`min_score - K_UNK_PENALTY`, `model.rs:246,346`), and `<s>`/`</s>`/`<pad>` are
reachable only by text containing those literal substrings. Setting all four
entries of every row to -500 changes predicted scores by exactly 0.000000.

## The corrected artifact is a transformation, not a retrain

    real tokens += log(5) = 1.6094379 ; special tokens := log(1e-12)

[R] The reason this is exact is the code diff, not any property of SentencePiece.
0.3.0 changes only the special-token branch and the final normalization; the real
token values entering the normalization come from identical code in both
versions, so `renormalize_over_real_tokens(released_row)` is exactly what the
fixed trainer computes from the same inputs. My original justification, that
SentencePiece emits normalized log-probabilities so the real tokens sum to one,
is not sufficient: the missing-token fallback assigns the base tokenizer's
log-prob to base-vocab tokens the sp model dropped, and that mass sits outside
SentencePiece's normalization. With such mass M the correction would be
log((5+M)/(1+M)), not log 5.

[R] What rules that out is measurement, and it is both cheaper and stronger than
retraining: over all 1,940 rows of each of the four models the real-token mass is
0.2 to within 8.4e-8, so the exact per-row correction `-log(real_mass)` is within
4.2e-7 nats of log 5 everywhere, and no row contains a
`MISSING_TOKEN_FILL_LOG_PROB` entry. This runs in seconds per model.

### Gates for step 1 (revised)

- **Gate A (replaces the retrain gate).** Over every row of every model to be
  shipped: real-token mass within 1e-6 of the model's median, no
  `MISSING_TOKEN_FILL_LOG_PROB` entries, corrected real-token mass 1.0 to float32
  tolerance, special-token mass below 1e-9.
- **Gate B.** The corrected argmin set equals the original argmin set per row,
  because `apply_unseen_token_constant` clamps by exact float32 equality against
  `row.min()`; if the transformation moved which entries are the minimum, the
  clamp would select a different plateau.
- **Gate C.** Pin one implementation and record the output sha256. [R] Two
  defensible implementations are not bit-identical: `float32(released + log5)`
  and `renormalize_over_real_tokens(released)` differ on 1,608 of 193,992,240
  entries, maximum 4.77e-7 nats. Both release gates use exact prediction
  equality, so the shipped bytes have to be reproducible from a named procedure.
- The retrain comparison stays as corroboration, not as the gate. It is also
  confounded: languages above the 100,000-line cap were subsampled and the corpus
  on store is the Apertus run's draw, so `zul_Latn` differs from the released row
  with signed mean -1.0e-3 and a maximum of 2.04 confined to tokens below p=1e-5
  carrying 1.6% of the mass, while two retrains of one corpus are bit-identical.

## What the correction does and does not change

[R] A uniform per-token constant does not cancel, confirmed in the scoring code.
`PreparedDP` is built from the trie and token ids only, so the candidate edge set
is identical across languages and independent of the weights; `score_only_f32` is
a max-plus DP, so a path with k real-token edges gains exactly k*c and paths with
different real-edge counts are reordered. Even where no segmentation changes, a
language whose best path uses k_L real edges gains k_L * log 5, so cross-language
comparison shifts wherever two languages produce different token counts.

[R] A third channel: `unk_score` is not drawn from the weights and is unchanged
by the correction, so real-token edges gain 1.609 nats against the unknown-token
edge. The gap between the lowest real token and that edge widens from 9.194 to
10.803 nats, so the unknown-token edge is selected on strictly fewer positions
for identical input. This is a behavioural change in its own right and belongs in
the appendix mechanism section.

Measured on the 250,000-line golden subset against the recorded gold labels, in
base mode (`analysis/correction_effect.py`,
`outputs/rerelease/correction_effect.json`):

| | released | corrected |
|---|---|---|
| macro F1 | 0.9454 | 0.9460 |
| macro FPR | 2.083e-05 | 2.081e-05 |
| accuracy | 0.9603 | 0.9604 |

1,807 of 250,000 predictions change (0.72%): 699 gold-correct gained, 669 lost.
The correction is a wash on quality. An earlier estimate of 0.9494 to 0.9509 with
63 gained and 32 lost was accuracy on a 20,000-line every-149th-line sample
rather than macro F1 on the golden subset, was not well enough powered, and is
superseded. The paper's numbers move in the third and fourth decimal, no claim
changes direction on this evidence, and the case for re-releasing is correctness
rather than a metric gain.

## Work

### 1. Produce and gate the corrected weights

DONE for the four stored models, written to
`/capstor/scratch/cscs/cmeister747/unilid_analysis/corrected/` by
`analysis/correct_special_token_mass.py`. Gates A to C above are still to be
implemented and run; the retrain corroboration is in `analysis/gate_correction.py`
and passes 8 of 8 languages.

### 2. Make the evaluation chain safe to point at a different model

[R] BLOCKING. Nothing may be scored until these are fixed. Each guard below was
verified by triggering it, not by reading it.

DONE so far: `full_test_eval.py` hashes the model into its resume fingerprint,
refuses to write a non-default model's results into the store-symlinked scratch
directory, and refuses to apply the released model's `learned_bias.npy` to a
different model. `length_bias.py` refuses to pair a non-default model with the
recorded prediction file. Remaining: `floor_equalization.run()`,
`solo_gates.run()`, `release_gates.py`, `build_release_calibration.py`,
`commonlid_calibrated.py`. Those four abort loudly against corrected weights
rather than producing wrong numbers, so they are lower risk than the two silent
paths above, but each still needs a model parameter before step 4.

- `analysis/full_test_eval.py:63-72` computes its resume fingerprint from the
  bias vectors, the language list, `CHUNK_LINES` and `TOTAL_LINES`. It does not
  hash the model or the weights, although its docstring claims to cover
  "everything that determines the memmap contents". The previous run completed,
  so pointing it at a corrected model would skip every chunk and recompute
  metrics from the old memmaps without an error. Add the weight-matrix sha256,
  following `analysis/mistralnemo_eval.py:461-469`, which already does this.
- `analysis/full_test_eval.py:38` writes to
  `/capstor/scratch/.../full_test_eval`, whose entries are symlinks into
  `/capstor/store/.../full_test_eval/`. That store directory holds `y_true.npy`,
  `pred_baseline.npy`, `pred_gate_flat4_prox21.npy` and
  `fingerprint_floor21.json`: the reference arrays for both gates and the
  provenance chain for every published GlotLID-C number. Redirect corrected-model
  runs to a fresh scratch root and make the store copies read-only for the
  duration. `analysis/solo_gates.py` writes fixed filenames too.
- [R] Other scripts resolve the model through
  `analysis.transfer_sweep.UNILID_MODEL_PATH` with no override and need code
  changes, not arguments: `floor_sweep.py`, `floor_equalization.py`,
  `solo_gates.py` (which also checks a sha256 against `fingerprint_floor21.json`,
  pinning the old weights), `release_gates.py`, `build_release_calibration.py`.
  `length_bias.py` takes a parameter no caller sets and reads its misclassified
  set from `glotlidc_y_pred.txt`, an old-model file with no consistency check, so
  a corrected model runs to completion and produces an internally inconsistent
  analysis. `commonlid_calibrated.py` asserts exact reproduction of
  `commonlid_carried_preds.npz` and four literal old-model constants, and aborts
  by design. `paper_breakdowns.py` and `regen_resource_tier_counts.py` score
  nothing and read the upstream outputs, so they run last.
- [R] `learned_bias.npy` was fit to the old model and is loaded unconditionally
  into the `learned_bias` config. Decide whether that config is still reported;
  refit it or drop it from `CONFIGS`.

### 3. Re-derive the calibration

- **Unseen-token constant c.** Measured: the released rows' unseen-token value
  runs -19.94 to -13.22 (median -17.66), the corrected -18.33 to -11.61 (median
  -16.05), the distribution shifted by 1.609. All 1,940 rows still sit above
  c = -21, so the constant still lowers every row, but the median distance it
  moves a row grows from 3.342 to 4.951 nats. [R] Holding c at -21 therefore
  penalizes unseen tokens harder relative to seen ones than before: the sweep is
  over a changed quantity, not merely a re-tuning. Clamping the corrected weights
  at -21 + log 5 = -19.3906 would reproduce the old clamped matrix plus a uniform
  log 5, since the clamp sets an absolute target.
- **Per-language thresholds tau. Measured: they cannot be carried or shifted, so
  all 1,084 must be re-estimated.** Probing six group-A languages spanning N_L 85
  to 17,989, with the released weights clamped at c = -21 and the corrected ones
  at -21 + log 5 so the clamped structures differ only by the uniform shift:
  tul_Latn 6.7418 to 5.8984 (-12.5%), bkv_Latn 12.0420 to 11.3768 (-5.5%),
  mpm_Latn 0.1501 to 0.0423 (-71.8%), cmo_Latn 0.0117 to 0.0262 (+123.3%), and
  two excluded under both. Mean change -0.40 nats, median -0.39, largest 0.84, in
  both directions. The own-won counts move too (1,954 to 1,942; 1,485 to 1,383),
  so the correction changes which lines a language wins and not only the
  percentile of the margins it wins by. This is the segmentation-length channel
  showing up directly in the quantity the thresholds are percentiles of.
  Contrast with c, which moves by exactly log 5 and can be carried across by
  addition. Re-estimate with the recipe of record. [R]
  Group A membership cannot change: `calibration.runtime_for` requires it to
  equal the languages with N_L < head_n, and N_L is a data property, so only the
  tau values move. Note in the protocol that the corpora on store are the Apertus
  draw, so languages above the 100,000-line cap are estimated on a different
  sample than the original thresholds were.
- **High-entropy group.** Re-run the identification. [R]
  `build_release_calibration.py:64` hardcodes
  `EXPECTED_GROUP_B_LANGS = {"sco_Latn","bjn_Latn","arg_Latn","vls_Latn"}` and
  cross-checks it, so it refuses to run until updated; the constants at `:48-59`
  are literals too.
- [R] The calibration JSON's provenance block pins the old artifacts
  (`base_weight_matrix_sha256`, `clamped_weight_matrix_sha256`, `langs_sha256`,
  two `tau_csv_sha256`). All must be regenerated.

### 4. Re-run the evaluations

[R] The release gates cannot serve as a smoke test. They compare against
`pred_baseline.npy` (exact equality) and `pred_gate_flat4_prox21.npy` (>= 0.999),
both recorded from the old weights; a correction whose purpose is to change
predictions fails the first by definition and, at 0.72% changed, the second too.
Freeze the current gate results under new filenames as the old artifact's record,
generate new references from the corrected model, and state in
`OPEN_SOURCE_STATUS.md` that the two generations are not comparable.

Compute, corrected. [R] My 6-hour estimate was extrapolated from the release
gates and is about 2.7x the recorded figure:
`EXPERIMENTS_CHRONOLOGICAL.md:257` records job 3032625 scoring all 45.6M lines
plus 250,000 calibration lines in 2h14m, and job 3038358, a top-k pass, in 3h13m.
Three things the estimate missed: `full_test_eval.py:186-196` scores once per
entry of `CONFIGS = ["baseline","freq_prior","learned_bias"]`, so one invocation
is three full passes; the modes differ in per-line cost; and threshold
re-estimation is unbudgeted at roughly 2.2M lines of top-k scoring for 1,084
languages.

Which numbers move, by owner:

**Re-runnable here**, after step 2: `lid_main` (the UniLID, calibrated and
Mistral-Nemo rows), `calibrated_heldout`, `calibrated_views`,
`calibration_provenance`, `commonlid`, `calibrated_nemo`, `resource-tier`,
`script-breakdown`, `viterbi_vs_marginal`, `lenbias-delta`, `lenbias-norm`.

**Needs the co-author's artifacts**: `unilid_llm_comparison`, `noise_robustness`,
`length_accuracy`, `samples-accuracy`, `vocab_size_efficiency`,
`tatoeba_udhr_comparison`, and the UniLID column of `per_language_f1`.

**[R] In the paper, no artifact on this machine, no owner identified**:
`lid_main.tex:90` and `:98` carry `\unilid-DeepSeek3.2` and `\unilid-Qwen3`, 24
cells of GlotLID-C-trained UniLID numbers that I missed entirely. Resolve who
holds those two models before step 4 starts. Conversely, the Apertus 200k and
131k models on store appear in no table.

**Unaffected**: `latency_glotlid`, `training_time`, `dialect_stats`,
`fasttext_epoch_sweep`, `latency_wili`, and every fastText, GlotLID and CLD3 row.

### 5. Update the paper (in place)

Prose sites quoting UniLID numbers, to be replaced from the re-runs:
`submission.tex:344` (macro F1 .929 to .957), `:824` (FPR against fastText),
`:833`, `:835`, `:850-851` (UDHR and FLORES). The abstract's `\camrev` sentence
carrying the same figures is currently removed in the working tree; whether it
returns is an editorial choice.

[R] Two appendix items that are not number substitutions:

- `submission.tex:1383-1384` states that for the Mistral-Nemo variant "two
  languages whose trained unseen-token values already lie below c = -21 are left
  unchanged". Verified here: the released file has exactly two such rows,
  `khm_Khmr` (-21.232) and `ory_Orya` (-21.016); the corrected file has zero. The
  sentence reverses. Sweep the appendix for every count and every directional
  claim about unseen-token values.
- `submission.tex:629-631` attributes the above-c values to "the training-time
  probability floor of 10^{-12} and renormalization". [R] That is wrong: the sp
  path applies the floor only to special tokens, never to real ones, so the value
  is whatever SentencePiece assigns its low-count pieces. My earlier plan said to
  replace it by naming the special-token defect; that would also be wrong, since
  the defect accounts for 1.609 of an 11.58-nat gap. **Measure the actual origin
  before editing this sentence.** Replacing one wrong explanation with another is
  worse than leaving it.

### 6. Ship

- Publish the corrected calibrated and uncalibrated models on the Hub; retire the
  polybox mirror.
- [R] The container cannot distinguish a corrected file from an uncorrected one:
  the header encodes only version 1 against 2, and `FORMAT_VERSION_MAX = 2` means
  every published reader rejects a version-3 file, so a bump is not free. Add a
  loud load-time report of each row's real-token mass (0.2 for pre-0.3.0, 1.0 for
  corrected), which needs no format change.
- [R] `UNILID/README.md:110` and `UNILID/REPRODUCING.md:84` print 0.929 to 0.957,
  and `OPEN_SOURCE_DESIGN.md:355` repeats them. All are merged upstream and all
  need updating with the paper.
- Model card, and a REPRODUCING section mapping artifacts to results. [R] With
  the paper updated in place there is one paper version, so that section maps the
  old artifact to nothing and needs rewriting rather than extending.

## Order and gating

Step 2 before any scoring. Steps 1 and 3 before step 4. Nothing ships before the
paper numbers are regenerated.

## Open items

- Who holds the DeepSeek3.2 and Qwen3 models.
- Whether the Apertus variants are published or only corrected locally.
- Whether the package offers users a migration for their own pre-0.3.0 models.
- Where the rest of the unseen-token gap comes from, which step 5 needs.
