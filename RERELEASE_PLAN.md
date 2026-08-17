# Re-releasing the 1,940-language model with the special-token correction

Written 2026-08-17, after the special-token defect was found and fixed in the
package (UNILID commit 9f7c1cf, version 0.3.0, folded into PR #3). This plan
covers the released artifact and the paper, neither of which the code fix
touches.

Status: APPROVED and IN EXECUTION.

## Decisions taken by the author (2026-08-17)

1. Re-release goes ahead. Every released model's weights are corrected and
   re-released, not just the calibrated one.
2. The paper is updated IN PLACE, not issued as an erratum.
3. The experiments that can be run on this machine are re-run here. The WiLI and
   DSL-ML evaluations stay with the co-author, whose models are not on this
   filesystem.
4. The polybox mirror of the original uncalibrated model is taken down.
   HuggingFace Hub carries both the calibrated and the uncalibrated model.

## What is wrong with the released weights

Per-language training gave each of the four special tokens a pre-normalization
log-probability of 0.0, i.e. probability 1.0. After normalization each holds
exactly 1/5, so every real token is a factor of five too small. Measured on the
released file: all 1,940 rows carry 0.800000 on the special tokens and 0.200000
on the real ones, with no variation.

The mass is unusable. No special token's stored weight is read when scoring: the
scorer takes its unknown-token score from a single model-wide constant
(`min_score - K_UNK_PENALTY` in `model.rs`), and `<s>`/`</s>`/`<pad>` are
reachable only by text containing those literal substrings. Setting all four
entries of every row to -500 changes predicted scores by exactly 0.000000.

All four stored GlotLID-C-scale models are affected identically (0.800000 special
mass): `glotlidc.unilid`, `glotlid_apertus200k_fp64.unilid`,
`glotlid_apertus131k_fp64.unilid`, `glotlid_mistralnemo_fp64.unilid`.

## The corrected artifact is a transformation, not a retrain

Because SentencePiece emits normalized log-probabilities, the real tokens sum to
exactly 1 before normalization and the four specials add exactly 4, so the
released rows are the corrected rows divided by 5. The correction is therefore:

    real tokens += log(5) = 1.6094379 ; special tokens := log(1e-12)

VERIFIED, not assumed: `aai_Latn` (24,580 training lines) was retrained from its
corpus with the fixed 0.3.0 code against the released base tokenizer, and
compared to (released row + log 5) over the 99,996 real tokens. Correlation
1.00000000, median absolute difference 1.7e-5, mean 1.8e-5, max 7.4e-3, and
99.69% of tokens within 1e-4. The residue is float32 storage precision in the
tail. Retraining 1,940 languages on 60M samples is not required.

This should still be gated on more than one language before anything ships: see
step 1 below.

## What the correction does and does not change

Adding the same constant to every real token of every row does not cancel,
because scoring is a Viterbi over segmentations: a per-token bonus favours
segmentations with more tokens, and cross-language comparisons then shift
wherever two languages segment a text into different numbers of tokens.

Measured, base mode, on 20,000 labelled lines sampled from the GlotLID-C test
pool: accuracy 0.9494 as shipped, 0.9509 corrected. 140 of 20,000 predictions
change (0.70%): 63 gold-correct predictions gained, 32 lost. This is a sample
statistic on a subsample, not the paper's macro F1 over 1,940 labels, and it says
nothing about the calibrated path, whose thresholds no longer apply.

## Work

### 1. Produce and gate the corrected weights

- Apply the transformation to `glotlidc.unilid`, writing a version-1 file.
- Gate A, extending the single-language check above: retrain 10 languages spread
  across the training-size range with 0.3.0 and require the same agreement
  (correlation 1.0, median absolute difference below 1e-4). Any language that
  fails is investigated before proceeding, not averaged away.
- Gate B: the corrected file's special-token mass is below 1e-9 for all 1,940
  rows and its real-token mass is 1.0 to float32 tolerance.
- Record the transformation as code in `analysis/` so the artifact is
  reproducible from the released file rather than by hand.

### 2. Re-derive the calibration

The existing calibration cannot be carried over: its thresholds are percentiles
of score margins, and margins move with the segmentation changes above.

- **Unseen-token constant c.** The plateau moves from about -19 to about -17.4,
  so the current c = -21 still clamps, but the value was chosen by a sweep on the
  old scale. Re-run the sweep (`analysis/floor_sweep.py`,
  `analysis/floor_equalization.py`). Note for interpretation, not as a shortcut:
  c = -21 + log(5) = -19.3906 would reproduce the old calibrated behaviour up to
  the same uniform per-token shift, so the sweep should be expected to land near
  it; if it lands far away, that is a finding worth understanding before shipping.
- **Per-language thresholds tau.** Re-estimate for the 1,080 group-A languages
  and the group-B members with the recipe of record (`analysis/solo_gates.py`,
  size-adaptive q_L on the clamped matrix). Inputs are present:
  `results_apertus200k/corpus/` holds all 1,940 per-language training files on
  store, and `glotlid_train_counts.json` holds N_L.
- **High-entropy group.** Its four members were identified by an error analysis
  of the base model's validation predictions, which change. Re-run the
  identification under the criterion in the paper's app:protocol (entropy
  z-scored within script via median/MAD > 1.5 and misclassified-validation
  absorption > 2x own support, or z > 5, restricted to >= 18,000 samples). If the
  membership changes, that is a paper-level change, not just an artifact change.
- Keep the selection discipline the paper states: constants selected on the
  validation sample, never on the test pool.

### 3. Re-run the evaluations

Which numbers move, by owner. Everything produced by a UNILID model moves;
nothing produced by fastText, GlotLID, or CLD3 moves; latency and training-time
numbers do not move, since neither the shapes nor the operations change and no
retraining happens.

**Re-runnable here** (GlotLID-C-trained models, data and scripts on store):

| Table | What it reports | Script |
|---|---|---|
| `lid_main` | UNILID, calibrated, Mistral-Nemo rows on GlotLID-C, UDHR, FLORES, and the CLD3-subset columns | `unilid_resources/eval_{glotlid,udhr,flores}.py` |
| `calibrated_heldout` | held-out macro F1 and FPR | `analysis/full_test_eval.py`, `analysis/release_gates.py` |
| `calibrated_views` | per-language F1 by resource tier, both metric views | `analysis/paper_breakdowns.py` |
| `calibration_provenance` | which component came from which data | regenerate from the new derivation |
| `commonlid` | CommonLID out-of-domain | `analysis/commonlid_eval.py`, `commonlid_calibrated.py` |
| `calibrated_nemo` | the variant-transfer check | `analysis/mistralnemo_*`, same recipe on the corrected Nemo file |
| `resource-tier`, `script-breakdown` | GlotLID-C stratified views | `analysis/paper_breakdowns.py`, `regen_resource_tier_counts.py` |
| `viterbi_vs_marginal` | Viterbi against forward marginalization | `analysis/full_test_eval.py` with `forward=` |
| `lenbias-delta`, `lenbias-norm` | length bias and length normalization | `analysis/length_bias.py` |

The GlotLID-C full test pass is the expensive one: 45,627,279 lines, against
about 2 minutes per 250,000 lines in the release gates, so on the order of 6
hours per full scoring run per mode. Budget for base, calibrated, and forward.

**Needs the co-author's artifacts** (WiLI-trained and DSL-ML-trained UNILID
models, none of which are on store or scratch here):

| Table | Why |
|---|---|
| `unilid_llm_comparison`, `noise_robustness`, `length_accuracy`, `samples-accuracy`, `vocab_size_efficiency` | all WiLI-trained UNILID models |
| `tatoeba_udhr_comparison` | UNILID trained on WiLI, evaluated on Tatoeba and UDHR |
| `per_language_f1`, `fasttext_epoch_sweep`, `dialect_stats` | DSL-ML dialect models; the fastText rows and the dataset statistics are unaffected, the UNILID column is not |
| `latency_wili` | latency is unaffected, but confirm the model it timed |

Those models must be corrected the same way before re-evaluation. The
transformation applies to any model trained with the `sp` per-language method;
confirm each one's special-token mass first, because a model trained with the
`soft`/`hard` method instead carries a per-language mass of p(unk) rather than a
uniform 0.8, and for those the correction is not a single constant.

**Unaffected**: `latency_glotlid`, `training_time`, and every fastText, GlotLID
and CLD3 row and dataset-statistics table.

### 4. Update the paper

- Replace every UNILID number the re-runs change, in the tables above and in the
  prose that quotes them (the abstract's and results' headline figures, and the
  `sec:calibration` paragraph).
- `app:protocol` gains the correction's provenance: what the defect was, that the
  corrected weights are a deterministic transformation of the submitted model,
  and which constants were re-derived.
- The appendix currently attributes the above-c unseen-token values to "a
  byproduct of its training pipeline and data scale". That sentence should name
  the actual cause.
- Check whether any claim's direction changes rather than only its digits. On the
  20,000-line sample the base-mode change is +0.0015 accuracy, which would not
  flip UNILID against fastText (.929 against .944 in `lid_main`), but the macro F1
  over 1,940 labels has not been measured and must be before any claim is
  restated.

### 5. Ship

- Upload the corrected file under a new name (for example
  `unilid-1940-calibrated-v2.unilid`) rather than replacing the current one, so
  the artifact behind the published paper stays retrievable.
- Model card: the correction, the new numbers, and which file matches which
  version of the paper.
- `REPRODUCING.md` gains a section mapping each paper version to its artifact.
- The polybox version-1 mirror needs the same treatment or an explicit note.

## Order and gating

1 and 2 must complete before 3, because the evaluations need the new calibration.
Within 3, run the golden-subset gates first as a smoke test, then the full pool.
Nothing ships before the paper numbers are regenerated, since the artifact and
the paper have to agree.

## Open items

- Whether the Apertus 200k and 131k variants are published, or only corrected
  locally so the numbers the paper cites can be regenerated. They are corrected
  either way; only publication is open.
- Whether the co-author's WiLI and DSL-ML models still exist, and when those
  evaluations can be re-run. Everything in this repository can proceed without
  them, but the paper cannot be finished until they are done.
- Whether the package should offer users a migration for their own pre-0.3.0
  models. The transformation here is a research-repo script; a
  `unilid-calibrate migrate` subcommand would give the same thing to anyone who
  trained their own model with an affected version.
