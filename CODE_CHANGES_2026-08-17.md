# Code changes, 2026-08-17: special-token correction and the re-release chain

A review-oriented record of every change made to this repository on 2026-08-17,
in the order it was made, with what was verified for each. The commit messages
carry the same reasoning; this file exists so the whole set can be read in one
place without walking the log.

Companion records: `EXPERIMENTS_RESULTS.md` (findings), `RERELEASE_PLAN.md`
(execution plan and gates), `EXPERIMENTAL_SETUP.md` (protocols and constants),
`OPEN_SOURCE_STATUS.md` (the package and its release).

Package changes (UNILID 0.3.0, PR #3 on the fork) were made earlier the same day
and are recorded in `OPEN_SOURCE_STATUS.md`; this file covers the analysis
repository.

---

## Commit `8662438`: bring the records current

**Why.** None of the four standing research documents had an entry after
2026-08-11, and three recorded the special-token structure as benign ("0.8 of all
mass, uniform across languages so argmax-neutral"). That reading is wrong twice
over, and both halves are now measured.

**New code.**

- `analysis/segmentation_shift.py` (new). Measures what the correction does to
  the Viterbi segmentation and to the score. 3,000 pool lines, seed 7. Output:
  `outputs/rerelease/segmentation_shift.json`.

**Documentation corrected in place** (old wording preserved and labelled, not
deleted):

| File | What was wrong |
|---|---|
| `EXPERIMENTAL_SETUP.md:217` | "uniform across languages so argmax-neutral" |
| `EXPERIMENTAL_SETUP.md:517` | Good-Turing budget derived from "0.2 is a structural property of the model" |
| `EXPERIMENTAL_SETUP.md:501` | "the four special tokens bit-identical" under the clamp |
| `EXPERIMENTAL_SETUP.md:455` | "four constraints" over a list of five |
| `EXPERIMENTS_CHRONOLOGICAL.md:921` | same argmax-neutral claim |
| `EXPERIMENTS_PLAN.md:950` | same argmax-neutral claim |
| `OPEN_SOURCE_STATUS.md` | attributed the -19 plateau to the special-token defect |

`EXPERIMENTS_RESULTS.md` gained six 2026-08-17 entries and its invalidated
section was split into **premise now false** (Exp 27's 0.2 budget, Exp 50's
`bgfloor`) versus **valid for the shipped artifact, superseded pending
regeneration** (Exp 20's c sweep, the threshold families, the high-entropy group,
Camera-ready E1 to E5).

**Verified.** `segmentation_shift.py` reproduces from the repo the same figures
measured ad hoc: 1,140 of 3,000 lines re-segment, all 1,140 toward more tokens,
mean token count 39.369 to 39.920, 14 predictions change, and on the 1,860 lines
with prediction and segmentation both unchanged the score delta equals
`n * log 5` to within 5.5e-4.

**One claim I withdrew during this commit.** I first wrote that Exp 27's 9x/12x
overstatement ratio "survives" the correction because the factor of 5 cancels
between its two sides. That is incomplete: n1 and T are counts under each
language's own Viterbi segmentation, which the correction shifts. The text now
says both counts must be recounted.

---

## Commit `20c6db3`: B0, the unseen-token plateau

**Why.** The paper's appendix (`submission.tex:629-631`) attributes each row's
unseen-token plateau to the training-time probability floor of 1e-12. It is not
that. The known alternative (Exp 10, `corr = -0.966` with corpus size) had never
been persisted as an artifact, and it is confounded: each of its 1,940 points is
a different language.

**New code.**

- `analysis/plateau_reference_fit.py` (new). Re-derives the cross-language
  relation from the committed `outputs/diagnostic/gt_counts.csv`, so the
  reference the experiment is compared against is itself traceable. Output:
  `outputs/rerelease/plateau_reference_fit.json`.
- `analysis/plateau_vs_corpus_size.py` (new). Removes the confound by holding
  language identity fixed: one corpus shuffled once (seed 20260817), nested
  prefixes of 1,000 / 3,000 / 10,000 / 30,000 / 100,000 lines retrained against
  the same unmodified base tokenizer. Reuses `gate_correction.retrain_row` rather
  than reimplementing training. Captures `real_missing` from the trainer's log
  record rather than by changing the shipped package. Output:
  `outputs/rerelease/plateau_vs_corpus_size.json`.

**Verified.** `plateau_reference_fit` reproduces the previously reported figures
exactly (`-5.539 - 2.039 * log10(T)`, `corr -0.9924`, R-squared 0.985).
`plateau_vs_corpus_size` passes 3/3 with within-language slopes -2.196 / -2.196 /
-2.184 at R-squared 0.999, and the two fits agree to 0.006 nats near the median
token count. `real_missing` is 0 in all 15 runs.

**Pass criterion was registered in the script before the run** (within 50% of the
cross-language slope, `real_missing` near zero) and the result cleared it by a
wide margin.

---

## Commit `834723a`: one place decides which model a run uses

**Why, and this is the load-bearing one.** Every script in the chain resolved its
model through `transfer_sweep.UNILID_MODEL_PATH` and its paths through
`full_test_eval.SCRATCH_DIR`, both module constants with no override. **40 entries
of that directory are symlinks into `/capstor/store/cscs/swissai`**, which holds
the reference arrays for both release gates and the provenance chain for every
published GlotLID-C number. Pointing any script at corrected weights would have
overwritten published artifacts in place, with no error.

**New code.**

- `analysis/model_context.py` (new). Resolves the (model, output root) pair and
  refuses a non-default model paired with the default root, or with any root
  having store-backed entries. Also `model_sha256`, `store_backed_entries`,
  `require_default_model`, and the shared `--model` / `--scratch-dir` flags.
- `analysis/model_context_selfcheck.py` (new). Fires every branch.

**Scripts wired.** `full_test_floor21.py`, `solo_gates.py`, `gate_variants.py`.
Each also now names its special columns from the model's own vocabulary when
clamping, so the clamp cannot be silently disabled on a 0.3.0-trained matrix.

`gate_variants.py` shares its model across a dozen free functions, so it
configures a module context once at entry rather than threading a parameter
through every signature. Unconfigured, it falls back to the released model, which
is the historical behaviour.

**A pre-existing breakage this surfaced and fixed.** `SPECIAL_P` had been deleted
from `floor_equalization.py` in commit `439836b`, leaving `full_test_gt.py`,
`full_test_bgfloor.py` and `gt_counts.py` with a dead import; all three had been
unimportable since. It is restored **with its meaning changed**: it is the
pre-0.3.0 assertion value, not a way to locate special columns (that is
`_special_columns`, which reads the vocabulary). Those three scripts therefore
keep aborting on a corrected matrix, which is wanted: their arithmetic is derived
from the 0.2 budget throughout and does not hold for one.

**Verified.** Six resolver branches fired as specified. Every module in
`analysis/` imports.

---

## Commit `d2cbc54`: wire the rest of the chain

**Scripts wired.** `release_gates.py`, `build_release_calibration.py`,
`commonlid_carried.py`, `commonlid_calibrated.py`, `mistralnemo_eval.py`.

Two needed more than plumbing, because a guard that lets an hour of scoring run
and then fails on a number is worse than one that refuses at the start:

- **`release_gates.py`** compares against reference arrays recorded from the
  released weights. The correction changes 0.72% of predictions by design, so a
  corrected model fails exact equality by construction and the 0.999 calibrated
  bar too. It now refuses a non-default model without an explicit `--ref`.
- **`commonlid_calibrated.py`** has three reproduction gates citing Exp 12/39
  values from the released weights. The score stage records the model identity in
  its sidecar; the eval stage reads it and, for a non-default model, names the
  four constants that must be re-recorded rather than firing as an opaque numeric
  mismatch.

**`mistralnemo_eval.py`** derives fifteen paths from a root that is itself a
symlink into the store, so `configure()` re-resolves and re-derives them.
`resolve()` gained a `default_model` parameter for it: that chain's default is
the packed variant, not the base model, so without it the ordinary invocation
would have been refused.

**Backward compatibility checked.** `build_release_calibration.py` still
reproduces the shipped `outputs/release/calibration_glotlidc.json`
**byte-for-byte**. The first version of this change added two provenance fields
unconditionally and broke that; they are now recorded only when the inputs are
not the defaults. `gate_variants.py`'s `TAU_FLAT4_CSV` and
`TAU_FLOOR21_GATE_CSV` were briefly renamed, which broke
`external_bench_eval.py` and `commonlid_calibrated.py`; both names are restored
as the default-root spellings alongside the configurable form.

**Verified.** `model_context_selfcheck` now also runs all seven entry points and
confirms each refuses the corrected model on its default root: 13/13.

---

## Uncommitted at the time of writing: `full_test_eval.py`

Three defects found while preparing the first corrected-model run, all of the
silent kind:

1. **The language list came from the wrong model.** `_load_model_data()` was
   called with no argument, so `langs` and `lang_to_idx` came from the default
   model even when a different one was being scored. Every prediction is written
   as an index into `langs`, so a model with a different language order would
   have been mislabelled throughout. Harmless for the corrected model, whose
   language list is identical, but not a property to rely on.
2. **The agreement gate measured the wrong thing for a second model.** Baseline
   predictions are checked against `pred_UniLID` from the sample pickle, recorded
   from the released weights, and the run aborts below 0.99. For a corrected
   model the predictions are *supposed* to differ; at 0.72% changed it lands near
   0.9928 and would have squeaked past the threshold by luck rather than by being
   checked. It is now a hard gate for the released model and a reported
   measurement, with an explicit note, for any other.
3. **No way to run a subset of configurations.** One invocation scored all three
   of `baseline`, `freq_prior`, `learned_bias`, which is three full-pool passes.
   Neither prior-side configuration appears anywhere in `submission.tex`, and
   `learned_bias.npy` was fit to the old model. A `--configs` flag was added;
   **author decision 2026-08-17: baseline only for the corrected run**, a 3x
   saving. The stale-`learned_bias` guard now applies only when that
   configuration is actually requested.

Ordering note: `configs` is normalized to `CONFIGS` order so `baseline` is first,
because the delta-against-baseline reporting slices `configs[1:]`.

---

## Commit `01828c6`: the two tables that had no generator, and a sweepable c grid

**New code.**

- `analysis/viterbi_vs_marginal.py` (new). Nothing in the chain called the
  forward (marginalizing) scorer over the pool, so `tab:viterbi_vs_marginal`
  could not be regenerated at all. Chunked and resumable in the manner of
  `full_test_eval.py`, fingerprinted on the model, and it refuses the released
  model's memmap directory outright because it writes new arrays there.
- `analysis/lenbias_norm_table.py` (new). `normalized_predict.py` was extended
  into an eleven-value alpha sweep whose tables carry neither the Original column
  nor the agreement check that gives the Raw rescore column its meaning. The
  agreement check is kept as a hard gate at exact agreement. For a non-default
  model the Original column is omitted rather than filled from another model's
  recorded predictions.

**Changed.** `analysis/floor_equalization.py` gained a `floor_grid` parameter, a
CLI, and `FLOORS_CORRECTED` (the published grid shifted by log 5). The clamp sets
an absolute target in log space and the correction moved every real token up by
that amount, so the shifted grid asks the published question of a corrected
model; the unshifted grid would ask a different one.

**A bug I introduced and caught in the same change.** The new parameter was first
called `floors`, which collided with the per-row minimum array of the same name
computed thirteen lines later. The sweep would have iterated over 1,940 row
minima instead of the four grid values. Renamed to `floor_grid` before running
anything.

---

## Runs launched

| Run | Job | Started | Expected |
|---|---|---|---|
| Corrected-model full-pool baseline, `--configs baseline` | SLURM 3107045 | 2026-08-17 | 92 chunks, about 2h14m of scoring |

Submission script: `slurm_full_test_eval_corrected.sh`. Scratch root
`/capstor/scratch/.../full_test_eval_corrected/`, tables to `outputs_corrected/`.

**Note on how it was launched.** Two attempts to run this detached from the
session (`nohup ... &`, then `setsid nohup ... &`) were both killed along with
the session, after 1 and 3 chunks respectively. The run is resumable and the
banked chunks were kept, but a multi-hour job belongs in the queue, not on a
login-node process tied to an interactive session.

---

## What has NOT been changed, deliberately

- The released artifacts on store and on the Hub. Untouched.
- `paper/submission.tex` and `paper/initial_version.tex`: uncommitted working-tree
  edits belonging to the user.
- The Good-Turing family (`gt_counts.py`, `full_test_gt.py`,
  `full_test_bgfloor.py`) and `token_tying.py` / `family_backoff.py`. Their
  arithmetic assumes the 0.2 budget throughout. They abort on a corrected matrix
  and are left that way until their derivations are redone.
- `EXPECTED_GROUP_B_LANGS` in `build_release_calibration.py`. Overridable by flag,
  but the assertion is not relaxed into a warning.
