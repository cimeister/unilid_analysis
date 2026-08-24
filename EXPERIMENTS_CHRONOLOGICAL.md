# UniLID Analysis — Chronological Experiment Log

> **Reconstruction provenance.** Rebuilt on 2026-05-27 after the original session
> transcript (`9729f7f3-3af8-42d5-818a-1f032a9f6f25`, 2026-03-26 → 2026-04-08) was lost.
> SLURM **job IDs, states, durations, and memory** are taken verbatim from the job-history
> tables in `EXPERIMENTS.md` and are reliable. **Absolute calendar dates** for each run are
> **[inferred]** from the recovered prompt timestamps and source-file modification times,
> not from SLURM records, so treat them as approximate. There is only one git commit for
> the whole project (`b7508fd`, 2026-04-08); per-experiment code versions are not separately
> tracked, so the "code version" for every entry below is "working tree as of `b7508fd`".
> Shared configuration is in `EXPERIMENTAL_SETUP.md`; outcomes are in
> `EXPERIMENTS_RESULTS.md`.

Reverse-chronological, grouped by experiment family. Within each family, jobs are listed
most-recent-first. All jobs ran on CSCS Clariden (account `infra01`, partition `normal`;
this header said `a139` until 2026-08-06, which SETUP.md records as wrong),
Python `/users/cmeister747/.pyenv/versions/3.11.5/bin/python3`, working dir
`/users/cmeister747/unilid_analysis`, data on scratch (`config.DATA_DIR`). See `SETUP.md`
for the infrastructure record.

---

## Family: Error analysis, calibration, and the balanced protocol (Exp 10–24)

**Window:** 2026-06-24 to present. Plan items: `EXPERIMENTS_PLAN.md` Exp 10–16 and the
"Next set of methods" items. Setup:
`EXPERIMENTAL_SETUP.md` (hierarchical pooling). Full plan:
`~/.claude/plans/yes-do-both-then-giggly-sprout.md`.

### 2026-08-21: WiLI models located, measured, and three retrains submitted (jobs 3138626/7/8)

- **Purpose / hypothesis:** whether the five WiLI-trained paper tables need
  regenerating. Settled by measurement: all three WiLI models from
  github.com/Ahmetcanyvz/UNILID/releases carry 0.800000 special-token mass per
  row, so they were `sp`-trained and carry the defect. Plan:
  `~/.claude/plans/this-session-focuses-on-shimmering-dusk.md`, approved after an
  adversarial review that returned two critical defects.
- **Assets:** two draft releases, eleven files. WiLI corpus (117,500 train and
  117,500 test, 235 languages, 500 per language), Tatoeba, UDHR, FLORES, DSL-ML,
  and three of eleven models. **No 10k / 20k / 50k / 200k model exists**, so
  `tab:vocab_size_efficiency` cannot be rebuilt from a container. Downloaded to
  `/capstor/scratch/.../wili_assets/`. CORRECTION 2026-08-23: this entry
  overstated the download. Only `wili-2018.zip` and the models actually landed
  in `wili_assets/`; `tatoeba.zip` (980 MB) was listed on the releases page but
  never downloaded — the filesystem, not this record, is right.
- **Phase 0, the instrument, PASSED:** `analysis/wili_eval.py` reproduces the
  published cells from the stored defective model at macro F1 0.960113, accuracy
  0.956502 and macro FPR 1.8589e-04. Written because no WiLI tooling existed here
  and `UNILID/eval.py` reports no macro FPR.
- **Base vocabulary provenance measured:** byte-identical between the WiLI and
  GlotLID-C containers for DeepSeek3.2 (`79b4c295...`) and Qwen3 (`311d4685...`),
  because an LLM tokenizer is carried unchanged; different for the 100k model
  (24,357 of 100,000 tokens shared), because a supplied-nothing model has its
  vocabulary trained on the corpus.
- **Init-from:** each model's own base tokenizer, extracted with
  `analysis/extract_base_tokenizer.py`, which writes only the base tokenizer.
  `unpack_unilid` would also write 235 defective per-language rows, and
  `convert.py` globs `langspec_soft_*` before `langspec_sp_*`, so the pack would
  pick the corrected set by naming coincidence rather than by construction.
- **Data:** one shared 235-language corpus split at
  `/capstor/scratch/.../wili_corpus_shared` (117,500 lines, 64 MB), built once
  with `train.prepare_corpus` so all three models train on byte-identical files.
- **Configuration:** `slurm_wili_train_fp64.sh`, parameterized by `MODEL_NAME` and
  `VOCAB_SIZE` via `--export`. Patched fp64 `spm_train` (fork commits d0208d9 +
  c5921a2) and UNILID 0.3.0, so the retrain changes **two** things, not one.
- **The trap the review caught, now guarded:** `--results-dir` and
  `--base-tokenizer-path` are both mandatory. Without them `train.py:450-452`
  defaults the base path to `results_<vocab//1000>k/tokenizers/`, the reuse test at
  `train.py:455` fails, and `train.py:465-492` **silently trains a fresh
  vocabulary and reports success**. The script also refuses a results directory
  resolving into the durable store, and refuses to start if per-language rows
  already exist, since `--skip-existing-langs` defaults to true and the loader
  validates token order but not real-token mass.
- **Expected:** WiLI is 64 MB over 235 languages against GlotLID-C's 1,940
  languages with far more text, so well under the 4h19m the DeepSeek GlotLID-C
  retrain took. 4h walltime requested.
- **Blocked and needing the co-author:** the four LLM-tokenizer variants
  (Mistral-Nemo extractable from the GlotLID-C container; Mistral, LLaMA3.2 and
  LLaMA2 have no container anywhere and their cached HuggingFace tokenizers are
  dangling symlinks with no blobs), and `tab:samples-accuracy`'s seed count.
  `tab:noise_robustness` is on hold by author instruction.

### 2026-08-21: GlotLID-C chain, five jobs completed

- **3112879** DeepSeek3.2 retrain COMPLETED 04:19:26; **3112846** Qwen3 retrain
  COMPLETED 04:49:30. Both clean: real-token mass 1.000000, defect absent.
  **Every corrupted row disappeared and every coverage row persisted**, which
  settles the open `mya_Mymr` question (corruption, not Burmese coverage) and
  corrects the earlier classification of `bod_Tibt` (corruption in both variants,
  not coverage). Results entry: "Both variant retrains completed".
- **3123324** group-A thresholds COMPLETED 00:12:48: 1,080 rows with 26 excluded,
  all `low_calibration`, matching the released model's expected counts exactly, so
  `build_release_calibration.py`'s group-A assertions pass unchanged. Read c = -17
  from the fingerprint. Artifact
  `outputs_corrected_round/diagnostic/tau_floor21_gate.csv`.
- **3129778** `tab:lenbias-norm` COMPLETED 00:02:09 on the golden subset, with the
  Original column filled and the implementation check at agreement 1.000000.
- **3130020 / 3130021** UDHR and FLORES score stages COMPLETED. **A provenance
  defect was introduced and fixed the same day**: the first run overwrote the
  released model's E2 scored artifacts and recorded the module-constant model path
  and floor target rather than the ones used. The scored data was correct (the
  sha256 check against the corrected fingerprint passed), but the mislabelled
  files were deleted, the recording fixed, output isolated per model, and both
  benchmarks re-scored.

### 2026-08-18: Qwen3-8B variant retrain under the patched trainer submitted

- **Purpose / hypothesis:** the Qwen3 model's `azj_Latn` row is corrupted in the
  manner of the fixed-vocabulary EM bug (Exp 41/42), independently of the
  special-token defect. Retraining with the patched fp64 trainer under UNILID
  0.3.0 repairs both at once. Plan item: `EXPERIMENTS_PLAN.md`, the DeepSeek3.2
  and Qwen3 rows. Author decision 2026-08-18.
- **Init-from:** not a continuation. Base tokenizer **extracted from the existing
  container** (`qwen3_8b_glotlid.unilid`) rather than re-converted from a
  HuggingFace tokenizer.json, so the vocabulary is bit-identical to the model
  being replaced and only the trainer and the special-token handling change.
  Written to `results_qwen3_8b_fp64/tokenizers/langspec_base_tokenizer.json`;
  `--reuse-base` loads it directly so `_convert_to_unigram_base` never runs.
  Preflight: Unigram type, 151,670 entries, all four special tokens present at
  indices 128245 / 128247 / 151669 / 128244, base scores uniform at
  log(1/151667) = -11.9294 with the specials at 0.0 (the defect's source, which
  0.3.0 no longer reads).
- **Data:** `results_apertus200k/corpus`, 1,940 files, the shared draw every
  retrain in this project has used. **Caveat to carry with the result:** for
  languages above the 100,000-line cap this is not necessarily the sample the
  original Qwen3 run saw, so the retrained rows are not expected to reproduce the
  originals exactly even where the original was sound.
- **Configuration:** `--vocab-size 151670 --byte-level --per-lang-counts-method
  sp --max-base-samples-per-lang 10000 --lang-batch-size 20 --reuse-corpus
  --skip-existing-langs --reuse-base`, patched `spm_train` from
  `~/.local/bin` (fork commits d0208d9 + c5921a2). Mirrors
  `slurm_mistralnemo_train_fp64.sh`.
- **Expected completion:** roughly 5 to 6 hours by interpolation between the
  recorded 131k retrain (4h36m, job 2903767) and the 200k retrain (7h28m, job
  2903768); 12h walltime is headroom.
- **Post-training gates, run in-job:** `analysis/inspect_variant_models.py` (real
  token mass must be 1.0 per row, not 0.2) and
  `analysis/variant_plateau_outliers.py` (`azj_Latn` must no longer appear;
  nothing new may appear that is not a shared minority-script coverage effect).
- **Artifacts:** `/capstor/scratch/.../glotlid_qwen3_8b_fp64.unilid`,
  `results_qwen3_8b_fp64/`, `outputs/rerelease/qwen3_fp64_inspect.json`,
  `outputs/rerelease/qwen3_fp64_plateau_outliers.json`. Logs
  `/capstor/scratch/.../logs/qwen3_fp64_%j.{out,err}`.

### 2026-08-18: round-grid c sweep submitted (job 3111471); floor-c pass 3110918 CANCELLED

- **Job 3111471** `unilid-cround-corr`, submitted after the pre-registration in
  `EXPERIMENTS_RESULTS.md` was written and committed (`3a9c65c`), not after
  seeing its result. Grid {-15, -17, -19, -21}, chosen by the rule the published
  grid follows: two values inside the model's own unseen-token plateau range, two
  below all of it. Selection procedure unchanged. Writes to
  `outputs_corrected_round/` so the shifted-grid sweep's table is not
  overwritten; both are reported. Script
  `slurm_floor_sweep_round_corrected.sh`.
- **Purpose:** the shifted grid {-15.3906, -17.3906, -19.3906, -21.3906} is the
  correct like-for-like comparison against the released model, but with no
  released predecessor for a reader to relate those values to, it prints as four
  arbitrary decimals. Author decision 2026-08-18: re-sweep on round numbers and
  take that as the constant of record for the paper.
- **Job 3110918 CANCELLED while still PENDING, nothing lost.** It would have run
  the floor-c full-pool pass at c = -17.3906 and written
  `fingerprint_floor21.json` at that constant into the corrected scratch root.
  `analysis/gate_variants.py` now reads its clamp target from that fingerprint by
  design, so leaving it to complete would have had the entire downstream chain
  build silently at a constant the round sweep is about to supersede. Verified no
  floor21 artifacts were written. Resubmit once job 3111471 names the constant.

### 2026-08-18: both corrected-model jobs COMPLETED

- **Job 3107045** `unilid-fulltest-corr` COMPLETED 01:42:36, MaxRSS 9.45G,
  exit 0:0. Full-pool baseline on the corrected weights, 45,377,279 lines.
  Outcome: overall macro F1 0.9292 to 0.9327 (+0.0035), accuracy 0.9608 to
  0.9609, tail 0.9132 to 0.9045 (-0.0087), magnets 0.9138 to 0.9067 (-0.0071),
  twins -0.0003, head -0.0006. **Qualifies the earlier "essentially a wash",
  which was the golden-subset measurement and stays true there.** Artifacts:
  `outputs_corrected/tables/full_test_eval.md`,
  `outputs_corrected/diagnostic/full_test_per_lang_f1.csv`.
- **Job 3107082** `unilid-csweep-corr` COMPLETED 00:13:49, MaxRSS 6.53G,
  exit 0:0. Selected c = **-17.3906**, one grid step above the pre-registered
  expectation of -19.3906. Aligned position by position against the released
  sweep, the rows clamped are identical at every position (452 / 1,821 / 1,940 /
  1,940) and every grid step is exactly log 5, so the shifted grid asked the same
  question. Positions 2 and 3 are tied in both models: the released model picks
  position 3 by 0.0001, the corrected model picks position 2 by 0.0002. **The
  constant did not move; a tie broke the other way.** All four values passed the
  all-strata guard in both models. Artifacts:
  `outputs_corrected/tables/floor_equalization.md`,
  `analysis/c_selection_comparison.py`,
  `outputs/rerelease/c_selection_comparison.json`.
- **Blocking decision raised to the author:** which c the re-release ships. The
  data does not determine it, and it propagates to all 1,084 thresholds, which
  are percentiles of margins on the clamped matrix.

### 2026-08-17: corrected-model unseen-token constant sweep submitted (job 3107082)

- **Purpose / hypothesis:** re-derive c for the corrected model. The probe showed
  c is consistent with carrying across by addition of log 5, but that was a
  60,000-line subsample on a flat optimum; the published protocol has to be run.
  Plan item: `EXPERIMENTS_PLAN.md` B2.
- **Protocol, unchanged from Exp 20:** sweep c, select on the validation half of
  the seed-42 500k draw under the all-strata guard, score the test half once.
- **Grid:** the published {-17, -19, -21, -23} shifted by log 5, that is
  {-15.3906, -17.3906, -19.3906, -21.3906}
  (`analysis/floor_equalization.FLOORS_CORRECTED`, `--corrected-grid`). The clamp
  sets an absolute target in log space and the correction raised every real token
  by exactly that amount, so the shifted grid asks the published question of a
  corrected model. The unshifted grid would ask a different one: at c = -21 a
  corrected row's unseen tokens sit 1.609 nats further below its seen tokens than
  the released model's did.
- **Expected:** near -19.3906, which would reproduce the released clamped
  structure up to the uniform shift. **A result far from that is a finding, not a
  tuning problem, and must be recorded as one.**
- **Artifacts:** `outputs_corrected/tables/`; submission script
  `slurm_floor_sweep_corrected.sh` (8h walltime, 64 cpus, 400G); logs
  `/capstor/scratch/.../logs/csweep_corrected_%j.{out,err}`.

### 2026-08-17: corrected-model full-pool baseline pass submitted (job 3107045)

- **Purpose:** the new base predictions for every regenerated GlotLID-C number.
  Plan item: `EXPERIMENTS_PLAN.md` B3 step 1.
- **Init-from:** not a training run. Model:
  `/capstor/scratch/.../corrected/glotlidc_corrected.unilid`, the
  special-token-corrected transformation of `glotlidc.unilid`.
- **Data:** the GlotLID test file, all 45,627,279 lines minus the 250,000
  validation lines, the same pool definition as the released run.
- **Configuration:** `--configs baseline` only. One full pool pass instead of
  three; neither `freq_prior` nor `learned_bias` appears in `submission.tex` and
  `learned_bias.npy` was fit against the released weights (author decision
  2026-08-17). Setup: `EXPERIMENTAL_SETUP.md`; submission script
  `slurm_full_test_eval_corrected.sh` (12h walltime, 64 cpus, 400G).
- **Artifacts:** scratch root
  `/capstor/scratch/.../full_test_eval_corrected/` (fresh, NOT the released
  model's store-symlinked directory; `analysis/model_context.py` enforces this
  rather than leaving it to be remembered). Tables under `outputs_corrected/`.
  Logs `/capstor/scratch/.../logs/fulltest_corrected_%j.{out,err}`.
- **Expected:** about 2h14m of scoring by the recorded rate for a full base pass.
  Resumable; 3 of 92 chunks were already banked by two login-node attempts before
  submission, both of which were killed along with the session that started them.

### 2026-08-17: B0, the unseen-token plateau is set by corpus size (login node, no SLURM, 15 spm_train runs)

- **Purpose / hypothesis:** the cross-language plateau relation is confounded,
  each of its 1,940 points being a different language. Does corpus size account
  for it with language identity held fixed? Plan item: `EXPERIMENTS_PLAN.md` B0.
- **Design:** one corpus shuffled once (seed 20260817), nested prefixes of 1,000 /
  3,000 / 10,000 / 30,000 / 100,000 lines retrained against the same unmodified
  base tokenizer. Three languages chosen deterministically from the 282 at the
  100,000-line cap: `abk_Cyrl`, `mam_Latn`, `zul_Latn`. Pass criterion registered
  in the script before the run: within-language slope inside 50% of the
  cross-language slope, `real_missing` near zero.
- **Outcome: PASS 3/3.** Slopes -2.196 / -2.196 / -2.184 nats per decade of
  tokens, R-squared 0.999, against -2.039 across 1,940 languages. On a common
  scale the within-language fit is `-4.628 - 2.192 * log10(T)` against
  `-5.539 - 2.039 * log10(T)`, agreeing to 0.006 nats at log10 T = 6 and to within
  0.30 nats over log10 T = 4 to 7. The plateau probability scales as `T^-0.95`,
  about one count in T. `real_missing` is 0 in all 15 runs, ruling out the
  base-tokenizer fallback as a source of the plateau.
- **Consequence:** `submission.tex:629-631` can be rewritten with a measured
  causal statement instead of the current floor attribution.
- **Artifacts:** `analysis/plateau_vs_corpus_size.py`,
  `analysis/plateau_reference_fit.py` (the reference fit, re-derived from the
  committed `outputs/diagnostic/gt_counts.csv` because it had never been
  persisted), `outputs/rerelease/plateau_vs_corpus_size.json`,
  `outputs/rerelease/plateau_reference_fit.json`.

### 2026-08-17: special-token defect found, fixed in the package, and the four stored models corrected (login node, no SLURM)

- **Purpose / hypothesis:** a setup report of large `--method sp` vs `--method em`
  score differences in `add_language`, initially assumed to be the reporter's
  configuration error. Plan item: `EXPERIMENTS_PLAN.md` "Special-token correction
  and re-release".
- **Defect:** `unilid/trainers/language_specific_trainer.py` gave each special
  token the base tokenizer's score; HF Unigram stores specials at `0.0`, read as a
  log-probability that is probability 1.0. Four of them dominate the
  normalization, each lands at 1/5, every real token is depressed by
  log 5 = 1.6094 nats. All four stored GlotLID-scale models carry exactly 0.800000
  special mass per row. No special token's weight is ever read when scoring
  (verified by perturbation: all four set to -500 changes scores by 0.000000), so
  the mass is unusable, not inert.
- **Package fix, version 0.3.0:** one enforcement point in
  `LanguageSpecificUnigramLMTokenizer.train` renormalizes over real tokens and
  parks the specials at the floor, whichever method produced the row
  (`unilid/vocab_io.py::renormalize_over_real_tokens`); `add_language` puts a new
  row on the model's own scale (`_match_real_token_scale`; new named constant
  `SCALE_SPREAD_REPORT_RATIO = 2.0`, diagnostic only). Branch
  `calibration-release`, PR #3, commits 9f7c1cf then 56e7fd4.
- **Correction of the stored models:** `analysis/correct_special_token_mass.py`
  (guard `MAX_REAL_MASS_SPREAD = 1.01`), a closed-form transformation rather than a
  retrain. Outputs on scratch under `corrected/`: glotlidc, apertus200k,
  apertus131k, mistralnemo.
- **Gate:** `analysis/gate_correction.py`, eight languages N_L 85 to 100,000
  retrained under the fixed code, 8/8 pass
  (`outputs/rerelease/gate_correction.json`). Criterion bounds the signed mean
  (0.01), mass-weighted difference (0.02) and correlation (0.9999); it does not
  require exact row reproduction, because languages above the 100,000-line cap
  were subsampled and the store corpus is the Apertus draw. The threshold was
  chosen after `zul_Latn` failed an exact-reproduction criterion; recorded here
  because that is the kind of change that has to stay visible.
- **Released artifact untouched by the code fix; both golden gates re-passed.**

### 2026-08-17: effect of the correction measured on the golden subset (login node, no SLURM)

- **Purpose:** decide whether the re-release is justified by metrics or only by
  correctness.
- **Outcome:** base mode, 250,000-line golden subset against recorded gold labels.
  Macro F1 0.9454 to 0.9460, macro FPR 2.083e-05 to 2.081e-05, accuracy 0.9603 to
  0.9604; 1,807 of 250,000 predictions change (0.72%), 699 fixed and 669 broken.
  A wash. **The case for re-releasing is correctness, not metrics.** Supersedes an
  earlier 0.9494 to 0.9509 estimate, which was accuracy on a 20,000-line
  every-149th-line sample and under-powered.
- **Artifacts:** `analysis/correction_effect.py`,
  `outputs/rerelease/correction_effect.json`.

### 2026-08-17: the correction is not a constant offset; segmentation moves (login node, no SLURM)

- **Purpose:** test the standing record's claim that the special-token mass is
  "uniform across languages so argmax-neutral".
- **Outcome:** false on both counts. Each language scores under its own Viterbi
  segmentation, so the per-token depression applies a different number of times per
  candidate; and the correction moves the segmentation, since the max-plus DP
  maximizes `sum(log p_i) + n * log 5` and a positive per-token constant favors
  more tokens. On 3,000 pool lines: 1,140 re-segment, all toward more tokens, mean
  token count 39.369 to 39.920, 14 predictions change. On the 1,860 lines with both
  prediction and segmentation unchanged, the score delta equals `n * log 5` to
  within 5.5e-4.
- **Artifacts:** `analysis/segmentation_shift.py`,
  `outputs/rerelease/segmentation_shift.json`. Three record sites corrected in
  place (this file's 2026-07-18 token-tying entry, `EXPERIMENTAL_SETUP.md:217`,
  `EXPERIMENTS_PLAN.md:950`).

### 2026-08-17: the 0.3.0 fix silently disabled the unseen-token constant; found and fixed (login node, no SLURM)

- **Purpose:** probe how far c moves under the correction. It surfaced a
  regression first.
- **Regression:** `analysis/probe_calibration_shift.py` returned `modified 0` of
  1,940 rows at every c for the corrected model against 1,940 for the released
  one. Parking the specials at `MIN_TOKEN_LOG_PROB` makes them each row's minimum,
  and `apply_unseen_token_constant` defines the unseen tokens as the exact
  minimum-value plateau, so the plateau is never located and the clamp does
  nothing. Every model trained by 0.3.0 as first shipped had the calibration's
  first correction disabled with no message.
- **Fix:** the clamp takes the special columns and excludes them from the minimum
  (`unilid/calibration.py`, `analysis/floor_equalization.py`); both callers find
  the columns by name from the vocabulary, and the old `SPECIAL_P = 0.2` detector
  was deleted because it cannot work on a corrected model. Pre-0.3.0 files
  unaffected (specials at -1.6094, never the minimum), asserted in a test
  alongside one for the broken case. Package commit 2d5f62d. Both release gates
  re-run because this is an inference-path change.

### 2026-08-17: calibration probes; c carries by addition, the thresholds do not (login node, no SLURM)

- **Purpose:** decide whether the calibration constants can be carried to the
  corrected model or must be re-derived, before paying for a full re-derivation.
  Both probes select on the validation half of the seed-42 draw and never touch
  the golden subset.
- **c:** 60,000 validation lines (seed 20260817), nine-value grid. Released optimum
  -19.5 (macro F1 0.95686) against -21 (0.95671); corrected optimum -17.5
  (0.95726). Shift +2.0 against log 5 = 1.609, and the optimum is flat enough here
  that the two are not distinguishable. `outputs/rerelease/probe_c.json`.
- **tau:** six group-A languages, released clamped at -21 and corrected at
  -19.3906. Two excluded in both (`kdr_Latn`, `chq_Latn`, `low_calibration`). The
  other four move `tul_Latn` -12.5%, `bkv_Latn` -5.5%, `mpm_Latn` -71.8%,
  `cmo_Latn` +123.3%; mean delta -0.40 nats. Both directions, two orders of
  magnitude apart in relative size, so no shift or scaling carries them: **all
  1,084 must be re-estimated.** `outputs/rerelease/probe_tau.json`.

### 2026-08-17: analysis-chain safety fixes so a second model cannot be scored through the first model's paths (login node, no SLURM)

- **Purpose:** the regeneration points existing scripts at a different model, and
  several of them write through scratch directories that are symlinks into the
  durable store, or assert against the released model's artifacts.
- **Done and each verified by triggering the guard, not by reading it:**
  `full_test_eval.py` (model sha256 in the fingerprint; refuses to write a
  non-default model into the store-symlinked root; refuses a stale
  `learned_bias.npy`), `length_bias.py` (refuses a model that does not match the
  recorded prediction file), `floor_equalization.py` (model parameter; special
  columns found by name).
- **Still to do:** eight further scripts, listed in `EXPERIMENTS_PLAN.md` and
  `RERELEASE_PLAN.md`, plus two paper tables with no reproducible generator at all
  (`viterbi_vs_marginal`, `lenbias-norm`).

### 2026-08-11: Open-source release SHIPPED and merged upstream

- **Outcome:** calibrated UniLID merged into github.com/Ahmetcanyvz/UNILID
  branch `release` (PR #1 merged by Ahmetcan, "lgtm!", 10:02Z; PR #2 with
  subsetting + docs restructure merged by the user, 16:57Z; upstream tip
  3867b1b). Apache-2.0 + weight notices LICENSE published via the merge.
  Weights public at huggingface.co/cmeister/unilid-1940 (v2 container +
  calibration.json + model card). Gates re-passed at every inference-path
  change (base and calibrated both 250,000/250,000 exact); final test count
  94. Full record, runbook, and decisions ledger: OPEN_SOURCE_STATUS.md
  (supersedes OPEN_SOURCE_HANDOFF.md).
- Post-design additions beyond OPEN_SOURCE_DESIGN.md: forward= marginalized
  scoring (Ahmetcan's, guarded to base mode), language subsetting (load-time
  languages= + unilid-calibrate subset; carried thresholds fire at most as
  often as calibrated, re-estimation optional), add-language worked example
  with committed toy data, lazy training imports (prediction-only installs),
  Python floor 3.9.

### 2026-08-10: Open-source release verification gates PASSED (login node, no SLURM)

- **Purpose:** OPEN_SOURCE_DESIGN.md section 5 blocking gates for the calibrated
  UniLID package (UNILID/ branch calibration-release): does the package reproduce
  the analysis chain's predictions?
- **Golden subset:** test half of the seed-42 500k draw (odd positions), 250,000
  lines; texts from config.TEST_FILE; references full_test_eval/pred_baseline.npy
  and pred_gate_flat4_prox21.npy on store. Runner: analysis/release_gates.py
  (RAYON_NUM_THREADS=32).
- **Base gate (exact equality required): PASS, 250,000/250,000** identical to
  pred_baseline.npy. Artifact: outputs/release/gate_base.json.
- **Calibrated gate (>= 99.9% + boundary-case forensics required): PASS,
  250,000/250,000 (agreement 1.000000, zero disagreements).** Model: the v2
  release bundle. Artifact: outputs/release/gate_calibrated.json.
- **Release artifacts:** outputs/release/calibration_glotlidc.json (built by
  analysis/build_release_calibration.py from tau_floor21_gate.csv,
  tau_flat4.csv, glotlid_train_counts.json; source CSV sha256s in provenance;
  value-level round trip verified) and
  /capstor/store/.../release/unilid-1940-calibrated.unilid (779,663,390 bytes;
  differs from glotlidc.unilid in exactly the header version byte plus the
  calibration trailer; bundled JSON byte-identical to the transcribed artifact).

### 2026-08-10: glotlid_train_counts.json and glotlid_correct_test.txt migrated to durable store (login node, no SLURM)

- **Purpose:** two release-critical inputs existed only on purgeable scratch: the
  per-language training-line-count artifact (N_L, 1,940 entries, sum 60,683,151 =
  config.TRAIN_LINES; consumed by ~15 analysis scripts via `config.TRAIN_COUNTS_FILE`
  and by the open-source calibration workflow) and the GlotLID-C test pool of record
  (`config.TEST_FILE`, 45,627,279 lines, backing y_true.npy and every pred_*.npy).
  Found during release-preparation artifact verification (open-source session).
- **Moved** to `/capstor/store/cscs/swissai/a0229/cmeister/unilid_analysis/glotlid_unilid/`:
  `glotlid_train_counts.json` (40,531 bytes, sha256 `77afede8...`) and
  `glotlid_correct_test.txt` (7,143,311,471 bytes, sha256 `8125d817...`); both sha256
  verified identical after copy; scratch paths replaced with symlinks to the store
  copies, so `config.TRAIN_COUNTS_FILE` and `config.TEST_FILE` resolve unchanged.

### 2026-08-10: E3 artifacts of record migrated to durable store (login node, no SLURM)

- **Purpose:** move the Mistral-Nemo E3 artifacts off purgeable scratch before the
  ~2026-08-22 14-day purge window (user instruction 2026-08-10; unilid-durable-storage
  pattern of 2026-08-06).
- **Moved (scratch -> /capstor/store/cscs/swissai/a0229/cmeister/unilid_analysis/):**
  `glotlid_mistralnemo_fp64.unilid` (1,021,692,427 bytes),
  `full_test_eval_mistralnemo/` (416M; memmaps, banked top-5 arrays, fingerprints),
  `results_mistralnemo/` (16G; per-language tokenizers). 3,990 files total.
- **Procedure:** sequential login-node rsync; sha256 manifests of both sides compared
  (3,990/3,990 identical, VERIFY_OK); scratch originals then replaced with symlinks to
  the store copies, so all recorded scratch paths keep resolving. Reads through the
  symlinks spot-checked (UNILID magic bytes; fingerprint files listed).
- Checkpoint hygiene: scratch originals deleted only after the manifest comparison
  passed, per the move instruction; no other deletions.

### 2026-08-09: camera-ready edit pass applied (review-driven; two login-node measurements, no SLURM)

- **Purpose:** implement the 36-finding review (paper/review_notes_2026-08-09.md) with
  every text edit wrapped in `\camrev{}` (red) for the user's check. Plan:
  `~/.claude/plans/steady-finding-abelson.md`, user-approved 2026-08-09 after an Opus
  adversarial consistency review (21 findings folded in, 3 blockers caught: the filled
  FLORES subset F1 cell, the "all evaluations on the scored pool" overstatement, a false
  order-of-magnitude claim).
- **F1 (resource-tier N_test recomputation):** `analysis/regen_resource_tier_counts.py`,
  single source `outputs/diagnostic/paper_eval_per_lang_f1_fullpool.csv`; gates: support
  sum = 45,377,279, tier language counts = 56/40/458/526/398/462; artifact
  `outputs/tables/resource_tier_ntest.md` -> the six N_test cells of
  `paper/tables/resource-tier.tex` (2,513 / 5,222 / 552,346 / 1,151,363 / 1,125,105 /
  42,540,730; previous column summed to 45,627,279, the full test file).
- **F2 (baseline false-positive concentration):**
  `analysis/baseline_error_concentration.py`, Opus pre-run review (12 findings; NO-GO ->
  GO after: np.load loader for headered .npy, denominator = fp.sum() with equality gate,
  support-share null printed, G6 set-equality with tau_floor21_gate.csv). Gates G1-G6
  passed. Artifact `outputs/tables/baseline_error_concentration.md` -> the section 4
  sentence: 578,270 of 1,779,499 baseline false positives (32.5%) are predictions into
  the union of the under-18,000 group and the high-entropy four (1,084 languages, 3.8%
  of the scored pool by gold label; ratio 8.48).
- **Other artifact-backed cells/claims:** latency prose realigned to
  tables/latency_glotlid (0.307 -> 1.85x/1.64x) and latency_wili (0.155); Viterbi
  footnote/caption to tab:viterbi_vs_marginal (0.001 accuracy / 0.002 F1); D2 sentence
  re-pointed to tab:calibrated_views (0.515->0.780, 0.628->0.892); the 282-capped-language
  fact and the 26-of-1,080 threshold exclusions verified from
  paper_eval_per_lang_f1_fullpool.csv and tau_floor21_gate.csv in-session; 3.4e-4
  fastText pool-difference from the Camera-ready E1 entry.
- **Setup-doc correction:** EXPERIMENTAL_SETUP.md "data-driven quantiles" sentence
  corrected (ZH_MAGNET/ZH_EXTREME/MAGNET_RATIO_MIN are fixed constants in
  analysis/diagnostic.py; code over prose).
- Checkpoint hygiene: no deletions; new artifacts are the two outputs/tables mds.

### 2026-08-06: camera-ready evaluation program opened (E1-E4 pre-registered; no job yet)

- **Purpose:** incorporate the promoted configuration gate_flat4_prox21 into the
  ICML camera-ready (user decision 2026-08-06: paper items 1-5 plus experiments
  E1-E4). Plan item: `EXPERIMENTS_PLAN.md` "Camera-ready evaluation program".
  Setup: `EXPERIMENTAL_SETUP.md` "Camera-ready reporting conventions" and
  adoption-rule amendment 9 (judge part gains the appendix reporting role;
  Table 1 stays full-pool). Working plan with the adversarial-review
  corrections: `~/.claude/plans/steady-finding-abelson.md` (Opus review
  2026-08-06, 11 findings, all folded in).
- **Fallback mode active** (user instruction 2026-08-06): proceed as though the
  paper team's files are unavailable. E2 rebuilds UDHR/FLORES from public
  sources with a 0.005 baseline-reproduction gate; E3 retrains the
  Mistral-Nemo variant (pre-authorized).
- **Data recovery:** `fasttext_y_pred.txt` and `glotlidc_y_pred.txt` on scratch
  were found to be broken symlinks (targets deleted with the old home-dir
  cleanup); the three recorded Google Drive folders were verified live
  2026-08-06 (both prediction files present at 410,645,511 bytes =
  45,627,279 lines x 9 bytes, plus the paper team's metrics/per-language
  JSONs). Recovery and import gates are pre-registered in the plan entry.
- **paper/submission.tex tracked** in its submitted state (commit 5cfbda4,
  user decision superseding the earlier keep-untracked instruction).
- **Checkpoint hygiene:** no deletions proposed; new artifacts are int16
  memmaps (~87 MB each) and small banked arrays for external benchmarks.

### 2026-08-07: provenance of the paper's appendix breakdown tables resolved (login-node diagnostics, no job)

- **Question:** the paper's script table (tab:script-breakdown) and resource-tier
  table (tab:resource-tier) did not reproduce under the global per-language view
  (Hebr 0.740 vs 0.6966; resource bins off in 5 of 6 rows). Which view produced
  them?
- **Answer, measured:** both tables are the WITHIN-STRATUM view (examples
  restricted to true labels inside the group, cross-group false positives
  excluded; the Exp 24 distinction). Under that view every cell reproduces from
  `pred_baseline.npy` to the printed precision: resource bins 0.8709 / 0.9748 /
  0.9897 / 0.9972 / 0.9919 / 0.9583 vs printed 0.871/0.975/0.990/0.997/0.992/
  0.958; script rows including the previously unexplained Cyrl 0.8774 -> 0.877
  and Hebr 0.7401 -> 0.740. The script table's "Other 82" basis excludes
  jpn_Jpan and kor_Hang (dropping them gives 82 languages, mean 0.9374 ->
  printed 0.937); total basis 1,938 = 1,940 minus those two.
- **Consequences:** (i) E4's reproduction gates are re-pointed at the
  within-stratum references, which now match exactly; (ii) the camera-ready
  appendix tables must state their view, and the promoted configuration's rows
  must be shown in both views, because its tail gain is a global-view fact and
  the within-stratum tail row moves the other way (the Exp 24 reversal, and
  the draw-201 confirmation pattern); (iii) the corresponding Ahmetcan ask
  (script-table basis) is resolved and can be dropped from the ask list.

### 2026-08-09: variant appendix comparison added to the paper (user-confirmed recommendation); discrepancy record consolidated

- **Paper:** appendix paragraph "Transfer to a pretrained-vocabulary
  variant" + `paper/tables/calibrated_nemo.tex` (tab:calibrated_nemo) added
  before "Remaining errors"; the published Table 1 variant row is untouched.
  Artifact -> cell mapping: all six table cells and the +0.050
  [+0.044, +0.057] interval from `outputs/tables/mistralnemo_eval.md`; the
  +0.041 vs +0.028 gain comparison derives from `paper_eval.md` (0.9292 ->
  0.9569) and `mistralnemo_eval.md` (0.9132 -> 0.9538); the high-entropy
  group bjn/sco/srp from `outputs/diagnostic/mistralnemo_flat_set.csv`; the
  two below-constant languages (khm_Khmr, ory_Orya) from
  `fingerprint_floor21_mistralnemo.json`.
- **Records:** the consolidated CLD3-subset discrepancy table (which printed
  right-side cells reproduce under which convention, and which do not) is
  written into the "Camera-ready E2" results entry with the closed-set
  interpretation and the implemented user decisions.

### 2026-08-09: E3 finished (topk job 3038358 COMPLETED in 3h13m; eval on the login node)

- **topk:** 2,098,722 affected lines banked under the variant's floor-21
  matrix; identity check (banked rank-1 == pred_nemo_floor21 at every banked
  line) passed; zero short candidate lists.
- **eval:** all gates passed (sentinel, identity, split re-derivation, sha
  bindings). Results of record: `EXPERIMENTS_RESULTS.md` "Camera-ready E3".
  Full pool: nemo_baseline 0.9132 / FPR(x1e5) 1.7927, nemo_floor21 0.9396 /
  1.7139, nemo_gated 0.9538 / 1.5588. Judge part: 0.8968 / 0.9278 / 0.9473;
  bootstrap (gated minus baseline) +0.0504 [+0.0438, +0.0573]. Retrain
  comparability: full-pool baseline +0.0012 from the paper's printed .912
  cell (different line sets noted). Verdict: the calibration mechanisms
  transfer across vocabularies with a larger gain than on the base model.
- **Checkpoint hygiene / purge safety:** the variant's artifacts of record
  (`glotlid_mistralnemo_fp64.unilid`, `full_test_eval_mistralnemo/` memmaps,
  banked arrays, fingerprints, `results_mistralnemo/` per-language
  tokenizers) live on scratch; store migration with scratch symlinks is
  pending (14-day purge window from 2026-08-08/09; migrate before
  2026-08-22, login-node chunked, no SLURM for transfers).

### 2026-08-09: calibration prose accuracy review after the user's hand edits; CLD3-subset F1 cells filled; user decisions recorded

- **User decisions (2026-08-09):** (1) the calibrated row's CLD3-subset F1
  cells are filled under the restricted-lines convention that reproduces the
  printed UniLID GlotLID-C and UDHR cells (.975 / .986 / .992 from
  `paper_eval_cld3_subset.md` and `paper_eval_cld3_subset_external.md`);
  subset FPR cells stay dashed; the existing fastText cells stay as printed
  (our measured fastText GlotLID-C subset value under the same convention is
  0.9767 vs the printed .990, reported to the user). (2) No further trimming:
  the compiled paper fits exactly; the arXiv version has no limit. (3) The
  disabled author notes stay in the source. (4) On the retrained Mistral-Nemo
  variant: recommendation delivered (appendix comparison, existing Table 1
  row untouched); awaiting the user's confirmation.
- **Accuracy review (Opus) of the calibration prose** against the record,
  requested because the user's jargon-reduction edits may have introduced
  errors. 17 findings; the substantive ones, all fixed (commit c7ef48e):
  the reassignment rule had become "reassign to the next language whose tau
  exceeds the margin" (tau plays no role in acceptance; corrected to the
  recorded ranks-2-to-5 walk with the 100,000-sample and 21-nat conditions);
  the re-examination scope had lost the under-18,000 / flat-four restriction;
  "we derive these constants using a validation set" was wrong for most
  constants (replaced with the disjoint-data statement + appendix pointer);
  the q_L formula was stated without its under-18,000 domain; the
  "held-out subset outside every selection" claim over-reached for
  RES_CAP=100,000 (selected on the full post-draw pool before the seed-301
  split existed; qualified in four places); the shared-threshold alternative
  now names the lowered replacement bar; "outperforming every system" scoped
  to full-label-coverage systems; the tail-gain citation re-pointed from
  tab:resource-tier (within-stratum view, where the small tiers decrease) to
  tab:calibrated_views; the unmeasured "mechanisms rarely fire" replaced with
  the measured FLORES accounting (1,484 of 192,280 examined, 799 moved);
  Table 1 caption now names its instrument and states the FLORES subset-cell
  convention mismatch honestly; \defn restored; the stale
  `paper/draft_new_sections.tex` deleted. The appendix mechanism
  specification and every checked number were verified accurate.

### 2026-08-08: E3 chain progressing (jobs 3032625 done, 3036767 failed-and-fixed, 3036829 submitted); flat set derived; paper trimming committed

- **Job 3032625 (baseline+calibval): COMPLETED** in 2h14m; all 45.6M lines
  baseline-scored into `full_test_eval_mistralnemo/`, 250,000 calibration-half
  lines scored, 0 empty.
- **flatrule (login node):** the variant's flat set by the recorded rule is
  THREE languages: bjn_Latn (zH 2.50, magnet_ratio 2.29), sco_Latn (2.56,
  2.49), srp_Latn (1.97, 3.08); 90 languages flag is_magnet overall.
  Differs from the base model's four (shares sco/bjn; drops arg/vls; adds
  srp_Latn). Recorded before use: `outputs/diagnostic/mistralnemo_flat_set.csv`.
- **Job 3036767 (tau): FAILED at 29 s by a gate misfire, fixed, resubmitted
  as 3036829.** `build_equalized_weights` modified 1,938 of 1,940 rows and
  the inherited `n_mod == n_lang` assertion fired. Diagnosis: two healthy
  rows have natural floors already below the target (khm_Khmr -21.232, 435
  above-minimum entries; ory_Orya -21.016, 6,433), which the recorded
  downward-clamp mechanism (Exp 20: min(floor_L, F), nothing raised)
  correctly leaves unchanged; the base model had no such row so the
  precedent gate never distinguished "set" from "clamp". Fix: the gate now
  asserts the precise invariant (every unmodified row's natural floor is at
  or below the target; anything else aborts) and records the skipped rows in
  `fingerprint_floor21_mistralnemo.json`.
- **Paper:** trimming items 2-5 plus the user's own shortenings committed
  (9839c26) after ref/label integrity checks (no undefined refs; active
  labels unique; the moved table appears once).

### 2026-08-08: E3 evaluation pipeline committed and the baseline+calibval job submitted (job 3032625)

- **Pipeline:** `analysis/mistralnemo_eval.py` (six stages: baseline / calibval /
  flatrule / tau / topk / eval) + three SLURM scripts, Sonnet-implemented,
  Opus pre-run review (16 findings), fixes applied by a verify-then-apply
  agent with per-fix evidence after a session-limit interruption left the
  first application partial. The review empirically confirmed the flat-rule
  construction reproduces the recorded flat four on the base model's
  artifacts.
- **Job 3032625** (`slurm_mistralnemo_baseline.sh`, infra01, normal, 100G,
  09:00:00): full-pool baseline scoring of `glotlid_mistralnemo_fp64.unilid`
  into `full_test_eval_mistralnemo/`, then the 250,000-line retired-val pass
  (calibval) for the flat-rule inputs. Subsequent stages: flatrule (login),
  tau (SLURM), topk (SLURM), eval (login), each gated on the prior stage's
  fingerprints.
- **Agent-ops note (user feedback, recorded in memory):** two worker agents
  resumed via SendMessage after stops ran on the session model instead of
  their explicit Sonnet override; both were stopped and relaunched fresh with
  explicit models. Standing rule: stopped workers get fresh Agent calls, not
  SendMessage resumes.

### 2026-08-07: E5 run and finished (job 3031609 + login-node eval); E3 training completed (job 3028465) with the degeneracy scan adjudicated

- **E5 (job 3031609, COMPLETED in 3m50s):** score stage passed both wiring
  gates at EXACT equality (1.000000 agreement with Exp 39's persisted
  baseline and floor-21 predictions); eval stage reproduced 0.8452/0.7228/
  0.8491/0.7181 through the new tag-mapping path, then measured the gated
  configuration: accuracy 0.8604, tag-level macro F1 0.7149; out-of-set rows
  32,901 -> 25,884. Results entry "Camera-ready E5"; integrated into the
  paper as tables/commonlid.tex + one Results sentence. Artifact -> cell
  mapping: all cells from `outputs/tables/commonlid_calibrated.md`.
- **E3 training (job 3028465, COMPLETED in 4h33m10s,** matching the 131k
  Apertus precedent 4h36m): per-language training, packing to
  `glotlid_mistralnemo_fp64.unilid`, and the in-job degeneracy scan all
  completed. Scan verdict: 32 of 1,940 rows flagged, ALL in minority scripts
  (Ethiopic, Canadian syllabics, Syriac, Tibetan, Cherokee, Coptic, Gothic,
  Lisu, and similar). Adjudication against the record
  (`outputs/tables/degenerate_rows.md`): the Apertus 131k and 200k models
  flag nearly the same rows (18 and 17) with near-identical estimated-token
  counts (e.g. arc_Syrc 29 in both 131k and Mistral-Nemo), and the recorded
  per-language F1 for such rows is 0.977 to 1.000 except within-script
  confusion pairs (arc_Syrc 0.000 in the 131k record). Mechanism: base-vocab
  script coverage caps the estimable entries for these scripts; NOT the
  azj_Latn-class trainer failure (which shows 7 entries on ordinary Latin
  prose and appears in no fp64 model). Mistral-Nemo adds ~14 rows over the
  Apertus set, mostly Ethiopic and Tibetan: its vocabulary covers those
  scripts worse. Recorded as a model property; evaluation proceeds.
- Session restart note: the two job watchers and the paper-trimming agent
  were orphaned by a Claude Code process restart; jobs were unaffected, the
  trimming agent was resumed from its transcript.

### 2026-08-07: E3 retrain launched (job 3028465)

- **Job 3028465** (`slurm_mistralnemo_train_fp64.sh`, infra01, normal, 64 CPU,
  400G, 12:00:00): per-language fixed-vocabulary EM retrain of all 1,940
  languages against the Mistral-Nemo-Base-2407 tokenizer, the E3
  pre-registration's pre-authorized path (no model arrived from the paper
  team). Pipeline: the recorded Apertus fp64 invocation reused EXACTLY
  (`UNILID/train.py --initial-vocab <tokenizer> --vocab-size 131072
  --byte-level --per-lang-counts-method sp --max-base-samples-per-lang 10000
  --lang-batch-size 20 --reuse-corpus --skip-existing-langs`, then
  `convert.py`, then the degeneracy scan), differing from job 2903767 only in
  the tokenizer, results dir (`results_mistralnemo/`), packed name
  (`glotlid_mistralnemo_fp64.unilid`, flat under scratch per the Apertus
  convention), and `--fasttext` omitted (train.txt no longer exists;
  `--corpus-dir` + `--reuse-corpus` is a sufficient input per train.py's own
  argument logic, verified). Tokenizer pinned: snapshot
  a4477a2f977929a969745b69bbd62e03043551a5, tokenizer.json sha256
  e11c71726323d33da7b8d6f6f269f1988931c0a52b7122bcdd8c05042974e0db, vocab
  131,072, byte-level BPE, all four special tokens present. Preflight passed
  (corpus 1,940 files, 60,683,151 lines).
- **Decisions on the preparation agent's flagged items:** the agent's
  in-process sweep driver (`mistralnemo_train_sweep.py`) is NOT used; the
  recorded train.py orchestration is (fidelity to the recorded pipeline over
  new code on the critical path). Packed model placed flat under scratch,
  matching the Apertus convention (`mistralnemo_constants.py` updated). The
  stale `degeneracy_scan.py` MODELS dict is left for a separate fix; the
  standalone `degeneracy_scan_mistralnemo.py` runs in-job. Expected duration
  ~4.6 h (the 131k Apertus job, same vocab size); 12 h cap.
- **Checkpoint hygiene:** new artifacts are `results_mistralnemo/` (per-language
  tokenizers, expected tens of GB, scratch, regenerable) and the packed model
  (~1 GB); no deletions proposed. Store migration decided after the variant's
  evaluation round closes.

### 2026-08-07: Ahmetcan's label lists received and verified; CLD3-subset conventions measured

- **Received** (`unilid_resources/`): full label sets for GlotLID-C (1,940),
  UDHR (366), FLORES-200 (190) and the CLD3-coverage subsets (83/80/77, bare
  ISO codes). All three full sets are EXACTLY equal to our reconstructions,
  independently confirming the E2 eval files as the paper's basis before the
  scoring job returns. Not received: the Mistral-Nemo model (E3 retrain path
  stands), the preprocessed eval files (ours now confirmed equivalent), the
  eval script (still needed, see below).
- **CLD3-subset convention, measured:** mapping each of the 83 bare codes to
  its largest-training-corpus lang_Script variant and computing per-language
  F1 on the kept lines whose true label is one of the 83 (23,293,775 lines)
  reproduces the paper's printed UniLID cell (0.9719 -> .971). Under that
  convention: baseline 0.9719, gate_flat4_prox21 0.9751, fasttext 0.9767
  (`outputs/tables/paper_eval_cld3_subset.md`). NOT resolved: the printed
  fastText subset cell (.990) is reproduced by none of the tested conventions
  (restricted-lines 0.9767; global-view 0.9719, exactly matching the paper
  team's own per-language JSON), and the printed subset FPR (1.63e-4) is
  reproduced by neither the restricted-pool (9.71e-5) nor the global-pool
  (7.77e-5) definition. The submission's right-side columns appear to mix
  conventions per system; flagged for the authors, eval script requested.

### 2026-08-07: E2 scoring submitted (job 3028291); paper items 2, 3, 5 and the appendix integrated into submission.tex

- **Job 3028291** (`slurm_external_bench.sh`, infra01, normal, 100G, 02:00:00):
  scores UDHR and FLORES-200 under the unmodified and floor-21 matrices with
  top-5 banking. Submitted after the user approved the mapping
  (2026-08-07) and after the bit-exact self-check. Eval stages run on the
  login node after it completes, acceptance gates first.
- **Paper integration (user approved the prose sample 2026-08-07):** the
  corrected training-floor sentence; the calibrated-configuration subsection
  (sec:calibration) at the end of the method section; the appendix
  (app:protocol: development protocol, provenance table, held-out comparison
  and bootstrap tables, both-views resource table, other-instrument
  measurements, development alternatives, remaining errors); the Discussion
  future-work paragraph; view statements added to the two pre-existing
  appendix table captions. Table 1 row and abstract/intro claims remain
  pending E2 cells.
- **Artifact -> cell mapping (number-tracing rule):** subsection floor-only
  0.912 -> 0.930 from `outputs/diagnostic/mixed_eval_judge_f1_bgfloor.csv`
  (means 0.9117 / 0.9300); full-pool 0.929 -> 0.957, FPR 2.03e-5 -> 1.77e-5
  and the held-out table + bootstrap CIs from `outputs/tables/paper_eval.md`
  and `paper_eval_appendix.tex`; tail 0.332 -> 0.732 from
  `outputs/tables/paper_eval_tail_note.md`; both-views table from
  `outputs/tables/paper_breakdowns_resource.tex`; draw-201 0.981 -> 0.978
  from `outputs/tables/gate_flat4_prox21_confirmation_201.md`; CommonLID
  -0.005 from Exp 39; development alternatives from the Exp 28/46/47 records;
  remaining-errors numbers from `outputs/tables/promoted_residual.md`.
  Internal configuration names do not appear in the tex (audited).

### 2026-08-07: E4 run and finished; E2 gate machinery self-checked (login node, no job)

- **Runs:** `python -m analysis.external_bench_eval --stage selfcheck` (the E2
  gate reimplementation replayed the 2,236,864 banked rows and reproduced
  `pred_gate_flat4_prox21.npy` on all 45,627,279 lines, 0 differ; group A
  267,681 examined / 229,769 moved, group B 85,463 / 84,545), then
  `python -m analysis.paper_breakdowns --part all` (both reproduction gates
  PASSED under the within-stratum view). Scripts at commit 9b1ed20 after the
  combined Opus review (16 findings, all applied by a fix agent and
  spot-checked).
- **Results of record:** `EXPERIMENTS_RESULTS.md` "Camera-ready E4". Both-views
  breakdown tables written; residual of record for the promoted configuration:
  926,299 wrong on the judge part, head-true 99.15%, head-head 88.64%, top
  pair ind->zsm 31,113, the eng->sco pair absent (flat-four re-examination).
- **E4 incident note:** an earlier accidental end-to-end test run of
  `paper_breakdowns.py` by the implementing agent (outputs deleted, nothing
  recorded) surfaced the view question that led to the provenance resolution
  above; recorded here for completeness.
- **E2 status:** scoring awaits the user's mapping approval; the SLURM script
  `slurm_external_bench.sh` is ready and not submitted.

### 2026-08-07: E1 run and finished (login node, no SLURM job)

- **Runs:** `python -m analysis.import_external_pred --source fasttext` (passed
  all gates), `--source glotlidc_file` (blocking gate correctly refused: the
  Drive file is bit-identical in content to `pred_baseline.npy`, 1.000000
  kept-pool agreement, while the pickle holds the paper-era scoring run at
  0.99514 agreement; `pred_glotlidc_file.npy` quarantined from use), then
  `python -m analysis.paper_eval` (all five gates passed). Code at commit
  02a346e after the Opus pre-run review (11 findings, all applied).
- **Results of record:** `EXPERIMENTS_RESULTS.md` "Camera-ready E1". Full pool:
  baseline 0.9292 / FPR(x1e5) 2.0263, gate_flat4_prox21 0.9569 / 1.7665,
  fasttext 0.9443 / 2.7063. Judge part: 0.9117 / 0.9498 / 0.9332. Bootstrap
  (judge): promoted minus baseline +0.0380 [+0.0328, +0.0434], promoted minus
  fasttext +0.0166 [+0.0112, +0.0223].
- **E2 data (2026-08-07, no scoring):** UDHR rebuilt from `cis-lmu/udhr-lid`
  (366-label exact intersection, matches the paper's count) and FLORES from
  the original FLORES-200 devtest tarball (190-label exact intersection,
  matches; flores_plus gave 205 and is diagnostic-only); details and revisions
  in `outputs/tables/external_bench_mapping.md`. Scoring awaits the user's
  mapping approval.

### 2026-08-06: gate_flat4_prox21 promoted; draw-201 confirmation recorded (no job)

- **No SLURM job.** User decision 2026-08-06: gate_flat4_prox21 promoted on
  the natural track after Exp 47-50, superseding floor21_gate (remains in
  the pool). Judge-part overall F1 0.9498, +0.0018 [+0.0010, +0.0026] over
  floor21_gate, zero supported collapses (Exp 49). The pre-registered
  composed step of Exp 50 (rebuilding the gate on the pooled-frequency floor
  matrix) is skipped by the same decision. Confirmation script ran as a
  memmap subset over the already-scored full-pool prediction memmaps
  (`/capstor/scratch/cscs/cmeister747/unilid_analysis/full_test_eval`); no
  scoring pass, no job submitted.
- Artifact: `outputs/tables/gate_flat4_prox21_confirmation_201.md`.
  Instrument: balanced test draw, seed 201, 185,204 lines. Within-stratum
  macro-F1: gate_flat4_prox21 overall 0.9781, tail 0.8763, magnets 0.8811,
  twins 0.9437, head 0.9814; baseline overall 0.9809, tail 0.9086, magnets
  0.9121, twins 0.9435, head 0.9817. Sanity gate (baseline overall row vs
  the recorded headline in `outputs/tables/two_sided_selection.md`): PASS,
  max abs diff 0.00004. Per-language collapse check on the draw-201 subset:
  3 supported collapses (support >= 10, F1 drop > 0.10), attributable to the
  draw's own per-language support cap of at most 100 lines; the
  promotion-gate clause (C), computed on the far larger judge part in Exp
  49, remains clean at zero supported collapses.

### 2026-08-06: Exp 47-50, the four candidate directions in order (jobs 3014614, 3015805, 3016337, 3016380)

- **3014614** gate_variants topk plus apply shared9_bar18k (Exp 47, 07:06):
  candidate arrays banked, shared-threshold variant built; verdict in the pool
  (best aggregate 0.9534, class-level clause-C fail, 9 collapses).
- **3015805** apply flat4_tau5 (Exp 48, 00:33): eligible, 0.9486, zero
  collapses; four flat large-corpus languages and their neighbours all gained.
- **3016337** apply flat4_prox21 (Exp 49, 00:18): in-job reproduction gate
  passed; eligible, 0.9498, the strongest eligible configuration; refinement
  contrast +0.0012 [+0.0007, +0.0016] over gate_flat4_tau5.
- **3016380** `slurm_full_test_bgfloor.sh` (Exp 50, running): full-pool scoring
  under the pooled-frequency unseen-token values (c = -8.4740; assigned plateau
  range -27.61 to -12.31; alignment of the base distribution verified four
  independent ways in review). Output pred_bgfloor.npy; first verdict is the
  gate-less judge-part comparison against floor21 solo.
- All four scripts Opus-reviewed before their runs; every verdict recorded in
  EXPERIMENTS_RESULTS.md Exp 47-50. Checkpoint hygiene: no deletions; new
  artifacts are three candidate arrays, four prediction memmaps, tau_flat4.csv.

### 2026-08-05: Exp 47 submitted (job 3014614): shared re-examination threshold

- **3014614** `slurm_gate_variants.sh`: 64 CPU, 100G, 03:00:00. Stage one saves
  the top five candidate languages and scores under the floor-21 matrix for the
  2,236,864 kept lines whose floor-21 prediction is a language with under
  18,000 training lines or in the flat-distribution category (the saved arrays
  also serve Experiments 48 and 49 with no further scoring). Stage two builds
  the Experiment 47 variant: one shared threshold of 9.0, replacement-candidate
  minimum 18,000 training lines. Pre-registration: EXPERIMENTS_RESULTS.md
  "Experiment 47 pre-registration". Code `analysis/gate_variants.py`, Opus
  review found two blockers (a label-set inflation to 12.0M lines and a false
  timing claim), both fixed with nine hardening items before submission.
- Directions 1 through 4 are being tried in order (user 2026-08-05).
  Checkpoint hygiene: no deletions; new artifacts are the three candidate
  arrays (about 130 MB total) and one prediction memmap per variant.

### 2026-07-30: Exp 46 mixed-matrix scoring submitted (job 2932154); Exp 44/45 completed

- **2932154** `unilid-mixed-matrix`: 64 CPU, 100G, 06:00:00. Four fail-fast
  stages: the pre-registered no-op scorer check (chunk 0 under W must be
  bit-identical to pred_baseline.npy), stage A full-pool scoring under the
  rule-v1 mixed matrix (sha 0c31f143..., 860 unmod rows + 1,080 floor-21 rows,
  fingerprint_mixed.json), stage B adaptive gate (tau recalibrated under the
  mixed matrix, tau_mixed.csv+json provenance binding), stage B_solotau (tau
  read from tau_floor21_gate.csv, isolating the tau-recalibration component).
  Outputs pred_mixed_nogate.npy, pred_mixed.npy, pred_mixed_solotau.npy.
  Hypotheses and criteria: EXPERIMENTS_RESULTS.md "Experiment 46
  pre-registration". Code reviewed (Opus, no blockers) with four hardening
  fixes applied before submission; `analysis/mixed_matrix.py`.
- Exp 44 (evidence base, seed-301 split) and Exp 45 (solo-gate references,
  jobs 2930701/2930702, 15:23 and 13:48; floor21_gate judge-part evaluation)
  recorded in EXPERIMENTS_RESULTS.md. User decisions 2026-07-30: rule v1
  signed off; bootstrap anchor switched to floor21_gate (condition met);
  clause-(A) cap question deferred to Exp 46 results; amendment scope
  confirmed (judge part is the confirmation instrument for
  derivation-informed candidates).
- Checkpoint hygiene: no deletions; three new ~91 MB prediction memmaps
  expected on scratch.

### 2026-07-30: floor21_gate promoted; draw-201 confirmation recorded (no job)

- **No SLURM job.** User decision 2026-07-30: floor21_gate promoted on the
  natural track after Exp 44-46 and amendment 8, superseding floor-21's
  provisional adoption and gt_margin_adaptive's configuration-to-beat status
  (both remain in the pool). Confirmation script ran as a memmap subset over
  the already-scored full-pool prediction memmaps
  (`/capstor/scratch/cscs/cmeister747/unilid_analysis/full_test_eval`); no
  scoring pass, no job submitted.
- Artifact: `outputs/tables/floor21_gate_confirmation_201.md`. Instrument:
  balanced test draw, seed 201, 185,204 lines. Within-stratum macro-F1:
  floor21_gate overall 0.9741, tail 0.8685, magnets 0.8758, twins 0.9425, head
  0.9813; baseline overall 0.9809, tail 0.9086, magnets 0.9121, twins 0.9435,
  head 0.9817. Sanity gate (baseline overall row vs the recorded headline in
  `outputs/tables/two_sided_selection.md`): PASS, max abs diff 0.00004.
  Per-language collapse check on the draw-201 subset: 8 supported collapses
  (support >= 10, F1 drop > 0.10), attributable to the draw's own
  per-language support cap of at most 100 lines; the promotion-gate clause
  (C), computed on the far larger judge part in Exp 45/46, remains clean at
  zero supported collapses.

### 2026-07-29: combined-method plan reviewed twice, amended, and started (no jobs yet)

- **No SLURM jobs.** Two adversarial Opus reviews of the implementation plan for
  the combined method (plan file
  `~/.claude/plans/steady-finding-abelson.md`): a mechanics review and a
  design-consistency review against all prior recorded decisions. Findings and
  the resulting user decisions are recorded in `EXPERIMENTS_RESULTS.md` "Current
  state (2026-07-29)" and as amendment 7 plus standing constraint 5 in
  `EXPERIMENTAL_SETUP.md`; the pre-registration amendments are recorded at the
  top of the combined-method section in `EXPERIMENTS_PLAN.md`.
- Key measured finding recorded as a result about the instruments: balanced
  draw 101 reverses the method ranking for the primary quantity (six of six
  group leaders disagree with the held-out remainder; gt_min leads every group
  on the draw), so the assignment rule derives from the seed-301 40/60
  remainder split instead (new pre-registered constants RULE_SPLIT_SEED=301,
  RULE_SPLIT_FRACTION=0.40, BOOT_B=10,000, BOOT_SEED=0).
- Execution started with the no-scoring steps: the feature-provenance artifact
  (`outputs/tables/combined_feature_provenance.md`), the two solo-gate
  reference builds (`analysis/solo_gates.py`: adaptive gate, 100k target bar,
  over `pred_baseline.npy` and `pred_floor21.npy`), and the evidence base
  (`analysis/combined_evidence.py`, derivation part only). Both scripts passed
  an Opus pre-run review (no blockers; seven should-fixes applied before any
  run, including the sha256_base_W provenance gate on the unmod branch, the
  memmap provenance table, the fixed contrasts against gt_margin_adaptive, and
  the carried-set oracle). No scoring before the step-2 user checkpoint (rule
  sign-off).
- **Solo-gate builds: jobs 2930701 (unmod) and 2930702 (floor21)**, each 1
  node, 64 CPU, 100G, 04:00:00, submitted 2026-07-29 after a login-node
  attempt was killed at exit 137 (memory) during the 2,175,310-line affected
  scoring stage; the affected count matched the review's predicted volume, so
  the failure is resource-side, not code-side. Expected run time about 15
  minutes each once scheduled (precedent: job 2895821, 00:14:11).
- Checkpoint hygiene: no deletions proposed. Upcoming artifacts: two solo-gate
  prediction memmaps (~91 MB each), later three mixed-configuration memmaps,
  and the split index file on scratch
  (`full_test_eval/rule_split_seed301.npz`, ~360 MB, regenerable from the
  seed). The `results_apertus200k/corpus` calibration files and the
  `full_test_eval/` memmaps must be re-touched before the scoring job (14-day
  scratch purge).

### 2026-07-27: Exp 43, clean re-measurement of the Apertus 131k branch (job 2911700)

- **2911700** `unilid-ft-131k-fp64`: COMPLETED, 02:06:08. Full-test baseline
  evaluation of `glotlid_apertus131k_fp64.unilid` against the 100k model, into the
  separate scratch dir `full_test_eval_131k_fp64`. Report
  `outputs/tables/full_test_eval_131k_fp64.md`, per-language values in
  `outputs/diagnostic/full_test_131k_fp64_per_lang_prf.csv`.
- Code: `analysis/full_test_eval_131k.py` parameterized over model path, scratch
  directory, outputs, and report label (reviewed: no writes reach the earlier
  scratch directories, no variable shadowing, fingerprint gate prevents mixing two
  models' predictions; one shadowing defect was caught and fixed before submission
  because the chunk loop already binds `label`).
- Outcome (`EXPERIMENTS_RESULTS.md` Exp 43): branch verdict HOLDS on clean
  evidence (within-stratum overall -0.0090, tail -0.0395 [CI -0.0476, -0.0322];
  global tail 0.4269 against the 100k model's 0.5618), with the magnitude
  overstated by the bug (false positives into tail 32,211 clean against 51,926
  corrupted, within 0.2% of the Exp 30 counterfactual prediction of 32,161).

### 2026-07-27: Exp 42, both fp64 retrains completed and verified

- **2903767** `unilid-apertus131k-fp64`: COMPLETED, 04:36:18. **2903768**
  `unilid-apertus200k-fp64`: COMPLETED, 07:28:27. Artifacts
  `glotlid_apertus131k_fp64.unilid` (1,021,914,972 bytes) and
  `glotlid_apertus200k_fp64.unilid` (1,559,144,154 bytes) on scratch; results
  dirs `results_apertus{131k,200k}_fp64`. Both under the patched trainer
  (fork commits d0208d9, c5921a2). Note: the 131k retrain took 4.6 h against
  9.8 h for the corrupted run, because the collapsed Azerbaijani row no longer
  wastes iterations.
- Post-training gate and effect measurement (`analysis/fp64_retrain_check.py`,
  `outputs/tables/fp64_retrain_check.md`): azj_Latn repaired in both models
  (131k: 7 -> 22,704 entries above the row minimum, entropy 1.609 -> 3.025;
  200k: 1,798 -> 31,210, entropy 2.473 -> 3.000, confirming the partial collapse
  diagnosed in Exp 41); the 17 minority-script rows unchanged in kind, which
  confirms the vocabulary-coverage class is not a numerical failure; no new
  degenerate rows; only 20 of 1,940 (131k) and 18 of 1,940 (200k) rows moved by
  more than 1 nat, concentrated in the long-line corpora the diagnosis predicted
  (quc_Latn 9.43 nats, pcm_Latn, fas_Arab, mam_Latn).
- Pending and specified, not run: a full-test baseline evaluation of the
  retrained 131k model to re-measure the Exp 29 branch verdict without the
  corrupted row (`EXPERIMENTS_PLAN.md` open item 1).

### 2026-07-26: fp64 trainer adopted; both Apertus models retraining (jobs 2903767, 2903768)

- Patch adopted into the sentencepiece fork (user directive): commits d0208d9
  (double-precision forward-backward in the trainer's E-step; inference paths
  untouched) and c5921a2 (hard CHECK on non-finite expected counts instead of
  silent zeroing), pushed to cimeister/sentencepiece branch fixed-vocab-em.
- Installed binary replaced (~/.local/bin/spm_train; backup spm_train.pre_fp64).
  Acceptance test: isolated azj retrain under the new binary yields 22,704
  above-minimum entries, entropy 3.025, floor -18.93 (was 7 / 1.609 / -27.63).
- Full retrains of BOTH Apertus models launched under the corrected trainer
  (2903767 = 131k, 2903768 = 200k; new results dirs and artifacts with the _fp64
  suffix; corpus split reused; originals kept as the Exp 15/29 record). Full
  rather than per-language retrains so each model has single-provenance weights.
  Post-completion gates: degeneracy scan on both new models, then a full-test
  baseline evaluation of the 131k_fp64 model to re-examine the Exp 29 branch
  verdict with the bug removed.

### 2026-07-26: Exp 40-41, oracle bound, per-tag CommonLID, EM bug diagnosed (multi-agent)

- Exp 40 (login node): oracle upper bound over the carried set, 0.9525 overall
  (+0.0191 over the best single configuration); headroom concentrated in tail
  (+0.0724) and flat_magnets (+0.0998). `analysis/oracle_bound.py`.
- Exp 39 extension (job 2903415): per-tag macro-averaged CommonLID F1 (agent-
  authored, reviewed; both baseline gates reproduced: 0.8452 accuracy, 0.7228 tag
  F1; per-line predictions persisted). Both carried leaders slightly NEGATIVE on
  the objective-consistent metric (floor-21 -0.0046, adaptive -0.0061): CommonLID's
  109 tags barely contain the repaired tail labels.
- Exp 41 (two agents: source analysis on a scratch clone of the sentencepiece
  fork; 14-run corpus bisection): the fixed-vocabulary EM bug fully diagnosed:
  float32 forward-backward breaks the log-posterior identity on very long lines
  (upstream's 4,192-byte cap masked it; the pipeline passes 1,000,000); the fork's
  isfinite-to-zero M-step guard turns the overflow into a silent collapse. Trigger:
  azj line 81,302 (142,136 bytes, longest in all corpora). Graded corruption in at
  least 94 Apertus-branch corpora; azj-200k partially collapsed; 100k production
  model unaffected (not trained through this path). Double-precision patch built
  and VERIFIED in scratch (unpatched rebuild bit-identical to the installed
  binary; patched azj healthy at objective 229.8); NOT applied anywhere
  (user decision). Artifacts: session scratchpad `em_debug/` (fix.patch, fb_sim.py,
  both build trees) and `em_bisect/` (minimal 390-line trigger file, run log).

### 2026-07-25: Exp 36-39, adaptive verdict, azj re-run, user decisions, carried-set and CommonLID checks

- Exp 36 (job 2895821): gt_margin_adaptive ELIGIBLE flagged (ota only); both
  pre-run predictions confirmed; floor-21 retains top rank 5/5 draws at a 0.0002
  margin. Exp 37 (login node): azj collapse reproduced byte-identically in
  isolation; deterministic numerical breakdown in the fixed-vocab EM fork.
- User decisions: near-tie co-selection (NEAR_TIE_BAND=0.001; six configurations
  carried forward), primary quantity = macro-averaged per-language F1 on
  natural-distribution test data (equal language weighting, extreme low-resource
  exemptions allowed); precise terminology definitions committed to memory.
- Exp 38 (login node): carried-set per-language comparison on the held-out
  remainder; complementary structure (adaptive leads overall 0.9334 via lowmid,
  floor-21 leads tail/magnets, learned_bias leads head/twins with 602 strict
  per-language wins). `analysis/carried_set_comparison.py`.
- Exp 39 (job 2898246; login-node attempt OOM-killed at exit 137, resubmitted via
  SLURM): CommonLID check of the carried leaders, floor-21 +0.0040 and
  gt_margin_adaptive +0.0070 over the reproduced 0.8452 baseline; gate portable
  without refitting (9,886 firings). `analysis/commonlid_carried.py`, reviewed
  with fingerprint and batch-length hardenings applied pre-run.

### 2026-07-25: Exp 34-35, round-3 verdict, EM-degeneracy bounded, adaptive variant launched

- **2895683** `unilid-gt-margin-100k`: COMPLETED 08:37. gt_margin_all_100k
  ELIGIBLE, flagged (single outlier ota_Arab); not selected (balanced-val overall
  0.9744 vs floor-21 0.9800); veto overall 0.9330 top-tier. ota dig-in: 395 new
  FPs are gt-weight-side high-margin flips (fas->ota 295), not reassignment;
  quantile gates cannot catch them. `EXPERIMENTS_RESULTS.md` Exp 34.
- EM-degeneracy investigation (user-raised): `analysis/degeneracy_scan.py`,
  `outputs/tables/degenerate_rows.md`: 0 flagged rows at 100k (main results
  clean), 17/18 near-identical sets at 200k/131k = deterministic vocabulary
  coverage (no multi-byte merges for those scripts; csw EM converges normally);
  azj at 131k is the single anomalous run of 3,880. Scan adopted as the
  post-training gate. Exp 15 magnitude caveat recorded. `EXPERIMENTS_RESULTS.md`
  Exp 35.
- Pre-registered `gt_margin_adaptive` (user-requested N-adaptive gate strength:
  q_L = MARGIN_Q * (1 - min(N,HEAD_N)/HEAD_N), target bar unchanged), built via
  `run(gate='nonhead', target_n=100_000, adaptive_q=True)`,
  `slurm_gt_margin_adaptive.sh`.

### 2026-07-25: Exp 33, gt_margin_all judged; round-3 launched (jobs 2895566, 2895683)

- **2895566** `unilid-gt-margin-all`: COMPLETED (2026-07-25 08:13). Verdict REJECTED
  on both tracks (4 barely-head collapses, worst llb_Latn -0.3211 precision-side)
  despite the best natural-veto aggregates of any candidate (overall 0.9331, lowmid
  FPs 451k -> 140k). Mechanism and the three-round reassignment law in
  `EXPERIMENTS_RESULTS.md` Exp 33.
- **2895683** `unilid-gt-margin-100k`: round-3 candidate `gt_margin_all_100k`
  (pre-registered in Exp 33 before the run; reassignment-target bar raised to
  RES_CAP=100,000, gated set unchanged; `run(gate='nonhead', target_n=100_000)`,
  `slurm_gt_margin_all_100k.sh`). Verdict via `two_sided_report` when complete.

### 2026-07-24: Exp 32, outlier-tolerant clause + victim dig-ins + round-2 launch (login node + SLURM)

- Clause (C) revised per user decision (MAX_LANG_COLLAPSE_OUTLIERS=2; dig-in
  instead of rejection for 1-2 outliers). Re-judged: learned_bias flagged-eligible
  (llb), gt_min = uniform-track champion flagged (mev/sbs), gt_margin still
  rejected (4 = class pattern). Commit series through Exp 31 pushed
  (a3ff2c2..046e147, 13 commits).
- Dig-ins (`analysis/victim_digin.py`, `outputs/tables/victim_digins.md`): every
  victim is FP inflow at a non-head label, never recall loss. 131k pathology: azj's
  row is a degenerate EM outcome (7 estimated tokens); 18 degenerate 131k rows vs 0
  at 100k; azj alone explains ~2/3 of the 131k FP-into-tail increase
  (counterfactual in Exp 32).
- Round-2 candidate `gt_margin_all` pre-registered (gate all N<18k labels) and
  built via `analysis/gt_margin.py run(gate='nonhead')`, `slurm_gt_margin_all.sh`.

### 2026-07-24: Exp 31, gating reconsideration + 131k error overlap + gt_margin build (login node)

- User-directed gating review. Amendments (EXPERIMENTAL_SETUP.md, pending final
  user confirmation): (B)-overall softened to a bounded drop; dual-track verdicts
  (natural-traffic + uniform-prior `passes_uniform`); ITERATE lane. Delta review:
  amendments verdict-neutral for the first round; seed-201 discipline preserved
  (one confirmed candidate per track; multiplicity note recorded).
- Dual-track run: floor-21 unchanged as natural-track champion; gt_min wins
  uniform-track selection but FAILS the balanced-test collapse confirmation
  (2 supported languages beyond 0.10, worst sbs_Latn -0.182), so the uniform track
  has no champion and the baseline holds there.
- Exp 30 (131k error overlap, `analysis/error_overlap_131k.py`): the 131k does not
  repeat the baseline's errors (57.7% shared; 42.3% fixed incl. Indic wins; 403
  languages regress; tat<-azj alone carries 17,603 FPs).
- Pre-registered composition `gt_margin` (gt_min weights + head-targeted margin
  gate, tau recalibrated under gt_min; `analysis/gt_margin.py`, reviewed, no
  defects) built on the login node; judged in the same dual-track report.

### 2026-07-24: Exp 28, gt_min full-test pass judged (job 2884210)

- **2884210** `unilid-full-test-gt`: COMPLETED (2026-07-24 01:37, ~2.2 h scoring).
  Verdict via `two_sided_report`: REJECTED (veto overall and tail/magnet global F1
  drop; FPs into tail 22,404 -> 79,113; 12 supported languages beyond the collapse
  bound) despite the best selection-view numbers ever measured (balanced-val overall
  0.9841, tail 0.9769; full-test within-stratum tail +0.0656 CI [+0.0603, +0.0729]).
  Floor-21 remains selected. Mechanism and the next-round composition hypothesis in
  `EXPERIMENTS_RESULTS.md` Exp 28. Artifacts: `outputs/tables/full_test_gt.md`,
  `pred_gt_min.npy`, `fingerprint_gt.json`.

### 2026-07-23: Good-Turing counting pass + margin diagnostic launched (plan B4/B3)

- **2883714** `unilid-gt-counts`: COMPLETED (2026-07-23 23:17, ~35 min run).
  Outcome (`EXPERIMENTS_RESULTS.md` Exp 27): plateau overstates unseen mass for
  all 1,940 languages (exact GT raises 0 rows; tail 9x, head 12x median
  overstatement); spot-checks exact. gt_min matrix built and gate-checked; scoring
  pass `analysis/full_test_gt.py` reviewed (no defects) and submitted as
  **2884210** `unilid-full-test-gt` (64 CPU, 100G, 6 h; pred_gt_min.npy +
  fingerprint_gt.json on the full-test scratch dir; verdict via
  `two_sided_report`, gt_min added to CONFIGS). Original submission record: plan B4
  prerequisite: per-language T, n1, plateau mass under each language's own Viterbi
  segmentation of its training corpus (`analysis/gt_counts.py`, resumable per
  language, review-fixed torn-line resume handling). Output
  `outputs/diagnostic/gt_counts.csv`; feeds the one-sided-min GT candidate
  (`full_test_gt.py`, to be written when counts exist). 64 CPU, 100G, 8 h.
- **2883715** `unilid-margin-diag`: COMPLETED (2026-07-23 22:39). Outcome
  (`EXPERIMENTS_RESULTS.md` Exp 26): VIABLE; FP catch 76.8%, test-side suppression
  6.7%, cascade 53 lines, per-language AUC 0.90-0.9998, 26 languages excluded.
  Follow-up candidate `margin_q5` (reassign to runner-up), built login-node by
  `analysis/full_test_margin.py` (reviewed; agreement 1.0000; 17,773 reassignments):
  REJECTED on clause (C), szy_Latn -0.107 via 82 reassigned pwn_Latn FPs (see
  Exp 26 addendum in EXPERIMENTS_RESULTS.md, mechanism verified from memmaps).
  Pre-registered final variant `margin_q5_head` (reassign to highest-scoring head
  candidate) recorded before its run, then built and judged same evening: ELIGIBLE
  (all stages pass, szy collapse gone) but not selected; floor-21 ranks higher on
  both instruments (val overall 0.9800 vs 0.9799; veto tail F1 0.6337 vs 0.5321).
  Margin family closed this round; composition path recorded in the plan. Original
  submission record: plan B3:
  margins on the 22,522 FP-into-tail lines, the 7,735 true-tail lines, and up to
  2,000 train lines per tail language; tau_L at the 5th percentile of self-won train
  margins; MIN_CALIB_LINES=200 exclusion, logged (`analysis/margin_diagnostic.py`).
  Outputs `outputs/tables/margin_diagnostic.md`,
  `outputs/diagnostic/tau_per_lang.csv`. 64 CPU, 100G, 2 h. Constants MARGIN_Q=5,
  MIN_CALIB_LINES=200, CALIB_MAX=2000, TOPK_MARGIN=5, CALIB_SEED=0 pre-registered in
  the approved plan.
- Both modules reviewed pre-launch (Opus adversarial pass: no defects that would
  produce a wrong headline number; fixed before submission: gt_counts torn-line
  resume gap, a diagnostic-column double-count, a hardcoded stratum size; the
  encode_batch-vs-scorer segmentation choice documented as deliberate).

### 2026-07-23: Exp 25, adoption-rule instruments + first verdicts + pnt/ell audit (plan B1/B2, no SLURM job)

- Login-node analysis, no new scoring. Code: `passes_shortlist`/`passes_two_sided`
  (`analysis/hierarchical_pool.py`), `build_test_draw`/`rebuild_stability_draws`
  (`analysis/balanced_split.py`), `analysis/two_sided_report.py`,
  `analysis/label_audit.py`; `balanced_sweeps.py` sweeps now shortlist. Reviewed
  pre-run (Opus adversarial pass: no correctness defects; two flags fixed:
  `run_bias_refit` guard -> shortlist, `balanced_split.__main__` de-pipelined).
- Instrument amendment at first run: the veto originally excluded all six balanced
  draws, leaving median ~1 true tail line per language (veto tail recall 0.2188);
  amended to exclude draws 101/201 only, with a conditional exclusion for candidates
  fit on stability draws and a median>=10 runtime gate. Delta review of the amendment
  (same agent): exclusion set sound for the four current configs, llb_Latn rejection
  confirmed genuine on the full pool (drop 0.111), freq_prior trace confirmed; one
  defect found and fixed: clause (C) now judges only languages with
  MIN_COLLAPSE_SUPPORT=10 true veto lines (at n=4 one line flip moves F1 by
  0.11-0.14 and false-trips the 0.10 bound), and the fit-draw conditional is
  enforced via `CONFIG_FIT_DRAWS`. Verdicts unchanged under the fixed clause.
- Outcomes (`EXPERIMENTS_RESULTS.md` Exp 25): floor-21 ELIGIBLE and selected
  (provisional adopted configuration; supersedes the Exp 20 recall-view verdict);
  freq_prior ELIGIBLE not selected; learned_bias reg=5.0 REJECTED on the
  per-language collapse clause (llb_Latn -0.113, n=4,181). Label audit: 50/50
  sampled pnt_Grek<-ell_Grek residual lines are standard Modern Greek (provisional),
  so that residual is model error, not label noise.
- Artifacts: `outputs/tables/two_sided_selection.md`,
  `outputs/tables/label_audit_pnt_ell.md`, `outputs/diagnostic/balanced_val/`
  (val_lines_seed201.npy new; seeds 102-105 regenerated, manifest annotated).

### 2026-07-23: Apertus 131k (preliminary_mul) retrain launched (plan Track A)

- **2883222** `unilid-apertus131k-train`: COMPLETED 2026-07-24 02:47 (~9.8 h total,
  inside the 12 h window; no resume needed, unlike the 200k run). All 1,940
  languages trained; `glotlid_apertus131k.unilid` packed (1,021,914,972 bytes).
  Evaluation: `analysis/full_test_eval_131k.py` (reviewed, no defects; one optional
  hardening applied) submitted as **2885941** `unilid-full-test-131k` (64 CPU,
  100G, 8 h): scores the b=0 baseline over the full pool into the separate scratch
  dir `full_test_eval_131k/`, reusing y_true read-only after a language-list gate;
  report `outputs/tables/full_test_eval_131k.md` +
  `outputs/diagnostic/full_test_131k_per_lang_prf.csv`. **2885941** COMPLETED
  2026-07-24 (~3.5 h): NEGATIVE on both views (within-stratum tail -0.0437
  [CI -0.0515, -0.0371], overall -0.0113; FPs into tail 22,522 -> 51,926; balanced
  val also lower). `EXPERIMENTS_RESULTS.md` Exp 29; branch discontinuation
  recommended, user decision pending. Original submission record: PENDING
  at submission (Resources), RUNNING on nid007559 within 8 minutes. Purpose:
  test whether a multilingual-focus 131k vocabulary reverses the Apertus 200k tail
  regression (-3.4pp, Exp 15); the tokenizer is documented in
  `~/apertus-tokenizer-development/README.md` as the balanced-multilingual candidate
  with the highest compression on Indic, Chinese, and the low-resource tail.
- Plan item: approved plan `~/.claude/plans/steady-finding-abelson.md` Track A;
  tokenizer choice (`preliminary_mul` over the stock Apertus tokenizer) fixed by the
  user 2026-07-23.
- Init: vocab seeding from
  `/users/cmeister747/apertus-tokenizer-development/preliminary_mul/tokenizer.json`
  (sha256 6f8c5ca267c94975081045a46686ae68f8a1335b70a104810904389272117d41, vocab
  131,072, BPE, NFC; specials `<unk>/<s>/</s>/<pad>` at ids 0-3), uniform Unigram
  init, per-language fixed-vocab EM (forked spm_train), standard setup as the 200k
  retrain (`EXPERIMENTAL_SETUP.md` Apertus retrain entry).
- Data: `train.txt` (60,683,151 lines) via the 200k run's per-language corpus split,
  reused read-only with `--corpus-dir` (preflight `analysis/preflight_131k.py`
  verified 1,940 files, line total exact, all checks passed pre-submission).
- Script: `slurm_apertus_train_131k.sh` (12 h, 64 CPU, 400G, infra01/normal;
  auto-adds `--reuse-base` on resume). Expected: 12 h timeout at ~1,700/1,940
  languages plus one resume, matching the 200k run; then `convert.py` packs
  `$SCR/glotlid_apertus131k.unilid` (~1.02 GB).
- Artifacts: `$SCR/results_apertus131k/` (per-language tokenizers),
  `$SCR/glotlid_apertus131k.unilid`, logs `apertus131k_train_2883222.{out,err}`.
- Checkpoint hygiene: no deletions; `results_apertus131k/` is new and the 200k corpus
  dir is reused read-only.

### 2026-07-23: Exp 24, metric decomposition of the saved full-test predictions (analysis only, no SLURM job)

- Login-node analysis, no new scoring. Script `analysis/metric_decomposition.py`,
  reviewed pre-run by an adversarial agent (no defects). Inputs: the Exp 16 prediction
  memmaps (job 2784115) and the floor-21 memmap (job 2791722) on scratch, plus
  `outputs/diagnostic/full_test_per_lang_f1.csv` and `lang_diagnostic.csv`.
  Consistency gates, all passed: kept-line count 45,377,279; every recorded
  within-stratum table value reproduced to 6e-5; saved per-language F1 reproduced
  exactly.
- Purpose: decompose the stratum rows (within-stratum macro-F1) against global
  per-language F1/precision/recall. Outcome (`EXPERIMENTS_RESULTS.md` Exp 24): the
  tail deficit is precision (0.459), not recall (0.874); the rejected configurations'
  tail ranking reverses under the global view (floor-21 reaches tail mean F1 0.7655
  versus baseline 0.5618); neither the guard nor the balanced val can register this
  failure mode. Follow-ups proposed as `EXPERIMENTS_PLAN.md` Open paths block E; the
  metric-view question added to the Decision required item.
- Artifacts: `outputs/tables/metric_decomposition.md`,
  `outputs/diagnostic/full_test_per_lang_prf.csv`.

### 2026-07-18 — Prior-centered regularizer + non-content token tying (plan items 3, 11)

Code changes before these runs, both reviewed pre-launch by an adversarial agent:
- `analysis/learned_prior.py`: (a) prior-centered penalty `reg*||b - gamma*log(N+1)||^2`,
  grid `PRIOR_GAMMAS = {0, 0.25, 0.5} x REGS`; outputs to `learned_prior_centered.md` /
  `learned_bias_centered.npy` so the Exp 14 artifacts stay intact. (b) GRADIENT BUG FIX
  (found by the review, present since Exp 14): the softmax-NLL gradient accumulated soft
  counts over ALL examples' top-k candidates while the loss conditions on the true label
  being in the top-k (recall 0.9971); the fitted b was therefore not the minimizer of the
  stated objective. Fixed by restricting the soft counts to present examples; verified by
  finite differences (max error 3e-8 with 33/40 absent examples). The Exp 14 measured
  deltas remain valid measurements of the b that was produced; the gamma=0 rows of this
  run give the corrected plain-L2 fit for comparison. Caution note added to
  `EXPERIMENTS_RESULTS.md` Exp 14.
- `analysis/token_tying.py` (new): pure non-content token tying; tied sets digits_ws
  (298 tokens), nonalpha_ascii (479), nonalpha_all (1,291) classified on byte-decoded
  token text. No renormalization: the review derived that renormalizing injects a
  per-language per-token offset `-log Z_L` up to 0.36 nats/token concentrated on flat
  confusers, conflating mechanisms; pure tying leaves untied columns bit-identical
  (unit-verified). Special tokens (each exactly p=0.2 per row, the peak-probability
  artifact investigated 2026-07-18: HF Unigram score-0 specials normalized into every
  row, 0.8 of all mass) are asserted and never touched.
  **Correction 2026-08-17:** "uniform across languages so argmax-neutral" was the
  original wording and is measured false. It is a training defect, not a property
  of the model; the mass is never read when scoring, and the resulting per-token
  depression is applied a different number of times per candidate because each
  language segments the text itself. See the 2026-08-17 entries above.

- **2794210** `unilid-bal-sweeps` — COMPLETED 00:08:52 (2026-07-19). Outcomes
  (`EXPERIMENTS_RESULTS.md` Exp 23): floor equalization rejected at selection (tail
  -0.0177 to -0.0269 now visible); punctuation partial pooling alpha=300 PASSES (all
  strata non-negative, effect at measurability edge); learned-bias refit on balanced
  data reg=0.3 PASSES (sel overall +0.0016, tail +0.0299, magnets +0.0252; suppressed
  list = head/twin sinks nya/por/heb, not flat magnets; ||b||_inf 11.3). Pending before
  adoption of either: refit-per-draw stability, balanced-test draw, full-test passes,
  and the explicit objective decision on individual-language suppression. First sweeps
  under the
  balanced protocol (Exp 22), three experiments in one job
  (`analysis/balanced_sweeps.py`): (a) floor-equalization re-selection (plan item 14
  follow-up, F grid {-17,-19,-21,-23}); (b) punctuation partial pooling (plan item 15,
  212 neutral dp columns toward within-script means, lam = alpha/(N+alpha), alpha
  {300, 3000, 30000}); (c) learned-bias refit on balanced data (plan item 16,
  per-language alternating fit/selection halves, plain L2, corrected gradient,
  interpretability table of most-negative offsets). Selection only, no test scoring;
  baseline validated against the saved full-test predictions at the balanced-val lines
  (expected agreement ~1.0; gate 0.99). Reviewed pre-launch (adversarial agent: no
  blocking defects; empties provably absent from the pool; all 1,940 languages in both
  refit halves, fit >= 4 / sel >= 3 examples each; one overclaiming conclusion sentence
  rewritten to single-draw wording per review). Artifacts:
  `outputs/tables/balanced_{floor_eq,punct_prior,bias_refit}.md`,
  `learned_bias_balanced.npy` (only on a guard pass). Script:
  `slurm_balanced_sweeps.sh`.
- **2793541** `unilid-tying-dp` — COMPLETED 00:09:08 (2026-07-19). NEGATIVE: dp_global
  val overall -0.0014 (twins -0.0060), dp_script -0.0016 (twins -0.0103, failing the
  twin guard alone); tail/magnets flat; baseline selected. The cost concentrating in
  twins shows digit/punctuation usage rates are within-pair discriminative signal
  (consistent with Exp 4's 10.5% punctuation share of within-pair KL). Tying is closed
  at every curation level; see the Exp 18 final reading. Curated re-run of the token
  tying after the user's critique of the Exp 18 design (whitespace/newlines should
  never have been tied; their frequencies encode spacing conventions). Tied set: 212
  tokens whose decoded text is entirely ASCII digits + neutral punctuation
  (`.,:;!?()[]{}/\|@#*+=<>~`_"%^`), with documented linguistic exclusions (apostrophes,
  hyphens/dashes, ampersand, currency, Spanish inverted marks, typographic quotes, all
  whitespace including leading-space Ġ-variants, all non-ASCII punctuation). Two
  configs: dp_script (primary, tie within script groups so writing-system conventions
  never cross scripts; single-script languages unchanged by construction) and dp_global
  (comparison). Pure tying, no renormalization; same guard. Reviewed pre-launch
  (adversarial agent: no defects; character inventory of the tied set audited
  linguistically; single-language-script invariance verified bit-exact; Exp 18 default
  path confirmed byte-identical). Note two in-scope caveats from the review: tied
  tokens like `,000`/`.000` smooth the decimal-separator locale convention
  (intentional), and ASCII click-letter risk is nil (orthographic clicks are alphabetic
  Unicode). Init-from: recovered `glotlidc.unilid`. Artifact:
  `outputs/tables/token_tying_dp.md`. Script: `slurm_token_tying_dp.sh`.
- **2791722** `unilid-ft-floor21` — COMPLETED 01:41:04 (2026-07-19). Resubmission of
  2791583 after fixing an over-strict startup gate. Full-test verdict on floor-21:
  overall +0.0129 (point), head -0.0003, twins -0.0001, mid +0.0001, but tail -0.0204
  [CI -0.0257, -0.0161] and magnets -0.0164 [CI -0.0210, -0.0129]; accuracy +0.0009.
  The tail cost is real (unlike the learned bias's test-half scare); floor-21 is a
  global-precision-for-tail-recall trade, dominated by the learned bias at equal
  overall gain, and is NOT adopted. Third val-selected point overturned at full scale.
  Results: `EXPERIMENTS_RESULTS.md` Exp 20. Artifact:
  `outputs/tables/full_test_floor21.md`.
- **2791583** `unilid-ft-floor21` — FAILED 00:00:23 (2026-07-19 00:13, exit 1:0). The
  startup completeness gate demanded no UNSEEN anywhere in `pred_baseline.npy`, but the
  Exp 16 run wrote predictions only for kept lines, so the 250,000 val positions are
  legitimately UNSEEN there (verified: 250,000 UNSEEN total, 0 on kept lines). Failed
  before any scoring or state changes; the gate now checks kept lines only. The pre-run
  review had asserted this check was verified against the memmaps; that verification
  claim was wrong. Original entry follows. Full-test evaluation of the
  Exp 20 guard-selected floor-21 matrix (plan item 14 follow-up): one scoring pass over
  the 45,377,279 non-val lines under the clamped matrix, compared against the SAVED
  Exp 16 baseline memmaps (job 2784115; opened read-only). Deterministic matrix rebuild
  + sha256 fingerprint; per-line label gate against the saved y_true; bidirectional
  val-partition cross-check; resumable. Decides whether floor-21 (test-half overall
  +0.0030) becomes a result of record or joins the tail-risk record: the test-half tail
  point is -0.0623 on ~35 items. Reviewed pre-launch (adversarial agent: no defects;
  walltime bound <= 5h10m vs 8h request; special-column assert added from its
  suggestion). Note for interpretation: absolute levels here use the zero-bias scorer
  path (like Exp 16), whereas the Exp 20 sweep used the unbiased predict path; the
  reported quantity is the internally consistent baseline-relative delta. Script:
  `analysis/full_test_floor21.py`, `slurm_full_test_floor21.sh`. Artifact:
  `outputs/tables/full_test_floor21.md`.
- **2791444** `unilid-flooreq-hier` — COMPLETED (2026-07-18). Outcomes: (a) floor
  equalization POSITIVE on overall, guard selects floor-21, test-half overall +0.0030
  [CI +0.0016, +0.0044], twins/head flat, magnets -0.0108 (crosses 0), tail -0.0623 (CI
  touching 0 on ~35 items; full-test check required before adoption; see Exp 20).
  (b) macrolanguage hierarchy NULL: deltas -0.0000 everywhere; macro-aware accuracy
  0.9680 vs exact 0.9603 measures the within-macro ceiling (Exp 21). Two experiments in
  one job
  (plan items 14 and 13, run in that order). (a) Downward floor equalization
  (`analysis/floor_equalization.py`): plateau clamped to min(floor_L, F), F in
  {-17, -19, -21, -23}; measured n_modified per F: 452 / 1,821 / 1,940 / 1,940; premise
  gate added (abort if corr(floor, log10 N) > -0.5). (b) Macrolanguage-hierarchical
  decision (`analysis/macro_hierarchy.py`): parameter-free logsumexp group marginal over
  SIL macrolanguages from top-50 candidates; guard is accept/reject; table reports
  hierarchical-vs-baseline deltas unconditionally (no tuning). Both reviewed pre-launch
  (adversarial agent: no correctness or crash bugs; Rust empty-string top-k early-return
  and K=50 marginal truncation verified; two reporting items fixed before submission;
  known cosmetic caveat: the hierarchy module's baseline uses top-k tie-breaking, which
  can differ from best_of on exact float32 ties). Init-from: recovered
  `glotlidc.unilid`. Artifacts: `outputs/tables/floor_equalization.md`,
  `macro_hierarchy.md`. Script: `slurm_floor_eq_hierarchy.sh`.
- **2790174** `unilid-backoff-wals` — COMPLETED 00:17:03 (2026-07-18). NEGATIVE, same
  monotone pattern as 2790155 within 0.0016 at every config (val overall -0.0036 at
  wals_lift_a300 down to -0.0304 at alpha=30000; tail/magnets drop at alpha >= 3000);
  nothing passes the guard; baseline selected. Genealogical grouping fidelity is
  immaterial to the outcome; the mass-lifting operation is the refuted element. Results:
  `EXPERIMENTS_RESULTS.md` Exp 19. Original entry follows. WALS genealogical grouping
  for the back-off (plan item 12, user-requested true families): tiered per-language
  fallback genus-within-script -> family-within-script (each requires >=
  `MIN_BACKBONE_GROUP = 3` backbone members) -> script. Source:
  `data/wals_languages.csv` (WALS export copied from `~/tokenizer-lm/data`, provenance
  in `data/README.md`; covers 1,159/1,940 languages; the parity-aware grouped config was
  evaluated as an alternative and rejected as primary source at 207/1,940 coverage).
  Tier assignment: genus 535, family 360, script 1,012, none 33; 37/96 tail languages
  get a genealogical tier (`outputs/diagnostic/backoff_groups_wals.csv`). Same six
  mode x alpha configs and guard as 2790155. Reviewed pre-launch (focused adversarial
  agent: no number-corrupting defects; nested-group semantics and None cascade verified
  on the real arrays). CAVEAT for interpretation: in genus groups with exactly 3
  backbone members (21 of 50 eligible genus groups), the EXCLUDE_K=3 confuser exclusion
  empties and falls back to all-but-self, so the confuser-excluded property is weaker at
  the genus tier. Artifact: `outputs/tables/family_backoff_wals.md`. Script:
  `slurm_family_backoff_wals.sh`.
- **2790155** `unilid-backoff` — COMPLETED 00:17:42 (2026-07-18). NEGATIVE: every config
  reduces val overall (lift_a300 -0.0028 ... lift_a30000 -0.0289; full mode within
  0.0007 of lift throughout); at alpha >= 3000 val tail drops 0.8710 -> 0.8387 and
  magnets 0.8797 -> 0.8609. Nothing passes the guard; baseline selected (test deltas
  zero by construction). Mechanism reading: lifting unseen-token mass toward the script
  mean makes languages MORE accepting of group-plausible foreign material, increasing
  theft; this is the direction Exp 10 warned about (small languages already
  under-penalize unseen tokens). The untried direction implied by Exp 10 is floor
  EQUALIZATION downward, not group-informed lifting. Original entry follows. Script-mean
  back-off at floor
  positions (plan item 12): each language's exact floor plateau (74,617-99,810 entries
  per row, measured; the emergent resource-tied unseen-token constant) is replaced by
  `lam_L * m_G(t)` with `lam_L = alpha/(N_L+alpha)`, m_G = confuser-excluded resource-
  weighted script backbone mean; modes lift/full x alpha {300, 3000, 30000}; observed
  tokens and specials bit-identical; no renormalization. 33 languages without a
  same-script backbone stay unmodified. Guard-selected on val, test half once. Reviewed
  pre-launch (adversarial agent: no number-corrupting defects; units of prior vs
  log-weights confirmed consistent because all rows are exactly normalized). Init-from:
  recovered `glotlidc.unilid`. Script: `analysis/family_backoff.py`,
  `slurm_family_backoff.sh`. Artifact: `outputs/tables/family_backoff.md`.
- **2790077** `unilid-learnprior` — COMPLETED (2026-07-18). Prior-centered learned bias
  sweep (18 fits, corrected gradient), val-guarded selection, test half once. Init-from:
  recovered `glotlidc.unilid`. Selected gamma=0.25, reg=10: test-half overall +0.0117
  [CI +0.0104, +0.0130], twins +0.0124, head +0.0089, magnets -0.0052 (crosses 0), tail
  -0.0320 (the noisy 250k-half tail; full-test read pending). Note: under the corrected
  gradient the previous operating point (gamma=0, reg=5) now FAILS the guard (val
  magnets -0.0119), so the gradient fix changed the fit materially. Marginal gain over
  plain-L2 reg=5 (+0.0112 on the same half); needs full-test confirmation before any
  supersession. Artifacts: `outputs/tables/learned_prior_centered.md`,
  `learned_bias_centered.npy`.
- **2790078** `unilid-tying` — COMPLETED 00:10:42 (2026-07-18). Non-content tying sweep
  (3 tied sets), val-guarded selection. NEGATIVE: every tied set reduces val overall
  (digits_ws -0.0010, nonalpha_ascii -0.0063, nonalpha_all -0.0078); nothing passes the
  guard; baseline selected (all test deltas zero by construction). Refinement hypothesis
  for a possible follow-up: the tied sets include the whitespace tokens (Ġ, Ċ), and
  whitespace frequency is genuinely language-discriminative (spaced vs unspaced
  scripts), so the negative may be dominated by tying whitespace; a digits+punctuation
  set that excludes whitespace was not run. Artifact: `outputs/tables/token_tying.md`.

### 2026-07-16 — Full-test-set evaluation (plan "Next methods" item 10, part 1)

- **2784115** `unilid-fulltest` — COMPLETED 05:06:50 (2026-07-18 00:02). Scores the 100k
  model on the full GlotLID test set minus the 250k val lines (45,377,279 lines) for three
  FIXED configurations: baseline, frequency prior gamma=0.5, learned bias reg=5.0
  (`outputs/tables/learned_bias.npy` from job 2731802). No selection: pure evaluation to
  tighten the stratified deltas; on the 250k test half every one of the 96 tail languages
  has <= 2 examples (67.7% have zero VAL examples), so the open question was whether the
  learned bias's tail delta (-0.0320, CI touching 0) is real. Outcome
  (`EXPERIMENTS_RESULTS.md` Exp 16): learned bias overall +0.0129, tail -0.0018
  [CI -0.0035, -0.0001] (the -0.0320 was split noise), magnets -0.0082, accuracy
  0.9608 -> 0.9751; frequency prior tail -0.0182 [CI -0.0225, -0.0146], i.e. NOT
  tail-safe (its Exp 14 tail 0.0000 was a tail-invisibility artifact). Baseline
  agreement with recorded predictions 0.9951 (check passed). Script:
  `analysis/full_test_eval.py` + `slurm_full_test_eval.sh`.
  Safety: seed-42 val-line reconstruction cross-checked against `val_mask.npy`; every
  sampled test-half line's label validated against the sample pickle (abort on first
  mismatch); zero-bias predictions validated against recorded UniLID predictions
  (abort if agreement < 0.99); resumable chunked memmaps on scratch guarded by a config
  fingerprint (sha256 of all three bias vectors + language list + chunking) so a resume
  with changed inputs aborts instead of mixing configurations. Reviewed before launch by
  an adversarial agent: numeric path confirmed correct; the fingerprint, atomic progress
  writes, and the baseline-agreement check were added from its findings. Bootstrap CIs
  (B=1000) for strata <= 3M examples (tail ~6k, magnets ~61k); point deltas only for
  twins/head/overall (n > 3M, item-level CI half-width < 0.001).
  Artifacts: `outputs/tables/full_test_eval.md`,
  `outputs/diagnostic/full_test_per_lang_f1.csv` (per-language F1 for plan items 5-6),
  memmaps + fingerprint in `/capstor/scratch/.../unilid_analysis/full_test_eval/`.

### 2026-07-10 — Selection-guard fix + re-selection re-runs (plan "Next methods" item 1)

Code change before these runs: the val-based selection guard in
`analysis/{hierarchical_pool,prior_sweep,learned_prior}.py` was unified into
`passes_guard` (`hierarchical_pool.py`) with `GUARD_STRATA = (tail, magnets, twins, head)`
and `GUARD_TOL = 0.01`; a config is eligible only if val overall macro-F1 improves and no
guarded stratum drops more than 0.01. Rule provenance and the tolerance decision (user
choice among {0.002, 0.01, 0.02} on 2026-07-10) are in `EXPERIMENTAL_SETUP.md`
("Selection guard"). `learned_prior.REGS` extended to {0.3, 1, 3, 5, 7, 10}.
`learned_prior.py`'s no-eligible-reg fallback now selects the baseline b=0 (the old code
fitted `max(REGS)` while printing "smallest reg"). A review agent found, and we fixed, a
rounding defect: `prior_sweep.py` had guarded on 4dp-rounded values; the guard and the
argmax now use unrounded values. Expected selections, precomputed from the saved val
tables (session scratchpad `test_guard.py`, all confirmed): 100k freq prior keeps
gamma=0.5; Apertus freq prior rejects all gammas (baseline selected, negative result);
learned bias moves off reg=0.3 (which costs val magnets -0.0318) to reg in {5, 7, 10}.

All three jobs: 500k seed-42 sample, val/test split `outputs/diagnostic/val_mask.npy`,
strata `outputs/diagnostic/lang_diagnostic.csv`, account `infra01`, 400 GB, expected
runtime 10-30 min each once scheduled. No checkpoints deleted for these runs (post-hoc
analyses; nothing to clean).

- **2731818** `unilid-commonlid` — COMPLETED (2026-07-10 22:22). CommonLID out-of-domain
  re-evaluation with the reg=5.0 bias (`learned_bias.npy` was overwritten by 2731802, so
  the recorded 0.8936 from job 2640066 belongs to the de-selected reg=0.3 vector and is
  superseded). Macro-aware accuracy: baseline 0.8452 -> freq prior gamma=0.5 0.8518
  (+0.0067, unchanged) -> learned bias reg=5.0 **0.8879** (+0.0427). Artifact:
  `outputs/tables/commonlid_eval.md` (overwritten).
- **2731802** `unilid-learnprior` — COMPLETED 00:09:05 (2026-07-10 22:15). Learned
  per-language bias re-run under the fixed guard + extended REGS. Init-from: recovered
  `glotlidc.unilid` (100k). Guard selected reg=5.0 (val magnets -0.0075; reg<=3 fail on
  magnets). TEST: overall +0.0112 [CI +0.0099,+0.0124], head +0.0094, twins +0.0135,
  magnets +0.0051 (CI crosses 0), tail -0.0320 [CI -0.0588,+0.0000], accuracy
  0.9603 -> 0.9749 (+0.0147). Supersedes job 2640065's selection (reg=0.3, +0.0180).
  NOTE: val tail macro-F1 is 0.8710 for every reg (and every gamma in the prior sweeps),
  so the val guard has no sensitivity on the tail stratum; the test tail movement was
  invisible to selection. Recorded as a guard limitation in `EXPERIMENTS_RESULTS.md`.
  Artifacts: `outputs/tables/learned_prior.md`, `learned_bias.npy` (both overwritten;
  npy is now the reg=5.0 fit).
- **2731803** `unilid-prior-apertus` — COMPLETED 00:15:25 (2026-07-10 22:21).
  Frequency-prior sweep on the Apertus 200k model under the fixed guard. Init-from:
  `glotlid_apertus200k.unilid`. Outcome as precomputed: NO gamma eligible (every gamma
  >= 0.25 drops val tail by >= 0.032), baseline selected, all test deltas zero. Replaces
  job 2649123's flawed gamma=3.0 selection (Exp 15 guard flaw); the frequency prior is
  rejected on the Apertus model. Val table identical to the 2649123 run (deterministic
  rescore). Note: the in-run "agreement with recorded UniLID preds" print (0.9608) checks
  against the 100k model's stored predictions, so below 1.0 is expected when model_path
  is overridden; the check is meaningful only for the default model. Artifact:
  `outputs/tables/prior_sweep_apertus.md` (overwritten).
- **2731804** `unilid-prior` — COMPLETED 00:16:36 (2026-07-10 22:23). Frequency-prior
  sweep on the 100k model under the fixed guard. Init-from: recovered `glotlidc.unilid`.
  Outcome as precomputed: gamma=0.5 re-selected (val magnets -0.0081 within tolerance);
  val and test tables identical to job 2639127 (deterministic rescore; gamma=0 agreement
  with recorded preds 0.9951, matching the known baseline self-agreement). The Exp 14
  frequency-prior result stands unchanged under the fixed guard; artifact header now
  records the guard rule. Artifact: `outputs/tables/prior_sweep.md` (overwritten).

### 2026-06-28 to 2026-06-29 — Prior redirect + Apertus retrain

> **Account change:** these runs used `--account=infra01` (not `a139`; the repo `SETUP.md`
> and older scripts are wrong — see memory `unilid-slurm-account`). All on CSCS Clariden,
> Python `/users/cmeister747/.pyenv/versions/3.11.5/bin/python3`, 500k uniform sample
> (`seed=42`), val/test split `outputs/diagnostic/val_mask.npy`, strata from
> `outputs/diagnostic/lang_diagnostic.csv`. Init-from: recovered `glotlidc.unilid`
> (100k model) unless noted.

Infra done this window: installed CPU `torch` (the default `pip install torch` pulled the
multi-GB CUDA build and blew the home quota); made the `transformers` import lazy in
`UNILID/unilid/api.py` so training doesn't clobber the custom Rust `tokenizers` build; added
Rust methods `best_of_cached_weight_sets_biased_batch` (per-language bias before argmax) and
`top_k_of_cached_weight_sets_batch` (top-k candidate scores for the learned-bias fit), both
rebuilt with `maturin develop --release` and validated (biased: zero-bias == unbiased;
top-k: gradient of the softmax fit checked to 2e-10).

- **2649123** `unilid-prior-apertus` — COMPLETED 00:16:19 (2026-06-29 23:11). Frequency-prior
  sweep on the Apertus 200k model. Plan: Exp 15. Init-from: `glotlid_apertus200k.unilid`.
  Guard selected gamma=3.0; overall +0.0203 but tail -0.0945, magnets -0.1102 (guard flaw).
  Artifact: `outputs/tables/prior_sweep_apertus.md`. Results: `EXPERIMENTS_RESULTS.md` Exp 15.
- **2641940** `unilid-apertus-train` — COMPLETED 01:41:24 (2026-06-29 13:45). Resume of 2639097
  (`--reuse-corpus --reuse-base --skip-existing-langs`); trained the last 250 languages and
  packed `glotlid_apertus200k.unilid` (1.56 GB, 1,940 langs, 200k vocab) via `convert.py`.
- **2639097** `unilid-apertus-train` — TIMEOUT 12:00:25 (2026-06-29 05:55). Standard-setup
  Apertus 200k retrain (no MAP prior): Apertus V2 200k byte-level BPE seeded into Unigram,
  SP per-language re-estimation on recovered `train.txt` (60,683,151 lines),
  `--max-base-samples-per-lang 10000 --lang-batch-size 20`. Reached 1,690/1,940 at the wall.
  Plan: Exp 15 (replaces the abandoned MAP-EM Exp 14 plan item). Script: `slurm_apertus_train.sh`.
- **2640066** `unilid-commonlid` — COMPLETED 00:06:13 (2026-06-28 21:05). CommonLID out-of-domain
  eval with the priors (baseline / freq gamma=0.5 / learned bias). 373,230 web lines, 109 tags.
  Macro-aware acc 0.8452 -> 0.8518 -> 0.8936. Artifact: `outputs/tables/commonlid_eval.md`.
- **2640065** `unilid-learnprior` — COMPLETED 00:07:56 (2026-06-28 20:56). Learned per-language
  bias: top-20 candidate extraction on val, L2-regularized softmax fit (reg swept {0.3,1,3,10},
  reg=0.3 selected on val), exact test eval via biased scorer. Test macro-F1 0.9454 -> 0.9638
  (+0.0180). Artifacts: `outputs/tables/learned_prior.md`, `learned_bias.npy`. Plan: Exp 14.
- **2639127** `unilid-prior` — COMPLETED 00:27:28 (2026-06-28 18:27). Frequency-prior sweep
  `b_L=gamma*log N_L` on the 100k model, gamma in {0..5}. gamma=0.5 selected: macro-F1 +0.0058.
  Artifact: `outputs/tables/prior_sweep.md`. Plan: Exp 14.
- **2639065** `unilid-hpool` — COMPLETED 00:28:16 (2026-06-28 18:19). Expanded Stage 1 sweep
  (uniform shrink, liability-scaled shrink, entropy sharpening; 11 configs). NEGATIVE:
  sharpening collapses magnets to ~0, shrink neutral at best. Artifact:
  `outputs/tables/hierarchical_pool.md`. Plan: Exp 13.
- **2638804 / 2638803** `unilid-hpool` / `unilid-commonlid` — COMPLETED 2026-06-28 ~17:28.
  Original Stage 1 shrinkage prototype + first CommonLID eval (Exp 13, Exp 12).
- **2626411 / 2626402** `unilid-hpool` / `unilid-commonlid` — FAILED 00:00:11 (2026-06-27).
  Early submissions before the CPU-torch / lazy-transformers infra fixes; import errors.

### 2026-06-26 — Artifact recovery + infra
- Recovered from Google Drive (folder `19sRPRiFHX8Lk3vZWlNGl0zzA88eAZ3Yx`) to scratch:
  `train.txt` (60,683,151 lines), `glotlid_correct_test.txt`, `glotlid_train_counts.json`,
  and all 5 prediction files (UniLID, DeepSeek, Qwen, Marg, fastText). Full 744 MB model
  recovered from the polybox link in `UNILID/README.md` to
  `/capstor/scratch/cscs/cmeister747/unilid_analysis/glotlidc.unilid` (verified full
  1940x100k matrix; the in-repo copy is truncated to ~7%).
- Rebuilt the custom Rust tokenizers (`maturin develop --release`); verified
  `set_weight_sets` + scoring on eng/deu/fra.
- Staged Apertus V2 200k tokenizer (`swiss-ai/apertus-tokenizer-development`,
  `preliminary_mul_200k`, byte-level BPE) to
  `/capstor/scratch/cscs/cmeister747/unilid_analysis/apertus_v2_200k/tokenizer.json`.

### 2026-06-24 — Exp 10 error analysis
- 7-cut agent workflow over a 28,527-error stratified sample + per-token decomposition +
  weight-matrix audit. Outcome: under-fit-tail attractor mechanism; two attractor types;
  ~80-85% recoverable ceiling. Drives the pooling direction. See `EXPERIMENTS_RESULTS.md`
  Exp 10 and project memory `unilid-error-analysis-findings`.

### Exp 11 — Per-language diagnostic (ongoing)
- `analysis/diagnostic.py`: structural flatness/entropy + full pairwise symmetric-KL +
  empirical magnet ratio (val half only) -> per-language category classifier.
  Output `outputs/diagnostic/lang_diagnostic.csv`.

---

## Family: Discriminative re-weighting and low-resource transfer (Exp 8–9)

**Window [inferred]:** ~2026-04-06 to 2026-04-07. Plan items: `EXPERIMENTS_PLAN.md` Exp 8,
Exp 9. Designed during the `/ultraplan` discussion on 2026-04-06 (recovered prompts 77–88).

### Exp 9 — Distribution transfer (related-language 9a + script-average 9b)
- **Job:** `1808399` (`unilid-transfer`), COMPLETED, 25 min, 400 GB.
- **Submission script:** `slurm_transfer_sweep.sh` → `analysis.transfer_sweep.generate_transfer_sweep(sample_size=500_000)`.
- **Hypothesis:** interpolating under-fit low-resource distributions toward a related
  high-resource language / script-average raises low-resource accuracy.
- **Search space:** `lambda ∈ {0.0,0.1,…,1.0}` (11), two approaches → 22 configs;
  223 related-language transfer pairs.
- **Init from:** scratch (analysis over the existing `glotlidc.unilid` weights; no model
  training).
- **Outcome:** 9a: <500 +10.6pp at λ=0.3, overall −1.3pp at the same λ; 9b: overall stays 0.960–0.961 across
  λ∈[0.1, 1.0], <500 does not exceed the 0.789 baseline at any λ<1.0. See `EXPERIMENTS_RESULTS.md` Exp 9.
- **Artifacts:** `outputs/tables/transfer_sweep.md`, `outputs/figures/transfer_sweep.png`.

### Exp 8a — Heuristic discriminative weighting
- **Job:** `1808414` (`unilid-disc8a`), COMPLETED, 13 min, 400 GB.
- **Submission script:** `slurm_discriminative_heuristic.sh` → `analysis.discriminative_finetune.generate_heuristic_discriminative(sample_size=500_000)`.
- **Hypothesis:** variance-based token up-weighting improves within-cluster separation.
- **Search space:** Setup A & B at `α ∈ {0.0,0.5,1.0,2.0,5.0}`, Setup C at
  `β ∈ {1.0,5.0,10.0}`, across the 7 confusion clusters.
- **Outcome:** all three setups reduce accuracy at every tested parameter; per-cluster accuracy 0 across all 7
  clusters for A and B at α≥1.0. See `EXPERIMENTS_RESULTS.md` Exp 8a.
- **Artifacts:** `outputs/tables/discriminative_heuristic.md`.

### Exp 8b — MMI discriminative fine-tuning (NOT RUN)
- **Status:** designed but not implemented. `analysis/discriminative_finetune.py` marks it
  `TODO` in the module docstring; no submission script exists. Carried in
  `EXPERIMENTS_PLAN.md` as not-started. This is where the session ended.

---

## Family: Training-data analysis (Exp 7)

**Window [inferred]:** ~2026-04-06. Plan item: `EXPERIMENTS_PLAN.md` Exp 7.
- **Run location:** login node, single pass, ~30 min (no SLURM job ID recorded).
- **Code:** `analysis/train_data_analysis.py`. Input: full training corpus
  (`config.TRAIN_FILE = $SCRATCH/train.txt`, 60,683,151 lines), downloaded from Google
  Drive on 2026-04-06 (recovered prompts 67–68).
- **Sub-analyses run:** 7.1 domain distribution, 7.4 per-language corpus quality, 7.5 script
  verification. **Deferred:** 7.2 mislabeling, 7.3 overlap (recovered prompt 82, "Skip
  analysis 7.2 and 7.3 for now").
- **Outcome:** 98.1% "other" domain; low-resource corpora short and small-vocab; 20
  off-script languages. See `EXPERIMENTS_RESULTS.md` Exp 7.
- **Artifacts:** `outputs/tables/train_data_analysis.md`,
  `outputs/figures/train_{domain_stacked,quality_scatter,script_purity}.png`.

---

## Family: Floor sweep (Exp 6)

**Window [inferred]:** ~2026-04-06 (recovered prompt 73).
- **Job:** `1806690` (`unilid-floor`), COMPLETED, 4.5 min, 400 GB.
- **Submission script:** `slurm_floor_sweep.sh` → `analysis.floor_sweep.generate_floor_sweep(sample_size=500_000)`.
- **Hypothesis:** clamping per-language weights at a higher floor (finite OOV probability)
  improves accuracy.
- **Search space:** `floor ∈ {None, -22.0, -15.0, -10.0}`.
- **Outcome:** floor=−22: 0/500k predictions changed; floor=−15: 3,372 changed, net −109; floor=−10: accuracy
  0.960 → 0.916. See `EXPERIMENTS_RESULTS.md` Exp 6.
- **Artifacts:** `outputs/tables/floor_sweep.{md,tex}`, `outputs/figures/floor_sweep.png`.
- **Note:** code comment in `floor_sweep.py` cites "OOV at -1e30"; user flagged this as not
  present in the repo (recovered prompt 75). Comment is inaccurate.

---

## Family: Normalization (Exp 2 re-classification + Exp 5 alpha sweep)

### Exp 5 — Alpha sweep (partial normalization)
**Window [inferred]:** ~2026-04-05/06 (recovered prompt 66, "did the alpha sweep finish").
- **Job:** `1804584` (`unilid-alpha`), COMPLETED, 12 min, 400 GB.
- **Submission script:** `slurm_alpha_sweep.sh` → `analysis.normalized_predict.generate_alpha_sweep(sample_size=500_000)`.
- **Search space:** `alpha ∈ {0.0,0.1,…,1.0}` (11), scoring `score / n_tokens^alpha`.
- **Outcome:** best `alpha=0.1` accuracy 0.961 (+0.001 over `alpha=0.0`); accuracy decreases monotonically for
  α>0.1. See `EXPERIMENTS_RESULTS.md` Exp 5.
- **Artifacts:** `outputs/tables/alpha_sweep.{md,tex}`, `outputs/figures/alpha_sweep.png`.

### Exp 2.5 — Full re-classification with normalized scores
**Window [inferred]:** ~2026-04-04 (recovered prompt 55).
- **Job:** `1795556` (`unilid-norm`), COMPLETED, 2.5 min, 400 GB.
- **Submission script:** `slurm_normalized.sh` → `analysis.normalized_predict.generate_normalized_analysis(sample_size=500_000)`.
- **Implementation:** added `best_of_cached_weight_sets_normalized` to the Rust Unigram
  tokenizer fork (`UNILID/tokenizers/`); PyO3 bindings; `predict_normalized` wrappers.
- **Outcome:** normalization drops accuracy 0.960 → 0.885; raw rescore reproduces originals
  exactly (100% agreement, validates implementation). See `EXPERIMENTS_RESULTS.md` Exp 2.
- **Artifacts:** `outputs/tables/normalized_comparison.{md,tex}`.

---

## Family: Tokenization length bias (Exp 2 token-delta + counterfactual)

**Window [inferred]:** ~2026-03-28 to 2026-04-03. This family hit repeated OOM/timeout
failures before succeeding; failures are recorded with the same rigor as the success.

- **Job `1791511`** (`unilid-lenbias`), COMPLETED, 5h, 400 GB. Full run: token deltas +
  scores + pairwise normalization counterfactual over 1,789,423 misclassifications
  (12h walltime). **The result of record for Exp 2.3–2.4.**
- **Job `1790440`**, TIMEOUT, 6h, 400 GB. Score computation roughly doubled runtime; hit
  the 6h walltime. Fixed by requesting 12h.
- **Job `1789048`**, OOM, 12 min, 128 GB. After a code refactor; still needed 400 GB.
- **Job `1752234`**, COMPLETED, 3h, 400 GB. Token-delta only (older code, no scores).
- **Job `1750406`**, OOM, 1.75h, 128 GB. Streaming fix applied; tokenizer cache still OOM
  at 128 GB.
- **Job `1747559`**, OOM, 2.5h, 128 GB. First attempt; storing texts in the pickle caused
  the OOM. Led to the decision to keep raw texts out of the sample pickle (see `SETUP.md`).
- **Memory lesson:** the per-language tokenizer cache (~1,895 tokenizers × 100k vocab)
  needs ~250 GB; 400 GB requests are required for any tokenizer-heavy job.
- **Code:** `analysis/length_bias.py`. **Artifacts:** `outputs/tables/length_bias.{md,tex}`,
  `outputs/figures/length_bias_histogram.png`.

---

## Family: Per-language distribution + token classification (Exp 3, Exp 4)

**Window [inferred]:** ~2026-04-05 to 2026-04-06 (recovered prompts 60–64).
- **Run location:** login node (no SLURM job IDs recorded).
- **Exp 3** (`analysis/distribution_analysis.py`): KL(lang‖base) for 1,940 languages;
  15 related-pair comparisons (symmetric KL, correlation, MAD); top divergent tokens.
- **Exp 4** (`analysis/token_classification.py`): heuristic 8-category classifier over the
  300 top divergent tokens (15 pairs × 20).
- **Outcomes:** low-resource (<500 sample) mean KL from base 0.32 vs 0.68–0.71 at 5k+; in token-category
  classification of top KL-contributors, morphological affixes 32.6% + content words 22.8% + function words 15.7%
  = 71.1% of category share; script/encoding 0%. See `EXPERIMENTS_RESULTS.md` Exp 3, Exp 4.
- **Artifacts:** `outputs/tables/distribution_analysis.{md,tex}`,
  `token_classification.{md,tex}`; figures `kl_vs_training_size.png`,
  `pairwise_logprob_scatter.png`, `pairwise_kl_vs_training.png`,
  `token_categories_stacked.png`.

---

## Family: Multi-system comparison + tables (Exp 1)

**Window [inferred]:** ~2026-03-26 to 2026-03-28. The project's first work.
- **Job `1747558`** (`unilid-tables`), COMPLETED, 16 min, 64 GB. Full-dataset tables
  (45.6M samples) via `analysis.run_all --sample-size 45627279 --format both`
  (`slurm_tables.sh`).
- **Sampling:** 500k uniform, `seed=42`, without replacement (`analysis/sample_data.py`).
- **Models added incrementally** (recovered prompts): UniLID + DeepSeek + Qwen first
  (prompt 10, 2026-03-26), then UniLID-Marg (prompt 18, 2026-03-27), then fastText
  (prompt 19, 2026-03-27).
- **Macro-F1 bug found and fixed** after a code-review agent (recovered prompt 20,
  2026-03-27): metrics had averaged F1 over `set(y_true) | set(y_pred)`; corrected to
  average over `set(y_true)` only (sklearn convention). See `SETUP.md` gotcha 5.
- **Artifacts:** `outputs/tables/table{1-7}_*.{md,tex}`, confusion-matrix PNGs/TeX for 7
  clusters; raw prediction outputs in `full_prob/` and `glotlid_e100_sanity/`.

---

## Infrastructure events (recovered from prompt history)

- **2026-04-02:** moved the UniLID model and data from home to scratch to free disk space
  (recovered prompts 38–39); `config.DATA_DIR` repointed to scratch.
- **2026-03-29/30:** bumped SLURM memory request to 400 GB after repeated 128 GB OOMs
  (recovered prompt 33).
- **2026-04-03:** permission-denied on a shared dataset path
  (`/capstor/store/.../stackv2-edu`) (recovered prompt 46); not central to UniLID analysis.

## Checkpoint / artifact-deletion assessment (updated 2026-08-06; deletions user-approved)

The 2026-05-27 version of this section stated that no model checkpoints are produced;
that stopped being true with the Apertus retrains (2026-07). Current assessment, with
the deletions the user approved on 2026-08-06 executed:

Deleted from scratch (about 62 GB freed):
- `glotlid_apertus131k.unilid` and `glotlid_apertus200k.unilid` (the two models trained
  through the float32 EM bug; superseded by the `_fp64` retrains for every recorded
  purpose; the bug itself remains reproducible from the documented azj_Latn recipe, the
  retained pre-fix binary `~/.local/bin/spm_train.pre_fp64`, and the trigger-line
  artifact `outputs/diagnostic/em_trigger_azj_81251_81640.txt`).
- `results_apertus131k/`, `results_apertus131k_fp64/`, `results_apertus200k_fp64/`
  (per-language training outputs behind those models; the retained packed `.unilid`
  files carry the results of record, and the training dirs are regenerable from the
  corpus plus the patched trainer and the recorded recipes). Consequence accepted with
  the approval: re-running the superseded corrupted-branch evaluation (Exp 29/30) is no
  longer possible; its prediction memmap and recorded entries remain.

Kept, with reasons:
- `glotlidc.unilid` (the production model behind every main-line result),
  `glotlid_apertus131k_fp64.unilid`, `glotlid_apertus200k_fp64.unilid` (the clean
  records of the closed vocabulary branch, Exp 42/43).
- `results_apertus200k/` in full: its `corpus/` subdirectory is load-bearing (the
  per-language calibration corpora read by the gate machinery and `gt_counts.py`), and
  its per-language tokenizer outputs back `gt_counts.csv` regeneration.
- Everything in `full_test_eval/` (y_true, all prediction memmaps including the
  superseded margin-family ones that the recorded two_sided_report wiring gates read,
  the seed-301 split record, the banked candidate arrays), plus
  `full_test_eval_131k/` and `full_test_eval_131k_fp64/` (the branch measurements of
  record, 88 MB each).
- All kept scratch artifacts re-touched 2026-08-06 against the 14-day purge.

Durable-storage migration (2026-08-06, user-directed): the artifacts of record
were moved from scratch to /capstor/store/cscs/swissai/a0229/cmeister/unilid_analysis
(5,975,047,573 bytes, 50 files, each byte-verified with cmp before the scratch
original was removed) and replaced on scratch by absolute symlinks, so every
path recorded in code and in these documents keeps working unchanged. Moved:
the three models of record (glotlidc.unilid and both _fp64 retrains),
sample_500k_all.pkl, and every file in full_test_eval/, full_test_eval_131k/,
and full_test_eval_131k_fp64/. Still on scratch under the re-touch policy:
results_apertus200k/ (the calibration corpus, regenerable from Drive) and
glotlid_unilid/ (the test file, regenerable from its zip). The user designated
the store path for anything that must not risk the 14-day purge.

Corpus migration completed (2026-08-06): results_apertus200k (35.1 GB, 5,821
files, including the 1,940 calibration corpus files) moved to
/capstor/store/cscs/swissai/a0229/cmeister/unilid_analysis/results_apertus200k
and replaced on scratch by a directory symlink; every file checksum-verified
before the scratch original was removed (chunked login-node verification; per
the user's instruction, no SLURM job is used for file management, since a
SLURM job here always occupies a node's four GPUs regardless of the request;
one earlier SLURM attempt, job 3019444, failed on a script quirk before
touching anything and is superseded by this record). CORPUS_DIR resolves
through the symlink unchanged; corpus reads verified. With this, everything
load-bearing lives on durable storage; scratch retains only regenerable data
(glotlid_unilid/) and logs.



## 2026-08-23: tab:length_accuracy rebuilt from the WiLI retrains

- **Plan item:** `EXPERIMENTS_PLAN.md:986` (the WiLI table set). `RERELEASE_PLAN.md:268`
  filed `length_accuracy` under "needs the co-author's artifacts"; that is now wrong for
  the UniLID column, which is re-runnable here. The fastText column still is not.
- **Instrument:** `analysis/wili_length_accuracy.py`, new. `analysis/wili_eval.py` was
  refactored to expose `predict_all` and `out_of_set_labels`, which the new script imports,
  so the empty-after-preprocess and out-of-label-set conventions cannot drift between the
  two. Pure extraction; no behavior change. wili_eval stores no per-line predictions, so
  predictions are recomputed.
- **Length definition, established before any accuracy was computed:** `len(raw_line)` in
  Unicode code points, no preprocessing, no stripping. It reproduces all six published
  bucket counts exactly (7,845 / 26,652 / 31,449 / 29,494 / 18,142 / 3,918). utf-8 bytes
  does not (2,947 / 16,851 / 25,389 / 32,660 / 27,363 / 12,290). Stripping changes nothing:
  no WiLI line has leading or trailing whitespace. Shortest line in both splits is 140
  chars, so nothing falls below the first bucket, as the caption claims. The count check
  aborts on mismatch and was confirmed to fire by feeding it the byte definition.
- **Instrument gate, PASSED:** the stored defective `wili_assets/wili_100k_500.unilid`
  reproduces all seven published UniLID cells to the two decimals printed
  (93.10 / 94.17 / 95.86 / 96.78 / 96.53 / 96.53, Overall 95.65; every delta 0.00 pp).
  `outputs/rerelease/wili_length_accuracy_wili_100k_500.json`.
- **Retrains measured** (login node, a few minutes each, 117,500 lines, 0 empty and 0
  out-of-set for all three):
  - `wili_100k_500_fp64`  93.04 / 94.11 / 95.83 / 96.79 / 96.60 / 96.61, Overall 95.64
  - `deepseek_v3.2_wili_fp64` 92.59 / 93.87 / 95.40 / 96.32 / 96.08 / 95.53, Overall 95.21
  - `qwen3_8b_wili_fp64`   91.56 / 92.95 / 94.78 / 95.64 / 95.69 / 95.30, Overall 94.52
  Each Overall matches the independently computed accuracy in the corresponding
  `outputs/rerelease/wili_eval_*.json` to all printed digits.
- **Reading:** the retrained base model shifts the published column by at most 0.08 pp in
  any bucket; the correction is not visible at table resolution. The two LLM-tokenizer
  variants lose most in the shortest bucket, which is the regime the paragraph at
  `submission.tex:984` argues about.
- **Artifacts:** `outputs/rerelease/wili_length_accuracy_{wili_100k_500,
  wili_100k_500_fp64,deepseek_v3.2_wili_fp64,qwen3_8b_wili_fp64}.json`.
- **Not done:** the fastText column. No fastText model trained on WiLI exists on this
  machine, so those six cells stay the co-author's.

## 2026-08-22/23: the queued wave completes; WiLI gate and evals; group B settled

All six queued jobs finished clean on 2026-08-22: 3138626/3138627/3138628 (WiLI
retrains, 20-25 min each), 3127704 (gate_variants topk on the corrected model,
6m40s, top-1 agreement 1.0000 with pred_floor21 on all 2,228,183 affected
lines), 3117575/3117576 (DeepSeek/Qwen full-pool evals, ~1h52m each).

2026-08-23, run on the login node: macro FPR recomputed for both variant evals
from their saved prediction arrays (variant_fp64_fulltest_fpr.json);
analysis/wili_transform_gate.py written by one agent, adversarially reviewed by
a second (verdict FIX FIRST: a live wrong-PASS on empty --models, exit-code
ambiguity, threshold-transfer disclosure; 11 fixes applied), then run — FAILs
read per model: DeepSeek 11 / Qwen 14 failing languages (corruption signature),
wili_100k_500 107 (systematic, needs the null arm); analysis/wili_eval.py run
on all three retrains (cells match published at three decimals except Qwen
.949->.948); group-B re-derivation measured (unchanged four).

Submitted: 3157817 (gate_variants apply flat4_tau5 then flat4_prox21 on the
corrected model; before submitting, gate_variants.py:1275's hardcoded
outputs/tables/gate_flat4_prox21_build.md was routed through _out() so the
corrected run cannot overwrite the released build record) and 3157851 (fp32
null-arm retrain of wili_100k_500: unpatched spm_train from fork commit
2b7ec9b built into the isolated sp_fp32_env/ prefix, sha256-discriminated
preflight; decides whether the 107-row gate FAIL is a build effect or a
non-reproducing stored model).

2026-08-23, recorded borrow: pred_fasttext.npy was COPIED (plain copy, sha256
4ff74fb55ce5668b...  verified equal) from full_test_eval/ into
full_test_eval_corrected/. The array is the external fastText model's
predictions over the same test file and is model-invariant given the identical
1,940-label order (verified); it cannot be regenerated into the corrected tree
because analysis/import_external_pred.py is not parametrized. A symlink was
deliberately NOT used: 40 of the released tree's 46 entries realpath into
/capstor/store, and a store-backed symlink in the corrected root would poison
analysis.model_context's guard for every later non-default resolve.

2026-08-23: tatoeba.zip (1,028,156,179 bytes, sha256 fe0c0292a2e50289...)
downloaded from the lid-eval-datasets DRAFT release into wili_assets/. The
public GitHub API returns drafts as an empty list, which is how the 2026-08-21
pass missed it; gh with authentication sees them. Contents: tatoeba_test.txt
(148 MB), tatoeba_train.txt, tatoeba_full.txt, sentences.csv,
reduced_tatoeba.csv.

2026-08-23, degeneracy-scan definition change (32 -> 34): the estimated-token
count in analysis/degeneracy_scan_mistralnemo.py now runs over REAL columns
only. The whole-row definition counted entries above the row minimum, which is
the plateau only while the four specials sit at log-prob 0.0 (the defect); in a
0.3.0-packed corrected container the specials hold the row MINIMUM and the
whole-row scan flags nothing (measured: 0 flagged on the corrected Nemo
container). Under the real-column definition both containers flag the identical
34 rows: the released record's 32 plus csw_Cans (98) and ike_Cans (99), whose
whole-row counts the 4 specials had inflated past MIN_ESTIMATED=100 (verified:
est_whole_row - est_real == 4 for all 1,940 rows). The released record at
outputs/tables/degenerate_rows_mistralnemo.md (32 rows) predates this change
and is kept as-is; re-running the scan on the released container at HEAD now
yields 34.

## 2026-08-23: Phase 2/3 WiLI trainings prepared and submitted; Phase 2b half-blocked

Prepared (each smoke-tested on a 2-language mini-corpus with the exact argv,
every preflight failure mode executed): Mistral-Nemo-WiLI (base tokenizer
extracted from the GlotLID-C container, 131,072 entries, specials at
[1,2,10,0]); Llama-3.2-1B-WiLI (tokenizer downloaded, revision 4e20de36,
byte-level, converts cleanly via train.py's own _convert_to_unigram_base);
the Phase 3 100k-from-defaults run plus analysis/wili_vocab_repro_check.py
(ordered token-list comparison, validated with three controls) and the
parameterized vocab-size script gated on that verdict.

Negative result (full record outputs/rerelease/wili_phase2_conversion_smoke.json):
Mistral-7B-v0.2 and Llama-2-7b-hf have non-byte-level vocabularies whose
entries include raw carriage returns (51 and 24 offending tokens); the
per-language SentencePiece seed-vocab writer refuses them
(unilid/vocab_io.py:120), so those two rows cannot train under
--per-lang-counts-method sp without an author decision (byte-level re-encoding,
dropping entries, or a different counts method -- each a different model).
Also: mistral-community/Mistral-7B-v0.2 has 32,000 vocab entries, not the
32,768 the plan stated; 32,768 is v0.3. Which repository the published
\unilid-Mistral row used is an open author question. Downloads succeeded for
all three repos (HF_TOKEN authorized on the gated meta-llama repos); manifest
with revisions and sha256s at outputs/rerelease/wili_phase2b_tokenizer_downloads.json.

Submitted: 3161886 (mistralnemo_wili), 3161887 (llama32_1b_wili), 3161889
(wili_100k_defaults + repro check), 3161890/91/92/93 (10k/20k/50k/200k,
afterok:3161889; the repro verdict decides only whether they are reported as
the published models or as new models built by the published procedure).

## 2026-08-23: Mistral identity verified; dropped-entry decision implemented; all seven WiLI variant trainings submitted

The author challenged the recorded "\unilid-Mistral cannot be Mistral-Nemo"
conclusion, believing the row used mistralai/Mistral-Nemo-Base-2407. Verified
adversarially (outputs/rerelease/mistral_identity_verification.json): the
conclusion HOLDS -- 0.921 and 0.958 are adjacent rows of ONE table
(tab:unilid_llm_comparison, WiLI corpus, caption stating rows differ only in
base tokenizer), so equal tokenizers would force equal cells. The author's
belief is confirmed for the OTHER row: Mistral-Nemo-Base-2407's ordered token
list is byte-identical to the base inside glotlid_mistralnemo_fp64.unilid
(131,072 entries), and submission.tex:784 names that repo for
\unilid-Mistral-Nemo. Two corrections to the record: v0.2 has 32,000 entries,
not the plan's 32,768 (that is v0.3, genuinely different); and
mistralai/Mistral-7B-v0.1 is byte-identical to the mistral-community v0.2 copy,
so v0.1-vs-v0.2 is moot. Corroboration: the only two sub-0.93 rows are exactly
the two 32k non-byte-level candidates. Author decision 2026-08-23:
\unilid-Mistral retrains on the v0.1/v0.2 32,000-entry tokenizer, stated
unconfirmed.

Author decision 2026-08-23, implemented: vocabulary entries the per-language
seed-vocab writer refuses (vocab_io.py:119-120, tab/newline/CR) are DROPPED
whole from the converted base. The filter mirrors the writer's predicate and is
verified in both directions against the real writer (every dropped token
individually refused; the survivors accepted). Mistral-7B-v0.2: 51 dropped ->
31,950 entries; Llama-2: 24 dropped -> 31,977; all 75 dropped tokens carry \r
only (code-punctuation pieces); full lists in
outputs/rerelease/wili_{mistral7b_v02,llama2_7b}_base_convert.json. Pre-drop
bases parked in superseded_bases_20260823/. Smoke tests pass end-to-end, real
mass 1.0, no refusal hit. Note: the refusal check dates to the UNILID package's
initial commit here, so how the ORIGINAL models trained past it depends on the
co-author's environment -- consistent with these rows being "new models built
by the published procedure", not byte-reproductions. Also note (recorded in the
smoke JSON): --byte-level is overridden to False from these bases by
language_specific_trainer.py:263, before and after the drop.

Submitted: 3162788 (mistral7b_v02_wili), 3162789 (llama2_7b_wili). All seven
WiLI variant/vocab-size trainings are now queued.

## 2026-08-24: the 18-job wave completes; compilation and chain continuation

All 18 jobs COMPLETED 0:0 (sacct): 3157817 apply stages (27s, verified
legitimate), 3157851 fp32 null arm (18m55s), 3158825 Nemo stages (1h58m), six
Tatoeba evals (3-6 min each), seven WiLI trainings (3161886/87/89/90/91/92/93,
3162788/89; 4 min - 1h34m). Three agents dispatched: null-arm verdict analysis;
WiLI compilation (report outputs/rerelease/wave_2026-08-24_compilation.md, nine
wili_eval runs added); corrected-chain continuation (report
outputs/rerelease/corrected_chain_2026-08-24.md; build_release_calibration
passed, paper_eval/paper_breakdowns corrected tables written, release_gates
base PASS; two blockers to a fix pass). Paper-edits agent launched to update
paper/PAPER_EDITS_pending.md and apply ready edits under \corrrev{}.

2026-08-24, chain completion: external_bench_eval eval-stage non-default fixes
(floor target from the model's own fingerprint; acceptance gate binding only
for the default model; resolve_out_root in configure -- selfcheck 42/42);
corrected UDHR/FLORES eval stages run; calibrated bundle packed via
unilid-calibrate bundle (version 2, c=-17); release_gates PASS in both modes at
exact equality. fp32 null-arm verdict recorded (build hypothesis refuted, 106
of 107 failures shared across builds); cap-4192 second null arm in preparation
(tests whether the stored model was trained at sentencepiece's default
max_sentence_length).

2026-08-24, cap null arm: job 3173500 submitted (fp32 build +
max_sentence_length 4192, the sentencepiece default; one variable vs the
fp32null arm). Unit correction: the cap applies to ENCODED bytes; 2,052 lines /
106 languages exceed it, and those 106 are exactly arm (a)'s failing set
(symmetric difference empty). The "101 lines over the cap" figure recorded
2026-08-21 was characters, and "65 of 106" was raw bytes -- both wrong units.
UNILID gains --max-sentence-length (default 1,000,000 unchanged, 111 tests
pass, patch in patches/unilid_max_sentence_length.patch); null-arm drivers
promoted to analysis/wili_null_arm_verdict.py / _augment.py with an ARMS
registry.

2026-08-24, CommonLID corrected (jobs 3174187 scoring 7m40s, 3174266 carried
re-run 3m49s, debug partition -- normal was 713 jobs deep): both scripts
parametrized (out-root routing, fingerprint-driven clamp, verify_one_sided_clamp,
non-binding recorded constants for non-default models; a cross-model bug fixed
in the eval stage's language-list load); selfcheck 46/46; released-tree replay
byte-identical except the git-commit provenance line. Corrected cells recorded
in EXPERIMENTS_RESULTS; B4 handed to the paper agent.

2026-08-24, corrected release SHIPPED. PR #3 merged upstream 19:24:32Z (merge
commit a47d4f5, tree identical to PR head 2d5f62d), so the package carries 0.3.0
before any weight was published. Annotated tag v0.3.0 created on a47d4f5 and
pushed to Ahmetcanyvz/UNILID. Because #3 was already merged, the load-time
generation report went to a NEW branch `generation-report` off a47d4f5, commit
6006095 (README, REPRODUCING, eval.py, tests/test_special_token_mass.py,
unilid/model_io.py; 119 tests pass on the committed tree alone), opened as PR #4.
The --max-sentence-length training-path changes (train.py, unilid/constants.py,
trainers/language_specific_trainer.py) were deliberately left uncommitted and
verified byte-identical to patches/unilid_max_sentence_length.patch after a
stash round-trip.

Doc numbers moved to the corrected generation in UNILID/README.md and
UNILID/REPRODUCING.md: c = -21 -> -17, macro F1 0.929/0.957 -> 0.933/0.956, UDHR
0.859/0.838 -> 0.856/0.842, CommonLID 0.845/0.860 -> 0.848/0.862 and 0.723/0.715
-> 0.722/0.717, out-of-set lines 32,901 -> 25,884 becomes 32,525 -> 25,994. Two
REPRODUCING claims were rewritten rather than substituted (the constant no longer
lowers every row; the "released file is unchanged" sentence has no referent). Two
further stale claims were caught during the sweep and are NOT in the drafted
diff: README's add-language and container sections said "the released model"
holds 0.2 real-token mass, which the overwrite makes false, so both now name the
2026-08-11 release by date; and "The original release's uncalibrated behavior is
one flag away" became "Base (uncalibrated) inference is one flag away", since
calibrated=False on the overwritten file no longer reproduces the 2026-08-11
model.

Published numbers re-measured directly on the store file before publishing, not
carried from the readiness pass: at c = -17 the clamp modifies 1,655 of 1,940
rows and leaves 285; with the special columns NOT excluded (the 0.2.1 path) it
modifies 0 of 1,940; real-token mass 1.000000 on every row; real-column row
minima -18.3292 to -11.6063, median -16.0486. The calibration JSON's constants
block reads unseen_token_constant -17.0, head_n 18000, replacement_min_n 100000,
proximity_bound 21.0, min_calib_lines 200. The 1,807-changed-predictions figure
in the model card was traced to its source (pr3_body_v2.md) and is base mode on
the 250,000-line golden subset, 699 gold-correct gained and 669 lost, accuracy
0.9603 -> 0.9604; the card now says base mode.

Hub upload: one atomic commit e0a524ed on cmeister/unilid-1940 overwriting
unilid-1940-calibrated.unilid and calibration.json in place and replacing the
model card, per the author's 2026-08-24 decision. Dry run first, then --confirm.
All three verified after upload by downloading and hashing: 135404c8..., 1ef3063b...,
and the card byte-identical to MODEL_CARD_final.md. Prior revision 8d4044d2
confirmed still to list the 2026-08-11 file (61d7f5fe...), and the model card
names that revision as the route to the superseded weights. The corrected
UNcalibrated version-1 model was NOT uploaded: the 2026-08-17 decision called for
it but the 2026-08-24 decision named three files, so it is deferred and flagged,
not cancelled.

Polybox sweep: the only actual link lived at UNILID/README.md:74. Removed with
its table row, and the "Both files contain the same trained model" sentence
repaired. Textual mentions cleared at UNILID/README.md:122,
UNILID/REPRODUCING.md:11 and :110, and OPEN_SOURCE_STATUS.md:39. Left in place as
records of past state, by their own headers: OPEN_SOURCE_DESIGN.md:359 ("Written
2026-08-10 ... IMPLEMENTED AND SHIPPED"), OPEN_SOURCE_HANDOFF.md:32 and :51
("SUPERSEDED 2026-08-11 ... Kept for the original spec and provenance"),
RERELEASE_PLAN.md:18 and :331, EXPERIMENTS_PLAN.md:930 and :982,
EXPERIMENTS_CHRONOLOGICAL.md:1575.
