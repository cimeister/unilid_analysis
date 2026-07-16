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
most-recent-first. All jobs ran on CSCS Clariden (account `a139`, partition `normal`),
Python `/users/cmeister747/.pyenv/versions/3.11.5/bin/python3`, working dir
`/users/cmeister747/unilid_analysis`, data on scratch (`config.DATA_DIR`). See `SETUP.md`
for the infrastructure record.

---

## Family: Error analysis + hierarchical pooling (Exp 10–15)

**Window:** 2026-06-24 to present. Plan items: `EXPERIMENTS_PLAN.md` Exp 10–15. Setup:
`EXPERIMENTAL_SETUP.md` (hierarchical pooling). Full plan:
`~/.claude/plans/yes-do-both-then-giggly-sprout.md`.

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

## Checkpoint / artifact-deletion assessment

No model checkpoints are produced by this project (all experiments are post-hoc analyses
over the fixed `glotlidc.unilid` weights). Large artifacts that exist:
`glotlidc.unilid` (~59 MB in repo; the 744 MB model is on scratch per `SETUP.md`), and the
410 MB prediction files in `full_prob/` and `glotlid_e100_sanity/`. No deletion is proposed
here; per project policy, deletion requires explicit per-artifact approval.
