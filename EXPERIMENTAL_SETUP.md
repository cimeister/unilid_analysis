# UniLID Analysis — Experimental Setup

> **Reconstruction provenance.** Rebuilt on 2026-05-27 after the original session
> transcript was lost. This file records *experimental design* choices and search spaces;
> the values below are read directly from the analysis code (`analysis/config.py` and the
> sweep modules) and the SLURM scripts, so they are code-accurate, not transcribed from
> prose. **Infrastructure** (cluster, filesystem, memory requirements, build steps, data
> re-download links, known gotchas) is in the existing `SETUP.md` — that file is not
> duplicated here; this one complements it with the "why" behind the experimental
> parameters. Where the rationale is inferred from the recovered prompt history rather than
> stated in code, it is marked **[inferred]**.

---

## System under evaluation

**UniLID** — language identification by a single shared Unigram tokenizer (100k vocabulary)
with 1,940 per-language log-probability weight vectors. Per-language weights are estimated
by Unigram EM (20 iterations, soft or hard) starting from the base tokenizer distribution
on each language's own corpus; the vocabulary is fixed, only probabilities change.
Regularization is implicit only: a probability floor and convergence early stopping (no
Dirichlet prior, no damping).

**Scoring.** For a text, each language scores `score(lang) = Σ log p(token_i | lang)` under
that language's own Viterbi segmentation (so token counts differ across languages). The
prediction is the argmax over all 1,940 languages.

**Model file.** `glotlidc.unilid` (weights matrix 1940 × 100k float32; ~744 MB full copy on
scratch, a ~59 MB copy in the repo). Loaded via `from unilid import load_model`.

---

## Evaluation data and sampling

- **Test set:** GlotLID test, `TOTAL_LINES = 45,627,279` samples, fastText format
  (`config.TEST_FILE`). Must be unzipped from `glotlid_correct_test.txt.zip` first
  (`SETUP.md` gotcha 2).
- **Default sample:** `DEFAULT_SAMPLE_SIZE = 500_000`, `SAMPLE_SEED = 42`, uniform
  **without replacement** (`analysis/sample_data.py`). The same seed is reused across all
  sweeps so results are comparable run-to-run. **[inferred rationale]** the user confirmed
  during the session that sampling is without replacement and asked for uniform coverage
  across languages rather than head-of-file sampling (recovered prompts 25, 54).
- **Sample pickle does not store raw texts** — only `y_true`, text lengths, train counts,
  and all five models' predictions. This was a deliberate choice after a full-dataset pickle
  with texts OOM'd at 128 GB (`SETUP.md` gotcha 3). Scripts needing raw text
  (`length_bias`, `normalized_predict`) stream the test file.
- **Full-dataset runs** use `--sample-size 45627279` (Exp 1 tables, Exp 2 length bias).

## Models compared (Exp 1)

From `config.PRED_FILES` (prediction files on scratch, one label per system):
`UniLID` (primary), `UniLID-DeepSeek` (DeepSeek v3.2 variant), `UniLID-Qwen` (Qwen3-8B
variant), `UniLID-Marg` (marginalized variant), `fastText` (e100 baseline). Each is a
precomputed `y_pred.txt` aligned line-for-line with the test file.

## Binning (`analysis/config.py`)

- **Text length (chars):** bins `[0, 30, 75, 150, 300, ∞]` → labels
  `<30, 30–75, 75–150, 150–300, 300+`.
- **Resource level (training doc count):** bins `[0, 500, 1000, 12000, 18000, 35000, ∞]` →
  `<500, 500–1k, 1k–12k, 12k–18k, 18k–35k, 35k+`. **Design note:** the `<1k` tier was split
  into `<500` and `500–1k` partway through (recovered prompts 23–24, "add the split at
  500"); the bin boundaries above ≥1k are quartiles of the remaining languages, rounded.
  The code comment in `config.py` ("Bin 1: <1k (fixed)") predates the 500-split and is
  stale; the bin array is the source of truth.
- **Scripts (Exp 1 table 4):** top 10 by sample count
  (`Latn, Cyrl, Arab, Grek, Deva, Hang, Hebr, Beng, Jpan, Armn`), rest grouped as "Other".

## Confusion clusters (`config.CONFUSION_CLUSTERS`)

Seven hand-defined clusters used for confusion matrices (Exp 1.4), distribution pair
analysis (Exp 3.4), token classification (Exp 4), and discriminative weighting (Exp 8a):
Arabic dialects (6 langs), Chinese varieties (5), Hindi belt / Devanagari (8),
Malay–Indonesian (3), Scandinavian (4), Hebrew (2), Persian–Iranian (4).

## Metrics (`analysis/metrics.py`)

- **Accuracy:** exact match.
- **Macro F1:** unweighted mean of per-class F1, **averaged over `set(y_true)` only**
  (sklearn convention). The original code averaged over `set(y_true) | set(y_pred)`, which
  added phantom zero-F1 terms for predicted-but-absent labels; this was found via a
  code-review agent and fixed (recovered prompt 20; `SETUP.md` gotcha 5). Any macro-F1
  number predating that fix is not comparable to the reported values.
- **Macro FPR:** unweighted mean of per-class false positive rate, displayed ×10⁵.

---

## Sweep search spaces (read from code)

These are the hyperparameter grids and the selection criterion for each sweep. All sweeps
run on the 500k sample unless noted. Search procedure: exhaustive grid (no adaptive search),
single seed (42).

| Experiment | Parameter | Grid (from code) | Source |
|-----------|-----------|------------------|--------|
| Exp 5 alpha | normalization exponent `alpha` in `score / n_tokens^alpha` | `{0.0, 0.1, 0.2, …, 1.0}` (11) | `normalized_predict.ALPHA_VALUES` |
| Exp 6 floor | weight clamp floor (log-prob) | `{None, -22.0, -15.0, -10.0}` | `floor_sweep.FLOOR_VALUES` |
| Exp 9 transfer | interpolation weight `lambda` (1.0 = baseline, no transfer), two approaches 9a/9b | `{0.0, 0.1, …, 1.0}` (11) × 2 | `transfer_sweep.LAMBDA_VALUES` |
| Exp 8a discriminative | Setup A (additive up-weight) and B (z-scored rescale): `alpha`; Setup C (sigmoid gate): `beta` | A/B `α ∈ {0.0, 0.5, 1.0, 2.0, 5.0}`; C `β ∈ {1.0, 5.0, 10.0}` | `discriminative_finetune.py` (`alpha_values_ab`, `beta_values_c`) |

**Selection criterion** for all sweeps: overall accuracy on the 500k held-out sample
(with macro-F1 and per-text-length / per-resource breakdowns as secondary diagnostics).
`alpha=0` / `lambda=1` / `floor=None` are the no-op baselines and each was validated to
reproduce the original predictions exactly (100% agreement), which serves as an
implementation check on the modified Rust scoring path.

**Important caveat on "tuning":** every sweep here tunes against the *test* sample, not a
separate validation split. This is acceptable for the project's actual question, which is
diagnostic ("does modifying the scoring help, and where") rather than producing a deployed
model selected on held-out data. It would not be acceptable to report any swept value
(e.g. `alpha=0.1`) as a tuned production setting without a separate validation split. Flag
this if the framing changes. **[inferred]** — the framing is diagnostic throughout the
recovered prompts; no validation/test split was ever set up.

---

## Code-modification record (Rust tokenizer fork)

The normalized-scoring experiments required changes to the Rust Unigram tokenizer in
`UNILID/tokenizers/` (a fork), rebuilt with `maturin develop --release` (`SETUP.md`
gotcha 6):
- `best_of_cached_weight_sets_normalized` (Viterbi DP) with an `alpha: f32` parameter,
  `score / n_tokens^alpha`; `alpha=0` reduces to raw scoring.
- PyO3 bindings + Python `predict_normalized` / `predict_normalized_batch` wrappers.

The floor sweep (Exp 6) and transfer sweep (Exp 9) need **no** Rust changes: they clamp /
interpolate the weight matrix in Python (`numpy`, memmap of the `.unilid` file) and push the
modified weights to the Rust cache via `set_weight_sets()` before predicting.

## Hierarchical pooling program (Exp 11–15, 2026-06-26)

**Goal:** raise macro-F1 / fairness by correcting the under-fit low-resource tail
(Exp 10), while preserving UniLID's modularity (a new language is one per-language
estimation under a frozen group prior; this rules out discriminative / global training).

**Diagnostic (`analysis/diagnostic.py`):** per-language features computed from the
1940x100k weight matrix: within-script flatness z-score `zH` (median/MAD of entropy),
full pairwise symmetric KL (one matmul `P @ logP.T`), nearest-confuser distance `d_nn`,
nearest higher-resource same-script distance `d_up`, promiscuity `k_close` (count within a
5th-percentile KL radius), and an empirical magnet ratio `FP/(support+1)` from the
validation half of the 500k sample only. Classes: `flat_magnet` (shrink hard),
`tight_lowres` (shrink gently toward `target_lang`), `twin` (do not pool), `isolated_tail`
(mild shrink to script mean), `head`/`mid` (protected). Thresholds are data-driven
quantiles (constants documented in the module header), not hand-tuned.

**Group means (empirical Bayes):** script-level mean (family-level is a later refinement),
resource-weighted (`w = min(N, cap)`), leave-one-out, with a language's own near-confusers
(`symKL < tau`) excluded from its prior so twins are never blurred. Backbone for the frozen
Stage-2 prior = languages with `N >= 18,000`.

**Stage 1 (post-hoc, `analysis/hierarchical_pool.py`):** shrink each row in probability
space `p' = (1-lambda_L) p + lambda_L m_g`, lambda gated by category, swept on the
validation half, evaluated once on the test half. Reuses the
memmap -> `set_weight_sets` -> `predict_batch` -> `compute_metrics` pattern from
`transfer_sweep.py`. Decision gate before any retrain.

**Stage 2 (retrain):** Apertus V2 200k vocab as a fixed Unigram vocabulary
(`train.py --initial-vocab .../apertus_v2_200k/tokenizer.json --vocab-size 200000
--byte-level --per-lang-counts-method soft`); MAP M-step
`p_L = (N_L c_L + alpha_L m_g)/(N_L + alpha_L)` injected at
`UNILID/unilid/trainers/em_trainer.py` (the soft per-language path; NOT
`_counts_to_log_probs`). Bridge from Stage 1: `alpha_L = N_L lambda_L / (1 - lambda_L)`.
Two passes: Pass 1 estimates and freezes `group_priors.json` from the backbone; Pass 2 is
MAP soft-EM for all 1,940. Training is pure-Python EM (no Rust build needed); only eval
needs the build + ~250 GB cache (SLURM 400 GB).

**Evaluation protocol (fixes tuning-on-test):** deterministic 50/50 val/test split of the
seed-42 500k sample (position parity, mask saved at `outputs/diagnostic/val_mask.npy`); all
hyperparameters and the diagnostic magnet signal use val only; test evaluated once.
Stratified macro-F1 over `tail` (N<1k) / `twins` / `head` plus overall, with 1,000-resample
item-level bootstrap CIs; accept only if no stratum regresses beyond its CI. External
validation on CommonLID (web domain, bare ISO 639-3 labels mapped via
`analysis/commonlid_map.py`).

**Selection guard (revised 2026-07-10).** All val-based selections in
`analysis/{hierarchical_pool,prior_sweep,learned_prior}.py` use one rule (`passes_guard`
in `analysis/hierarchical_pool.py`): a candidate config is eligible only if its val
overall macro-F1 beats the baseline and no guarded stratum (tail, magnets, twins, head)
drops by more than `GUARD_TOL = 0.01`; among eligible configs the one with the highest
val overall macro-F1 is selected, and if none is eligible the baseline is selected and
the negative result reported. Provenance of the rule: the original guard covered only
twins/head at tolerance 0.002 and selected the tail-collapsing gamma=3.0 on the Apertus
model (Exp 15). A CI-only rule was considered and rejected because the small strata have
bootstrap CIs wide enough that even that collapse would pass (its test tail CI upper
bound was exactly 0), so the rule needs a point-estimate floor. The 0.01 tolerance was
chosen by the user on 2026-07-10 from the consequences computed on the saved val tables
at tolerances {0.002, 0.01, 0.02}: 0.01 keeps the frequency prior at gamma=0.5 on the
100k model (val magnets -0.0081) and de-selects reg<=3 for the learned bias (val magnets
-0.0155 to -0.0318), so the Exp 14 learned-bias headline is superseded by the re-run at
the re-selected reg. `learned_prior.REGS` was extended from {0.3, 1, 3, 10} to
{0.3, 1, 3, 5, 7, 10} (2026-07-10) because the guarded region's boundary lies between
reg=3 (fails on magnets) and reg=10 (passes), and the original grid would select its own
endpoint.

## Reproducibility limitations of this record

- Only one git commit exists (`b7508fd`, 2026-04-08); per-experiment code versions are not
  separately tracked. Source-file mtimes are the only finer-grained timing signal.
- SLURM submissions did not log seeds/commit/launch-command into the output beyond the
  submission scripts themselves; the scripts in `slurm_*.sh` are the reproducibility record
  for each job (kept in the repo).
- The recovered prompt history (`EXPERIMENTS_CHRONOLOGICAL.md` cites specific prompt numbers)
  is the only surviving record of design rationale that was not written into code or
  `EXPERIMENTS.md`.
