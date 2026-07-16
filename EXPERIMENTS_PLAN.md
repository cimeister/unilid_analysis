# UniLID Analysis — Experiment Plan

> **Reconstruction provenance.** Rebuilt on 2026-05-27 after the original session
> transcript (`9729f7f3-3af8-42d5-818a-1f032a9f6f25`, 2026-03-26 → 2026-04-08) was lost.
> Statuses are inferred from `EXPERIMENTS.md`, the analysis code, generated outputs, and the
> recovered prompt history. The session ended mid-stream (last activity 2026-04-08); the
> "not started" items below are the live frontier where work stopped, not abandoned ideas.

Status values: `not started` | `waiting on dependency` | `ongoing` | `finished` |
`abandoned`. Finished items link to `EXPERIMENTS_RESULTS.md`; run records are in
`EXPERIMENTS_CHRONOLOGICAL.md`.

---

## Finished

| # | Experiment | Status | Results |
|---|-----------|--------|---------|
| 1 | Multi-system comparison (UniLID vs DeepSeek/Qwen/Marg/fastText) across length, resource, script; confusion clusters | `finished` | `EXPERIMENTS_RESULTS.md` Exp 1 |
| 2 | Tokenization length bias: token-delta analysis + pairwise normalization counterfactual + full normalized re-classification | `finished` | `EXPERIMENTS_RESULTS.md` Exp 2 |
| 3 | Per-language distribution analysis (KL vs base, related-pair KL/MAD, top divergent tokens, EM-noise diagnosis) | `finished` | `EXPERIMENTS_RESULTS.md` Exp 3 |
| 4 | Token classification of divergent tokens (8-category heuristic) | `finished` | `EXPERIMENTS_RESULTS.md` Exp 4 |
| 5 | Alpha sweep (partial length normalization `score / n_tokens^alpha`) | `finished` | `EXPERIMENTS_RESULTS.md` Exp 5 |
| 6 | Log-probability floor sweep | `finished` | `EXPERIMENTS_RESULTS.md` Exp 6 |
| 7.1 / 7.4 / 7.5 | Training-data analysis: domain distribution, corpus quality, script verification | `finished` | `EXPERIMENTS_RESULTS.md` Exp 7 |
| 8a | Heuristic discriminative weighting (variance-based, setups A/B/C) | `finished` | `EXPERIMENTS_RESULTS.md` Exp 8a |
| 9 | Distribution transfer for low-resource languages (9a related-language, 9b script-average) | `finished` | `EXPERIMENTS_RESULTS.md` Exp 9 |

---

## Not started (frontier where the session stopped)

### 8b — MMI discriminative fine-tuning
- **Status:** `not started` (designed, not implemented).
- **Idea:** gradient-based optimization of per-language weight vectors within confusion
  clusters using a softmax cross-entropy (MMI) objective.
- **Evidence of design intent:** discussed in the `/ultraplan` exchange on 2026-04-06
  (recovered prompts 81–88, "is there a more principled way... optimizing for...").
  `analysis/discriminative_finetune.py` marks 8b as `TODO` in its docstring; no
  implementation or submission script exists.
- **Open design question (recovered prompt 82):** what exactly to optimize for, and whether
  cluster-local fine-tuning can preserve the calibration of the EM-trained weights.

### 7.2 — Training-data mislabeling analysis
- **Status:** `not started` (explicitly deferred, recovered prompt 82, "Skip analysis 7.2
  and 7.3 for now. We can possibly revisit later.").
- **Idea:** run the model on its own training data to surface systematic mislabeling.

### 7.3 — Training-data overlap analysis
- **Status:** `not started` (deferred with 7.2).
- **Idea:** exact-duplicate and n-gram overlap between confusable language pairs.

### Per-script / per-group normalization
- **Status:** `not started`.
- **Idea:** different normalization exponent per script family, or normalize only within
  confusable groups, rather than one global alpha. Listed under future work in
  `EXPERIMENTS.md`.

---

## Active program: hierarchical pooling (started 2026-06-26)

Direction chosen after the 2026-06-24 error analysis (see `EXPERIMENTS_RESULTS.md`
Experiment 10): hierarchical empirical-Bayes pooling of the per-language Unigram
distributions toward script/family group means, with shrinkage gated by each
language's flatness and distance to its nearest confuser. Objective is macro-F1 /
fairness. Modularity is a hard constraint (adding a language stays a single
per-language estimation under a frozen group prior), which is why discriminative /
MMI training was rejected. Full plan file: `~/.claude/plans/yes-do-both-then-giggly-sprout.md`.

| # | Item | Status |
|---|------|--------|
| 10 | Error analysis (qualitative + per-token decomposition + weight-matrix audit) | `finished` (`EXPERIMENTS_RESULTS.md` Exp 10) |
| 11 | Per-language diagnostic (flatness / distance-to-confuser classifier) -> `analysis/diagnostic.py`, `outputs/diagnostic/lang_diagnostic.csv` | `finished` |
| 12 | CommonLID external validation (UniLID, macro-aware) | `finished` (`EXPERIMENTS_RESULTS.md` Exp 12; trends partially hold) |
| 13 | Stage 1 post-hoc gated shrinkage prototype -> `analysis/hierarchical_pool.py` | `finished` — **NEGATIVE** (`EXPERIMENTS_RESULTS.md` Exp 13; shrinkage reduces macro-F1) |
| 14 | Stage 2 MAP-EM re-estimation on Apertus 200k with the script-mean prior | `abandoned` — MAP-EM is the same fixed point as Stage 1, so it would reproduce the negative result. Replaced in the redirect by a STANDARD-setup Apertus 200k retrain (no prior) to test vocab coverage: see `EXPERIMENTS_RESULTS.md` Exp 15 (mixed; hurt the tail). |
| 15 | Stratified macro-F1 evaluation (val/test split, bootstrap CIs) | `finished` (built into Stage 1; reusable for the redirect) |

### Redirect after the Stage 1 negative result
- **Per-language frequency prior** (`b_L = gamma*log N_L`, Rust biased scorer): `finished` —
  **POSITIVE** (`EXPERIMENTS_RESULTS.md` Exp 14). gamma=0.5 gives test macro-F1 +0.0058
  [CI +0.0048,+0.0069], accuracy +0.0032, head/twins up, tail/magnets unharmed. First
  modification to beat baseline with a CI excluding zero. `analysis/prior_sweep.py`.
- **Expanded magnet shrink + entropy-sharpening sweep:** `finished` — NEGATIVE. Sharpening
  catastrophic (magnets -> ~0), shrink neutral at best. `outputs/tables/hierarchical_pool.md`.
- **Learned per-language bias** (free `b_L` fit on val by regularized softmax over each
  example's top-k candidates; Rust `top_k_of_cached_weight_sets_batch`): `finished`,
  **POSITIVE** (`EXPERIMENTS_RESULTS.md` Exp 14). Guard-revised 2026-07-10 (job 2731802):
  selected reg=5.0, test macro-F1 0.9454 -> 0.9567 (+0.0112 [CI +0.0099,+0.0124]),
  accuracy +0.0147, head/twins up, magnets flat, tail point estimate -0.0320 (CI touches
  0; the val guard is tail-blind, see Exp 14). The earlier reg=0.3 result (+0.0180) is
  superseded. `analysis/learned_prior.py`.
- **CommonLID out-of-domain validation of the priors:** `finished` (`EXPERIMENTS_RESULTS.md`
  Exp 14). With the guarded reg=5.0 bias (job 2731818): baseline 0.8452 -> freq prior
  0.8518 -> learned bias 0.8879 (+0.0427). Both transfer; learned is larger (CommonLID is
  all common languages). The 0.8936 number from the reg=0.3 vector is superseded.
- **Apertus 200k vocab retrain + frequency prior on it:** `finished`, **MIXED / cautionary**
  (`EXPERIMENTS_RESULTS.md` Exp 15). Jobs 2639097 (timeout) + 2641940 (resume, complete) +
  2649123 (prior). The 200k vocab raises accuracy (+0.41pp) but LOWERS tail macro-F1 (-3.4pp);
  the frequency prior at the guard-selected gamma=3.0 boosts overall +0.0203 while collapsing
  tail (-0.0945) and magnets (-0.1102). Exposed a guard flaw (protects only twins/head).

## Next set of methods to test

Ordered by priority. The learned bias (Exp 14) is the current best result and the template.
Items 3-7 were added 2026-07-10 from the assessment of the results record; items 3, 4, and 7
extend the calibration template, items 5-6 are analyses that pick the direction after it.

1. **Fix the selection guard to protect all strata** (`analysis/prior_sweep.py`,
   `learned_prior.py`, `hierarchical_pool.py`): require that no stratum (tail, magnets,
   twins, head) regresses on val beyond a tolerance, not just twins/head; also require the
   candidate to beat the val baseline overall. Re-select and re-report on both models.
   `finished` (2026-07-10; jobs 2731802/2731803/2731804 + CommonLID re-eval 2731818;
   results in `EXPERIMENTS_RESULTS.md` Exp 14, Exp 15, and Invalidated / superseded).
   Rationale: Exp 15 showed "overall macro-F1" can rise while the tail collapses, because
   the 1,940-language macro-average is dominated by the head; the guard must reflect the
   macro-F1 / fairness objective.
   Rule adopted: `GUARD_TOL = 0.01` on all four strata plus required val overall
   improvement (`passes_guard` in `analysis/hierarchical_pool.py`; decision record in
   `EXPERIMENTAL_SETUP.md` "Selection guard"). A CI-only guard was rejected: the small
   strata have CIs wide enough that even the Exp 15 tail collapse (test CI upper bound
   exactly 0) would pass.
   Outcomes: 100k frequency prior keeps gamma=0.5 (numbers unchanged); Apertus frequency
   prior rejected (no eligible gamma); learned bias re-selected at reg=5.0, overall
   +0.0112, CommonLID 0.8879 (the reg=0.3 result, +0.0180 / 0.8936, is superseded).
   New finding: val tail macro-F1 is constant (0.8710) across ALL configs in every sweep,
   so the val guard has no sensitivity on the tail stratum; the reg=5.0 test tail delta
   is -0.0320 with CI touching 0. This raises the priority of item 10 (split redesign).

2. **Learned per-language bias on the Apertus 200k model** (`slurm_learnprior_apertus.sh`,
   to write): the precise instrument, not the blunt frequency prior. Expected to give a
   clean gain that protects the Apertus tail, unlike the frequency prior (Exp 15). `not
   started`. Confirms whether the learned-bias win is model-agnostic. Should inherit the
   fixed guard (item 1) and the prior-centered regularizer (item 3) if that lands first.

3. **Regularize the learned bias toward the frequency prior** (`analysis/learned_prior.py`):
   replace `reg*||b||^2` with `reg*||b - gamma*log(N_L+1)||^2`, gamma fixed at 0.5 or
   selected on val jointly with reg. Two effects: b_L for languages with few val examples
   shrinks toward a sensible default instead of toward 0, and a newly added language gets
   the starting bias `gamma*log N_new`, which restores the add-a-language modularity story
   for the learned variant. One-line change to the loss plus a re-run. `not started`.

4. **Length-aware calibration** `score + b_L + c_L * n_tokens(L, x)`: a per-token offset
   c_L targets the Exp 10 mechanism directly (the per-language smoothing floor is
   resource-tied, corr(floor, log10 N) = -0.966, and 86.4% of the stolen score margin
   comes from short non-content tokens), which a constant b_L cannot correct. Scoring
   needs no new Rust: adding c_L to every entry of row L of the weight matrix is exactly
   a per-token offset (`set_weight_sets`, with b_L via the biased scorer). The softmax
   fit stays convex (logits linear in b and c). The top-k extraction for the fit needs
   per-candidate token counts: a small extension of `top_k_of_cached_weight_sets_batch`,
   or a fixed-segmentation approximation validated by exact rescoring (the same
   reproduction check used for gamma=0). Caveat to validate: folding c_L into a row also
   shifts that language's Viterbi segmentation. `not started`.

5. **Interpret the fitted b_L** (`outputs/tables/learned_bias.npy`): correlation with
   log N_L, and whether the most negative offsets are the diagnosed flat magnets
   (kzn, tly, vol, ido, mlt, qus). If the discriminative fit independently rediscovers
   the Exp 10 magnet list, the diagnosis and the fix corroborate each other. Pure
   analysis, no new runs. `not started`.

6. **Residual error decomposition on the learned-bias model**: rerun the Exp 10 cuts on
   the errors the selected learned-bias model still makes (test half), and size the
   macrolanguage-split share by applying the SIL macro-aware scoring (mapping from
   `analysis/commonlid_eval.py`) to the GlotLID test sample. The outcome picks the next
   method (twin tie-breaking vs short-text handling). `not started`.

7. **Unsupervised prior adaptation to a target domain**: re-estimate the class prior
   (equivalently b_L) on unlabeled target text by EM over the model's posteriors
   (Saerens et al., 2002), starting from the GlotLID-fit bias; evaluate on CommonLID
   without using its labels. Separates the attractor-suppression part of the learned-bias
   gain from the part that fits the GlotLID label distribution, and addresses the caveat
   that the fitted bias encodes the tuning distribution's label prior. `not started`.

8. **Diagnose the Apertus tail regression** (`analysis/diagnostic.py` on the Apertus model):
   compare per-language tail F1 100k vs Apertus; test whether the 200k byte-level vocab
   fragments rare-tail languages (longer token sequences, flatter per-language
   distributions) or whether it is a recovered-data mismatch. `not started`.

9. **Constrained / capped prior variants** (if 1-4 leave residual tail cost): a per-language
   bias with an explicit per-stratum fairness penalty in the fit objective, or a prior that
   excludes tail languages (`N < threshold`) from down-weighting. `not started`, contingent.

10. **Multi-seed / variance and final numbers** for the reported deltas: current CIs are
    single-run bootstraps over examples; add resampled val/test splits to bound
    split-selection variance. In addition, score the final selected configurations once on
    the full test set minus the val examples (~45.4M lines) and use those as the headline
    numbers; the current test half is only 250k examples. `not started`.

Data note: scratch was purged; the original GlotLID `train.txt` (60,683,151 lines),
`glotlid_correct_test.txt`, train counts, all 5 prediction files, and the full 744 MB
model were recovered from the Google Drive folders in `SETUP.md` on 2026-06-26. The
custom Rust tokenizers build was rebuilt (`maturin develop --release`).

## Notes for whoever resumes this

- All evaluated scoring modifications (Exp 2, 5, 6, 8a) reduced 500k-sample accuracy
  relative to the 0.960 baseline.
- All experiments are post-hoc analyses over the fixed `glotlidc.unilid` weights; none train
  a model. 8b would be the first to modify weights via optimization and requires
  infrastructure not present in the repo (a training loop, not the sweep harness used for
  5/6/8a/9).
- The last three recovered prompts (2026-04-08) were about consolidating results into
  `EXPERIMENTS.md`, identifying files to archive, and re-reading the method and
  discriminative-training direction.
