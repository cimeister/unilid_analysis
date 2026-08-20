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

## Camera-ready evaluation program (2026-08-06; pre-registered before any run)

User decision 2026-08-06: incorporate the promoted configuration into the ICML
camera-ready (new Table 1 row, mechanisms subsection, appendix) and run E1-E4.
Governing conventions: `EXPERIMENTAL_SETUP.md` "Camera-ready reporting
conventions" and adoption-rule amendment 9. Full working plan with the
adversarial-review corrections: `~/.claude/plans/steady-finding-abelson.md`
(11 findings, all folded in). Proceeding on fallback paths (user instruction:
act as though the paper team's files are not available); the ask list is with
the user to send.

- **E1: common reporting set** | status: finished (2026-08-07; results in
  `EXPERIMENTS_RESULTS.md` "Camera-ready E1"; full pool 0.9292/0.9569/0.9443,
  judge 0.9117/0.9498/0.9332, both bootstrap intervals above zero) | Recover
  `fasttext_y_pred.txt` and `glotlidc_y_pred.txt` from the recorded Drive
  folders (verified live 2026-08-06, byte sizes 410,645,511 = 45,627,279 x 9);
  import to int16 memmaps (blocking gates: exact line count, 100% agreement
  with `sample_500k_all.pkl` on the re-derived seed-42 indices); comparability
  measurements (non-blocking, pre-registered branches): fastText full-pool
  macro F1 vs the paper's .944 within 0.005, imported UniLID file vs
  `pred_baseline.npy` agreement (recorded context 0.9951). Then macro F1 +
  macro FPR for {baseline, gate_flat4_prox21, fastText} on {full pool, judge
  part}, after wiring gates (carried CSV 1e-9; judge 0.9117/0.9498; full pool
  0.9292/2.0263e-5); paired bootstrap (B=10,000, seed 0, percentile 95%, over
  the 1,940 languages) for (promoted - baseline) and (promoted - fastText) on
  the judge part. Scripts `analysis/import_external_pred.py`,
  `analysis/paper_eval.py`; outputs `outputs/tables/paper_eval.md` + `.tex`.
- **E2: UDHR and FLORES-200 transfer test** | status: finished (2026-08-07,
  job 3028291 + login-node eval; both acceptance gates passed: baseline UDHR
  0.858977 vs printed .859, FLORES 0.931741 vs .932, and both baseline FPR
  cells reproduce exactly; transfer outcome: FLORES 0.9317 -> 0.9326 gated,
  UDHR 0.8590 -> 0.8383 gated, the pre-registered balanced-set reversal;
  label sets independently confirmed by the paper team's lists) | Rebuilt
  from `cis-lmu/udhr-lid` (366-label
  intersection, matching the paper's count exactly) and, amended 2026-08-07
  from the pre-registered `flores_plus` source, the original FLORES-200
  devtest tarball: the official 204-code list intersects the 1,940 labels at
  exactly the paper's 190 while flores_plus gives 205, so the original release
  is the paper's basis (`outputs/tables/external_bench_mapping.md`, addendum).
  Label mapping table user-approved before scoring; acceptance gate:
  reproduce the paper's UniLID cells (UDHR .859, FLORES .932) within 0.005
  before any new-configuration number. Scoring on SLURM unconditionally (full
  model load). Script `analysis/external_bench_eval.py`: baseline pass under
  W; floor-21 pass under sha-verified W_f21 persisting top-5 ids/scores; gate
  with `tau_floor21_gate.csv` + `tau_flat4.csv` unchanged, disjoint groups
  each from a fresh floor-21 prediction copy, replacement via
  `_walk_replacement(21.0, 100000, ...)`. All constants transferred without
  refitting (provenance stated per the conventions entry).
- **E3: Mistral-Nemo variant** | status: finished (2026-08-09; results in
  `EXPERIMENTS_RESULTS.md` "Camera-ready E3": baseline 0.9132 -> gated 0.9538
  full pool, judge bootstrap +0.0504 [+0.0438, +0.0573]; the mechanisms
  transfer with a larger gain than on the base model; paper presentation
  awaiting the user's confirmation of the appendix recommendation) |
  Pin the local HF tokenizer snapshot revision; per-language
  fixed-vocabulary EM (fp64 spm_train) over the 1,940 corpus files;
  `analysis.degeneracy_scan` before evaluation; full-pool baseline pass; the
  variant's floor-21 matrix; tau recalibration (CALIB_MAX=2000, CALIB_SEED=0,
  MIN_CALIB_LINES=200, q_L = 5*(1 - min(N,18000)/18000)); flat set by the
  recorded rule (one scoring pass over the retired 250k validation half for
  magnet_ratio; user decision 2026-08-06); full-pool floor-21 pass banking
  top-5; gate; eval vs the variant's own baseline. Both paper rows for the
  variant come from this retrain; the submission's row is left untouched with
  a footnote. Sequenced last.
- **E5: CommonLID for the camera-ready** | status: finished (2026-08-07, job
  3031609 + login-node eval; results in `EXPERIMENTS_RESULTS.md` "Camera-ready
  E5": gated accuracy 0.8604 vs baseline 0.8452, tag F1 0.7149 vs 0.7228, all
  gates passed at exact equality; integrated as tab:commonlid) |
  Evaluate the promoted configuration on CommonLID (373,230 lines, 109 bare
  ISO-639-3 tags, web domain) under the recorded conventions
  (`analysis/commonlid_eval.py`: script stripped from predictions, correct if
  equal to the gold tag or a member of the gold macrolanguage per the SIL
  table, plus the documented tgl-fil bridge; metrics = accuracy and tag-level
  macro F1 via `compute_metrics` after mapping). One SLURM scoring pass under
  the sha-verified floor-21 matrix banking top-5 ids/scores (the recorded
  Exp 39 pass discarded them), then the gate via the shared
  `_gate_walk_and_merge` (the self-checked E2 code path), thresholds
  unchanged. Wiring gates before any new number: reproduce the recorded
  baseline accuracy 0.8452 and tag-level macro F1 0.7228 (Exp 12/39) and
  floor-21 0.7181, and match the persisted per-line predictions in
  `outputs/diagnostic/commonlid_carried_preds.npz` for baseline and floor-21.
  Output feeds a table + paragraph in submission.tex (replacing the appendix's
  one-sentence CommonLID mention); no fastText row (no fastText model binary
  is available for CommonLID scoring, only its GlotLID-C predictions).
- **E4: breakdowns and residual re-measurement** | status: finished
  (2026-08-07; results in `EXPERIMENTS_RESULTS.md` "Camera-ready E4"; both
  reproduction gates passed under the within-stratum view; residual of record
  926,299 / 99.15% / 88.64%) | Script/resource-bin aggregation of {baseline, promoted,
  fastText} on the full pool; reproduction gate against the paper's published
  script table (the Hebr row 0.740 vs our 0.6966 and the 1,938-vs-1,940 basis
  are expected mismatches; on mismatch the affected table goes to the user, no
  silently inconsistent second table). Residual close-pair analysis recomputed
  on `pred_gate_flat4_prox21.npy`'s judge-part residual (feeds paper item 5).

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

Item numbers are stable identifiers; entries appear in the order they were added, not in
numeric order. **The state paragraph below is frozen at 2026-07-19 and is retained only
as the context in which these items were written. For current state read
`EXPERIMENTS_RESULTS.md` "Current state (2026-08-06)" and the open items at the top of
this file.** Historical context (2026-07-19): selection runs under the balanced protocol
(item 10 / Exp 22), which encodes the uniform-prior objective; under it the two live
candidates are the punctuation partial pooling (item 15, guard-passed at alpha=300) and
the balanced-data bias refit (item 16, guard-passed at reg=0.3, adoption blocked on
stability, evaluation, and an objective decision). Under the natural-traffic objective
the learned bias reg=5.0 (Exp 16) remains the reference result. The consolidated list of
open paths is at the end of this file.

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
   started`, deprioritized with the bias family (2026-07-18 user decision) and pending
   the item-16 outcome; if ever run, it starts under the balanced protocol.

3. **Regularize the learned bias toward the frequency prior** (`analysis/learned_prior.py`):
   replace `reg*||b||^2` with `reg*||b - gamma*log(N_L+1)||^2`, gamma selected on val
   jointly with reg (grid {0, 0.25, 0.5}; gamma=0 is the plain L2). Two effects: b_L for
   languages with few val examples shrinks toward a default instead of toward 0, and a
   newly added language gets the starting bias `gamma*log N_new` (modularity). Exp 16
   caution: the anchor itself costs full-test tail -0.0182, so large gamma is not
   presumed safe. `finished` (job 2790077, 2026-07-18; `EXPERIMENTS_RESULTS.md` Exp 17):
   guard selects gamma=0.25/reg=10 under the corrected NLL gradient, test-half overall
   +0.0117, marginal over plain-L2 reg=5 (+0.0112); the old operating point fails the
   guard under the corrected gradient. Not a result of record pending full-test
   evaluation, and the whole bias family is deprioritized by the 2026-07-18 user
   decision (tail risk; fitting for a new language needs global val data, breaking
   add-a-language modularity). Items 2 and 4 inherit that deprioritization.

4. **Affine score recalibration** `score'(L) = a_L*score + b_L + c_L*n_tokens(L, x)`
   (refined 2026-07-18 from the length-aware-calibration item): multinomial logistic
   recalibration, convex in (a, b, c); nests Exp 14 (b only). The per-token c_L is the
   degree of freedom row normalization removes (all rows have identical real-token mass
   0.2), equivalently a per-language geometric length model; it targets the Exp 10
   mechanism directly (resource-tied floor, corr -0.966; 86.4% of the stolen score
   margin from short non-content tokens), which a constant b_L cannot correct. Scoring
   needs no new Rust for (b, c): fold c_L into row L via `set_weight_sets` and b_L via
   the biased scorer. The top-k extraction for the fit needs per-candidate token counts
   (small extension of `top_k_of_cached_weight_sets_batch`) or a fixed-segmentation
   approximation validated by exact rescoring. Caveat to validate: folding c_L into a
   row also shifts that language's Viterbi segmentation. Start with (b, c); a_L
   (temperature) only if the residuals justify it. `not started`.

5. **Interpret the fitted b_L** (`outputs/tables/learned_bias.npy`): correlation with
   log N_L, and whether the most negative offsets are the diagnosed flat magnets.
   Partially superseded 2026-07-19: the balanced refit (Exp 23c) answered the
   substantive question for the uniform-prior fit (suppressed languages are head/twin
   sinks, not flat magnets). The natural-traffic vector's interpretation remains
   undone; only worth doing if the paper reports that vector. `not started`, low
   priority.

6. **Residual error decomposition on the current best configuration**: rerun the Exp 10
   cuts on the errors that remain (which configuration is "best" now depends on the
   objective decision; see Open paths). The macro-aware sizing half of this item is
   `finished` via Exp 21 (0.77pp of test-half accuracy is within-macro confusion).
   Decomposition `not started`.

7. **Unsupervised prior adaptation to a target domain** (Saerens et al., 2002):
   re-estimate the class prior on unlabeled target text by EM over the model's
   posteriors. `not started`. Reframed 2026-07-19: this is a natural-traffic-objective
   tool (it adapts the deployment prior); under the balanced selection objective it is
   relevant only to the paper's deployment story, not to selection.

8. **Diagnose the Apertus tail regression** (`analysis/diagnostic.py` on the Apertus model):
   compare per-language tail F1 100k vs Apertus; test whether the 200k byte-level vocab
   fragments rare-tail languages (longer token sequences, flatter per-language
   distributions) or whether it is a recovered-data mismatch. `not started`.

9. **Constrained / capped bias variants**: reframed 2026-07-19 as the follow-up to the
   balanced refit (Exp 23c): cap |b_L| (e.g. at 1-2 nats) or add a per-language recall
   floor to the fit objective, so satellite languages gain without deep suppression of
   any single language (the unconstrained fit puts -8.9 nats on por). Runs only if the
   user accepts a bounded version of the suppression trade. `not started`, contingent
   on the objective decision in Open paths.

10. **Multi-seed / variance and final numbers** for the reported deltas: current CIs are
    single-run bootstraps over examples; add resampled val/test splits to bound
    split-selection variance. In addition, score the final selected configurations once on
    the full test set minus the val examples (~45.4M lines) and use those as the headline
    numbers; the current test half is only 250k examples. Full-test part `finished`
    (job 2784115, 2026-07-18; `EXPERIMENTS_RESULTS.md` Exp 16): learned bias confirmed
    (overall +0.0129, tail -0.0018 [CI -0.0035, -0.0001]); the frequency prior is NOT
    tail-safe at full scale (tail -0.0182 [CI -0.0225, -0.0146]); the reg=0.3-era
    -0.0320 tail scare was split noise. Resampled splits still `not started`; the
    tail-weighted val allocation matters for all future sweeps (65 of 96 tail languages
    have zero val examples). **Split part `finished` (2026-07-19,
    `EXPERIMENTS_RESULTS.md` Exp 22):** balanced protocol implemented
    (`analysis/balanced_split.py`; 188,061-line val, all languages represented, five
    seeds); re-baseline shows freq prior and floor-21 now FAIL selection on the visible
    tail, and the learned bias fails on balanced-val overall (a fitted prior loses
    under the uniform-prior view by construction). Resampled-draw stability checks are
    part of every future selection.

15. **Punctuation back-off / hierarchical prior on non-content columns** (added
    2026-07-19 from the user's reading of Exp 18/dp): the tying experiments localized
    punctuation/digit usage rates as well-estimated twin-discriminative signal at high
    N and noisy theft-channel estimates at low N (Exp 4: 10.5% of twin KL; Exp 10: 20%
    of stolen margin). Instead of the refuted full tie (weight 1), shrink ONLY the
    ~212 neutral digit/punctuation columns toward the within-script mean with the
    standard data-dependent weight `lam_L = alpha/(N_L + alpha)`: head/twin rates
    (N=100k, lam ~ 0) keep their signal; low-N estimates get stabilized. Equivalent to
    a script-level hierarchical (Dirichlet-style) prior on punctuation rates,
    MAP-estimated per language; post-hoc testable with the existing tying machinery;
    modular (new language: its own counts + the frozen script mean). Selection under
    the balanced protocol (item 10). Selection `finished` (job 2794210, 2026-07-19;
    `EXPERIMENTS_RESULTS.md` Exp 23b): **alpha=300 passes the guard with no negative
    stratum** (overall +0.0001, tail +0.0004); stronger alphas turn twins negative,
    consistent with twin conventions being signal. Effect at the measurability edge;
    full-test pass and balanced-test evaluation pending before any claim.

16. **Learned bias refit on balanced data** (added 2026-07-19,
    `analysis/balanced_sweeps.py` run_bias_refit): fit b on the language-balanced fit
    half (per-language alternating split of the seed-101 val), select reg on the other
    half under the guard. Fitting on balanced data removes the traffic-prior component
    of the objective, so the fitted b isolates attractor suppression; if no reg passes,
    the Exp 14 gain under the uniform-prior view is prior fitting, and if one passes,
    its most negative offsets should name the diagnosed magnets. Plain L2 only
    (centering on a frequency prior would contradict the uniform-prior objective);
    corrected NLL gradient. Selection `finished` (job 2794210, 2026-07-19;
    `EXPERIMENTS_RESULTS.md` Exp 23c): **reg=0.3 passes** (sel overall +0.0016, tail
    +0.0299, magnets +0.0252, twins -0.0016); attractor suppression survives the
    uniform-prior objective. Suppressed languages are head/twin sinks (nya, por, heb),
    not flat magnets, matching the 40%-FP-on-head-sinks diagnostic. Pending before
    adoption: refit-per-draw stability (draws 102-105), a balanced-test draw disjoint
    from val, a full-test pass, and an explicit decision on individual-language
    suppression (b = -8.9 on por trades Portuguese marginal recall for its
    satellites; the guard does not bound per-language harm). Floor-equalization
    re-selection ran in the same job: rejected at selection (item 14 closed,
    Exp 23a).

11. **Non-content token tying** (`analysis/token_tying.py`): tie the probabilities of
    tokens with no language identity (digits/whitespace 298 tokens, ASCII non-alpha 479,
    all-script non-alpha 1,291) to one shared resource-weighted value, so they cancel
    from every score difference. Pure tying, no renormalization. `finished` (job
    2790078, 2026-07-18; `EXPERIMENTS_RESULTS.md` Exp 18): NEGATIVE at every scope,
    val overall -0.0010 to -0.0078, nothing passes the guard. The Exp 18 design tied
    whitespace, a user-flagged error (spacing conventions are signal). Curated re-run
    also NEGATIVE (job 2793541, 2026-07-19): 212 digit + neutral-punctuation tokens
    with linguistic exclusions, tied within script groups and globally; val overall
    -0.0014 to -0.0016 with the cost concentrated in twins (dp_script twins -0.0103).
    Non-content usage rates are twin-discriminative signal; tying is closed as a
    direction at every curation level.

12. **Family back-off at unseen positions** (`analysis/family_backoff.py`): floor-plateau
    entries (exact per-language minimum, 74,617-99,810 per row) replaced by
    `lam_L * m_G(t)`, `lam_L = alpha/(N_L + alpha)`, m_G = confuser-excluded script
    backbone mean (script is the family proxy; a genealogical grouping is a possible
    refinement). Modes lift/full x alpha {300, 3000, 30000}; observed tokens
    bit-identical (the difference from refuted Exp 13 shrinkage); no renormalization.
    Modular: a new language needs only its own N and the frozen backbone mean.
    `finished` (jobs 2790155 script grouping + 2790174 WALS genealogical tiers,
    2026-07-18; `EXPERIMENTS_RESULTS.md` Exp 19): NEGATIVE under both groupings, val
    overall -0.0028 to -0.0304, monotone in alpha, grouping choice immaterial
    (within 0.0016). Adding unseen-token mass toward group typicality increases theft;
    joins Exp 9/13/18 in that refuted family. Implication recorded in Exp 19: the
    family-initialized retrain is not supported by this post-hoc surrogate. Untried
    remainder: DOWNWARD floor equalization (lower low-resource floors toward
    high-resource levels; Exp 6 only ever clamped upward).

13. **Macrolanguage-hierarchical decision** (`analysis/macro_hierarchy.py`): treat the
    variety within a macrolanguage as latent; score macrolanguages by log-sum-exp over
    members (top-50 candidates), argmax at the macro level first, then within the
    winner. Parameter-free; SIL mapping from `analysis/commonlid_eval.py` (83
    multi-member groups covering 289 languages: ara 11, msa 12, zho 8, que 27, zap 30).
    Targets the 40% of val false-positive mass on head-level dialect sinks and the ~20%
    arbitrary-split error ceiling. Also reports macro-aware accuracy. `finished`
    (job 2791444, 2026-07-18; `EXPERIMENTS_RESULTS.md` Exp 21): NULL, the group
    marginal never flips a decision; its product is the ceiling measurement
    (macro-aware accuracy 0.9680 vs exact 0.9603 on the test half, so 0.77pp of
    accuracy is within-macro confusion, an evaluation-convention question).

14. **Downward floor equalization** (`analysis/floor_equalization.py`, added 2026-07-18):
    clamp each language's exact floor plateau to `min(floor_L, F)` for one global
    constant F swept over {-17, -19, -21, -23}; nothing is ever raised (Exp 6 only
    tested the upward direction), observed tokens and specials bit-identical, no
    renormalization, fully modular (one shared constant). The direction implied by
    Exp 10 (resource-tied floors under-penalize unseen tokens for small languages) and
    by the four mass-adding negatives (Exp 9/13/18/19). `finished`, NOT ADOPTED
    (jobs 2791444 sweep + 2791722 full-test; `EXPERIMENTS_RESULTS.md` Exp 20). Full
    test: overall +0.0129 but tail -0.0204 [CI -0.0257, -0.0161] and magnets -0.0164
    [CI -0.0210, -0.0129]; a global-precision-for-tail-recall trade, dominated by the
    learned bias (equal overall, tail -0.0018). Third val-selected operating point
    overturned at full scale; item 10's split redesign is now a prerequisite for any
    further sweep selection.

Data note: scratch was purged; the original GlotLID `train.txt` (60,683,151 lines),
`glotlid_correct_test.txt`, train counts, all 5 prediction files, and the full 744 MB
model were recovered from the Google Drive folders in `SETUP.md` on 2026-06-26. The
custom Rust tokenizers build was rebuilt (`maturin develop --release`).

## Open paths (2026-07-19, updated 2026-07-23)

Ordered by dependency, not preference. The first block completes the two guard-passing
candidates; the second refines methods; the third is analysis; the fourth is
longer-range; the fifth (block E) holds the Exp 24 follow-ups, proposed 2026-07-23 and
awaiting approval. One decision gates several items and is listed first.

**Decision required (user):** the objective, the suppression bound, and the metric
view. (a) Whether the paper's headline objective is natural-traffic macro-F1 (learned
bias reg=5.0 is the reference result, Exp 16) or the uniform-prior view (baseline is
best adopted; items 15/16 are the live candidates). (b) Whether any per-language bias
may suppress an individual language, and by how much: the unconstrained balanced fit
puts -8.9 nats on por_Latn to benefit its satellites, and the guard bounds strata, not
languages. Item 9 (capped variants) implements the bounded version if a bound is
chosen. (c) RESOLVED 2026-07-23: the user chose the precision-primary adoption rule.
A candidate passes iff it passes the balanced-val guard with the tail within-stratum
tolerance widened to TAIL_RECALL_TOL=0.03 when tail global per-language F1 improves by
more than the recall loss, AND tail and magnet global mean F1 do not drop
(PREC_TOL=0.0), AND no single language loses more than 0.10 global F1. Implementation
is `passes_two_sided` (plan item E1); constants recorded in EXPERIMENTAL_SETUP.md.
Background: every stratum row and guard column is within-stratum (recall view), and
global per-language F1 reverses the tail ranking of the rejected configurations
(Exp 24: tail mean F1 baseline 0.5618, learned bias 0.6003, freq prior 0.6800,
floor-21 0.7655); the "not tail-safe" and "not adopted" verdicts of Exp 16/20 were
conditional on this choice.

**A. Complete the balanced-protocol pipeline for the passing candidates.**
1. Balanced-test draw: a language-balanced evaluation draw disjoint from the val draws
   (e.g. seed 201, K=100/language from pool minus draw-101 val), so balanced-objective
   FINAL numbers exist independently of selection data. Small extension of
   `analysis/balanced_split.py`. Not yet built; prerequisite for any balanced-objective
   claim about items 15/16.
2. Refit-per-draw stability for item 16 (protocol caveat 3): refit on draws 102-105 fit
   halves, compare selected reg, selection-half deltas, and the suppressed-language
   lists. Cheap (four more fits in the existing harness).
3. Full-test passes (natural-traffic view) for `punct_a300` and, conditional on the
   decision above, the (possibly capped) balanced bias; the `full_test_floor21.py`
   pattern generalizes (one scoring pass each against the saved baseline memmaps).
4. CommonLID out-of-domain checks for both candidates: distinguishes orthographic
   convention from register/domain artifact, the open question from the linguistic
   reading of the tying results.

**B. Method refinements, contingent on A.**
5. Punctuation prior refinements if alpha=300 confirms: finer alpha grid (30, 100, 300,
   1000), digits pooled separately from punctuation, genus-within-script grouping (the
   WALS tiers from item 12 are already built and reviewed).
6. Combination test: punctuation prior (likelihood-side) plus a capped balanced bias
   (decision-side); the mechanisms are orthogonal.
7. Item 4 (affine recalibration with per-token c_L) under the balanced protocol, if the
   bias family is revived by the decision above; the c_L term is likelihood-side and
   modularity-compatible in its frequency-anchored form.

**C. Analyses that sharpen the paper regardless of method outcomes.**
8. Residual error decomposition on the decided best configuration (item 6).
9. Training-data items 7.2 (mislabeling) and 7.3 (overlap): now directly relevant to the
   twin story, since part of the twin punctuation/domain signal (e.g. the JW.org marker
   in ind/zsm, Exp 4) may be corpus artifact rather than linguistic convention.
10. Apertus tail diagnosis (item 8) only if the Apertus branch is revived.

**D. Longer-range, training-time.**
11. Move the punctuation prior inside training (MAP counts on the dp columns during
    per-language estimation) and give the emergent floor an explicit, resource-indexed
    smoothing constant at training time; both are estimator changes that keep
    add-a-language modularity, motivated by the finding that every post-hoc
    distribution edit except mild dp pooling has failed.

**E. Exp 24 follow-ups (approved 2026-07-23 as part of the plan in
`~/.claude/plans/steady-finding-abelson.md`; user decisions: precision-primary
adoption rule, one-sided-min Good-Turing, `preliminary_mul` tokenizer, day-0 retrain
submission).** Statuses: E1 FINISHED (Exp 25: floor-21 provisionally adopted,
learned bias rejected on the collapse clause, freq prior eligible-not-selected);
E6 FINISHED (Exp 25: pnt/ell residual is model error, 50/50 standard Greek);
E2 FINISHED (Exp 26: diagnostic viable; `margin_q5` rejected on the szy_Latn
reassignment mechanism, pre-registered `margin_q5_head` ELIGIBLE but not selected,
floor-21 ranks higher on both instruments; margin family closed this round). New
open path from E2: compose the weight-side winner with the margin gate, tau
recalibrated under the composed weights; not pre-registered yet. E3 FINISHED
(Exp 27/28: counts landed, plateau overstated everywhere; gt_min REJECTED by the
veto despite the best selection-view numbers on record; the floor pathology is a
between-language externality, so per-language honesty and cross-language
equalization are separate corrections). New open paths from E3: (i) cross-language
equalization at the GT-implied shared level (replaces the swept floor-21 constant
with a counts-derived one); (ii) gt_min plus the head-targeted margin gate with
recalibrated tau. E4 (floor-21 + learned bias probe) is deprioritized below these
two: the bias is rejected for adoption and both new compositions target the same
strata more directly. E5 not started. Track A (Apertus 131k retrain + evaluation)
FINISHED: negative on both views (Exp 29, within-stratum tail -0.0437, FPs into
tail 2.3x). That measurement used the model containing a collapsed row (Exp 41);
the branch is NOT discontinued and a clean re-measurement on the retrained model
is open item 1 at the top of this file. Motivation and numbers in
`EXPERIMENTS_RESULTS.md` Exp 24-30, 42.
- E1. Two-sided guard columns: add global per-language F1 and mean precision for tail
  and magnets to every selection report. For finished configurations the numbers come
  from the saved memmaps at no scoring cost; `analysis/metric_decomposition.py` has
  the computation. Without this, selection is systematically directed against
  tail-precision configurations.
- E2. Margin diagnostic: one scoring pass over the 22,522 baseline false-positive
  lines plus the 7,735 true tail lines, recording the score gap between the predicted
  tail language and the runner-up. Decides whether a per-language decision margin
  tau_L, calibrated as a low quantile of the gap on L's own training lines (own-recall
  loss bounded by the quantile, no global data, add-a-language modular), can separate
  false positives from genuine tail lines. Login-node scale.
- E3. Good-Turing unseen-token mass: rescale each language's floor plateau so its
  total unseen mass equals the Good-Turing estimate n1/T from its own token counts
  (n1 = singleton types, T = total tokens; no global data, no tuned constant).
  Principled replacement for the floor-21 clamp; floor-21 (tail global F1 0.7655) is
  the baseline it must beat.
- E4. Composition pass: floor-21 plus learned bias reg=5.0 in one scoring run. The
  Exp 24 decomposition shows the two act on different categories (flat_magnets versus
  head/mid/twins); the existing memmap harness covers this.
- E5. tight_lowres check: anp_Deva (global F1 0.019), inh_Cyrl (0.150), gom_Deva
  (0.373), syl_Beng, arq_Arab sit at 2.5k-4.5k docs under a larger same-script
  neighbor and appear in no stratum row; verify any adopted mechanism on these five
  explicitly.
- E6. Label audit: manually inspect 50 of the 2,644 residual pnt_Grek <- ell_Grek
  lines (floor-21 view) to split model error from corpus label noise before investing
  in E2 for that pair.

## Plan: per-language combined method (drafted 2026-07-27, pre-registration pending)

**Amendments 2026-07-29 (user decisions, taken after two adversarial reviews of
the implementation plan; measured findings recorded in `EXPERIMENTS_RESULTS.md`
"Current state (2026-07-29)"). The committed text below stands as drafted, with
three changes:**
1. The assignment rule is derived from a seeded 40/60 split of the held-out
   remainder (RULE_SPLIT_SEED=301, RULE_SPLIT_FRACTION=0.40; rule derived on the
   40% part, judged on the 60% part, comparators recomputed there), not from
   balanced draw 101 as the "Protocol point" paragraph below states. Measured
   reason: draw 101 reverses the method ranking in all six groups and would
   select gt_min everywhere.
2. The decision criteria below are replaced by a paired bootstrap over the 1,940
   languages (BOOT_B=10,000, BOOT_SEED=0, percentile, 95%) of mixed minus
   gt_margin_adaptive on the judge part, with statuses per amendment 7 of the
   adoption rule (`EXPERIMENTAL_SETUP.md`): promotion requires the interval
   entirely above zero plus clauses (A), (B), (C) plus a user decision; an
   interval containing zero keeps the method in the pool for deeper exploration;
   hard rejection requires being worse or equal on every recorded instrument and
   group.
3. The treatment set stays the committed six combinations; the two combinations
   without solo references, (unmodified, gate on) and (floor-21, gate on), get
   solo-gate reference builds (gate post-process over `pred_baseline.npy` and
   `pred_floor21.npy`, about 15-30 minutes each) before the rule is fixed.

**Status (2026-07-30).** Steps 0-7 executed 2026-07-29/30 (Exp 44, 45, 46).
Outcome: the mixed matrix under rule v1 is indistinguishable from uniform
floor21_gate on the judge part (paired bootstrap +0.0002 [-0.0003, +0.0006]);
the per-language assignment direction is closed for this treatment space by a
null interaction, not by harm. floor21_gate is promoted under amendment 8,
with the draw-201 confirmation recorded in
`outputs/tables/floor21_gate_confirmation_201.md`. Nothing in the committed
step text below remains open.

**Idea.** Exp 38 showed the carried configurations are complementary rather than
redundant, and Exp 40 measured an oracle that picks the best per language at
0.9525 against 0.9334 for the best single configuration. Every carried method is
either a transformation of one language's row or a decision threshold for one
language's predictions, so a per-language assignment is implementable and keeps
add-a-language modularity: a new language's treatment depends only on its own
statistics.

**The caveat that shapes the whole design.** The oracle number was computed by
taking each language's F1 from a run in which all 1,940 languages used that
configuration. A mixed matrix is a different system: prediction is an argmax
across rows, so if language A receives a deepened floor while language B keeps a
shallow one, the comparison between A and B changes. The oracle is therefore
evidence that per-language heterogeneity exists, not an achievable target, and
the combined configuration must be SCORED, never inferred. Measuring the size of
that interaction is the experiment's guaranteed output regardless of whether the
method wins.

**Treatment set (modularity-preserving only).** Per language, one row treatment
from {unmodified, floor-21 clamp, Good-Turing rescale} and the margin gate either
off or on (adaptive quantile, top-resource reassignment target). The two
prior-style methods (frequency prior, learned bias) are excluded from the combined
candidate: they need global data to fit for a new language, which the standing
modularity constraint rules out, and the learned bias was rejected for adoption on
per-language harm. Consequence to measure first: the oracle restricted to the
modular subset is the honest ceiling for this design, and it is cheaper than the
full oracle to compute (step 1).

**Assignment features, training-side only.** Document count, plus quantities
derived from the weight matrix itself (row flatness, distance to the nearest
confuser, whether that confuser has much higher resources) and the Good-Turing
ratio n1/T. Provenance caution: `outputs/diagnostic/lang_diagnostic.csv` mixes
weight-derived columns with `magnet_ratio`, which was computed from validation-half
false-positive counts. Validation-derived features are acceptable (validation is
selection data) but test-derived features are not, so step 0 audits every column
before any is used.

**Protocol point that keeps the evaluation clean.** The assignment rule must be
derived from the BALANCED VALIDATION set (draw 101), not from the Exp 38 table,
which was computed on the held-out remainder that will judge the result. Deriving
the rule from the remainder and then evaluating on it would be selection on the
evaluation data. Step 1 therefore recomputes the per-group leader table on draw
101, and the rule is fixed and pre-registered from that table alone.

**Steps, with agent delegation.** Model choice follows the standing policy:
Sonnet for mechanical and search work, Opus only for correctness-critical
verification, Haiku for trivial single-file lookups.

- **Step 0, feature-provenance audit (agent, Sonnet).** For every column of
  `lang_diagnostic.csv` and `gt_counts.csv`, state whether it derives from the
  weight matrix alone, from training data, from validation data, or from test
  data. Read-only, mechanical. Output: a short table appended to the step-1
  artifact. This is exactly the kind of bounded lookup an agent should do.
- **Step 1, evidence base (agent, Sonnet).** One script,
  `analysis/combined_evidence.py`: (a) per-language F1 for the seven
  configurations on the balanced validation draw 101, and the per-group leader
  table computed there; (b) the oracle over the full carried set and over the
  modular subset only, on the same draw. Reuses `_per_lang_stats` and the saved
  prediction memmaps, so no scoring is required. Output:
  `outputs/tables/combined_evidence.md`.
- **Step 2, fix and pre-register the rule (assistant).** Write the assignment
  rule into `EXPERIMENTS_RESULTS.md` before any scoring, in the form "if
  <training-side condition> then <treatment>", with every threshold stated. The
  expected shape from current evidence, to be confirmed or revised by step 1:
  large languages unmodified and ungated, the smallest and the flat-confusion
  languages on the floor clamp, the middle band on Good-Turing plus the gate.
- **Step 3, implementation (agent, Sonnet, with a precise spec).**
  `analysis/mixed_matrix.py`: build the mixed weight matrix by applying each
  language's assigned row treatment, then score the full pool once and apply the
  per-language gate at inference. It is a structural clone of
  `analysis/full_test_gt.py` plus the gate loop from `analysis/gt_margin.py`, so
  the agent's task is mechanical assembly against a known pattern, with the
  standard gates: special-token columns bit-identical, row normalization checked,
  fingerprints recorded, resumable chunked memmap, bit-identity for untreated
  languages against the baseline predictions.
- **Step 4, adversarial pre-run review (agent, Opus).** The standing discipline
  for any code whose numbers enter the record. Focus: that each language receives
  exactly its assigned treatment, that no test-derived feature entered the rule,
  and that the interaction measurement in step 6 is computed correctly.
- **Step 5, scoring (assistant, SLURM).** One pass over the full pool, about two
  hours at 64 CPUs and 100 GB, mirroring `slurm_full_test_gt.sh`.
- **Step 6, evaluation and the interaction measurement (agent, Sonnet; assistant
  verifies).** Report the primary quantity against the carried set and the
  adoption rule on both tracks. Then the scientific payload: for each language,
  compare its F1 in the mixed system against its F1 in the single-method run of
  its assigned treatment. The difference is the cross-language interaction, and
  its distribution answers whether per-language treatment can be chosen
  independently of the competition it participates in.
- **Step 7, decision and documentation (assistant).** Adopt, iterate with a named
  mechanism, or reject; update the four documents and commit.

**Pre-registered decision criteria.** Success: the primary quantity exceeds
0.9334 and the amended collapse clause is satisfied (at most two supported
per-language collapses, which trigger investigation rather than rejection).
Partial success worth iterating: the aggregate falls between 0.9309 and 0.9334
but the small-language groups beat floor-21's 0.6337 and 0.5345. Informative
failure: the aggregate falls below 0.9309, in which case the interaction
measurement from step 6 quantifies why, and that result closes the naive
per-language-assignment direction.

**Compute budget.** One scoring pass plus cheap analyses over saved artifacts.

## Candidate directions from the post-promotion error analysis (2026-07-30, none started)

Source: a joint analysis of the errors that remain after the promoted
configuration, run on 2026-07-30. None of these is pre-registered; each needs
its own pre-registration (exact rule, constants, decision criteria) before any
scoring run. The quoted numbers trace to the committed per-language CSVs in
`outputs/diagnostic/` and to `outputs/diagnostic/gate_threshold_sweep_20260730.csv`;
the larger intermediate arrays were session artifacts and are regenerable from
the prediction memmaps on scratch.

Definitions used throughout this section. The promoted configuration
(floor21_gate) does two things. First, in every language's row (its vector of
100,000 log token probabilities), all entries at the row minimum, which are the
tokens never seen in that language's training data, are lowered to the value
-21. Second, a decision-time re-examination step: for every line whose
predicted language has fewer than 18,000 training lines, if the winning score
exceeds the second-place score by less than a threshold calibrated on that
language's own training lines, the prediction is moved to the highest-scoring
alternative among the top five candidates whose training corpus has at least
100,000 lines. Evaluation uses the seed-301 split of the held-out data: rules
and constants may be chosen on the derivation part (18,001,573 lines), and
final judgment uses only the judge part (27,002,441 lines). "Average F1" below
always means per-language F1 averaged unweighted over the 1,940 languages, the
project's primary quantity. After the promoted configuration, 962,633 wrong
predictions remain on the judge part.

**Direction 1: one shared re-examination threshold instead of 1,080
per-language thresholds, and a lower size requirement for replacement
candidates.** Replace the per-language thresholds with a single shared value
of 9 (scores are natural-log values, so the unit is nats), and allow
replacement candidates with at least 18,000 training lines instead of at
least 100,000. Measured on the derivation part: average F1 0.9530 against the
promoted configuration's 0.9478; languages under 1,000 training lines 0.7433
against 0.7330; the 118 languages with unusually flat token distributions
0.6763 against 0.6434; languages with at least 18,000 training lines 0.9553
against 0.9586. The optimum in the shared value is flat between 7 and 12.
Supporting pattern: this is the third case in the record where one shared
constant beats per-language estimates (the -21 level itself, which the pooled
Good-Turing statistics independently derive as -20.60; and the refutation of
per-script or per-size floor levels). It also removes the per-language
calibration requirement that currently excludes 26 languages with too few
usable training lines. Caution, flagged for a user decision: subtracting
nothing from large-language candidates while re-examining small-language wins
with one shared threshold is equivalent to lowering every small language's
effective score by a constant, which is an adjustment indexed on corpus size,
and the project has previously set aside score adjustments of that family.
Cost: a post-processing pass over saved predictions, no rescoring. Refuted
if, on the judge part, the paired bootstrap interval against the promoted
configuration contains zero, or the group of languages with at least 18,000
training lines loses more than 0.01, or the balanced-validation check fails
by more than the promoted configuration's recorded amounts.

**Outcome (2026-08-05/06, Exp 47).** Tried. The shared threshold
(SHARED_TAU=9.0, replacement bar lowered to 18,000 training lines) scored
average F1 0.9534 on the judge part, the highest aggregate on record, but
failed clause C at class level: 9 languages with judge-part support 15 to
2,947 lost more than 0.10 F1 against baseline. In the pool, not promotable in
current form. A hybrid follow-up (the smaller of 9.0 and a per-language cap
from own-train margins) is recorded but not pre-registered or run.

**Direction 2: choose the re-examined languages by distribution shape instead
of corpus size.** Four languages with corpora above 18,000 lines (Scots,
Banjar Latin-script, Aragonese, West Flemish) are each written almost
identically to a much larger language (English, Indonesian, Spanish, Dutch)
and have token distributions that are unusually flat for their script, per
the zH column of `outputs/diagnostic/lang_diagnostic.csv` (flatness scores 1.567 to 3.228, median 2.51,
against an overall median of 0.00). Because the
re-examination step only applies to languages under 18,000 training lines,
these four are exempt, and they receive 63,842 of the 118,006 remaining wrong
predictions into small-language or flat-distribution labels (54.1%). The
proposal is to add a language to the re-examined set when its flatness score
is high, regardless of corpus size. Upper bound if all such wrong predictions
were removed: +0.0012 average F1, +0.0204 on the flat-distribution group.
Cost: rescoring only the lines predicted as those languages. Refuted if the
four languages lose more from re-examination of their genuine lines than
their neighbors gain in precision.

**Outcome (2026-08-06, Exp 48).** Tried. Re-examining the four flat
large-corpus languages at their own 5th-percentile thresholds scored
average F1 0.9486 on the judge part, +0.0006 [+0.0001, +0.0013] over
floor21_gate, zero supported collapses. Eligible; a component of the
promoted configuration gate_flat4_prox21.

**Direction 3: a quality condition on the replacement label.** Of the 138,077
predictions the re-examination step moved on the judge part, 78,651 landed on
the correct language and 59,426 did not; 10,504 of the wrong ones landed on
Maltese alone, against 442 correct ones there. Simply moving fewer
predictions is measured to be worse (reverting all wrong moves' lines to
their pre-move labels scores 0.9397, below 0.9478, because a wrong prediction
on a large language costs less average F1 than the same wrong prediction on a
small one). The proposal is a per-candidate acceptance condition on the
replacement label, with the exact form fixed at pre-registration (it must be
computable from training-side or selection-side data only, never from
held-out data). Upper bound if every wrong move were prevented while keeping
every correct one: +0.0056 average F1. Cost: post-processing only.

**Outcome (2026-08-06, Exp 49).** Tried. The score-proximity condition
(D3_PROX=21.0 natural-log units) scored average F1 0.9498 on the judge
part, +0.0012 [+0.0007, +0.0016] over the direction-2 configuration
(gate_flat4_tau5), zero supported collapses. Eligible; a component of the
promoted configuration gate_flat4_prox21. Target-identity forms of this condition
were also measured and set aside in favor of the score-proximity form (see
`EXPERIMENTS_RESULTS.md`, "Patterns established by Experiments 44 to 50",
pattern (c)).

**Direction 4: give unseen tokens values according to their overall
frequency, instead of one identical value.** In the promoted configuration
every token unseen in a language's training data gets the same value, -21,
and such tokens are 92.4% of a typical row. The proposal is standard back-off
to a background distribution: the value becomes a shared constant plus the
token's log frequency in the pooled training data of all languages, which is
already stored in the model file as the base tokenizer distribution. A token
common across languages but absent from one language's data then scores
higher in that language than a token rare everywhere. Measured on 80,000
random derivation-part lines: line-level accuracy 96.138% against 96.097%,
with 52 lines flipping to correct against 20 flipping to wrong, and no
increase in wrong predictions onto small-language labels. The effect is
small and its sign on average F1 is not yet established; one full scoring
pass would establish it.

**Outcome (2026-08-06, Exp 50).** Tried. The pooled-frequency unseen-token
value (bgfloor) scored +0.000412 [+0.000043, +0.000837] in average F1 over
floor-21 solo on the judge part (paired bootstrap, B=10,000, seed 0), a
real but small gain at the edge of resolution. Eligible. The pre-registered
composed step (rebuilding the Exp 49 gate on this matrix) was declined by the user
(2026-08-06); bgfloor stays in the pool at this gate-less result, not
incorporated into gate_flat4_prox21.

**Direction 5: keep a small language's unseen-token value higher when -21
would make it lose on its own text.** For any ordered pair of languages, the
expected score difference per token when text genuinely from the first
language is scored under both rows is computable from the weight matrix
alone. Under the unmodified matrix this difference is positive for every one
of the 3,761,660 pairs, meaning every language is expected to win on its own
text. After lowering all unseen-token values to -21, it is negative for 762
pairs involving 12 languages, each with 85 to 410 training lines. The two of
those twelve with at least 10 judge-part lines lose 0.0333 mean recall
against the unmodified baseline, against 0.0065 for all other languages. The
proposal: for exactly those languages, lower the unseen-token value only to
the deepest level at which the expected difference stays non-negative. Cost:
one matrix computation plus one scoring pass; 12 rows change.

**Outcome.** Not tried. Still open.

**Measured and set aside.** Honest per-language unseen-token mass (the
Good-Turing rescale) recovers most of the small-language recall that -21
sacrifices but adds 40% more wrong predictions onto small labels and loses
on line accuracy (187 against 8 paired flips), so it runs against the
precision-limited objective. Per-script levels for the unseen-token value
cannot address the 99.5% of remaining errors that occur between languages of
the same script. Corpus-size-indexed levels can recover at most 4,079 of the
962,633 remaining wrong predictions. Scaling the re-examination threshold
with line length scores 0.9512 or lower against 0.9530 for a constant.
Doubling every per-language threshold scores 0.9325.

**The boundary this family cannot cross.** 98.7% of the remaining wrong
predictions are lines whose true language has at least 18,000 training
lines, and 88.2% of those are confused with another such language,
concentrated in close pairs (Indonesian and Malay, English and Scots,
Mandarin and Wu Chinese). Both rows in such a pair cover the text well, so
no treatment of unseen tokens changes their comparison materially (the
measured change is at most 0.14 nats per token). Progress past roughly
0.955 average F1 requires a mechanism that separates specific pairs. The
open question, unresolved and not yet designed: whether such a mechanism can
keep the add-a-language property, since adding a new language would only
require computing pair corrections between the new language and the
existing, unchanged rows, from the new language's own data.

## Open items after Exp 38-42 (updated 2026-07-27)

Ordered by readiness. Items 1 and 2 are specified well enough to launch without
further design; items 3 and 4 need a decision first.

**1. RESOLVED 2026-07-27 (job 2911700, Exp 43): the clean re-measurement ran and the branch verdict holds.** The retrained model is negative against the 100k model on both metric views (within-stratum overall -0.0090, tail -0.0395; global tail 0.4269 against 0.5618), so the regression is a real property of the larger vocabulary. The bug had overstated it: false positives into tail labels are 32,211 clean against 51,926 corrupted, within 0.2% of the Exp 30 counterfactual. The 200k retrain could receive the same treatment but is lower priority. Original item text follows.

**1 (original).** Clean re-measurement of the Apertus 131k branch. The
Exp 29 verdict (negative on both metric views) was measured on a model containing
a collapsed Azerbaijani row that Exp 30 showed carried about two thirds of the
false-positive increase. The retrained model `glotlid_apertus131k_fp64.unilid`
exists and passed the degeneracy gate (Exp 42). What is needed: a full-test
baseline evaluation of the retrained model against the 100k production model.
`analysis/full_test_eval_131k.py` does exactly this but hardcodes the model path
and its scratch directory at module level, so it needs those two values
parameterized (and a separate scratch directory, since the fingerprint gate will
correctly refuse to reuse the old one). Roughly 3.5 hours on one node. Until it
runs, the Exp 29 numbers stand as a measurement of the corrupted model and their
magnitude is known to be overstated. The 200k retrain is available for the same
treatment but is lower priority, since the 200k branch was refuted on a different
axis (Exp 15) and its Azerbaijani row was only partially damaged.

**2. RESOLVED 2026-07-30 (Exp 44-46, amendment 8), then superseded 2026-08-06
(Exp 47-50).** The combined-method plan (the per-language mixed weight
matrix) ran to completion: the mixed configuration scored indistinguishably
from the uniform floor21_gate configuration on the judge part (paired
bootstrap of per-language F1 averaged over the 1,940 languages, mixed minus
floor21_gate, +0.0002 [-0.0003, +0.0006]), so the per-language-assignment
direction closed for the combined method's six-combination treatment space
by a null interaction, not by harm. floor21_gate, the simpler uniform
configuration, was promoted instead (2026-07-30). floor21_gate was itself
superseded 2026-08-06 by gate_flat4_prox21 after Exp 47-50 (see "Candidate directions from the
post-promotion error analysis" above and `EXPERIMENTS_RESULTS.md`, "Current
state (2026-08-06)"). Original item text follows.

**2 (original). Per-language method chooser (the strongest open method direction).** Exp 40
measured an oracle that picks the best of the seven configurations per language at
0.9525 against 0.9334 for the best single configuration, with the headroom
concentrated in languages under 1,000 documents (+0.0724) and flat-confusion
languages (+0.0998). A real chooser must select using training-side information
only (document count, Good-Turing statistics, margin distributions, flatness and
nearest-confuser distance from the per-language diagnostic), never using test
labels, and its gap to the oracle measures its quality. Modularity is preserved
if the choice rule is a function of a language's own statistics. Not
pre-registered; the pre-registration should fix the feature set and the choice
rule before any measurement.

**3. Objective interpretation (needs a user decision).** The primary quantity
averages per-language F1 over natural-distribution test data. The alternative
reading of the same instruction averages over equal-volume test data (100 lines
per language). Exp 38's reasoning for the current choice is recorded; a
correction is a one-line change to the record and a re-ranking of the carried set.

**4. Carried-set narrowing (deliberately deferred).** Six configurations remain
live because the selection data cannot separate them (Exp 38 shows they are
complementary rather than redundant). No narrowing is needed until the paper's
final table, and the oracle result argues for combination rather than selection.

**Completed since the last plan revision:** the fp64 trainer fix adopted and both
Apertus models retrained (Exp 41, 42); the carried-set comparison under the
primary quantity (Exp 38); the CommonLID out-of-domain check including the
objective-consistent per-tag metric (Exp 39); the oracle bound (Exp 40).

## Special-token correction and re-release (2026-08-17)

Full execution detail is in `RERELEASE_PLAN.md`; this section carries the status
of record. Context: per-language training gave four never-read special tokens 0.8
of every row's mass, depressing every real token by log 5 = 1.6094 nats. Fixed in
package version 0.3.0 and applied to all four stored models as a closed-form
transformation.

**Author decisions taken 2026-08-18:**
5. **c = -17.3906 for the corrected model**, which is what the published Exp 20
   procedure selected when re-run (job 3107082), rather than -21 + log 5 =
   -19.3906, the pre-registered expectation. Reason: overriding a pre-registered
   procedure with a prior expectation after seeing its result is the pattern the
   project's own rules exist to prevent. The two grid points are tied (released
   picks -21 over -19 by 0.0001, corrected picks -17.3906 over -19.3906 by
   0.0002), and that tie must be disclosed in the paper rather than presented as
   a clean selection.
6. **The paper reports the full-pool stratum regressions alongside the overall
   gain** (overall macro F1 +0.0035, tail -0.0087, magnets -0.0071), stating the
   mechanism: the stratum rows are the within-stratum recall view, so a falling
   tail figure means tail-language examples are misclassified more often.

**Author decisions taken 2026-08-17, not to be silently revisited:**
1. Regenerate what this machine can. Table 1 carries corrected UniLID, calibrated
   and Mistral-Nemo rows next to DeepSeek3.2 and Qwen3 rows computed on
   pre-correction weights, with the mixture stated in the caption. Same for the
   co-author's WiLI and DSL-ML tables.
2. Ship the corrected artifacts once their own gates pass, without waiting for the
   paper. The model card states which paper version its numbers match.
3. Special tokens must not contribute to a score under any training method.
4. Both a calibrated and an uncalibrated model go on HuggingFace; the polybox
   copy of the original uncalibrated model is retired.

- **Defect found, root cause proven, package fixed (0.3.0)** — `finished`.
  Results: the 2026-08-17 entries at the top of `EXPERIMENTS_RESULTS.md`.
- **Four stored models corrected and gated 8/8** — `finished`.
  `analysis/correct_special_token_mass.py`, `analysis/gate_correction.py`.
- **Effect measured on the golden subset** — `finished`. Macro F1 0.9454 to
  0.9460, a wash; the re-release is justified by correctness, not metrics.
- **0.3.0 clamp regression found and fixed** — `finished`. Package commit 2d5f62d.
- **Calibration probes (c, tau)** — `finished`. c is consistent with carrying by
  addition; the thresholds are not carryable and all 1,084 must be re-estimated.
- **B0: separate corpus size from language identity in the unseen-token plateau**
  — `finished` 2026-08-17, PASS 3/3. Three languages at the 100,000-line cap
  (`abk_Cyrl`, `mam_Latn`, `zul_Latn`), five nested subsample sizes each.
  Within-language slope -2.196 / -2.196 / -2.184 nats per decade of tokens at
  R-squared 0.999, against -2.039 across 1,940 languages; the two fits agree to
  0.006 nats near the median T. The plateau probability scales as `T^-0.95`, about
  one count in T, which is estimator behaviour rather than a floor.
  `real_missing` is 0 in all 15 runs, ruling out the base-tokenizer fallback.
  Results: `EXPERIMENTS_RESULTS.md` "B0: corpus size alone sets the plateau".
  Artifacts: `analysis/plateau_vs_corpus_size.py`,
  `analysis/plateau_reference_fit.py`, `outputs/rerelease/`.
- **B1: make the analysis chain safe to point at a second model** — `finished`
  2026-08-17. `analysis/model_context.py` is the single resolution point and
  refuses a non-default model paired with the default output root or any
  store-backed root. Wired and each verified by triggering the guard:
  `full_test_eval.py`, `length_bias.py`, `floor_equalization.py`,
  `full_test_floor21.py`, `solo_gates.py`, `gate_variants.py`,
  `mistralnemo_eval.py`, `release_gates.py`, `build_release_calibration.py`,
  `commonlid_calibrated.py`, `commonlid_carried.py`.
  `analysis/model_context_selfcheck.py` fires all six resolver branches and all
  seven entry points: 13/13. The two missing generators are written:
  `analysis/viterbi_vs_marginal.py` and `analysis/lenbias_norm_table.py`.
- **B2a: the proximity bound (21 nats) is untracked and may need re-deriving** —
  `not started`, found 2026-08-19. It is a score difference, the same kind of
  quantity as the per-language thresholds, which the probe measured moving by up
  to 123%. The published grid search found macro F1 flat to within 0.0003 for
  bounds from roughly 15 to 35, so 21 may still sit inside that plateau on the
  corrected model, but that is a hypothesis and not a measurement. Re-running the
  grid search is one pass over the 18.0M-line development part.
- **B2: re-derive the calibration** — `ongoing`. c: **finished**, job 3107082,
  selected **-17.3906**; see the 2026-08-18 results entry for the tie. Still to
  do: all 1,084 group-A thresholds (`solo_gates.py floor21`, needs the floor-c
  full-pool pass first); the high-entropy group re-identified under the published
  criterion (`build_release_calibration.py` asserts the current four and aborts
  until updated, overridable only by an explicit flag). Group A membership cannot
  change, being defined by N_L.
- **B3: regenerate the paper numbers** — `ongoing`. Step 1 (corrected full-pool
  baseline) **finished**, job 3107045: overall macro F1 0.9292 to 0.9327. Step 2
  (floor-c full-pool pass at c = -17.3906) is job 3110918, submitted 2026-08-18.
  Remaining steps and their dependency order in `RERELEASE_PLAN.md`.
- **B4: ship the corrected artifacts, retire polybox** — `waiting on dependency`
  (B2 for the calibrated model; the uncalibrated model is ready now).
- **DeepSeek3.2 and Qwen3 rows (24 cells of `lid_main.tex`)** — both models were
  located in the co-author's Drive folder on 2026-08-18 and downloaded, so this is
  no longer blocked on artifacts. Both carry the special-token defect.
  - **DeepSeek3.2** — `not started`. Correct, gate, re-evaluate. No sign of the
    fp64 EM corruption (3 plateau outliers, all shared minority-script coverage
    effects).
  - **Qwen3** — `ongoing`. **Its azj_Latn row is corrupted independently of the
    special-token defect** (plateau at the training floor, 20.1 sd below the
    corpus-size expectation, flagged in no other model, and the documented fp64
    EM bug trigger language). **Author decision 2026-08-18: retrain the variant
    with the patched trainer** rather than report it with the defect stated or
    drop the row. A 0.3.0 retrain fixes both defects at once, since the trainer
    no longer gives the special tokens the base tokenizer's score-0 entries.
    Submission script `slurm_qwen3_train_fp64.sh`.
  - **`mya_Mymr` in the Qwen3 model** (-8.5 sd, also unique to it, also
    N_L = 100,000) is unresolved: Burmese script coverage is the competing
    explanation and has not been separated from a second EM casualty. The retrain
    settles it either way, since a coverage effect survives a retrain and an EM
    casualty does not.

## Notes for whoever resumes this (updated 2026-07-27; pointer corrected 2026-08-06)

Read `EXPERIMENTS_RESULTS.md` "Current state (2026-08-06)" first; it is the
single authoritative summary. These are the traps and conventions that are easy
to get wrong.

- **Three evaluation datasets, never interchanged.** Selection uses the balanced
  validation set (draw 101, 188,061 lines, up to 100 per language). Confirmation
  of a ranked candidate uses the held-out remainder (45,004,014 lines outside
  draws 101 and 201), where per-language F1 counts every false positive. Final
  reporting uses the balanced test set (draw 201, 185,204 lines). Selection never
  touches draw 201, and exactly one candidate per track is confirmed there per
  round, to keep that dataset's confirmation value. Amendment 2026-07-29: a
  candidate whose parameters or rule are fit on remainder data (currently only
  the mixed configuration) is confirmed on the judge part of the seed-301 split
  (27,002,441 lines), and every comparison involving such a candidate is
  computed on the judge part with comparators recomputed there; the full
  remainder stays the confirmation instrument of record for ordinary candidates.
- **Two ways to compute a stratum number, and they can rank methods oppositely.**
  A within-stratum figure restricts to examples whose true label is in the
  stratum and therefore excludes false positives arriving from outside it; a
  global per-language figure counts the whole confusion. The tail deficit is a
  precision problem (0.459) invisible to the within-stratum view (0.913). Always
  state which is used. Details in `EXPERIMENTAL_SETUP.md` "Stratified-metric
  views" and `outputs/tables/metric_decomposition.md`.
- **The adoption rule has nine amendments** (`EXPERIMENTAL_SETUP.md`
  "Precision-primary adoption rule"), all user decisions. The ones most likely to
  surprise: the collapse clause tolerates up to two per-language collapses as
  flagged investigations rather than rejecting, and near-tied candidates are all
  carried forward rather than narrowed to one. `passes_guard` is retained only to
  reproduce historical reports; current selection uses `passes_shortlist`,
  `passes_two_sided`, and `passes_uniform`.
- **The objective is decided** (2026-07-25): macro-averaged per-language F1 with
  every language weighted equally. One interpretation choice (natural-distribution
  versus equal-volume test data) is open and flagged in the plan's open items.
- **Trainer integrity.** The installed `spm_train` is the patched double-precision
  build; the pre-fix binary is kept as `~/.local/bin/spm_train.pre_fp64` for
  reproducing historical training only. Run `analysis/degeneracy_scan.py` on every
  newly trained model before evaluating it, and read `EXPERIMENTAL_SETUP.md`
  "Per-language training pipeline and the trainer fix" to tell the two causes of
  degenerate rows apart.
- **Refuted families, do not revisit without a new mechanism:** moving mass toward
  group typicality (Exp 9, 13, 18, 19 including the curated tying re-run), length
  normalization (Exp 2, 5), floor clamps as a naive family (Exp 6, 20, 23a; the
  calibrated descendants survive), heuristic variance reweighting (Exp 8a),
  entropy sharpening.
- **Anchors worth knowing:** the six carried configurations and their per-group
  strengths (Exp 38); the oracle bound at 0.9525 (Exp 40); the reassignment law
  from the margin family (Exp 31, 33, 34); the Good-Turing finding that the floor
  overstates unseen mass for all 1,940 languages (Exp 27); the macrolanguage
  ceiling measurement (Exp 21); the special-token structure (four tokens at
  exactly 0.2 per row). **Correction 2026-08-17:** this read "argmax-neutral",
  which is measured false. It is a training defect, corrected in version 0.3.0 and
  in all four stored models; see the re-release section below.
- **Everything is committed and pushed** to `origin/main`; artifacts of record are
  in `outputs/tables/` and `outputs/diagnostic/`, prediction memmaps and models on
  scratch under `/capstor/scratch/cscs/cmeister747/unilid_analysis/`. Scratch is
  purged after roughly two weeks without access, so re-touch the memmaps if a gap
  in work is expected.
