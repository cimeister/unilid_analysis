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

**Prior-centered learned bias (2026-07-18).** `learned_prior.py` penalty generalized to
`reg*||b - gamma*log(N+1)||^2`, grid `PRIOR_GAMMAS = {0, 0.25, 0.5}` x `REGS` (18 fits,
gamma=0 is the plain L2). Selection: same all-strata guard, max val overall. Outputs are
tagged `_centered` so the Exp 14 artifacts remain the plain-L2 record. The same change
fixes an NLL gradient bug present since Exp 14 (soft counts were accumulated over all
examples' top-k candidates while the loss conditions on top-k containing the true label;
finite-difference verified). Rationale for the grid cap at gamma=0.5: Exp 16 measured the
pure frequency prior (the anchor) at tail -0.0182 on the full test set, so large gamma is
not presumed safe.

**Non-content token tying (2026-07-18).** `token_tying.py`: tied sets classified on
byte-decoded token text (never tied if any character is alphabetic or the token does not
decode): digits_ws (298 tokens), nonalpha_ascii (479), nonalpha_all (1,291). Tied value:
log of the resource-weighted mean probability (`w = min(N, 100k)`, all languages). PURE
tying, no renormalization: the scorer sums unnormalized log-weights, and renormalizing
would apply `-log Z_L` to every untied token (up to 0.36 nats/token spread, concentrated
on flat confusers), conflating tying with a per-token bias. Special tokens
(`<s> </s> <pad> <unk>`, each exactly p=0.2 in every row: HF score-0 specials normalized
into the rows, 0.8 of all mass, uniform across languages so argmax-neutral) are asserted
and never tied. Selection: all-strata guard over the three sets, val half; test half
once.

## Balanced validation protocol (2026-07-19, plan item 10)

All future selection uses the language-balanced validation protocol
(`analysis/balanced_split.py`; constants documented in its docstring). Pool = the
45,377,279 kept full-test lines (the original 250k val is RETIRED: it was tuning data
for Exp 13-20 and is never reused). Five draws (seeds 101-105), each sampling
`min(K=100, floor(0.5 * n_L))` lines per language without replacement; draw 101 is the
working val (188,061 lines, all 1,940 languages, tail median support 33, 89/96 tail
languages with >= 10 examples); the other draws bound split-selection variance. Final
numbers come from pool-minus-val. Artifacts:
`outputs/diagnostic/balanced_val/{manifest.json, val_lines_seed*.npy}`, seed-101 text
cache on scratch.

**Objective note (measured, Exp 22):** a language-balanced val is the uniform-prior
deployment view. Prior-style methods whose gain is deployment-prior fitting (the
frequency prior, the learned bias) fail selection under it BY CONSTRUCTION, and did in
the re-baseline; natural-traffic macro-F1 remains reported on the test side. Which view
is the paper's headline objective is a framing decision to make explicit; selection
uses the balanced view.

**Protocol caveats (from the 2026-07-19 adversarial review, which verified all draws
and the re-baseline bit-exactly):**
1. Every language has >= 7 val examples (minimum pool count is 15, `otw_Latn`), and the
   smallest surviving test support is 8 items (`otw_Latn`, `kdr_Latn`, `xum_Latn`);
   per-language claims for the few smallest languages rest on that many items.
2. The multi-draw variance check has reduced power for the smallest tail languages:
   the 0.5 fraction cap makes any two draws share ~50% of a tail language's lines
   (~6.5% overlap overall), so the check understates resampling variance there.
3. Draws 102-105 are a clean stability check only for operating points that were NOT
   fit on draw-101 texts. For fitted methods (e.g. any bias refit), stability must be
   assessed by refitting per draw and comparing the selections, not by treating other
   draws as held-out data.
4. The seed-101 text cache stores raw parsed text; consumers must apply
   `model.preprocess` themselves (empty-after-preprocess lines score as wrong,
   matching the memmap convention). The cache is deterministically regenerable from
   `val_lines_seed101.npy` and the test file.
5. RESOLVED 2026-07-23: the language-balanced TEST draw exists (seed 201; see the
   Precision-primary adoption rule section for its construction and the regeneration
   of the stability draws it required). Balanced-objective headline numbers now come
   from it. Note for caveat 3: after the regeneration, draws 102-105 exclude the test
   draw, and any candidate fit on them is vetoed on data excluding them as well.

## Stratified-metric views (2026-07-23, Exp 24)

Two stratified views exist in this record and answer different questions:

- **Within-stratum macro-F1**: truth and predictions restricted to examples whose true
  label is in the stratum (`compute_metrics(yk[m], preds[m])`, m = true-label-in-stratum).
  Every stratum row in the full-test tables and every guard column uses this view.
  Cross-stratum false positives are excluded by construction, so for the small strata it
  approximates a recall view (baseline tail: 0.9132 within-stratum versus a 0.9154
  perfect-precision counterfactual).
- **Global per-language F1**: the full confusion row and column per language. The
  overall rows already use it (macro-F1 over all 1,940 languages). For the tail it
  includes the false positives assigned to tail labels under natural traffic (baseline:
  22,522 against 7,735 true tail examples; tail mean F1 0.5618).

Consequences for selection: the guard bounds only the within-stratum view, and the
balanced val (K=100 per language) additionally removes the volume asymmetry between
head and tail that produces the false positives, so neither selection instrument can
register the tail-precision failure mode or its repair. Configurations whose mechanism
is tail-precision repair (floor-21, the frequency prior) fail the guard's tail column
while raising tail global F1 (Exp 24, `outputs/tables/metric_decomposition.md`,
`analysis/metric_decomposition.py`). Any tail or magnet claim must state which view it
uses.

## Precision-primary adoption rule (2026-07-23, user decision)

Three instruments, each held out from the next's selection:
1. **Selection**: balanced-val draw 101, existing `passes_guard` (within-stratum), with
   one widening below.
2. **Precision veto**: for shortlisted candidates, global per-language F1/precision
   computed from the candidate's full-pool prediction memmap on pool minus the
   selection draw (101) and the headline draw (201). Amended 2026-07-23 at first run:
   the original union-of-all-draws exclusion left median ~1 true line per tail
   language (six independent half-draws exhaust a 66-line pool; measured veto tail
   recall 0.2188), breaking per-language F1 exactly where the veto needs it. The
   stability draws 102-105 stay inside the veto; a candidate FIT on any stability
   draw (refit-per-draw checks, caveat 3) must additionally exclude the draws it was
   fit on. A runtime gate aborts if the veto retains median < 10 true lines per tail
   language. Veto LEVELS are not comparable to full-pool numbers (about half of each
   tail language's true lines are excluded while all false positives remain); the
   rule uses gains and drops only. A sampled natural-prevalence val was rejected on
   power grounds (a 2M uniform sample carries ~1,333 tail-predicted lines across 96
   languages, so per-language precision is undefined for most of the tail). The veto
   is a pre-registered pass/fail check, not an argmax over test evaluations.
3. **Headline**: adopted configurations report on the balanced test draw (seed 201,
   185,204 lines, all 1,940 languages, tail median support 16). Construction
   (2026-07-23): drawn disjoint from the working val draw 101 ONLY, with
   k = min(K, floor(0.5 * remaining)) so a natural remainder survives; excluding the
   union of all five val draws is infeasible (the same exhaustion arithmetic as
   above). To keep the test draw disjoint from all potential tuning data, the unused
   stability draws 102-105 were regenerated to exclude it (k rule matched to draw
   101's for exchangeability; zero languages at reduced k; manifest annotated). This
   closes protocol caveat 5.

A candidate is **adopted** iff (A) on balanced-val draw 101: overall may drop at most
GUARD_TOL=0.01, twins/head at most GUARD_TOL, and tail/magnets at most
TAIL_RECALL_TOL=0.03 when that stratum's global mean F1 gain on the veto exceeds its
within-stratum loss, else GUARD_TOL (symmetric widening, user decision 2026-07-23);
and (B) overall global macro-F1 improves and tail/magnet global mean F1 do not drop
(PREC_TOL=0.0) on the veto instrument; and (C) no single language with at least
MIN_COLLAPSE_SUPPORT=10 true veto lines loses more than 0.10 global F1 versus
baseline. The support floor in (C) was added after the 2026-07-23 delta review: at
n=4 a single line flip moves F1 by 0.11-0.14, so an unsupported max-over-languages
test false-trips on quantization noise; sub-floor languages exceeding the bound are
reported informationally, never silently dropped. The fit-draw conditional is
enforced in code: `two_sided_report.CONFIG_FIT_DRAWS` must declare each candidate's
fit draws and the run refuses candidates whose fit draws are not excluded from the
veto.

Constants (pre-registered 2026-07-23, not swept): TAIL_RECALL_TOL=0.03, PREC_TOL=0.0,
per-language collapse bound 0.10, BALANCED_TEST_SEED=201. Rationale for the widening:
Exp 24 measured the recall/precision trade of the leading configs at <= 3.3pp recall
for +16 to +30pp precision; a strict two-sided AND at 0.01 would reject every
tail-precision config (floor-21 drops within-stratum tail 0.0204). No CI on the veto:
it is a population statistic over the full held-out remainder, and the CI-only-rule
rejection in the guard provenance note applies.

Implementation: `passes_two_sided` in `analysis/hierarchical_pool.py` (single
implementation, imported by all reports); reporting in
`analysis/two_sided_report.py` -> `outputs/tables/two_sided_selection.md`.

**Amendments (2026-07-24, gating reconsideration; user invited the review and
directed continuation under it, final confirmation pending).** Audit of the first
round's verdicts: clause (C) was productive twice (the margin_q5 rejection surfaced
the small-relative reassignment mechanism, and its pre-registered fix passed;
learned-bias suppression confirmed on the full pool) but the binary verdict
conflated method quality with the objective choice for gt_min (best uniform-prior
numbers on record, rejected only on natural-traffic clauses). Three amendments:
1. Stage (B) overall softened from "must improve" to "must not drop more than
   GUARD_TOL": a tail-repair candidate must not fail on a sub-noise overall dip.
   Flips no first-round verdict (gt_min's veto overall drop was 0.0007 but its
   tail/magnet precision drops and clause (C) stand).
2. Dual-track verdicts. The natural-traffic track is the rule above. The
   uniform-prior track (`passes_uniform`): balanced-val overall must improve, no
   stratum drops more than GUARD_TOL; the single track-selected candidate (highest
   balanced-val overall) is confirmed on the balanced test draw with the
   per-language collapse bound (0.10 at support >= 10 on the draw). Within-stratum
   macro-F1 on balanced data is that track's deployment view, so it needs no
   separate precision veto. Each track names its own champion; the paper's headline
   choice between them remains the user's decision, but the machinery no longer
   blocks on it.
3. Verdict vocabulary gains an ITERATE lane: a candidate that wins one track and
   fails the other with an identified, addressable mechanism is recorded as
   "iterate" with the mechanism, not merely "rejected" (first assignment: gt_min,
   natural-traffic failure mode = FP inflow, addressable by composition with an
   FP-side repair).
The seed-201 discipline is unchanged: selection never touches the test draw;
exactly one candidate per track is confirmed there per round.
4. Outlier tolerance in the collapse clause (user decision 2026-07-24): the bound
   exists to catch methods that harm a (sub)class of languages as a whole, so up to
   MAX_LANG_COLLAPSE_OUTLIERS=2 supported collapses flag a REQUIRED dig-in on those
   specific languages without blocking eligibility; three or more reject as a
   class-level pattern. Applied identically on both tracks. Effect on the record:
   learned_bias returns to eligible (flagged: llb_Latn), gt_min becomes the
   uniform-track champion (flagged: mev_Latn, sbs_Latn on the test draw); gt_margin
   remains rejected (4 supported collapses forming the lowmid-under-dominant
   pattern).

## Reproducibility limitations of this record

- Only one git commit exists (`b7508fd`, 2026-04-08); per-experiment code versions are not
  separately tracked. Source-file mtimes are the only finer-grained timing signal.
- SLURM submissions did not log seeds/commit/launch-command into the output beyond the
  submission scripts themselves; the scripts in `slurm_*.sh` are the reproducibility record
  for each job (kept in the repo).
- The recovered prompt history (`EXPERIMENTS_CHRONOLOGICAL.md` cites specific prompt numbers)
  is the only surviving record of design rationale that was not written into code or
  `EXPERIMENTS.md`.
