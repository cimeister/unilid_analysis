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
[Correction 2026-08-09: this sentence overstates. In analysis/diagnostic.py only
KCLOSE_QUANTILE and DNN_LO_Q are quantiles; ZH_MAGNET=1.5, ZH_EXTREME=5.0, and
MAGNET_RATIO_MIN=2.0 are fixed values with no recorded search. The camera-ready
provenance paragraph states them as fixed during development (code over prose,
consistency review finding 12).]

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
5. Near-tie co-selection (user decision 2026-07-25): eligible configurations whose
   balanced-val overall macro-F1 lies within NEAR_TIE_BAND=0.001 of the top one are
   ALL carried forward as live candidates; selection does not narrow to a single
   method when the selection data cannot separate them. Balanced-test reporting
   stays restricted to the top-ranked candidate per track (confirmation
   discipline). First application: six configurations carried
   (freq_prior, learned_bias, floor21, margin_q5, margin_q5_head,
   gt_margin_adaptive).
6. Objective (user decision 2026-07-25): the project's primary quantity is
   macro-averaged per-language score: every language weighted equally in the
   average, within reason (an extreme low-resource set may be exempted as
   unworkable and reported separately). Interpretation adopted pending
   correction: primary evaluation = per-language F1 computed over the full
   natural-distribution test data with all false positives counted, averaged
   unweighted over languages; the balanced draws remain selection and
   confirmation instruments.
7. Status vocabulary and pool semantics (user decision 2026-07-29). The words
   "pass" and "adopted" had accumulated several meanings in this record
   (eligible, flagged eligible, selected, carried, track champion, provisionally
   adopted, rejected); this amendment fixes three statuses with defined
   consequences. Evaluation is unconditional: every candidate is measured on all
   instruments of both tracks (selection view, veto, per-group global F1,
   collapse diagnostics), and clauses (A), (B), (C) are computed and reported
   for every candidate regardless of status.
   - IN THE POOL (default): a candidate that improves at least one recorded
     instrument or group by more than NEAR_TIE_BAND=0.001 and is not
     hard-rejected stays a live candidate and is explored. Clause failures do
     not remove a candidate from the pool; they are recorded as named
     weaknesses, and the clause-C dig-in requirement is unchanged.
   - PROMOTED (at most one per track): the configuration the paper reports for
     that track. Promotion requires clauses (A), (B), (C) on that track plus an
     explicit user decision; the clauses gate promotion only, not pool
     membership. This keeps the protection against promoting a candidate over
     the baseline on one number while it harms others.
   - HARD-REJECTED: only when definitively worse: worse or equal on every
     recorded instrument and group with at least one strict loss, or the
     apparent improvement is traced to a bug or measurement artifact. A hard
     reject closes that specific configuration, not its family or direction.
   Prior verdicts re-read under this vocabulary, numbers unchanged: gt_min,
   learned_bias (reg 5.0), and gt_margin move from "rejected for adoption" to
   "in the pool, not promotable in current form", each keeping its recorded
   mechanism; the ITERATE lane of amendment 3 is subsumed by pool membership
   with a named mechanism; floor-21's provisional adoption and
   gt_margin_adaptive's configuration-to-beat status are unchanged.
8. Conditional widening of the clause-(A) cap (user decision 2026-07-30, taken
   with the Exp 45/46 evidence in hand). For the tail and magnet strata only,
   the clause-(A) cap extends from TAIL_RECALL_TOL=0.03 to WIDE_CAP=0.05 when
   that stratum's veto-instrument global mean F1 gain exceeds
   GAIN_RATIO_MIN=5 times its within-stratum loss on the selection view.
   WIDE_CAP is a hard outer bound: no gain ratio extends the cap further.
   Both constants are pre-registered here, not swept; the motivating case is
   floor21_gate (Exp 45: balanced-val tail drop 0.0426 and magnets 0.0383
   against veto-side gains of +0.3987 and +0.3590, gain-to-loss ratios above
   9). Applied prospectively; recorded verdicts are not rewritten, but a
   configuration may be explicitly re-evaluated under this amendment with
   both verdicts recorded. Implemented in `passes_two_sided`
   (`analysis/hierarchical_pool.py`).
9. Camera-ready reporting role for the judge part (user decision 2026-08-06,
   taken with the adversarial plan review's findings in hand). For the ICML
   camera-ready, Table 1 cells are computed on the full kept pool (45,377,279
   lines), the only instrument comparable with the submission's existing rows
   (the submission's UniLID cell .929 and FPR 2.03e-5 reproduce from our
   full-pool baseline as 0.9292 and 2.0263e-5). The judge part of the seed-301
   split (27,002,441 lines) gains one additional role: the paper appendix's
   held-out comparison instrument, carrying the paired-bootstrap intervals,
   because it is the only kept-pool subset outside every selection listed in
   the camera-ready disclosure table (floor level, flat-four membership,
   D3_PROX, clause-A selection, draw-201 confirmation). Draw 201 remains the
   balanced final-reporting instrument of the internal protocol; its role is
   unchanged. This amendment adds a reporting use; it does not relax the
   one-confirming-measurement-per-candidate discipline for adoption decisions.

## Standing design constraints (why the methods look the way they do)

Every method in this project is designed against four constraints that were set by
the user and are not negotiable per experiment. They explain choices that would
otherwise look arbitrary.

1. **Add-a-language modularity.** Adding a new language must be a local operation:
   estimate that language's parameters from that language's own data, with the
   other 1,939 languages untouched. This rules out any globally coupled fit
   (discriminative training, MMI, a softmax over all languages) regardless of its
   accuracy, and it is why every adopted mechanism is either a per-language
   transformation of that language's own row or a decision rule whose parameters
   are calibrated on that language's own training text.
2. **Principled over ad hoc.** Preference for standard statistical constructions
   with few free constants (Good-Turing missing-mass estimation, quantile
   calibration) over tuned corrections. Where a constant is unavoidable it is
   pre-registered before the run that judges it, never swept afterwards.
3. **Likelihood-side preferred over prior-side.** Additive per-language biases
   (a class prior) are measured and kept as reference points but disfavored for
   adoption: they need global data to fit for a new language, and they showed
   per-language harm (Exp 16, 25).
4. **Per-language harm is bounded explicitly.** Aggregate improvement does not
   license a collapse on individual languages. This is encoded in the adoption
   rule's collapse clause and is why several strong-on-average candidates were
   rejected or flagged for investigation.
5. **Plan consistency review (user decision 2026-07-29).** Every experiment plan
   is checked against all prior recorded decisions (this document's adoption
   rule and amendments, the standing constraints, and the decision entries in
   the results and plan documents) before execution, by an adversarial review
   whose explicit charge includes finding silent reverts of or unlabeled
   amendments to earlier decisions. Any divergence from committed text must be
   labeled as an amendment with its own user decision. Origin: the 2026-07-29
   review of the combined-method plan found four such divergences (dropped
   adoption-rule clauses, a dropped verdict track, an unrecorded instrument
   redefinition, an unaddressed re-admission of a rejected treatment).

## Method families (2026-07-23 onward): configurations and motivation

Each family below states what the method computes, the constants and where they
came from, the reason for the design, and the outcome. Per-experiment results are
in `EXPERIMENTS_RESULTS.md`; this section is the configuration record.

### Floor equalization (`floor21`, Exp 20)

Every row contains a large block of entries at the row's exact minimum, which are
the tokens the estimator never observed for that language. That minimum is
resource-tied: correlation between a row's minimum and log10 of the language's
document count is -0.966, so small languages penalize unseen tokens far less than
large ones and attract text they should not. `floor21` clamps each row's minimum
block to `min(row_minimum, -21)`, one shared constant, nothing raised, observed
tokens and the four special tokens bit-identical
(`analysis/floor_equalization.py`). Motivation: it is the subtractive direction,
chosen after every mass-adding variant failed (Exp 9, 13, 18, 19), and it is fully
modular because the constant is shared and nothing is fitted. The constant was
selected from the grid {-17, -19, -21, -23} on validation data. Status: carried,
and top-ranked on the balanced validation set.

### Good-Turing unseen mass (`gt_min`, Exp 27, 28)

The principled replacement for the shared clamp: instead of one constant for all
languages, set each language's unseen-token mass to the Good-Turing plug-in
estimate from its own token counts. Procedure (`analysis/gt_counts.py`,
`analysis/full_test_gt.py`): count each language's training tokens under its own
Viterbi segmentation, giving total tokens T and the number n1 of token types
occurring exactly once; the four special tokens occupy exactly 0.8 of every row's
mass, so the seen-plus-unseen budget is exactly 0.2 and the target unseen mass is
`0.2 * n1/T`, spread uniformly over the row's minimum block, with the seen
entries rescaled by `(0.2 - target)/(0.2 - current)` so the row stays normalized.
No tuned constant exists anywhere in this method: n1 and T come from the
language's own data and 0.2 is a structural property of the model.

Pre-registered decision (fixed before the judging run, since every mass-adding
edit had failed): the one-sided variant `target = min(current, 0.2 * n1/T)`, which
never raises unseen mass. The counting pass then found the direction never binds:
the emergent floor overstates unseen mass for all 1,940 languages (tail median 9x,
head median 12x), so the one-sided rule and the exact plug-in coincide on this
model. Outcome: the best numbers ever measured on equal-volume validation data
(macro-F1 0.9841) and a false-positive explosion on natural-distribution data
(22,404 to 79,113 into tail labels). The mechanism is that per-language honesty
preserves and widens the between-language gap in unseen-token penalties, which is
what actually drives cross-language competition. Recorded conclusion: the floor
pathology is a between-language externality, and per-language calibration and
cross-language equalization are separate corrections that fix different halves.

### Per-language decision margins (the margin gate family, Exp 26, 31, 33, 34, 36)

A decision-layer method rather than a weight edit. For a line whose predicted
language L is small, compute the score gap between L and the runner-up; if the gap
is below a per-language threshold tau_L, the prediction is reassigned to a
larger-language candidate instead. Design decisions and their reasons:

- **tau_L is calibrated on L's own training lines** (the 5th percentile of the
  gaps on lines L itself wins), never on validation or test data. This satisfies
  modularity exactly: a new language brings its own threshold. It also bounds
  L's own recall loss by the quantile by construction, which is why a quantile
  was chosen over a tuned threshold.
- **Constants, all pre-registered:** quantile 5; minimum 200 self-won training
  lines to calibrate at all (languages below this are excluded from gating,
  listed in the report, and keep baseline behavior, never silently defaulted);
  at most 2,000 calibration lines per language; calibration seed 0; top-5
  candidate list.
- **The reassignment target evolved through three measured failures, and the
  progression is the family's main scientific content.** Reassigning to the
  runner-up moved false positives onto small close relatives (szy_Latn absorbed
  82 of pwn_Latn's lines). Gating only the smallest languages moved the burden
  onto mid-size languages under dominant neighbors (llb, arq, skr, vmk). Gating
  all languages below 18,000 documents moved it onto languages just above that
  threshold (aba, bam, llb, twx). The general law recorded: reassignment
  relocates false-positive burden to the lowest-capacity permitted target near
  the cluster. The fix is to require the target to be a top-resource language
  (at least 100,000 documents, matching the measured source profile of the
  original false positives: 98.9% come from sources with median 100,000
  documents), and to keep the prediction unchanged when no such candidate is in
  the top five.
- **Adaptive strength (`gt_margin_adaptive`, the user-requested variant).** A
  hard threshold at 18,000 documents creates boundary victims, so the quantile
  decays linearly with the language's size: `q_L = 5 * (1 - min(N_L, 18000)/18000)`.
  Full strength for the smallest languages, zero at the boundary. No new
  constants. This recovered nearly all of the validation-set cost of the
  all-language gate.

### Composition candidates (Exp 31, 33, 34, 36)

The Good-Turing rescale repairs within-language calibration and the margin gate
repairs cross-language false-positive inflow, so the compositions apply both, with
tau recalibrated under the modified weights (thresholds do not transfer when the
weights change). `gt_margin_adaptive` is the surviving member and is carried.

### The re-examination family and the promoted configuration (Exp 44 to 50)

This family extends the margin gate family above (same mechanism: a
per-language decision-time threshold, plus a minimum training-corpus size
for the replacement candidate), applied uniformly to all 1,940 languages on
top of the floor-21 weight matrix, then refined by four post-promotion
candidates (Exp 47 to 50). gate_flat4_prox21, described below, is the
configuration the project currently promotes.

**floor21_gate (Exp 44, 45; promoted 2026-07-30).** Floor-21 rows for all
1,940 languages (the "Floor equalization" mechanism above), plus the margin
gate applied to every language with fewer than 18,000 training lines
instead of only the smallest languages, with the replacement-candidate bar
raised to RES_CAP = 100,000 training lines. RES_CAP was established in Exp
33/34 (`EXPERIMENTS_RESULTS.md`, Exp 33 pre-registration of
`gt_margin_all_100k`): 98.9% of the false positives this family repairs
come from source languages with a median of 100,000 training lines, so a
reassignment target below that bar is not a genuine repair. Thresholds:
`tau_floor21_gate.csv`, one per gated language, at the size-adaptive
quantile of that language's own training-line score margins, q_L = 5 * (1 -
N_L / 18,000) for a language with N_L training lines, the same calibration
as `gt_margin_adaptive` above. Judge-part overall F1 0.9480, +0.0151
[+0.0112, +0.0191] over gt_margin_adaptive.

**Exp 47: shared re-examination threshold (in the pool, not promoted).**
Replaces the 1,080 per-language thresholds with SHARED_TAU = 9.0
natural-log units and lowers the replacement-candidate bar to 18,000
training lines. SHARED_TAU was chosen from a derivation-part sweep
(`outputs/diagnostic/gate_threshold_sweep_20260730.csv`, optimum flat
between 7 and 12). Scores highest on aggregate (judge-part overall 0.9534)
but fails the per-language collapse clause at class level: 9 languages with
judge-part support 15 to 2,947 lose more than 0.10 F1 against baseline,
because a per-language threshold set at a percentile of that language's own
margins bounds its own recall loss by that percentile and a shared value
does not. Not incorporated into the promoted configuration for this reason.
Recorded follow-up, not pre-registered: a hybrid threshold, the smaller of 9.0 and a
per-language cap from own-train margins.

**Exp 48: the four flat large-corpus languages (component of the promoted
configuration).** Four languages with more than 18,000 training lines each
have a token distribution unusually flat for their script: sco_Latn,
bjn_Latn, arg_Latn, vls_Latn, identified by the zH column of
`outputs/diagnostic/lang_diagnostic.csv` (their zH values are 1.567 to
3.228, median 2.51, against a median of 0.00 over all 1,940 languages).
floor21_gate's corpus-size-only gate exempts these four even though they
receive 63,842 of the 118,006 wrong predictions into small-language or
flat-distribution labels that remain after floor21_gate (54.1%). Exp 48
re-examines the four under their own thresholds (`tau_flat4.csv`), at a
fixed 5th percentile rather than floor21_gate's size-adaptive quantile,
which is zero at or above 18,000 training lines, and with the same RES_CAP
= 100,000 replacement-candidate bar. Judge-part overall F1 0.9486, +0.0006
[+0.0001, +0.0013] over floor21_gate.

**Exp 49: score-proximity condition on the replacement candidate (component
of the promoted configuration).** Adds one condition to the
replacement-candidate walk used by both the under-18,000 gate and the flat4
gate: a candidate must have a saved score within D3_PROX = 21.0 natural-log
units of the top-1 saved score, in addition to the existing RES_CAP =
100,000 training-line bar. D3_PROX was chosen on the derivation part from a
grid search of 0.5 to 100 in steps of 1; the optimum plateau spans roughly
15 to 35 within 0.0003 of the derivation-part score, so 21.0 represents the
plateau rather than a finely tuned value. Judge-part overall F1 0.9498,
+0.0018 [+0.0010, +0.0026] over floor21_gate. This condition was chosen
over target-identity conditions (accepting or rejecting a replacement by
which language it is): every target-identity form that repaired the
motivating errors was measured to cost aggregate F1, because it relocates a
false positive from a large language back onto the small source language,
which costs more under unweighted per-language averaging than it saves
(`EXPERIMENTS_PLAN.md`, "Candidate directions from the post-promotion error
analysis", direction 3).

**gate_flat4_prox21: floor21_gate's gate extended to the four flat
languages (Exp 48), with the score-proximity condition added (Exp 49).**
Judge-part overall F1 0.9498; promoted 2026-08-06 by user decision,
superseding floor21_gate, which remains in the pool. Confirmation on the
balanced test draw (seed 201): `outputs/tables/gate_flat4_prox21_confirmation_201.md`.

**Exp 50: pooled-frequency unseen-token values (in the pool, not
composed).** An alternative to the floor-21 weight-matrix edit: each row's
minimum-block entries are set to a shared constant `c` plus the token's log
probability in the pooled training-data distribution (`p_base`, already
stored in the model file as the base tokenizer distribution), with `c`
fixed so the mean assigned value over the non-special vocabulary equals
-21.0 (measured: c = -8.4740, assigned plateau range -27.61 to -12.31). The
flat -21 value is the special case of a flat `p_base`. Judge-part overall
F1 +0.000412 [+0.000043, +0.000837] over floor-21 solo (paired bootstrap,
B=10,000, seed 0), a gate-less comparison. The pre-registered composed step
(rebuilding the Exp 49 gate on this matrix) was declined by
the user (2026-08-06); this candidate stays in the pool at its gate-less
result, not incorporated into gate_flat4_prox21.

**User decisions in this family.** The order of trying directions 1 through
4 (2026-08-05); the floor21_gate promotion (2026-07-30); the amendment-8
conditional widening of the clause-(A) cap that made floor21_gate eligible
(2026-07-30, "Precision-primary adoption rule" amendment 8 above); the
gate_flat4_prox21 promotion (2026-08-06); declining the Exp 50 composed
step (2026-08-06). Of the constants, SHARED_TAU and D3_PROX were chosen on
the derivation part of the seed-301 split before the judging run. RES_CAP =
100,000 was established earlier, in Exp 33/34, from the measured resource
profile of the false positives this family repairs, and the per-language
threshold quantile (MARGIN_Q = 5) is the margin gate family's
pre-registered constant; neither was refit here. All four follow the
"principled over ad hoc" standing constraint above.

**The seed-301 split (Exp 44).** For any candidate whose rule or constants
are chosen using held-out remainder data (Exp 47 through 50, and the mixed
matrix before them), the 45,004,014-line held-out remainder is partitioned
by a seeded 40/60 split: RULE_SPLIT_SEED = 301, RULE_SPLIT_FRACTION = 0.40.
The resulting derivation part (18,001,573 lines, about 18.0M) is where
rules and constants may be chosen; the judge part (27,002,441 lines, about
27.0M) is used for the one confirming measurement per candidate, with
comparators recomputed there. This split replaced an earlier plan to
derive the combined-method rule from the balanced validation draw (seed
101): on that draw the per-group leader disagreed with the
held-out-remainder leader in all six groups tested, so the draw cannot
rank methods for the primary quantity (`EXPERIMENTS_RESULTS.md`, "Patterns
established by Experiments 44 to 50", pattern (a)).

### Post-hoc infrastructure conventions shared by all of the above

- Each candidate produces a full prediction memmap over the whole test pool, and
  every builder asserts bit-identity with its parent configuration outside the
  lines it is supposed to change, plus at least 99% agreement between the
  recomputed top-1 and the parent's stored prediction.
- Every rebuilt weight matrix is verified by sha256 against the fingerprint of
  the matrix that produced the recorded numbers, so a later evaluation cannot
  silently score a different matrix.
- Reports are regenerated from artifacts by scripts that gate on previously
  recorded values before emitting anything, so a wiring error aborts instead of
  producing plausible numbers.

## Camera-ready reporting conventions (2026-08-06, program E1-E4)

Conventions fixed before any camera-ready number is computed; the experiments
are pre-registered in `EXPERIMENTS_PLAN.md` ("Camera-ready evaluation program").

- **Instruments.** Table 1 cells: full kept pool (45,377,279 lines). Appendix
  held-out comparison: the judge part (amendment 9 above). No delta pairs terms
  from different line sets (Exp 16 conclusion 3). Every reported number's
  sentence or caption names its instrument (line set + metric view).
- **Macro F1.** Per-language F1 with the full confusion (all false positives
  counted), averaged unweighted over the 1,940 languages
  (`_per_lang_stats`, `analysis/metric_decomposition.py`). On both reporting
  instruments every language has support, so this coincides with
  `compute_metrics`' average over labels present in the gold labels.
- **Macro FPR.** Per language, FP / (FP + TN) with TN = kept_lines - support -
  FP, averaged unweighted; reproduces the paper's printed baseline cell
  (2.0263e-5 -> 2.03e-5).
- **Empty-after-preprocess lines** score as wrong for the scored system
  (`EMPTY=-1` sentinel), the recorded repo convention. Measured effect on the
  kept pool: zero such lines; on external benchmarks the count is reported.
- **External benchmark label mapping** (E2 fallback): labels mapped to
  `lang_Script`, intersected with the 1,940-label set, mapping table
  user-approved before scoring; label counts reported against the paper's
  366 (UDHR) and 190 (FLORES-200); ambiguous mappings abort. Acceptance gate
  before any new-configuration number: the rebuilt sets must reproduce the
  paper's UniLID baseline cells (UDHR .859, FLORES .932) within 0.005. The
  macro denominator for external benchmarks follows whatever convention that
  reproduction requires, and the choice is recorded here once measured.
- **Transfer statement of record** (corrects an earlier draft claim): in the
  promoted configuration, the per-language thresholds and q_L are own-train
  quantities; the 18,000 boundary and RES_CAP=100,000 are training-count
  quantities; D3_PROX=21.0 was selected on the GlotLID-C derivation part;
  flat-four membership uses one validation-derived input (magnet_ratio from
  the retired 250k validation half). Applying all of them unchanged to
  UDHR/FLORES is a no-refitting transfer test and is described as exactly
  that, not as "training-data-only".

## Per-language training pipeline and the trainer fix (2026-07-26)

**Pipeline.** A retrain takes a tokenizer's token inventory as a fixed
vocabulary and estimates per-language probabilities over it:
`UNILID/train.py --initial-vocab <tokenizer.json> --vocab-size <n> --byte-level
--per-lang-counts-method sp --lang-batch-size 20 --results-dir <dir>
--corpus-dir <dir> --reuse-corpus --skip-existing-langs`, then `convert.py` packs
the per-language tokenizers into a `.unilid` file. The vocabulary is seeded with
uniform log-probabilities (`_convert_to_unigram_base`), then each language runs
20 iterations of expectation-maximization over its own corpus using the forked
`spm_train` with pruning disabled, so every language ends with a probability for
every vocabulary entry. The per-language corpus split is tokenizer-independent
and is reused across retrains via `--corpus-dir`.

**Pre-flight checks (`analysis/preflight_131k.py`), all aborting on failure:** the
four special-token strings must exist in the tokenizer's vocabulary (the trainer
otherwise silently falls back to token id 0), the vocabulary size must match, and
the corpus split must contain 1,940 files totaling the recorded line count.

**Post-training gate (`analysis/degeneracy_scan.py`), added 2026-07-25.** Run on
every new model before any evaluation. It flags rows with fewer than 100 entries
above the row minimum. Two distinct causes produce such rows and must not be
confused: deterministic vocabulary coverage (a script with no multi-byte pieces
in the inventory; harmless for unique-script languages, damaging where several
languages share a script, as with the six Cree-syllabics languages) and genuine
training failure. The 100k production model has zero flagged rows.

**The fixed-vocabulary EM bug (found 2026-07-26, Exp 41; fixed 2026-07-27,
Exp 42).** The trainer's expectation step accumulated the forward-backward
quantities in 32-bit floats. On very long training lines the rounding error
breaks the identity that a lattice node's log-posterior is at most zero (measured
+351 on the trigger line), expected counts overflow to infinity, and the fork's
own guard mapped non-finite counts to zero, deleting exactly the most frequent
tokens and leaving a plausible-looking but collapsed model. Upstream
SentencePiece never encounters this because it skips lines longer than 4,192
bytes; this pipeline passes `--max_sentence_length=1000000` because discarding
training lines silently is worse. The trigger was one 142,136-byte line in the
Azerbaijani corpus, the longest line in all 1,940 corpora.

The fix (fork commits d0208d9, c5921a2; copy at
`patches/sentencepiece_fp64_estep.patch`) computes the trainer's forward-backward
in double precision and converts the non-finite-count guard into a hard failure.
It changes only the precision at which the expectation step's defining identity is
evaluated: the vocabulary, iteration count, prior, objective, and maximization
step are untouched, and the inference code paths keep their float versions so
previously shipped models behave identically. An unpatched rebuild reproduces the
previously installed binary's trained scores bit-for-bit, which is how the fix was
validated. The corruption was graded rather than binary: at least 94 corpora
contain lines above the upstream cap, and the 200k model's Azerbaijani row was
partially collapsed without ever crossing the degeneracy threshold. Both Apertus
models were retrained in full rather than repaired per language, so each model has
single-provenance weights.

## Reproducibility limitations of this record

- Only one git commit exists (`b7508fd`, 2026-04-08); per-experiment code versions are not
  separately tracked. Source-file mtimes are the only finer-grained timing signal.
  **Superseded as of 2026-08-06**: this described the state of the reconstructed
  repository as of the 2026-05-27 rebuild. Commits resumed on 2026-07-16, and
  since then the project has committed after most experiment launches and
  decisions; `git log --oneline` shows 56 commits as of 2026-08-06, one dated
  2026-04-08 and the other 55 dated 2026-07-16 or later. The one-commit
  limitation applies to code versions before 2026-07-16.
  `EXPERIMENTS_CHRONOLOGICAL.md` still cites no per-entry commit hash for this
  repository, so its entries must be matched to commits by date, not by a cited
  hash.
- SLURM submissions did not log seeds/commit/launch-command into the output beyond the
  submission scripts themselves; the scripts in `slurm_*.sh` are the reproducibility record
  for each job (kept in the repo).
- The recovered prompt history (`EXPERIMENTS_CHRONOLOGICAL.md` cites specific prompt numbers)
  is the only surviving record of design rationale that was not written into code or
  `EXPERIMENTS.md`.
