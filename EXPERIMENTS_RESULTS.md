# UniLID Analysis — Results Summary

> **Reconstruction provenance.** This file was reconstructed on 2026-05-27 after the
> original Claude session transcript (session `9729f7f3-3af8-42d5-818a-1f032a9f6f25`,
> 91 prompts, 2026-03-26 → 2026-04-08) was lost. It is a reorganization of content that
> already existed in `EXPERIMENTS.md` (the session's own write-up, last modified
> 2026-04-08) cross-checked against the analysis code, the generated tables in
> `outputs/tables/`, and the recovered prompt history. Every number below traces to a
> table file under `outputs/tables/` and the corresponding section of `EXPERIMENTS.md`.
> No results were regenerated; if a number cannot be reproduced from the cited artifact,
> treat it as unverified. Inferences are marked **[inferred]**.

This summarizes the most important findings. The detailed narrative (per-experiment
prose, by-axis breakdowns, full tables) lives in `EXPERIMENTS.md`. Experiment design and
search spaces are in `EXPERIMENTAL_SETUP.md`. Status of planned work is in
`EXPERIMENTS_PLAN.md`. Chronology and SLURM job records are in
`EXPERIMENTS_CHRONOLOGICAL.md`.

The system under study, **UniLID**, is a language-ID model: a single shared Unigram
tokenizer (100k vocabulary) with 1,940 per-language log-probability weight vectors.
A text is scored per language by `score(lang) = Σ log p(token_i | lang)` under that
language's own Viterbi segmentation; the argmax over 1,940 languages is the prediction.
Evaluation is on the GlotLID test set (45,627,279 samples). Most sweeps run on a 500k
uniform sample (`seed=42`, without replacement).

---

## Current state (2026-07-23)

- **Adopted configuration (provisional): floor-21**, selected by the precision-primary
  adoption rule (Exp 25; rule fixed by the user 2026-07-23, EXPERIMENTAL_SETUP.md).
  Provisional because the Good-Turing unseen-mass candidate (plan E3) is specified as
  the principled version that must beat it, and the margin method (E2) targets its
  residual. The learned bias reg=5.0 is REJECTED for adoption by the per-language
  collapse clause (llb_Latn -0.113) but remains the natural-traffic reference
  measurement (Exp 16).
- **Evaluation:** selection = balanced-val draw 101 shortlist (`passes_shortlist`);
  adoption = `passes_two_sided` (balanced-val stage with symmetric tail/magnet
  widening, full-pool precision veto minus draws 101/201, per-language collapse
  bound); headline = balanced test draw seed 201. Three earlier val-selected operating
  points were overturned at full scale before the balanced protocol existed.
- **Apertus 131k branch (Exp 29):** negative on both views (tail -0.0437
  within-stratum, FPs into tail 2.3x); the vocab-size regression is not an
  allocation problem. Discontinuation recommended, user decision pending.
- **Objective (user decision 2026-07-25):** the primary quantity is macro-averaged
  per-language score, every language weighted equally within reason (extreme
  low-resource exemptions allowed). Adopted interpretation, pending correction:
  per-language F1 on the full natural-distribution test data (all false positives
  counted), averaged unweighted over languages.
- **Carried-forward set (near-tie co-selection, user decision 2026-07-25):** the
  selection data cannot separate six eligible configurations (freq_prior,
  learned_bias, floor21, margin_q5, margin_q5_head, gt_margin_adaptive; balanced-val
  overall 0.9794-0.9800); all six stay live. Top-ranked floor-21; on the primary
  quantity the order is gt_margin_adaptive 0.9334, floor-21 0.9309 (Exp 36).
  Uniform-track champion gt_min (flagged: mev/sbs dig-ins done, Exp 32).
- **Next-round candidate (recorded, not pre-registered):** extend the margin gate
  to all non-head predicted labels, or repair the gt mid-band redistribution at the
  weight level (Exp 31 mechanism).
- **Natural-traffic objective:** the learned per-language bias reg=5.0 is the reference
  result (Exp 16: full-test overall +0.0129, tail -0.0018 [CI -0.0035, -0.0001],
  CommonLID 0.8452 -> 0.8879). The frequency prior (tail -0.0182) and floor
  equalization (tail -0.0204) are not adopted; Exp 24 shows those tail deltas are
  within-stratum (recall-view) numbers, and under global per-language F1 the same
  configurations raise tail mean F1 (baseline 0.5618, learned bias 0.6003, freq prior
  0.6800, floor-21 0.7655). Which view tail claims use is an open decision.
- **Metric views (Exp 24):** every stratum row and guard column is within-stratum
  macro-F1 and excludes cross-stratum false positives; the tail deficit under global
  per-language F1 is precision (0.459), not recall (0.874), with 22,522 false
  positives into tail labels, 98.9% from head sources.
- **Uniform-prior (balanced) objective:** the unmodified baseline is the best adopted
  configuration; two candidates passed selection on 2026-07-19 and are pending
  confirmation (Exp 23): punctuation partial pooling alpha=300 (no negative stratum,
  effect at the measurability edge) and the balanced-data bias refit reg=0.3
  (sel tail +0.0299, magnets +0.0252; adoption blocked on per-draw stability, a
  balanced-test draw, a full-test pass, and the per-language-suppression decision).
- **Refuted families:** mass-toward-group-typicality edits (Exp 9/13/18/19), length
  normalization (Exp 2/5), floor clamps in both directions (Exp 6, Exp 20/23a),
  heuristic variance reweighting (Exp 8a), entropy sharpening. The macrolanguage
  hierarchy is a null with a useful ceiling measurement (Exp 21).
- Open paths for the next experiments are consolidated at the end of
  `EXPERIMENTS_PLAN.md`.

## Headline observations (historical, Exp 1-9 era)

1. All evaluated scoring modifications reduce 500k-sample accuracy relative to the 0.960
   UniLID baseline: full length normalization (α=1) → 0.885; partial normalization at any
   α > 0.1 (best α=0.1 → 0.961, +0.001); floor=−10 → 0.916; heuristic discriminative
   weighting in all three setups (best A α=0.5 → 0.866). The accuracy reduction concentrates
   on the <30 char bin in every case.

2. Token-count delta (pred − true) on full-test-set misclassifications has mean −0.17
   (Cohen's d −0.092; one-sample t = −122.5, p ≈ 0). Mean magnitude grows monotonically with
   input length (<30: −0.11; 300+: −2.71). Median is 0 across all length bins except 300+
   (−1.00); 61.1% of misclassifications have equal token count under the predicted and true
   languages.

3. Mean KL(lang‖base) by resource bin: <500 → 0.32; 500–1k → 0.50; 1k–12k → 0.70; 35k+
   → 0.68. Saturates above ~10k samples; log(training count) vs KL r = 0.261.

Possible (unconfirmed) conclusions:
- The existing UniLID parameterization is near-optimal under sum-of-log-prob scoring;
  modifications that reduce per-token discrimination reduce accuracy most on short inputs.
- Low-resource (<5k sample) per-language distributions are under-fit (close to the base)
  rather than over-fit.
- Post-hoc probability-space blending alone does not jointly improve overall and very-low-
  resource accuracy in the configurations tested.

---

## Experiment 1 — Multi-system comparison

**Question:** How does UniLID compare to four alternatives (UniLID-DeepSeek, UniLID-Qwen,
UniLID-Marg, fastText) across text length, training-resource level, and script?

**Findings** (500k sample; full-dataset tables also generated):
- Overall accuracy is tight across systems: UniLID 0.960, Marg 0.961, DeepSeek 0.959,
  Qwen 0.951, fastText 0.947. fastText has the highest macro-F1 (0.947).
- Accuracy is strongly length-dependent: <30 chars ≈ 79% for UniLID, >300 chars ≈ 99.5%.
- Resource sweet spot at 12k–18k training samples (99.5%); the 35k+ bin is lower (95.8%)
  but is 92.8% of the data.
- Latin script (1,659 languages): 96.3%; Devanagari 89.6%, Arabic 90.7%; unique-script languages reach ~100%.
- UniLID's worst relative weakness: `lzh_Hani` (Literary Chinese), 25.0% error vs
  DeepSeek 3.9%. Its standout strength: `azj_Latn`, 1.1% error vs DeepSeek 61.2%.

**Artifacts:** `outputs/tables/table1_overall`, `table2_by_length`, `table3_by_resource`,
`table4_by_script`, `table5_error_overlap`, `table6_per_script_winner`,
`table7_divergences` (`.md`/`.tex`); confusion-matrix PNGs/TeX for 7 clusters.
**Detail:** `EXPERIMENTS.md` §1.

---

## Experiment 2 — Tokenization length bias and normalization

**Hypothesis:** Because UniLID sums per-token log-probs with no length prior, it may favor
languages that produce fewer tokens for the same text.

**Findings:**
- The bias exists and scales with length. Full-dataset misclassifications (1,789,423):
  mean token delta (pred − true) = −0.17, Cohen's d = −0.092. At 300+ chars the mean delta
  is −2.71.
- Pairwise counterfactual: normalizing by token count would flip 18.6% of errors toward
  the true language. When the predicted language uses ≥2 fewer tokens, ~75% of those errors
  are pairwise-correctable.
- **But full re-classification with normalized scores made accuracy worse: 0.960 → 0.885.**
  Raw rescore reproduced the original predictions exactly (100% agreement, validating the
  Rust implementation). Normalization broke far more predictions than it fixed (net
  −37,740), with the damage concentrated on short texts (<30 chars: 0.792 → 0.566).

**Possible (unconfirmed) conclusion:** the unnormalized sum-of-log-probs scoring carries
signal that simple length normalization removes; the per-token average is noisiest for
short texts.

**Artifacts:** `outputs/tables/length_bias`, `normalized_comparison`;
`outputs/figures/length_bias_histogram.png`. **Detail:** `EXPERIMENTS.md` §2.

---

## Experiment 5 — Partial length normalization (alpha sweep)

**Question:** Does a partial correction `score / n_tokens^alpha` for `alpha ∈ {0.0,…,1.0}`
help, even if full normalization (alpha=1) hurts?

**Observations:** Best is `alpha=0.1` at 0.961 accuracy (+0.001 over `alpha=0.0`),
net +114 corrections out of 1,749 changed predictions. Accuracy decreases monotonically for
`alpha > 0.1`; the <30 char bin drops fastest (0.792 → 0.566 at `alpha=1.0`).

**Artifacts:** `outputs/tables/alpha_sweep`; `outputs/figures/alpha_sweep.png`.
**Detail:** `EXPERIMENTS.md` §5.

---

## Experiment 6 — Log-probability floor sweep

**Question:** Does clamping all per-language weights at a higher floor (giving OOV tokens a
finite low probability) help?

**Observations:** `floor=-22` clamps 0 elements; predictions identical to baseline.
`floor=-15` clamps 90.7% of the matrix and changes 3,372/500k predictions, net −109.
`floor=-10` clamps 99.5%, accuracy 0.960 → 0.916 with the <30 char bin going 0.792 → 0.608;
25,485 predictions broken vs 3,390 corrected.

**Note:** A code comment in `analysis/floor_sweep.py` refers to "OOV at -1e30". The user
flagged (recovered prompt, 2026-04-06) that `-1e30` is never used in the repo and the
actual clamp value differs. The comment is inaccurate; trust the runtime weight matrix.

**Artifacts:** `outputs/tables/floor_sweep`; `outputs/figures/floor_sweep.png`.
**Detail:** `EXPERIMENTS.md` §6.

---

## Experiment 3 — Per-language distribution analysis

**Question:** How do the EM-estimated per-language distributions differ from the base
distribution and from each other, and where does EM noise appear?

**Findings:**
- KL(lang‖base) is highest for unique-script languages (~1.2–1.3) and lowest for
  low-resource Latin-script languages (~0.20–0.24), which barely moved from the base.
- Mean KL rises with resource level then saturates above ~10k samples
  (log-count vs KL: r=0.261).
- Pair with one data-poor language: Hindi/Angika (anp 4,499 samples) has correlation 0.884
  (Indonesian/Malay 0.896 at 100k each) but MAD 2.218 (Indonesian/Malay 0.364), with many
  tokens near the probability floor.
- Within-cluster pairwise KL vs min training size: r ≈ −0.03 (≈ zero).

**Resource-level structure:** mean KL from base 0.32 at <500 samples vs 0.68–0.71 at 5k+;
mean MAD 1.17 vs 4.84–5.90. EM uses no explicit regularization (only a probability floor
and convergence early stopping).

**Possible (unconfirmed) conclusion:** low-resource (<500 sample) distributions are
under-fit (close to the base) rather than over-fit; for mixed-resource pairs the lower-
resource distribution shows distinct EM noise.

**Artifacts:** `outputs/tables/distribution_analysis`;
`outputs/figures/kl_vs_training_size.png`, `pairwise_logprob_scatter.png`,
`pairwise_kl_vs_training.png`. **Detail:** `EXPERIMENTS.md` §3.

---

## Experiment 4 — Token classification for confused pairs

**Question:** Are the tokens that drive within-pair divergence linguistically meaningful or
artifacts (punctuation, encoding, domain markers)?

**Observations:** Across 300 tokens (15 pairs × 20), morphological affixes contribute 32.6%
of KL and content words 22.8% (combined 55.4%); function words 15.7%; punctuation 10.5%;
character/phonotactic 10.2%; multi-word units 7.9%; script/encoding 0.0%; domain/religious
0.3%. The one mixed-resource pair (Hindi/Angika) is 75% content words, with the Angika
distribution showing many tokens at or near the probability floor (Exp 3.5). Indonesian/
Malay shows 7.3% domain KL from `ĠYehuwa` (JW.org marker).

**Possible (unconfirmed) conclusions:** discriminative features in the top 20 KL tokens per
pair are dominated by linguistic units rather than tokenization or encoding artifacts; the
Indonesian/Malay distinction is partially dependent on religious-text domain markers.

**Artifacts:** `outputs/tables/token_classification`;
`outputs/figures/token_categories_stacked.png`. **Detail:** `EXPERIMENTS.md` §4.

---

## Experiment 7 — Training-data analysis

**Question:** Does the training corpus have domain skew, quality problems, or script
mislabeling that explains confusions?

**Findings** (full corpus, 60,683,151 lines):
- 98.1% of training data classifies as "other" (not religious, not Wikipedia); 1.9%
  religious; Wikipedia markers 0.002%. By resource bin religious share ranges 0.3%–2.7%.
  Confusable cluster examples: Indonesian (ind_Latn) 2.2% religious vs Malay (zsm_Latn)
  0.2%; Bokmål (nob_Latn) 6.2%. **Caveat:** the heuristics are conservative (keyword/
  pattern matching); true religious fraction is likely higher.
- Low-resource languages (<500) have shorter texts (~84 chars) and tiny vocabularies
  (~1.4k tokens) vs ~100k for the 35k+ bin.
- Script labels: 20 languages have >5% off-script characters. 6 "Canadian Aboriginal
  Syllabics" languages are actually 100% Latin romanization; Japanese `jpn_Jpan` is mostly
  Hiragana (a script-code mapping artifact, not a data problem). 99.0% of languages are
  >95% in-script.

**Scope note:** sub-analyses 7.2 (mislabeling) and 7.3 (overlap) were deferred. See
`EXPERIMENTS_PLAN.md`.

**Artifacts:** `outputs/tables/train_data_analysis.md`;
`outputs/figures/train_domain_stacked.png`, `train_quality_scatter.png`,
`train_script_purity.png`. **Detail:** `EXPERIMENTS.md` §7.

---

## Experiment 8a — Heuristic discriminative weighting

**Hypothesis:** Variance-based up-weighting of discriminative tokens within a confusion
cluster could improve within-cluster separation.

**Observations:** All three setups reduce accuracy at every parameter setting tested. At
the mildest settings: A(α=0.5) 0.866, B(α=0.5) 0.889, C(β=1.0) 0.899, vs 0.960 baseline.
Per-cluster accuracy is 0 across all seven clusters for A and B at α ≥ 1.0.

**Possible (unconfirmed) conclusion:** variance-based token re-weighting at the granularity
used here does not improve within-cluster discrimination; any improvement, if it exists,
requires a mechanism other than per-token additive/multiplicative adjustments to the EM-
trained weights.

**Artifacts:** `outputs/tables/discriminative_heuristic.md`. **Detail:** `EXPERIMENTS.md`
§8a.

---

## Experiment 9 — Distribution transfer for low-resource languages

**Hypothesis:** Interpolating an under-fit low-resource distribution toward a related
high-resource language (9a) or a script-average (9b) could raise low-resource accuracy.

**Observations** (probability-space interpolation, `lambda=1` is baseline):
- **9a related-language transfer:** <500 accuracy 0.789 → 0.895 at λ=0.3 (+10.6pp);
  500–5k peaks at 0.968 at λ=0.7 (+1.0pp); overall accuracy 0.960 → 0.947 at λ=0.3. For
  λ ≤ 0.3, accuracy on all three groups drops sharply (e.g. <500: 0.053 at λ=0.0).
- **9b script-average transfer:** overall accuracy stays 0.960–0.961 across λ ∈ [0.1, 1.0];
  <500 accuracy does not exceed 0.789 (baseline) at any λ<1.0 and falls to 0.526 at λ=0.1.
- Neither approach increases both overall and <500-group accuracy simultaneously in the
  tested range.

**Possible (unconfirmed) conclusion:** probability-space interpolation of EM-trained
per-language distributions toward a related-language or script-average distribution does not
jointly improve overall and very-low-resource accuracy in the configurations tested.

**Artifacts:** `outputs/tables/transfer_sweep.md`;
`outputs/figures/transfer_sweep.png`. **Detail:** `EXPERIMENTS.md` §9.

---

## Experiment 10 — Error analysis (signal for the pooling direction)

**Question:** Is there a qualitative trend in UniLID's errors that points to a focus for
improvement? Run 2026-06-24 via a 7-cut agent workflow over a 28,527-error stratified
sample of the full test set plus a per-token score decomposition and a weight-matrix audit
(scorer validated: `score(pred) >= score(true)` for 97.3% of recorded errors).

**Central finding:** UniLID's errors are dominated by the **under-fit low-resource tail
acting as false-positive attractors that steal predictions from the high-resource head.**
In every dominant confusion pair the low-resource sibling wins (eng->sco 48,620 forward vs
151 reverse, 322x). 84.7% of short-text errors route a high-resource truth to a
non-high-resource sink (true high-resource 64.5%, pred high-resource 13.2%). 86% of errors
predict a language ~30x rarer than the truth. The direct cause is the absent language prior
plus under-regularized EM vectors. The per-language smoothing floor is resource-tied
(`corr(floor, log10 count) = -0.966`, range -13.2 to -19.9), so small languages
under-penalize the unseen; 86.4% of the score gap toward the wrong language comes from short
non-content tokens (<=3-char subwords 51%, punctuation 20%), content words only 13.6%.
Whitespace/digit columns are the cleanest part of the matrix, not noisy magnets.

**Two attractor types with opposite pooling prescriptions:** (a) FLAT promiscuous magnets
(kzn, tly, vol, ido, mlt, qus) pulling from 25-111 unrelated languages, 0 true-label
appearances, 40% all-caps; all-caps is 62x over-represented in errors. Shrinking these is a
strict win. (b) TIGHT sibling sinks (glk<-fas, anp<-hin, wuu<-cmn); symmetric pooling would
blur them, so shrink only the low-resource member one-directionally. 62% of top-confusion
volume is equal-resource near-twins (ind/zsm, kin/run) where pooling is a blur risk.

**Ceiling:** ~4% clean-text mislabels, ~14% unverifiable content, ~20% arbitrary
macrolanguage splits (hbs/srp; arb->arz is 91% diacritized scripture, a diacritization
domain shift). fastText recovers 63.7% of UniLID's errors but is no better on the genuine
twins. So ~80-85% of the error budget is recoverable, and pooling must NOT be credited with
the twin share.

**Implication:** pooling must be gated by flatness and distance-to-confuser and evaluated
stratified (tail vs twins vs head) or aggregate macro-F1 can move the wrong way. Detailed
findings are in the project memory (`unilid-error-analysis-findings`).

**Artifacts:** `scratchpad/error_stats.json`, `scratchpad/errors_sample.jsonl` (analysis
inputs). **Status:** drives the hierarchical-pooling program (`EXPERIMENTS_PLAN.md` Exp 11-15).

---

## Experiment 37: the azj_Latn collapse is deterministic and numerically diagnosed (2026-07-25, isolated re-run)

**Procedure:** azj_Latn's per-language training re-run in isolation (same corpus
file, same 131k base tokenizer, same trainer call as job 2883222; output to a
scratch directory, the packed model untouched; recipe in the chronological log).

**Result: byte-identical reproduction.** 7 entries above the row minimum,
entropy 1.609 nats, same EM trace. The collapse is DETERMINISTIC, not stochastic:
across the whole project there is now zero evidence of random EM degeneration.
The same corpus file produced a healthy azj row in the 200k retrain, so the
trigger is the corpus-plus-131k-seed-vocabulary pair specifically.

**Diagnosis.** The 7 surviving vocabulary entries are the four special tokens
plus 'ĠMun' (all at probability 0.2) and 'gin'/'ayar' at trace level; the corpus
is ordinary Azerbaijani prose. The EM objective WORSENS from its start (374 to
2746 at sub-iteration 1, total distribution replacement, then frozen at machine
zero deltas): a likelihood that deteriorates and freezes is a numerical breakdown
in the fixed-vocab EM fork (cimeister/sentencepiece, branch fixed-vocab-em), not
a legitimate optimum. Root cause inside the C++ E/M steps is an open item for the
fork; the one-command reproduction is the handoff.

## Experiment 36: gt_margin_adaptive judged; both pre-run predictions confirmed; floor-21 retains selection 5/5 (2026-07-25, job 2895821)

**Verdict: ELIGIBLE, flagged (ota_Arab only); not selected.** Both on-record
predictions confirmed: reassignments 325,546 (down from round 3's 407,562;
207,241 to the true label) and ota_Arab remains the sole flag (its FPs are
high-margin weight-side flips no quantile catches; mechanism unchanged from
Exp 34). Rows: balanced-val 0.9798 / 0.9580 / 0.9503 / 0.9408 / 0.9808
(overall/tail/magnets/twins/head); veto overall 0.9334, the best of all eleven
configurations tested (floor-21 0.9309).

**Selection.** The natural-track ranking margin over floor-21 is 0.0002 of
balanced-val overall. Stability check across all five val draws (the instrument
built for exactly this): floor-21 leads 5/5 with margins 0.0000-0.0002,
consistently signed, so floor-21 retains selection under the standing rule and
the margin is not draw noise. The two candidates are near-equivalent on the
selection instrument while gt_margin_adaptive is better on the veto and on
balanced-val tail (0.9580 vs 0.8942). Open rule-design question for the user,
recorded not decided: whether near-ties on the selection instrument (margin
below, say, 0.001) should break on the veto overall, which would select
gt_margin_adaptive here.

**Family arc closed (four rounds).** gt_min (rejected, FP explosion) ->
runner-up gate (szy mechanism) -> head-targeted (lowmid class) -> 100k bar
(barely-head class closed, ota residue) -> N-adaptive strength (boundary cliff
removed, val cost recovered from 0.9744 to 0.9798). Each rejection produced the
next candidate's mechanism; the family ends with two eligible compositions and
the reassignment law as a documented result.

## Experiment 35: the EM-degeneracy question bounded (2026-07-25, analysis)

**User concern:** if per-language EM occasionally degenerates, previous "failures"
are in question. **Finding: the concern is bounded to one language-run.** Scan of
all three packed models (`analysis/degeneracy_scan.py`,
`outputs/tables/degenerate_rows.md`, threshold 100 estimated tokens):
- 100k model: 0 degenerate rows. Every adoption verdict and every method
  experiment (Exp 1-28, 31-34) rests on this model; none is affected.
- 200k: 17 flagged rows; 131k: 18. The two sets are near-identical (Syriac,
  Cherokee, Coptic, Cree syllabics, Gothic, Kali, Limbu, Lisu, Meetei, Mongolian):
  this class is DETERMINISTIC vocabulary coverage, not EM instability. The Apertus
  BPE inventory holds no multi-byte merges for these scripts, so byte-level
  pretokenized text exposes only the ~60-90 single-byte pieces and the fixed-vocab
  EM correctly estimates only those (csw's EM log converges normally, L1 delta
  0.4 to 0.03). Unique-script members are harmless (cop/lis/chr at F1 ~1.0); the
  six Cree-syllabics languages share the script and byte-only pieces cannot
  discriminate within it (csw 0.088, cwd 0.542). The same class was present,
  undetected, in the Exp 15 200k retrain: its tail deficit includes this coverage
  effect, a recorded caveat on the Exp 15 magnitude (direction unchanged).
- azj_Latn at 131k is the ONLY anomaly outside that class in 3,880 Apertus
  per-language EM runs (absent from the 200k and 100k flagged sets); the adjacent
  EM trace freezes at machine-zero deltas from sub-iteration 2, a genuine collapse.
  Open item: re-run azj's EM in isolation to test determinism, and attribute the
  trace conclusively (batched logs interleave languages).

**Process change:** `degeneracy_scan.py` is the post-training gate for any future
retrain: run it on every new .unilid before evaluation; flagged rows outside the
known unique-script exemption block the evaluation.

**Pre-registration: candidate `gt_margin_adaptive` (recorded before any run;
user-requested direction).** Identical to gt_margin_all_100k except the gate
strength adapts to training size: the calibration quantile becomes
q_L = MARGIN_Q * (1 - min(N_L, HEAD_N)/HEAD_N), so suppression strength decays
linearly from q=5 at N=0 to zero at the head boundary. No new constants (reuses
MARGIN_Q and HEAD_N); removes the threshold cliff that created boundary victims;
gated set and the 100k target bar unchanged. One candidate; both tracks.

## Experiment 34: gt_margin_all_100k, the first fully eligible gt-family candidate; floor-21 retains selection (2026-07-25, job 2895683)

**Verdict: ELIGIBLE, flagged (single outlier ota_Arab); not selected.** Build:
407,562 reassignments (251,419 to the true label, the highest recovery of the
family; 94,462 below-tau lines kept for lack of a 100k-bar candidate in the
top-5). Rows: balanced-val 0.9744 / 0.9579 / 0.9479 / 0.9392 / 0.9808
(overall/tail/magnets/twins/head); veto overall 0.9330 (top tier, floor-21
0.9309), tail global F1 0.4621, FPs into tail 28,743. Raising the target bar
eliminated the barely-head collapses entirely (aba/bam/llb/twx all recover), and
the natural-track ranking still selects floor-21 (balanced-val overall 0.9800 vs
0.9744: the gate's suppression cost on balanced data). Not a uniform-track passer
(val overall must improve there).

**ota_Arab dig-in (required by the flag).** ota (N=674, tail, Ottoman Turkish)
receives 395 new false positives, 295 from fas_Arab, 0 recall loss. It is NOT a
reassignment receiver (below the target bar): these are gt-weight-side flips whose
margins exceed tau_ota, i.e. confidently wrong under the GT floors. A quantile
gate cannot catch high-margin flips without destroying recall; the fas/ota pair
(Persian-influenced Ottoman orthography) is the gt-side residue class, distinct
from the reassignment law of Exp 31/33.

**Round and family summary.** Three margin-composition rounds converged: the
reassignment law (burden relocates to the lowest-capacity valid target) is closed
by the 100k bar at the cost of one gt-side residual flag. Final standings under
the amended rule: natural-traffic champion floor-21 (selection margin 0.0056 of
balanced-val overall over the fully-repaired composition); uniform-prior champion
gt_min (flagged, mev/sbs dig-ins on record); gt_margin_all_100k stands as the
eligible composition demonstrating that the two-correction decomposition
(within-language calibration + FP-side repair) can be made per-language-safe.

## Experiment 33: gt_margin_all judged; reassignment burden relocates to barely-head sinks (2026-07-25, job 2895566)

**Verdict: REJECTED on both tracks** (val overall drop > 0.01; 4 supported
collapses, worst -0.3211), with the best natural-veto aggregates of any candidate
tested: overall 0.9121 to 0.9331 (above floor-21's 0.9309), tail global F1 0.5373,
lowmid global F1 0.9267 to 0.9554, FPs into lowmid 451,042 to 139,506, FPs into
tail 19,390. Build: 461,605 reassignments over 1,080 gated labels, 235,421 to the
true label (`outputs/tables/gt_margin_all_build.md`).

**Mechanism (new, third of its kind).** The four collapsed languages (aba_Latn
N=18,107; bam_Latn N=18,697; llb_Latn N=25,228; twx_Latn N=26,573) are all
barely-head languages just above the 18k threshold: valid reassignment targets
that received the relocated burden. Pure precision loss (llb precision 0.878 to
0.437; recall unchanged or improved in all four). Across three rounds the same
law: runner-up targeting moved FP burden to small relatives (szy), tail-only
gating moved it to lowmid sinks (arq/skr/llb/vmk), all-label gating moves it to
barely-head sinks. Reassignment relocates FP burden to the lowest-capacity valid
target near the cluster.

**Pre-registration: round-3 candidate `gt_margin_all_100k` (recorded before any
run).** Identical to gt_margin_all except the reassignment-target bar rises from
HEAD_N=18,000 to RES_CAP=100,000 (the established resource cap; 98.9% of the
original false positives come from sources with median N = 100,000, so the
returned lines belong at top-resource labels). Gated set unchanged (N < 18,000);
if the top-5 holds no candidate at the bar, keep the gt_min prediction. One
candidate; both tracks; round closes on its verdict.

## Experiment 32: victim dig-ins, a degenerate-row finding in the 131k model, and the round-2 pre-registration (2026-07-24)

**Dig-ins required by the outlier-tolerant clause
(`outputs/tables/victim_digins.md`, `analysis/victim_digin.py`).** One mechanism
covers every flagged or collapsed victim: false-positive INFLOW at a non-head
label, never recall loss (recall lost is 0-9 lines in every case). llb_Latn gains
1,356 FPs under the learned bias (many small Bantu sources whose suppression
redirects their lines into the mid-sized Bantu sink) and 2,695 under gt_margin
(ndo/kua/bem/nya/zul); arq_Arab gains 765 from the Arabic cluster (ary/arb/fas);
skr_Arab 636 from pnb/urd; vmk_Latn 463 from vmw/ngl; sbs/mev on the balanced draw
gain scattered small-language FPs. Conclusion: the flagged outliers and the
rejected class share one addressable failure mode, and a margin gate that defends
ALL non-head labels addresses every observed case.

**Degenerate rows in the 131k model.** azj_Latn's 131k row collapsed in EM:
entropy 1.609 nats, 131,065 of 131,072 entries at the floor (about 7 estimated
tokens), recall 0.0000; its 229k test lines scatter to tly (161,886!), crh, tat,
tur. A matrix scan finds 18 rows with fewer than 100 estimated tokens (0 such rows
in the 100k model); most are unique-script languages where this is harmless
(cop/lis/chr at F1 ~1.0), but azj_Latn (head, Latin) and the Cree-syllabics
cluster (csw 0.088, cwd 0.542) are genuine per-language training failures.
Counterfactual with azj-true lines removed: 131k FPs into tail fall from 51,926 to
32,161 (the single failed row explains about two thirds of the FP increase);
tail global F1 gap narrows to 0.5627 vs 0.4258 and overall to 0.9287 vs 0.9196.
The Exp 29 verdict stands (the branch loses every aggregate even without azj) but
its magnitude was overstated by one EM failure. Repair path if the branch is ever
revived: delete the affected langspec files and re-run per-language EM with
--skip-existing-langs (independent per language), then repack; not run now.

**Pre-registration: round-2 candidate `gt_margin_all` (recorded before any run).**
gt_min weights plus the margin gate extended to ALL predicted labels with
N < 18,000 (tail and lowmid), tau recalibrated under gt_min per gated language on
its own train lines, head-targeted reassignment, all constants unchanged
(MARGIN_Q=5, MIN_CALIB_LINES=200 with exclusion logged, CALIB_MAX=2000,
CALIB_SEED=0, TOPK_MARGIN=5, HEAD_N=18,000). Directly motivated by the dig-in
mechanism. One candidate; judged on both tracks under the amended rule.

## Experiment 31: amended gating, dual-track verdicts, and the gt_margin composition (2026-07-24)

**Gating amendments** (user-invited reconsideration; EXPERIMENTAL_SETUP.md
"Amendments"): (B)-overall softened to a bounded drop; a uniform-prior track added
(`passes_uniform` selection on balanced val, collapse-checked confirmation on the
balanced test draw for the single track-selected candidate); ITERATE verdict lane.
Delta-reviewed: verdict-neutral for the first round; select-on-val/confirm-on-test
discipline preserved with the multiplicity count recorded.

**Dual-track outcomes (final, under the outlier-tolerant collapse clause added
the same day, amendment 4).**
- Natural-traffic track: floor-21 remains champion. learned_bias returns to
  eligible, flagged for the llb_Latn dig-in; ranking unchanged (floor-21 0.9800
  vs 0.9799).
- Uniform-prior track: **gt_min is the champion, flagged** with two required
  dig-ins from its balanced-test confirmation (mev_Latn -0.172 n=12, sbs_Latn
  -0.182 n=12). Under the first, outlier-intolerant clause it had been blocked
  outright; the amendment converts that block into targeted investigations, which
  is the intended semantics: the bound catches class-level harm, and two flagged
  outliers are not a class.

**gt_margin (pre-registered composition, built and judged; reviewed, no defects;
`outputs/tables/gt_margin_build.md`).** The recalibrated gate reassigns 60,320 of
gt_min's 86,924 tail predictions (28,533 to the true label; 22 languages under the
calibration floor). It repairs gt_min's headline pathologies: FPs into tail fall
from 79,113 to 19,390 (below the baseline's 22,404) and supported clause-C
collapses fall from 12 to 4. It passes the natural track's stages (A) and (B)
outright. Verdicts:
- Natural track: REJECTED on clause (C) alone: arq_Arab -0.131 (n=271), llb_Latn
  -0.206 (n=4,181), skr_Arab -0.192 (n=157), vmk_Latn -0.187 (n=93). These are
  LOWMID victims of the gt weight side; the gate defends only tail labels, so
  mid-band languages with dominant neighbors inherit gt_min's damage. llb_Latn is
  now a chronic victim across method families (learned bias -0.113, gt family
  -0.206).
- Uniform track: passes selection (0.9827, ranked under gt_min); a recorded second
  look at the test draw (multiplicity noted) shows it would also fail that
  confirmation (knx_Latn -0.111 n=15, sbs_Latn -0.107 n=12, sdc_Latn -0.160 n=30).

**Round closed per pre-registration.** Recorded mechanism for the next round: the
gt-family damage that survives composition is concentrated in lowmid languages
under dominant neighbors (the arq/skr/llb/vmk pattern, the same population as the
tight_lowres category), and the decision-layer defense must cover those labels,
not only tail: either extend the margin gate to all non-head predicted labels
(tau for every language with sufficient train data) or repair the gt mid-band
redistribution at the weight level. Not pre-registered; next round's candidate.

## Pre-registration: composition candidate gt_margin (recorded 2026-07-24 before any run)

Config `gt_margin`: the gt_min weight matrix (Exp 28) combined with the
head-targeted margin gate (Exp 26 rule), with tau_L RECALIBRATED under the gt_min
matrix (margins change when weights change; the Exp 26 tau values do not transfer).
All constants unchanged from their pre-registrations: MARGIN_Q=5,
MIN_CALIB_LINES=200, CALIB_MAX=2000, CALIB_SEED=0, TOPK_MARGIN=5, HEAD_N=18,000,
one-sided-min GT. Rationale: Exp 28 showed gt_min repairs the within-language
ranking (recall) and the margin gate repairs FP inflow (precision); this is the
first candidate that separates the two corrections explicitly. Judged on both
tracks of the amended rule. One candidate; if it fails, the failure mechanism is
recorded and the round closes.

## Experiment 30: the 131k does not repeat the baseline's errors; its regression is concentrated (2026-07-24, analysis)

**Question (user):** does the 131k model repeat exactly the same errors as the 100k
baseline? **No.** Line-exact overlap (`outputs/tables/error_overlap_131k.md`,
`analysis/error_overlap_131k.py`, accuracy gates passed):
- Of the 100k's 1,779,499 errors, 57.7% recur under 131k, and only 66.2% of those
  pick the same wrong label. The 131k fixes 753,463 errors (42.3%) and introduces
  733,388 (net -20,075; accuracy +0.0004).
- The tokenizer's documented strengths show up: 35% of the 100k's Indic-script
  errors are fixed (net positive there), and the improvement list is tail/lowmid
  heavy (syl_Latn +0.381, pwn_Latn +0.258, tig_Ethi +0.248, lad_Hebr +0.241,
  tcy_Knda +0.198). 190 languages improve by more than 0.01.
- But 403 languages regress, and the tail-FP explosion is CONCENTRATED: the single
  pair tat_Latn <- azj_Latn carries 17,603 false positives (azj_Latn, a head
  language and tat's close Turkic relative, collapses to F1 0.006 from 0.999),
  accounting for over half of the FP increase (51,926 vs 22,522). The remaining
  structure mirrors the baseline's known pairs (pnt<-ell 2,810, sbs<-Bantu,
  mrq<-tah).

**Reading:** the 131k base is a different error trade, not a uniform degradation:
real wins where the tokenizer adds coverage, broad small-language losses from
flatter distributions, plus one pathological relative pair (azj/tat) that a repair
layer (floor edit or margin gate on that model) or an EM inspection of tat's 131k
row might address. This softens the Exp 29 discontinuation reading from "the branch
is dead" to "the branch loses on net as a drop-in but contains recoverable
structure"; the discontinuation recommendation stands for the drop-in use, and the
131k memmap remains available for hybrid analyses.

## Experiment 29: Apertus 131k multilingual base does not reverse the vocab-size regression (2026-07-24, jobs 2883222 + 2885941)

**Hypothesis:** the 200k retrain's tail regression (-3.4pp, Exp 15) came from
vocabulary allocation, so `preliminary_mul` (131,072; documented in the tokenizer
project as highest compression on Indic, Chinese, and the low-resource tail) should
reverse it.

**Result: refuted; the regression tracks vocabulary size, not allocation.** Retrain
completed in one 9.8 h window (all 1,940 languages, standard setup, corpus split
reused; `glotlid_apertus131k.unilid`). Full-test b=0 baseline vs the 100k baseline
(`outputs/tables/full_test_eval_131k.md`):
- Within-stratum: overall -0.0113, tail -0.0437 [CI -0.0515, -0.0371], magnets
  -0.0352, twins -0.0044, head -0.0019; accuracy +0.0004.
- Global view: tail mean F1 0.5618 to 0.4046; false positives into tail labels
  22,522 to 51,926 (2.3x); flat_magnet 0.4716 to 0.3551; every group lower.
- Balanced val: overall 0.9766 vs 0.9811, tail 0.8679 vs 0.9170.

Both objectives agree, so this is not metric-conditional. Reading: with the training
data fixed, a larger vocabulary means more parameters per language and flatter
low-resource distributions, so tail models under-fit harder and their labels take
more head traffic (the FP doubling is the Exp 24 absorption mechanism amplified).
Better tail compression in the tokenizer does not compensate; the 131k regression
(-0.0437 within-stratum tail) is in line with the 200k's (-3.4pp on the older eval).
Per the plan's branch criterion the 131k base should not be continued;
recommendation recorded, decision with the user. A retrain-side counterfactual worth
noting for any future vocab work: per-language vocabulary truncation (each language
keeping only its top-k pieces) would decouple shared-vocab size from per-language
parameter count.

**Artifacts:** `outputs/tables/full_test_eval_131k.md`,
`outputs/diagnostic/full_test_131k_per_lang_prf.csv`, model + memmaps on scratch
(`glotlid_apertus131k.unilid`, `full_test_eval_131k/`). Scripts:
`slurm_apertus_train_131k.sh`, `analysis/full_test_eval_131k.py` (both reviewed).

## Experiment 28: gt_min judged; per-language honesty is not the fix for a between-language problem (2026-07-24, job 2884210)

**Verdict: REJECTED by the adoption rule; floor-21 remains selected.** Single
pre-registered candidate, no sweep. Full tables in `outputs/tables/full_test_gt.md`
and `two_sided_selection.md`.

**The two views split harder than for any previous config.**
- Selection view (balanced val): gt_min is the best configuration measured on this
  protocol: overall 0.9841 (baseline 0.9811), tail 0.9769 (+0.0599), magnets 0.9688
  (+0.0514), twins/head flat. Full-test within-stratum: tail +0.0656
  [CI +0.0603, +0.0729], magnets +0.0528 [+0.0478, +0.0588].
- Veto view (natural traffic): false positives into tail labels rise from 22,404 to
  79,113 (3.5x); tail global mean F1 drops 0.3382 to 0.2950 despite tail recall
  rising 0.8664 to 0.9675; overall global macro-F1 drops; 12 languages above the
  support floor lose more than 0.1 (worst -0.2123).

**Mechanism.** Exp 27 showed every floor is individually overstated (~10x), and
correcting each language against its own data fixes the ranking on genuinely
low-resource text: that is the recall/selection-view gain. But the argmax competes
ACROSS languages, and the honest per-language estimates preserve and even widen the
resource-tied floor gap (median tail-to-head plateau-mass ratio grows from ~87x to
~115x), so out-of-model head text flows into tail labels harder than before. The
floor pathology is a between-language externality, not per-language miscalibration.
Floor-21 works on natural traffic precisely because it is NOT per-language-honest:
one shared constant flattens the cross-language gap. gt_min and floor-21 are exact
mirror images: within-language calibration buys recall and pays precision;
cross-language equalization buys precision and pays recall.

**What this sharpens.**
1. The open objective decision now has concrete champions: under the uniform-prior
   view gt_min is the best configuration ever tested here; under the natural-traffic
   view it is disqualified and floor-21 stands. The adoption rule encodes the
   natural-traffic veto, so floor-21 remains the provisionally adopted config.
2. Next-round composition hypothesis (recorded, NOT run, not yet pre-registered):
   combine the two mechanisms explicitly, e.g. cross-language floor equalization at
   the GT-implied level (equalize per-token floor values across languages, with the
   shared level set from the GT counts instead of a swept constant), or gt_min plus
   the head-targeted margin gate (Exp 26) with tau recalibrated under the gt_min
   matrix. Either would be the first candidate family that separates the
   within-language and between-language corrections.

**Artifacts:** `outputs/tables/full_test_gt.md`, updated `two_sided_selection.md`,
`pred_gt_min.npy` + `fingerprint_gt.json` on the full-test scratch dir. Script
`analysis/full_test_gt.py` (reviewed pre-run, no defects).

## Experiment 27: Good-Turing counts, and the floor overstates unseen mass everywhere (2026-07-23, job 2883714)

**Hypothesis:** the emergent floor plateau misstates each language's unseen-token
probability; the Good-Turing plug-in n1/T from the language's own Viterbi counts
gives the calibrated value.

**Result (`outputs/diagnostic/gt_counts.csv`, 1,940 languages, 2.43B tokens
counted; three languages re-verified by hand end-to-end, exact match):** the
emergent plateau OVERSTATES unseen mass for every language without exception.
Exact GT would raise the plateau for 0/1,940 languages, so the pre-registered
one-sided-min rule coincides with exact GT on this model. Tail median: current
plateau mass 8.7e-2 against a GT target of 9.7e-3 (9x); head median 1.0e-3 against
8.4e-5 (12x). The gt_min matrix (built and gate-checked; floor drops -0.67 to
-9.18 nats, mean -3.08) is the per-language-calibrated version of the mechanism
floor-21 applied as one global constant. No language has n1=0; no tuned constant
anywhere (n1, T from own data; 0.2 is the fixed non-special budget).

**Pending:** the full-test scoring pass (`analysis/full_test_gt.py`, in review,
then SLURM) and the adoption verdict via the two-sided report. gt_min must beat
floor-21 (veto tail F1 0.6337) to displace it.

## Experiment 26: margin diagnostic, viable (2026-07-23, job 2883715)

**Hypothesis:** on lines the baseline routes into a tail label, the score gap between
the winning tail language and the runner-up is separable from the gaps on genuine
lines of that language, so a per-language threshold tau_L calibrated on L's OWN
training data (5th percentile of self-won train margins; MIN_CALIB_LINES=200
exclusion) can remove false positives at a bounded recall cost.

**Result: viable.** Aggregate over the 96 tail languages
(`outputs/tables/margin_diagnostic.md`, tau values in
`outputs/diagnostic/tau_per_lang.csv`):
- 17,299 of the 22,522 FP-into-tail lines fall below tau (76.8% catch rate); 5,413 of
  the caught lines (31.3%) have the true label as runner-up and would be recovered
  outright by reassignment.
- Test-side genuine suppression: 474 of 7,084 self-won true-tail lines (6.7%,
  against the 5% train-side bound; the gap is ordinary train-test shift). Only 53
  suppressed lines have another tail language as runner-up, so there is no
  tail-to-tail cascade.
- Per-language margin AUC (genuine train vs FP) is 0.90-0.9998 for the large
  receivers: sbs_Latn 0.9022 (catch 0.588), pnt_Grek 0.9409 (catch 0.738, recovery
  0.741, consistent with the Exp 25 audit finding that the residual is standard
  Greek with ell as runner-up), mrq_Latn 0.9763, pwn_Latn 0.999, arb_Latn 0.9998.
- 26 languages are excluded from gating (under 200 scoreable self-won train lines;
  listed in the report); they keep baseline behavior.

**Candidate pass 1, `margin_q5` (reassign to the runner-up): REJECTED on clause (C),
with the failure mechanism identified.** Build (`analysis/full_test_margin.py`,
login node, top-1 agreement 1.0000, 17,773 reassignments = 17,299 caught FPs + 474
suppressions, exactly matching the diagnostic): passed stage A (val tail drop 0.0281
inside the widened 0.03) and stage B with the largest FP reduction of any config
(veto FPs into tail 22,404 to 6,594; tail global F1 +0.1743), but one language above
the support floor collapses: szy_Latn -0.107 (n=175). Mechanism (verified from the
memmaps): szy_Latn receives 86 new false positives, 82 of them pwn_Latn's caught
lines handed to their runner-up (szy is pwn's Formosan neighbor), with head true
labels (ind/zsm/hbs). Globally 7,582 of the 17,773 reassignments land on languages
with N < 18,000 and 1,465 on tail languages: runner-up reassignment moves FP burden
onto other precision-fragile labels instead of returning it to the head sources that
produced 98.9% of the FPs (Exp 24).

**Pre-registered follow-up (recorded 2026-07-23 before the run), config
`margin_q5_head`:** identical gate, but reassign to the highest-scoring candidate
with N >= 18,000 (the established head threshold) in the top-5; if the top-5 holds
no head candidate, keep the baseline prediction. No new constants. This is the
second and final candidate from the margin family in this round (multiplicity note:
two candidates from this family have been judged against the veto).

**Candidate pass 2, `margin_q5_head`: ELIGIBLE, not selected.** Build: 16,239
reassignments, 6,858 to the true label (against 5,413 under runner-up targeting;
1,534 below-tau lines kept at baseline for lack of a head candidate). All three
stages pass; the szy_Latn collapse is gone (head targeting removes the
small-relative reassignment path). Balanced-val row 0.9799 / 0.8981 / 0.9036 /
0.9406 / 0.9814 (overall/tail/magnets/twins/head); veto row: overall 0.9215, tail
F1 0.5321 (precision 0.4445), FPs into tail 6,560, the lowest of all six configs.
Floor-21 remains selected: it ranks higher on the selection instrument (val overall
0.9800 vs 0.9799) and is also better on the veto (overall 0.9309 vs 0.9215, tail
F1 0.6337 vs 0.5321).

**Reading of the round:** the two eligible mechanisms act at different layers.
Floor-21 (weight-side) removes the unseen-token score advantage at the source and
wins outright; the margin gate (decision-side) reaches the same-script relative
residual that no floor edit can touch (pnt/ell) and achieves the largest FP
reduction, but pays recall for it twice (its own suppression plus the caught lines
it cannot recover). The natural next composition, deliberately NOT run this round
(no pre-registration, and the weight-side component may change when the
Good-Turing candidate lands): weight-side winner plus the margin gate with tau
RECALIBRATED under the composed weight matrix (margins change when weights change,
so Exp 26 tau values do not transfer). Recorded in Open paths.

## Experiment 25: precision-primary adoption rule, first verdicts, and the pnt/ell label audit (2026-07-23)

Implements the adoption rule the user fixed on 2026-07-23 (EXPERIMENTAL_SETUP.md
"Precision-primary adoption rule"; symmetric widening follow-up decision same day) and
applies it to the four finished configurations. Analysis only, login node, no new
scoring; code reviewed pre-run (Opus adversarial pass: no correctness defects, two
flags fixed: `run_bias_refit` now shortlists like the other sweeps, and
`balanced_split.__main__` no longer runs a pipeline that would undo the regenerated
draws).

**Instruments built.**
- Balanced test draw seed 201 (headline): 185,204 lines, all 1,940 languages, tail
  median support 16, 70/96 tail languages with >= 10 examples. Drawn disjoint from the
  working val (draw 101) only: excluding the union of all five val draws would leave
  ~2 of a 66-line tail pool (deviation from the first plan wording, recorded in
  `balanced_split.py` and EXPERIMENTAL_SETUP.md).
- Stability draws 102-105 regenerated to exclude the test draw (they had no consumers;
  each is again 188,061 lines with zero languages at reduced k).
- Veto instrument: pool minus the selection and headline draws, 45,004,014 lines,
  retaining median 17 (minimum 4) true lines per tail language. The first run used
  pool minus all six draws and measured veto tail recall 0.2188: six half-draws
  exhaust small pools, so per-language F1 was recall-broken exactly where the veto
  needs it. Amended same day; a runtime gate now aborts if the veto retains median
  < 10 true tail lines per language. Veto levels are not comparable to the Exp 24
  full-pool numbers (half the tail's true lines are excluded, all false positives
  remain); the rule uses gains and drops only.

**Verdicts (`outputs/tables/two_sided_selection.md`).**
- **floor-21: ELIGIBLE and selected** (highest balanced-val overall among eligible,
  0.9800). This supersedes the Exp 20 "not adopted" verdict, which was conditional on
  the recall-only view (Exp 24). Floor-21 is the provisionally adopted configuration:
  provisional because E3 (Good-Turing) is specified as the principled replacement that
  must beat it, and E2 (margin) targets its residual.
- **freq_prior: ELIGIBLE** (val tail/magnet losses 0.0195/0.0197 inside the widened
  0.03 with veto gains +0.1434/+0.1278; worst per-language drop zsm_Arab -0.085 at 13
  veto lines). Not selected (balanced-val overall 0.9798 < floor-21's 0.9800).
- **learned_bias reg=5.0: REJECTED** by the per-language collapse clause: llb_Latn
  loses 0.113 global F1 on 4,181 veto lines (shu_Arab -0.100, led_Latn -0.079 close
  behind). This is the bias suppression pattern at the individual-language level, now
  formally bounded. The Exp 16 numbers stand as the natural-traffic reference
  measurement; what is rejected is adoption under the precision-primary rule. The
  delta review confirmed the llb_Latn drop on the full pool (0.111, false-positive
  inflation 579 to 1,861) and added a support floor to the collapse clause
  (MIN_COLLAPSE_SUPPORT=10 veto lines; at n=4 a single line flip moves F1 by
  0.11-0.14); verdicts are unchanged under the fixed clause.

**Headline (balanced test draw, within-stratum) for the selected configuration:**
baseline 0.9809 / 0.9086 / 0.9121 / 0.9435 / 0.9817
(overall/tail/magnets/twins/head); floor-21 0.9804 / 0.8924 / 0.8984 / 0.9433 /
0.9817. The tail recall cost of the adopted configuration is on the record here; its
precision gain lives in the veto view (tail global F1 0.3382 -> 0.6337 on the veto
instrument; 0.5618 -> 0.7655 full-pool, Exp 24).

**Label audit (plan B2, `outputs/tables/label_audit_pnt_ell.md`).** 50 of the 2,644
lines labeled ell_Grek that floor-21 predicts as pnt_Grek, deterministic sample
(seed 0): all 50 read as standard Modern Greek, short subtitle-register lines (sample
median ~25 characters), none with Pontic diagnostics (provisional assistant
classification, open to override). 48/50 are also flipped by the baseline. Conclusion:
this residual is model error on short low-evidence lines, not corpus label noise, so
the margin method's recoverable ceiling on the pair is the full 2,644 lines.

**Artifacts:** `outputs/tables/two_sided_selection.md`,
`outputs/tables/label_audit_pnt_ell.md`,
`outputs/diagnostic/balanced_val/val_lines_seed201.npy` (+ regenerated seeds 102-105,
manifest annotations). Code: `analysis/two_sided_report.py`, `analysis/label_audit.py`,
`passes_shortlist`/`passes_two_sided` in `analysis/hierarchical_pool.py`,
`build_test_draw`/`rebuild_stability_draws` in `analysis/balanced_split.py`.

## Experiment 24: within-stratum vs global per-language F1 (metric decomposition, 2026-07-23)

**Question:** do the stratum rows of the full-test tables and global per-language F1
agree about the tail? Analysis of the saved prediction memmaps (Exp 16 job 2784115,
floor-21 job 2791722); no new scoring. Script `analysis/metric_decomposition.py`
(reviewed pre-run, no defects); before reporting it reproduces every recorded
within-stratum value (gate tolerance 6e-5) and the saved per-language F1 exactly.

**Finding 1: the two views disagree about the tail, structurally.** Every stratum row
in `full_test_eval.md` / `full_test_floor21.md` and every guard column restricts truth
and predictions to examples whose true label is in the stratum, so a head-true line
predicted as a tail language is excluded from the tail row. The overall rows are global
per-language F1 and include it. Baseline: tail within-stratum 0.9132; tail global mean
F1 0.5618 over the same 96 languages (mean precision 0.4590, mean recall 0.8741). Tail
labels receive 22,522 false positives against 7,735 true tail examples; 98.9% of them
come from head sources (median source N = 100,000; the head stratum has 43.67M test
lines against the tail's 7,735, so a leak rate near 0.1% from one head language exceeds
a tail language's whole test support; 3,426 lines labeled ell_Grek are predicted as
pnt_Grek, whose true support is 150). With precision fixed at 1.0, tail global mean F1
would be 0.9154:
measured globally, the tail deficit is precision, and the stratum rows cannot show it.
57/96 tail languages have precision below 0.5; 7/96 have recall below 0.5.

**Finding 2: the config ranking on tail inverts between the views.**

| config | tail within-stratum (reported) | tail global F1 | precision | recall | FPs into tail labels |
|---|---|---|---|---|---|
| baseline | 0.9132 | 0.5618 | 0.459 | 0.874 | 22,522 |
| learned bias reg=5.0 | 0.9114 | 0.6003 | 0.502 | 0.871 | 17,496 |
| freq prior gamma=0.5 | 0.8950 | 0.6800 | 0.616 | 0.850 | 12,381 |
| floor-21 | 0.8928 | 0.7655 | 0.763 | 0.842 | 9,103 |

The "not tail-safe" (Exp 16) and "not adopted" (Exp 20) verdicts are conclusions about
the within-stratum view only. Under global per-language F1 each of those configurations
raises tail mean F1 (+0.04 to +0.20) at a tail recall cost of at most 3.3pp, and the
ordering is exactly reversed: floor-21, the configuration with the largest
within-stratum tail drop, raises tail global F1 most and raises flat_magnet mean F1
from 0.4716 to 0.6402.

**Finding 3: the mechanisms are complementary.** Decomposing each overall +delta into
per-category contributions: floor-21's +0.0129 comes mostly from flat_magnets (+0.0103
of it), with head and twins flat; the learned bias reaches the same +0.0129 from head
(+0.0031), mid (+0.0039), and twins (+0.0009), and is the only configuration that
raises twin global F1 (0.8887 to 0.9103). A composition test is proposed (Open paths
E4).

**Finding 4: the selection protocol cannot register this failure mode, twice over.**
(a) The guard's stratum columns are within-stratum, so cross-stratum false positives
are excluded by construction; (b) the balanced val caps every language at K=100 lines,
which removes the volume asymmetry that produces the false positives in the first
place. Selection under the current guard is therefore systematically directed against
tail-precision configurations. The two views answer different questions (is a genuine
tail line recognized, versus is an emitted tail label correct); which one the paper's
tail claims use is now part of the objective decision in Open paths.

**Finding 5: residual structure under floor-21.** 9,103 residual false positives into
tail labels, 100% same-script, concentrated in directed pairs of close relatives:
pnt_Grek from ell_Grek (2,644), sbs_Latn from loz/bem/kng/toi/kqn (about 1,040
combined), mrq_Latn from tah/rar (480), tat_Latn from tur (209), min_Arab from fas
(199), rme_Latn from eng (193), mns_Cyrl from rus (158). Floor manipulation does not
separate close relatives; the proposed follow-ups for that residue are a calibrated
decision margin (E2) and a label audit of the pnt/ell pair (E6).

**Artifacts:** `outputs/tables/metric_decomposition.md`;
`outputs/diagnostic/full_test_per_lang_prf.csv` (per-language precision/recall/F1/FP
for all four configurations, including floor-21, which `full_test_per_lang_f1.csv`
lacks). Script: `analysis/metric_decomposition.py`. **Status:** analysis of record;
proposed follow-ups in `EXPERIMENTS_PLAN.md` Open paths block E; the metric-view
question added to the Decision required item there.

## Experiment 23 — First sweeps under the balanced protocol (2026-07-19)

Three experiments, selection-only on the seed-101 balanced val (job 2794210; baseline
validated against the saved full-test predictions, agreement 1.0000 expected path).

**23a. Floor equalization: rejected at selection time.** Every F drops val tail
(-0.0177 at F=-17 down to -0.0269 at F=-23) and magnets similarly; nothing passes. The
tail-sighted guard reaches in nine minutes the verdict that previously required a
five-hour full-test pass (Exp 20). Plan item 14 closes as a selection outcome.

**23b. Punctuation partial pooling (plan item 15): alpha=300 PASSES the guard.** All
strata non-negative (overall +0.0001, tail +0.0004, magnets +0.0003, twins +0.0001,
head +0.0000); stronger alphas turn twins negative (-0.0031 at alpha=30000), consistent
with the tying result that twin conventions are signal. The effect at alpha=300 is at
the edge of measurability; a full-test pass and a balanced-test evaluation are required
before calling it real. Direction: the only configuration in the program to date with
no negative stratum at selection.

**23c. Learned-bias refit on balanced data (plan item 16): reg=0.3 passes with a
substantial selection-half gain,** overall 0.9818 -> 0.9834 (+0.0016), tail +0.0299,
magnets +0.0252, head +0.0001, twins -0.0016 (within tolerance). Answer to the design
question: attractor suppression survives the uniform-prior objective; the Exp 14 gain
was not purely traffic-prior fitting. The fitted vector is aggressive
(||b||_inf = 11.3) and its most-suppressed languages are NOT the flat magnets but
head/twin sinks (nya_Latn -11.3, por_Latn -8.9, heb_Hebr -7.2, swc_Latn -5.9),
matching the diagnostic finding that 40% of false-positive mass sits on head-level
sinks: under a uniform prior the optimum suppresses dominant cluster members to free
their many satellites. CAUTIONS before any adoption: (1) single draw; refit-per-draw
stability (draws 102-105) is required by protocol caveat 3; (2) the guard's stratum
tolerances do not bound INDIVIDUAL-language harm, and b = -8.9 on Portuguese trades
that language's marginal recall for its satellites, an objective-level question to
decide explicitly; (3) selection-half optimism (fit and selection halves share the
draw's candidate structure); (4) a full-test pass and a balanced-test draw (disjoint
from val) are needed for final numbers.

**Artifacts:** `outputs/tables/balanced_{floor_eq,punct_prior,bias_refit}.md`,
`learned_bias_balanced.npy`; `analysis/balanced_sweeps.py`.

## Experiment 22 — Balanced validation protocol and re-baseline (2026-07-19)

**What:** the split redesign (plan item 10). Language-balanced val drawn from the kept
full-test pool (K=100 per language, fraction cap 0.5; 188,061 lines; all 1,940
languages represented, tail median support 33 vs 0 under the old protocol; five seeds
for split-variance checks; the original 250k val is retired). The four saved full-test
prediction sets were re-scored under the new protocol with no new model scoring.

**Selection view (balanced val, guard verdicts):** frequency prior FAIL (tail -0.0195),
floor-21 FAIL (tail -0.0228, magnets -0.0196): the balanced val catches, at selection
time, both failures that previously required full-test passes to discover. Learned
bias FAIL for a different and structural reason: balanced-val overall drops -0.0012
below baseline. A fitted per-language bias approximates the log prior of natural
traffic; on a language-balanced sample the optimal prior is uniform, so any nonzero
bias loses by construction. **Under the balanced (uniform-prior) objective, the
unmodified baseline is the best configuration tested to date.**

**Final view (pool minus val):** natural-traffic numbers persist (baseline 0.9210,
freq prior 0.9343, learned bias 0.9342, floor-21 0.9372 overall; tail 0.9069 / 0.8908 /
0.9062 / 0.8883). The two views answer different deployment questions (natural traffic
vs every-language-equal); the protocol makes the choice explicit instead of conflating
them, and selection uses the balanced view.

**Artifacts:** `outputs/tables/balanced_split_rebaseline.md`,
`outputs/diagnostic/balanced_val/`; `analysis/balanced_split.py`.

## Experiment 21 — Macrolanguage-hierarchical decision (NULL, 2026-07-18)

**Hypothesis:** treating the variety within a macrolanguage as latent (group score =
logsumexp over members from the top-50 candidates, argmax group, best member within it)
recovers errors inside the 83 multi-member SIL macrolanguage groups (289 languages).
Parameter-free; job 2791444.

**Result: exact null.** The marginal essentially never flips a group decision (test
deltas -0.0000 on every stratum; exact accuracy unchanged at 0.9603). Score gaps
between candidates are large in nats, so the group marginal is dominated by its top
member and argmax-of-groups equals group-of-argmax. The useful output is the ceiling
measurement: macro-aware accuracy on the test half is 0.9680 against exact 0.9603, so
0.77pp of accuracy (and the ~20% Exp 10 error share) is within-macro confusion that no
decision rule can recover; it is an evaluation-convention question, not a modeling one.

**Artifacts:** `outputs/tables/macro_hierarchy.md`; `analysis/macro_hierarchy.py`.

## Experiment 20 — Downward floor equalization (POSITIVE on overall; tail pending
full-test, 2026-07-18)

**Hypothesis:** the resource-tied floor means small languages under-penalize unseen
tokens (Exp 10); every mass-ADDING fix failed (Exp 9/13/18/19), so equalize DOWNWARD:
clamp each language's exact floor plateau to `min(floor_L, F)`, one global constant,
nothing raised, observed tokens and specials bit-identical. F swept over
{-17, -19, -21, -23} (n_modified 452 / 1,821 / 1,940 / 1,940). Job 2791444.

**Result:** val overall rises at every F (+0.0024 to +0.0038, peak at F=-21) with
twins/head/magnets flat on val; the guard selects **floor-21**. Test half:

| stratum | base | equalized | delta | 95% CI |
|---|---|---|---|---|
| overall | 0.9454 | 0.9494 | +0.0030 | [+0.0016, +0.0044] |
| twins | 0.9224 | 0.9228 | +0.0003 | [-0.0004, +0.0014] |
| head | 0.9603 | 0.9600 | -0.0003 | [-0.0009, +0.0003] |
| magnets | 0.8832 | 0.8630 | -0.0108 | [-0.0429, +0.0295] |
| tail | 0.9310 | 0.8621 | -0.0623 | [-0.1111, +0.0000] |

This is the first likelihood-side modification to beat baseline with a CI excluding
zero, it is fully modular (one shared constant, no fitting), and the mechanism is the
subtractive direction the four negatives pointed to. Adoption was blocked on the tail
question pending a full-test pass.

**Full-test verdict (job 2791722, 2026-07-19): the tail cost is real; NOT adopted.**
One scoring pass under the floor-21 matrix against the saved Exp 16 baseline
(45,377,279 lines):

| stratum | base | floor-21 | delta | 95% CI |
|---|---|---|---|---|
| overall | 0.9292 | 0.9421 | +0.0129 | point only (n > cap) |
| head | 0.9602 | 0.9599 | -0.0003 | point only |
| twins | 0.9167 | 0.9166 | -0.0001 | point only |
| tail | 0.9132 | 0.8928 | -0.0204 | [-0.0257, -0.0161] |
| magnets | 0.9138 | 0.8974 | -0.0164 | [-0.0210, -0.0129] |

Mid stratum (1k <= N < 18k, 984 languages): +0.0001. Accuracy +0.0009. Decomposition:
the stratum tables are computed on stratum-masked example subsets, so the overall
+0.0129 is a global-precision effect (languages stop receiving stolen cross-stratum
false positives) while the tail -0.0204 is a recall-side harm (examples truly written
in tail languages are misclassified more; lowering a tail language's floor penalizes
its own out-of-corpus tokens). Under the fairness objective the tail regression is
disqualifying: at equal overall gain (+0.0129 both), the learned bias costs the tail
-0.0018 (Exp 16) versus floor-21's -0.0204, so floor equalization is dominated and is
recorded as an overall-for-tail trade alongside the frequency prior, not adopted.
[Metric-conditional, added 2026-07-23: the -0.0204 and the domination claim hold on
the within-stratum (recall) view only. Under global per-language F1, floor-21 is the
strongest tested configuration for tail (0.5618 to 0.7655) and flat_magnet (0.4716 to
0.6402) mean F1, and the learned bias acts on different strata (head/mid/twins); see
Exp 24.]

Third structural lesson about selection: the val guard passed floor-21 because val is
blind on tail AND magnets; the full test refuted both strata. After the Apertus
gamma=3.0 flaw (Exp 15) and the freq-prior tail artifact (Exp 16), this is the third
val-selected operating point overturned at full scale. No further sweep selections
should be trusted on tail/magnet claims until the split redesign (plan item 10) adds
tail-sensitive validation.

**Artifacts:** `outputs/tables/floor_equalization.md`, `full_test_floor21.md`;
`analysis/floor_equalization.py`, `full_test_floor21.py`.

## Experiment 19 — Group-mean back-off at floor positions (NEGATIVE, 2026-07-18)

**Hypothesis:** the resource-tied unseen-token floor (an exact per-language plateau over
74,617-99,810 of 100k entries; Exp 10: corr(floor, log10 N) = -0.966) is the tail's
under-fitting mechanism, so replacing the flat floor with a group-informed profile
(`lam_L * m_G(t)`, `lam_L = alpha/(N_L+alpha)`, observed tokens bit-identical, no
renormalization) should improve discrimination. Two groupings: script backbone means
(job 2790155) and, at the user's request, WALS genealogical tiers
(genus-within-script 535 languages, family-within-script 360, script fallback 1,012;
source `data/wals_languages.csv`; job 2790174). Modes lift/full x alpha
{300, 3000, 30000}.

**Result: negative at every strength under both groupings, and the grouping barely
matters.** Val overall falls monotonically with alpha (script: -0.0028 to -0.0289; WALS:
-0.0036 to -0.0304); at alpha >= 3000 val tail falls 0.8710 -> 0.8387 and magnets
0.8797 -> 0.8609 under both. Full mode tracks lift mode within 0.0007 everywhere. No
config passes the guard; baseline selected in both runs.

**Mechanism reading:** lifting a language's unseen-token mass toward any group mean makes
it MORE accepting of group-plausible foreign material, increasing exactly the theft that
Exp 10 diagnosed (small languages already under-penalize unseen tokens). Together with
Exp 9 (transfer), Exp 13 (shrinkage/sharpening), and Exp 18 (tying), every intervention
that moves probability mass toward group typicality has now failed in the same
direction. Consequence for the family-initialization idea: this was its post-hoc
surrogate (initialization persists only where data is absent, i.e. at these floor
positions), so a family-initialized or family-MAP retrain is not supported by current
evidence. The direction Exp 10 actually implies remains untried: equalizing the
per-token unseen penalty DOWNWARD (lowering low-resource floors toward high-resource
levels), the opposite of every mass-adding fix tested so far. Note Exp 6 clamped floors
UPWARD (also negative), so downward equalization is genuinely untested.

**Artifacts:** `outputs/tables/family_backoff.md`, `family_backoff_wals.md`,
`outputs/diagnostic/backoff_groups_{script,wals}.csv`; `analysis/family_backoff.py`;
`data/wals_languages.csv` (provenance `data/README.md`).

## Experiment 18 — Non-content token tying (NEGATIVE, 2026-07-18)

**Hypothesis:** tokens with no language identity (digits, whitespace, punctuation)
contribute only estimation noise to score differences (Exp 10: 86.4% of the stolen margin
is short non-content tokens), so tying their probabilities to one shared value across all
languages should help. Pure tying, no renormalization (renormalizing would inject a
per-language per-token offset up to 0.36 nats/token; derived in the pre-run review).
Three tied sets classified on byte-decoded token text: digits_ws (298 tokens),
nonalpha_ascii (479), nonalpha_all (1,291). Job 2790078.

**Result: negative at every scope.** Val overall macro-F1 drops monotonically with tied-set
size: digits_ws -0.0010, nonalpha_ascii -0.0063, nonalpha_all -0.0078; nonalpha_all also
drops val tail -0.0108. No config passes the guard; baseline selected.

**Reading:** cross-language variation in non-content token probabilities is not pure noise;
it carries usable signal. The most likely single cause is whitespace: the tied sets include
the space and newline tokens, and whitespace frequency separates spaced from unspaced
scripts. A refinement (tie digits+punctuation but not whitespace) was not run.

**Curated re-run (2026-07-19, job 2793541): still negative; the hypothesis is refuted at
every curation level.** After the user's critique of the original design (whitespace should
never have been tied), the tied set was rebuilt: 212 tokens of ASCII digits plus neutral
punctuation only, with documented linguistic exclusions (apostrophes, hyphens/dashes,
ampersand, currency, Spanish inverted marks, typographic quotes, all whitespace including
leading-space variants, all non-ASCII punctuation), tied within script groups (primary) and
globally (comparison). Val outcome: dp_global overall -0.0014 with twins -0.0060; dp_script
overall -0.0016 with twins -0.0103 (fails the twin guard on its own); tail and magnets
flat under both. Baseline selected; all test deltas zero by construction.

**Final reading:** the cost concentrates in the twins stratum, so within-pair differences in
digit and punctuation usage rates are genuine discriminative signal for near-twin pairs,
consistent with Exp 4 (punctuation is 10.5% of within-pair KL; that KL is signal, not
estimation noise). The Exp 10 stolen-margin finding needs the sharper interpretation this
gives: short non-content tokens dominate margins because they are frequent and their
per-language rates are informative; the pathological part is only their UNSEEN (floor)
positions, and both floor directions have now been tested directly (raising toward group
means, Exp 19: worse; lowering to a common value, Exp 20: tail-harmful). Full tying
(weight 1 on the shared value) is refuted at every curation level; the constructive
reading is that these experiments LOCALIZED where punctuation/digit statistics are
signal (well-estimated head/twin rates) versus noise (low-N estimates and floor
positions), which motivates partial, N-indexed treatment of exactly these columns
(punctuation back-off / hierarchical prior, plan item 15) rather than the
all-or-nothing tie tested here.

**Artifacts:** `outputs/tables/token_tying.md`, `token_tying_dp.md`;
`analysis/token_tying.py`.

## Experiment 17 — Prior-centered learned bias with corrected gradient (2026-07-18)

**Setup:** the learned-bias penalty generalized to `reg*||b - gamma*log(N+1)||^2`
(gamma in {0, 0.25, 0.5} x reg grid; gamma=0 is plain L2), fit with the corrected NLL
gradient (see the Exp 14 estimator caution). Job 2790077.

**Result:** the guard selects gamma=0.25, reg=10. Test half: overall +0.0117
[CI +0.0104, +0.0130], twins +0.0124, head +0.0089, magnets -0.0052 (crosses 0), tail
-0.0320 (the 250k-half tail, which Exp 16 showed is noise-dominated; a full-test read is
required before interpreting it). Under the corrected gradient the previous operating
point (gamma=0, reg=5) fails the guard (val magnets -0.0119), so the Exp 14 revised
selection does not survive the estimator fix as-is. The centered gamma=0.25 point is
marginally above the old plain-L2 reg=5 on the same half (+0.0117 vs +0.0112,
overlapping CIs). Not adopted as a result of record pending a full-test evaluation;
method status also depends on the modularity concern recorded 2026-07-18 (the
discriminative fit couples all languages through the softmax, so adding a language
requires refitting on global data, unlike the likelihood-side methods).

**Artifacts:** `outputs/tables/learned_prior_centered.md`, `learned_bias_centered.npy`.

## Experiment 16 — Full-test-set evaluation of the fixed configurations (2026-07-18)

**Question:** do the guard-selected results (Exp 14) hold at full test-set scale, where the
tail is measurable? On the 250k test half every one of the 96 tail languages has <= 2
examples, so the Exp 14 tail deltas (freq prior 0.0000; learned bias -0.0320 with CI
touching 0) rested on ~35 items. Job 2784115 (05:06:50) scored the 100k model on all
45,377,279 non-val test lines for three configurations FIXED on val: baseline, frequency
prior gamma=0.5, learned bias reg=5.0. No selection; pure evaluation. Zero-bias
predictions validated against the recorded UniLID predictions (agreement 0.9951, matching
the known baseline self-agreement). All 1,940 languages have test support (tail stratum
7,735 examples, magnets 64,657).

| stratum | base | freq delta | freq 95% CI | learned delta | learned 95% CI |
|---|---|---|---|---|---|
| overall | 0.9292 | +0.0116 | point only | +0.0129 | point only |
| head | 0.9602 | +0.0011 | point only | +0.0101 | point only |
| twins | 0.9167 | +0.0011 | point only | +0.0116 | point only |
| tail | 0.9132 | -0.0182 | [-0.0225, -0.0146] | -0.0018 | [-0.0035, -0.0001] |
| magnets | 0.9138 | -0.0173 | [-0.0207, -0.0141] | -0.0082 | [-0.0099, -0.0066] |

Overall accuracy: baseline 0.9608, freq prior 0.9638, learned bias 0.9751. CIs (B=1000
item bootstrap) are computed for strata under 3M examples; for the others the item-level
CI half-width is below 0.001.

**Conclusions.**
1. The learned bias result is confirmed and its tail scare is resolved: the true tail
   cost is -0.0018 [CI -0.0035, -0.0001], small though nonzero; the -0.0320 point
   estimate on the 250k half was split noise. Magnets cost is real but modest (-0.0082).
   Overall +0.0129, head +0.0101, twins +0.0116, accuracy +0.0143. These are the numbers
   of record for the learned bias.
2. The frequency prior is NOT tail-safe: tail -0.0182 [CI -0.0225, -0.0146]. Its Exp 14
   "tail exactly 0.0000" was an artifact of the test half containing almost no tail
   examples. The Exp 14 claim that the frequency prior is the safer minimal version is
   withdrawn: on the full test set the learned bias has BOTH the larger gain and the
   10x smaller tail cost. [Metric-conditional, added 2026-07-23: these tail deltas are
   within-stratum (recall-view) numbers. Under global per-language F1 the frequency
   prior raises tail mean F1 from 0.5618 to 0.6800 by reducing false positives into
   tail labels (22,522 to 12,381); see Exp 24.]
3. Macro-F1 LEVELS are not comparable between the 250k half and the full set (baseline
   0.9454 vs 0.9292): languages absent from the half's true-label set contributed no term
   there, and the full set adds every hard rare language. Deltas are the comparable
   quantity.

**Artifacts:** `outputs/tables/full_test_eval.md`;
`outputs/diagnostic/full_test_per_lang_f1.csv` (per-language F1 under all three configs,
input for plan items 5-6); memmaps + config fingerprint in
`/capstor/scratch/.../unilid_analysis/full_test_eval/`. Script:
`analysis/full_test_eval.py` (reviewed pre-launch; resumable; fail-loud alignment and
scorer checks).

## Experiment 15 — Apertus 200k retrain + frequency prior on it (MIXED / cautionary)

Tests the orthogonal vocab-coverage hypothesis: does a larger, better-covered vocabulary
(Apertus V2 200k byte-level, seeded into the Unigram and re-estimated per language) improve
macro-F1 on its own? Standard-setup retrain (NOT the refuted MAP-EM prior), all 1,940
languages, SP per-language re-estimation on the recovered GlotLID corpus. Job 2639097 (timed
out at 12 h, 1,690/1,940) + 2641940 (resumed, COMPLETED, all 1,940). Model
`glotlid_apertus200k.unilid` (1.56 GB). Evaluated on the same 500k sample / val-test split /
strata as the 100k model, so baselines are directly comparable.

**Retrain baseline vs the 100k model (test half, gamma=0 in each):**

| stratum | 100k base | Apertus base | delta |
|---|---|---|---|
| overall macroF1 | 0.9454 | 0.9447 | -0.0007 |
| tail | 0.9310 | 0.8966 | **-0.0344** |
| magnets | 0.8832 | 0.8999 | +0.0167 |
| twins | 0.9224 | 0.9219 | -0.0005 |
| head | 0.9603 | 0.9608 | +0.0005 |
| accuracy | 0.9603 | 0.9644 | +0.0041 |

The 200k vocab RAISES overall accuracy (+0.41pp, head-driven) but LOWERS tail macro-F1
(-3.4pp). For a macro-F1 / fairness goal the retrain is a net negative on the tail: bigger
vocab helps common languages and hurts rare-tail discrimination. The vocab-coverage
hypothesis is not supported for the tail.

**Frequency prior on the Apertus model: reveals a guard flaw.** The val guard (no
twin/head regression) selected gamma=3.0: test overall 0.9447 -> 0.9673 (+0.0203
[CI +0.0179,+0.0228]) BUT tail -0.0945 [CI -0.1667,0] and magnets -0.1102 [-0.1598,-0.0570].
The +0.0203 "overall" is an artifact of macro-averaging over 1,940 languages: the many head
languages (+0.0061) outnumber the collapsing tail. On the Apertus model even the mildest
gamma=0.25 already drops val_tail (0.8387 -> 0.8065), so NO frequency-prior strength helps
the Apertus tail. **The guard is insufficient: it protected only twins/head, not tail/magnets,
and so selected a tail-destroying operating point.**

**Resolution (2026-07-10, job 2731803).** Under the fixed all-strata guard (`GUARD_TOL =
0.01`, `EXPERIMENTAL_SETUP.md` "Selection guard") no gamma is eligible on the Apertus model
(every gamma >= 0.25 drops val tail by >= 0.032); the baseline is selected and the frequency
prior is formally rejected on this model. The gamma=3.0 selection above is superseded and
kept as the record of the guard flaw.

**Conclusions.** (1) The frequency prior `gamma*log N_L` is a blunt frequency reweighting
that trades tail for head; it looked tail-safe on the 100k model only because gamma=0.5 was
mild there. (2) The learned per-language bias (Exp 14) is the precise instrument: on the 100k
model it improved overall +0.0180 with the tail FLAT, because a free per-language offset can
down-weight specific attractors without penalizing every rare language. (3) The Apertus
retrain is a mostly-negative branch for the tail. Next: fix the selection guard to protect
all strata, and run the LEARNED bias (not the frequency prior) on the Apertus model.

**Artifacts:** `outputs/tables/prior_sweep_apertus.md`; model `glotlid_apertus200k.unilid`;
`slurm_apertus_train.sh`, `slurm_prior_apertus.sh`.

## Experiment 14 — Per-language frequency prior (POSITIVE, first real improvement)

**Redirect after Exp 13.** The Stage 1 failure (shrinkage removes a magnet's recall together
with its false positives) pointed to a per-language PRIOR instead: a constant offset `b_L`
added to each language's summed score before the argmax (Rust `best_of_cached_weight_sets_biased_batch`,
added + validated). A constant matters most on SHORT text (few tokens, small |score|) where
the rare-attractor problem is worst, and it is selective: a magnet still wins on its own text
(large likelihood margin) but loses the marginal cases it was stealing. Prior family swept:
`b_L = gamma * log(N_L + 1)` (frequency prior P(L) ~ N_L^gamma). Tuned on val, scored once on
test (job 2639127).

**Result: significant improvement.** Val overall macro-F1 rises monotonically with gamma
(0.9451 -> 0.9639 at gamma=5), but twins/tail/magnets regress at high gamma (over-favouring
common languages). The val guard (no twin/head regression) selects **gamma=0.5**, where BOTH
twins and head already improve. On TEST at gamma=0.5:

| stratum | base | prior | delta | 95% CI |
|---|---|---|---|---|
| overall | 0.9454 | 0.9524 | +0.0058 | [+0.0048, +0.0069] |
| head | 0.9603 | 0.9615 | +0.0012 | [+0.0008, +0.0017] |
| twins | 0.9224 | 0.9243 | +0.0019 | [+0.0004, +0.0042] |
| tail | 0.9310 | 0.9310 | +0.0000 | [0, 0] |
| magnets | 0.8832 | 0.8811 | -0.0026 | [-0.0130, +0.0035] |

Overall accuracy 0.9603 -> 0.9634 (+0.0032). The macro-F1, head, and twins CIs all exclude 0;
the tail is untouched and magnets are flat (NOT destroyed as shrinkage destroyed them). This
is the selective effect predicted: the constant offset removes marginal false wins (so the
victim head/twin languages recover) without blunting any language's own-text margin. Modular
(one scalar per language from its train count), no retraining. This is the first modification
in the project to beat baseline with a CI that excludes zero. Re-run 2026-07-10 under the
revised all-strata guard (job 2731804): gamma=0.5 is again selected (val magnets -0.0081 is
within the 0.01 tolerance) and the test numbers are unchanged; the artifact header now
records the guard rule. **Full-test-set update (2026-07-18, Exp 16):** at full scale the
frequency prior costs tail -0.0182 [CI -0.0225, -0.0146]; the "tail +0.0000" in the table
above is an artifact of the 250k half's tail invisibility (every tail language has <= 2
examples there). The tail-safety claim for this prior is withdrawn.

**Learned per-language bias (guard-revised result of record, 2026-07-10).** Generalizing the
1-param frequency prior to a free `b_L` per language, fit on val by L2-regularized softmax over
each example's top-k candidate scores (top-20 recall of the true label 0.9971; Rust
`top_k_of_cached_weight_sets_batch` added; gradient verified to 2e-10). The original run (job
2640065) selected reg=0.3 under the twins/head-only guard and reported overall +0.0180; that
selection is superseded (see Invalidated / superseded results) because reg=0.3 costs val
magnets -0.0318, which the revised all-strata guard disallows. Under the fixed guard
(`GUARD_TOL = 0.01`, see `EXPERIMENTAL_SETUP.md` "Selection guard"; REGS extended with 5 and
7), the re-run (job 2731802) selects **reg=5.0** (val magnets -0.0075). On GlotLID test:

| stratum | base | learned | delta | 95% CI |
|---|---|---|---|---|
| overall | 0.9454 | 0.9567 | +0.0112 | [+0.0099, +0.0124] |
| head | 0.9603 | 0.9696 | +0.0094 | [+0.0086, +0.0101] |
| twins | 0.9224 | 0.9358 | +0.0135 | [+0.0101, +0.0170] |
| tail | 0.9310 | 0.8966 | -0.0320 | [-0.0588, +0.0000] |
| magnets | 0.8832 | 0.8862 | +0.0051 | [-0.0302, +0.0463] |

Overall accuracy 0.9603 -> 0.9749 (+0.0147); overall macro-F1 +0.0112, about 2x the frequency
prior; overall/head/twins CIs exclude zero; magnets crosses zero. This is the project's best
guard-compliant result.

**Estimator caution (found in review, 2026-07-18).** The fit's NLL gradient accumulated
softmax soft counts over ALL examples' top-k candidates while the loss conditions on the
true label being inside the top-k (recall 0.9971), so the fitted b was not exactly the
minimizer of the stated objective (finite-difference verified; the perturbation
concentrates on the confuser languages that populate absent examples' top-k lists). The
test deltas above are valid measurements of the b that was produced; only the estimator
description was wrong. The gradient is fixed in `analysis/learned_prior.py` and the
prior-centered re-run (job 2790077) re-fits the plain-L2 bias with the corrected
gradient as its gamma=0 rows.

**Tail caution and a guard blind spot.** The test tail delta is -0.0320 with CI
[-0.0588, +0.0000] (upper bound exactly 0): not significantly negative at the 95% level, but
the point estimate is large. The val guard could not have seen this: val tail macro-F1 is
0.8710 for every reg (and for every gamma <= 1.5 in the frequency-prior sweep), i.e. the val
half contains too few decision-sensitive tail examples for the guard to register tail
movement at all. Addressing this needs a split-design change (plan item 10: resampled
val/test splits, possibly a tail-weighted val allocation), not a tolerance change.
**Resolved (2026-07-18, Exp 16):** on the full test set (7,735 tail examples, all 96
languages) the learned bias's tail delta is -0.0018 [CI -0.0035, -0.0001]; the -0.0320 was
split noise. The guard blind spot itself remains a val-design problem for future sweeps.

**Out-of-domain validation (CommonLID web text, Exp 12 pipeline + priors).** With the guarded
reg=5.0 bias (job 2731818): baseline macro-aware accuracy 0.8452 -> frequency prior
(gamma=0.5) 0.8518 (+0.0067) -> learned bias 0.8879 (+0.0427). The gain holds out of domain;
the superseded reg=0.3 vector gave 0.8936, so the milder guarded vector keeps most of it.
CommonLID's 109 labels are all common languages, so suppressing the rare attractors there is
nearly pure gain.

**Caveats:** the bias down-weights rare languages, so a deployment whose inputs are genuinely
rare-language-heavy could see tail regression (full-test tail delta -0.0018
[CI -0.0035, -0.0001], Exp 16). The earlier framing of the frequency prior as the safer
minimal version is withdrawn: on the full test set (Exp 16) the frequency prior costs tail
-0.0182 while the learned bias costs -0.0018, so the learned bias has both the larger gain
and the smaller tail cost. Pending: learned bias on the Apertus-200k model (plan item 2);
the prior-centered regularizer (plan item 3) may reduce the residual tail/magnet cost.

**Artifacts:** `outputs/tables/prior_sweep.md`, `learned_prior.md`, `commonlid_eval.md`,
`learned_bias.npy`; `analysis/{prior_sweep,learned_prior,commonlid_eval}.py`; Rust
`best_of_cached_weight_sets_biased_batch` + `top_k_of_cached_weight_sets_batch`.

## Experiment 13 — Stage 1 post-hoc gated shrinkage + sweep (NEGATIVE)

**Hypothesis:** shrinking the diagnosed flat_magnet / tight_lowres / isolated_tail rows
toward a confuser-excluded backbone script mean (gated by category) raises stratified
macro-F1 without regressing twins/head. Run on the 500k sample, tuned on the val half,
scored once on the test half (job 2638804, 16 min). Baseline self-agreement 0.9951.

**Result: shrinkage REDUCES macro-F1.** Val overall fell monotonically with shrinkage
strength (0.9451 baseline -> 0.9429 at mag=0.3 -> 0.9378 at mag=0.9). On test the mildest
config gave overall -0.0013 [CI -0.0025, +0.0001], magnets -0.0298, tail -0.0622, twins
+0.0000, head +0.0003. Every stronger config was worse.

**Why (mechanism):** the flat magnets are recalled well on their own (rare) true examples
(magnet-stratum baseline macro-F1 0.88), so shrinking their distributions toward the mean
destroys that recall. The hoped-for victim recovery did not materialise: the head stratum
barely moved (+0.0003), because (a) magnet false-positives are spread thinly across many
high-support victims and (b) removing a magnet as the argmax winner sends the example to the
2nd-place language, which is often still wrong (another magnet or a sibling), not the true
label. So shrinkage trades away magnet/tail recall for a victim-recovery that does not occur.

**Consequence for Stage 2:** the MAP-EM posterior mean `(N c + alpha m)/(N+alpha)` equals
`(1-lambda) p + lambda m` at the EM fixed point (lambda = alpha/(N+alpha)), i.e. the same
operation Stage 1 applied. The re-segmentation difference is second-order. So a full Apertus
MAP-EM retrain with this script-mean prior would very likely reproduce this negative result;
it is not worth the 1,940-language retrain. The diagnosis (magnets steal) is correct; the
*shrinkage-toward-mean* fix is refuted. Redirect candidates: a per-language prior /
calibration offset (down-weights rare attractors without blunting their own-text margin),
or entropy-sharpening the magnet rows (addresses flatness without pulling toward a foreign
mean). Both are cheap post-hoc tests.

**Artifacts:** `outputs/tables/hierarchical_pool.md`, `analysis/hierarchical_pool.py`,
`outputs/diagnostic/lang_diagnostic.csv`.

## Experiment 12 — CommonLID external validation (trends partially hold)

**Question:** do the GlotLID-test error trends hold on out-of-domain web text? UniLID run on
CommonLID (Common Crawl, 373,230 lines, 109 bare-ISO-639-3 tags, macro-aware scoring via the
SIL macrolanguage table; job 2638803, 2.5 min).

**Findings:** macro-aware accuracy 0.8452 (vs 0.9615 on GlotLID test; web text is harder, as
the CommonLID paper intends). The attractor trends transfer: diagnosed flat_magnets account
for 27.7% of error predictions (uzb->tly/vol/ido, eng->sco/nov, msa->abs/bew are all
magnet thefts), confirming the magnet phenomenon is real and domain-general. Resource
asymmetry is weaker but present: predicted language is rarer than the truth in 61.6% of
errors (vs ~86% on GlotLID test). Top confusions are the same closely-related pairs
(arb->ars/ary/acm, ind->zsm, eng->sco, fas->mzn/glk).

**Artifacts:** `outputs/tables/commonlid_eval.md`, `analysis/commonlid_eval.py`,
`/capstor/scratch/.../commonlid/unilid_preds.npz`.

## Invalidated / superseded results

- **Exp 14 learned bias at reg=0.3 (overall macro-F1 +0.0180; accuracy +0.0169; CommonLID
  0.8936), jobs 2640065 / 2640066.** Superseded 2026-07-10. The reg=0.3 operating point was
  selected under the twins/head-only guard and costs val magnets -0.0318, which the revised
  all-strata guard (tolerance 0.01) disallows; the measurements themselves are valid, but the
  configuration is no longer the selected one. Result of record: reg=5.0 (job 2731802),
  overall +0.0112, CommonLID 0.8879 (job 2731818). See Exp 14.
- **Exp 15 Apertus frequency prior at gamma=3.0 (overall +0.0203), job 2649123.** Superseded
  2026-07-10. Selected by the flawed guard while collapsing tail (-0.0945) and magnets
  (-0.1102). Under the fixed guard no gamma is eligible on the Apertus model (job 2731803);
  the frequency prior is rejected there. The sweep table remains valid as a sweep record.

No results were invalidated during the 2026-05-27 reconstruction.

**Uncommitted-results caution.** The committed `EXPERIMENTS.md` (commit `b7508fd`, the only
commit) contains only Experiments 1–6. The working-tree copy adds Experiments 7, 8a, and 9
(+144 lines, purely additive — no committed content was changed). The summaries above use
the working-tree copy, which is the more complete version. This means the Exp 7 / 8a / 9
results currently exist **only** in the uncommitted working tree plus their
`outputs/tables/` files (`train_data_analysis.md`, `discriminative_heuristic.md`,
`transfer_sweep.md`). They are not in git history and would be lost if the working tree were
reset. Committing them would secure the provenance chain. The table files are the artifacts
of record for those numbers.
