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

## Headline observations

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
records the guard rule.

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

**Tail caution and a guard blind spot.** The test tail delta is -0.0320 with CI
[-0.0588, +0.0000] (upper bound exactly 0): not significantly negative at the 95% level, but
the point estimate is large. The val guard could not have seen this: val tail macro-F1 is
0.8710 for every reg (and for every gamma <= 1.5 in the frequency-prior sweep), i.e. the val
half contains too few decision-sensitive tail examples for the guard to register tail
movement at all. Addressing this needs a split-design change (plan item 10: resampled
val/test splits, possibly a tail-weighted val allocation), not a tolerance change.

**Out-of-domain validation (CommonLID web text, Exp 12 pipeline + priors).** With the guarded
reg=5.0 bias (job 2731818): baseline macro-aware accuracy 0.8452 -> frequency prior
(gamma=0.5) 0.8518 (+0.0067) -> learned bias 0.8879 (+0.0427). The gain holds out of domain;
the superseded reg=0.3 vector gave 0.8936, so the milder guarded vector keeps most of it.
CommonLID's 109 labels are all common languages, so suppressing the rare attractors there is
nearly pure gain.

**Caveats:** the bias down-weights rare languages, so a deployment whose inputs are genuinely
rare-language-heavy could see tail regression (test tail point estimate -0.0320, CI touching
zero). The frequency prior is the safer minimal version; the learned bias is the higher-gain
version. Pending: learned bias on the Apertus-200k model (plan item 2); the prior-centered
regularizer (plan item 3) may reduce the tail cost by shrinking tail biases toward
`gamma*log N` instead of toward the unregularized fit.

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
