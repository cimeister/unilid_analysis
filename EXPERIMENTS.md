# UniLID Analysis — Experiment Results

## Overview

Scientific evaluation of UniLID, a language identification system based on a Unigram tokenizer with per-language log-probability weights (1,940 languages, 100k shared vocabulary). Compared against four alternative systems on the GlotLID test set (45,627,279 samples).

### Models evaluated
| Model name | Description | Prediction file |
|-----------|-------------|-----------------|
| UniLID | Primary system (Unigram LM with per-language weights) | `glotlidc_y_pred.txt` |
| UniLID-DeepSeek | DeepSeek v3.2 variant | `deepseek_v3.2_glotlid_y_pred.txt` |
| UniLID-Qwen | Qwen3 8B variant | `qwen3_8b_glotlid_y_pred.txt` |
| UniLID-Marg | Marginalized variant | `marg_y_pred.txt` |
| fastText | fastText baseline (e100) | `fasttext_y_pred.txt` |

### Metrics
- **Accuracy**: exact match
- **Macro F1**: unweighted mean of per-class F1 (averaged over true labels only, matching sklearn convention)
- **Macro FPR**: unweighted mean of per-class false positive rate (×10⁵ for display)

---

## Experiment 1: Multi-System Comparison

### 1.1 Overall Results (500k uniform sample, seed=42)

| Model | Accuracy | Macro F1 | Ma-FPR (×10⁵) |
|-------|----------|----------|----------------|
| UniLID | 0.960 | 0.941 | 2.1 |
| UniLID-DeepSeek | 0.959 | 0.928 | 2.1 |
| UniLID-Qwen | 0.951 | 0.922 | 2.6 |
| UniLID-Marg | 0.961 | 0.943 | 2.1 |
| fastText | 0.947 | 0.947 | 2.8 |

**Full dataset** (45.6M samples) tables were generated via SLURM job 1747558 (completed in 16 min). Outputs in `outputs/tables/table{1-7}_*.{md,tex}`.

### 1.2 Observations by analysis axis

**By text length**: Strong monotonic trend — accuracy drops sharply for short texts (<30 chars: 79.2% for UniLID) and converges above 99% for texts >300 chars. All models follow the same pattern.

**By resource level** (6 bins: <500, 500-1k, 1k-12k, 12k-18k, 18k-35k, 35k+):
- Peak accuracy at 12k-18k training samples: 99.5%
- High-resource bin (35k+): 95.8% accuracy, 92.8% of test samples
- Very low resource (<500, 96 languages): high run-to-run variance, small per-language N

**By script**: Scripts with unique character sets (Hangul, Tamil, Gujarati, etc.) achieve 100% accuracy. Latin script (76% of data, 1,659 languages): 96.3%. Devanagari 89.6%, Arabic 90.7%.

### 1.3 Cross-system comparison

**Error overlap** (500k sample):
- All 5 wrong: 28.9% of any-wrong
- UniLID-DeepSeek and UniLID-Qwen are highly correlated (r=0.93); UniLID is more distinct (r≈0.80)

**Notable per-language divergences**:
- `azj_Latn` (Azerbaijani): UniLID 1.1% error vs DeepSeek 61.2% error
- `lzh_Hani` (Literary Chinese): UniLID 25.0% error vs DeepSeek 3.9% — UniLID's worst relative weakness

### 1.4 Confusion matrices

Seven confusion clusters identified and visualized (PNG heatmaps + LaTeX tables, per model):
1. Arabic dialects (arb, arz, ary, ars, apc, acm)
2. Chinese varieties (cmn, wuu, yue, lzh, hak) — cmn→wuu accounts for 66% of Hani errors
3. Hindi belt (hin, anp, bho, mai, mag, hne, kas, doi) — hin→anp is 56% of Deva errors
4. Malay-Indonesian (ind, zsm, bjn)
5. Scandinavian (dan, nob, nno, swe)
6. Hebrew (heb, hbo)
7. Persian-Iranian (fas, glk, mzn, sdh)

---

## Experiment 2: Tokenization Length Bias Analysis

### 2.1 Hypothesis
UniLID scores languages by summing per-token log-probabilities without a length prior. Since different languages produce different numbers of tokens for the same text (due to per-language Viterbi segmentation), the model may be biased toward languages that produce fewer tokens (fewer negative terms summed).

### 2.2 Architecture details
- Single shared Unigram tokenizer with 100k vocabulary
- 1,940 per-language weight vectors (log-probabilities) stored in `.unilid` binary
- Scoring: `score(lang) = Σ log p(token_i | lang)` where tokenization is language-specific
- The Viterbi segmentation uses per-language weights, so the same text produces different token sequences (and different token counts) for different languages

### 2.3 Token count delta analysis (full dataset: 1,789,423 misclassifications)

SLURM job 1791511 (completed in 5h, 400 GB memory, 1,895 per-language tokenizers built).

**Overall**: Mean delta = -0.17 tokens (pred - true), p ≈ 0, Cohen's d = -0.092
- 24.9% of misclassifications predict a language with fewer tokens
- 14.0% predict a language with more tokens (1.8:1 ratio)
- 61.1% have the same token count

**By text length** — bias scales with text length:
| Length bin | Mean delta | % fewer | % more |
|-----------|-----------|---------|--------|
| <30 chars | -0.11 | 17.7% | 7.7% |
| 75-150 | -0.21 | 32.1% | 20.2% |
| 300+ | -2.71 | 53.1% | 31.8% |

**By confusion cluster**:
- Persian-Iranian: strongest bias (-0.22)
- Chinese varieties: near zero (-0.01, 93% same token count)

### 2.4 Length normalization counterfactual (pairwise)

For each misclassified sample, checked whether normalizing scores by token count (score/n_tokens) would flip the true language's score above the predicted language's score.

**Overall**: 332,034 / 1,789,423 (18.6%) would be pairwise correctable.

**By token delta** — the smoking gun:
| Delta | N | % correctable |
|-------|---|--------------|
| <-2 | 112,762 | 73.4% |
| -2 to -1 | 333,452 | 74.8% |
| 0 | 1,092,902 | 0.0% |
| >0 | 250,307 | 0.0% |

When the predicted language uses 2+ fewer tokens, normalization fixes ~75% of those errors.

### 2.5 Full re-classification with normalized scores (500k sample)

SLURM job 1795556 (completed in 2.5 min). Modified the Rust Unigram tokenizer to add `best_of_cached_weight_sets_normalized` which divides scores by token count before argmax over all 1,940 languages.

**Critical result: normalization makes things significantly worse.**

| Model | Accuracy | Macro F1 |
|-------|----------|----------|
| UniLID (original) | 0.960 | 0.941 |
| UniLID (raw rescore) | 0.960 | 0.941 |
| UniLID (normalized) | 0.885 | 0.875 |

- Raw rescore has **100% agreement** with original predictions (validates the Rust implementation)
- Normalization changes 10% of predictions: 79.6% of changes are wrong (break correct predictions), only 4.4% correct errors
- **Net loss: -37,740 samples** (7.5 percentage point accuracy drop)

**By text length** — normalization hurts most on short texts:
| Length | Original Acc | Normalized Acc | Drop |
|--------|-------------|---------------|------|
| <30 | 0.792 | 0.566 | -0.226 |
| 30-75 | 0.951 | 0.842 | -0.109 |
| 75-150 | 0.978 | 0.925 | -0.053 |
| 300+ | 0.995 | 0.991 | -0.004 |

### 2.6 Pairwise counterfactual vs full re-classification

Of the misclassifications the pairwise counterfactual (§2.4) flagged as correctable (332,034 of 1,789,423, 18.6%), the full re-classification (§2.5) changed 10% of predictions: 79.6% of changes broke previously correct predictions and 4.4% corrected previous errors (net −37,740 predictions, −7.5pp overall accuracy). The largest accuracy drop is on the <30 char bin (0.792 → 0.566) and the smallest on 300+ chars (0.995 → 0.991), while §2.3 records the largest mean token-count delta in the 300+ bin (−2.71).

Possible (unconfirmed) conclusion: the unnormalized sum-of-log-probs scoring carries signal that simple length normalization removes, with the per-token average noisiest for short texts.

---

## SLURM Job History

| Job ID | Name | State | Duration | Memory | Notes |
|--------|------|-------|----------|--------|-------|
| 1747558 | unilid-tables | COMPLETED | 16 min | 64 GB | Full dataset tables (45.6M) |
| 1747559 | unilid-lenbias | OOM | 2.5h | 128 GB | First attempt — texts in pickle caused OOM |
| 1750406 | unilid-lenbias | OOM | 1.75h | 128 GB | Streaming fix — tokenizer cache still OOM at 128 GB |
| 1752234 | unilid-lenbias | COMPLETED | 3h | 400 GB | Token delta only (old version, no scores) |
| 1789048 | unilid-lenbias | OOM | 12 min | 128 GB | After code refactor — still needs 400 GB |
| 1790440 | unilid-lenbias | TIMEOUT | 6h | 400 GB | Score computation added ~2x runtime, hit 6h walltime |
| 1791511 | unilid-lenbias | COMPLETED | 5h | 400 GB | Full run with scores + counterfactual (12h walltime) |
| 1795556 | unilid-norm | COMPLETED | 2.5 min | 400 GB | Normalized prediction on 500k sample |

---

## Output Files

### Tables (markdown + LaTeX)
- `table1_overall.{md,tex}` — 5-model comparison (500k sample, also run on full dataset)
- `table2_by_length.{md,tex}` — by text length bins
- `table3_by_resource.{md,tex}` — by training resource level
- `table4_by_script.{md,tex}` — by script (top 10 + Other)
- `table5_error_overlap.{md,tex}` — cross-system error overlap
- `table6_per_script_winner.{md,tex}` — best model per script
- `table7_divergences.{md,tex}` — per-language divergences (UniLID vs alternatives)
- `length_bias.{md,tex}` — token delta + counterfactual tables (full dataset)
- `normalized_comparison.{md,tex}` — raw vs normalized scoring (500k sample)

### Confusion matrices
- `cm_{cluster}.tex` — LaTeX with cellcolor shading (7 clusters × 5 models)
- `cm_{cluster}_{model}.png` — heatmap images (35 files)
- `length_bias_histogram.png` — distribution of token count deltas

### Cached pickles (on scratch)
- `sample_500k_all.pkl` — 500k uniform sample (no texts, has all 5 model predictions)
- `sample_45627k_all.pkl` — full dataset (same format)

---

## Experiment 3: Per-Language Distribution Analysis

### 3.1 Context

UniLID uses a single shared vocabulary (100k tokens) with per-language log-probability weights estimated via EM re-estimation. Each language starts from the base tokenizer distribution and runs 20 iterations of Unigram EM (soft or hard mode) on its own training corpus. The vocabulary is fixed — only the probabilities change. A probability floor of 1e-12 and convergence early stopping provide minimal regularization.

### 3.2 Languages vs Base Distribution

Computed KL(lang || base) for all 1,940 languages using proper probability normalization (softmax of log-scores).

**Most divergent from base** (KL ~1.2–1.3): Languages with unique scripts — Lisu (`lis_Lisu`, KL=1.28), Hebrew (`ydd_Hebr`, KL=1.27), Tamil (`tam_Taml`, KL=1.25), Myanmar (`ksw_Mymr`, KL=1.24). These concentrate probability mass on script-specific tokens that the base assigns low probability to.

**Closest to base** (KL ~0.20–0.24): All low-resource Latin-script languages with <500 training samples — e.g., `ldn_Latn` (106 samples, KL=0.20), `otw_Latn` (85 samples, KL=0.22). The EM-estimated distributions for these languages remain close to the base distribution.

**By resource level:**

| Resource bin | # Langs | Mean KL | Median KL | Mean corr w/ base | Mean MAD |
|--------------|---------|---------|-----------|-------------------|----------|
| <500 | 56 | 0.32 | 0.27 | 0.189 | 1.17 |
| 500–1k | 40 | 0.50 | 0.43 | 0.206 | 1.78 |
| 1k–12k | 458 | 0.70 | 0.67 | 0.258 | 3.86 |
| 12k–18k | 526 | 0.71 | 0.69 | 0.277 | 4.84 |
| 18k–35k | 398 | 0.70 | 0.69 | 0.289 | 5.23 |
| 35k+ | 462 | 0.68 | 0.62 | 0.321 | 5.90 |

Correlation of log(training count) vs KL: r = 0.261. Correlation of log(training count) vs correlation-with-base: r = 0.351. Both moderate — more data produces more confident divergence from the base, but the relationship saturates above ~10k samples.

### 3.3 Top Divergent Tokens vs Base (selected languages)

For each language, computed per-token KL contribution: `p_lang(i) * log(p_lang(i) / p_base(i))`. The top tokens are those that contribute most to the distributional divergence.

**High-resource English (100k samples):** Top tokens are English function words and multi-word units: `Ġsaid` (+5.69), `Ġwith` (+4.84), `ĠofĠthe` (+4.61), `ĠinĠthe` (+4.57). These are genuinely discriminative — clear signal, not noise.

**Low-resource Ojibwe (otw_Latn, 85 samples):** Top tokens are recognizable Ojibwe morphemes: `zhi` (+3.38), `waa` (+2.24), `gii` (+3.68), `maa` (+2.07). Per-token deltas are 2–4x smaller than for high-resource languages (+2–4 vs +5–12).

**Low-resource Ladin (ldn_Latn, 106 samples):** Similar pattern — small deltas, tokens partially shifted from base but not strongly.

### 3.4 Related Language Pair Analysis

For 15 related language pairs, computed symmetric KL divergence, Pearson correlation of log-probs, and mean absolute deviation (MAD), plus top 20 tokens by KL contribution.

| Pair | Sym KL | Corr | MAD | Train A | Train B |
|------|--------|------|-----|---------|---------|
| Indonesian / Malay | 0.06 | 0.896 | 0.364 | 100k | 100k |
| Hindi / Angika | 0.08 | 0.884 | **2.218** | 100k | 4,499 |
| Kinyarwanda / Kirundi | 0.08 | 0.870 | 0.439 | 100k | 100k |
| Bokmål / Nynorsk | 0.14 | 0.868 | 0.370 | 100k | 100k |
| MSA / Moroccan Arabic | 0.16 | 0.881 | 0.591 | 100k | 100k |
| Persian / Gilaki | 0.16 | 0.839 | **1.279** | 100k | 22,263 |
| Danish / Bokmål | 0.19 | 0.853 | 0.450 | 100k | 100k |
| English / Scots | 0.22 | 0.811 | 0.466 | 100k | 87,458 |
| MSA / Egyptian Arabic | 0.26 | 0.753 | 0.620 | 100k | 100k |
| Hebrew / Biblical Hebrew | 0.26 | 0.640 | 0.457 | 100k | 100k |
| Slovak / Czech | 0.30 | 0.798 | 0.555 | 100k | 100k |
| Spanish / Portuguese | 0.42 | 0.765 | 0.628 | 100k | 100k |
| Russian / Ukrainian | 0.48 | 0.788 | 0.522 | 100k | 100k |

### 3.5 Evidence of EM Noise

**Hindi / Angika (hin_Deva vs anp_Deva)**: The most suspicious pair. Correlation is high (0.884) — comparable to Indonesian/Malay (0.896) — yet MAD is 2.218, which is **6x higher** than Indonesian/Malay (0.364). Angika has only 4,499 training samples. The top divergent tokens are common Hindi function words and punctuation (`Ġ,`, delta +7.95; `Ġà¤¤à¥ģà¤®` = "tum"/you, delta +4.67) that the Angika EM over-suppressed. The scatter plot shows a horizontal band at anp ≈ -17, indicating many tokens collapsed to near-floor probability.

**Persian / Gilaki (fas_Arab vs glk_Arab)**: Correlation 0.839, MAD = 1.279. Gilaki has 22,263 samples.

**Within-cluster pair KL vs training size**: r = -0.03, essentially zero. The divergence between related languages within confusion clusters is **not explained by training data size**.

### 3.6 Resource-level patterns

Observations:

1. Low-resource languages (<500 samples): mean KL from base 0.32, mean MAD 1.17, mean per-language correlation with base 0.189.

2. Mid-to-high-resource languages (5k+ samples): mean KL 0.68–0.71, mean MAD 4.84–5.90. Same-resource related pairs include ind/zsm (KL=0.06) and dan/nob (KL=0.19). For mixed-resource pairs such as hin/anp (100k vs 4,499 samples), the lower-resource distribution has MAD 2.218 at correlation 0.884, with a band of tokens near the probability floor visible in the scatter (§3.5).

EM training uses no explicit regularization (no Dirichlet prior, no damping); the only implicit guards are the probability floor (1e-12) and convergence early stopping.

Possible (unconfirmed) conclusion: low-resource (<500 sample) distributions are under-fit (close to the base) rather than over-fit; for mixed-resource pairs the lower-resource distribution shows EM noise distinct from the higher-resource partner.

### 3.7 Output files

- `outputs/tables/distribution_analysis.{md,tex}` — all tables (KL rankings, resource-level summary, pairwise summary, 13 vs-base token tables, 15 pairwise token tables)
- `outputs/figures/kl_vs_training_size.png` — scatter: KL from base vs training count
- `outputs/figures/pairwise_logprob_scatter.png` — 6 scatter plots of log-probs for related pairs
- `outputs/figures/pairwise_kl_vs_training.png` — within-cluster pairwise KL vs min training size

---

## Experiment 4: Token Classification for Confused Pairs

### 4.1 Context

The distribution analysis (Experiment 3) identified the top KL-contributing tokens for 15 confused language pairs but did not classify *what kind* of tokens drive the divergence. This experiment applies a heuristic classifier to categorize each token and aggregate statistics to understand whether the model's discriminative features are linguistically meaningful or dominated by artifacts.

### 4.2 Token categories

Eight categories defined by heuristic rules (priority order):
1. **Multi-word unit** — tokens containing 2+ space markers (tokenizer artifacts like `ĠofĠthe`)
2. **Punctuation** — mostly non-alphanumeric tokens (`Ġ-`, `,"Ġ`, `?`)
3. **Script/encoding** — invalid Unicode or encoding artifacts
4. **Domain/religious** — domain-specific terms, especially from religious text (JW.org) (`ĠYehuwa`)
5. **Function word** — space-prefixed token matching a cross-lingual function word list (`Ġkarena`, `Ġof`, `Ġaf`)
6. **Character/phonotactic** — very short (1-2 char) alphabetic tokens (`zh`, `yi`)
7. **Morphological affix** — subword tokens without space prefix, ≤5 chars (`tion`, `hed`, `nya`)
8. **Content word** — space-prefixed words not in the function word list (`Ġbisa`, `Ġartikel`)

### 4.3 Aggregate results (300 tokens across 15 pairs)

| Category | # Tokens | % Tokens | % of KL |
|----------|----------|----------|---------|
| **Morph. affix** | 92 | 30.7% | **32.6%** |
| **Content word** | 80 | 26.7% | 22.8% |
| Char/phonotactic | 49 | 16.3% | 10.2% |
| Function word | 33 | 11.0% | 15.7% |
| Punctuation | 24 | 8.0% | 10.5% |
| Multi-word unit | 20 | 6.7% | 7.9% |
| Domain/religious | 2 | 0.7% | 0.3% |
| Script/encoding | 0 | 0.0% | 0.0% |

Morphological affixes (32.6% of KL) and content words (22.8%) together account for 55.4% of the discriminative power. Function words add another 15.7%. Punctuation and formatting contribute 10.5%. Script/encoding artifacts contribute 0%, and domain/religious terms 0.3%.

Possible (unconfirmed) conclusion: discriminative features in the top 20 KL-contributing tokens per pair are dominated by linguistic units (morphological affixes, content words, function words) rather than tokenization or encoding artifacts.

### 4.4 By resource level

**High-resource pairs (both >50k, 13 pairs):**

| Category | % Tokens |
|----------|----------|
| Morph. affix | 33.5% |
| Content word | 19.6% |
| Char/phonotactic | 18.5% |
| Function word | 12.7% |
| Punctuation | 7.3% |
| Multi-word unit | 7.7% |

**Mixed-resource pairs (one <10k, 1 pair: Hindi/Angika):**

| Category | % Tokens |
|----------|----------|
| **Content word** | **75.0%** |
| Morph. affix | 10.0% |
| Punctuation | 15.0% |

The Hindi/Angika pair (the only mixed-resource pair in the analyzed set) shows a different profile: content words 75%, with no function words or character/phonotactic tokens. The Angika distribution (4,499 samples) has many tokens at or near the probability floor (§3.5).

### 4.5 Domain dependency

Only one pair shows significant domain/religious KL contribution:

| Pair | % KL from domain terms |
|------|----------------------|
| Indonesian / Malay | 7.3% |

The discriminative token is `ĠYehuwa` (Jehovah), associated with JW.org Bible translations and present in Indonesian training data at a higher rate than in Malay.

Possible (unconfirmed) conclusion: the Indonesian/Malay distinction is partially dependent on religious-text domain markers.

### 4.6 Per-pair variation (stacked bar chart)

The stacked bar chart (`outputs/figures/token_categories_stacked.png`) shows substantial variation across pairs:

- **English/Scots**: function words ~65%
- **Hindi/Angika**: content words ~75% (the Angika distribution has many tokens near the probability floor, §3.5)
- **Danish/Bokmål** and other Scandinavian pairs: mix of morphological affixes and content words
- **Arabic pairs**: morphological affixes and punctuation most prominent
- **Chinese pairs**: content words and character/phonotactic tokens
- **Spanish/Portuguese** and **Slovak/Czech**: function words ~40%

### 4.7 Output files

- `outputs/tables/token_classification.{md,tex}` — aggregate table, resource-level breakdown, domain dependency, per-pair classified token tables (15 pairs × 20 tokens each)
- `outputs/figures/token_categories_stacked.png` — stacked bar chart of category proportions per pair

---

## Experiment 5: Partial Length Normalization (Alpha Sweep)

### 5.1 Context

Full normalization (alpha=1.0, `score/n_tokens`) dropped accuracy from 96.0% to 88.5% (Experiment 2). We added a partial normalization parameter `score / n_tokens^alpha` to the Rust Unigram model and sweep `alpha ∈ {0.0, 0.1, 0.2, ..., 1.0}` to find whether a partial correction exists.

### 5.2 Implementation

Added `alpha: f32` parameter to:
- `best_of_cached_weight_sets_normalized` in Rust core (Viterbi DP)
- PyO3 Python bindings (with default `alpha=1.0`)
- Python `predict_normalized` / `predict_normalized_batch` wrappers

When `alpha=0`: `(c as f32).powf(0.0) = 1.0`, so normalization reduces to raw scoring (validated: 100% agreement with original predictions).

### 5.3 Results

SLURM job 1804584 (completed in 12 min, 400 GB memory).

| Alpha | Accuracy | Macro F1 |
|-------|----------|----------|
| 0.0 | 0.960 | 0.941 |
| **0.1** | **0.961** | **0.942** |
| 0.2 | 0.960 | 0.942 |
| 0.3 | 0.958 | 0.941 |
| 0.4 | 0.956 | 0.940 |
| 0.5 | 0.951 | 0.936 |
| 0.6 | 0.945 | 0.930 |
| 0.7 | 0.936 | 0.923 |
| 0.8 | 0.923 | 0.912 |
| 0.9 | 0.907 | 0.896 |
| 1.0 | 0.885 | 0.875 |

At alpha = 0.1: accuracy 0.961 (+0.001 over alpha=0.0), macro F1 0.942 (+0.001). 1,749 predictions changed, net +114 corrections. The accuracy curve decreases monotonically for alpha > 0.1; the <30 char bin drops fastest (0.792 → 0.566 at alpha=1.0); the 300+ char bin drops from 0.995 to 0.991 across the full sweep.

**By text length:**

| Alpha |   <30 | 30--75 | 75--150 | 150--300 |  300+ |
|-------|-------|--------|---------|----------|-------|
| 0.0 | 0.792 |  0.951 |   0.978 |    0.987 | 0.995 |
| 0.1 | 0.793 |  0.952 |   0.978 |    0.987 | 0.995 |
| 0.5 | 0.755 |  0.939 |   0.972 |    0.985 | 0.995 |
| 1.0 | 0.566 |  0.842 |   0.925 |    0.966 | 0.991 |

Possible (unconfirmed) conclusion: any benefit of partial length normalization is at most marginal; higher alpha reduces accuracy most on the inputs where the token-count bias (§2.3) is smallest in magnitude.

### 5.4 Output files

- `outputs/tables/alpha_sweep.{md,tex}` — alpha vs accuracy/macro-F1
- `outputs/figures/alpha_sweep.png` — accuracy vs alpha curve (overall + per text length bin)

---

## Experiment 6: Log-Probability Floor Sweep

### 6.1 Context

UniLID's per-language distributions are estimated via EM. Tokens not in a language's vocabulary are assigned a very low log-probability floor. Experiment 3 showed that low-resource languages have noisy EM estimates with many tokens collapsed to near-floor values. This experiment tests whether clamping all per-language log-probability weights at different floor values improves or degrades prediction accuracy.

By raising the floor, OOV tokens (those structurally absent from a language's vocabulary) are given a finite low probability — effectively giving every language access to the full 100k shared vocabulary with a uniform low-probability baseline.

### 6.2 Implementation

No Rust code changes needed. The clamping is applied in Python before pushing weights to the Rust tokenizer cache:
1. Load the weight matrix from the `.unilid` file (memmap, shape 1940 × 100k, float32)
2. For each floor value, clamp: `weights[weights < floor] = floor` (all values including OOV)
3. Push clamped weights to Rust via `set_weight_sets()`
4. Run predictions on 500k sample

New file: `analysis/floor_sweep.py`.

### 6.3 Results

SLURM job 1806690 (completed in 4.5 min, 400 GB memory).

**Diagnostic — elements clamped per floor value** (out of 194M total):

| Floor | Elements clamped | % of matrix |
|-------|-----------------|-------------|
| -22 | 0 | 0.0% |
| -15 | 176,000,253 | 90.7% |
| -10 | 192,947,960 | 99.5% |

The zero-clamp result at -22 indicates all stored weight values are ≥ -22. The 90.7% clamped at -15 reflects the large number of OOV tokens per language (~95k of 100k tokens are OOV for a typical language).

**Overall results:**

| Floor | Accuracy | Macro F1 | Changed | Corrected | Broken | Net |
|-------|----------|----------|---------|-----------|--------|-----|
| None (baseline) | 0.960 | 0.941 | -- | -- | -- | -- |
| -22 | 0.960 | 0.941 | 0 | 0 | 0 | 0 |
| -15 | 0.960 | 0.943 | 3,372 | 1,129 | 1,238 | -109 |
| -10 | 0.916 | 0.837 | 36,467 | 3,390 | 25,485 | -22,095 |

**By text length:**

| Floor |   <30 | 30--75 | 75--150 | 150--300 |  300+ |
|-------|-------|--------|---------|----------|-------|
| None | 0.792 |  0.951 |   0.978 |    0.987 | 0.995 |
| -22 | 0.792 |  0.951 |   0.978 |    0.987 | 0.995 |
| -15 | 0.788 |  0.951 |   0.978 |    0.987 | 0.994 |
| -10 | 0.608 |  0.887 |   0.954 |    0.976 | 0.992 |

### 6.4 Observations

- Floor=-22: 0 elements clamped; predictions identical to baseline (0/500k changed).
- Floor=-15: 90.7% of the weight matrix clamped (176M of 194M elements); 3,372 of 500k predictions changed. Accuracy 0.960 (unchanged at three decimals), macro F1 +0.002 (0.941 → 0.943), net −109 predictions.
- Floor=-10: 99.5% of the matrix clamped; accuracy 0.916 (−0.044), macro F1 0.837 (−0.104). The <30 char bin drops 0.792 → 0.608; 25,485 predictions broken vs 3,390 corrected.
- Same monotonic pattern as the alpha sweep (Experiment 5): accuracy decreases as modification strength increases, with the largest reductions on the <30 char bin.

Possible (unconfirmed) conclusion: the existing parameterization is near-optimal under sum-of-log-prob scoring; modifications that reduce per-token discrimination reduce accuracy most on short inputs.

### 6.5 Output files

- `outputs/tables/floor_sweep.{md,tex}` — floor vs accuracy/macro-F1 (overall + by text length)
- `outputs/figures/floor_sweep.png` — accuracy vs floor curve (overall + per text length bin)

---

## Experiment 7: Training Data Analysis

### 7.1 Domain Distribution

Streamed the full training corpus (60,683,151 lines, 1,940 languages) with heuristic domain classifiers. Ran on login node (~30 min single-pass).

**Overall**: 98.1% of training data is classified as "other" (not religious, not Wikipedia); 1.9% detected as religious/Bible text; Wikipedia markers 0.002%.

**By resource level:**

| Resource bin | Religious % | Wikipedia % | Other % |
|-------------|-------------|-------------|---------|
| <500 | 2.5 | 0.0 | 97.5 |
| 500-1k | 1.9 | 0.0 | 98.1 |
| 1k-12k | 0.5 | 0.0 | 99.5 |
| 12k-18k | 0.3 | 0.0 | 99.7 |
| 18k-35k | 0.6 | 0.0 | 99.4 |
| 35k+ | 2.7 | 0.0 | 97.3 |

**Confusion cluster languages**: religious-domain share by language: Indonesian (ind_Latn) 2.2%, Malay (zsm_Latn) 0.2% (consistent with the `Yehuwa` finding from Experiment 4); Bokmål (nob_Latn) 6.2% (highest among the cluster languages).

**Caveat**: The domain heuristics are conservative (keyword/pattern matching). Many Bible translations lack explicit markers like `Yehuwa` or verse references.

### 7.4 Per-Language Corpus Quality

| Resource bin | # Langs | Mean text len | Mean char entropy | Mean vocab size |
|-------------|---------|---------------|-------------------|-----------------|
| <500 | 56 | 83.8 | 4.47 | 1,447 |
| 500-1k | 40 | 94.7 | 4.51 | 3,670 |
| 1k-12k | 458 | 148.0 | 4.39 | 15,718 |
| 12k-18k | 526 | 166.7 | 4.34 | 19,597 |
| 18k-35k | 398 | 164.1 | 4.28 | 29,619 |
| 35k+ | 462 | 104.6 | 4.55 | 102,776 |

Low-resource languages (<500) have shorter texts (mean 84 chars vs 105-167 for larger bins) and much smaller vocabularies (~1.4k vs 100k+). Character entropy is similar across bins (~4.3-4.6 bits). The high-resource bin (35k+) has shorter mean text length (104.6) than the mid-resource bins (148-167).

### 7.5 Script Verification

20 languages flagged with >5% unexpected script characters:

- **6 Canadian Aboriginal Syllabics languages** (`crk_Cans`, `crm_Cans`, etc.): 100% Latin, 0% Syllabics. These are written in Latin romanization despite the `_Cans` script label.
- **Japanese** (`jpn_Jpan`): 35.4% Han, 52.3% Hiragana. Expected: `Jpan` maps to Han, but Japanese text naturally uses Hiragana/Katakana prominently. This is a script code mapping issue, not a data quality issue.
- **Several Cyrillic languages** (`kca_Cyrl`, `kpv_Cyrl`, etc.): 91-95% Cyrillic with some Latin (transliterated content or code-switching).

The vast majority of languages (1,901/1,921 = 99.0%) have >95% of their characters in the expected script.

### 7 Output files

- `outputs/tables/train_data_analysis.md` — all domain, quality, and script tables
- `outputs/figures/train_domain_stacked.png` — domain distribution for cluster languages
- `outputs/figures/train_quality_scatter.png` — entropy and vocab size vs training count
- `outputs/figures/train_script_purity.png` — script purity histogram

---

## Experiment 8a: Heuristic Discriminative Weighting

### 8a.1 Context

Tested whether variance-based token weighting could improve within-cluster discrimination. For each confusion cluster, computed per-token variance across cluster languages and applied three setups: (A) additive upweighting of discriminative tokens, (B) additive rescaling with z-scored variance, (C) sigmoid gate replacing non-discriminative tokens with base distribution.

### 8a.2 Results

SLURM job 1808414 (completed in 13 min, 400 GB memory).

**All setups reduce accuracy at every tested parameter:**

| Setup | Param | Overall Acc | Overall Ma-F1 |
|-------|-------|-------------|---------------|
| Baseline | -- | 0.960 | 0.941 |
| A (upweight) | α=0.5 | 0.866 | 0.914 |
| B (rescale) | α=0.5 | 0.889 | 0.918 |
| C (gate) | β=1.0 | 0.899 | 0.923 |

Per-cluster accuracy collapses to near 0% for all clusters at α≥1.0 in setups A and B. Setup C is the least destructive but still drops from 96.0% to 89.9%.

### 8a.3 Observations

All three setups reduce accuracy at every parameter setting tested (A: α ∈ {0.5, 1.0, 2.0, 5.0}; B: α ∈ {0.5, 1.0, 2.0, 5.0}; C: β ∈ {1.0, 5.0, 10.0}). Best config across setups: A α=0.5, net −30,433 predictions. At α ≥ 1.0 in setups A and B, per-cluster accuracy is 0 across all seven clusters.

Possible (unconfirmed) conclusion: variance-based token re-weighting at the granularity used here does not improve within-cluster discrimination; any improvement, if it exists, requires a mechanism other than per-token additive/multiplicative adjustments to the EM-trained weights.

### 8a.4 Output files

- `outputs/tables/discriminative_heuristic.md` — overall and per-cluster accuracy for all setups

---

## Experiment 9: Distribution Transfer for Low-Resource Languages

### 9.1 Context

Low-resource languages (<5k samples) are under-fit: barely diverged from the base distribution (KL~0.3). Tested two transfer approaches, both operating in probability space (arithmetic interpolation):

- **9a**: Interpolate with the closest high-resource same-script language (223 transfer pairs identified)
- **9b**: Interpolate with the script-average distribution (average of all languages sharing the same script)

### 9.2 Results

SLURM job 1808399 (completed in 25 min, 400 GB memory).

**9a: Related language transfer**

| Lambda | Overall | <500 | 500-5k | High-res |
|--------|---------|------|--------|----------|
| 1.0 (baseline) | 0.960 | 0.789 | 0.958 | 0.960 |
| 0.9 | 0.959 | 0.789 | 0.964 | 0.959 |
| 0.7 | 0.957 | 0.842 | 0.968 | 0.957 |
| 0.3 | 0.947 | **0.895** | 0.965 | 0.947 |
| 0.0 | 0.878 | 0.053 | 0.169 | 0.879 |

Best for <500 group: λ=0.3 (89.5%, +10.6pp), but overall drops to 94.7%. The 500-5k group peaks at 96.8% at λ=0.7 (+1.0pp). Heavy transfer (λ<0.3) damages high-resource predictions because all 1940 languages compete in the argmax.

**9b: Script-average transfer**

| Lambda | Overall | <500 | 500-5k | High-res |
|--------|---------|------|--------|----------|
| 1.0 (baseline) | 0.960 | 0.789 | 0.958 | 0.960 |
| 0.9 | 0.960 | 0.789 | 0.957 | 0.960 |
| 0.7 | 0.960 | 0.789 | 0.952 | 0.960 |
| 0.1 | **0.961** | 0.526 | 0.837 | 0.962 |
| 0.0 | 0.961 | 0.053 | 0.083 | 0.962 |

Script average is much more stable than related-language transfer: overall accuracy stays at 96.0-96.1% across all λ values. At λ=0.1, overall ticks up to 96.15% and high-resource improves to 96.2%. However, the <500 group degrades at any λ<1.0 — the script average washes out what little distinctive signal these tiny languages have.

### 9.3 Observations

- 9a (related-language transfer): <500 accuracy peaks at λ=0.3 (0.895, +10.6pp over baseline 0.789); 500-5k peaks at λ=0.7 (0.968, +1.0pp); overall drops from 0.960 (λ=1.0) to 0.947 at λ=0.3. For λ ≤ 0.3, accuracy on all three groups drops sharply (e.g. <500: 0.053 at λ=0.0).
- 9b (script-average transfer): overall accuracy stays 0.960-0.961 across λ ∈ [0.1, 1.0]; <500 accuracy does not exceed 0.789 (baseline) at any λ < 1.0 and falls to 0.526 at λ=0.1.
- Neither approach increases both overall and <500-group accuracy simultaneously in the tested range.

Possible (unconfirmed) conclusion: probability-space interpolation of EM-trained per-language distributions toward a related-language or script-average distribution does not jointly improve overall and very-low-resource accuracy in the configurations tested.

### 9.4 Output files

- `outputs/tables/transfer_sweep.md` — λ vs accuracy (overall, <500, 500-5k, high-resource)
- `outputs/figures/transfer_sweep.png` — accuracy curves for both approaches

---

## SLURM Job History

| Job ID | Name | State | Duration | Memory | Notes |
|--------|------|-------|----------|--------|-------|
| 1747558 | unilid-tables | COMPLETED | 16 min | 64 GB | Full dataset tables (45.6M) |
| 1747559 | unilid-lenbias | OOM | 2.5h | 128 GB | First attempt — texts in pickle caused OOM |
| 1750406 | unilid-lenbias | OOM | 1.75h | 128 GB | Streaming fix — tokenizer cache still OOM at 128 GB |
| 1752234 | unilid-lenbias | COMPLETED | 3h | 400 GB | Token delta only (old version, no scores) |
| 1789048 | unilid-lenbias | OOM | 12 min | 128 GB | After code refactor — still needs 400 GB |
| 1790440 | unilid-lenbias | TIMEOUT | 6h | 400 GB | Score computation added ~2x runtime, hit 6h walltime |
| 1791511 | unilid-lenbias | COMPLETED | 5h | 400 GB | Full run with scores + counterfactual (12h walltime) |
| 1795556 | unilid-norm | COMPLETED | 2.5 min | 400 GB | Normalized prediction on 500k sample |
| 1804584 | unilid-alpha | COMPLETED | 12 min | 400 GB | Alpha sweep (11 values) on 500k sample |
| 1806690 | unilid-floor | COMPLETED | 4.5 min | 400 GB | Floor sweep (3 values + baseline) on 500k sample |
| 1808414 | unilid-disc8a | COMPLETED | 13 min | 400 GB | Heuristic discriminative weighting (3 setups) |
| 1808399 | unilid-transfer | COMPLETED | 25 min | 400 GB | Transfer sweep (related + script avg, 22 configs) |

---

## Pending / Future Work

1. **MMI discriminative fine-tuning (Experiment 8b)** — gradient-based optimization of per-language weight vectors within confusion clusters using softmax cross-entropy loss.
2. **Training data mislabeling analysis** — run model on its own training data to identify systematic mislabeling (deferred from Experiment 7)
3. **Training data overlap analysis** — exact duplicate and n-gram overlap between confusable pairs (deferred from Experiment 7)
4. **Per-script normalization** — different alpha values per script family, or normalize only within confusable groups
