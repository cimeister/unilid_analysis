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

### 1.2 Key findings by analysis axis

**By text length**: Strong monotonic trend — accuracy drops sharply for short texts (<30 chars: 79.2% for UniLID) and converges above 99% for texts >300 chars. All models follow the same pattern.

**By resource level** (6 bins: <500, 500-1k, 1k-12k, 12k-18k, 18k-35k, 35k+):
- Sweet spot at 12k-18k training samples: 99.5% accuracy
- High-resource bin (35k+) has lower accuracy (95.8%) but dominates the data (92.8% of samples)
- Very low resource (<500, 96 languages) has high variance due to tiny sample sizes

**By script**: Scripts with unique character sets (Hangul, Tamil, Gujarati, etc.) achieve 100% accuracy. Latin script (76% of data, 1,659 languages) is the hardest at 96.3%. Devanagari (89.6%) and Arabic (90.7%) also challenging due to dialect continua.

### 1.3 Cross-system comparison

**Error overlap** (500k sample):
- All 5 wrong: 28.9% of any-wrong — these are likely genuinely ambiguous samples
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

### 2.6 Interpretation

The pairwise counterfactual was misleading. While normalization does lift the true language above the predicted language in 18.6% of error cases, it simultaneously lifts many *other* incorrect languages above the correct one. The unnormalized sum-of-log-probs scoring carries genuine signal: languages that tokenize a text into fewer tokens often genuinely fit the text better. Simple length normalization destroys this signal, especially for short texts where the per-token average is noisy.

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

**Closest to base** (KL ~0.20–0.24): All low-resource Latin-script languages with <500 training samples — e.g., `ldn_Latn` (106 samples, KL=0.20), `otw_Latn` (85 samples, KL=0.22). The EM barely moved these from the base, suggesting insufficient data for meaningful distribution estimation.

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

**Low-resource Ojibwe (otw_Latn, 85 samples):** Top tokens are recognizable Ojibwe morphemes: `zhi` (+3.38), `waa` (+2.24), `gii` (+3.68), `maa` (+2.07). The EM did learn real signal even with 85 samples, but the deltas are 2–4x smaller than high-resource languages (+2–4 vs +5–12), suggesting the distribution is only partially specialized.

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

**Hindi / Angika (hin_Deva vs anp_Deva)**: The most suspicious pair. Correlation is high (0.884) — comparable to Indonesian/Malay (0.896) — yet MAD is 2.218, which is **6x higher** than Indonesian/Malay (0.364). Angika has only 4,499 training samples. The top divergent tokens are common Hindi function words and punctuation (`Ġ,`, delta +7.95; `Ġà¤¤à¥ģà¤®` = "tum"/you, delta +4.67) that the Angika EM over-suppressed. The scatter plot shows a horizontal band at anp ≈ -17, indicating many tokens collapsed to near-floor probability — a signature of insufficient EM data.

**Persian / Gilaki (fas_Arab vs glk_Arab)**: Similar pattern. Correlation 0.839 but MAD = 1.279 (3x higher than expected for this correlation level). Gilaki has 22,263 samples — more than Angika but still showing noise.

**Within-cluster pair KL vs training size**: r = -0.03, essentially zero. The divergence between related languages within confusion clusters is **not explained by training data size**, suggesting the pairwise KL mostly reflects genuine linguistic differences, with noise adding variance but not dominating.

### 3.6 Key finding: noise vs signal

The analysis reveals two regimes:

1. **Low-resource languages (<500 samples)**: EM barely moves the distribution from the base (KL ~0.3, MAD ~1.2). The distributions are under-specialized. These languages are close to the base but also close to each other, making them hard to distinguish. The risk is **under-fitting**, not over-fitting.

2. **Mid-to-high-resource languages (5k+ samples)**: EM produces well-separated distributions (KL ~0.7, MAD 4–6). For language pairs that are genuinely similar, the distributions are appropriately close (e.g., ind/zsm: KL=0.06). But when one member of a pair has much less data (e.g., anp with 4.5k vs hin with 100k), the lower-resource distribution shows noise: high MAD despite high correlation, with many tokens collapsed to near-floor probability.

The EM process has **no explicit regularization** (no Dirichlet prior, no damping). The only implicit guards are the probability floor (1e-12) and convergence early stopping. For languages with <5k training samples, the lack of regularization allows the distribution to drift, particularly on tokens that appear rarely or never in the training corpus.

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

**Key finding: The model's discriminative features are overwhelmingly linguistic, not artifactual.** Morphological affixes (32.6% of KL) and content words (22.8%) together account for 55.4% of the discriminative power. Function words add another 15.7%. Punctuation and formatting contribute 10.5% — noticeable but not dominant. Script/encoding artifacts contribute 0%, and domain/religious terms only 0.3%.

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

The Hindi/Angika pair (the only mixed-resource pair in the analyzed set) shows a dramatically different profile: content words dominate at 75%, with no function words or character/phonotactic tokens. This is consistent with the EM noise hypothesis — the Angika distribution (4,499 samples) has collapsed many tokens to near-floor, leaving only content-word-level differences.

### 4.5 Domain dependency

Only one pair shows significant domain/religious KL contribution:

| Pair | % KL from domain terms |
|------|----------------------|
| Indonesian / Malay | 7.3% |

The discriminative token is `ĠYehuwa` (Jehovah) — a marker of JW.org Bible translations that are common in Indonesian training data but rare in Malay. This suggests the Indonesian/Malay distinction is partially dependent on religious text domain, which may not generalize to secular text.

### 4.6 Per-pair variation (stacked bar chart)

The stacked bar chart (`outputs/figures/token_categories_stacked.png`) shows substantial variation across pairs:

- **English/Scots**: dominated by function words (~65%) — the distinction is primarily syntactic
- **Hindi/Angika**: dominated by content words (~75%) — EM noise signature
- **Danish/Bokmål** and **Scandinavian pairs**: mix of morphological affixes and content words
- **Arabic pairs**: morphological affixes and punctuation prominent
- **Chinese pairs**: content words and character/phonotactic tokens
- **Spanish/Portuguese** and **Slovak/Czech**: function words ~40% — closely related languages distinguished by common words

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

**Best alpha = 0.1**: marginal improvement (+0.1pp accuracy, +0.1pp macro F1). Only 1,749 predictions changed, net +114 corrections. The accuracy curve is monotonically decreasing after alpha=0.1, with short texts (<30 chars) degrading fastest. The 300+ char bin is nearly immune (0.995 → 0.991 even at alpha=1.0).

**By text length:**

| Alpha |   <30 | 30--75 | 75--150 | 150--300 |  300+ |
|-------|-------|--------|---------|----------|-------|
| 0.0 | 0.792 |  0.951 |   0.978 |    0.987 | 0.995 |
| 0.1 | 0.793 |  0.952 |   0.978 |    0.987 | 0.995 |
| 0.5 | 0.755 |  0.939 |   0.972 |    0.985 | 0.995 |
| 1.0 | 0.566 |  0.842 |   0.925 |    0.966 | 0.991 |

**Conclusion**: Partial normalization provides negligible benefit. The length bias is real but the raw scoring already captures genuine signal that normalization destroys.

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

### 6.4 Interpretation

- **Floor=-22 is a no-op**: all weight values are already ≥ -22, so no clamping occurs.
- **Floor=-15 is near-neutral**: despite clamping 90.7% of the weight matrix (mostly OOV tokens), only 3,372 of 500k predictions change. Macro F1 ticks up by 0.002 but accuracy is flat and the net effect is slightly negative (-109). Making OOV tokens accessible at probability exp(-15) ≈ 3×10⁻⁷ barely matters because Viterbi still prefers in-vocabulary tokens.
- **Floor=-10 is destructive**: 4.4pp accuracy drop, driven by short texts (<30 chars: 79.2% → 60.8%). At exp(-10) ≈ 4.5×10⁻⁵, OOV tokens become competitive enough to distort the Viterbi segmentation. 25,485 predictions broken vs 3,390 corrected.

The pattern mirrors the alpha sweep (Experiment 5): the model's existing parameterization is near-optimal for the current scoring approach. Modifications that reduce discriminative power — whether by normalizing scores or by raising the probability floor — hurt more than they help, especially for short texts where there is less signal to work with.

### 6.5 Output files

- `outputs/tables/floor_sweep.{md,tex}` — floor vs accuracy/macro-F1 (overall + by text length)
- `outputs/figures/floor_sweep.png` — accuracy vs floor curve (overall + per text length bin)

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

---

## Pending / Future Work

1. **Regularized EM re-estimation** — add Dirichlet prior during per-language EM, evaluate on low-resource languages
2. **Tokenization length bias for other models** — compare whether UniLID-Marg or fastText show similar biases
3. **Per-script normalization** — different alpha values per script family, or normalize only within confusable groups
