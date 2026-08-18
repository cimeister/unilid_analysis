# Paper edits required by the corrected model

Concrete edit list. Every row names the site, the current text, and either the
new value or the specific run it waits on. Line numbers are `paper/submission.tex`
at commit `e6bf689`.

No disclosure of the selection history is included: the preprint is unreleased,
so the paper states the procedure and the constant it selected, not what an
earlier model selected.

---

## Decision needed first, because it changes several numbers below

**The grid the constant was swept over.** The published grid
$\{-17,-19,-21,-23\}$ was round numbers spanning the released model's
unseen-token range ($-19.94$ to $-13.22$). The corrected model's range is
$-18.33$ to $-11.61$, so the same grid-choosing logic gives different round
numbers. I swept the published grid shifted by $\log 5$
($\{-15.39,-17.39,-19.39,-21.39\}$) because that is the like-for-like comparison
against the released model, and it selected $c = -17.3906$.

For a paper with no released predecessor, that grid prints as an arbitrary set of
four-decimal numbers. Two options:

| | grid | prints as | cost |
|---|---|---|---|
| **A. Keep the shifted grid** | $\{-15.39,-17.39,-19.39,-21.39\}$ | $c = -17.39$ | none, already run |
| **B. Re-sweep on round numbers** | $\{-15,-17,-19,-21\}$ | $c$ likely $-17$ | one 14-minute job |

I recommend **B**. The original grid was itself round numbers chosen against the
model's own range, so applying that logic to this model is the faithful thing,
not a shifted grid that only makes sense relative to a model no reader sees. It
is a re-sweep of the same protocol on the same validation data, not a search for
a better answer, and I would record it as such.

Everything marked $c$ below takes the value this decision fixes. The row counts
in item 3 also depend on it.

---

## A. Ready now

| # | Site | Current | New | Source |
|---|---|---|---|---|
| 1 | `:754` | "shared unseen-token constant $c = -21$" | $c = -17.39$ (or $-17$ under B) | job 3107082 |
| 2 | `:1287` | "$c = -21$ ... sweep over $\{-17,-19,-21,-23\}$" | new constant and grid | job 3107082 |
| 3 | `:627-628` | "every unseen-token value exceeds $c$, so all of them are set to $c$" | **1,821 of 1,940 are set to $c$; 119 already lie below it and are left unchanged** | job 3107082 |
| 4 | `:629-631` | "that value is a byproduct of the training-time probability floor of $10^{-12}$ and renormalization" | **the smallest value the per-language fit assigns; scales as one count in the training-token count $T$; the $10^{-12}$ floor is never reached** | B0 |
| 5 | `tab:lid_main`, \unilid row, GlotLID-C | `.929` / `2.03e-5` | **`.933` / `2.02e-5`** | job 3107045 |
| 6 | `:344`, `:833` | "from .929 to .957" | `.929` becomes **`.933`**; the `.957` is item B1 | job 3107045 |
| 7 | `:824` | "2.03e-5 vs 2.71e-5", "roughly 25\%" | **`2.02e-5`** vs 2.71e-5; "roughly 25\%" still holds (25.5\%) | job 3107045 |
| 8 | `tab:calibration_provenance` | "unseen-token constant $c=-21$" | new constant | job 3107082 |

**Item 4 is the only one that is a claim rather than a number, and it is the one
the paper currently gets wrong.** Replacement wording and the reasoning behind it
are in `paper/appendix_revision_draft_2026-08-17.md` item 1.

**Item 3 turns the base model into the case the Mistral-Nemo paragraph already
describes** (some rows already below $c$). Item B7 must be written to match.

---

## B. Blocked, each on a named run

| # | Site | What is needed | Blocked on |
|---|---|---|---|
| B1 | `tab:lid_main` calibrated row, GlotLID-C cells (`.957`/`1.77e-5`); `:344`, `:833`, `:835` | gated predictions on the corrected model | job 3110918, then `solo_gates.py floor21` (1,084 thresholds), then `gate_variants.py` |
| B2 | `tab:lid_main` \unilid and calibrated UDHR / FLORES cells; `:850-851` (`0.859`, `0.838`, `0.932`, `0.933`) | the E2 external-benchmark chain re-run | `external_bench_eval.py`, after B1 |
| B3 | `tab:calibrated_heldout`, `tab:calibrated_views` | held-out and both-views tables | after B1 |
| B4 | `tab:commonlid` | CommonLID re-scored | `commonlid_carried.py` then `commonlid_calibrated.py`, after B1 |
| B5 | `tab:resource-tier`, `tab:script-breakdown`, `tab:per_language_f1` (\unilid column) | breakdowns | `paper_breakdowns.py`, `regen_resource_tier_counts.py`, after B1 |
| B6 | `:1284` "held-out macro F1 rises from 0.912 to 0.930" | the constant-alone ablation | after B1 |
| B7 | `:1383-1384` "two languages whose trained unseen-token values already lie below $c = -21$"; the variant's high-entropy group (Banjar, Scots, Serbian-Latin); `tab:calibrated_nemo` | the corrected Mistral-Nemo chain end to end | `mistralnemo_eval.py`, six stages, not started |
| B8 | `tab:viterbi_vs_marginal` (`.961`/`.929`, `.962`/`.931`) | both decoders on the corrected model | **job 3110925, queued** |
| B9 | `tab:lenbias-norm` | alpha 0 and 1 on the corrected model | **job 3110926, queued** |
| B10 | `tab:lenbias-delta` | the token-count delta on misclassifications; `:1247` quotes "0.17 tokens" | `length_bias.py`, after B1 |

---

## C. Needs the co-author, or has no artifact here

| # | Site | Note |
|---|---|---|
| C1 | `tab:unilid_llm_comparison`, `tab:noise_robustness`, `tab:length_accuracy`, `tab:samples-accuracy`, `tab:vocab_size_efficiency`, `tab:tatoeba_udhr_comparison` | WiLI and DSL-ML models are not on this machine |
| C2 | `tab:lid_main` \unilid-DeepSeek3.2 and \unilid-Qwen3 rows, 24 cells | no artifact here, no owner identified. Author decision 2026-08-17: they stay on pre-correction weights with the caption stating the mixture |
| C3 | `:975` "all within 0.025 macro F1" | **This comparison will straddle two model generations** once C2 stands: corrected Mistral-Nemo against pre-correction DeepSeek3.2 and Qwen3. Needs rewording, not a new number, unless the two variants are recomputed |

---

## D. Unaffected, listed so they are not re-checked

`tab:latency_glotlid`, `tab:latency_wili`, `tab:training_time`,
`tab:dialect_stats`, `tab:fasttext_epoch_sweep`, and every \fasttext, \glotlid
and \cld row in every table. Also `:754`'s statement that token probabilities are
floored at $10^{-12}$ during training, which remains true of the code; what
changes is only the claim that the floor explains the unseen-token values.

---

## Caption work implied by the above

- `tab:lid_main`'s caption must state that the DeepSeek3.2 and Qwen3 rows are
  computed on different weights from the \unilid, calibrated and Mistral-Nemo
  rows (C2).
- The existing editorial note in that caption about the UDHR-subset FPR of
  1.06e-5 is unresolved and independent of this work.
