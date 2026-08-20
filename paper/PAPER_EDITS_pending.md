# Paper edits required by the corrected model

Concrete edit list. Each row names the site, the current text, and either the new
value or the run it waits on. Line numbers are `paper/submission.tex`.

No disclosure of selection history is included: the preprint is unreleased, so
the paper states the procedure and the constant it selected.

**The unseen-token constant is settled: c = -17** (job 3117581, round grid
{-15,-17,-19,-21} chosen by the rule the published grid follows; pre-registered
clamp counts and selection both hit exactly). 1,655 of 1,940 rows are clamped and
285 already lie below it.

---

## A. APPLIED 2026-08-19, commit `6374b67`

All wrapped in a new `\corrrev{}` macro (blue), kept separable from the
camera-ready `\camrev{}` pass. To accept all of this round, redefine as
`\newcommand{\corrrev}[1]{#1}`. Brace balance checked before and after: my edits
introduced no imbalance (the file carries a pre-existing delta of 1 that my
crude check flags in both versions).

Item 11 (`tab:lenbias-norm`) is **not** applied; it is being regenerated, see
below.

| # | Site | Current | New |
|---|---|---|---|
| 1 | `:754` | "shared unseen-token constant $c = -21$" | **$c = -17$** |
| 2 | `:1287` | "$c = -21$ ... sweep over $\{-17,-19,-21,-23\}$" | **$c = -17$, sweep over $\{-15,-17,-19,-21\}$** |
| 3 | `:627-628` | "every unseen-token value exceeds $c$, so all of them are set to $c$" | **1,655 of the 1,940 exceed $c$ and are set to it; the remaining 285 already lie below and are left unchanged** |
| 4 | `:629-631` | "byproduct of the training-time probability floor of $10^{-12}$ and renormalization" | **the smallest value the per-language fit assigns, scaling as one count in the training-token count $T$; the floor is never reached.** Wording in the appendix draft, item 1 |
| 5 | `tab:lid_main`, \unilid row, GlotLID-C | `.929` / `2.03e-5` | **`.933` / `2.02e-5`** |
| 6 | `tab:lid_main`, \unilid-Mistral-Nemo, GlotLID-C | `.912` / `1.84e-5` | **`.912` / `1.86e-5`** (F1 unchanged to three decimals) |
| 7 | `:344`, `:833` | "from .929 to .957" | `.929` becomes **`.933`**; the `.957` is B1 |
| 8 | `:824` | "2.03e-5 vs 2.71e-5", "roughly 25\%" | **`2.02e-5`**; "roughly 25\%" still holds (25.5\%) |
| 9 | `tab:calibration_provenance` | "unseen-token constant $c=-21$" | **$c=-17$** |
| 10 | `tab:viterbi_vs_marginal` | `.961`/`.929`, `.962`/`.931` | **`.961`/`.933`, `.961`/`.935`.** The "+0.002 from marginalization" claim survives (+0.0023). **Two caption fixes:** the accuracy cells now round to the same value so the bolding must go, and measured cost is about 1.7x Viterbi, not "approximately $2\times$" |
| 11 | `tab:lenbias-norm` | Original / Raw rescore / Normalized, overall `.960`/`.960`/`.885` | **REGENERATING**, job 3129778. See below |

### `tab:lenbias-norm`: regenerating rather than applying

The published table came from `outputs/tables/normalized_comparison.md`
(2026-04-04), whose per-bin figures match it exactly. Its Original column is the
sample pickle's stored `pred_UniLID`.

The first corrected run omitted that column, because half of the 500,000-line
draw is the validation half that the full-pool runs exclude, so the corrected
model has no plain-scorer prediction there. Per the author's instruction to redo
it on a new subset, it is being rebuilt on the **golden subset**: the test half
of the same draw, 250,000 lines, which is inside the scored pool. That fills
Original from this model's own `pred_baseline.npy`, restores the implementation
check (Raw rescore must reproduce the plain scorer exactly, enforced as a hard
gate), and keeps the validation half out of a reported number. It is the same
subset both release gates use.

The caption's sample size changes from 500k to 250k accordingly.

---

## B. Blocked, each on a named run

| # | Site | Blocked on |
|---|---|---|
| B1 | `tab:lid_main` calibrated row GlotLID-C cells (`.957`/`1.77e-5`); `:344`, `:833`, `:835` | jobs 3123324 (group-A thresholds) and 3127704 (topk), then `gate_variants apply` |
| B2 | `tab:lid_main` \unilid and calibrated UDHR / FLORES cells; `:850-851` | `external_bench_eval.py`, after B1 |
| B3 | `tab:calibrated_heldout`, `tab:calibrated_views` | after B1 |
| B4 | `tab:commonlid` | `commonlid_carried.py` then `commonlid_calibrated.py`, after B1 |
| B5 | `tab:resource-tier`, `tab:script-breakdown`, `tab:per_language_f1` | `paper_breakdowns.py`, `regen_resource_tier_counts.py`, after B1 |
| B6 | `:1284` "held-out macro F1 rises from 0.912 to 0.930" | after B1 |
| B7 | `:1383-1384` Mistral-Nemo unseen-token parenthetical; its high-entropy group; `tab:calibrated_nemo` | Mistral-Nemo stages after `baseline`, which clamp |
| B8 | `tab:lid_main` \unilid-DeepSeek3.2 and \unilid-Qwen3 rows | retrains 3112879 / 3112846, then evals 3117575 / 3117576 |
| B9 | `tab:lenbias-delta`; `:1247`'s "0.17 tokens" | `length_bias.py`, plus the instrument decision below |

---

## C. Needs the co-author

`tab:unilid_llm_comparison`, `tab:noise_robustness`, `tab:length_accuracy`,
`tab:samples-accuracy`, `tab:vocab_size_efficiency`,
`tab:tatoeba_udhr_comparison`. The WiLI and DSL-ML models are not on this machine
and were not in the Drive folder.

---

## D. Unaffected

`tab:latency_glotlid`, `tab:latency_wili`, `tab:training_time`,
`tab:dialect_stats`, `tab:fasttext_epoch_sweep`, and every \fasttext, \glotlid
and \cld row. Also `:754`'s statement that token probabilities are floored at
$10^{-12}$ during training, which remains true of the code; only the claim that
the floor explains the unseen-token values changes.

---

## Caption and framing work

- **`tab:lid_main`'s caption loses its split.** It currently says the \unilid and
  calibrated rows are on the 45,377,279-line scored pool while the others carry
  over from the original submission on all 45,627,279 lines. The DeepSeek3.2 and
  Qwen3 rows are being recomputed on the scored pool, so the table becomes
  internally consistent for the first time and that sentence goes.
- **`:975` "all within 0.025 macro F1"** no longer straddles model generations,
  since both variants are being retrained. Recheck the span against the new
  numbers rather than assuming it still holds.
- **The stratum regressions are reported alongside the overall gain** (author
  decision 2026-08-18), with the mechanism stated. Full-pool, uncalibrated:
  overall +0.0035, tail -0.0087, magnets -0.0071. Under the clamp the tail lands
  at 0.8875 within-stratum against the released model's 0.8928, while global
  per-language tail F1 rises to 0.7743 against 0.7655 with false positives into
  tail labels down to 8,727 from 22,522. Both views belong in the appendix; they
  disagree by construction (Exp 24).
- The existing editorial note about the UDHR-subset FPR of 1.06e-5 is unresolved
  and independent of this work.

---

## Gap found 2026-08-19: which calibration constants still need re-deriving

`tab:calibration_provenance` lists five selected components. Sorting them by
whether the special-token correction can move them:

| component | moves? | status |
|---|---|---|
| unseen-token constant $c$ | **yes**, it is an absolute target in log space | re-derived, $c = -17$ |
| thresholds $\tau_\lang$ | **yes**, percentiles of score margins | job 3123324 |
| high-entropy-group membership | **yes**, identified from predictions | not started |
| **proximity bound $21$** | **yes, and this is not tracked anywhere** | see below |
| $100{,}000$-sample requirement | no, a corpus-size property | unchanged |
| $18{,}000$-sample boundary (`head_n`) | no, a corpus-size property | unchanged |
| $q_\lang$ form, percentiles, rank cutoff | no, dimensionless | unchanged |

**The proximity bound is the gap.** It is a score difference in natural-log units:
a replacement candidate is accepted when the top candidate's score minus the
candidate's score is at most 21. Score differences between two languages are
exactly the quantity the correction moves, because each language segments a line
into a different number of tokens and each token gained $\log 5$. This is the
same mechanism that moved the per-language thresholds by up to 123% when it was
probed, and none of that reasoning was applied to the proximity bound.

It was selected by a grid search from 0.5 to 100 on the development part, and the
paper states that overall macro F1 on that part varies by less than 0.0003 across
bounds from roughly 15 to 35, so 21 is a representative value of a plateau rather
than a tuned optimum. **That flat plateau is the reason this may not matter, and
it is also the reason it has to be checked rather than assumed**: if the plateau
has moved or narrowed on the corrected model, 21 may no longer sit inside it.

Cost: the recorded grid search ran on the development part, so re-running it is
one pass over 18.0M lines, comparable to the other full-pool jobs.

---

## Open items needing an author call

- **`tab:lenbias-delta`'s basis**, the same question as `tab:lenbias-norm` but not
  yet resolved: the corrected predictions exclude the 250,000 validation lines
  while the published table used all 45,627,279. The golden-subset treatment
  applied to `lenbias-norm` is the obvious match.
