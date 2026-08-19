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

## A. Ready now

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
| 11 | `tab:lenbias-norm` | Original / Raw rescore / Normalized, overall `.960`/`.960`/`.885` | **Raw rescore `.961`, Normalized `.838`** (per-bin table in `outputs_corrected/tables/lenbias_norm.md`). Normalization is **more** damaging, so `:1247`'s conclusion strengthens. The Original column is omitted; see the open item below |

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

## Open items needing an author call

- **`tab:lenbias-norm`'s Original column and `tab:lenbias-delta`'s basis.** The
  corrected predictions exclude the 250,000 validation lines; the published
  tables used all 45,627,279. The Original column also supported an
  implementation check ("Raw rescore reproduces the original predictions
  exactly"), which can be partly recovered by comparing the $\alpha=0$ rescore
  against `pred_baseline.npy` on the test half.
