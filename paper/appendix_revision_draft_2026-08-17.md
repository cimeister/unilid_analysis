# Draft appendix revisions after the special-token correction

> The concrete, itemized edit list is `paper/PAPER_EDITS_pending.md`. This file
> holds the proposed *wording* for the items that are prose rewrites rather than
> number substitutions.
>
> An earlier item 1c proposed disclosing that the sweep did not separate the two
> central grid values. **Removed 2026-08-18**: the preprint is unreleased, so
> there is no prior constant to reconcile against and no disclosure is owed. The
> paper states the procedure and the constant it selected.

The ready items were applied to `submission.tex` on 2026-08-19, each wrapped in
`\corrrev{}`. The wording below is the record of what went in and why.

Items 1, 1b and 1d are APPLIED (commit `6374b67`). Items 2 and 3 depend on runs
that have not finished and are listed so they are not lost.

---

## 1. READY: the unseen-token value's origin (`submission.tex:629-631`)

The current sentence attributes the value to the training floor. That is wrong,
and it is the one paper claim the special-token work proved false outright rather
than merely shifting.

### Current text

> Without this correction, that value is a byproduct of the training-time
> probability floor of $10^{-12}$ and renormalization, and it varies with
> $|\corpus^\lang|$; since \cref{eq:lang-id} compares scores across languages, the
> variation acts as a language-dependent offset on every unseen token.

### Applied 2026-08-19 (commit `6374b67`, revised for style)

> Without this correction, that value is the smallest value of
> $\log\unigramdistlang$ over $\vocab$, and it varies with the size of
> $\corpus^\lang$. Across the 1{,}940 languages, the value decreases by 2.04 nats
> for each tenfold increase in the number of training tokens, and the correlation
> between the value and $\log_{10}$ of that token count is $-0.99$. For each of
> three languages whose corpora were subsampled across two orders of magnitude,
> the same decrease per tenfold increase is measured within that one language, so
> the quantity that determines the value is corpus size and not the identity of
> the language. The training-time probability floor of $10^{-12}$ does not
> determine it: every value lies above that floor, and the smallest distance to
> it is 7.7 nats.

An earlier draft of this passage used the phrase "nats per decade of training
tokens", which is coined shorthand rather than a defined term, and gave processes
and values agency ("subsampling reproduces the same slope", "no language's value
reaches that floor"). It also quoted both the correlation and $R^2$, which are
redundant for a simple linear fit. Corrected on author instruction 2026-08-19.

### What changed and why

- **Removed** the attribution to the training floor and renormalization. In the
  SentencePiece training path `MIN_TOKEN_LOG_PROB` is assigned to the special
  tokens and to nothing else; the separate safety clamp applies to every token but
  never binds. Every observed value sits between $-19.94$ and $-13.22$, that is
  7.7 to 14.4 nats above $\log 10^{-12} = -27.63$.
- **Kept** "it varies with $|\corpus^\lang|$", which was already correct, and
  quantified it.
- **Added** the within-language result, which is what upgrades this from a
  correlation to a statement about corpus size. The cross-language correlation is
  measured over 1,940 different languages, so corpus size and language identity
  are confounded in it. Holding language identity fixed and subsampling
  (`abk_Cyrl`, `mam_Latn`, `zul_Latn`, 1,000 to 100,000 lines) gives slopes
  slopes of $-2.196$, $-2.196$ and $-2.184$ nats for each tenfold increase in the
  training-token count, against $-2.039$ across languages.
- **Left out** the underlying scaling law. The slope corresponds to the value
  scaling as $T^{-0.95}$ in the training-token count $T$, approximately one count
  in $T$, which is what an unsmoothed maximum-likelihood fit gives a type it never
  effectively observes. It is a cleaner statement but goes further than the
  measurement licenses without a derivation, so it is offered as an option rather
  than included.

Artifacts: `outputs/rerelease/plateau_reference_fit.json`,
`outputs/rerelease/plateau_vs_corpus_size.json`;
`analysis/plateau_reference_fit.py`, `analysis/plateau_vs_corpus_size.py`.

---

## 1b. READY: "all of them are set to $c$" becomes false (`submission.tex:627-628`)

### Current text

> for the base \unilid model every unseen-token value exceeds $c$, so all of them
> are set to $c$.

### Why it changes

True for the released model at $c = -21$: all 1,940 rows have their unseen-token
value above $-21$, so all 1,940 are clamped. **False for the corrected model at
its own selected $c = -17$**: 1,655 of 1,940 rows are clamped and 285 already lie
below the target, so they are unchanged.

### Applied 2026-08-19 (commit `6374b67`, revised for style)

> for the base \unilid model 1{,}655 of the 1{,}940 unseen-token values lie above
> $c$ and are set to $c$; the remaining 285 already lie below $c$ and are
> unchanged, because the minimum above never raises a value.

An earlier draft ended "since the rule only ever lowers", which gives a rule
agency; the applied text names the operation instead. The counts also come from
the round-grid sweep's $c = -17$, not the shifted grid's $-17.3906$.

Note this makes the base model behave like the Mistral-Nemo variant described at
`:1383-1384`, where some rows already lie below $c$. The two passages should be
read together for consistency once both are settled.

**Code consequence, already applied.** The chain asserted `n_mod == n_lang`,
which encoded the incidental fact that at $c = -21$ every released row moved.
That assertion is replaced by
`analysis.floor_equalization.verify_one_sided_clamp`, which checks the property
that actually has to hold: no row was skipped that should have been lowered.

---

## 1d. READY: the full-pool effect of the correction (`submission.tex:344` and the stratum tables)

Uncalibrated, full pool, 45,377,279 lines:

| stratum | released | corrected | delta |
|---|---|---|---|
| overall macro F1 | 0.9292 | 0.9327 | +0.0035 |
| overall accuracy | 0.9608 | 0.9609 | +0.0001 |
| tail | 0.9132 | 0.9045 | -0.0087 |
| magnets | 0.9138 | 0.9067 | -0.0071 |
| twins | 0.9167 | 0.9164 | -0.0003 |
| head | 0.9602 | 0.9596 | -0.0006 |

`:344` quotes macro F1 .929, which becomes .933.

Per the author decision of 2026-08-18 the stratum regressions are reported
alongside the overall gain, with the mechanism stated: the stratum rows are the
within-stratum recall view and exclude false positives into tail labels, so a
falling tail figure means examples truly written in tail languages are
misclassified more often. This is the same decomposition Exp 20 recorded for the
floor-21 clamp, where an overall gain from global precision sat alongside a
recall-side loss on the tail.

---

## 2. BLOCKED on the corrected Mistral-Nemo chain: a claim that reverses (`submission.tex:1383-1384`)

> its own unseen-token treatment (two languages whose trained unseen-token values
> already lie below $c = -21$ are left unchanged)

Measured on the stored files: the **released** Mistral-Nemo variant has exactly
two such rows, `khm_Khmr` ($-21.232$) and `ory_Orya` ($-21.016$); the
**corrected** variant has none *at $c = -21$*, because the correction raises
every real token by 1.6094 nats. So the parenthetical reverses at that constant.
Whether it reverses at the variant's own re-derived $c$ is not yet known: the base
model's re-derived $c = -17$ leaves 285 of its 1,940 rows below the target
(item 1b), so a re-derived constant for the variant may leave some of its rows
unclamped too. The count has to be measured against the variant's own
constant, not assumed from this one.

The final wording waits on the corrected variant's own re-derived $c$, since the
count is a comparison against it. The same paragraph names Banjar, Scots and
Serbian in Latin script as the variant's high-entropy group; that identification
is made from predictions and has to be re-run.

**Sweep the appendix for every other count and every directional claim about
unseen-token values before signing off**, not only this one.

---

## 3. BLOCKED on the regeneration: numeric substitutions

Prose sites quoting UniLID numbers, to be replaced from the re-runs:
`submission.tex:344` (macro F1 .929 to .957), `:824` (FPR against fastText),
`:833`, `:835`, `:850-851` (UDHR and FLORES). The abstract's `\camrev` sentence
carrying the same figures is currently removed in the working tree; whether it
returns is an editorial choice.

`:975` claims vocabulary-sensitivity results are "all within 0.025 macro F1".
That comparison will straddle two model generations once the DeepSeek3.2 and
Qwen3 rows stay on pre-correction weights, so it needs rewording rather than a
new number.
