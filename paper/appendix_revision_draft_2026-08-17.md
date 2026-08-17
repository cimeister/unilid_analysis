# Draft appendix revisions after the special-token correction

Not applied to `submission.tex`. Two reasons: your own uncommitted edits to that
file are in the working tree, and most of the numbers are still being
regenerated. This is the wording sample for sign-off before anything is edited in
place.

Only item 1 is ready now. Items 2 and 3 depend on runs that have not finished and
are listed so they are not lost.

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

### Proposed replacement

> Without this correction, that value is the smallest value $\log\unigramdistlang$
> takes, and it depends on the size of $\corpus^\lang$: across the 1{,}940
> languages it falls by 2.04 nats per decade of training tokens (correlation
> $-0.99$ between the value and $\log_{10}$ of the language's training-token
> count, $R^2 = 0.99$), and subsampling one language's corpus over two decades
> reproduces the same slope, so the dependence is on corpus size and not on which
> languages happen to have small corpora. It is not a byproduct of the
> training-time probability floor of $10^{-12}$: no language's value reaches that
> floor, the closest being 7.7 nats above it. Since \cref{eq:lang-id} compares
> scores across languages, the variation acts as a language-dependent offset on
> every unseen token.

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
  $-2.196$, $-2.196$, $-2.184$ nats per decade of training tokens at $R^2 = 0.999$,
  against $-2.039$ across languages.
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

## 2. BLOCKED on the corrected Mistral-Nemo chain: a claim that reverses (`submission.tex:1383-1384`)

> its own unseen-token treatment (two languages whose trained unseen-token values
> already lie below $c = -21$ are left unchanged)

Measured on the stored files: the **released** Mistral-Nemo variant has exactly
two such rows, `khm_Khmr` ($-21.232$) and `ory_Orya` ($-21.016$); the
**corrected** variant has none, because the correction raises every real token by
1.6094 nats. So the parenthetical reverses, and after correction the variant
behaves like the base model, where every unseen-token value exceeds $c$ and all
are set to $c$.

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
