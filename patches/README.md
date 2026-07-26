# Local modifications to the nested repositories

The working tree contains two nested git repositories that this repository does not
track (see `.gitignore`): `UNILID/` and its submodule `UNILID/tokenizers/`. Both carry
local modifications that are required to reproduce the results in
`EXPERIMENTS_RESULTS.md`. Those modifications are recorded here as patches.

As of 2026-07-16 the same modifications are also committed and pushed to forks, which
are the preferred way to obtain them:

- https://github.com/cimeister/UNILID, branch `bias-scoring`, commit `1d26844`
  (based on `816189e`; also points the tokenizers submodule at the fork below)
- https://github.com/cimeister/tokenizers, branch `unilid-scorers`, commit `156f6c51`
  (based on `b1c1c736`)

The patches are retained so this repository stays self-contained.

- `unilid.patch`: against UNILID commit `816189e`
  (clone of https://github.com/Ahmetcanyvz/UNILID). Makes the `transformers` import in
  `unilid/api.py` lazy (so training does not clobber the custom Rust `tokenizers` build)
  and adds `predict_normalized` / `predict_normalized_batch` wrappers to
  `unilid/model_io.py` (Exp 2/5 length normalization).
- `tokenizers.patch`: against tokenizers commit `b1c1c736`
  (clone of https://github.com/Ahmetcanyvz/tokenizers). Adds to the Unigram model and the
  PyO3 bindings: `best_of_cached_weight_sets_normalized(_batch)` (Exp 2/5),
  `best_of_cached_weight_sets_biased(_batch)` (Exp 14 per-language bias), and
  `top_k_of_cached_weight_sets(_batch)` (Exp 14 learned-bias fit). Rebuild with
  `maturin develop --release` after applying (see `SETUP.md` gotcha 6).

To reproduce the working tree: clone each repository at the commit above, then
`git apply` the corresponding patch from inside it, e.g.
`git -C UNILID apply ../patches/unilid.patch`.

## sentencepiece_fp64_estep.patch (Exp 41, 2026-07-26; NOT applied)
Double-precision forward-backward for the fixed-vocab EM trainer path in the
sentencepiece fork (branch fixed-vocab-em). Diagnosis and verification in
EXPERIMENTS_RESULTS.md Exp 41: float32 accumulation breaks the E-step's
log-posterior identity on very long lines (the pipeline passes
--max_sentence_length=1000000, bypassing upstream's 4,192-byte cap), and the
fork's isfinite-to-zero M-step guard converts the overflow into a silent model
collapse (azj_Latn at 131k). The patch leaves inference paths untouched; an
unpatched rebuild reproduces the installed binary bit-for-bit. Adoption into the
fork is a pending user decision; if adopted, the recommended companion change is
a hard CHECK on non-finite expected counts, and the 33 Apertus-branch corpora
with lines above 10,000 characters need retraining. The minimal 390-line trigger
corpus is preserved at outputs/diagnostic/em_trigger_azj_81251_81640.txt.
