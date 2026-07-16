# Local modifications to the nested repositories

The working tree contains two nested git repositories that this repository does not
track (see `.gitignore`): `UNILID/` and its submodule `UNILID/tokenizers/`. Both carry
uncommitted local modifications that are required to reproduce the results in
`EXPERIMENTS_RESULTS.md`. Those modifications are recorded here as patches.

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
