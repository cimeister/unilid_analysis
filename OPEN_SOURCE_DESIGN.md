# Calibrated UniLID open-source release: API and artifact-format design

Written 2026-08-10 (release session, checklist step 1 of OPEN_SOURCE_HANDOFF.md).
Status: APPROVED by the user 2026-08-10 (overall design + three sub-decisions: v2
container, loud error on missing calibration, new fork binding for tokens).
IMPLEMENTED AND SHIPPED (merged upstream 2026-08-11); current state and post-design
additions (forward= integration behind a base-mode guard, lazy training imports,
language subsetting with carried-threshold semantics, the add-language worked
example, README/REPRODUCING restructure) are recorded in OPEN_SOURCE_STATUS.md,
which is the file to read first when resuming this effort.
Revision 1, same day: an adversarial plan-consistency review (Opus, per the standing
review-before-execution practice) returned 19 findings (3 blocking); all fixes are
applied in this revision and marked [R1] where substantive. None change the four
approved decisions. Companion to OPEN_SOURCE_HANDOFF.md; supersedes nothing.

## 1. Inputs

Three read-only reconnaissance passes over (a) the UNILID/ package at HEAD 1d26844,
(b) the reference calibration code in analysis/, and (c) the artifacts on store and in
outputs/diagnostic/. Findings that shape the design:

- The `.unilid` container is a positional binary: 32-byte header
  (magic "UNILID\0\0", version=1, num_langs, vocab_size, base_tok_len, langs_len,
  4 reserved bytes), then base-tokenizer JSON, langs JSON, then the raw
  float32[num_langs x vocab_size] weights. The loader memmaps exactly
  num_langs*vocab_size*4 bytes and ignores trailing bytes; it rejects files with
  version > its own. There is no metadata section.
- Commit 1d26844's per-language bias is a single scalar added to the whole-sentence
  score after the DP; it cannot express the unseen-token clamp. But the same commit
  chain (tokenizers fork commit 156f6c51) added
  `top_k_of_cached_weight_sets_batch(texts, k) -> [[(lang_idx, f32 score)] * k]`,
  which is exactly the primitive the re-examination gate needs (k=5).
- No exposed function returns the token segmentation under a *specified* language;
  only under the argmax (`best_of_cached_weight_sets*`). The Rust helper
  (`tokens_and_score_f32`) exists but has no per-language binding.
- The clamp of record (`analysis/floor_equalization.py build_equalized_weights`) is
  self-contained given the weight matrix: per language row, floor = row.min(); if
  floor > c, set row[row == floor] = c; fp32; no renormalization; rows already at or
  below c untouched. No unseen-token mask artifact is needed.
- The tau table of record (`outputs/diagnostic/tau_floor21_gate.csv`) was produced by
  `analysis/solo_gates.py` with the size-adaptive quantile
  q_L = 5 * (1 - min(N_L, 18000)/18000), computed on the clamped matrix via the Rust
  top-k path; `margin_diagnostic.py` (fixed q=5, unclamped matrix, N<1000 only) is the
  constants' origin but not the recipe of record. Percentile: `np.percentile` default
  linear interpolation. Sampling: one `np.random.default_rng(0)` shared sequentially
  across languages in CSV order, `rng.choice(n, 2000, replace=False)` then sorted.
- The cleanest reusable inference form is
  `analysis/external_bench_eval.py::_load_gate_thresholds` / `_gate_walk_and_merge`
  (Opus-reviewed; asserts group disjointness, tau-CSV set equality, excluded -> -inf,
  bookkeeping identities; fresh-copy two-step apply).
- PRECISION CORRECTION TO THE HANDOFF: the promoted analysis chain is float32 end to
  end for the matrix, clamp, banked top-5 scores, margins, and proximity comparisons
  (`external_bench_eval.py:723-730` and `:999-1002` guard the dtype explicitly). The
  handoff's "analysis chain scored with an fp64 clamped matrix" does not match the
  code; fp64 appears only in the Mistral-Nemo training runs and auxiliary statistics
  (N arrays, tau floats). The package will therefore use the same fp32 arithmetic,
  and the residual disagreement risk vs the reference predictions comes from
  summation order / code-path differences, not a dtype mismatch.
- Package defects found (fixed on the release branch as part of this work):
  both `[project.scripts]` entry points point at functions that do not exist
  (`unilid.model_io:_cli_convert`, `unilid.cli:eval_main`); the assembly-time
  default for tokens missing from a language's file is an unnamed `-1e30` local
  duplicated in three places (model_io.py x2, eval.py); pyproject declares
  `license = MIT` while the repo has no LICENSE file and the proposal is Apache-2.0.
- The paper team's eval scripts (unilid_resources/) call
  `predict_batch(texts, forward=...)`; the package's `predict_batch` accepts no such
  argument (TypeError as written). The bundled benchmark label-list .txt files are
  read by no script. Both flagged; neither is a blocker for this design.

## 2. Decisions taken by the handoff (not revisited here)

Calibration default ON with a base flag; fork-then-PR on cimeister/UNILID; weights to
HF Hub only after the licensing check and explicit OK; Apache-2.0 proposed pending
co-author sign-off; every external action gated.

## 3. API design

### 3.1 Loading

```python
from unilid import UnilidModel

model = UnilidModel("unilid-1940-calibrated.unilid")      # calibrated (default)
model = UnilidModel("unilid-1940-calibrated.unilid", calibrated=False)  # base
model = UnilidModel("model.unilid", calibration="cal.json")  # explicit sidecar
```

- `UnilidModel(model_path, calibrated: bool = True, calibration: str | None = None)`.
- `calibrated=True` (default) requires calibration artifacts: either bundled in the
  container (v2 file, section 4) or supplied via `calibration=`. If neither is
  present, raise `UnilidCalibrationError` naming the missing artifact and the two
  remedies (`calibrated=False`, or supply/derive a calibration). No auto-downgrade.
- `calibrated=False` reproduces today's behavior exactly (same code path; the clamp
  is never applied, the gate never runs).
- `model.calibration` exposes the loaded `Calibration` object (or `None`).
- Loader checks (all loud, mirroring `_load_gate_thresholds`): group A language set
  == {lang : N_L < 18000} from the bundled train counts; group B members all have
  N_L >= 18000; groups disjoint; excluded entries map to tau = -inf; every group
  language present in the model's language list.

### 3.2 Prediction

`predict(text)` and `predict_batch(texts)` keep their exact signatures and return
types: `(lang, tokens, score)` per text, `(None, [], -inf)` for texts that
preprocess to empty. Under `calibrated=True`:

1. At load, the clamp is applied to the fp32 weight matrix (port of
   `build_equalized_weights`, identical semantics) before `set_weight_sets`.
2. Scoring runs `top_k_of_cached_weight_sets_batch(texts, k=5)`.
3. Per text: base prediction = rank-1. If its language is in group A use tau_A, else
   if in group B use tau_B, else no re-examination. If (s1 - s2) < tau, walk ranks
   2..5: first candidate with N >= 100000 and (s1 - s_j) <= 21.0 replaces the
   prediction; otherwise it is unchanged. Per-row logic is exactly equivalent to the
   reference's fresh-copy two-step apply (review-verified: group membership is read
   from rank-1 only; the walk reads only ids/scores/N; the groups are disjoint; and
   no re-examined language has N >= 100,000, so a walk can never land inside either
   group); the reference's assertions (disjoint sets, bookkeeping identity) are kept.
   [R1] Comparison dtypes are pinned to the reference: tau stays float64 and the
   gate compares float(gap) < tau in float64 (the reference's array comparison
   widens the fp32 gap losslessly); the proximity comparison stays float32 (as in
   `_walk_replacement`).
4. `lang` is the final language; `tokens` and `score` come from the segmentation
   pass under the final language (section 3.4), so the calibrated score has the
   same definition as the base path's (`tokens_and_score_f32`), not the top-k
   scoring loop's. [R1]
5. [R1] Known tie-break difference vs the reference base predictions: `best_of`
   breaks exact score ties by first index (strict >), `top_k` sorts unstably; the
   full-test build measured 1 affected row in 2,236,864 (~5e-7). This is a listed
   allowed cause in the agreement-gate forensics (section 5), not a bug to chase.

Constants (c=-21, HEAD_N=18000, RES_CAP=100000, D3_PROX=21.0, TOPK=5,
CALIB_MAX=2000, MIN_CALIB_LINES=200, CALIB_SEED=0) are read from the calibration
artifact, not hardcoded, so a custom deployment can state its own RES_CAP (the
uncapped-corpus caveat from the handoff is documented at that field).

### 3.3 New module `unilid/calibration.py`

- `Calibration` (dataclass): constants, per-language group tables (tau, excluded,
  cause, n_scoreable, n_self_won), high-entropy member list, train_counts,
  provenance. `Calibration.from_json / to_json`, `from_bundle / to_bundle`.
- `apply_unseen_token_constant(W, c) -> (W_cal, n_modified)`: the clamp port, fp32,
  raises on non-finite output (as the reference does).
- `re_examine(topk_ids, topk_scores, calibration) -> final_ids + stats`: the gate +
  walk port (from `_gate_walk_and_merge`, minus the replay-only agree_mask).
- `estimate_tau(model, lang, lines, n_l) -> TauRow` (implemented signature; the
  constants come from `model.calibration` rather than a separate argument, so
  they cannot disagree with the matrix the model was clamped with): the
  per-language threshold recipe of record, ported from `solo_gates.py` with BOTH
  exclusion branches [R1]: top-k pass over up to CALIB_MAX own lines (sampled with
  `np.random.default_rng(calib_seed).choice(n, CALIB_MAX, replace=False)`, sorted),
  wins filter top1==lang, finite margins only; `low_calibration` if fewer than
  MIN_CALIB_LINES finite own-won margins, `zero_strength` if
  q_L = margin_q*(1 - min(N_L, head_n)/head_n) <= 0 (low_calibration wins when both
  apply); excluded either way (tau = -inf, never re-examined); else
  `np.percentile(gaps, q_L)` (numpy default linear interpolation). `n_self_won`
  counts wins BEFORE the finiteness filter (documented; the reference CSV does the
  same). [R1] `estimate_tau` REQUIRES the model's cached weights to be the clamped
  matrix (margins are defined on the distributions with the unseen-token constant
  applied); it asserts this via the caller passing a calibrated-loaded model, and
  refuses to run on a base-loaded one.

### 3.4 Token segmentation under the final language

`top_k` returns no tokens, so the calibrated path needs one of:

- (a) RECOMMENDED (approved and implemented): add one PyO3 binding to the
  tokenizers fork (we own it; precedent: commit 156f6c51):
  `tokens_of_cached_weight_set_batch(texts, indices) -> [(tokens, score)]`, a thin
  wrapper over the existing Rust `tokens_and_score_f32`. Work per text: one
  scoring pass over all languages plus one segmentation pass for the chosen
  language, like today's `best_of`, plus one extra DP-graph preparation per text
  (prepare_dp runs once in the top-k call and once in the tokens call; a fused
  call could remove this later).
- (b) Python-side Viterbi (`unilid/algorithms/viterbi.py`) for the final language.
  No Rust change, but a slow pure-Python pass per text.
- (c) Return `tokens=[]` in calibrated mode. Smallest change, breaks the contract.

### 3.5 Customization: `add_language()` and CLI

`unilid.add_language(model_path, lang, train_file, output_path, *, method="sp"|"em",
em_rounds=20)` and console script `unilid-add-language`:

1. Extract the base tokenizer from the container; train the new language's
   distribution over the existing vocabulary
   (`LanguageSpecificUnigramLMTokenizer`, 20 EM rounds, floor 1e-12). [R1] Default
   `method="sp"` (the forked-SentencePiece path with soft re-estimation), which is
   the E3-verified end-to-end recipe; the pure-Python `EMUnigramTrainer` path
   ("em") ships documented as unverified against the E3 chain. "sp" needs the built
   spm binary; its absence is a loud error naming the binary, never a silent method
   switch.
2. [R1] Rebuild the container from an EXPLICIT ordered language list: the existing
   num_langs rows in their existing order (copied from the loaded matrix,
   byte-identical), the new language appended at index num_langs. `save_unilid`'s
   filename-glob discovery is NOT used here (its soft-before-sp glob preference and
   alphabetical sort would silently drop or reorder languages).
3. [R1] Post-training guards on the new row, all loud: abort if any entry is at or
   below -1e29 (the assembly fill for base-vocab tokens missing from the trained
   file; means the training did not cover the vocabulary); abort if the row is
   non-finite; then apply the clamp per the reference semantics and REPORT the
   outcome explicitly: clamped (plateau lowered to c), or left unclamped because
   row.min() <= c already (the reference leaves such rows entirely untouched; the
   trainer floor log(1e-12) = -27.63 lies below c = -21, so this case is expected
   for some training regimes and must be printed, not silent).
4. Update the calibration with only the new language's data: N_L = its training-line
   count; if N_L < 18000, run `estimate_tau` (both exclusion branches) against the
   full new model loaded calibrated; if N_L >= 18000, no threshold; the language
   becomes a replacement candidate iff N_L >= RES_CAP. [R1] Two documented
   approximations, stated as such (not as identities): existing languages' taus are
   kept although the new language changes the margin distribution they were
   estimated on (a full re-derivation would re-run the tau pass for all 1,080+4
   languages); and a new language with N_L >= RES_CAP becomes a replacement
   candidate for every gated language without any recalibration of those languages.
   High-entropy-group membership is NOT recomputed (needs cross-language statistics
   and a validation scoring pass); the docs state this as the one non-incremental
   piece, with the criterion quoted from the paper.

Also `unilid-calibrate`: (re)estimate tau for named languages of an existing model
from their training files and write/update the calibration artifact. This plus
add_language covers the E3-verified customization recipe; full from-scratch
derivation including high-entropy identification stays in the analysis repo for v1.

## 4. Artifact format

### 4.1 Recommended: `.unilid` container version 2 (bundled calibration)

```
Header (32 bytes, unchanged struct "<8sIIIII4x"): magic, version=2, num_langs,
  vocab_size, base_tok_len, langs_len, reserved
Body (unchanged): base-tokenizer JSON, langs JSON, float32 weights
v2 trailer: cal_len (uint64 LE), calibration JSON (utf-8, cal_len bytes)
```

- Weights stored are the BASE (unclamped) matrix; the clamp is applied at load when
  calibrated. One canonical matrix serves both modes.
- `save_unilid(model_dir, output_path, calibration=None)`: without a calibration it
  writes version 1 byte-identical to today; with one it writes version 2. [R1] Two
  distinct constants: `FORMAT_VERSION_MAX = 2` (reader's acceptance bound) and the
  per-write version (1 or 2 depending on calibration presence); the current single
  `__version__` doing both jobs would silently change every v1 write when bumped.
- Old readers see version 2 > 1 and fail loudly ("model file version 2; this unilid
  version reads <= 1; upgrade"). This is deliberate: the alternative (version 1 +
  trailing section) lets an old package silently return base predictions from a
  file advertised with calibrated numbers, which is a silent behavioral fallback.
- New readers accept v1 (no calibration; `calibrated=True` then errors per 3.1) and
  v2 (trailer required; truncated trailer is a loud corruption error).
- `unpack_unilid` additionally writes `calibration.json` beside the tokenizers.

Alternative considered (sidecar JSON next to a v1 file): keeps old readers working
on the same file, but the model and its calibration can separate in transit, and
"out of the box" then depends on users downloading two files. Rejected as default;
the standalone JSON remains supported via the `calibration=` load argument and
`unpack_unilid`/`unilid-calibrate` for transparency and custom workflows.

### 4.2 Calibration JSON schema (format_version 1)

```json
{
  "format_version": 1,
  "constants": {"unseen_token_constant": -21.0, "head_n": 18000,
                 "replacement_min_n": 100000, "proximity_bound": 21.0,
                 "topk": 5, "margin_q": 5.0, "group_b_percentile": 5.0,
                 "calib_max": 2000, "min_calib_lines": 200, "calib_seed": 0},
  "group_a": {"criterion": "predicted language has fewer than head_n training samples",
               "thresholds": {"<lang>": {"tau": 7.310638256072998, "excluded": false,
                                          "cause": "", "n_scoreable": 1335,
                                          "n_self_won": 1316}, ...}},
  "group_b": {"criterion": "the high-entropy group (identification criteria in the paper, app:protocol)",
               "thresholds": {"<lang>": {...same row shape...}, ...}},
  "train_counts": {"<lang>": 6735, ...},
  "provenance": {"source_model_sha256": "...", "clamped_matrix_sha256": "...",
                  "tau_csv_sha256": {"group_a": "...", "group_b": "..."},
                  "derived": "analysis repo, gate_flat4_prox21, promoted 2026-08-06"}
}
```

- [R1] `margin_q` (the 5 in q_L = margin_q*(1 - min(N_L, head_n)/head_n)) and
  `group_b_percentile` (the fixed 5th percentile) are in the constants block; no
  literal 5 is hardcoded in the recipe code.
- [R1] Keys are `group_a` / `group_b` (the reference code's own names), each with a
  `criterion` field spelling out the paper's wording; no coined labels.
- JSON has no -inf: excluded rows store `"tau": null, "excluded": true`; the loader
  maps excluded/missing to -inf (exactly `_load_tau_csv`'s forced -inf semantics).
  [R1] Loader validations beyond section 3.1: duplicate language keys rejected
  (JSON objects cannot express them, but the CSV-to-JSON transcriber checks);
  `excluded` must be a JSON boolean; a row with `excluded: false` and `tau: null`
  is rejected (the CSV analogue is `_load_tau_csv`'s bool-dtype guard).
- [R1] Tau values are transcribed at full float64 round-trip precision
  (`repr(float)`), never rounded: the release gate re-verifies value-level equality
  against the repo CSVs, and a rounded tau shifts gated predictions. The CSV's
  empty `cause` reads as NaN in pandas; the transcriber maps NaN to "".
- Group names in code/docs follow the paper: "languages with fewer than 18,000
  training samples" (group A) and "the high-entropy group" (group B). No new terms.
- For the released model the two tables are transcribed from
  `outputs/diagnostic/tau_floor21_gate.csv` (1080 rows, 26 excluded) and
  `tau_flat4.csv` (4 rows); `train_counts` from `glotlid_train_counts.json`
  (now durable on store); provenance records each source CSV's sha256, and the
  release gate re-verifies value-level equality against the repo CSVs.

## 5. Precision policy (explicit, per the handoff's blocking requirement)

- Arithmetic of record: float32 for weights, clamp, scores, margins, and the
  proximity comparison, matching the reference chain exactly. Threshold comparison
  in float64 as in the reference (section 3.2 item 3).
- [R1] Golden subset: the TEST HALF of the seed-42 500k draw, i.e.
  `sample_idx[(np.arange(500000) % 2) == 1]`, 250,000 lines. The even-parity half
  is the excluded validation sample and is masked to negative values in the
  reference arrays (comparing on it fails by construction). The gate asserts
  `reference >= 0` on every compared line before comparing.
- [R1] Base gate (blocking): package-base vs `pred_baseline.npy` on the golden
  subset, EXACT equality. Precedent: the package file is the identical code path
  the analysis chain loaded (`transfer_sweep._load_unilid_model` execs the
  package's model_io.py), `solo_gates` enforces 0.9999 for the unmodified path,
  and the recovered Drive prediction file agreed with `pred_baseline.npy` at
  1.000000. Any disagreement is investigated, not tolerated.
- Calibrated gate (blocking): package-calibrated vs `pred_gate_flat4_prox21.npy`
  on the golden subset, >= 99.9% agreement, AND every disagreeing line
  individually shown to be a boundary case: top-1/top-2 scores within eps, or
  |gap - tau| <= eps, or |(s1 - s_j) - 21.0| <= eps, or [R1] an exact rank-1/rank-2
  score tie resolved differently by `best_of` (first-index) vs `top_k` (unstable
  sort), the one structural difference between the package path and the reference
  build (measured rate ~5e-7). eps chosen from the observed disagreement
  distribution and recorded (starting point 1e-3 nats). Any disagreement not
  explained by a listed cause is a blocking failure, not a gate-tuning exercise.
- [R1] Provenance note: the handoff's recorded 99.51% self-agreement was the
  paper-era scoring run (sample_500k_all.pkl) vs the current run, not an
  fp32-vs-fp64 artifact; the fp64 attribution in the handoff is corrected by this
  design (section 1), and the recovered prediction file agrees with
  `pred_baseline.npy` at 1.000000.
- No bit-exactness chase across environments; the policy above is the standard.

## 6. Tests (checklist step 3)

Unit: clamp (one-sided, plateau-only, no renorm, n_modified, fp32, non-finite
raise, row.min() <= c untouched); calibration loader (excluded -> -inf, missing
lang -> -inf, set-equality and disjointness assertions fire, [R1] non-bool
excluded rejected, excluded:false with tau:null rejected); estimate_tau ([R1] both
exclusion branches and cause precedence, clamped-matrix requirement enforced);
walk (rank order 2..5, RES_CAP, proximity, unfilled-slot skip,
no-candidate-unchanged, blocked-vs-nocand accounting, [R1] float64 tau comparison
vs float32 proximity comparison); excluded-language passthrough; container v1/v2
round trip, v1 trailing-bytes tolerance, old-reader-style rejection of v2,
truncated-trailer error, [R1] v1 writes byte-identical with FORMAT_VERSION_MAX=2;
synthetic add-language round trip (tiny vocab, 3+1 languages: [R1] existing rows
byte-identical AND in original order with the new language at the last index, tau
estimated or excluded by line count, counts updated, [R1] the -1e29 assembly-fill
guard and the unclamped-row report both exercised).
Golden gates: section 5, run on the login node via the package's Rust batch path.

## 7. Docs, packaging, branch plan (checklist steps 4-5, summarized)

- Branch `calibration-release` off `release`, developed on the fork (push OK;
  PR/LICENSE/HF each gated on explicit user OK).
- README rewrite: install (submodule + maturin story, spm optional), quick start
  (calibrated default), model download, the calibration trade-off with both numbers
  (GlotLID-C macro F1 .929 -> .957; UDHR .859 -> .838 on parallel equal-support
  data; the corrected generation published 2026-08-24 reads .933 -> .956 and
  .856 -> .842, and the package docs now carry those), base-flag documentation, add-your-own-language guide, eval conventions
  copied from the paper's stated definitions. [R1] The README and model card must
  both state the migration consequence explicitly: the new package default
  (calibrated=True) raises on the existing v1 polybox file (users either download
  the calibrated v2 bundle or pass calibrated=False), and evaluation scripts that
  reproduce the paper's base rows must pass calibrated=False. HF model card
  mirrors the same numbers (prepared, not uploaded).
- pyproject: version 0.1.0 -> 0.2.0; fix the two broken console scripts; add
  `unilid-add-language` / `unilid-calibrate`; the `license = MIT` metadata is
  corrected on the PR branch alongside the proposed Apache-2.0 LICENSE, both marked
  pending co-author sign-off (metadata currently contradicts the repo, which has no
  license file at all).
- New named constant replacing the triple-defined `-1e30` assembly default in
  `unilid/constants.py` (explicitly flagged here per the magic-number rule).

## 8. Decision points presented to the user

1. Overall design approval (sections 3-7).
2. Container format: v2 version bump (recommended) vs v1 + trailing section vs
   sidecar-only.
3. Behavior when `calibrated=True` (default) finds no calibration: loud error
   (recommended) vs auto-downgrade with a warning.
4. Tokens under the final language: new fork binding (recommended) vs Python
   Viterbi fallback vs dropping tokens in calibrated mode.
