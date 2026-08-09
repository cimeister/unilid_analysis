# Open-source release handoff (calibrated UniLID)

Written 2026-08-09 for a fresh session tasked with open-sourcing a working, customizable
UniLID system. This file is the source of truth for that task: goal, user decisions,
where every needed piece lives, the exact mechanism spec, verification gates, and the
release checklist. Internal doc; the paper (paper/submission.tex, app:protocol) is the
public-facing spec of record.

## Goal

Ship a version of UniLID that (a) works out of the box: install, download the model,
predict; and (b) is easy to customize: users can add their own languages/data without
retraining anything else, including the calibration for the new language.

## User decisions (2026-08-09, binding)

1. **Target**: develop on the fork `github.com/cimeister/UNILID`; open a PR to
   `github.com/Ahmetcanyvz/UNILID` (the repo the paper links) when ready. Do not push to
   origin directly.
2. **Calibration ships DEFAULT ON**: calibrated inference is the default; base
   (uncalibrated) behavior available via a flag. This changes results for existing users
   of the released model; the README must say so and document the trade-off (GlotLID-C
   macro F1 .929 -> .957; UDHR .859 -> .838 on parallel equal-support data).
3. **Weights on HuggingFace Hub**, polybox kept as mirror. BLOCKER before upload: verify
   the licensing of weights derived from the GlotLID-C corpus (the corpus aggregates
   sources with their own licenses; check the GlotLID-C release terms and ask the user
   if unclear). Until verified, prepare everything but do not upload.
4. **License: propose Apache-2.0** in the PR. The repo currently has NO license file.
   Ahmetcan's sign-off is REQUIRED before the license (or the PR) is published as final.

Standing conduct rules: the user's global CLAUDE.md applies in full (clarify before
non-trivial ambiguity, no silent fallbacks, fail loudly). External actions gated:
pushing branches to the fork is fine; creating the PR, uploading to HF, or adding a
LICENSE file to a published branch each need explicit user OK first.

## The two codebases

**`UNILID/` (checkout at /users/cmeister747/unilid_analysis/UNILID)** is the package to
extend. Remotes: `origin` = Ahmetcanyvz/UNILID, `fork` = cimeister/UNILID. Layout:
`unilid/` (api.py, model_io.py `UnilidModel`/`save_unilid`, trainers/ incl.
`language_specific_trainer.LanguageSpecificUnigramLMTokenizer`, algorithms/,
constants.py `SPECIAL_TOKENS`), top-level `train.py`, `eval.py`, `convert.py`,
`pyproject.toml`, README.md (has quick start + polybox link for unilid-1940), and two
submodules (`tokenizers` -> points at the user's fork, `sentencepiece`). Rust-backed
batch inference (`model.model.best_of_cached_weight_sets_batch`,
`top_k_of_cached_weight_sets_batch`); commit 1d26844 already added "per-language bias
scoring support", which may be usable groundwork for the clamp; read it before writing
new scoring code. NO LICENSE file. NO calibration code.

**This analysis repo** holds the reference implementations and artifacts of record for
the calibration (next section). The eval scripts from the paper team are in
`unilid_resources/` (show the intended user-facing API surface: `UnilidModel(path)`,
`.predict_batch(texts, forward=False)`, label lists per benchmark).

## Calibrated UniLID: the exact mechanism, constants, and reference code

The public spec is paper/submission.tex sections 4 (Calibrated UniLID paragraph) and
app:protocol. Repo-internal spec: EXPERIMENTAL_SETUP.md (promoted configuration
gate_flat4_prox21). Constants and reference code:

| Piece | Value | Reference code | Artifact |
|---|---|---|---|
| Unseen-token constant | c = -21 (natural log); one-sided clamp min(trained, c) applied to each language's unseen-token log-probs; NO renormalization | analysis/floor_equalization.py `build_equalized_weights` | fingerprint_floor21.json (sha of the clamped matrix) |
| Group A (re-examined) | languages with N < 18,000 training lines (HEAD_N, analysis/full_test_margin.py:36); 1,080 languages in the released model | gate application: analysis/gate_variants.py | outputs/diagnostic/tau_floor21_gate.csv (1,080 rows; 26 rows excluded=True, cause low_calibration, tau=-inf, never re-examined) |
| Group B (re-examined) | the four high-entropy languages sco_Latn, bjn_Latn, arg_Latn, vls_Latn; fixed 5th percentile | membership rule: analysis/diagnostic.py (ZH_MAGNET=1.5, ZH_EXTREME=5.0, MAGNET_RATIO_MIN=2.0, z of entropy within script via median/MAD; restricted to N >= 18,000) | outputs/diagnostic/tau_flat4.csv |
| Threshold tau_L | q_L-th percentile, q_L = 5(1 - N_L/18,000), of margins (top score minus runner-up, clamped matrix) on up to CALIB_MAX=2,000 of the language's own training lines where it is top-scoring; MIN_CALIB_LINES=200 else no threshold; CALIB_SEED=0 | analysis/margin_diagnostic.py:49-53 | same CSVs |
| Re-assignment walk | candidates rank 2..5 (TOPK_MARGIN=5); accept first with N >= RES_CAP=100,000 AND score within D3_PROX=21.0 nats of top-1; else prediction unchanged | analysis/gate_variants.py `_walk_replacement` (pure, dataset-agnostic) and the two-step disjoint-group apply semantics (`_flat4_prox21_two_step_pred`); cleanest reusable form: analysis/external_bench_eval.py `_load_gate_thresholds` / `_gate_walk_and_merge` (Opus-reviewed, used for UDHR/FLORES/CommonLID) | |

Facts that must survive into the package docs: the two groups are disjoint; each starts
from a fresh copy of the clamped-matrix prediction; RES_CAP equals the per-language
training-data cap of the released model's corpus (it admits exactly the 282 capped
languages); the clamp is one-sided (a language whose trained unseen value is already
below c is left unchanged; happens for 2 languages in the Mistral-Nemo variant, 0 in
the base model); margins and the walk operate on the clamped matrix, not the base one.

## Models and artifacts (all durable on store)

- Base model: /capstor/store/cscs/swissai/a0229/cmeister/unilid_analysis/glotlidc.unilid
  (779,503,019 bytes, float32, 1,940 languages x 100k vocab; scratch symlink at
  /capstor/scratch/cscs/cmeister747/unilid_analysis/glotlidc.unilid). Reader:
  analysis/transfer_sweep.py `_load_model_data` or the package's unilid.model_io.
- Calibration artifacts for the base model: the two tau CSVs above + the flat-four list
  + c and the constants table. These are small; ship them WITH the model (either bundled
  into the .unilid container via unilid.model_io `save_unilid`, or as a sidecar JSON in
  the HF repo; bundling is cleaner for out-of-box use; the format choice is the new
  session's design call, presented to the user).
- Optional second release, the Mistral-Nemo-vocabulary variant (calibration re-derived
  from scratch on a new vocab; proof the customization recipe works):
  glotlid_mistralnemo_fp64.unilid + its own tau CSVs + flat set (bjn/sco/srp) under
  results_mistralnemo/ and outputs/diagnostic/ (see EXPERIMENTS_RESULTS.md
  "Camera-ready E3"). NOTE: these E3 artifacts are due for store migration before the
  ~2026-08-22 scratch purge (SESSION_STATUS.md pending action).
- Reference predictions for verification gates: full_test_eval/ store dir has
  y_true.npy, pred_baseline.npy, pred_gate_flat4_prox21.npy (.npy with headers, int16,
  45,627,279 lines; y_true < 0 marks 250k excluded validation lines; scored pool =
  45,377,279 lines).

## Customization workflow to productize ("add your own language")

1. Train the new language's distribution over the EXISTING shared vocabulary:
   per-language fixed-vocabulary EM (uniform init, 20 rounds, training floor 1e-12).
   Package path: unilid.trainers.language_specific_trainer; working end-to-end example
   against a pretrained vocab: analysis/mistralnemo_train_sweep.py.
2. Calibrate the new language with ONLY its own data: clamp its unseen-token values to
   c=-21; if its training-line count N < 18,000, estimate tau_L by the recipe above
   (needs a scoring pass of its own training lines against the full model to get
   margins); if N >= 18,000, no threshold (its predictions are not re-examined).
3. Caveats to document: the high-entropy-group membership is the one non-incremental
   piece (needs cross-language entropy statistics and a validation scoring pass); a new
   language also becomes a possible re-assignment CANDIDATE only if N >= 100,000; for
   corpora without the 100k cap the RES_CAP semantics must be stated (the paper says the
   requirement coincides with the cap; an uncapped deployment needs a documented choice).
4. End-to-end evidence the recipe works on a fresh vocabulary: the E3 chain
   (analysis/mistralnemo_eval.py, six stages baseline/calibval/flatrule/tau/topk/eval)
   re-derived every component for a new vocab and gained +0.041 full-pool macro F1.

## Verification gates for the release (blocking, no silent fallbacks)

- Calibrated inference in the package must reproduce the analysis chain's predictions:
  compare against pred_gate_flat4_prox21.npy on a manageable subset (the seed-42 500k
  sample indices are re-derivable via analysis/full_test_eval.py `_sample_line_indices`;
  or draw 201). KNOWN PRECISION ISSUE to handle up front: the analysis chain scored
  with an fp64 clamped matrix; the released model is fp32, and near-tie lines flip
  under precision/order changes (a recorded prediction-file self-agreement of 99.51%
  came from exactly this class of difference). Set an explicit agreement gate (e.g.
  >= 99.9% with every disagreement shown to be a near-tie within a stated score
  epsilon), and record the chosen precision policy; do NOT chase bit-exactness across
  precisions silently.
- Base (uncalibrated) path must keep matching pred_baseline.npy under the same gate.
- The shipped tau/flat artifacts must hash-match the repo's CSVs.
- README numbers must be copied from the paper's tables with their conventions (macro
  F1 / macro FPR definitions now printed in section 5.3 of the tex; scored-pool vs
  full-file instrument split stated in the Table 1 caption).
- Login-node note: the package's own Rust batch inference is fine on the login node;
  the analysis-repo numpy full-matrix scoring path is the thing bounded at ~373k lines
  (SETUP.md); use SLURM only if reproducing full-pool numbers with the analysis code.

## Release checklist (order matters)

1. Read UNILID/ commit 1d26844 (per-language bias scoring) and decide reuse vs new code
   for the clamp; design the calibrated-inference API (default on, `calibrated=False`
   escape hatch) and the artifact bundling format; present the design to the user
   before implementing (CLAUDE.md planning rule).
2. Implement: clamp + gate + walk in the package (port from the reference code above);
   `add_language()` / CLI for the customization workflow incl. per-language tau
   estimation; loader support for bundled calibration artifacts.
3. Tests: golden-subset agreement gates above; unit tests for the walk semantics
   (disjoint groups, fresh-copy, no-candidate-unchanged, excluded-language passthrough);
   a tiny synthetic add-language round trip.
4. Docs: README rewrite (install incl. Rust/submodule story, quick start, model
   download, calibration trade-off with the UDHR number, add-your-own-language guide,
   eval conventions); HF model card mirroring the same numbers.
5. Packaging: pyproject versioning, wheel/build story for the Rust parts, CI if cheap.
6. LICENSE: add proposed Apache-2.0 on the PR branch, marked as pending co-author
   approval in the PR description. Do not publish as decided.
7. Weights: prepare the HF repo content (model + bundled calibration + model card);
   UPLOAD ONLY after the GlotLID-C-derived-weights licensing check and explicit user OK.
8. PR to origin only after the user reviews the fork branch.

## Open items the new session must NOT resolve alone

- License sign-off (Ahmetcan), HF org/repo name (user), weight-licensing verdict
  (user after the check), whether the Mistral-Nemo variant ships in v1 (user),
  any change to the paper's linked URL (co-authors).

## Prompt for the new session (copy verbatim)

> Open-source a working version of calibrated UniLID. Read OPEN_SOURCE_HANDOFF.md at
> the repo root of /users/cmeister747/unilid_analysis first; it is the source of truth
> for goal, binding user decisions (fork-then-PR on cimeister/UNILID; calibration
> default ON with a base flag; weights to HuggingFace Hub after a licensing check;
> Apache-2.0 proposed pending co-author sign-off), the exact calibration mechanism with
> constants and reference code paths, model/artifact locations, verification gates, and
> the release checklist. The package to extend is the UNILID/ checkout inside this
> repo. Follow the checklist order: design first and present the API + artifact-format
> design for approval before implementing; implement with the blocking agreement gates
> (fp32-vs-fp64 near-tie policy stated explicitly, no silent fallbacks); keep every
> external action gated as the handoff specifies (push to fork OK; PR, HF upload, and
> LICENSE publication each need my explicit OK). The global CLAUDE.md working rules
> apply. Record progress in SESSION_STATUS.md and log substantive decisions in the
> repo as you go.
