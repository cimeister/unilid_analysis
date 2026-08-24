# Open-source release: status and handoff (calibrated UniLID)

Living source of truth for the open-sourcing effort. Written 2026-08-11 at the
point where the release SHIPPED AND MERGED UPSTREAM; a new session managing
this effort should read this file first, then OPEN_SOURCE_DESIGN.md (the
approved design of record) and, for history only, OPEN_SOURCE_HANDOFF.md (the
original 2026-08-09 spec, now superseded by this file; three of its factual
claims were corrected during execution, see "Corrections" below).

## State: SHIPPED. The calibrated release is merged into the upstream repo.

- **Upstream**: github.com/Ahmetcanyvz/UNILID, branch `release`, tip a47d4f5,
  tagged **v0.3.0** (annotated, pushed 2026-08-24).
  - PR #1 (core release: calibrated inference default-on, v2 container,
    add-language, tests, README, proposed LICENSE) merged by Ahmetcanyvz
    2026-08-11T10:02Z, comment "lgtm!", merge commit e34f4d8. **The Apache-2.0
    + notices LICENSE is thereby published upstream; the sign-off condition in
    the PR description was met by the merge.**
  - PR #2 (out-of-box fixes + language subsetting + docs restructure) opened
    and merged by the user's own account (cimeister) 2026-08-11T16:57Z.
- **Working branch**: `calibration-release` on github.com/cimeister/UNILID
  (local checkout UNILID/ in this repo). Commits, in order: 366f942 (core
  implementation + 72 tests), bc4eef3 (README + proposed LICENSE), fcb3369
  (tokenizers merge + numpy loader), 40514fc (merge of upstream's parallel
  work; forward= integrated behind a base-mode guard), a67d47e (worked example
  + integration tests + api.py bug fixes), fcc293f (lazy imports, Python floor
  3.9, README out-of-box fixes), fd3da96 (subsetting + README/REPRODUCING
  restructure), 8dd90ec (post-merge clone-note cleanup, **pushed to the fork
  but NOT yet upstream** — the one commit upstream lacks; the user PRs from
  this branch themselves, see PR #2 precedent).
- **Tokenizers fork**: github.com/cimeister/tokenizers, branch
  `unilid-scorers` at d79bad0e = merge of the scorer additions (156f6c51,
  22025269: biased/top-k/normalized scorers, tokens_of_cached_weight_set_batch)
  with Ahmetcan's unimixlm_fast (5b1e6ab0 set_weight_sets_numpy, 0ee36d4d
  forward scoring). The UNILID gitlink points here.
- **Weights**: huggingface.co/cmeister/unilid-1940 (public):
  `unilid-1940-calibrated.unilid` (779,663,390 bytes, version-2 container),
  `calibration.json` (160,363 bytes, identical to the bundled section),
  model-card README (license apache-2.0 + notices; install pointer updated
  post-merge). **Superseded 2026-08-24**: both files were overwritten in place
  by the corrected generation, see "Re-release of the corrected weights" below.
  The polybox mirror is retired and its link is gone from the package docs.
- **Package**: version 0.3.0, Python >= 3.9. Test suite: 119 passing
  (unit + real-trainer integration + lazy-import + subsetting + doctor +
  SentencePiece-path + special-token-mass coverage).

## Setup-feedback follow-up (2026-08-15, on the fork branch, not yet upstream)

SETUP_FEEDBACK.md (a new user's clean-macOS install report) drove commit
8a09bd6 on `calibration-release`. Plan of record:
~/.claude/plans/this-session-focuses-on-shimmering-dusk.md. What changed and
what it was measured against:

- Two defects kept the suite from running on the documented `[dev]`-only
  install: an unconditional `import torch` in standard_trainer.py (torch is in
  `[train]`, used only in `pad()`), and the base tokenizer reaching `spm_train`
  even when the caller passed `use_sentencepiece=False`, because
  `train_language_specific_tokenizer` fed one em_mode to both the base and the
  per-language step. Base and per-language are now separate: `base_em_mode`,
  `use_sp_seed_vocab`, `use_sp_em` on the trainer and the api helpers (defaults
  unchanged), and `--base-seed-vocab` / `--base-em-impl` on train.py (defaults
  `sp`, recorded in training_summary.json).
- ROOT CAUSE of the reported `FileNotFoundError: 'spm_train'`, and a correction
  to an adversarial review that claimed the reporter must have had `[train]`
  installed: the `sentencepiece/` submodule directory at the repo root makes
  `import sentencepiece` succeed as a NAMESPACE PACKAGE whenever the pip
  package is absent, so every `spm is None` guard was dead in a source
  checkout. Guards now test `hasattr(spm, "SentencePieceProcessor")`, and the
  test skip conditions require both the binary and the package. The reporter's
  account of their environment was right.
- doctor.py (top level, run as `python doctor.py`): submodules, rustc version
  (MIN_RUST_VERSION = (1, 85), derived from the fork's pinned crate manifests),
  maturin, the extension's three scorer methods, spm_train. It imports nothing
  from `unilid` by design, because `import unilid` pulls in the tokenizers
  extension and would crash in exactly the broken setups it diagnoses; a
  console script or `python -m unilid.doctor` cannot work for that reason.
- Python range MEASURED, not assumed: 3.9, 3.12, 3.13, 3.14 each in a fresh
  uv venv with `[dev]` plus the abi3 wheel, all 102 passed / 2 skipped
  (SentencePiece tests, correctly skipped); 3.11.5 dev env 104 passed. No cap
  on requires-python; classifiers extended to 3.13/3.14; `datetime.utcnow()`
  replaced (deprecated from 3.12, on a removal path).
- .github/workflows/ci.yml: one abi3 build feeding a 3.9/3.11/3.12/3.13/3.14
  matrix, deliberately without building SentencePiece, so the minimal install
  staying green is enforced.
- Tokenizers fork: both `rust-toolchain` files pinned from `stable` to 1.93.1
  (cimeister/tokenizers a731efdf, pushed), gitlink bumped here. `stable` is why
  the reporter's 2021 toolchain was selected. Extension rebuilt against the pin
  and BOTH GATES RE-RUN AND RE-PASSED: base 250,000/250,000 exact, calibrated
  250,000/250,000 with zero disagreements (outputs/release/gate_*.json).
- Open item 3 (packaging extras: CI) is thereby DONE for CI; a published wheel
  for the Rust parts is still unstarted.
- PR #3 open: github.com/Ahmetcanyvz/UNILID/pull/3, carrying 8dd90ec, 3427640,
  6ab2201, and (folded in per author decision 2026-08-17) the special-token
  training fix 9f7c1cf, the over-attribution correction 56e7fd4, and the clamp
  regression fix 2d5f62d. Branch `calibration-release` pushed at 2d5f62d. PR
  retitled "Special tokens hold no probability mass, plus the out-of-box setup
  fixes"; description rewritten to lead with the training fix. Package version
  0.3.0.

## Special tokens hold no probability mass (2026-08-17, version 0.3.0)

Author decision, taken after the second setup-feedback pass reported a default
`--method sp` add-language scoring 0.14 against 0.98 for `--method em` on
identical data: special tokens must not contribute to a score under any method.

What was wrong. `train_with_sentencepiece_direct` gave every special token the
base tokenizer's score, and HuggingFace stores specials with score 0.0, read
here as a log-probability, i.e. probability 1.0. Four of them then dominated the
normalization: each landed at exactly 1/5 and every real token was depressed by
log(5) = 1.6094 nats. All 1,940 rows of the released model carry exactly 0.8 on
specials.

> **Correction 2026-08-17.** This paragraph originally continued "which is why
> their unseen-token plateau sits near -19 instead of at the -27.63 training
> floor". That attribution is measured false and was itself an over-correction of
> the paper's wrong explanation. The defect contributes a uniform 1.609 nats:
> removing it moves the median plateau only from -17.66 to -16.05, nowhere near
> -27.63, and the floor is never reached under the sp path at all. The plateau is
> set by the per-language fit and tracks corpus size,
> `corr(plateau, log10 N_L) = -0.9659` over all 1,940 rows. Full record in
> `EXPERIMENTAL_SETUP.md`, "The unseen-token plateau is set by corpus size".

Why it matters mechanically: no special token's stored weight is ever read when
scoring. `model.rs` takes the unknown-token score from a single model-wide
constant (`min_score - K_UNK_PENALTY`), not from the per-language row, and
`<s>`/`</s>`/`<pad>` are reachable only by text containing those literal
substrings. Verified by perturbation, not by reading: setting all four entries
of every row to -500 changes predicted scores by exactly 0.000000. So mass on
them is purely mass taken from the tokens that decide predictions.

The fix, all in the training and customization paths, none in the inference
path: one enforcement point in `LanguageSpecificUnigramLMTokenizer.train`
renormalizes whichever method produced the row over the real tokens and parks
the specials at the floor; `add_language` puts the new row on the model's own
scale so a corrected row cannot outscore a pre-0.3.0 model's rows by a constant
per token. Measurements, released-model behaviour, and the re-release question
are recorded in SESSION_STATUS.md.

The released artifact is unchanged and both golden gates re-passed at
250,000/250,000 after the change.

### The 0.3.0 fix introduced a regression, found and fixed the same day

Parking the specials at the training floor made them each row's minimum, and
`apply_unseen_token_constant` defines a row's unseen tokens as its exact
minimum-value plateau. So for **any model trained by 0.3.0 as first shipped, the
unseen-token constant was silently a no-op**: the plateau of unseen real tokens
was never located and the calibration's first correction disappeared without a
message. Found by a probe reporting `modified 0` of 1,940 rows at every candidate
c.

Fixed in `unilid/calibration.py`: the clamp takes the special columns and excludes
them from the minimum, with callers finding those columns by name from the
vocabulary. Pre-0.3.0 files are unaffected, their specials sitting at -1.6094 and
never being the minimum; a test asserts both the working and the broken case.
Package commit 2d5f62d. Both release gates were re-run because this is an
inference-path change.

This is the reason `analysis/floor_equalization.py` no longer detects special
columns by looking for the 0.2 probability: that detector cannot work on a
corrected model, where the specials hold no mass at all.

### Re-release of the corrected weights: SHIPPED 2026-08-24

**What is published.** huggingface.co/cmeister/unilid-1940, in two atomic
commits, both 2026-08-24: `e0a524ed9e47dd295702de823f636bd5107415b0` (calibrated
file, calibration artifact, model card) and
`d2af79507f8cb9a55ad77ba74583a9324352f1dc` (uncalibrated file plus the card
update). Every file sha256-verified after upload by download:

| repo path | source | sha256 |
|---|---|---|
| `unilid-1940-calibrated.unilid` | store `corrected/glotlidc_corrected_calibrated.unilid` (779,663,677 B) | `135404c834e9e07435b99551c1c3a570cf3b2ac94cff6c26691e90796381dc91` |
| `unilid-1940.unilid` | store `corrected/glotlidc_corrected.unilid` (779,503,019 B) | `31c3d956db7b00c939c4985c86a82a8e8d1af963f8cf3921cebaab257d0d74fd` |
| `calibration.json` | store `corrected/release/calibration_glotlidc_corrected.json` (160,650 B) | `1ef3063b9f9a2a04d2997b8c762d035cf52a33dbc613ccf57567c5f81638b174` |
| `README.md` (model card, as of `d2af7950`) | session scratchpad `card_new.md` | `6bf61b74f48628180f25e84af766170574db06716d49272280d2f6c096d39530` |

The card at commit `e0a524ed`, before the uncalibrated file was added, was
`4cb583e48fe7624448d7faa89024e6ec1f69d7f6608c3d3e691251c5b903d58f`.

**Author decision trail.** 2026-08-17: publish the corrected calibrated and
uncalibrated models to the Hub and retire the polybox mirror. 2026-08-24:
**overwrite `unilid-1940-calibrated.unilid` in place** rather than publish under
new names, upload authorized, add a git tag, remove the polybox link from the
README and everywhere else it appears.

**The uncalibrated file: deferred earlier on 2026-08-24, EXECUTED 2026-08-24.**
The author authorized the upload the same day, and `glotlidc_corrected.unilid`
went up as `unilid-1940.unilid` in Hub commit `d2af7950`, atomically with the
model card update. sha256 `31c3d956db7b00c939c4985c86a82a8e8d1af963f8cf3921cebaab257d0d74fd`
verified before the upload and again by a fresh forced re-download.

The filename was not prescribed anywhere. RERELEASE_PLAN decision 4 and section 6
say only that the Hub carries both models. `unilid-1940.unilid` follows the
author's stated default; `unilid-1940-base.unilid` was the alternative considered,
parallel to `unilid-1940-calibrated.unilid`. Renaming is cheap until someone
depends on the name.

The two `.unilid` files hold the same weights, measured, not assumed: sha256 of
the float32 weight matrix is
`a4aeff199464032c223ae7c77eaa6f128307180f6758e28b1f8e8cd7c985662e` in both and
sha256 of the language list is
`00ad6a35b85c3b0b3816598c86534075cdc57f34e50d7e99bc7c74a379e53e19` in both. The
first also equals `base_weight_matrix_sha256` in the bundled calibration
artifact, whose provenance block names `corrected/glotlidc_corrected.unilid` as
its source model. Only the container version differs (1 against 2).

Package docs updated to match, on PR #4's branch (`generation-report`, commit
`cf9f44c`): README's download table and wget block list both files, and
REPRODUCING's "or use a version-1 `.unilid` file" now names the download it
never had.

**The 0.2.1 no-op hazard, and its mitigation.** A pre-0.3.0 loader takes each
row's minimum over the whole row. In a corrected file the special tokens sit at
the training floor, log(1e-12), below every real token, so the minimum finds
them, the unseen-token plateau is never located, and the clamp does nothing
while printing a line that reports it as applied. Measured on the published
file: **0.2.1 modifies 0 of 1,940 rows, 0.3.0 modifies 1,655** (285 left as
trained). Mitigations, all in place: 0.3.0 is merged upstream (a47d4f5) and
tagged v0.3.0 before the upload; the model card leads with "requires UNILID
0.3.0 or later" and carries the 0-of-1,940 measurement; `_special_columns`
excludes the specials from the minimum and is present in the tagged commit.

**The container still cannot distinguish the generations.** The header encodes
only version 1 against 2, and `FORMAT_VERSION_MAX = 2` means every published
reader rejects a version-3 file, so a version bump is not free. The substitute
is the load-time report of each row's real-token mass, 0.2 for a pre-0.3.0 sp
file and 1.0 for a corrected one. It is implemented (`real_token_mass` and
`special_columns_of` in `unilid/model_io.py`, `eval.py` printing to stderr) and
sits in **PR #4**, https://github.com/Ahmetcanyvz/UNILID/pull/4, commit 6006095,
NOT yet merged. Until it merges, the self-check in the model card is the manual
route. The special columns are located by token string, not position, because a
base tokenizer converted from an LLM's holds them at non-contiguous high indices
(qwen3_8b: columns 128,244/128,245/128,247/151,669).

**Because the overwrite reuses the filename**, a copy downloaded before
2026-08-24 has the same name and different contents. The model card states this
and names revision `8d4044d2b69429e16ce256bde6acfa0c02e68203`, verified still to
list the 2026-08-11 file (sha256 `61d7f5fe86422112a336c2cba4fa834faa896255156e44b83e89bb45c586bd72`),
as the route to the superseded weights.

**Two generations, two reference sets, not comparable.** Gates for the released
generation are `outputs/release/gate_*.json` against references recorded from
the 2026-08-11 weights; gates for the corrected generation are
`outputs_corrected_round/release/gate_*_corrected.json` against
`full_test_eval_corrected/`. c is **-17** for the corrected calibration and -21
for the released one: the constant is an absolute target and the correction
raised every real token by 1.6094 nats, so the two values are not directly
comparable. Both corrected gates were re-run after the load-path change and
passed at exact equality, 250,000/250,000 in base mode and 250,000/250,000 in
calibrated mode with zero disagreements
(`gate_*_corrected_postmassreport.json`). Minimum package version for these
weights: **0.3.0**. Full plan: `RERELEASE_PLAN.md`.

### Found, not fixed (out of scope, flagged for a later decision)

`unilid/algorithms/accumulate.py:111` opens a `multiprocessing.Pool` per
`_accumulate_usage` call, i.e. once per EM iteration per language. By then the
tokenizers Rust extension has rayon threads running, so on Python 3.12 and 3.13
each fork raises `DeprecationWarning: This process is multi-threaded, use of
fork() may lead to deadlocks in the child` (560 of them in the integration test
alone). Python 3.14 does not warn because its Linux default start method is
`forkserver`. The suite passes on all of 3.9/3.11/3.12/3.13/3.14, but this is a
real fork-safety hazard, not only noise. Fixing it means choosing a start
method for the trainer's pools, which changes training startup cost and
picklability requirements, so it was left alone rather than changed quietly.

## Corrections to the original handoff (details in OPEN_SOURCE_DESIGN.md rev 1)

1. The reference analysis chain is float32 end to end for the matrix, scores,
   margins, and proximity comparisons; the handoff's "fp64 clamped matrix"
   claim was wrong (fp64 was the Mistral-Nemo training runs).
2. The tau recipe of record is analysis/solo_gates.py (size-adaptive q_L on
   the clamped matrix), not margin_diagnostic.py (fixed q=5, unclamped,
   superseded).
3. The recorded 99.51% prediction self-agreement was the paper-era scoring run
   vs the current run; the recovered file agrees with pred_baseline.npy at
   1.000000. The agreement gates were consequently set strict (exact).

## Verification record

- Golden subset = test half of the seed-42 500k draw from the GlotLID-C pool
  (250,000 lines; the even half is the excluded validation sample).
- Base gate: package base predictions == pred_baseline.npy, 250,000/250,000.
- Calibrated gate: v2 bundle predictions == pred_gate_flat4_prox21.npy,
  250,000/250,000 (zero disagreements; the >= 99.9% + near-tie-forensics gate
  was passed at exact equality). Both gates were re-run and re-passed after
  the loader switched to set_weight_sets_numpy.
- Byte provenance: the v2 release file differs from store glotlidc.unilid in
  exactly the header version byte plus the calibration trailer; the bundled
  JSON is byte-identical to outputs/release/calibration_glotlidc.json, whose
  provenance block records the source CSV sha256s.
- Worked example (examples/add_language): before-add held-out accuracy 0.00,
  after-add 0.98, base languages 1.00, re-examination gate active; the sp
  training method also completes but estimates a flatter distribution at toy
  scale (0.60; documented in the example README).

## Runbook (exact commands; run from this repo's root, login node)

- Tests: `cd UNILID && /users/cmeister747/.pyenv/versions/3.11.5/bin/python3 -m pytest tests/ -q`
- Gates: `RAYON_NUM_THREADS=32 /users/cmeister747/.pyenv/versions/3.11.5/bin/python3 -m analysis.release_gates --mode base` (then `--mode calibrated`); results to outputs/release/gate_*.json; both must PASS before any release-affecting change ships.
- Rebuild the calibration artifact from the CSVs of record:
  `python3 -m analysis.build_release_calibration` (validates against the real
  model and round-trips; writes outputs/release/calibration_glotlidc.json).
- Rebuild the v2 bundle through the package path:
  `cd UNILID && python3 -m unilid.calibrate_cli bundle <store>/glotlidc.unilid ../outputs/release/calibration_glotlidc.json -o <store>/release/unilid-1940-calibrated.unilid`
- Rebuild the Rust extension after submodule changes:
  `cd UNILID/tokenizers/bindings/python && VIRTUAL_ENV=/users/cmeister747/.pyenv/versions/3.11.5 maturin develop --release`
- Worked example end to end:
  `PYTHON=/users/cmeister747/.pyenv/versions/3.11.5/bin/python3 bash UNILID/examples/add_language/run_example.sh`
- HF uploads (huggingface_hub, logged in as cmeister):
  `HfApi().upload_file(path_or_fileobj=..., path_in_repo=..., repo_id="cmeister/unilid-1940", commit_message=...)`

## Data and artifact locations (all durable on store)

Store root: /capstor/store/cscs/swissai/a0229/cmeister/unilid_analysis/
- glotlidc.unilid (779,503,019 B, version-1 base model of record)
- release/unilid-1940-calibrated.unilid (779,663,390 B, the shipped v2 file)
- full_test_eval/ (y_true.npy, pred_baseline.npy, pred_gate_flat4_prox21.npy,
  fingerprint_floor21.json; int16 length 45,627,279)
- glotlid_unilid/glotlid_correct_test.txt (test pool of record, 7.1 GB) and
  glotlid_unilid/glotlid_train_counts.json (N_L, 1,940 entries) — both
  migrated from purgeable scratch 2026-08-10, sha256-verified, scratch paths
  are symlinks
- results_apertus200k/corpus/ (1,940 per-language training files, needed for
  threshold estimation)
Repo-side: outputs/release/ (calibration JSON + gate results),
outputs/diagnostic/tau_floor21_gate.csv and tau_flat4.csv (threshold sources
of record), analysis/build_release_calibration.py, analysis/release_gates.py.

## Decisions ledger (user authorizations, all 2026-08-10/11)

1. Design approved with three sub-decisions: v2 container version bump; loud
   UnilidCalibrationError when calibrated=True finds no artifact (never
   auto-downgrade); tokens under the final language via a new fork binding.
2. README direction: explicit, oriented to readers new to the method (never
   assume terms like the training floor are known); CommonLID numbers
   included.
3. Weight licensing: Apache-2.0 plus notices mirroring the GlotLID model
   release (weights are aggregated statistical patterns; no training data
   included; rights-holder contact clause). Approved after the check of the
   GlotLID-C corpus terms and the GlotLID model precedent.
4. PR authorized and opened; HF username cmeister given for the upload (repo
   name cmeister/unilid-1940 was this session's choice; public).
5. Ahmetcan's requests handled: unimixlm_fast merged into unilid-scorers;
   PR #1 conflicts (from his parallel release-branch work) resolved.
6. Subsetting: both tools (load-time languages= and unilid-calibrate subset);
   recalibration must not be required (carried thresholds are the default and
   fire at most as often as calibrated; --recalibrate is optional).
7. Docs: README rewritten as one coherent document; paper-results material in
   REPRODUCING.md; calibration described briefly in the README; forward
   marginalization at the end, framed as experimental.

## Standing conduct rules for external actions

- Pushing branches to the cimeister forks (UNILID, tokenizers): allowed
  without asking (handoff rule, exercised throughout).
- Creating NEW pull requests, new HF repos, or publishing new license terms:
  each needs the user's explicit OK (PR #1 was explicitly authorized; the
  user opened PR #2 themselves).
- Maintenance of already-shipped artifacts (e.g. correcting a stale factual
  line in the cmeister/unilid-1940 model card) has been treated as covered by
  the original authorization; anything beyond factual maintenance should be
  asked about.
- Never push to Ahmetcanyvz/UNILID directly; changes flow via the fork branch
  and PRs the user controls.
- Global CLAUDE.md applies in full: no silent fallbacks, fail loudly, gates
  re-run after any inference-path change.

## Open items

1. Commits 8dd90ec (drops the now-stale fork-branch clone note from the README)
   and 8a09bd6 (the setup-feedback fixes above) are on the fork branch, not yet
   upstream. PR #3 is authorized by the user for this work; it carries both.
2. Mistral-Nemo variant release (glotlid_mistralnemo_fp64.unilid + its own
   calibration, all on store): open user decision, unstarted. The E3 chain is
   the evidence the recipe transfers; a release would repeat the
   build_release_calibration + bundle + gates pattern with the nemo artifacts
   (tau_mistralnemo_*.csv, its 3-language high-entropy group bjn/sco/srp).
3. Packaging extras: CI DONE 2026-08-15 (see the setup-feedback section above).
   A published wheel for the Rust parts is still unstarted.
4. Known interaction for the paper team: the four unilid_resources/eval_*.py
   scripts construct UnilidModel(path) with default arguments, which now
   raises on files without a calibration artifact and would run CALIBRATED on
   the v2 file; reproducing the paper's base rows needs calibrated=False (the
   forward= keyword they pass works after the unimixlm_fast merge). Flag to
   Ahmetcan if the scripts are ever moved into the repo.
5. If the user's own fork repo (cimeister/UNILID) is ever archived, nothing
   depends on it post-merge except as the PR staging area.

## Prompt for a new session (copy verbatim)

> Take over managing the open-source release of calibrated UniLID. Read
> OPEN_SOURCE_STATUS.md at the repo root of
> /users/cmeister747/unilid_analysis first; it is the source of truth for the
> shipped state, the verification record and runbook, the decisions ledger,
> the standing conduct rules for external actions, and the open items. The
> design of record is OPEN_SOURCE_DESIGN.md; OPEN_SOURCE_HANDOFF.md is the
> superseded original spec (its fp64 claim and two reference pointers were
> corrected; see the status file). The package lives in the UNILID/ checkout
> (branch calibration-release, merged upstream through PR #2 plus one pending
> cleanup commit). Before any change that touches inference, re-run the test
> suite and both golden-subset gates per the runbook, and keep every external
> action within the standing conduct rules. Record progress in
> SESSION_STATUS.md and update this status file as state changes.
