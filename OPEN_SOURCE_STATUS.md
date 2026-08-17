# Open-source release: status and handoff (calibrated UniLID)

Living source of truth for the open-sourcing effort. Written 2026-08-11 at the
point where the release SHIPPED AND MERGED UPSTREAM; a new session managing
this effort should read this file first, then OPEN_SOURCE_DESIGN.md (the
approved design of record) and, for history only, OPEN_SOURCE_HANDOFF.md (the
original 2026-08-09 spec, now superseded by this file; three of its factual
claims were corrected during execution, see "Corrections" below).

## State: SHIPPED. The calibrated release is merged into the upstream repo.

- **Upstream**: github.com/Ahmetcanyvz/UNILID, branch `release`, tip 3867b1b.
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
  post-merge). Polybox stays as the mirror for the version-1 base file.
- **Package**: version 0.2.1, Python >= 3.9. Test suite: 104 passing
  (unit + real-trainer integration + lazy-import + subsetting + doctor +
  SentencePiece-path coverage).

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
  6ab2201.

## Special tokens hold no probability mass (2026-08-17, version 0.3.0)

Author decision, taken after the second setup-feedback pass reported a default
`--method sp` add-language scoring 0.14 against 0.98 for `--method em` on
identical data: special tokens must not contribute to a score under any method.

What was wrong. `train_with_sentencepiece_direct` gave every special token the
base tokenizer's score, and HuggingFace stores specials with score 0.0, read
here as a log-probability, i.e. probability 1.0. Four of them then dominated the
normalization: each landed at exactly 1/5 and every real token was depressed by
log(5) = 1.6094 nats. All 1,940 rows of the released model carry exactly 0.8 on
specials, which is why their unseen-token plateau sits near -19 instead of at
the -27.63 training floor, the phenomenon the unseen-token constant corrects.

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
