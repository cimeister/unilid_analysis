# Session Status

Snapshot, 2026-08-24. Two workstreams run in parallel: correcting and
regenerating the GlotLID-C numbers, and regenerating the WiLI-trained models.
The corrected open-source release SHIPPED 2026-08-24 (see the readiness section
at the bottom, now a shipped record).

## Read these first

- `paper/PAPER_EDITS_pending.md`: the itemized paper edit list, marked applied or
  blocked. **14 edits are already applied to `submission.tex`**, wrapped in a
  `\corrrev{}` macro (blue) so this round stays separable from the camera-ready
  `\camrev{}` pass. Accept all with `\newcommand{\corrrev}[1]{#1}`.
- `~/.claude/plans/this-session-focuses-on-shimmering-dusk.md`: the approved WiLI
  training plan, revised after an adversarial review.
- `EXPERIMENTS_RESULTS.md`: entries dated 2026-08-17 through 2026-08-23 at the top.
- `RERELEASE_PLAN.md`, `EXPERIMENTS_PLAN.md`, `CODE_CHANGES_2026-08-17.md`,
  `OPEN_SOURCE_STATUS.md`.

## Running

All 18 SLURM jobs of the 2026-08-23 wave COMPLETED 0:0. No jobs queued.
Compilations: `outputs/rerelease/wave_2026-08-24_compilation.md` (WiLI) and
`outputs/rerelease/corrected_chain_2026-08-24.md` (GlotLID-C chain).

CLD3 regeneration in flight (2026-09-01): trainings 3246937/39/41 (12 h,
subset-vocabulary models per author ruling of approximate reproduction) with
chained evals 3246938/40/42; restricted-argmax GlotLID-C jobs 3244447-3244450
also queued. On completion: cld3_regenerated_2026-09-01.md fills in; the
right half of tab:lid_main is then applied as ONE convention (\unilid +
calibrated rows together, variants via restricted argmax).

Agents in flight (2026-08-24):

| agent | what | react how |
|---|---|---|
| null-arm analysis | fp32null vs stored-transformed AND vs fp64 retrain | decides the wording for the wili_100k_500 gate FAIL (build effect vs non-reproduction) |
| blocker fix | DONE: external_bench fixed (selfcheck 42/42), corrected UDHR/FLORES cells produced, calibrated bundle packed, release_gates PASS both modes at exact equality | cells sent to the paper agent |
| cap-4192 null arm | job 3173500 RUNNING/PENDING: fp32 build + default cap 4192 | membership separation already EXACT (106 encoded-over-cap languages = the 106 failing); on completion run `analysis/wili_null_arm_verdict.py --arm fp32null_cap4192` then `_augment` |
| paper edits + verification | CLOSED 2026-08-24: nine table files + submission.tex applied under \corrrev{}; all cells verified at full precision; the 3 verification findings fixed (+0.039/+0.024, 0.916, the two seed-free 95.64 sites); ledger carries A2.11, marking conventions, PD-1..PD-9 | `paper/PAPER_EDITS_pending.md` is the authority |
| CommonLID | DONE: all binding gates pass, corrected cells 0.848/0.722, 0.851/0.720, 0.862/0.717; B4 sent to the paper agent | outputs_corrected_round/tables/commonlid_calibrated.md |

## Settled since 2026-08-21, with the numbers

- **All three WiLI retrains clean and evaluated.** Real mass 1.000000. WiLI test
  cells vs published: 100k 0.9601/1.8629e-4/0.9564 (pub .960/1.859e-4/0.9565);
  DeepSeek 0.9552/2.0484e-4 (pub .955/2.042e-4); Qwen 0.9481/2.3412e-4
  (pub .949/2.310e-4). Only Qwen F1 moves at three decimals.
- **`tab:tatoeba_udhr_comparison` is rebuilt for the \unilid row.** Instrument
  `analysis/wili_external_eval.py`, gated on the STORED `wili_100k_500`: Tatoeba
  0.414278 / 9.60632e-4 over 201 languages (pub 0.414 / 9.61e-4 / 201),
  UDHR 0.867971 / 5.87469e-4 over 142 languages (pub 0.868 / 5.88e-4 / 142). Both
  benches MATCH on every cell, so the instrument is gated. Split determined by
  measurement: `tatoeba_full.txt` (13,101,022 lines, 428 labels) filtered to the
  model's label set gives exactly 201 languages and 11,848,300 rows, matching
  submission.tex:1131; `tatoeba_test.txt` gives 197 / 2,371,336 and is ruled out.
  Scope is the paper's own reading (rows whose gold label is in the model's label
  set). Retrained fp64 `wili_100k_500`: Tatoeba 0.4200 / 9.2300e-4, UDHR 0.8659 /
  5.8604e-4. UDHR also run for the DeepSeek and Qwen WiLI models, stored and
  retrained (no published cell for either in this table).
- **The Tatoeba pass cannot run on the login node.** The three-model `--fp64` run
  was killed part-way through the second model (9.6M of 13.1M lines, no traceback,
  no json); the first model had already finished. `slurm_wili_external_eval.sh`
  now carries it, one model per job.
- **`tab:length_accuracy` is rebuilt and no longer needs the co-author** (for the
  UniLID column). Length is `len(raw_line)` in Unicode chars; that definition
  reproduces all six published bucket counts exactly, utf-8 bytes does not. The
  stored defective 100k model reproduces all seven published cells to the printed
  two decimals (every delta 0.00 pp), so the instrument is gated. Retrained:
  100k Overall 95.64 (pub 95.65, max bucket shift 0.08 pp), DeepSeek 95.21,
  Qwen 94.52. `analysis/wili_length_accuracy.py`,
  `outputs/rerelease/wili_length_accuracy_*.json`. fastText column not
  re-runnable: no WiLI-trained fastText model exists here.
- **WiLI transformation gate run** (reviewed first; exit abort=2/FAIL=1/PASS=0):
  DeepSeek 11 / Qwen 14 failing languages (minority-script, corruption
  signature, `bod` +6.13 nats); `wili_100k_500` 107 failing (systematic ~-0.3
  nats on non-Latin rows). Thresholds are TRANSFERRED from gate_correction's
  same-build calibration; a FAIL is not "corruption" until the null arm reads.
- **Retrained GlotLID-C variants reproduce Table 1**: DeepSeek 0.9089/1.976e-5
  (pub .909/2.08e-5), Qwen 0.9049/2.341e-5 (pub .904/2.55e-5), scored pool,
  fingerprints record the fp64 model paths. All six flagged rows at F1 0.94-1.0.
- **Group B unchanged on the corrected model, by measurement**:
  {sco_Latn, bjn_Latn, arg_Latn, vls_Latn}. zH affine-invariant (zero flips);
  magnet_ratio recomputed by re-scoring the 250k val half (stated substitution:
  full_test_eval scorer, not the original y_pred pipeline; same verdict). topk
  universe 1,084 and group A's 1,080 intact. `build_release_calibration`'s
  group-B assert can stand. `outputs/rerelease/groupb_rederivation.json`.
- `gate_variants topk` corrected: top-1 agreement 1.0000 on all affected lines.
- Everything in the 2026-08-21 snapshot (c = -17, corrected base cells, tail
  views, group-A thresholds, WiLI instrument gate) still stands; see
  `EXPERIMENTS_RESULTS.md`.

## Facts a new session must not re-derive

- **The special-token defect is `sp`-only.** DSL-ML is `em`-trained and needs
  nothing.
- **All three stored WiLI models are `sp`-trained and carry the defect**
  (0.800000 special mass per row), measured.
- **Base vocabularies split two ways.** LLM tokenizers are byte-identical across
  corpora; the WiLI 100k base shares only 24,357/100,000 tokens with GlotLID-C's.
  The base vocabulary is untouched by the defect.
- **`train.py` silently trains a fresh vocabulary if `--results-dir` and
  `--base-tokenizer-path` are not both passed.**
- **`UNILID/eval.py` computes no macro FPR.** Use `analysis/wili_eval.py`.
- **The trainer's patched component is the `spm_train` binary resolved via PATH**
  (`~/.local/bin/spm_train`, fork d0208d9+c5921a2); the pip `sentencepiece`
  wheel is only a reader. The unpatched fp32 build sits isolated in
  `sp_fp32_env/bin/`, discriminable only by sha256 (both builds print 0.2.2).
- **WiLI language order is the tokenizer-filename sort** (diverges from
  `sorted()` at index 146). Align by `langs.index(lang)`, never by position.
- **The full-pool arrays cannot supply val-half predictions**: the 250k val
  lines are EXCLUDED=-2/UNSEEN=-3 by construction (`full_test_eval.py:233-235`).
  Sample indices are reproducible: `random.seed(42)` draw, helper at
  `full_test_eval.py:49-53`.
- **`gate_variants` reads DIAG_CSV/PRF_CSV as module-level literals from the
  released `outputs/diagnostic/`** even with `--out-dir`; the corrected apply
  run is consistent with the topk banking only because both use the released
  categories (now validated: group B unchanged). Its tau CSVs DO route through
  `_out()`.
- **`variant_plateau_outliers.py` cannot be used on WiLI** (zero variance in
  corpus size).
- **A plateau anomaly that survives a retrain is coverage; one that disappears
  was corruption.**
- **Latency is closed** (author, 2026-08-19).

## Open decisions

The paper-side decisions are itemized as PD-1..PD-9 in
`paper/PAPER_EDITS_pending.md` (fastText carry; the three linked breakdown
tables; variant-row swap; the "unchanged" Nemo claim; samples-accuracy seeds;
noise on hold; LLaMA3.2 repo confirmation). The list below is the repo-side
remainder.

- **Swap the lid_main variant rows to corrected numbers?** GlotLID-C cells
  match at paper precision; doing it retires the caption's pool mixture but
  needs the variant UDHR/FLORES/CLD-subset columns re-run (not yet scored).
- **How to read the `wili_100k_500` gate FAIL** — waits on the fp32 null arm
  (job 3157851).
- **Mistral / LLaMA3.2 / LLaMA2 base tokenizers**: author-designated HF repos,
  unconfirmed against the originals; vocab-size sanity checks required.
  `\unilid-Mistral` cannot be Mistral-Nemo (0.921 vs 0.958).
- **`tab:vocab_size_efficiency`**: no 10k/20k/50k/200k container exists; base
  vocabularies trained here, gated by the 100k reproducibility check (not yet
  run).
- **`tab:samples-accuracy`**: needs the seed count behind its standard
  deviations (author).
- ~~Mistral identity~~ SETTLED 2026-08-23: \unilid-Mistral is a 32k
  Mistral (same-table adjacency proof + F1-vs-vocab pattern;
  outputs/rerelease/mistral_identity_verification.json); Mistral-Nemo-Base-2407
  verified byte-identical to the Nemo container base. Author chose the
  v0.1/v0.2 32,000-entry tokenizer (byte-identical pair), stated unconfirmed.
- ~~CR-token blocker~~ SETTLED 2026-08-23 (author): refused entries DROPPED
  whole (51 / 24, all \r-only); filter verified both directions against
  vocab_io's writer; jobs 3162788/3162789 submitted.
- **`tab:lenbias-delta`**: same golden-subset instrument question as
  `lenbias-norm`, not yet decided.
- **The fastText halves** of the WiLI tables: unaffected by the defect;
  carrying them must be a stated choice.
- Whether the Apertus 200k and 131k variants are published; whether the package
  offers a migration for pre-0.3.0 models.
- ~~Corrected-weight filenames on the Hub~~ SETTLED 2026-08-24 (author):
  overwrite `unilid-1940-calibrated.unilid` and `calibration.json` in place.
  Shipped; see the release record at the bottom.
- ~~Publish the corrected UNcalibrated (version-1) model?~~ SHIPPED 2026-08-24
  (author authorized): `unilid-1940.unilid` in Hub commit `d2af7950`, atomic with
  the card update. The docs now name a version-1 download (PR #4, `cf9f44c`).
  Open only if the filename should be `unilid-1940-base.unilid` instead; nothing
  depends on the current name yet.
- `commonlid_calibrated.py` asserts old-model reproduction and needs the
  carried npz regenerated first (`commonlid_carried` corrected run not yet
  done) -- the one remaining unparametrized corrected-chain piece.

## Corrected release: SHIPPED 2026-08-24

All three readiness blockers cleared, then published. Full record in
`OPEN_SOURCE_STATUS.md` section "Re-release of the corrected weights: SHIPPED
2026-08-24" and in `EXPERIMENTS_CHRONOLOGICAL.md` (last entry).

- **Weights**: huggingface.co/cmeister/unilid-1940, commit `e0a524ed`, one
  atomic commit. `unilid-1940-calibrated.unilid` (135404c834e9e074...) and
  `calibration.json` (1ef3063b9f9a2a04...) overwritten IN PLACE, model card
  replaced. All three verified post-upload by download and hash. Prior revision
  `8d4044d2` still lists the 2026-08-11 file (61d7f5fe...) and the model card
  names it as the route to the superseded weights.
- **Package**: PR #3 merged upstream 2026-08-24T19:24Z, merge commit a47d4f5.
  Annotated tag **v0.3.0** on a47d4f5, pushed to Ahmetcanyvz/UNILID.
- **PR #4 OPEN**: https://github.com/Ahmetcanyvz/UNILID/pull/4, now at commit
  `cf9f44c` on branch `generation-report` off a47d4f5. Carries the load-time
  real-token-mass report, the corrected doc numbers, and (2026-08-24) the
  download entries for `unilid-1940.unilid`. 119 tests pass. Not in v0.3.0; if
  the tag should include it, v0.3.1 on the merge commit is the clean move.
- **PR #5 OPEN**: https://github.com/Ahmetcanyvz/UNILID/pull/5, commit `795e5db`
  on branch `max-sentence-length` off a47d4f5, base `release`. The three
  `--max-sentence-length` files only, no PR #4 content. 111 tests pass on the
  branch. Committed from a temporary worktree, so the UNILID/ working tree still
  carries the three modified files for the pending cap-arm job 3173500, still
  byte-identical to `patches/unilid_max_sentence_length.patch`.
- **The 0.2.1 no-op hazard**: a pre-0.3.0 loader modifies **0 of 1,940 rows**
  where 0.3.0 modifies 1,655, and prints a success line either way. Mitigated by
  merging and tagging 0.3.0 before the upload and by the model card leading with
  "requires UNILID 0.3.0 or later" plus the 0-of-1,940 measurement.
- **Re-measured before publishing, not carried**: 1,655 of 1,940 rows clamped at
  c = -17 (285 untouched), 0 of 1,940 on the 0.2.1 path, real-token mass
  1.000000 every row, real-column row minima -18.3292 to -11.6063 median
  -16.0486.
- STILL SCRATCH-ONLY and purge-exposed: `full_test_eval_corrected/` (690 MB),
  the reference arrays both corrected gates are measured against.

## Known damage, recorded

The released model's E2 scored artifacts (`external_bench/scored_udhr.npz`,
`scored_flores.npz`, from 2026-08-07) were overwritten and then deleted
(2026-08-21). Scratch-only; regenerate if the released model's E2 numbers are
ever recomputed. `external_bench_eval.py` now writes non-default models to
`external_bench/scored_<model stem>/`.

## Carried over: camera-ready

- Edit pass applied 2026-08-09, wrapped in `\camrev{}`; dispositions in
  `paper/review_notes_2026-08-09.md`.
- Ahmetcan ask list, reduced: the subset-evaluation script or command; the
  UDHR-subset FPR of 1.06e-5; the DSL-ML competitor-score source and split.
- The user compiles the PDF (no icml2026.sty here).

## Author decisions 2026-08-24 (all PD items resolved)

Noise table removed; fastText carried; PD-2 three tables corrected (gated
instrument); PD-3/PD-5 leave carried (CLD-subset instrument absent); PD-4
lenbias-delta on the golden subset (B9 run pending); PD-6 pair breakdown
applied; PD-7 sampled rows left; PD-9 LLaMA3.2 confirmed. Commit authorized
"if appropriate".

Release decisions, same day, verbatim: PR #3 merges within the day; overwrite
`unilid-1940-calibrated.unilid` in place; upload authorized; add the new model to
PR #3 if unmerged, otherwise open a new PR; add a git tag; remove the polybox
link from the README and anywhere else it appears. All executed 2026-08-24; PR #3
was already merged, so the package change went to PR #4.
