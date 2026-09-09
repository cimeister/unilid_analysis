# Session Status

Snapshot, 2026-09-09 (second update). The correction campaign is COMPLETE:
every table cell in the revised paper traces to a gated instrument or a
recorded, stated carry. The corrected release SHIPPED 2026-08-24; PR #4 and
PR #5 MERGED upstream 2026-08-31 (left untagged by author ruling). R1, the
reviewer-feedback round, was applied 2026-09-09 (commits 4180a06/4cc9168),
and the author then ANSWERED the R1.HOLD questions; those rulings are applied
as **R2** (ledger section "R2: the author's rulings on R1.HOLD"). What remains
open is only the short list under "Open decisions" below.

## Read these first

- `paper/PAPER_EDITS_pending.md`: the itemized paper edit list, marked applied or
  blocked. Its last section, **R1 (2026-09-09)**, is the reviewer-feedback round
  and is the current front. All edits are wrapped in a `\corrrev{}` macro (blue)
  so this round stays separable from the camera-ready `\camrev{}` pass; 121 new
  spans were added in R1, against 192 pre-existing. Accept all with
  `\newcommand{\corrrev}[1]{#1}`.
- `PAPER_FEEDBACK.md`: the reviewer feedback driving R1.
- `paper/r1_bib_entries.bib`: four verified bib entries that must be merged into
  `custom.bib` (not in this repo) before the paper compiles (fourth added in R2:
  chen-etal-2024-fumbling, the LLMs-on-LID citation).
- `~/.claude/plans/this-session-focuses-on-shimmering-dusk.md`: the approved WiLI
  training plan, revised after an adversarial review.
- `EXPERIMENTS_RESULTS.md`: entries dated 2026-08-17 through 2026-08-23 at the top.
- `RERELEASE_PLAN.md`, `EXPERIMENTS_PLAN.md`, `CODE_CHANGES_2026-08-17.md`,
  `OPEN_SOURCE_STATUS.md`.

## Running

Nothing. No SLURM jobs, no agents, no watches. Every wave through 2026-09-02
completed and is recorded in `EXPERIMENTS_RESULTS.md` /
`EXPERIMENTS_CHRONOLOGICAL.md`.

## The author's own uncommitted working-tree changes -- DO NOT commit or revert

- ~75 deleted `slurm_*.sh`/`run_*.sh` plus `OPEN_SOURCE_HANDOFF.md` and
  `SETUP_FEEDBACK.md`: the author's cleanup. Leave exactly as found.
- `analysis/cld3_calibrated_transfer.py`: modified by the author.
- (No longer on this list: `paper/tables/noise_robustness.tex` and the
  author's noise-appendix hunk in `submission.tex` were committed in R2 by
  the author's ruling "I added this back manually. Check that its there";
  the re-add is now official paper content.)

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
- **The sentence-length cap**: sentencepiece's default 4,192 SKIPS over-cap
  lines, counted in ENCODED bytes; the per-language step here overrides to
  1,000,000 (now `--max-sentence-length`, PR #5). The published
  `wili_100k_500` was trained at the DEFAULT cap -- measured: the cap-4192
  fp32 retrain reproduces it on all 235 languages, closing the 107-row gate
  question (100 cap + 7 build).
- **Capstor stale tails**: a cross-node overwrite can expose the old file
  tail to another node; verify written records by regeneration diff.
- **Approximate reproduction is the author's bar** for regeneration decisions
  (stated repeatedly); exact-match gates are for instrument validation only.
- **Paper terminology since R1**: "log-probability margin" is the defined
  term; the threshold symbol is delta (renamed from tau); the tab:lid_main
  subset convention is specialists in the main table (incl. the calibrated
  row: procedure applied with all constants carried unswept; 87% of its
  GlotLID-C gain is Corsican, disclosed) with transfer in tab:cld3_refit.

## Open decisions

### After R2 (the author's R1.HOLD rulings, applied 2026-09-09)

Full record: section **R2** at the end of `paper/PAPER_EDITS_pending.md`.
R1.HOLD-1/-2/-3/-9/-10/-11/-13 applied; -4/-5/-8 closed no-action by ruling.
Still open:

- **R1.HOLD-12, blocks compilation**: `paper/custom.bib` is not in this repo;
  `paper/r1_bib_entries.bib` must be merged before the paper builds (now FOUR
  `\cite` keys depend on it, chen-etal-2024-fumbling added in R2).
- **R1.HOLD-6(c)-(f), co-author TODO** (author: "I'll ask about the rest. Keep
  it as a todo list"): GlotLID-M version, label set, out-of-label scoring;
  where UniLID's base hyperparameters (100k vocab, 20 EM rounds, 1e-12 floor)
  were selected; hardware behind Tables 13-15; the Tatoeba label mapping.
- **HOLD-7 flag for the author**: the ruling said the noise table's p=0%
  mismatch comes from a different evaluation set, but the UniLID p=0% cells
  match the paper's other WiLI tables to every printed digit; only fastText
  differs, which points at a different fastText artifact or run. The paper
  sentence was scoped to fastText and kept vague; confirm the provenance.
- **R1 HOLD-MEASUREMENT, unchanged, highest value first**: the shared-phi
  naive-Bayes control isolating the central claim; branching factor `b`;
  train/test dedup; fastText epoch curve on GlotLID-C (expensive, can invert
  the 9x claim); alpha = 0.25/0.5 length normalization; mean input length for
  the CLD3 latency explanation. Do not start without the author's word.

### Repo-side remainder (everything else is settled)

Historical PD-1..PD-9 and their dispositions live in the ledger; the 2026-08
open list below it is SUPERSEDED by later measurement -- do not re-open from
stale copies: the variant rows, subset columns, lenbias tables, vocab-size
table, CLD3 columns, and the commonlid chain are all applied and recorded
(ledger A2.1-A2.20 + R1). Still genuinely open on the repo side:

- Whether the Apertus 200k/131k variants are ever published; whether the
  package offers a migration for pre-0.3.0 models.
- `full_test_eval_corrected/` also has a store copy now (2026-08-24,
  byte-verified), as do the six CLD3 subset containers -- nothing load-bearing
  is scratch-only.
- The base-model Hub filename (`unilid-1940.unilid` vs a `-base` suffix) is
  free to change until adoption.

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
- **PR #4 MERGED** upstream 2026-08-31 (008f76aa): was at commit
  `cf9f44c` on branch `generation-report` off a47d4f5. Carries the load-time
  real-token-mass report, the corrected doc numbers, and (2026-08-24) the
  download entries for `unilid-1940.unilid`. 119 tests pass. Not in v0.3.0;
  the author ruled 2026-09-02: leave untagged.
- **PR #5 MERGED** upstream 2026-08-31 (5504cc22): commit `795e5db`
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
- `full_test_eval_corrected/` (690 MB, the gates' reference arrays) copied to
  the durable store 2026-08-24, byte-verified; nothing load-bearing is
  scratch-only.

## Known damage, recorded

The released model's E2 scored artifacts (`external_bench/scored_udhr.npz`,
`scored_flores.npz`, from 2026-08-07) were overwritten and then deleted
(2026-08-21). Scratch-only; regenerate if the released model's E2 numbers are
ever recomputed. `external_bench_eval.py` now writes non-default models to
`external_bench/scored_<model stem>/`.

## Carried over: camera-ready

- Edit pass applied 2026-08-09, wrapped in `\camrev{}`; dispositions in
  `paper/review_notes_2026-08-09.md`.
- Ahmetcan ask list, reduced to TWO: the CLD3 prediction label space in the
  subset columns (R1.HOLD-1) and the Table 5 fastText K=10 diagnosis
  (R1.HOLD-8). CLOSED since: the DSL-ML source/split (answered by the
  2026-09-02 research + vetting, applied in R1) and the UDHR-subset FPR
  1.06e-5 (a measured decimal-exponent reading; the cell is superseded by the
  regenerated subset columns anyway).
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
