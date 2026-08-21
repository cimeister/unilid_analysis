# Session Status

Snapshot, 2026-08-21. Two workstreams run in parallel: correcting and
regenerating the GlotLID-C numbers, and regenerating the WiLI-trained models.
The open-source package fix is done and sits in PR #3.

## Read these first

- `paper/PAPER_EDITS_pending.md`: the itemized paper edit list, marked applied or
  blocked. **14 edits are already applied to `submission.tex`**, wrapped in a
  `\corrrev{}` macro (blue) so this round stays separable from the camera-ready
  `\camrev{}` pass. Accept all with `\newcommand{\corrrev}[1]{#1}`.
- `~/.claude/plans/this-session-focuses-on-shimmering-dusk.md`: the approved WiLI
  training plan, revised after an adversarial review.
- `EXPERIMENTS_RESULTS.md`: entries dated 2026-08-17 through 2026-08-21 at the top.
- `RERELEASE_PLAN.md`, `EXPERIMENTS_PLAN.md`, `CODE_CHANGES_2026-08-17.md`,
  `OPEN_SOURCE_STATUS.md`.

## Running

| job | what |
|---|---|
| 3138626 / 3138627 / 3138628 | WiLI retrains: 100k base, DeepSeek3.2, Qwen3 |
| 3127704 | `gate_variants topk`, full-pool candidate banking (GlotLID-C) |
| 3117575 / 3117576 | DeepSeek3.2 and Qwen3 full-pool evals, chained on their retrains |

## Settled, with the numbers

- **c = -17** for the corrected GlotLID-C model. Round grid {-15,-17,-19,-21};
  the pre-registration was exact (clamp counts 317 / 1,655 / 1,940 / 1,940 and the
  selection). 1,655 of 1,940 rows clamped, 285 already below.
- **Corrected base, full pool**: macro F1 0.9292 to 0.9327, macro FPR 2.03e-5 to
  2.02e-5. **Plus clamp**: 0.9419, against the released model's 0.9421 at c = -21.
- **Tail, both views**: within-stratum 0.8875 against the released clamped 0.8928;
  global per-language F1 **0.7743 against 0.7655**, false positives into tail
  labels 8,727 against 22,522. The views disagree by construction (Exp 24).
  Author decision 2026-08-19: priorities unchanged, c stays selected on validation
  overall macro F1 under the all-strata guard.
- **Group-A thresholds regenerated**: 1,080 rows, 26 excluded, all
  `low_calibration`, matching the released expected counts exactly.
- **Both GlotLID-C variant retrains clean.** Real-token mass 1.000000. Every
  corrupted row disappeared (`azj_Latn`, `bod_Tibt`, `mya_Mymr`) and every
  coverage row persisted (`got_Goth`, `nqo_Nkoo`, `kyu_Kali`).
- **Tables regenerated**: `viterbi_vs_marginal` (.961/.933 and .961/.935),
  `lenbias-norm` (0.960 to 0.837 normalized, with the implementation check at
  agreement 1.000000), Mistral-Nemo GlotLID-C cells (0.9119 / 1.858e-5, unchanged
  to three decimals).
- **UDHR and FLORES score stages done** on the corrected model; the eval stage
  waits on the group B thresholds.
- **The WiLI instrument reproduces the paper exactly**: macro F1 0.960113,
  accuracy 0.956502, macro FPR 1.8589e-04 against .960 / 0.9565 / 1.859e-4.

## Facts a new session must not re-derive

- **The special-token defect is `sp`-only.** The pure-Python `em` path never had
  it. DSL-ML is `em`-trained (author confirmation 2026-08-19) and needs nothing.
- **All three WiLI models are `sp`-trained and carry the defect** (0.800000
  special mass per row), measured, not asked.
- **Base vocabularies split two ways.** An LLM tokenizer is carried unchanged
  across corpora: DeepSeek3.2 and Qwen3 are byte-identical between their WiLI and
  GlotLID-C containers. A model with no supplied tokenizer has its vocabulary
  trained on the corpus: the WiLI 100k base shares only 24,357 of 100,000 tokens
  with GlotLID-C's. The base vocabulary is untouched by the defect, so extracting
  a container's vocabulary is exact.
- **`train.py` silently trains a fresh vocabulary if `--results-dir` and
  `--base-tokenizer-path` are not both passed**, and reports success. This is the
  most dangerous trap in the training path.
- **`UNILID/eval.py` computes no macro FPR.** Use `analysis/wili_eval.py`, whose
  FPR matches the convention in `analysis/paper_eval.py`.
- **`analysis/variant_plateau_outliers.py` cannot be used on WiLI**: it hardcodes
  the GlotLID-C counts, and every WiLI language has exactly 500 lines, so
  `log10(N_L)` has zero variance and the regression has no slope to fit.
- **WiLI language order is the tokenizer-filename sort**, which puts `nds-nl`
  before `nds` and first diverges from `sorted()` at index 146. Align matrices by
  `langs.index(lang)`, never by position.
- **WiLI's longest training line is 40,578 bytes and 101 lines exceed the
  4,192-byte upstream cap**, so the fp64 EM overflow is not excluded there. An
  earlier note claiming it was impossible was wrong.
- **A row whose plateau anomaly survives a retrain is vocabulary coverage; one
  that disappears was corruption.**
- **Latency is closed** (author confirmation 2026-08-19): it varies with hardware
  and label-set size, and is not revisited.

## Open decisions

- **Mistral / LLaMA3.2 / LLaMA2 base tokenizers.** No container anywhere, and the
  cached HuggingFace copies are dangling symlinks with no blobs. Author decision
  2026-08-21: use `mistral-community/Mistral-7B-v0.2`,
  `meta-llama/Llama-3.2-1B`, `meta-llama/Llama-2-7b-hf`, with vocabulary-size
  sanity checks, and state that they are unconfirmed. `\unilid-Mistral` cannot be
  Mistral-Nemo: the published rows differ, 0.921 against 0.958.
- **`tab:vocab_size_efficiency`**: no 10k/20k/50k/200k container exists on the
  releases page. Base vocabularies are trained here, gated by the 100k
  reproducibility check.
- **`tab:samples-accuracy`**: needs the seed count behind its standard deviations.
- **`tab:lenbias-delta`**: same golden-subset instrument question as
  `lenbias-norm`, not yet decided.
- **Group B (high-entropy) membership** is read from `lang_diagnostic.csv`,
  computed on the released model. `build_release_calibration.py` asserts the
  current four and aborts until re-derived.
- **The fastText halves** of the WiLI tables: unaffected by the defect, so
  carrying them is defensible, but it must be a stated choice.
- Whether the Apertus 200k and 131k variants are published; whether the package
  offers users a migration for pre-0.3.0 models.

## Known damage, recorded

The released model's E2 scored artifacts (`external_bench/scored_udhr.npz`,
`scored_flores.npz`, from 2026-08-07) were overwritten and then deleted. The
directory is on scratch, not store-backed, so nothing published was lost, but
those two files need regenerating if the released model's E2 numbers are ever
recomputed. `analysis/external_bench_eval.py` now writes a non-default model's
arrays to `external_bench/scored_<model stem>/`.

## Carried over: camera-ready

- Edit pass applied 2026-08-09, wrapped in `\camrev{}`; dispositions in
  `paper/review_notes_2026-08-09.md`.
- Ahmetcan ask list, reduced: the subset-evaluation script or command; the
  UDHR-subset FPR of 1.06e-5; the WiLI training method (now answered by
  measurement); the DSL-ML competitor-score source and split.
- The user compiles the PDF (no icml2026.sty here).
