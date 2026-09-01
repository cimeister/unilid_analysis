# Paper edits required by the corrected model

Concrete edit list. Each row names the site, the current text, and either the new
value or the run it waits on. Line numbers are `paper/submission.tex`.

**Line numbers in entries A2.1 through A2.12 are pre-removal.** A2.13 deleted 15
lines, so every site after the deletion has moved up. Measured anchors for the
appendix sites quoted in earlier entries: `:1247` is now `:1232`, `:1253` is now
`:1238`, `:1261` is now `:1246`, `:1298` is now `:1283`, the Mistral-Nemo transfer
paragraph starting at `:1391` now starts at `:1376`, and the remaining-errors
paragraph at `:1409` now starts at `:1394`. Sites before `:1182` are unmoved.
Every edit in this file was matched on its text rather than on a line number, so
the numbers are navigation aids, not addresses.

No disclosure of selection history is included: the preprint is unreleased, so
the paper states the procedure and the constant it selected.

**The unseen-token constant is settled: c = -17** (job 3117581, round grid
{-15,-17,-19,-21} chosen by the rule the published grid follows; pre-registered
clamp counts and selection both hit exactly). 1,655 of 1,940 rows are clamped and
285 already lie below it.

Status vocabulary used below: APPLIED (in the working tree, wrapped in
`\corrrev{}`), DONE / CLOSED (nothing further to do), PENDING-DECISION (the number
or the framing needs the author). Nothing is BLOCKED or IN-PROGRESS any more.

## Census, 2026-08-25

| group | count | state |
|---|---|---|
| A, the 2026-08-19 round | 11 items | all applied |
| A2.1 -- A2.16, the corrected round | 16 entries | all applied |
| B1 -- B9, the blocked list | 9 items | 8 DONE, 1 CLOSED by ruling (B8) |
| PD-1 -- PD-9, the author's decisions | 9 items | all decided: 4 applied to the paper, 5 closed with no paper edit |

**Nothing in this file is waiting on a measurement this repository can make.**
What remains, in full:

1. **Two co-author asks** (C3): the \cld-subset evaluation script or command,
   which after 2026-09-01 governs only the three carried variant rows and is what
   would reopen PD-3 and PD-5; and the DSL-ML competitor-score source and split.
2. **`tab:lid_main`'s caption split**, which stays as written because PD-3 closed
   with the variant rows carried. One small question rides with it: the
   UDHR-subset FPR column's lowest value is now \glotlid's 2.09e-5 and no cell in
   that column is bold, which was already true before this round (A2.16). Bolding
   it or leaving the column as published is an author's call.
3. **The noise analysis**: removed by ruling (A2.13), and the author will re-add
   `tab:noise_robustness` manually now that the co-author has rerun the
   experiments. Recorded rather than open; no session work should touch
   `paper/tables/noise_robustness.tex`.

Resolved and kept here only as pointers: **PD-5r**, closed 2026-08-24 by reverting
the Mistral-Nemo GlotLID-C FPR cell to `\camrev{1.84e-5}` so the row is fully
carried (see the PD-5r resolution section below), and the **UDHR-subset FPR of
1.06e-5**, retired on 2026-09-01 when that cell was regenerated as 5.73e-5.

Everything else in this file is a record of work that is finished.

---

## A2. APPLIED 2026-08-24 (this round), working tree

Every cell and every prose span below is wrapped in `\corrrev{}`. Sources:
`outputs/rerelease/wave_2026-08-24_compilation.md` (WiLI wave, 14 SLURM jobs, all
`COMPLETED 0:0`), `outputs/rerelease/corrected_chain_2026-08-24.md` (GlotLID-C
chain), and the per-run JSON under `outputs/rerelease/` and the `.tex`/`.md`
fragments under `outputs_corrected_round/tables/`. Every cell was read back from
the JSON or fragment rather than from a summary. Brace balance re-checked per
file after editing: every edited file balances, and `submission.tex` keeps its
pre-existing delta of 1 (line 542, unchanged by this round).

### A2.1 `tab:unilid_llm_comparison` (`paper/tables/unilid_llm_comparison.tex`)

All seven \unilid rows moved to the retrained models. fastText row untouched.
Source JSON: `outputs/rerelease/wili_eval_<model>_fp64.json`, 117,500 WiLI test
lines each.

| row | F1 old -> new | FPR old -> new | model |
|---|---|---|---|
| `\unilid (base)` | 0.960 -> 0.960 | 1.859e-4 -> **1.863e-4** | `wili_100k_500_fp64` (0.9600884 / 1.862884e-4) |
| `\unilid-Mistral-Nemo` | 0.958 -> **0.959** | 1.925e-4 -> **1.894e-4** | `mistralnemo_wili_fp64` (0.9588978 / 1.893799e-4) |
| `\unilid-Mistral` | 0.921 -> **0.920** | 3.365e-4 -> **3.380e-4** | `mistral7b_v02_wili_fp64` (0.9202152 / 3.379524e-4) |
| `\unilid-LLaMA3.2` | 0.954 -> 0.954 | 2.084e-4 -> **2.081e-4** | `llama32_1b_wili_fp64` (0.9543169 / 2.081469e-4) |
| `\unilid-LLaMA2` | 0.911 -> **0.910** | 3.698e-4 -> **3.733e-4** | `llama2_7b_wili_fp64` (0.9096351 / 3.732679e-4) |
| `\unilid-DeepSeek3.2` | 0.955 -> 0.955 | 2.042e-4 -> **2.048e-4** | `deepseek_v3.2_wili_fp64` (0.9551710 / 2.048372e-4) |
| `\unilid-Qwen3` | 0.949 -> **0.948** | 2.310e-4 -> **2.341e-4** | `qwen3_8b_wili_fp64` (0.9481253 / 2.341153e-4) |

Bolding unchanged: the base row is still the best F1 and the best FPR in the
table (0.960 against 0.959; 1.863e-4 against 1.894e-4).

**Caption**, one added sentence pair, the minimal accurate statement:
"For \unilid-Mistral and \unilid-LLaMA2 we could not confirm which repository the
original variant's base tokenizer came from. From the base vocabulary of each we
dropped the entries that contain a carriage return, 51 and 24 respectively,
because the SentencePiece seed-vocabulary writer does not accept them."

Not put in the caption, and not put in an appendix note either: that all seven
rows are models trained in this round. Every number in the paper is a measurement
of the authors' own models, so saying it for one table would read as a
distinction that does not exist. The two facts a reader cannot reconstruct are
the unconfirmed repository and the dropped entries, and those are stated.
Counts and vocabulary sizes: 51 dropped / 31,950 entries (Mistral,
`wili_mistral7b_v02_base_convert.json`), 24 dropped / 31,977 entries (LLaMA2,
`wili_llama2_7b_base_convert.json`); both records carry
`identity_rebuild_byte_identical: true` and `refused_entries_remaining: 0`.

### A2.2 `tab:length_accuracy` (`paper/tables/length_accuracy.tex`)

\unilid column only, from `outputs/rerelease/wili_length_accuracy_wili_100k_500_fp64.json`
(retrained 100k). Sample counts unchanged, and they still reproduce the published
column exactly (7,845 / 26,652 / 31,449 / 29,494 / 18,142 / 3,918 / 117,500).

| bucket | old -> new |
|---|---|
| 101--150 | 93.10 -> **93.04** |
| 151--200 | 94.17 -> **94.11** |
| 201--300 | 95.86 -> **95.83** |
| 301--500 | 96.78 -> **96.79** |
| 501--1000 | 96.53 -> **96.60** |
| 1000+ | 96.53 -> **96.61** |
| Overall | 95.65 -> **95.64** |

Bolding unchanged: \unilid is still above \fasttext in every row, by 0.31 pp at
the narrowest (1000+: 96.61 against 96.30).

### A2.3 `tab:tatoeba_udhr_comparison` (`paper/tables/tatoeba_udhr_comparison.tex`)

\unilid row only. Tatoeba from `wili_tatoeba_wili_100k_500_fp64.json`
(0.4199704 / 9.230050e-4, 201 languages, 11,848,300 rows); UDHR from
`wili_udhr_wili_100k_500_fp64.json` (0.8659475 / 5.860380e-4, 142 languages,
10,027 rows).

| cell | old -> new |
|---|---|
| Tatoeba F1 | 0.414 -> **0.420** |
| Tatoeba FPR | 9.61e-4 -> **9.23e-4** |
| UDHR F1 | 0.868 -> **0.866** |
| UDHR FPR | 5.88e-4 -> **5.86e-4** |

Language counts in the column headers (201, 142) are unchanged and were
re-confirmed against both JSONs.

### A2.4 `tab:vocab_size_efficiency` (`paper/tables/vocab_size_efficiency.tex`)

F1 and FPR for all five sizes, from
`outputs/rerelease/wili_eval_wili_<size>_defaults_fp64.json`.

| vocab | F1 old -> new | FPR old -> new |
|---|---|---|
| 10k | 0.945 -> **0.944** | 2.514e-4 -> **2.556e-4** |
| 20k | 0.951 -> **0.950** | 2.278e-4 -> **2.303e-4** |
| 50k | 0.957 -> 0.957 | 2.019e-4 -> **2.015e-4** |
| 100k | 0.960 -> 0.960 | 1.859e-4 -> **1.863e-4** |
| 200k | 0.9606 -> 0.9606 | 1.8382e-4 -> **1.8418e-4** |

Latency and Samples/s: left as published, with a caption sentence saying so.
They cannot be regenerated here: `analysis/wili_eval.py` writes no timing field
and no throughput harness was run in this wave.

**Caption**, added: "Macro F1 and macro FPR come from base tokenizers retrained
for this table; at 100k the retrained vocabulary contains the same 100,000 tokens
as the published one, in a different order at 132 of the positions. The latency
and throughput columns are the original measurements and were not re-measured on
the retrained models."

The 132 is a measurement made for this edit, not a carried claim:
`wili_vocab_repro_check.json` records only the first divergence (index 18,484)
and full set overlap, so the ordered token lists of
`results_wili_100k_defaults_fp64/tokenizers/langspec_base_tokenizer.json` and of
the stored `wili_assets/wili_100k_500.unilid` base were loaded and compared
directly: sets equal, 132 of 100,000 positions hold a different token. Consistent
with this, `wili_eval_wili_100k_defaults_fp64.json` and
`wili_eval_wili_100k_500_fp64.json` agree to 15 decimal places on macro F1, which
is what an identical token set predicts, since a per-token log-probability does
not depend on the token's index.

### A2.5 `tab:calibrated_nemo` (`paper/tables/calibrated_nemo.tex`)

From `outputs_corrected_round/tables/mistralnemo_eval.{md,tex}` (job 3158825,
five stages, all gates passed; clamp 1,431 of 1,940 at c = -17).

| row | full-pool F1 | Ma-FPR (x1e5) | held-out F1 |
|---|---|---|---|
| retrained baseline | 0.913 -> **0.912** | 1.79 -> **1.86** | 0.897 -> **0.895** |
| + unseen-token constant | 0.940 -> **0.935** | 1.71 -> **1.79** | 0.928 -> **0.923** |
| + re-examination (calibrated) | 0.954 -> **0.950** | 1.56 -> **1.62** | 0.947 -> **0.944** |

Caption left as it stands: its only quantitative claim, that the retrained
baseline is within 0.002 macro F1 of the \cref{tab:lid_main} variant row, still
holds (measured difference -0.0001 against that row's 0.912).

### A2.6 `tab:lid_main` calibrated row, GlotLID-C

`\camrev{\textbf{.957}}` -> `\corrrev{\textbf{.956}}` (corrected 0.9564 against
the released 0.9569). The FPR cell keeps its `\camrev{\textbf{1.77e-5}}`: the
corrected value is 1.7745e-5, which prints as the same 1.77e-5, so nothing in
that span is superseded. Bolding unchanged (still the best F1 in the column).
The `\unilid` row's GlotLID-C cells already carried `\corrrev{.933}` /
`\corrrev{2.02e-5}` from the 2026-08-19 round and are exactly this run's
corrected baseline (0.9327 / 2.0187e-5); re-verified, not re-edited.

### A2.7 `tab:calibrated_heldout` (`paper/tables/calibrated_heldout.tex`)

This is item B3's first half, unblocked by the gate apply. Cell-for-cell from
`outputs_corrected_round/tables/paper_eval_appendix.tex` and the 4-decimal table
in `outputs_corrected_round/tables/paper_eval.md`, judge part, 27,002,441 lines.

| cell | old -> new |
|---|---|
| \unilid F1 / FPR | 0.912 / 2.04 -> **0.916 / 2.03** |
| calibrated \unilid F1 / FPR | 0.950 / 1.77 -> **0.949 / 1.78** |
| \fasttext F1 / FPR | 0.933 / 2.72 -> unchanged (bit-identical input array in both rounds) |
| bootstrap vs \unilid | +0.038 [+0.033, +0.043] -> **+0.034 [+0.029, +0.038]** |
| bootstrap vs \fasttext | +0.017 [+0.011, +0.022] -> **+0.016**, interval unchanged |

### A2.8 `tab:lenbias-norm` (`paper/tables/lenbias-norm.tex`)

Item 11 of the 2026-08-19 round, held back then because it was regenerating.
The regeneration is complete: `outputs_corrected_round/tables/lenbias_norm.{md,json}`,
250,000 lines (the test half of the seed-42 500,000-line draw, the golden
subset), Raw rescore reproduces the plain scorer at agreement 1.000000, which is
the implementation check the table exists to report.

| bucket | N old -> new | Original | Raw rescore | Normalized old -> new |
|---|---|---|---|---|
| <30 | 27,328 -> 13,708 | 0.792 -> 0.795 | 0.792 -> 0.795 | 0.566 -> **0.494** |
| 30--75 | 177,256 -> 88,503 | 0.951 -> 0.951 | 0.951 -> 0.951 | 0.842 -> **0.776** |
| 75--150 | 195,267 -> 97,861 | 0.978 -> 0.977 | 0.978 -> 0.977 | 0.925 -> **0.883** |
| 150--300 | 87,096 -> 43,566 | 0.987 -> 0.988 | 0.987 -> 0.988 | 0.966 -> **0.946** |
| 300+ | 13,053 -> 6,362 | 0.995 -> 0.994 | 0.995 -> 0.994 | 0.991 -> **0.986** |
| Overall | 500,000 -> 250,000 | 0.960 -> 0.960 | 0.960 -> 0.960 | 0.885 -> **0.837** |

Caption: "Evaluated on a $500k$-sample subset" becomes "Evaluated on the
$250k$-line test half of a $500k$-line uniform sample"; the two quoted numbers
follow the table (0.960 to 0.837; <30 chars 0.795 -> 0.494).

### A2.9 Prose in `submission.tex`

| line | old | new |
|---|---|---|
| 348 | "from `\corrrev{.933}` to .957" | ".957" -> **`\corrrev{.956}`** |
| 847 | "from `\corrrev{0.933}` to 0.957" | "0.957" -> **`\corrrev{0.956}`** |
| 852 | "the paired bootstrap interval of the improvement over \unilid is $+0.038$ with 95\% interval $[+0.033, +0.043]$" | **$+0.034$, $[+0.029, +0.038]$** (matches `tab:calibrated_heldout`) |
| 974 | "more than doubling the Macro F1 of \fasttext on Tatoeba (0.414 vs. 0.160)" | **0.420**; "more than doubling" still holds (0.420 / 0.160 = 2.6) |
| 989 | "all are `\camrev{within 0.025 macro F1}` of base UniLID on GlotLID-C" | **`\corrrev{within 0.03 macro F1}`**. The published claim is now false: the corrected base cell is .933 and the lowest variant cell is .904, a gap of 0.029. 0.03 holds under both the current variant cells (.912/.909/.904) and the corrected retrains (0.9119/0.9089/0.9049, gap 0.028), so it survives the B8 decision either way |
| 1198 | "0.002 higher macro F1 (0.931 against 0.929) and 0.001 higher accuracy at approximately $2\times$ the inference cost" | **"(0.935 against 0.933) and the same accuracy to three decimals, at approximately $1.7\times$ the inference cost"**. Completes 2026-08-19 item 10, whose table edit landed but whose prose did not |
| 542 (footnote) | "0.002 higher macro F1 and 0.001 higher accuracy ... but took roughly twice as much time" | **"and the same accuracy to three decimals ... roughly 1.7 times as much time"**. Same item 10 |
| 1397 | "(two languages whose trained unseen-token values already lie below $c = -21$ are left unchanged)" | **"(509 languages whose trained unseen-token values already lie at or below $c = -17$ are left unchanged)"**, from the clamp line of job 3158825: 1,431 of 1,940 clamped, 509 already at or below. Closes item B7 |
| 1401--1405 | "raises the variant's macro F1 from 0.913 to 0.954 ... $+0.050$ with 95\% interval $[+0.044, +0.057]$ ... ($+0.041$ against $+0.028$ in full-pool macro F1)" | **"from 0.912 to 0.950 ... $+0.0489$ with 95\% interval $[+0.0424, +0.0555]$ ... ($+0.039$ against $+0.024$)"**. The two gains were corrected from $+0.038$ / $+0.023$ to $+0.039$ / $+0.024$ on 2026-08-24, see A2.11 |

The bootstrap interval at 1401--1405 is quoted at four decimals, as
`mistralnemo_eval.md` reports it, rather than rounded to three. Rounding
+0.0555 to three decimals is a coin flip on the last digit and the source has no
more precision to settle it; the paper already quotes a four-decimal interval two
paragraphs earlier ($+0.0002$, $[-0.0003, +0.0006]$), so this is in house style.
The two full-pool gains are differences computed at full precision and then
rounded, the same convention as every other derived number in this round:
0.9504030563 - 0.9118863405 = 0.0385167 -> $+0.039$, and
0.9564272222 - 0.9327099739 = 0.0237172 -> $+0.024$. My first pass took them as
differences of the printed three-decimal cells (0.950 - 0.912; 0.956 - 0.933),
which gives $+0.038$ and $+0.023$; that convention is wrong here. It is
inconsistent with the bootstrap cells in the same round, which are full-precision,
and the published $+0.041$ / $+0.028$ are consistent with both conventions, so
they cannot be cited as evidence for either.

**Prose deliberately left unchanged**, each with the reason:

| line | text | why left |
|---|---|---|
| 838 | "reduces FPR by roughly 25\% compared to \fasttext (`\corrrev{2.02e-5}` vs 2.71e-5)" | still true: 1 - 2.0187/2.7063 = 25.4\%. \fasttext's 2.7063e-5 is bit-identical in both rounds |
| 849 | "lowering macro FPR from `\corrrev{2.02e-5}` to 1.77e-5" | corrected calibrated FPR is 1.7745e-5, which prints as 1.77e-5 |
| 864--865 | UDHR and FLORES sentence | **updated after all**, see A2.10 |
| 855--859 | "mean per-language F1 rises from 0.515 to 0.780 ... 0.628 to 0.892" | **updated**, see A2.14 |
| 867--869, 1359--1366 | CommonLID 0.845 / 0.860 / 0.723 / 0.715 | **updated after all**, see A2.12 |
| 1247, now 1232 | script gaps "F1 0.940 vs. 0.946", "Greek -0.248, Hebrew -0.227, Devanagari -0.121, Bengali -0.100" | **updated**, see A2.14 |
| 1253 | resource-tier claim "matches or exceeds \fasttext for languages with 500+ training samples" | still true under the corrected within-stratum column, but the table it describes is not applied, see PD-2 |
| 1261 | "on average 0.17 tokens shorter" | B9, `tab:lenbias-delta` |
| 1279--1281 | high-entropy group "Scots, Banjar, Aragonese, and West Flemish" | re-measured on the corrected model and unchanged (`groupb_rederivation.json`) |
| 1288--1290 | "the 26 of the 1,080 languages ... receive no threshold" | `build_release_calibration` on the corrected model reports exactly 1,080 group A rows with 26 excluded |
| 1298 | "held-out macro F1 rises from 0.912 to 0.930" | B6; the second number has no corrected counterpart yet, see below |
| 1354--1357 | balanced-draw 0.978 against 0.981 | no corrected balanced-draw run |
| 1376 | "22,404 to 79,113" | development-time measurement on the released model, not regenerated |
| 1379--1391 | the three within-family alternatives, including 0.953 and "+0.0002" | development-time measurements, not regenerated |
| 1409--1413, now 1394--1398 | "926,299 wrong predictions ... 99.2\% ... 88.6\% ... 31,113" | **updated whole**, see A2.14 |
| 1394--1396 | "(an independent training run; the published row in \cref{tab:lid_main} is unchanged)" | see PD-5, a framing question rather than a number |

### A2.10 `tab:lid_main` UDHR and FLORES cells, \unilid and calibrated rows

Applied 2026-08-24, after the two `external_bench_eval` blockers were cleared
(addendum B of `outputs/rerelease/corrected_chain_2026-08-24.md`; both eval stages
exited 0 against `external_bench/scored_glotlidc_corrected/`). This closes B2.

| row | bench | old -> new |
|---|---|---|
| `\unilid` | UDHR | .859 / 1.43e-4 -> **.856 / 1.52e-4** (0.8560 / 1.52e-4) |
| `\unilid` | FLORES | .932 / 2.78e-4 -> **.931 / 2.83e-4** (0.9313 / 2.83e-4) |
| `\unilid (calibrated)` | UDHR | .838 / 2.08e-4 -> **.842 / 2.03e-4** (0.8419 / 2.03e-4) |
| `\unilid (calibrated)` | FLORES | .933 / 2.91e-4 -> **.932** / 2.91e-4 (0.9324 / 2.91e-4) |

The calibrated FLORES FPR keeps its printed 2.91e-4 and is left unwrapped: the
corrected value prints the same. Bolding unchanged in all four columns (\glotlid
holds the best UDHR and FLORES cells).

Prose: `:864-865` "calibration lowers macro F1 (0.859 to 0.838) ... changes little
(0.932 to 0.933)" becomes **(0.856 to 0.842) ... (0.931 to 0.932)**, inside the
existing `\camrev{}` span. Both claims survive: calibration still lowers UDHR
macro F1 and still moves FLORES by 0.001.

Recorded direction, for the internal record and not for the paper: on UDHR the
correction lowers the baseline row (-0.003 F1) and raises the calibrated row
(+0.004 F1), the opposite sign to the internal GlotLID-C pool, where the
correction lifted the baseline (+0.0035) and left the gate flat (-0.0005). On
FLORES no cell moves by more than 0.001 F1. Both release gates, base and
calibrated, now pass at exact equality for the corrected generation, so nothing
further is expected to move these four cells.

### A2.11 Fixes from the adversarial verification, applied 2026-08-24

Source: `outputs/rerelease/paper_edit_verification_2026-08-24.md`. Every numeric
cell of A2.1 through A2.10 was re-derived independently and confirmed, including
the 132-position vocabulary re-measurement and the arithmetic behind every
surviving claim. Three findings, all applied.

| # | site | old -> new |
|---|---|---|
| V1 | `:1404` | "($+0.038$ against $+0.023$ in full-pool macro F1)" -> **"($+0.039$ against $+0.024$)"**. Full-precision differences: 0.0385167 and 0.0237172. See the convention note under A2.9 |
| V2 | `:1298` | "held-out macro F1 rises from 0.912 to 0.930" -> **"rises from `\corrrev{0.916}` to 0.930"**. The 0.912 was the released judge-part baseline; `tab:calibrated_heldout` now prints 0.916 for the same quantity, so the paper was printing two values for one number. The 0.930 half is still B6 |
| V3a | `tab:samples-accuracy`, 500-samples row | 95.65 $\pm$ 0.00 -> **`\corrrev{95.64}` $\pm$ 0.00**. This row is the all-500-samples case, that is the whole WiLI training split, so it is the same deterministic quantity as `tab:length_accuracy`'s Overall cell and needs none of the seeds PD-7 asks for. The standard deviation stays 0.00, which is what a single deterministic run gives |
| V3b | `:953`, `fig:samples-accuracy` plot data | `(500,95.65) +- (0,0.00)` -> **`(500,95.64) +- (0,0.00)`**. The same quantity again, in the pgfplots coordinate list behind the figure |

**V3b carries no `\corrrev{}` marker.** It sits inside a pgfplots `coordinates {...}`
list, where a `\textcolor` wrapper is not valid syntax. It is the one number in
this round that changes without the blue marking, recorded here so the sweep for
`\corrrev` spans does not read the file as complete.

`tab:noise_robustness`'s p=0\% \unilid accuracy cell (0.957) was the fourth site of
the same quantity and was **not** touched at the time, because that table was then
on hold. **Superseded later the same day:** the author ruled the noise analysis out
of the paper, so the table and every sentence about it are gone (A2.13) and this
mismatch no longer exists in the document.

Two further occurrences of 95.65 sit in `submission.tex` at `:819` and `:1164` and
were left alone: both lines are commented out, so neither renders. They are an
older draft of the same figure and table.

### A2.12 `tab:commonlid` and its prose, applied 2026-08-24

Item B4, unblocked: the corrected CommonLID chain completed with its binding
gates passing. Source, cross-checked cell by cell before editing:
`outputs_corrected_round/tables/commonlid_calibrated.md` (373,230 rows, 109 tags,
0 rows empty after preprocess, 0 rows with fewer than 5 saved candidates;
FLOOR_TARGET -17.0 read from the corrected fingerprint).

| row | accuracy old -> new | tag-level macro F1 old -> new | measured |
|---|---|---|---|
| `\unilid` | 0.845 -> **0.848** | 0.723 -> **0.722** | 0.8476 / 0.7218 |
| `\unilid`, unseen-token constant only | 0.849 -> **0.851** | 0.718 -> **0.720** | 0.8512 / 0.7203 |
| calibrated `\unilid` | 0.860 -> **0.862** | 0.715 -> **0.717** | 0.8624 / 0.7171 |

Caption: out-of-set line counts 32,901 -> **32,525** and 25,884 -> **25,994**
(measured baseline 32,525 lines over 1,095 distinct labels, gated 25,994 over
810).

Prose, both sites, every quoted number replaced: `:867-869` and `:1359-1366`
"from 0.845 to 0.860 ... from 0.723 to 0.715 ... (32,901 to 25,884 lines)" become
**0.848 to 0.862 ... 0.722 to 0.717 ... (32,525 to 25,994 lines)**. A sweep of
`submission.tex` and every table for 0.845 / 0.849 / 0.860 / 0.723 / 0.718 /
0.715 / 32,901 / 25,884 found no other CommonLID site; the one further hit,
`tab:tatoeba_udhr_comparison`'s 0.849, is \fasttext's UDHR macro F1 and is
unrelated.

All three surrounding claims survive and no wording changed: calibration still
raises accuracy (0.848 to 0.862), still lowers tag-level macro F1 (0.722 to
0.717), and out-of-set predictions still fall (32,525 to 25,994). Neither
sentence quotes a delta, so the changed magnitudes (accuracy +0.014 rather than
+0.015; tag-level macro F1 -0.005 rather than -0.008; 6,531 fewer out-of-set
lines rather than 7,017) appear nowhere in the text. The caption's "decreases
slightly" is still supported by a drop of 0.005.

Recorded for the internal record: the module's own four informational
comparisons against the released model all exceed its EVAL_GATE_TOL of 0.0005
(baseline accuracy +0.0024, baseline tag F1 -0.0010, floor-21 tag F1 +0.0022,
floor-21 accuracy +0.0021). That is the expected cross-model difference for a
non-default model, not a reproduction failure, and the module states so itself.

### A2.13 `tab:noise_robustness` removed from the paper, 2026-08-24

Author ruling, verbatim: *"The script for creating the data for the noise tables
got deleted. It may be best just to remove this analysis."* and *"See answer
above about this table. It may be best to delete it."*

Four spans were deleted. This is a deletion, not a `\corrrev{}` wrap, so each is
recorded verbatim here and is recoverable from this entry alone.

**1. `:1182`, the table input.**

```
\input{tables/noise_robustness}
```

**2. `:1202-1215`, the whole appendix subsection.**

```
\subsection{Robustness to Orthographic Noise}
\label{app:noise_robustness}

To evaluate robustness under realistic input corruption, we apply stochastic character-level perturbations to the WiLI test set: with probability $p$, each non-whitespace character is independently replaced with another character drawn uniformly from the inventory of characters observed in the WiLI training set. We evaluate at $p \in \{0\%, 5\%, 10\%, 25\%, 50\%\}$. Example inputs at $p=5\%$ and $p=10\%$ are shown below; \cref{tab:noise_robustness} reports accuracy, macro F1, and macro FPR for \unilid and \fasttext on the perturbed test sets.

\paragraph{Example perturbations.}
\begin{description}
    \item[Original] \emph{Anton (or Antonius) Maria Schyrleus (also Schyrl, Schyrle) of Rheita (1604--1660) was an astronomer and optician. He developed several inverting and erecting eyepieces\ldots}
    \item[$p=5\%$] \emph{Anton (or Antonius) Maria Schyrlezs $\ldots$ Anton\'in $\ldots$ astronmmer and optijian\ldots}
    \item[$p=10\%$] \emph{Anton (or Aston,us) MDria SchyrKeus $\ldots$ of iheith\ldots}
\end{description}

At low noise ($p < 10\%$), \unilid maintains a small edge in F1 and accuracy. At moderate-to-high noise ($p \ge 25\%$), \fasttext degrades more gracefully than \unilid: the character n-gram representations underlying \fasttext appear to absorb localized character corruptions more robustly than \unilid's segmentation-based scoring, where corrupted characters can fragment otherwise-high-probability tokens. This points to a possible avenue for future work --- explicit noise-aware token scoring or character-level smoothing within the \unilid framework.
```

**3. `:428`, the clause in the introduction's "This work" paragraph.** The
sentence read "On out-of-domain inputs it gives partial improvements; ...
(\cref{sec:results})."; the deleted middle is

```
; \camrev{on orthographic noise \unilid and \fasttext are roughly tied at corruption rates up to 10\%, and \fasttext degrades less at higher rates}
```

and it now reads "On out-of-domain inputs it gives partial improvements
(\cref{sec:results})."

**4. `:986`, the second half of the "Robustness Analysis" paragraph.**

```
We additionally evaluate robustness to character-level corruption on WiLI by stochastically replacing non-whitespace characters at rates of 5\% and 10\%. Full setup details and results can be found in \cref{app:noise_robustness}. \camrev{In short, at corruption rates up to 10\% the two systems are roughly tied, with \unilid slightly ahead; at 25\% and above \fasttext degrades less (accuracy 0.906 against 0.824 at the 25\% rate; \cref{tab:noise_robustness}).}
```

The paragraph now ends at "such as determining the language of social media
posts." and is about input length only. Its `\paragraph{Robustness Analysis.}`
heading still names what is left, robustness to short inputs, so it was not
touched.

**Kept deliberately: `:424-425`, the related-work paragraph "Domain shift and
orthographic noise".** It states a challenge in the literature with its own
citations, it does not report our analysis, and the introduction's "the first
two challenges" counts it. Deleting it would break that count and would remove
cited background rather than a result of ours.

**No dangling references.** `\cref{tab:noise_robustness}` and
`\cref{app:noise_robustness}` appeared only inside the deleted spans; both labels
are now defined nowhere the document reaches.

**`paper/tables/noise_robustness.tex` is left on disk, unreferenced.** Nothing
`\input`s it, so it does not compile into the paper. It is kept rather than
deleted so the table survives if the author reverses this, and this entry plus
that file together restore the analysis in full.

### A2.14 PD-2 and PD-6 applied, 2026-08-24

Source, cross-checked cell by cell before editing:
`outputs/rerelease/pd_compute_2026-08-24.md`, with the corrected values read back
from `outputs_corrected_round/tables/resource_tier_fpr.md`,
`resource_tier_ntest.md`, `paper_breakdowns.md` and
`diagnostic/promoted_residual_pairs.csv`. The instrument was gated on the
released model first: all 24 published resource-tier F1 and FPR cells reproduce,
worst relative FPR gap 4.0% at the two cells the paper prints to one significant
figure, worst F1 gap 0.0004.

#### `tab:resource-tier`, five cells

| tier | UniLID F1 | UniLID FPR | measured |
|---|---|---|---|
| `<500` | 0.871 -> **0.857** | 7.2e-5 -> **6.5e-5** | 0.8572 / 6.5053e-05 |
| `500--1k` | 0.975 -> **0.973** | 1.5e-5 -> **1.0e-5** | 0.9731 / 9.8602e-06 |
| `1k--12k` | 0.990 (same) | 8.0e-6 (same) | 0.9895 / 8.0751e-06 |
| `12k--18k` | 0.997 (same) | 2.0e-6 (same) | 0.9971 / 1.9765e-06 |
| `18k--35k` | 0.992 (same) | 7.0e-6 (same) | 0.9918 / 6.7805e-06 |
| `35k+` | 0.958 (same) | 5.3e-5 -> **5.4e-5** | 0.9576 / 5.3737e-05 |

$N_{\text{test}}$ and both \fasttext columns are unchanged, and that is verified
rather than assumed: the `support` and `N` columns of the released and corrected
per-language CSVs are element-wise identical, and `pred_fasttext.npy` and
`y_true.npy` are byte-identical across the two scratch roots. The caption's claim
that \unilid matches or exceeds \fasttext above 500 training samples survives
(0.973/0.964, 0.990/0.979, 0.997/0.986, 0.992/0.981, 0.958/0.942), and `<500` is
still the exception (0.857 against 0.915).

#### `tab:script-breakdown`, seven \unilid cells and seven $\Delta$ cells

| script | \unilid | $\Delta$ |
|---|---|---|
| Latn | 0.940 -> **0.944** | -0.006 -> **-0.002** |
| Cyrl | 0.877 -> **0.880** | -0.093 -> **-0.090** |
| Arab | 0.691 -> **0.693** | -0.056 -> **-0.054** |
| Deva | 0.811 (same) | -0.121 (same) |
| Beng | 0.885 -> **0.879** | -0.100 -> **-0.106** |
| Grek | 0.677 -> **0.675** | -0.248 -> **-0.250** |
| Hebr | 0.740 -> **0.738** | -0.227 -> **-0.229** |
| Armn | 0.974 -> **0.972** | -0.012 -> **-0.014** |
| Other | 0.937 (same) | -0.036 (same) |

The Other row uses the paper's 82-language basis (0.9374), not the
`paper_breakdowns_script.tex` fragment's 84-language Other, which is why that
fragment's Other cell was not applied. The caption's claim that Greek, Hebrew and
Devanagari are the largest gaps survives: -0.250, -0.229, -0.121, with Bengali
fourth at -0.106.

#### `tab:calibrated_views`, 16 of 36 cells

Global view: `<500` 0.515 -> **0.596** and 0.780 -> **0.781**; `500--1k`
0.628 -> **0.676** and 0.892 -> **0.893**; `1k--12k` 0.891 -> **0.894** and
0.945 -> **0.944**; `18k--35k` 0.963 -> **0.962**; `35k+` 0.958 -> **0.957** and
0.957 -> **0.956**. Within-stratum view: `<500` 0.871 -> **0.857** and
0.827 -> **0.820**; `500--1k` 0.975 -> **0.973** and 0.955 -> **0.954**;
`1k--12k` 0.987 -> **0.986**; `35k+` 0.958 -> **0.957** in the calibrated column.
Every \fasttext cell and the `12k--18k` row are unchanged.

The two tables that print the same quantity now agree, which is what PD-2 asked
for: this table's within-stratum \unilid column and `tab:resource-tier`'s \unilid
F1 column are both 0.857 / 0.973 / 0.990 / 0.997 / 0.992 / 0.958. The caption's
mechanism claim survives: the views still rank the methods oppositely in the
smallest tier (global 0.596 -> 0.781, within-stratum 0.857 -> 0.820).

#### Prose, three sites

| line (post-A2.13) | old -> new |
|---|---|
| `:857-858` | "rises from 0.515 to 0.780 ... and from 0.628 to 0.892" -> **0.596 to 0.781 ... 0.676 to 0.893** |
| `:1232` | "F1 0.940 vs. 0.946" -> **0.944 vs. 0.946**; "Greek -0.248, Hebrew -0.227, Devanagari -0.121, Bengali -0.100" -> **-0.250, -0.229, -0.121, -0.106** |
| `:1238` | untouched: its only claim, that \unilid matches or exceeds \fasttext above 500 training samples, quotes no number and still holds |

#### PD-6, the remaining-errors sentence at `:1394-1398`

| quantity | old -> new | exact |
|---|---|---|
| wrong predictions | 926,299 -> **930,576** | 930,576 |
| share with a head true language | 99.2\% -> **99.1\%** | 922,578 / 930,576 = 0.9914053 |
| of those, share confused with another head language | 88.6\% -> 88.6\% | 816,947 / 922,578 = 0.8855045, a rounding boundary checked rather than assumed |
| Indonesian and Standard Malay | 31,113 -> **31,105 lines** | 31,105 |

The two other pairs the sentence names keep their places in the ranking: Standard
Arabic and Najdi Arabic 2nd in both rounds, Mandarin and Wu Chinese 9th in both.
Nothing qualitative in the sentence moved.

**Correction to this ledger's own PD-6 text.** It said the pair count was one
"which the corrected round does not report". That was wrong:
`outputs_corrected_round/diagnostic/promoted_residual_pairs.csv` and
`outputs_corrected_round/tables/promoted_residual.md` carry all twenty pairs, and
the released-model run of the same script reproduces the published 926,299 /
99.2\% / 88.6\% / 31,113 exactly, which is what gates the instrument.

### A2.15 B6, B9/PD-4 and the proximity bound, 2026-08-25

Source, cross-checked before editing: `outputs/rerelease/b6_b9_proximity_2026-08-25.md`,
with the cells read back from `outputs_corrected_round/tables/length_bias_golden.md`
and the sweep records under `outputs{,_corrected_round}/diagnostic/prox_bound_sweep.json`.
Each of the three was measured on the released model first and checked against a
published or pre-registered value before the corrected number was computed.

#### B6: closed, and the printed cell does not move

The sentence now at `:1283` reads "held-out macro F1 rises from `\corrrev{0.916}`
to 0.930". The corrected floor-21-only judge-part macro F1 is **0.9302**
(0.930168), which prints as the same 0.930. **No `\corrrev{}` wrap was added**,
following convention 1: this is a single-cell substitution inside an otherwise
carried sentence, and the printed value does not change. Recorded here so a later
reader does not re-derive it.

Instrument gate on the released model: baseline 0.9117 (0.911731) against
`paper_eval.GATE_B_ANCHORS["baseline"]`, and floor-21 0.9300 (0.929981) against
the published 0.930. Both reproduce. The corrected baseline came back at 0.9159,
which is the same 0.916 this ledger records for the V2 cell, and the corrected
baseline and gated rows reproduce `outputs_corrected_round/tables/paper_eval.md`'s
judge-part table to four decimals.

**The sentence is no longer mixed-generation**, which is what B6 was tracking. One
naming point worth keeping: the corrected chain calls its configuration "floor21"
throughout while its constant is c = -17, so the corrected 0.9302 and the released
0.9300 are the same configuration (unseen-token constant applied, no
re-examination) at different constants.

#### B9 / PD-4: `tab:lenbias-delta` rebuilt on the golden subset

Basis, per PD-4: the 250,000-line test half of the seed-42 500,000-line draw, the
same subset as `tab:lenbias-norm`, selected by the same imported helper so the two
tables cannot drift onto different lines.

| bucket | N old -> new | mean $\Delta$ | median $\Delta$ | \% fewer | \% same | \% more |
|---|---|---|---|---|---|---|
| All misclassified | 1,789,423 -> **9,906** | -0.17 -> **-0.05** | 0.00 | 24.94 -> **25.29** | 61.08 -> **52.12** | 13.99 -> **22.59** |
| `<30` | 515,094 -> **2,814** | -0.11 -> **-0.05** | 0.00 | 17.69 -> **19.76** | 74.62 -> **65.28** | 7.69 -> **14.96** |
| `30--75` | 771,812 -> **4,316** | -0.15 -> **-0.01** | 0.00 | 24.28 -> **23.96** | 62.76 -> **53.59** | 12.96 -> **22.45** |
| `75--150` | 392,549 -> **2,206** | -0.21 -> **-0.04** | 0.00 | 32.10 -> **31.14** | 47.71 -> **39.30** | 20.19 -> **29.56** |
| `150--300` | 102,497 -> **532** | -0.24 -> **-0.23** | 0.00 | 36.80 -> **39.66** | 34.85 -> **26.50** | 28.34 -> **33.83** |
| `300+` | 7,471 -> **38** | -2.71 -> **-2.13** | -1.00 -> **0.00** | 53.09 -> **44.74** | 15.11 -> **13.16** | 31.80 -> **42.11** |

The N columns of the old and new tables count over different line sets and must
not be compared term by term, which is why the caption now states the basis.

**The claim that changed, not just the number.** The published prose said the gap
grows "for longer inputs", full stop. On the corrected model the mean $\Delta$ is
flat across the three bins below 150 characters (-0.05, -0.01, -0.04) and grows
only above them (-0.23, then -2.13). Two sites were reworded to the measurement:

- prose, now `:1246`: "on average \corrrev{0.05} tokens shorter than under the
  true language, \corrrev{with that difference growing only for inputs above 150
  characters}". The published clause said "the gap"; "that difference" names the
  quantity the same sentence has just defined, which the style rules prefer to a
  bare label.
- caption: "The mean $\Delta$ is negative in every row and \corrrev{grows in
  magnitude only for inputs above $150$ characters}, while the median is zero and
  \corrrev{$52\%$} of errors leave the token count unchanged".

**"Systematic" survives and was checked rather than assumed.** The bias is still
distinguishable from zero on the corrected model: one-sample t-test against a mean
of zero gives p = 3.49e-04 (released, p = 9.54e-63), and the Wilcoxon signed-rank
test excluding zeros gives p = 1.35e-03. The effect is smaller: Cohen's d, the
mean $\Delta$ divided by its standard deviation, moves from -0.1689 to -0.0359.
The caption's "small but systematic" therefore still holds. No prose quotes a
p-value, so nothing else needed changing.

The instrument gate here is directional rather than numeric, because the published
table and this one cover different line sets: on the released model over the same
250,000 lines, every mean $\Delta$ keeps its sign, the ordering of "\% fewer"
against "\% more" holds in every row, and the four bins below 300 characters agree
with the published table to within 0.02 tokens and 1.6 points. The subset is not
an unrepresentative slice of the error set: 3.974\% of its lines are misclassified
against 3.922\% over the full test file.

One caveat carried from the source report rather than hidden: the `300+` row rests
on 38 misclassifications, and `150--300` on 532. Those two rows are the reason the
released and corrected columns differ most where the counts are smallest.

#### The proximity bound: verified, nothing changed

Recorded in the framing section below. The paper's sentence stands as written.

### A2.16 The \cld-subset right half of `tab:lid_main`, 2026-09-01

Source, cross-checked cell by cell before editing:
`outputs/rerelease/cld3_regenerated_2026-09-01.md`, section 6.7 for the cells and
section 4 for the full-precision measurements. Applied under **option (b)** of
that record's section 6.5, chosen by the coordinator as the execution of the
author's standing ruling that approximate reproduction is the bar, not as a fresh
decision.

**The mechanism, and why only one row could move.** Two ways exist to confine a
model to a \cld subset. *Refit* trains the base vocabulary on the subset's own
corpora and estimates the rows over it, which is the published procedure applied
to a smaller language set; *Restrict* takes a restricted argmax over the full
model's rows. The three variant rows have a fixed LLM tokenizer that cannot be
refitted, so Restrict is all they can use, and section 6.2 measures Restrict
missing all six published variant cells in one direction. The \unilid row was
regenerated by Refit (subset containers, SLURM 3246937/3246939/3246941 trained and
3247404/3247405/3247406 evaluated).

#### `\unilid` row, all six \cld-subset cells

| cell | column | old -> new | measured |
|---|---|---|---|
| 7 | GlotLID-C, 83 languages, F1 | .971 -> **.974** | 0.97419 |
| 8 | GlotLID-C, 83 languages, FPR | 1.63e-4 -> **1.52e-4** | 1.5150e-04 |
| 9 | UDHR, 80 languages, F1 | .992 -> **.995** | 0.99521 |
| 10 | UDHR, 80 languages, FPR | 1.06e-5 -> **5.73e-5** | 5.7332e-05 |
| 11 | FLORES, 77 languages, F1 | .997 -> **.997** | 0.99701 |
| 12 | FLORES, 77 languages, FPR | 3.29e-5 -> **3.93e-5** | 3.9343e-05 |

Cell 11 prints the same value and **is wrapped anyway**, under convention 1's
whole-regeneration rule: all six cells come from the new subset models, so the
marker means "this cell is a measurement of the subset model", not "this digit
changed". That is the same rule applied to `tab:unilid_llm_comparison` and
`tab:vocab_size_efficiency`, and the opposite of the B6 case, where a single cell
inside a carried sentence kept its printed value and took no wrap.

#### `\unilid (calibrated)` row, three F1 cells blanked

Cells 7, 9 and 11 go from .975 / .986 / .992 to **`--`**, extending the `--`
already printed in that row's three subset FPR positions. The reason, from section
6.3 and now stated in the caption: the promoted configuration `gate_flat4_prox21`
is a table of per-language thresholds over two groups defined on the full
1,940-language training distribution, and neither survives a subset model. Group B
is the four high-entropy languages `sco_Latn`, `bjn_Latn`, `arg_Latn`,
`vls_Latn`, **none of which is a \cld language**, so that half of the
configuration has nothing to act on. Group A, the gate-eligible rows, falls from
1,080 of 1,940 (56%) to 12-14 of 93-99 (13-14%). A calibration refitted on a
subset container would be a different method, and printing its output in a row
whose caption cites `\cref{sec:calibration}` would attribute numbers to a method
that did not produce them. The instrument exists (`cld_subset_eval.py --calibrated`,
default off and verified unchanged), so this is a definitional limit, not a
missing run.

The three published F1 values that go away came from a third convention anyway
(lines filtered to the subset, predictions not restricted, each bare ISO mapped to
its largest-corpus `lang_Script` variant), which is neither Refit nor Restrict.

#### Variant rows: carried, unchanged

Consistent with PD-3 and PD-5, which closed with the rows carried. Their measured
Restrict values are on record in section 6.2 if that is ever overruled:
Mistral-Nemo .977 / .995 / .995, DeepSeek3.2 .974 / .995 / .995, Qwen3 .971 /
.994 / .994, against published .972 / .992 / .994, .971 / .990 / .994 and .964 /
.984 / .989.

#### Caption

Two `\corrrev{}` insertions and two deletions, reconciled with what the caption
already said rather than appended to it. The left half's pool sentence is
untouched: it is about the GlotLID-C column and stays true. Removed as superseded:
"Its \cld-subset F1 cells restrict each benchmark to the lines whose gold label is
in the \cld subset; this convention reproduces the \unilid GlotLID-C subset cell
and matches the \unilid UDHR subset cell within 0.005" (it describes cells that no
longer exist) and "The convention behind the original rows' subset FPR values
could not be determined, so the calibrated row's subset FPR cells are omitted"
(superseded by the stronger reason now given for all six). Added:

> The \unilid \cld-subset cells come from models whose base vocabulary is trained
> on each benchmark's subset of languages, by the same procedure as the full-set
> models. The \cld-subset cells of the remaining rows are those of the original
> submission.

and, after the sentence introducing the calibrated row:

> Its \cld-subset cells are omitted because the configuration of
> \cref{sec:calibration} is not defined on a subset model: none of its four
> high-entropy languages is in any \cld subset, and the share of languages
> eligible for a re-examination threshold falls from 56\% to 13\%.

#### The editorial note was deleted, and why

The caption ended with "[Editorial note, to resolve before submission: the \unilid
UDHR-subset FPR value, 1.06e-5, is the lowest in its column, a factor of two below
\glotlid's 2.09e-5, and is awaiting confirmation against the original
computation.]" That note is about cell 10, which now reads 5.73e-5, so it pointed
at a value the table no longer prints and its premise was no longer true. Removed
verbatim here for recoverability. The anomaly it flagged is resolved by
measurement rather than by confirmation: the regenerated cell is 5.73e-5, and
section 6.1 of the source record argues the published 1.06e-5 was an exponent typo
for 1.06e-4, under which reading the two sit within a factor of two.

**One consequence needs an author's eye.** With \unilid's UDHR-subset FPR at
5.73e-5, the lowest value in that column is now \glotlid's 2.09e-5. Neither cell
is bold, and neither was bold before, so the caption's "bold marks the best value
in a column" was already not honored in that one column. I did not add bold to a
competitor row: it would be a new claim rather than a consequence of this edit.
The author can bold \glotlid's 2.09e-5 or leave the column as published.

#### No prose to update

A sweep of `submission.tex` for .971, .975, .986 and .992 found no sentence
quoting a \cld-subset cell; `:733` refers to the right half of the table without
naming a number.

---

## A. APPLIED 2026-08-19, commit `6374b67`

All wrapped in the `\corrrev{}` macro (blue), kept separable from the
camera-ready `\camrev{}` pass. To accept all of both rounds, redefine as
`\newcommand{\corrrev}[1]{#1}`.

Item 11 (`tab:lenbias-norm`) was held back then and is applied in this round as
A2.8. Item 10's table edit landed then; its two prose sites are applied in this
round as A2.9.

| # | Site | Current | New |
|---|---|---|---|
| 1 | `:754` | "shared unseen-token constant $c = -21$" | **$c = -17$** |
| 2 | `:1287` | "$c = -21$ ... sweep over $\{-17,-19,-21,-23\}$" | **$c = -17$, sweep over $\{-15,-17,-19,-21\}$** |
| 3 | `:627-628` | "every unseen-token value exceeds $c$, so all of them are set to $c$" | **1,655 of the 1,940 exceed $c$ and are set to it; the remaining 285 already lie below and are left unchanged** |
| 4 | `:629-631` | "byproduct of the training-time probability floor of $10^{-12}$ and renormalization" | **the smallest value the per-language fit assigns, scaling as one count in the training-token count $T$; the floor is never reached** |
| 5 | `tab:lid_main`, \unilid row, GlotLID-C | `.929` / `2.03e-5` | **`.933` / `2.02e-5`** |
| 6 | `tab:lid_main`, \unilid-Mistral-Nemo, GlotLID-C | `.912` / `1.84e-5` | **`.912` / `1.86e-5`** (F1 unchanged to three decimals). **Reverted 2026-08-24**: the cell is back to `\camrev{1.84e-5}` under the PD-5r resolution below, so this row is fully carried |
| 7 | `:344`, `:833` | "from .929 to .957" | `.929` becomes **`.933`**; the `.957` is settled in this round as `.956` (A2.6) |
| 8 | `:824` | "2.03e-5 vs 2.71e-5", "roughly 25\%" | **`2.02e-5`**; "roughly 25\%" still holds (25.4\%) |
| 9 | `tab:calibration_provenance` | "unseen-token constant $c=-21$" | **$c=-17$** |
| 10 | `tab:viterbi_vs_marginal` | `.961`/`.929`, `.962`/`.931` | **`.961`/`.933`, `.961`/`.935`**; bolding on accuracy removed; cost 1.7x. Prose completed this round |
| 11 | `tab:lenbias-norm` | overall `.960`/`.960`/`.885` | applied this round, A2.8 |

---

## Marking conventions used by this round

Recorded because a later sweep for `\corrrev{}` spans, or a decision to accept
the round wholesale, depends on knowing them.

**1. Cells whose printed value did not change.** Two rules are in force, and
which one applies depends on whether the table was regenerated wholesale.

- *Whole-table regeneration: wrap every regenerated cell, including the ones that
  print the same value.* This holds for `tab:unilid_llm_comparison` (base F1
  0.960, LLaMA3.2 F1 0.954, DeepSeek3.2 F1 0.955 all print as before),
  `tab:vocab_size_efficiency` (50k F1 0.957, 100k F1 0.960, 200k F1 0.9606),
  `tab:lenbias-norm` (the Original and Raw rescore 0.951 and the Overall 0.960),
  and `tab:length_accuracy` and `tab:tatoeba_udhr_comparison`, where every cell in
  the regenerated column moved anyway. The marker there means "this cell is a
  measurement of the new model", not "this digit changed".
- *Single-cell substitution inside an otherwise carried table: wrap only what
  changed.* This holds for `tab:lid_main`, where the calibrated row's GlotLID-C
  FPR keeps `\camrev{\textbf{1.77e-5}}` and its FLORES FPR stays unwrapped at
  2.91e-4 because both print the same, and for `tab:calibrated_heldout`, where the
  \fasttext row and the \fasttext confidence interval are unwrapped. In
  `tab:calibrated_heldout` the \fasttext row is not a regenerated cell at all: the
  same prediction array (sha256 `4ff74fb55ce5668b...`) is the input in both rounds,
  so the number is bit-identical rather than coincidentally equal.

**2. Two `\camrev{}` spans were replaced by `\corrrev{}` and have lost their
camera-ready marking.** Both are recorded here so the camera-ready pass is
recoverable if the author wants that marking back:

| site | original span | now |
|---|---|---|
| `tab:lid_main`, calibrated row, GlotLID-C F1 | `\camrev{\textbf{.957}}` | `\corrrev{\textbf{.956}}` |
| `submission.tex:989` | `\camrev{within 0.025 macro F1}` | `\corrrev{within 0.03 macro F1}` |

Nothing else in either round removed a `\camrev{}` span. Where a corrected number
sits inside a longer camera-ready sentence, the `\corrrev{}` is nested inside the
surviving `\camrev{}` (`:864-865`, `:1198`, and the `:542` footnote), so both
marks are preserved and the inner colour wins.

**3. `paper/initial_version.tex` is deliberately untouched.** It is a frozen
record of the original submission and still carries every pre-correction value
(.929, c = -21, 0.885, 95.65, and the rest). A sweep for stale numbers will hit it
on every one of them. That is what the file is for; it is not a site to edit. The
same holds for `paper/review_notes_2026-08-09.md`, which records the camera-ready
review pass as it stood on its date and quotes the CommonLID pair 0.723 -> 0.715
twice; it is a dated record, not a paper source.

**4. The $\Delta$ column of `tab:script-breakdown` is the difference of the
printed cells, not of the full-precision values.** This was determined on the
released model rather than assumed: the two conventions disagree at Cyrl, Armn
and Other, and at all three the published $\Delta$ is the difference of the two
printed cells (Cyrl 0.877 - 0.970 = -0.093, where full precision gives -0.092).
A2.14's corrected $\Delta$ column follows the same rule. The full-precision
alternative is -0.0022, -0.0896, -0.0548, -0.1208, -0.1056, -0.2504, -0.2290,
-0.0135, -0.0352, which differs in one cell only (Arab, -0.055 rather than
-0.054), if the author prefers it.

**5. `tab:resource-tier`'s FPR cells are printed at two precisions.** Read off
the released model's measurements: cells at or above 1e-5 print to two
significant figures (5.2873e-05 -> 5.3e-5), cells below 1e-5 print to one with a
trailing zero (6.7179e-06 -> 7.0e-6, not 6.7e-6). Every unchanged cell holds
under that reading and `35k+` moves under it (5.3737e-05 -> 5.4e-5) where one
significant figure would hide the move. The `500--1k` cell is the one the two
rules split: its corrected 9.8602e-06 has crossed below 1e-5, so the sub-1e-5
rule gives **1.0e-5** and the two-figure rule would give 9.9e-6. Applied as
1.0e-5 per the author's instruction. Either way it is a real move from the
published 1.5e-5, and the measured value is 9.8602e-06.

**6. New constant: `FPR_GATE_REL_TOL = 0.05`**, in
`analysis/regen_resource_tier_counts.py`, alongside the existing
`F1_GATE_TOL = 0.005` that `paper_breakdowns.py` had already pre-registered for
the same column. It governs the published-cell comparison and nothing else, and
it binds only for the released model. It is 5% because the published FPR cells
are printed at one or two significant figures, so neither an absolute bound nor a
significant-figure test expresses "agrees at printed precision"; 5% is the
smallest round bound above the 4.0% worst case the released model produces
against the one-significant-figure cells. Flagged here because it is a new
threshold in the codebase.


---

## B. The blocked list, re-stated 2026-08-24

| # | Site | State |
|---|---|---|
| B1 | `tab:lid_main` calibrated row GlotLID-C cells; `:344`, `:833`, `:835` | **DONE.** Jobs 3123324 and 3127704 then `gate_variants apply` (job 3157817, both stages verified). Applied as A2.6 and A2.9 |
| B2 | `tab:lid_main` \unilid and calibrated UDHR / FLORES cells | **DONE, applied 2026-08-24** (A2.10). The eval stage's two non-default-model spots were fixed (fingerprint-derived floor target; the acceptance gate made informational for a non-default model, matching `paper_breakdowns`), both benches then exited 0, and the four cells plus the prose at `:864-865` carry the corrected values |
| B3 | `tab:calibrated_heldout`, `tab:calibrated_views` | **DONE.** `tab:calibrated_heldout` applied 2026-08-24 (A2.7), `tab:calibrated_views` applied the same day with the other two linked tables (A2.14) |
| B4 | `tab:commonlid`; `:867-869`, `:1359-1366` | **DONE, applied 2026-08-24** (A2.12). The corrected chain completed with its binding gates passing (baseline exact equality, floor-21 agreement 1.000000); source `outputs_corrected_round/tables/commonlid_calibrated.md`. Six table cells, two caption counts and both prose sites carry the corrected values |
| B5 | `tab:resource-tier`, `tab:script-breakdown`, `tab:per_language_f1`; the prose now at `:1232` and `:1238` | **DONE.** `tab:resource-tier` and `tab:script-breakdown` applied 2026-08-24 (A2.14), together with the `:1232` prose; `:1238` quotes no number and still holds. `tab:per_language_f1` needs nothing: it is the DSL-ML dialect table and DSL-ML is `em`-trained, so the defect never touched it (C0) |
| B6 | the held-out sentence, now `:1283` | **DONE, closed 2026-08-25** (A2.15). The corrected floor-21-only judge-part macro F1 is 0.9302, which prints as the 0.930 the sentence already carries, so no cell moved and no wrap was added. Both numbers in the sentence are now corrected-model measurements |
| B7 | `:1383-1384` Mistral-Nemo unseen-token parenthetical; its high-entropy group; `tab:calibrated_nemo` | **DONE.** Applied as A2.5 and A2.9. The high-entropy group needed no edit: the corrected variant's flat set is still `{bjn_Latn, sco_Latn, srp_Latn}`, that is Banjar, Scots, and Serbian in Latin script (`outputs_corrected_round/tables/mistralnemo_flat_set.md`) |
| B8 | `tab:lid_main` \unilid-DeepSeek3.2 and \unilid-Qwen3 rows | **CLOSED, rows left carried**, by the author's PD-3 ruling and the failed \cld-subset feasibility gate. The retrains and their GlotLID-C evals are done and match the published cells at paper precision; they are not applied because six of each row's twelve columns cannot be produced |
| B9 | `tab:lenbias-delta`; the token-count sentence, now `:1246` | **DONE, applied 2026-08-25** (A2.15). All six rows rebuilt on the PD-4 golden subset, caption basis and counts restated, the mean rewritten from 0.17 to 0.05 tokens, and the growth claim narrowed to inputs above 150 characters |

---

## Author decisions, 2026-08-24, and what each one left

Every open decision has a ruling. Four are applied; four wait on numbers another
agent is computing; one is closed by verification. Rulings are quoted where the
wording matters.

| # | ruling | state |
|---|---|---|
| PD-1 | *"accept that fasttext numbers"* | **DECIDED, no paper edit needed.** The \fasttext column of `tab:length_accuracy` and the \fasttext row of `tab:tatoeba_udhr_comparison` stay as published, alongside \unilid numbers from this round's retrains. Verified before closing that no caption claims either was re-measured: `tab:length_accuracy`'s caption states only what the columns are, `tab:tatoeba_udhr_comparison`'s states only the training set and the two benchmarks, and the provenance sentence added in A2.4 is in `tab:vocab_size_efficiency`, a \unilid-only table. Carrying them is now a stated choice on the record rather than a silent one |
| PD-2 | *"apply all 3"* | **DECIDED, APPLIED as A2.14.** All three tables carry the corrected cells and the two tables printing the same quantity now agree. Both movements this ledger predicted are confirmed: Beng within-stratum 0.885 -> 0.879, and the `<500` tier 0.871 -> 0.857 within-stratum while its global value rises 0.515 -> 0.596 |
| PD-3 | *"Do the swap if the other columns can be computed. Otherwise leave it"* | **DECIDED, CLOSED: leave the rows carried.** The condition fails. Of the twelve columns in each variant row, the three \cld-subset FPR pairs cannot be produced: the subset definition files are here (`unilid_resources/{glotlidc_cld3subset_83,udhr_cld3subset_80,flores_cld3subset_77}.txt`) but the subset-evaluation instrument is not, which is the standing C3 ask. `outputs/tables/paper_eval_cld3_subset.md` records the reconstruction attempt: it reproduces the published \unilid GlotLID-C subset F1 (.971) but no tested convention reproduces the printed subset FPR (measured 9.71e-5 and 7.77e-5 against 1.63e-4), and it fails the published \fasttext subset F1 as well. Swapping would put six new cells beside six carried ones inside one row. Evidence: `outputs/rerelease/pd_compute_2026-08-24.md` section 3. The variant UDHR and FLORES stages were deliberately not run, since six cells cannot complete a row. **Unchanged by the 2026-09-01 \cld regeneration** (A2.16): that work refits a base vocabulary, which the variant rows' fixed LLM tokenizers cannot do, so the only mechanism open to them is a restricted argmax, and it misses all six published variant cells in one direction |
| PD-4 | *"Do the obvious match"* | **DECIDED, APPLIED as A2.15.** The basis is the 250,000-line test half of the seed-42 draw, the same subset as `tab:lenbias-norm`, and `tab:lenbias-delta` now carries it |
| PD-5 | *"If this repo's corrected variant is available and all columns can be computed, then use it"* | **DECIDED, CLOSED: leave the row carried**, on the same failed condition as PD-3. The corrected Mistral-Nemo variant exists, but its \cld-subset columns are not computable either. One residual follows from this closure and needs a line from the author, see PD-5r |
| PD-6 | *"run the pair breakdown on promoted_residual_pairs.csv to keep the paper consistent"* | **DECIDED, APPLIED as A2.14.** 930,576 wrong predictions, 99.1\%, 88.6\% unchanged at the printed precision, Indonesian and Standard Malay 31,105 lines. This ledger's earlier claim that the corrected round does not report the pair count was wrong and is corrected in A2.14 |
| PD-7 | *"We do not have these [seeds]. Can it be left? We also do not have the fasttext models."* | **DECIDED, closed, no paper edit.** The eight sampled rows of `tab:samples-accuracy`, 5 through 400 samples per language, stay as published. The seeds and repeat count behind their standard deviations are not available, and neither are the \fasttext models. The correction's effect where it can be measured is at most about 0.001 on these quantities, which sits inside those rows' own printed standard deviations (0.02 to 0.90), so carrying them does not put a visibly wrong number in the table. The 500-samples row is the deterministic whole-split case and was corrected to 95.64 in A2.11 |
| PD-8 | *"The script for creating the data for the noise tables got deleted. It may be best just to remove this analysis."* | **DECIDED, APPLIED as A2.13.** `tab:noise_robustness` is removed from the paper: the table input, the whole appendix subsection, the introduction clause and the results sentences. Every removed span is recorded verbatim in A2.13 and `paper/tables/noise_robustness.tex` is left on disk unreferenced, so the analysis is recoverable. This also retires the known-stale p=0\% accuracy cell that PD-8 was tracking |
| PD-9 | *"yes, it is"* | **DECIDED, closed, no paper edit.** The \unilid-LLaMA3.2 base tokenizer is the repository the original variant used. Verified after the ruling that `tab:unilid_llm_comparison`'s caption names exactly \unilid-Mistral and \unilid-LLaMA2 as the unconfirmed pair and does not mention LLaMA3.2, which is now correct rather than merely defensible. `SESSION_STATUS.md`'s open-decision line listing three unconfirmed repositories is superseded by this ruling |
| PD-5r | **New, and the only open paper item.** `tab:lid_main`'s caption says the rows other than \unilid and calibrated "carry over from the original submission, computed on all 45,627,279 lines". Verified after the PD-3 and PD-5 closure: that is exactly true of the \unilid-DeepSeek3.2 and \unilid-Qwen3 rows, which carry no corrected cell. It is **not** true of one cell of the \unilid-Mistral-Nemo row, whose GlotLID-C FPR reads `\corrrev{1.86e-5}` from the 2026-08-19 round. That value is this repo's corrected retrain measured on the 45,377,279-line pool (1.8583e-5), so a single cell of an otherwise carried row is from another generation, which is the mixing PD-5's condition was written to prevent | RESOLVED 2026-08-24: reverted to `\camrev{1.84e-5}` by the coordinator, applying the author's own PD-5 principle (corrected values only with all columns computable). See the "PD-5r resolution" section at the end of this file for the verification against `git show 6374b67~1`. |

---

## C. Ownership, standing asks

### C0. RESOLVED: DSL-ML needs nothing; the defect is `sp`-only

DSL-ML was trained entirely with `--method em` (author confirmation 2026-08-19),
so no DSL-ML result is regenerated. `tab:dialect_stats` and the dialect column of
`tab:per_language_f1` are off the list. The `sp` training path is the only one
that carried the defect: `language_specific_trainer.py:203-204` sits inside
`train_with_sentencepiece_direct`, and the pure-Python EM path left the special
tokens at the training floor.

The latency discrepancy is resolved and is not to be revisited (author,
2026-08-19): latency varies with hardware and with label-set size, so the
`full_prob` run's 1,075 samples/s and `tab:latency_glotlid`'s 3,253 are not
expected to agree.

### C1. The WiLI ask list, closed out 2026-08-24

The three stored WiLI models carry 0.800000 special-token mass in every row, so
they are `sp`-trained and the WiLI tables did need regenerating
(`outputs/rerelease/wili_models_inspect.json`). That regeneration is now
complete. Of the eight model files missing from the GitHub draft releases, all
eight were retrained here in the 2026-08-23 wave and all nine models in that wave
pass the post-training instrument gate (235 languages, real-token mass 1.000000
to six decimals, no defect).

The original ask list is closed. The `tab:samples-accuracy` seeds are not
available and the rows stay as published (PD-7); the LLaMA3.2 tokenizer
repository is confirmed as the original (PD-9), with Mistral and LLaMA2 recorded
as unconfirmed in the caption. One loose end of no consequence to any table
remains: `datasets_reduced.zip` (5 MB) in the datasets release has unidentified
contents.

### C2. Weights to publish

Per the author instruction of 2026-08-19, every re-fitted or corrected weight
file goes to the HuggingFace Hub rather than being reproduced by the co-author:
corrected base GlotLID-C uncalibrated (ready), corrected base calibrated at
c = -17 (needs the version-2 pack, see below), corrected Mistral-Nemo (ready),
retrained DeepSeek3.2 and Qwen3-8B (ready), the nine WiLI models of the
2026-08-23 wave (ready), corrected Apertus 200k / 131k (ready; in no paper
table).

Both release gates now pass for the corrected generation, each against a
reference recorded from that generation: `--mode base` at 250,000/250,000, and
`--mode calibrated` at 250,000/250,000 with `n_disagree` 0, against the version-2
bundle `glotlidc_corrected_calibrated.unilid` packed 2026-08-24 from
`outputs_corrected_round/release/calibration_glotlidc_corrected.json` (c = -17,
group A 1,080, group B the expected four, weight matrix bit-identical to the
version-1 input).

### C3. Standing asks only the co-author can answer

- The subset-evaluation script or command behind the \cld-subset cells. **Reduced
  in scope 2026-09-01**: the \unilid row no longer depends on it, because those
  six cells were regenerated by the published procedure on subset-fitted
  vocabularies (A2.16). It still governs the three variant rows, which stay
  carried under an unknown convention, and it is what would reopen PD-3 and PD-5.
- ~~The UDHR-subset FPR of 1.06e-5~~ **retired 2026-09-01**: that cell now reads
  the regenerated 5.73e-5 and the editorial note asking for confirmation was
  removed with it (A2.16).
- The DSL-ML competitor-score source and split.

## D. Unaffected

`tab:latency_glotlid`, `tab:latency_wili`, `tab:training_time`,
`tab:dialect_stats`, `tab:fasttext_epoch_sweep`, `tab:calibration_provenance`
(other than the c value already applied), and every \glotlid and \cld row. Also
`:754`'s statement that token probabilities are floored at $10^{-12}$ during
training, which remains true of the code.

---

## Caption and framing work still open

- **`tab:lid_main`'s caption split.** It says the \unilid and calibrated rows use
  the 45,377,279-line scored pool while the others carry over from the original
  submission on all 45,627,279 lines. That sentence goes only if PD-3 is decided
  in favour of swapping the variant rows. `paper_eval` also prints a standing
  note: if the camera-ready table restates N per row, every row's N has to move,
  not only the new one.
- **The stratum regressions are reported alongside the overall gain** (author
  decision 2026-08-18), with the mechanism stated. Full-pool, uncalibrated:
  overall +0.0035, tail -0.0087, magnets -0.0071. The two views disagree by
  construction and both belong in the appendix (Exp 24).
- **The proximity bound 21: re-derived 2026-08-25, and the paper's sentence
  stands.** This was the one selected constant never re-derived, and it mattered
  because the bound is a score difference in natural-log units and the correction
  moves score differences. Measured on the corrected model over the 18,001,573-line
  derivation part, the set the original selection used: derivation-part macro F1
  varies by **0.000150** across bounds 15 to 35, against the paper's stated 0.0003.
  The full 101-point grid from 0.5 to 99.5 puts the plateau within 0.0003 of the
  grid optimum at **13.5 to 40.5**, wider than the released model's 14.5 to 35.5,
  so "roughly 15 to 35" is true on both models and 21 sits further from either edge
  than before. The argmax is 20.5 on both models, and 21 is 0.000046 below it on
  the corrected model (0.000002 on the released one). The sweep is gated by
  bit-identity: at bound 21.0 the walk reproduces the saved
  `pred_gate_flat4_prox21.npy` exactly on both models, and with the proximity
  condition disabled it reproduces `pred_gate_flat4_tau5.npy` exactly. Nothing was
  changed: `D3_PROX` is still 21.0 and no prediction array was rewritten. Artifacts:
  `outputs/rerelease/b6_b9_proximity_2026-08-25.md` section 3,
  `outputs/diagnostic/prox_bound_sweep.json`,
  `outputs_corrected_round/diagnostic/prox_bound_sweep.json`.


## PD-5r resolution (2026-08-24, coordinator, applying the author's PD-5 principle)

The Mistral-Nemo lid_main row's GlotLID-C FPR cell is reverted from
\corrrev{1.86e-5} (commit 6374b67, this repo's corrected retrain on the scored
pool) to its pre-6374b67 form \camrev{1.84e-5}, verified against
`git show 6374b67~1:paper/tables/lid_main.tex`. Reason: the author's PD-5
ruling makes corrected values conditional on ALL columns of the row being
computable; the CLD-subset instrument is absent, so the row stays fully
carried, and one corrected-generation cell inside it was exactly the mixing
the condition forbids. The corrected value (1.858e-5, rounds to 1.86e-5)
remains on record in EXPERIMENTS_RESULTS.md and outputs_corrected_round/ if
the row is ever swapped whole.

## Noise table reinstatement (author, 2026-08-25)

The co-author has rerun the robustness experiments with the fixed code, and
the author will re-add `tab:noise_robustness` manually. The A2.13 removal
stands as applied; `paper/tables/noise_robustness.tex` remains on disk
untouched for that purpose, and no session work should regenerate or delete
it. The reinstatement is the author's manual task, not an open item here.


## C3 ask narrowed; UDHR-subset FPR likely a typo (2026-08-26)

The CLD-subset convention sweep (`outputs/rerelease/cld_subset_convention_sweep.md`)
excludes every free-prediction FPR convention by impossible implied
denominators and identifies confinement of predictions to the subset label
set as the only surviving family (identity-validated, not cell-reproduced).
The C3 ask to the co-author narrows from "the subset-evaluation script" to:
restricted argmax over the full model, or models trained on the subset
languages only? PD-3/PD-5 remain closed until answered. Separately, the
standing UDHR-subset FPR item (1.06e-5) now carries measured evidence of a
decimal-exponent typo for 1.06e-4 -- the author should confirm against the
original computation before any cell changes.


## C3 narrowed a second time; the swap stays closed on measurement (2026-08-31)

The author's row-equivalence resolution was verified exactly, but the
restricted-argmax gate misses all 8 measurable published subset cells in both
directions (outputs/rerelease/cld_subset_gate_2026-08-31.md). PD-3/PD-5
remain closed, now on measured grounds. Adopting the ruled convention would
move the published \unilid subset cells themselves (.992/.997 -> .996/.996),
so it is not adopted. The C3 ask to the co-author is now: which models
produced the CLD3-subset columns, and were they the same models (same base
vocabularies) as the corresponding full-set columns? The suspected UDHR-FPR
typo remains unresolved by this gate.


## C3 ANSWERED by the author (2026-08-31)

The plain-UniLID CLD3 columns were produced with a base tokenizer TRAINED ON
THE SUBSET of languages -- a different vocabulary from the model behind the
full-dataset columns. This is the gate record's reading (b), confirmed for
the \unilid row, and it explains that row's side of the both-directions miss.
No subset-vocabulary model exists on this filesystem (measured census), so
reproducing those columns for the corrected generation means fresh
subset-vocabulary trainings. What it implies for the VARIANT rows' subset
cells (fixed LLM tokenizers cannot be subset-fitted, yet their measured
restricted-argmax cells also miss) is under reassessment; PD-3/PD-5 stay
closed pending it.
