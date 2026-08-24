# Adversarial verification of the 2026-08-24 paper edit pass

Internal documentation: a verification record for this repository's own tracking,
not prose written for an external reader.

Scope: every numeric change in `git diff paper/` (nine table `.tex` files plus
`paper/submission.tex`), the ledger `paper/PAPER_EDITS_pending.md`, and a sweep
for stale survivals across `paper/`. Nothing in the edit report was taken on
trust; every cell below was re-read from the measurement record, and where the
record prints only four decimals the value was recomputed from the per-language
diagnostic CSVs.

**Rounding rule applied:** round-half-away-from-zero at the table's printed
precision. **No cell in this diff is an exact tie**, so round-half-even gives the
identical result everywhere; the choice of rule is immaterial to every verdict
below. (The nearest approach to a tie is `tab:calibrated_heldout`'s calibrated
\unilid F1: the true value is 0.94949536, which is *below* 0.9495 and therefore
0.949 under every rule.)

---

## 1. Cell-by-cell verification

### 1.1 `tab:unilid_llm_comparison` (WiLI, 117,500 test lines)

Source: `outputs/rerelease/wili_eval_<model>_fp64.json`.

| cell | old | new | source file : field | measured | verdict |
|---|---|---|---|---|---|
| \unilid (base) F1 | 0.960 | 0.960 | `wili_eval_wili_100k_500_fp64.json : macro_f1` | 0.9600883786631457 | VERIFIED |
| \unilid (base) FPR | 1.859e-4 | 1.863e-4 | same : `macro_fpr` | 1.862884160756501e-4 | VERIFIED |
| Mistral-Nemo F1 | 0.958 | 0.959 | `wili_eval_mistralnemo_wili_fp64.json` | 0.9588977552255618 | VERIFIED |
| Mistral-Nemo FPR | 1.925e-4 | 1.894e-4 | same | 1.8937988725222768e-4 | VERIFIED |
| Mistral F1 | 0.921 | 0.920 | `wili_eval_mistral7b_v02_wili_fp64.json` | 0.9202152408179711 | VERIFIED |
| Mistral FPR | 3.365e-4 | 3.380e-4 | same | 3.379523549736316e-4 | VERIFIED |
| LLaMA3.2 F1 | 0.954 | 0.954 | `wili_eval_llama32_1b_wili_fp64.json` | 0.9543169433772456 | VERIFIED |
| LLaMA3.2 FPR | 2.084e-4 | 2.081e-4 | same | 2.0814693580651025e-4 | VERIFIED |
| LLaMA2 F1 | 0.911 | 0.910 | `wili_eval_llama2_7b_wili_fp64.json` | 0.9096350614747829 | VERIFIED |
| LLaMA2 FPR | 3.698e-4 | 3.733e-4 | same | 3.732678668848882e-4 | VERIFIED |
| DeepSeek3.2 F1 | 0.955 | 0.955 | `wili_eval_deepseek_v3.2_wili_fp64.json` | 0.9551709540312575 | VERIFIED |
| DeepSeek3.2 FPR | 2.042e-4 | 2.048e-4 | same | 2.048372431351155e-4 | VERIFIED |
| Qwen3 F1 | 0.949 | 0.948 | `wili_eval_qwen3_8b_wili_fp64.json` | 0.9481252709386385 | VERIFIED |
| Qwen3 FPR | 2.310e-4 | 2.341e-4 | same | 2.3411529368976176e-4 | VERIFIED |
| \fasttext row | 0.946 / 2.331e-4 | unchanged | -- | -- | VERIFIED unchanged |

Bolding: base row still holds the best F1 (0.960 vs 0.959) and the best FPR
(1.863e-4 vs 1.894e-4). Correct.

**Caption sentence, re-measured independently:**
- 51 dropped entries, Mistral: `wili_mistral7b_v02_base_convert.json`
  `sp_seed_vocab_drop.dropped_count = 51`, `converted_entries_before_drop 32001`,
  `entries_after_drop 31950`. All 51 decoded tokens contain U+000D; none contains
  a tab or a newline. VERIFIED.
- 24 dropped entries, LLaMA2: `wili_llama2_7b_base_convert.json`, 32001 -> 31977.
  All 24 contain U+000D. VERIFIED.
- Refusal site: `UNILID/unilid/vocab_io.py:119` raises on `\t`, `\n` or `\r` in
  `_write_sp_seed_vocab_file`. The caption's "the SentencePiece seed-vocabulary
  writer does not accept them" matches the code. VERIFIED.

### 1.2 `tab:vocab_size_efficiency` (WiLI)

Source: `outputs/rerelease/wili_eval_wili_<size>_defaults_fp64.json`.

| cell | old | new | measured | verdict |
|---|---|---|---|---|
| 10k F1 | 0.945 | 0.944 | 0.9439884395856981 | VERIFIED |
| 10k FPR | 2.514e-4 | 2.556e-4 | 2.555737406801236e-4 | VERIFIED |
| 20k F1 | 0.951 | 0.950 | 0.9498818443557467 | VERIFIED |
| 20k FPR | 2.278e-4 | 2.303e-4 | 2.3033278777959628e-4 | VERIFIED |
| 50k F1 | 0.957 | 0.957 | 0.9567189881087633 | VERIFIED |
| 50k FPR | 2.019e-4 | 2.015e-4 | 2.0149118021458445e-4 | VERIFIED |
| 100k F1 | 0.960 | 0.960 | 0.9600883786631459 | VERIFIED |
| 100k FPR | 1.859e-4 | 1.863e-4 | 1.8628841607565012e-4 | VERIFIED |
| 200k F1 | 0.9606 | 0.9606 | 0.9605653435591045 | VERIFIED |
| 200k FPR | 1.8382e-4 | 1.8418e-4 | 1.8417894162575015e-4 | VERIFIED |
| latency / samples-per-second, all rows | -- | unchanged | -- | VERIFIED unchanged |

**Caption sentence, re-measured independently (not taken from the edit report).**
Loaded the ordered `model.vocab` token lists of
`results_wili_100k_defaults_fp64/tokenizers/langspec_base_tokenizer.json` and of
the base tokenizer inside `wili_assets/wili_100k_500.unilid` (via
`unilid.model_io.load_unilid_raw`) and compared them directly:

```
len fresh 100000   len stored 100000
sets equal True    multiset equal True    duplicate tokens 0
positions differing 132    first divergence 18484    n_languages 235
```

**132 CONFIRMED**, independently of `wili_vocab_repro_check.json` (which records
only the first divergence, 18,484, and full set overlap). Caption accurate.
"Macro F1 and macro FPR come from base tokenizers retrained for this table" is
accurate: all five rows are `wili_*_defaults_fp64`.

### 1.3 `tab:length_accuracy` (WiLI, \unilid column only)

Source: `outputs/rerelease/wili_length_accuracy_wili_100k_500_fp64.json : bins`.

| bucket | old | new | measured `accuracy_pct` | verdict |
|---|---|---|---|---|
| 101--150 | 93.10 | 93.04 | 93.04015296367113 | VERIFIED |
| 151--200 | 94.17 | 94.11 | 94.11301215668618 | VERIFIED |
| 201--300 | 95.86 | 95.83 | 95.82816623740024 | VERIFIED |
| 301--500 | 96.78 | 96.79 | 96.79256797992812 | VERIFIED |
| 501--1000 | 96.53 | 96.60 | 96.60456399514938 | VERIFIED |
| 1000+ | 96.53 | 96.61 | 96.60541092394078 | VERIFIED |
| Overall | 95.65 | 95.64 | 95.64085106382979 | VERIFIED |
| sample counts, all rows | -- | unchanged | 7,845 / 26,652 / 31,449 / 29,494 / 18,142 / 3,918 / 117,500 | VERIFIED |

Note the 501--1000 / 1000+ pair rounds in opposite directions from nearly equal
values (96.60456 -> 96.60; 96.60541 -> 96.61). Both are correct.
Bolding: \unilid above \fasttext in every row, narrowest 0.31 pp at 1000+. Correct.

### 1.4 `tab:tatoeba_udhr_comparison` (\unilid row only)

| cell | old | new | source | measured | verdict |
|---|---|---|---|---|---|
| Tatoeba F1 | 0.414 | 0.420 | `wili_tatoeba_wili_100k_500_fp64.json : macro_f1` | 0.4199703506581937 | VERIFIED |
| Tatoeba FPR | 9.61e-4 | 9.23e-4 | same : `macro_fpr` | 9.230049734638726e-4 | VERIFIED |
| UDHR F1 | 0.868 | 0.866 | `wili_udhr_wili_100k_500_fp64.json : macro_f1` | 0.8659475147486313 | VERIFIED |
| UDHR FPR | 5.88e-4 | 5.86e-4 | same : `macro_fpr` | 5.860379968635932e-4 | VERIFIED |
| language counts 201 / 142 | -- | unchanged | both JSONs `n_languages_scored` | 201 / 142 | VERIFIED |

Bolding correct in all four columns.

### 1.5 `tab:lenbias-norm`

Source: `outputs_corrected_round/tables/lenbias_norm.json` (250,000-line golden
subset = test half of the seed-42 500,000-line draw). Cross-checked against
`EXPERIMENTS_RESULTS.md:472` (job 3129778, 2026-08-21), which supersedes the
earlier full-500k run of job 3117582 recorded at `EXPERIMENTS_RESULTS.md:639`.

| bucket | N old -> new | Original | Raw | Normalized old -> new | measured Normalized | verdict |
|---|---|---|---|---|---|---|
| <30 | 27,328 -> 13,708 | 0.792 -> 0.795 | 0.795 | 0.566 -> 0.494 | 0.494382842135979 | VERIFIED |
| 30--75 | 177,256 -> 88,503 | 0.951 | 0.951 | 0.842 -> 0.776 | 0.7761092844310362 | VERIFIED |
| 75--150 | 195,267 -> 97,861 | 0.978 -> 0.977 | 0.977 | 0.925 -> 0.883 | 0.8830484053913203 | VERIFIED |
| 150--300 | 87,096 -> 43,566 | 0.987 -> 0.988 | 0.988 | 0.966 -> 0.946 | 0.9455309186062526 | VERIFIED |
| 300+ | 13,053 -> 6,362 | 0.995 -> 0.994 | 0.994 | 0.991 -> 0.986 | 0.9864822382898459 | VERIFIED |
| Overall | 500,000 -> 250,000 | 0.960 | 0.960 | 0.885 -> 0.837 | 0.8374 | VERIFIED |

Original overall 0.960376 -> 0.960 VERIFIED; cross-checks against the corrected
model's recorded golden-subset accuracy 0.9604 (`EXPERIMENTS_RESULTS.md:931`).
Raw-rescore agreement 1.000000 (`raw_agreement_with_recorded`), so the caption's
"reproduces the original predictions exactly (100% agreement)" holds. VERIFIED.

Caption "Evaluated on the 250k-line test half of a 500k-line uniform sample":
`analysis/full_test_eval.py:49-53` draws the sample as
`random.sample(range(TOTAL_LINES), 500_000)` under `SAMPLE_SEED = 42`, i.e. a
uniform draw without replacement, and the test half is the odd-parity half of the
sorted draw. Wording accurate. VERIFIED.

### 1.6 `tab:lid_main`

GlotLID-C source: `outputs_corrected_round/diagnostic/paper_eval_per_lang_f1_fullpool.csv`
(45,377,279-line kept pool), macro F1 and macro FPR recomputed here from the
per-language rows. UDHR/FLORES source:
`outputs_corrected_round/tables/external_bench_{udhr,flores}.md`.

| cell | old | new | measured | verdict |
|---|---|---|---|---|
| \unilid GlotLID-C F1 | .933 | .933 (not re-edited) | 0.9327099739222764 | VERIFIED |
| \unilid GlotLID-C FPR | 2.02e-5 | 2.02e-5 (not re-edited) | 2.0187446708e-5 | VERIFIED |
| \unilid UDHR F1 | .859 | .856 | 0.856046 | VERIFIED |
| \unilid UDHR FPR | 1.43e-4 | 1.52e-4 | 15.2027e-5 | VERIFIED |
| \unilid FLORES F1 | .932 | .931 | 0.931265 | VERIFIED |
| \unilid FLORES FPR | 2.78e-4 | 2.83e-4 | 28.2794e-5 | VERIFIED |
| calibrated GlotLID-C F1 | .957 | .956 | 0.9564272221627717 | VERIFIED |
| calibrated GlotLID-C FPR | 1.77e-5 | 1.77e-5 (kept \camrev) | 1.7745086445e-5 | VERIFIED (prints identically) |
| calibrated UDHR F1 | .838 | .842 | 0.8419 | VERIFIED |
| calibrated UDHR FPR | 2.08e-4 | 2.03e-4 | 20.2808e-5 | VERIFIED |
| calibrated FLORES F1 | .933 | .932 | 0.9324 | VERIFIED |
| calibrated FLORES FPR | 2.91e-4 | unchanged, unwrapped | 29.1324e-5 | VERIFIED (prints identically) |

Bolding: .956 is the highest GlotLID-C F1 in the column (fastText .944, \unilid
.933, Nemo .912, DeepSeek .909, Qwen .904). Correct.
Label counts quoted in the header (366 UDHR, 190 FLORES) match the bench reports
(366 labels / 24,115 rows; 190 labels / 192,280 rows). VERIFIED.

### 1.7 `tab:calibrated_heldout` (judge part, 27,002,441 lines)

Source: `outputs_corrected_round/tables/paper_eval_appendix.tex` and
`paper_eval.md`; F1 and FPR independently recomputed from
`paper_eval_per_lang_f1_judge.csv` (FPR convention fp/(N - support), the
convention recorded in `analysis/metrics.py`).

| cell | old | new | measured | verdict |
|---|---|---|---|---|
| \unilid F1 | 0.912 | 0.916 | 0.9159374583497665 | VERIFIED |
| \unilid FPR (x1e5) | 2.04 | 2.03 | 2.030085062006831 | VERIFIED |
| calibrated F1 | 0.950 | 0.949 | 0.9494953601662065 | VERIFIED |
| calibrated FPR (x1e5) | 1.77 | 1.78 | 1.7825327738267995 | VERIFIED |
| \fasttext F1 / FPR | 0.933 / 2.72 | unchanged | 0.9331534309859375 / 2.7164746258 | VERIFIED unchanged |
| bootstrap vs \unilid mean | +0.038 | +0.034 | +0.0336 (`paper_eval.md`) | VERIFIED |
| bootstrap vs \unilid CI | [+0.033,+0.043] | [+0.029,+0.038] | [+0.0290,+0.0383] | VERIFIED |
| bootstrap vs \fasttext mean | +0.017 | +0.016 | +0.0163 | VERIFIED |
| bootstrap vs \fasttext CI | [+0.011,+0.022] | unchanged | [+0.0108,+0.0221] | VERIFIED |

Cross-check: 0.9494954 - 0.9159375 = 0.033558 -> +0.034 and
0.9494954 - 0.9331534 = 0.016342 -> +0.016. The bootstrap point estimates agree
with the full-precision differences of the same table's own cells. Consistent.
Caption claim "held-out levels sit below the full-pool levels for every system":
0.916<.933, 0.949<.956, 0.933<.944. Holds. VERIFIED.

### 1.8 `tab:calibrated_nemo`

Source: `outputs_corrected_round/tables/mistralnemo_eval.{md,tex}`; F1 and FPR
independently recomputed from `mistralnemo_per_lang_f1_{fullpool,judge}.csv`.

| cell | old | new | measured | verdict |
|---|---|---|---|---|
| retrained baseline, full-pool F1 | 0.913 | 0.912 | 0.9118863405012766 | VERIFIED |
| retrained baseline, FPR (x1e5) | 1.79 | 1.86 | 1.858261036226707 | VERIFIED |
| retrained baseline, held-out F1 | 0.897 | 0.895 | 0.8950966665187564 | VERIFIED |
| + unseen-token, full-pool F1 | 0.940 | 0.935 | 0.9350312233556158 | VERIFIED |
| + unseen-token, FPR | 1.71 | 1.79 | 1.789740263053053 | VERIFIED |
| + unseen-token, held-out F1 | 0.928 | 0.923 | 0.9231935518761833 | VERIFIED |
| + re-examination, full-pool F1 | 0.954 | 0.950 | 0.9504030563476834 | VERIFIED |
| + re-examination, FPR | 1.56 | 1.62 | 1.6246805989716402 | VERIFIED |
| + re-examination, held-out F1 | 0.947 | 0.944 | 0.9439787198316901 | VERIFIED |

Caption claim "within 0.002 macro F1 of the published variant row": lid_main's
Mistral-Nemo GlotLID-C cell is .912, measured baseline 0.9119, diff -0.0001.
Quantitatively holds. (See MINOR-3 on "which is left unchanged".)

### 1.9 `submission.tex` prose

| line | old | new | source | verdict |
|---|---|---|---|---|
| 345 | ".933 to .957" | ".933 to .956" | matches `tab:lid_main` / 0.9564272 | VERIFIED |
| 542 (footnote) | "0.001 higher accuracy ... roughly twice as much time" | "the same accuracy to three decimals ... roughly 1.7 times as much time" | `EXPERIMENTS_RESULTS.md:625-637`, job 3110925: 0.961/0.961 accuracy; 4h36m both decoders vs 1h42m Viterbi alone => marginalization 2h54m/1h42m = 1.71x | VERIFIED |
| 847 | "0.933 to 0.957" | "0.933 to 0.956" | as line 345 | VERIFIED |
| 849 | "to 1.77e-5" | unchanged | corrected 1.7745e-5 prints as 1.77e-5 | VERIFIED |
| 853 | "+0.038 [+0.033,+0.043]" | "+0.034 [+0.029,+0.038]" | `paper_eval.md` bootstrap | VERIFIED |
| 864 | "(0.859 to 0.838)" | "(0.856 to 0.842)" | UDHR baseline 0.856046 -> gated 0.8419 | VERIFIED |
| 865 | "(0.932 to 0.933)" | "(0.931 to 0.932)" | FLORES baseline 0.931265 -> gated 0.9324 | VERIFIED |
| 974 | "(0.414 vs. 0.160)" | "(0.420 vs. 0.160)" | 0.4199704; 0.420/0.160 = 2.625, so "more than doubling" holds | VERIFIED |
| 989 | "within 0.025 macro F1" | "within 0.03 macro F1" | lid_main .933 vs {.912,.909,.904}; max gap .933-.904 = 0.029 <= 0.03. The old 0.025 was false | VERIFIED |
| 1198 | "(0.931 against 0.929) ... 0.001 higher accuracy at approximately 2x" | "(0.935 against 0.933) ... the same accuracy to three decimals, at approximately 1.7x" | `tab:viterbi_vs_marginal` + job 3110925 record | VERIFIED |
| 1397 | "two languages ... below c = -21" | "509 languages ... at or below c = -17" | **Context is the retrained Mistral-Nemo variant**, not the base model. `corrected_chain_2026-08-24.md` section 2 clamp line: "mistralnemo floor -17.0: 1,431 of 1,940 rows clamped; 509 already at or below the target" | VERIFIED (509 is the Nemo count and the sentence is about Nemo; the base model's 1,655 / 285 appear correctly at :627-631) |
| 1401 | "from 0.913 to 0.954" | "from 0.912 to 0.950" | `mistralnemo_eval.md` full pool | VERIFIED |
| 1403 | "+0.050 [+0.044,+0.057]" | "+0.0489 [+0.0424,+0.0555]" | `mistralnemo_eval.md` paired bootstrap, judge part | VERIFIED |
| 1404 | "(+0.041 against +0.028)" | "(+0.038 against +0.023)" | true full-pool gains are +0.03852 and +0.02372 | **MISMATCH -- see CRITICAL-1** |
| 838 | "roughly 25% ... (2.02e-5 vs 2.71e-5)" | unchanged | 1 - 2.0187447/2.7062870 = 25.406% | VERIFIED unchanged |

---

## 2. Findings

### CRITICAL-1 -- `submission.tex:1404`, both full-pool gains are off by one in the last digit

The sentence reads "(+0.038 against +0.023 in full-pool macro F1)". Re-derived
from the per-language diagnostic CSVs at full precision:

- Mistral-Nemo: 0.9504030563476834 - 0.9118863405012766 = **0.03851672** -> **+0.039**
- dedicated tokenizer: 0.9564272221627717 - 0.9327099739222764 = **0.02371725** -> **+0.024**

Neither is a rounding tie. The paper prints +0.038 and +0.023, which are the
differences of the *rounded three-decimal table cells* (0.950-0.912; 0.956-0.933).
The ledger discloses this convention at `PAPER_EDITS_pending.md:218-219` and
argues the published sentence computed its +0.041 / +0.028 the same way -- but
those two released values are consistent with *both* conventions (released
0.9538-0.9132 = 0.0406 -> 0.041; 0.9569-0.9292 = 0.0277 -> 0.028), so they do not
establish the printed-cell convention, and the rest of this same edit pass uses
full-precision rounding everywhere (including `tab:calibrated_heldout`'s own
bootstrap point estimates, which do match the full-precision differences).

The relative claim ("the gain is larger than for the dedicated-tokenizer model")
survives either way. Author call required: print +0.039 / +0.024, or keep the
cell-difference values and say in the sentence that they are differences of the
tabulated cells.

### MAJOR-1 -- `submission.tex:1298` now contradicts the table it points at

":1298" reads "Under the unseen-token constant alone, with no re-examination,
held-out macro F1 rises from 0.912 to 0.930." The 0.912 was the *old*
`tab:calibrated_heldout` \unilid cell. That cell is now **0.916** in the same
paper. Two different values for the same quantity (uncalibrated \unilid macro F1
on the 27,002,441-line held-out part) now appear in the camera-ready.

The ledger records the block correctly (B6: the 0.930 has no corrected
counterpart because `paper_eval` reports only `baseline` and
`gate_flat4_prox21`, and "the first number would become 0.916"), but it treats
this as "left whole rather than half-updated" without noting that the applied
table edit turns it into a live internal contradiction. Either the sentence gets
a `\corrrev` on 0.912 plus the missing floor-21-only judge-part run, or it must
be cut for the camera-ready.

### MAJOR-2 -- `tab:length_accuracy` Overall 95.65 -> 95.64 breaks three other sites

The same quantity -- \unilid accuracy on the full 117,500-line WiLI test set --
is printed in four places. Only one moved:

| site | prints | consistent with 95.64? |
|---|---|---|
| `tab:length_accuracy` Overall (edited) | 95.64 | -- |
| `tab:samples-accuracy`, 500 samples/lang row | **95.65 $\pm$ 0.00** | no |
| `fig:samples-accuracy` plot data, `submission.tex:953` | **(500,95.65)** | no |
| `tab:noise_robustness`, p=0% \unilid accuracy | **0.957** | no (0.9564085 -> 0.956) |

Before the edit all four agreed (95.65021 -> 95.65 and -> 0.957). The ledger's
PD-7 and PD-8 record that `tab:samples-accuracy` and `tab:noise_robustness` are
not regenerated, but neither entry notes that applying A2.2 makes them
numerically inconsistent with an applied table. `fig:samples-accuracy`'s data
point is not mentioned anywhere in the ledger.

(The equivalent \fasttext discrepancy -- 94.54 vs 94.55 vs 0.954 -- predates this
pass and is already recorded in `review_notes_2026-08-09.md:50-51`.)

### MINOR-1 -- two different `\corrrev` wrapping conventions in one pass

The WiLI tables wrap cells whose *printed value did not change* but whose
underlying measurement did: `unilid_llm_comparison` base F1 `\corrrev{0.960}`,
LLaMA3.2 `\corrrev{0.954}`, DeepSeek `\corrrev{0.955}`; `vocab_size_efficiency`
50k `\corrrev{0.957}`, 100k `\corrrev{0.960}`, 200k `\corrrev{0.9606}`.
`tab:lid_main` does the opposite in the identical situation and deliberately
leaves the calibrated FLORES FPR (2.91e-4) unwrapped and the calibrated GlotLID-C
FPR as `\camrev{1.77e-5}`. Both choices are individually disclosed in the ledger
(A2.10, A2.6) but they contradict each other; a reader tracking blue text gets
two different meanings for it.

### MINOR-2 -- `\camrev` replaced by `\corrrev` at two sites

`tab:lid_main` calibrated GlotLID-C F1 (`\camrev{\textbf{.957}}` ->
`\corrrev{\textbf{.956}}`) and `submission.tex:989`
(`\camrev{within 0.025 macro F1}` -> `\corrrev{within 0.03 macro F1}`). Both are
noted in the ledger (A2.6, A2.9). Consequence worth stating: the ledger's own
acceptance recipe ("redefine `\corrrev` as identity to accept") no longer has an
inverse at these two spans -- the camera-ready marking is gone, so the author can
no longer accept the `\camrev` round while rejecting the `\corrrev` round there.
Both macros are colour-only (`submission.tex:140,144`), so nothing renders wrong.

### MINOR-3 -- `tab:calibrated_nemo` caption "which is left unchanged"

The caption still says the published variant row in `tab:lid_main` "is left
unchanged". That row's FPR cell carries `\corrrev{1.86e-5}` from the 2026-08-19
round, so the claim is not literally true. Disclosed as PD-5; flagged here
because the caption is camera-ready prose, not a ledger entry. The quantitative
half of the sentence (within 0.002 macro F1) is verified above.

### MINOR-4 -- `paper/initial_version.tex` retains every superseded value

`.859`, `1.43e-4`, `.932`, `2.78e-4`, `0.414`, `9.61e-4`, `0.868`, `5.88e-4`,
`0.885`, `0.792`, `0.566`, `95.65`, `0.945`, `2.514e-4`, `1.925e-4`, `3.365e-4`,
`3.698e-4`, `2.310e-4`, `1.859e-4` all survive there. **This is a legitimate
survival, not a miss**: the file is the frozen original submission -- it carries
its own `\documentclass`, inlines all 27 tables rather than `\input`-ing
`tables/`, and contains zero `\camrev` / `\corrrev` markers. Nothing in the
ledger says so, though, and a future pass could reasonably read it as a missed
file. One line in the ledger's section D would close it.

### MINOR-5 -- cosmetic nesting inconsistency

`\textbf{\corrrev{...}}` in `length_accuracy`, `tatoeba_udhr_comparison`,
`unilid_llm_comparison`, `vocab_size_efficiency`; `\corrrev{\textbf{...}}` in
`lid_main`. Renders identically. No action needed.

### Observations (no action implied)

- **Cross-pool comparison behind ":989".** The "within 0.03 macro F1" claim
  compares `tab:lid_main`'s \unilid GlotLID-C cell (45,377,279-line pool) against
  the three variant cells (45,627,279 lines, carried from the original
  submission). Disclosed in the lid_main caption and open as PD-3.
- **`tab:latency_wili` vs `tab:vocab_size_efficiency`.** 0.155 ms / 6435 s^-1
  against the 100k row's 0.175 ms / 5717.68 s^-1. Pre-existing; the newly added
  caption sentence ("the original measurements ... not re-measured") is accurate
  and does not make it worse.
- **Reproduction tolerances.** Per `wave_2026-08-24_compilation.md`, Mistral-Nemo,
  Mistral, LLaMA2, Qwen3 and the 10k/20k vocab rows all fall outside
  `wili_eval.py`'s own +/-5e-4 F1 tolerance against the published cells. Applying
  the retrained values is the right call; the ledger's decision not to state in
  the caption that all seven rows are new models is recorded and is an author
  matter.

---

## 3. Mechanical conventions

| check | result |
|---|---|
| every changed span wrapped in `\corrrev{}` | PASS -- every numeric change in the diff is inside a `\corrrev`; no bare edited number found |
| `\camrev` content altered without a ledger note | PASS -- 4 spans altered in place (`:542`, `:861-865`, `:1195-1198`, plus nested edits), 2 spans replaced (`tab:lid_main` calibrated F1, `:989`); all six appear in the ledger. See MINOR-2 for the consequence |
| new caption sentences accurate | PASS -- 51/24 CR-dropped entries re-derived from the convert records; 132 reordered positions re-measured from the stored container and the defaults-trained base tokenizer |
| AI / assistant mentions in the diff | PASS -- none in any `.tex`. The only hit anywhere in the diff is the word "agent" in the ledger's own status vocabulary (`PAPER_EDITS_pending.md:21`), an internal file |
| brace balance vs HEAD, per changed file | PASS -- identical in all nine files: 0 for every table, and `submission.tex` keeps its pre-existing delta of 1. Matches the ledger's claim |

---

## 4. Ledger spot-checks

| claim | verdict |
|---|---|
| A2.1 applied (seven \unilid rows, fastText untouched) | CONFIRMED against the diff and the seven JSONs |
| A2.4 caption applied; "132 is a measurement made for this edit" | CONFIRMED; re-measured here to 132 independently |
| A2.7 applied cell-for-cell from `paper_eval_appendix.tex` | CONFIRMED; every cell matches that fragment and the recomputed CSV values |
| A2.9 "deliberately left unchanged" at `:838` and `:855-859` | CONFIRMED -- neither line appears in the diff |
| PD-2 items genuinely unapplied (`calibrated_views`, `resource-tier`, `script-breakdown`) | CONFIRMED -- none of the three files is modified in the working tree |
| PD-4 / B9 (`lenbias-delta`) genuinely unapplied | CONFIRMED -- file unmodified |
| A2.6 "the \unilid GlotLID-C cells ... re-verified, not re-edited" | CONFIRMED -- absent from the diff, and equal to 0.9327100 / 2.0187447e-5 |

---

## 5. Verdict

**FIXES NEEDED.** Every table cell in the diff is correct against its
authoritative record at the printed precision. Three items need action:

1. **CRITICAL-1** -- `submission.tex:1404`: +0.038 -> +0.039 and +0.023 -> +0.024,
   or reword to say the two figures are differences of the tabulated cells.
2. **MAJOR-1** -- `submission.tex:1298`: 0.912 contradicts the now-applied
   `tab:calibrated_heldout` cell 0.916.
3. **MAJOR-2** -- `tab:length_accuracy` Overall 95.64 contradicts
   `tab:samples-accuracy` (95.65), `fig:samples-accuracy`'s plot data at
   `submission.tex:953` (95.65) and `tab:noise_robustness` p=0% (0.957).

Minors 1-5 are convention or framing items for the author.

Verifier: adversarial re-derivation pass, 2026-08-24. Repo commit at run time:
dd9c570 plus the working-tree paper edits.
