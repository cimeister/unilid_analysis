# PD-2, PD-6, PD-3/PD-5: the compute round of 2026-08-24

Scope: the three decisions the author took on 2026-08-24 that have a compute
side. Repo commit `dd9c570ad761ceba672aea5fa9ed57f492144c07`. Corrected model
`/capstor/scratch/cscs/cmeister747/unilid_analysis/corrected/glotlidc_corrected.unilid`,
its predictions under `.../full_test_eval_corrected`, reports under
`outputs_corrected_round/`. No SLURM was needed: every run below finished in
under ten seconds on the login node.

Nothing in `paper/`, `EXPERIMENTS_*.md` or `SESSION_STATUS.md` was edited.

Summary of outcomes:

| item | outcome |
|---|---|
| PD-2 | **Done.** All three tables have a complete corrected cell set below. `regen_resource_tier_counts.py` was parametrized and extended; the FPR convention was identified and gated |
| PD-6 | **Done, and the ledger entry was wrong.** The corrected pair breakdown already existed; the pair count is 31,105 |
| PD-3 / PD-5 | **Feasibility gate FAILS -> leave the rows carried.** The \cld-subset instrument does not exist in this repository. No UDHR/FLORES stage was run for the variants |

---

## 1. PD-2: the three linked tables

### 1.1 What was missing, and the instrument built for it

`paper_breakdowns.py` already produced the corrected F1 grid for all three
tables. Two things were missing.

**The N_test column.** `analysis/regen_resource_tier_counts.py` produced it, but
had no `--model` / `--scratch-dir` / `--out-dir` flags and read
`outputs/diagnostic/...` and wrote `outputs/tables/...` as hardcoded literals, so
it could only ever report the released model.

**The FPR columns.** No script in the repository produced them, and their
provenance was not recorded anywhere. `paper_breakdowns.py` says so explicitly at
`PAPER_RESOURCE_TABLE`: "The paper's FPR columns are not reproduced here (not
part of this task's spec)."

Both are now in `analysis/regen_resource_tier_counts.py`:

- The `analysis.model_context` pattern is applied exactly as
  `analysis/paper_breakdowns.py` uses it (`add_arguments` / `resolve` /
  `resolve_out_root`, plus `--out-dir`), with the same input inventory comment
  classifying every input as model-derived or corpus-derived.
- **The default run is byte-identical.** `python -m analysis.regen_resource_tier_counts`
  with no flags rewrote `outputs/tables/resource_tier_ntest.md` and `diff`
  against the published copy is empty. The new FPR content goes to a **new**
  file, `tables/resource_tier_fpr.md`, precisely so that the 2026-08-09 artifact
  keeps its bytes.
- The script now needs `python -m analysis.regen_resource_tier_counts` rather
  than `python analysis/regen_resource_tier_counts.py`, because it imports from
  the `analysis` package. That is the repo convention for every other script in
  the chain.

### 1.2 The FPR convention, and the instrument gate

The published FPR columns had never been reproduced, so the convention was
determined by measurement on the **released** artifacts before any corrected
number was computed.

- The **global** view (each language's false positives over the whole
  45,377,279-line pool) misses every published cell, by up to an order of
  magnitude: `<500` measures 4.26e-6 against the published 7.2e-5.
- The **within-stratum** macro FPR reproduces all of them. Definition: restrict
  to lines whose true label is in the tier, then average
  `FP_L / (N_tier - support_L)` over the tier's languages. This is the same
  restriction the F1 columns of this table already use.

Instrument gate, released model, `outputs/tables/resource_tier_fpr.md`:
**all 24 F1 and FPR cells reproduce, gate PASSED.** Worst relative FPR gap 4.0%,
at the two cells the paper prints to one significant figure (measured 6.718e-6
against a printed 7.0e-6, and 1.925e-6 against 2.0e-6). Worst F1 gap 0.0004.

The micro form (summed FP over summed negatives) agrees with the macro form to
the printed precision of every published cell, so the published table does not
distinguish the two. Macro is reported, because every other F1/FPR pair in this
repo's tables is macro-averaged; micro is printed alongside in the artifact.

Two new constants were defined, both used only by that published-cell comparison
and nowhere else: `F1_GATE_TOL = 0.005` (the same value `paper_breakdowns.py`
pre-registered for the same column) and `FPR_GATE_REL_TOL = 0.05`. The second is
new and needs the author's eye: the published FPR cells are printed at one or two
significant figures, so neither an absolute bound nor a significant-figure test
expresses "agrees at printed precision"; 5% is the smallest round bound above the
4.0% worst case measured on the released model.

The comparison against the published cells **binds the released model only**.
Under any other `--model` it is computed and reported in full but is
informational and does not set the exit code, following
`_cross_model_message` / `_breakdowns_exit_code` in `paper_breakdowns.py`.

Gates that bind for every model, and that all passed on both runs: 1,940
languages in the per-language CSV; support summing to exactly 45,377,279; tier
language counts 56/40/458/526/398/462; the kept-line count of `y_true.npy`; and
an **alignment gate** requiring that per-language F1 and FP counts recomputed
from the memmaps equal the per-language CSV's own columns, so a run cannot build
a table from one model's predictions and another model's CSV.

**Independent cross-check.** The within-stratum F1 values this script computes
agree with `paper_breakdowns.py`'s independent implementation to all four printed
decimals, on both models and both systems (corrected UniLID 0.8572 / 0.9731 /
0.9895 / 0.9971 / 0.9918 / 0.9576; fastText 0.9150 / 0.9641 / 0.9787 / 0.9864 /
0.9808 / 0.9421).

### 1.3 `tab:resource-tier` (`paper/tables/resource-tier.tex`)

Sources: `outputs_corrected_round/tables/resource_tier_fpr.md` (F1 and FPR),
`outputs_corrected_round/tables/resource_tier_ntest.md` (N_test), both written by
the run below; F1 column independently confirmed by
`outputs_corrected_round/tables/paper_breakdowns_resource.tex`.

```
python -m analysis.regen_resource_tier_counts \
  --model       ${SCR}/corrected/glotlidc_corrected.unilid \
  --scratch-dir ${SCR}/full_test_eval_corrected \
  --out-dir     outputs_corrected_round
```

Complete corrected cell set, published cell first:

| tier | # Langs | N_test | UniLID F1 | UniLID FPR (measured) | fastText F1 | fastText FPR |
|---|---|---|---|---|---|---|
| `<500` | 56 (same) | 2,513 (same) | 0.871 -> **0.857** | 7.2e-5 -> **6.5e-5** (6.5053e-05) | 0.915 (same) | 1.15e-4 (same) |
| `500-1k` | 40 (same) | 5,222 (same) | 0.975 -> **0.973** | 1.5e-5 -> **see note** (9.8602e-06) | 0.964 (same) | 1.9e-5 (same) |
| `1k-12k` | 458 (same) | 552,346 (same) | 0.990 (same) | 8.0e-6 (same) (8.0751e-06) | 0.979 (same) | 8.0e-6 (same) |
| `12k-18k` | 526 (same) | 1,151,363 (same) | 0.997 (same) | 2.0e-6 (same) (1.9765e-06) | 0.986 (same) | 1.0e-5 (same) |
| `18k-35k` | 398 (same) | 1,125,105 (same) | 0.992 (same) | 7.0e-6 (same) (6.7805e-06) | 0.981 (same) | 1.6e-5 (same) |
| `35k+` | 462 (same) | 42,540,730 (same) | 0.958 (same) | 5.3e-5 -> **5.4e-5** (5.3737e-05) | 0.942 (same) | 9.1e-5 (same) |

Measured UniLID F1: 0.8572, 0.9731, 0.9895, 0.9971, 0.9918, 0.9576.

**Note on how the FPR cells are printed, which two cells depend on.** The
published table does not use one precision. Reading it against the released
model's measurements, cells at or above 1e-5 are printed to two significant
figures (5.2873e-05 -> 5.3e-5) and cells below 1e-5 to one with a trailing zero
(6.7179e-06 -> 7.0e-6, not 6.7e-6; 8.1738e-06 -> 8.0e-6, not 8.2e-6). Every
"(same)" above holds under that reading, and `35k+` moves under it
(5.3737e-05 -> 5.4e-5) where it would not move under one significant figure.

The `500-1k` cell is the one the convention does not settle: 9.8602e-06 has
crossed below 1e-5, so the sub-1e-5 rule prints **1.0e-5** and the two-figure
rule prints **9.9e-6**. Both are a real move from the published 1.5e-5. This is a
typesetting call for the author, not a measurement question; the measured value
is 9.8602e-06 either way.

Three facts worth stating plainly:

1. **The N_test column does not move, and this is verified, not assumed.** The
   `support` and `N` columns of the released and corrected per-language CSVs are
   element-wise identical (checked directly), because both are properties of the
   test pool and the training corpus, not of the weights.
2. **The fastText columns do not move.** `pred_fasttext.npy` is byte-identical in
   the two scratch roots (sha256 `4ff74fb5...`), as is `y_true.npy`
   (`9d62ce57...`).
3. **Five UniLID cells change at printed precision**: two F1 (`<500`, `500-1k`)
   and three FPR (`<500`, `500-1k`, `35k+`). Note the direction in the two small
   tiers: they lose F1 and also lose FPR, so the correction moves both halves of
   the same trade. `35k+` moves the other way, gaining FPR while its F1 holds.

The caption's claim at `:1253` -- "matches or exceeds \fasttext for languages
with 500+ training samples" -- survives: 0.973/0.964, 0.990/0.979, 0.997/0.986,
0.992/0.981, 0.958/0.942, and the `<500` exception is still an exception
(0.857 against 0.915).

### 1.4 `tab:script-breakdown` (`paper/tables/script-breakdown.tex`)

The paper's basis, not the fragment's: `Other` is our 84-language group minus
`jpn_Jpan` and `kor_Hang`, so `Other` = 82. The corrected fragment
`paper_breakdowns_script.tex` uses the full 1,940-language basis (Other = 84) and
its `Other` row is therefore **not** the cell to apply.

Instrument gate on the released model, paper basis: the recomputed UniLID column
reproduces every published cell, and the recomputed fastText column reproduces
every published cell, at printed precision -- with one 0.001 residual at Bengali
(measured 0.8858, printed 0.885). That residual predates this round and is
consistent with the paper's stated 1,938-language basis against our 1,940.

Complete corrected cell set:

| Script | # Langs | UniLID | fastText | Delta |
|---|---|---|---|---|
| Latn | 1,700 | 0.940 -> **0.944** | 0.946 (same) | -0.006 -> **-0.002** |
| Cyrl | 70 | 0.877 -> **0.880** | 0.970 (same) | -0.093 -> **-0.090** |
| Arab | 38 | 0.691 -> **0.693** | 0.747 (same) | -0.056 -> **-0.054** |
| Deva | 32 | 0.811 (same) | 0.932 (same) | -0.121 (same) |
| Beng | 6 | 0.885 -> **0.879** | 0.985 (same) | -0.100 -> **-0.106** |
| Grek | 4 | 0.677 -> **0.675** | 0.925 (same) | -0.248 -> **-0.250** |
| Hebr | 4 | 0.740 -> **0.738** | 0.967 (same) | -0.227 -> **-0.229** |
| Armn | 2 | 0.974 -> **0.972** | 0.986 (same) | -0.012 -> **-0.014** |
| Other | 82 | 0.937 (same) | 0.973 (same) | -0.036 (same) |

Measured UniLID values, paper basis: 0.9443, 0.8801, 0.6926, 0.8109, 0.8790,
0.6750, 0.7376, 0.9721, 0.9374. Measured fastText: 0.9465, 0.9697, 0.7474,
0.9317, 0.9846, 0.9253, 0.9666, 0.9856, 0.9726 -- identical in both rounds, as
the byte-identical prediction array requires.

**The Delta column follows the printed cells, not full precision.** This was
determined, not assumed: on the released model the two conventions disagree at
Cyrl, Armn and Other, and in all three the published value is the difference of
the two printed cells (Cyrl 0.877 - 0.970 = -0.093, while the full-precision
difference is -0.0922 -> -0.092). The corrected Delta column above uses the same
printed-cell convention. If the author would rather have full-precision
differences, they are -0.0022, -0.0896, -0.0548, -0.1208, -0.1056, -0.2504,
-0.2290, -0.0135, -0.0352, which changes only the Arab cell (-0.055 rather than
-0.054).

The prose at `:1247` survives with new numbers: "F1 0.940 vs. 0.946" becomes
**0.944 vs. 0.946**; "Greek -0.248, Hebrew -0.227, Devanagari -0.121, Bengali
-0.100" becomes **Greek -0.250, Hebrew -0.229, Devanagari -0.121, Bengali
-0.106**. The caption's qualitative claims hold: Greek, Hebrew and Devanagari are
still the three largest gaps, every non-Latin gap is still negative, and Latin
still "matches", by a narrower margin than before.

### 1.5 `tab:calibrated_views` (`paper/tables/calibrated_views.tex`)

Source: `outputs_corrected_round/tables/paper_breakdowns_resource.tex`, read
cell for cell. The "calib. \unilid" column is `gate_flat4_prox21`.

Global view:

| tier | # Langs | UniLID | calib. UniLID | fastText |
|---|---|---|---|---|
| `<500` | 56 | 0.515 -> **0.596** | 0.780 -> **0.781** | 0.750 (same) |
| `500-1k` | 40 | 0.628 -> **0.676** | 0.892 -> **0.893** | 0.861 (same) |
| `1k-12k` | 458 | 0.891 -> **0.894** | 0.945 -> **0.944** | 0.941 (same) |
| `12k-18k` | 526 | 0.979 (same) | 0.985 (same) | 0.971 (same) |
| `18k-35k` | 398 | 0.963 -> **0.962** | 0.965 (same) | 0.953 (same) |
| `35k+` | 462 | 0.958 -> **0.957** | 0.957 -> **0.956** | 0.941 (same) |

Within-stratum view:

| tier | # Langs | UniLID | calib. UniLID | fastText |
|---|---|---|---|---|
| `<500` | 56 | 0.871 -> **0.857** | 0.827 -> **0.820** | 0.915 (same) |
| `500-1k` | 40 | 0.975 -> **0.973** | 0.955 -> **0.954** | 0.964 (same) |
| `1k-12k` | 458 | 0.990 (same) | 0.987 -> **0.986** | 0.979 (same) |
| `12k-18k` | 526 | 0.997 (same) | 0.997 (same) | 0.986 (same) |
| `18k-35k` | 398 | 0.992 (same) | 0.992 (same) | 0.981 (same) |
| `35k+` | 462 | 0.958 (same) | 0.958 -> **0.957** | 0.942 (same) |

Measured 4-decimal values are in
`outputs_corrected_round/tables/paper_breakdowns.md`.

The two tables that print the same quantity now agree: the within-stratum UniLID
column here and the UniLID F1 column of `tab:resource-tier` are both
0.857 / 0.973 / 0.990 / 0.997 / 0.992 / 0.958, which is what PD-2 asked for.

The caption's mechanism claim survives: the two views still rank the methods
oppositely in the smallest tiers (global `<500`, calibration 0.596 -> 0.781;
within-stratum, 0.857 -> 0.820).

Prose at `:855-859`: "mean per-language F1 rises from 0.515 to 0.780 ... 0.628 to
0.892" becomes **0.596 to 0.781 ... 0.676 to 0.893**, as the ledger predicted.

### 1.6 Both PD-2 predictions confirmed

The ledger predicted exactly two published-cell movements: "Beng within-stratum
0.885 -> 0.879, and the `<500` tier within-stratum 0.871 -> 0.857 while its global
value rises 0.515 -> 0.596". Both are confirmed. The tables above add the cells
the ledger did not enumerate.

---

## 2. PD-6: the pair breakdown

**The ledger entry is stale.** PD-6 says the pair count is one "which the
corrected round does not report". It does. The corrected
`paper_breakdowns.py --part residual` run of 2026-08-24 14:45 wrote
`outputs_corrected_round/diagnostic/promoted_residual_pairs.csv` with all twenty
pairs, and `outputs_corrected_round/tables/promoted_residual.md` prints them.

The run was repeated to capture the exact numerators and denominators, which the
markdown rounds to four decimals. It reproduced both artifacts byte for byte, all
gates passing (the seed-301 split re-derived and bit-matched against
`rule_split_seed301.npz`; 18,001,573 derivation / 27,002,441 judge).

Instrument gate: the same script on the released model reports 926,299 /
0.9915 / 0.8864 and a top pair of 31,113 lines, which are the four numbers the
paper prints. The instrument reproduces the published sentence exactly.

Corrected values for `:1394-1398`, judge part, 27,002,441 lines,
`gate_flat4_prox21`:

| quantity | published | corrected | exact |
|---|---|---|---|
| wrong predictions | 926,299 | **930,576** | 930,576 |
| share with a head true language | 99.2% | **99.1%** | 922,578 / 930,576 = 0.9914053 |
| of those, share confused with another head language | 88.6% | **88.6%** (unchanged) | 816,947 / 922,578 = 0.8855045 |
| Indonesian -- Standard Malay | 31,113 lines | **31,105 lines** | 31,105 |

The head-head share is genuinely unchanged at the printed precision, and the
boundary case was checked rather than assumed: 88.55045% rounds to 88.6%, not to
88.5%.

The other two pairs the sentence names are still there and still in the same
places in the ranking: Standard Arabic -- Najdi Arabic is 2nd in both rounds
(23,608 -> 22,184 lines) and Mandarin -- Wu Chinese is 9th in both
(15,651 -> 15,976). Nothing in the sentence's qualitative content needs to move.

So the sentence can now be updated whole, which is what the author's decision
asked for: four numbers change, one of them (88.6%) to itself.

Full corrected top-20 table: `outputs_corrected_round/tables/promoted_residual.md`.

---

## 3. PD-3 / PD-5: the feasibility gate

Both decisions were conditional on all columns being computable. **They are not.
The condition fails, and both resolve to "leave the rows carried".**

### 3.1 What the twelve columns are

Each variant row of `tab:lid_main` carries six benchmark column pairs:
GlotLID-C, UDHR and FLORES-200 on the full label sets, and the same three
restricted to the `\cld` (CLD3) label subsets of 83, 80 and 77 languages.

- **GlotLID-C (F1, FPR)**: available for all three corrected models.
- **UDHR and FLORES (F1, FPR)**: computable. `analysis/external_bench_eval.py`
  is parametrized, and its two blockers were cleared for the B2 work earlier
  this round, so both stages exit 0 against a non-default model.
- **\cld-subset (six cells per row)**: **not computable in this repository.**

### 3.2 The \cld-subset finding

The subset *definitions* are here, supplied by the co-author and cross-verified:
`unilid_resources/glotlidc_cld3subset_83.txt`, `udhr_cld3subset_80.txt`,
`flores_cld3subset_77.txt`.

The *instrument* is not, and this is stated in the repository's own records
rather than inferred:

- `paper/PAPER_EDITS_pending.md:591`, under "Standing asks only the co-author can
  answer": "The subset-evaluation script or command behind the \cld-subset
  cells, and the UDHR-subset FPR of 1.06e-5".
- `SESSION_STATUS.md:169` and `EXPERIMENTS_RESULTS.md:1301` carry the same ask.
- The four eval scripts the co-author did send (`unilid_resources/eval_*.py`)
  contain no occurrence of "subset" or "cld" -- verified by grep. Neither does
  anything in `analysis/`.
- `outputs/tables/paper_eval_cld3_subset.md` records a reconstruction attempt on
  the released model. It reproduces the published UniLID GlotLID-C subset **F1**
  cell exactly (.971) under an open-set restricted-lines convention, but states:
  "FPR cells are NOT computed here: neither the restricted-pool nor the
  global-pool convention reproduces the paper's printed 1.63e-4 (measured
  9.71e-5 and 7.77e-5); the paper team's eval script is needed first." It also
  fails to reproduce the published fastText subset F1 (.990) under any tested
  convention.
- The paper already concedes the point in `tab:lid_main`'s own caption: "The
  convention behind the original rows' subset FPR values could not be
  determined, so the calibrated row's subset FPR cells are omitted", plus a
  standing editorial note that the UDHR-subset FPR of 1.06e-5 "is awaiting
  confirmation against the original computation".

So of the twelve columns in each variant row, **three (the \cld-subset FPR
pairs) cannot be produced under the convention the published row uses**, and two
of the three subset F1 cells rest on a convention that is reproduced for
GlotLID-C, matched within 0.005 for UDHR, and unverified for FLORES.

### 3.3 Verdict and what was deliberately not run

- **PD-3**: do not swap the `\unilid-DeepSeek3.2` and `\unilid-Qwen3` rows. A
  swap would replace a row of twelve consistent published cells with a row of
  six new cells plus six carried ones, silently mixing generations inside a
  single row -- the failure PD-1 is a decision about, made worse by being
  invisible.
- **PD-5**: the same condition, the same answer. The Mistral-Nemo row stays as
  it is.
- **Not run, deliberately**: the UDHR and FLORES score and eval stages for
  `deepseek_v3.2`, `qwen3_8b` and `mistralnemo_corrected`. Six benchmark cells
  per row that cannot complete a row are not worth the compute, and putting them
  in the record would invite exactly the half-updated row the author's condition
  was written to prevent. The score stages were previously SLURM jobs
  (3130020 / 3130021) and would need resubmitting whenever the \cld ask returns.
- **What unblocks this**: one artifact from the co-author, the subset-evaluation
  script or command. With it, all twelve columns become computable, the
  reconstruction can be gated against the published cells the way every other
  instrument in this round was, and PD-3 and PD-5 can be reopened as real
  choices rather than foreclosed ones.

Note for the author on the GlotLID-C halves, which are unaffected by the above:
the two variants' corrected GlotLID-C cells already match the published ones at
paper precision (0.9089 / 1.976e-5 against .909 / 2.08e-5, and 0.9049 / 2.341e-5
against .904 / 2.55e-5), so leaving the rows carried costs nothing in that
column. The FPR cells differ in the third digit, which the table does not print.

---

## 4. Artifacts written by this round

| path | what |
|---|---|
| `analysis/regen_resource_tier_counts.py` | parametrized for `--model` / `--scratch-dir` / `--out-dir`; FPR/F1 part added |
| `outputs/tables/resource_tier_fpr.md` | new; the released model's within-stratum F1/FPR and the passing instrument gate |
| `outputs/tables/resource_tier_ntest.md` | rewritten by the default run, byte-identical to the published copy |
| `outputs_corrected_round/tables/resource_tier_fpr.md` | new; the corrected model's cells |
| `outputs_corrected_round/tables/resource_tier_ntest.md` | new; identical values to the released one, as the shared support column requires |
| `outputs_corrected_round/tables/promoted_residual.md` | re-run, byte-identical |
| `outputs_corrected_round/diagnostic/promoted_residual_pairs.csv` | re-run, byte-identical |
| this file | the report |

## 5. Open items for the author

1. `FPR_GATE_REL_TOL = 0.05` is a new constant, chosen as the smallest round
   bound above the 4.0% worst case the released model produces against
   one-significant-figure published cells. It governs a published-cell
   comparison only.
2. The `Delta` column of `tab:script-breakdown` is the difference of the printed
   cells, which is what the published table does. The full-precision
   alternative is given in section 1.4 and differs in one cell.
3. PD-6's ledger text should be corrected: the corrected round does report the
   pair count.
4. The `\cld`-subset eval script remains the single artifact blocking PD-3 and
   PD-5.
