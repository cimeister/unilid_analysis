# Corrected GlotLID-C release chain, 2026-08-24 login-node session

Model under test: `/capstor/scratch/cscs/cmeister747/unilid_analysis/corrected/glotlidc_corrected.unilid`
(sha256 31c3d956db7b00c9..., base_W sha256 a4aeff1994640322..., version-1 container).
Scratch root (SCR): `/capstor/scratch/cscs/cmeister747/unilid_analysis`.
Reports root: `outputs_corrected_round/`. Repo commit at run time: dd9c570.

Every command below was run on the login node with the argv recorded verbatim in
each section. Nothing was substituted for a missing artifact.

---

## 1. Job 3157817 (gate_variants apply flat4_tau5 then flat4_prox21) -- VERIFIED

Log: `${SCR}/logs/gateapply_corrected_3157817.out` (10,199 bytes); `.err` empty.
Wall clock: `Sun Aug 23 08:36:45 PM CEST` to `08:37:12 PM` = **27 s**, both stages.

**The 27 s is legitimate, and the "18-minute historical apply" premise does not
match any recorded apply-stage log.** The released model's own two apply stages
were also fast:

| job | stage | start | finish | wall |
|---|---|---|---|---|
| 3015805 | apply flat4_tau5 (released) | Aug 6 00:34:34 | 00:35:01 | 27 s |
| 3016337 | apply flat4_prox21 (released) | Aug 6 01:34:49 | 01:34:59 | 10 s |
| 3157817 | apply flat4_tau5 + flat4_prox21 (corrected) | Aug 23 20:36:45 | 20:37:12 | 27 s |

The historical 37 s total for the same two stages brackets the corrected 27 s.
The multi-minute jobs in this family are the *topk* stage (3127704, 6 m 40 s) and
the *sologates* floor-21 stage (3123324, 07:37 to 07:50, 13 min) -- both already
complete, and both stages this apply run legitimately skips because it reads
their persisted arrays (`gate_topk_ids/lines/scores.npy`, Aug 21 22:42).

### Decisive log lines

Stage 1 (flat4_tau5) did real scoring work, not a short-circuit:

```
variant 'flat4_tau5' re-examined-set languages (4): ['arg_Latn', 'bjn_Latn', 'sco_Latn', 'vls_Latn']
  floor -17.0: 1,655 of 1,940 rows clamped; 285 already at or below the target
Pushing weights to Rust cache...
Model ready
Caching floor-21 weights for flat4_tau5 calibration...
flat4_tau5 calibration: arg_Latn tau=2.9338 nats (1,903 finite winning margins of 1,903 self-won lines, 2,000 scoreable of 2,000 sampled)
flat4_tau5 calibration: bjn_Latn tau=4.4250 nats (1,863 ...)
flat4_tau5 calibration: sco_Latn tau=5.5799 nats (1,969 ...)
flat4_tau5 calibration: vls_Latn tau=3.6451 nats (1,950 ...)
Wrote outputs_corrected_round/diagnostic/tau_flat4.csv
```

Stage 2 (flat4_prox21) ran its own self-check across the full array:

```
flat4_prox21 self-check passed: the D=None two-step walk is bit-identical to
  .../full_test_eval_corrected/pred_gate_flat4_tau5.npy on all 45,627,279 lines.
```

### Outputs present

| artifact | size | mtime |
|---|---|---|
| `outputs_corrected_round/diagnostic/tau_flat4.csv` | 225 B, 4 data rows | Aug 23 20:37 |
| `${SCR}/full_test_eval_corrected/pred_gate_flat4_tau5.npy` | 91,254,686 B | Aug 23 20:37 |
| `${SCR}/full_test_eval_corrected/pred_gate_flat4_prox21.npy` | 91,254,686 B | Aug 23 20:37 |
| `${SCR}/full_test_eval_corrected/pred_gate_flat4_prox21_meta.json` | 2,268 B | Aug 23 20:37 |
| `outputs_corrected_round/tables/gate_flat4_prox21_build.md` | 3,969 B | Aug 23 20:37 |
| `outputs_corrected_round/tables/gate_flat4_tau5_build.md` | 3,507 B | Aug 23 20:37 |

### Independent numeric re-verification (not from the log)

Loaded the arrays directly and recounted:

| check | measured | log claim |
|---|---|---|
| `pred_gate_flat4_prox21 != pred_floor21` | 306,371 | 306,371 (n_moved) |
| `pred_gate_flat4_tau5 != pred_floor21_gate` | 80,078 | 80,078 (n_moved) |
| `pred_gate_flat4_prox21 != pred_gate_flat4_tau5` | 12,325 | 12,325 (blocked_by_proximity) |
| `pred_gate_flat4_prox21 == pred_floor21_gate` | False | (a no-op would be True) |

Accuracy against `y_true.npy`, all 45,627,279 lines: floor21 0.95642, floor21_gate
0.95916, flat4_tau5 0.96028, flat4_prox21 0.96040. The chain moves monotonically
in the expected direction. The build record's own accounting is internally
consistent: step 1 moved 226,321 (135,002 to true) and step 2 moved 80,050
(52,265 to true), combined 306,371 moved / 187,267 to true.

**Verdict: both apply stages ran in full and wrote every expected output.**

---

## 2. Job 3158825 (corrected Mistral-Nemo stages) -- VERIFIED

Log: `${SCR}/logs/nemo_stages_corrected_3158825.out` (183,854 bytes, 1,371 lines);
`.err` empty. `Mon Aug 24 05:35:41` to `07:34:11` = 1 h 58 m 30 s.

All five stages ran, in order, with their own banners:

| stage | banner timestamp |
|---|---|
| calibval | 05:35:42 |
| flatrule | 05:36:41 |
| tau | 05:36:46 |
| topk | 05:43:15 |
| eval | 07:34:02 |

Preflight line: `OK: floor target -17.0, matching .../full_test_eval_corrected/fingerprint_floor21.json.`

### Clamp line -- matches expectation exactly

```
  mistralnemo floor -17.0: 1,431 of 1,940 rows clamped; 509 already at or below the target
floor -17.0: 1431 of 1940 rows clamped; 509 row(s) already at or below the target and left unchanged per the downward-clamp mechanism: aak_Latn (-17.7397), abk_Cyrl (-17.6931), ...
```

(Expected 1,431 of 1,940. Confirmed.)

### Eval-stage gates, all passed

- Language order matches the canonical 1,940-language list.
- `y_true.npy` shape (45,627,279,), no UNSEEN, 250,000 EXCLUDED (== EXPECTED_VAL_LINES).
- Full kept pool 45,377,279 (== EXPECTED_KEPT).
- Seed-301 judge split 18,001,573 / 27,002,441, bit-matches the stored record.
- Sentinel guard: no value < -1 on the kept pool for any of the three configs.
- Banked-array identity, weight-matrix sha cross-check, topk fingerprint, and
  gate-group membership all matched.

### Headline numbers, and the delta against the released round

Full kept pool, 45,377,279 lines:

| config | corrected F1 | released F1 | dF1 | corrected FPR x1e5 | released FPR x1e5 | dFPR |
|---|---|---|---|---|---|---|
| nemo_baseline | 0.9119 | 0.9132 | -0.0013 | 1.8583 | 1.7927 | +0.0656 |
| nemo_floor21 | 0.9350 | 0.9396 | -0.0046 | 1.7897 | 1.7139 | +0.0758 |
| nemo_gated | 0.9504 | 0.9538 | -0.0034 | 1.6247 | 1.5588 | +0.0659 |

Judge part, 27,002,441 lines:

| config | corrected F1 | released F1 | dF1 | corrected FPR x1e5 | released FPR x1e5 | dFPR |
|---|---|---|---|---|---|---|
| nemo_baseline | 0.8951 | 0.8968 | -0.0017 | 1.8653 | 1.7993 | +0.0660 |
| nemo_floor21 | 0.9232 | 0.9278 | -0.0046 | 1.7960 | 1.7199 | +0.0761 |
| nemo_gated | 0.9440 | 0.9473 | -0.0033 | 1.6283 | 1.5627 | +0.0656 |

Paired bootstrap (gated minus baseline, judge part, B=10,000, seed 0):
corrected **+0.0489 [+0.0424, +0.0555]** vs released +0.0504 [+0.0438, +0.0573].
The intervals overlap heavily; the correction does not move the headline claim.

### Against the paper's `paper/tables/calibrated_nemo.tex`

Published cells (3 dp), and the corrected round's regenerated `.tex` cells:

| row | paper full-pool F1 | corrected | paper FPR x1e5 | corrected | paper held-out F1 | corrected |
|---|---|---|---|---|---|---|
| retrained baseline | 0.913 | 0.912 | 1.79 | 1.86 | 0.897 | 0.895 |
| + unseen-token constant | 0.940 | 0.935 | 1.71 | 1.79 | 0.928 | 0.923 |
| + re-examination (calibrated) | 0.954 | 0.950 | 1.56 | 1.62 | 0.947 | 0.944 |

Every corrected F1 cell is 0.001 to 0.005 below the published cell; every FPR cell
is 0.06 to 0.08 (x1e5) above it. The direction is uniform across all three rows
and both pools: under the special-token correction this Mistral-Nemo retrain is
slightly worse on both metrics, not better. The `caption`'s claim that the
retrained baseline is "within 0.002 macro F1 of the published variant row" still
holds under the correction: the module's own recorded comparison reads
`Measured nemo_baseline full-pool: F1 0.9119 (diff -0.0001), FPR 1.8583e-05
(diff +1.8261e-07)` against `lid_main.tex`'s 0.912 / 1.86e-5 -- and note that
`lid_main.tex`'s Mistral-Nemo FPR cell already carries a `\corrrev{1.86e-5}`
marker, i.e. that cell has already been updated to the corrected value.

### Other corrected-round differences worth recording

- Re-examination accounting: group A examined 201,516 / moved 170,333 (released
  184,251 / 155,731); group B examined 58,165 / moved 57,514 (released 58,257 /
  57,759). Lines with `nemo_gated != nemo_floor21`: 227,847 (released 213,490).
- Group A tau exclusions: **33** of 1,080 (released 32). Group B: 0 of 3, both.
- Degeneracy caveat: **34** of 1,940 rows flagged (released 32). This is the
  2026-08-23 definition change (estimated-token count now over real columns only),
  not a new defect.

### Outputs present

`outputs_corrected_round/tables/mistralnemo_eval.md` (6,328 B) and `.tex` (1,146 B),
`outputs_corrected_round/diagnostic/mistralnemo_per_lang_f1_fullpool.csv` (177,547 B),
`outputs_corrected_round/diagnostic/mistralnemo_per_lang_f1_judge.csv` (174,202 B),
`${SCR}/full_test_eval_mistralnemo_corrected/pred_nemo_gated.npy`. All Aug 24 07:34.

---

## 3. `build_release_calibration` for the corrected model -- PASSED

```
python3 -m analysis.build_release_calibration \
  --model ${SCR}/corrected/glotlidc_corrected.unilid \
  --fingerprint ${SCR}/full_test_eval_corrected/fingerprint_floor21.json \
  --unseen-token-constant -17.0 \
  --tau-group-a outputs_corrected_round/diagnostic/tau_floor21_gate.csv \
  --tau-group-b outputs_corrected_round/diagnostic/tau_flat4.csv \
  --out outputs_corrected_round/release/calibration_glotlidc_corrected.json
```

Result:

```
Loaded 1940 languages, vocab size 100000
Wrote outputs_corrected_round/release/calibration_glotlidc_corrected.json (160,650 bytes):
  1080 group A rows (26 excluded), 4 group B rows, 1940 train counts
```

Both pre-verified expectations held without any `--expect-*` override: group A
1,080 rows / 26 excluded, group B exactly `{arg_Latn, bjn_Latn, sco_Latn, vls_Latn}`.
The script's own round-trip re-verification and its `cal.runtime_for(langs)`
validation against the corrected model's language list both passed.

Bundled constants: `unseen_token_constant -17.0`, head_n 18000, replacement_min_n
100000, proximity_bound 21.0, topk 5, margin_q 5.0, group_b_percentile 5.0,
calib_max 2000, min_calib_lines 200, calib_seed 0. Provenance records the
corrected `model_path` and `fingerprint_json` (the script adds those two fields
only for a non-default model, so the released JSON's byte-for-byte reproduction
check is unaffected), base_W sha a4aeff1994640322..., clamped sha 0f1812e0c73d4e2b...

---

## 4. Corrected `paper_eval` and `paper_breakdowns` -- BOTH RAN CLEAN

```
python3 -m analysis.paper_eval --model ${SCR}/corrected/glotlidc_corrected.unilid \
  --scratch-dir ${SCR}/full_test_eval_corrected --out-dir outputs_corrected_round \
  --waive-released-model-gates

python3 -m analysis.paper_breakdowns --part all \
  --model ${SCR}/corrected/glotlidc_corrected.unilid \
  --scratch-dir ${SCR}/full_test_eval_corrected --out-dir outputs_corrected_round
```

Note: `paper_breakdowns` has no `--waive-released-model-gates` flag. Its argparse
takes `--part`, `--out-dir`, `--model`, `--scratch-dir` only; it detects the
non-default model itself and downgrades the published-cell comparisons to
informational.

### `paper_eval` banners

Gates 1-2 PASSED (language order; seed-301 split 18,001,573 / 27,002,441, matching
the stored `rule_split_seed301.npz`). Wiring gates A, B and C WAIVED, with the
reason printed in full: their reference CSVs and anchors (`{'baseline': 0.9117,
'gate_flat4_prox21': 0.9498}`, gate C's 0.9292 / 2.0263e-05) are released-model
measurements. The module states explicitly that none of them was read while
scoring, and that the seven CARRIED prediction memmaps were not loaded.
Sentinel guard: zero UNSEEN/EXCLUDED on the kept pool; EMPTY counts 0 / 0 / 0.

The module also emits a standing note for the camera-ready table: the submission's
Table 1 rows state N = 45,627,279 while these cells use the 45,377,279-line kept
pool, so **every** restated row's N must be updated, not only the new one.

### `paper_eval` headline cells

Full kept pool, 45,377,279 lines:

| config | corrected F1 | released F1 | dF1 | corrected FPR x1e5 | released FPR x1e5 | dFPR |
|---|---|---|---|---|---|---|
| baseline | 0.9327 | 0.9292 | +0.0035 | 2.0187 | 2.0263 | -0.0076 |
| gate_flat4_prox21 | 0.9564 | 0.9569 | -0.0005 | 1.7745 | 1.7665 | +0.0080 |
| fasttext | 0.9443 | 0.9443 | 0.0000 | 2.7063 | 2.7063 | 0.0000 |

Judge part, 27,002,441 lines:

| config | corrected F1 | released F1 | dF1 | corrected FPR x1e5 | released FPR x1e5 | dFPR |
|---|---|---|---|---|---|---|
| baseline | 0.9159 | 0.9117 | +0.0042 | 2.0301 | 2.0373 | -0.0072 |
| gate_flat4_prox21 | 0.9495 | 0.9498 | -0.0003 | 1.7825 | 1.7743 | +0.0082 |
| fasttext | 0.9332 | 0.9332 | 0.0000 | 2.7165 | 2.7165 | 0.0000 |

The fastText row is bit-identical in both rounds, as it must be: the same borrowed
`pred_fasttext.npy` (sha256 4ff74fb55ce5668b...) is the input to both.

Paired bootstrap, judge part, B=10,000, seed 0, gate_flat4_prox21 minus comparator:

| comparator | corrected | released |
|---|---|---|
| baseline | +0.0336 [+0.0290, +0.0383] | +0.0380 [+0.0328, +0.0434] |
| fasttext | +0.0163 [+0.0108, +0.0221] | +0.0166 [+0.0112, +0.0223] |

The correction lifts the baseline (+0.0035 full pool) and leaves the promoted
configuration essentially where it was (-0.0005), so the gate-minus-baseline
margin narrows by about 0.004 while the gate-minus-fastText margin is unchanged.

`lid_main.tex`'s UniLID GlotLID-C cells already read `\corrrev{.933}` /
`\corrrev{2.02e-5}`, which are exactly this run's corrected baseline cells.
The calibrated row's GlotLID-C cells still read `\camrev{.957}` / `\camrev{1.77e-5}`,
which are the *released* 0.9569 / 1.7665; the corrected values are 0.9564 / 1.7745,
so both round to the same 3 dp F1 (.956 vs .957 differ) -- flagging for the paper
owner, not changed here.

### `paper_breakdowns` -- published-cell comparison (informational, quoted in full)

The module printed the banner it was designed to print for a non-default model:

> INFORMATIONAL, NOT A GATE: script-table rows ['Beng'] differ from
> paper/submission.tex. Those published cells are measurements of the RELEASED
> model and this run scored .../corrected/glotlidc_corrected.unilid, so a
> difference here is an expected cross-model difference, not a regression and not
> a reproduction failure. The .tex fragment WAS written, with this run's
> regenerated numbers.

and the same sentence again for `resource-tier rows ['<500']`. It wrote
`paper_breakdowns_script.tex` and `paper_breakdowns_resource.tex` with this run's
numbers, each carrying a four-line provenance comment naming the corrected model,
the corrected prediction directory, and an explicit "NOT the paper/submission.tex
published numbers" line. The process exited 0.

Script-table within-stratum comparison (tolerance 0.005):

| group | our F1 | paper F1 | diff | status |
|---|---|---|---|---|
| Latn (1700) | 0.9443 | 0.9400 | +0.0043 | OK |
| Cyrl (70) | 0.8801 | 0.8770 | +0.0031 | OK |
| Arab (38) | 0.6926 | 0.6910 | +0.0016 | OK |
| Deva (32) | 0.8109 | 0.8110 | -0.0001 | OK |
| Beng (6) | 0.8790 | 0.8850 | -0.0060 | **MISMATCH** |
| Grek (4) | 0.6750 | 0.6770 | -0.0020 | OK |
| Hebr (4) | 0.7376 | 0.7400 | -0.0024 | OK |
| Armn (2) | 0.9721 | 0.9740 | -0.0019 | OK |
| Other (82, paper basis) | 0.9374 | 0.9370 | +0.0004 | OK |

Resource-tier within-stratum comparison:

| group | our F1 | paper F1 | diff | status |
|---|---|---|---|---|
| <500 (56) | 0.8572 | 0.8710 | -0.0138 | **MISMATCH** |
| 500--1k (40) | 0.9731 | 0.9750 | -0.0019 | OK |
| 1k--12k (458) | 0.9895 | 0.9900 | -0.0005 | OK |
| 12k--18k (526) | 0.9971 | 0.9970 | +0.0001 | OK |
| 18k--35k (398) | 0.9918 | 0.9920 | -0.0002 | OK |
| 35k+ (462) | 0.9576 | 0.9580 | -0.0004 | OK |

Under the released model both of these tables reproduce within tolerance; the two
mismatches are the corrected model's own movement. Beng within-stratum baseline
drops 0.8858 to 0.8790; the `<500` tier's within-stratum baseline drops 0.8709 to
0.8572 while its *global* baseline rises 0.5145 to 0.5957 -- the correction
reduces this tier's false positives into other strata much more than it helps its
own within-stratum recall. (The global-view comparison against the paper is a
recorded expected-mismatch cross-view comparison, not a gate, in both rounds.)

### `paper_breakdowns` residual part

Gates passed (language order; seed-301 split re-derived and bit-matching).

| config | n_wrong | EMPTY | head-true share | head-head share |
|---|---|---|---|---|
| gate_flat4_prox21 | 930,576 | 0 | 0.9914 | 0.8855 |
| floor21_gate | 963,563 | 0 | 0.9879 | 0.8917 |

The head-head mechanism remains the dominant residual, as recorded.

### Outputs written

`outputs_corrected_round/tables/`: `paper_eval.md`, `paper_eval_table1_row.tex`,
`paper_eval_appendix.tex`, `paper_breakdowns.md`, `paper_breakdowns_gate.md`,
`paper_breakdowns_script.tex`, `paper_breakdowns_resource.tex`, `promoted_residual.md`.
`outputs_corrected_round/diagnostic/`: `paper_eval_per_lang_f1_fullpool.csv`,
`paper_eval_per_lang_f1_judge.csv`, `promoted_residual_pairs.csv`.

---

## 5. `release_gates` -- base PASSED; calibrated BLOCKED (no corrected bundle exists)

### Ref convention (resolved from the code, not guessed)

`_resolve_paths` refuses a non-default model without `--ref`, with the message
"Record a reference from this model first and pass it with --ref." The reference
must be a full-length prediction array from the model under test; the gate takes
the seed-42 250,000-line test half out of it. So the corrected refs are exactly
`${SCR}/full_test_eval_corrected/pred_baseline.npy` (base) and
`.../pred_gate_flat4_prox21.npy` (calibrated). `--label` was used so the corrected
gate JSON does not overwrite the released one.

### `--mode base` -- PASS

```
RAYON_NUM_THREADS=32 python3 -m analysis.release_gates --mode base \
  --model ${SCR}/corrected/glotlidc_corrected.unilid \
  --ref ${SCR}/full_test_eval_corrected/pred_baseline.npy \
  --out-dir outputs_corrected_round/release --label corrected
```

```
gate base: model .../corrected/glotlidc_corrected.unilid
          reference .../full_test_eval_corrected/pred_baseline.npy
Collecting 250,000 golden lines from the test pool...
Loaded 1940 languages, vocab size 100000
Scoring (base)...
  scored 250,000/250,000
base gate: agreement 1.000000 (250,000/250,000) -> PASS; wrote outputs_corrected_round/release/gate_base_corrected.json
```

Exact equality on all 250,000 golden lines. The packaged corrected model's base
inference path reproduces the corrected analysis pipeline bit-for-bit.

### `--mode calibrated` -- NOT RUN, blocked on a missing artifact

`run()` constructs `UnilidModel(model_path, calibrated=True)` and passes **no**
`calibration=` argument; `release_gates.py` exposes no CLI flag for one. Calibrated
inference therefore requires a version-2 `.unilid` file with the calibration
section appended. Measured container versions:

| file | header version | calibration |
|---|---|---|
| `${SCR}/corrected/glotlidc_corrected.unilid` | 1 | none |
| `/capstor/store/.../release/unilid-1940-calibrated.unilid` | 2 | present, c = -21.0 |

A filesystem sweep of both the scratch and store trees found no corrected
calibrated bundle. The released version-2 file carries c = -21.0 -- the released
constant, not the corrected chain's -17.0 -- and gating it against the corrected
`pred_gate_flat4_prox21.npy` is precisely the cross-generation comparison
`_resolve_paths` exists to refuse.

**What the calibrated gate needs before it can run:** a version-2 corrected model
packed from `${SCR}/corrected/glotlidc_corrected.unilid` plus the calibration JSON
built in step 3 (`outputs_corrected_round/release/calibration_glotlidc_corrected.json`,
c = -17.0). `unilid.model_io.write_unilid`/`pack` takes a `calibration=` argument
and the CLI has a `--calibration` flag, so the artifact is producible; packing it
was not in this session's scope and was not done. No substitute was gated.

---

## 6. External-bench eval stages (UDHR, FLORES) -- BOTH ABORT

### The scored arrays are sound

The corrected score stages were re-run after the 2026-08-21 routing fix and their
outputs sit at `${SCR}/external_bench/scored_glotlidc_corrected/` (Aug 21 17:58 and
18:00), written at commit 7fb935e. Their meta sidecars record
`model_path: .../corrected/glotlidc_corrected.unilid` and
`sha256_W_loaded: a4aeff1994640322...`, which is the corrected model's base_W sha --
so these arrays genuinely came from the corrected model.

**Recorded for the log:** the *earlier* score jobs 3130020 (UDHR, Aug 21 08:06) and
3130021 (FLORES, Aug 21 08:31), at commit 31ee9f2, printed a banner naming the
corrected model but then logged `Loading model (.../unilid_analysis/glotlidc.unilid)`
-- the **released** model. That is the bug the 2026-08-21 fix addressed. Those runs
wrote to `${SCR}/external_bench/scored_{udhr,flores}.npz` at the top level; those
files are no longer present, so the bad artifacts were cleaned up and cannot be
picked up by the eval stage. `_bench_out_dir()` now routes any non-default model
into `scored_<model stem>/`, which is what prevents a recurrence.

### The abort

```
python3 -m analysis.external_bench_eval --stage eval --bench udhr \
  --model ${SCR}/corrected/glotlidc_corrected.unilid \
  --scratch-dir ${SCR}/full_test_eval_corrected --out-dir outputs_corrected_round
```

```
external benchmarks against a NON-DEFAULT model
  model   /capstor/scratch/cscs/cmeister747/unilid_analysis/corrected/glotlidc_corrected.unilid
  outputs /capstor/scratch/cscs/cmeister747/unilid_analysis/full_test_eval_corrected
  tables under outputs_corrected_round/tables
Traceback (most recent call last):
  ...
  File "/users/cmeister747/unilid_analysis/analysis/external_bench_eval.py", line 826, in run_eval
    raise RuntimeError(f"{meta_path} records floor_target {meta['floor_target']}, "
RuntimeError: /capstor/scratch/cscs/cmeister747/unilid_analysis/external_bench/scored_glotlidc_corrected/scored_udhr_meta.json records floor_target -17.0, expected FLOOR_TARGET -21.0 (analysis.full_test_floor21)
```

FLORES aborts identically at the same line, same root cause:

```
RuntimeError: .../scored_glotlidc_corrected/scored_flores_meta.json records floor_target -17.0, expected FLOOR_TARGET -21.0 (analysis.full_test_floor21)
```

Nothing was worked around and no analysis file was edited.

### Root cause (a finding, not a fix)

`analysis/external_bench_eval.py` is inconsistent about where it gets the clamp
constant across its own two stages:

- **score stage, line 518:** `target = float(fp.get("floor_target", FLOOR_TARGET))`
  -- reads the fingerprint, so the corrected run correctly used c = -17.0 and
  recorded `"floor_target": -17.0` in the meta sidecar.
- **eval stage, line 825:** `if meta["floor_target"] != FLOOR_TARGET:` -- compares
  against the module constant `FLOOR_TARGET = -21.0` (imported from
  `analysis.full_test_floor21`, the Exp 20 guard-selected released constant),
  with no fingerprint read and no CLI override.

The corrected chain's `fingerprint_floor21.json` carries `floor_target: -17.0`;
the released chain's carries `-21.0`. So the eval stage is reachable only for the
released model, and the score stage will always hand it a sidecar it rejects for
any model whose fingerprint uses a different constant. The check itself is doing
its job (it exists to catch an eval run against a score-stage npz built at a
different floor); the defect is that its reference is the module constant rather
than the same fingerprint the score stage read. `analysis/mistralnemo_eval.py`
solves the same problem with a `--floor-target` override; `external_bench_eval.py`
has no equivalent.

**A second, downstream blocker in the same function.** Even with the floor check
satisfied, `run_eval` at line 867 applies an unconditional acceptance gate:

```python
diff = abs(baseline_macro_f1 - reg["paper_baseline_f1"])
if diff > ACCEPTANCE_TOL:      # writes a failure .md and sys.exit(1)
```

`reg["paper_baseline_f1"]` is the **paper's released-model** cell (0.859 UDHR,
0.932 FLORES) and there is no non-default-model waiver on this path, unlike
`paper_eval`'s `--waive-released-model-gates` or `paper_breakdowns`' automatic
downgrade to informational. The module is otherwise non-default-model aware
(`_bench_out_dir()` at line 249, and the tau-CSV guard at line 619 that refuses to
read released-tree thresholds for a non-default model), so lines 825 and 867 look
like two spots that the non-default-model pass missed rather than a deliberate
design choice. Both need a decision before the corrected UDHR/FLORES cells exist.

### The paper cells these stages were meant to regenerate

Actual `paper/tables/lid_main.tex` cells (the task's "approximately .851/1.77e-4
and .925/3.16e-4" does not match the file; quoting what is there):

| row | UDHR F1 | UDHR FPR | FLORES F1 | FLORES FPR | GlotLID-C F1 | GlotLID-C FPR |
|---|---|---|---|---|---|---|
| `\unilid` | .859 | 1.43e-4 | .932 | 2.78e-4 | `\corrrev{.933}` | `\corrrev{2.02e-5}` |
| `\unilid (calibrated)` | .838 | 2.08e-4 | .933 | 2.91e-4 | `\camrev{.957}` | `\camrev{1.77e-5}` |

Those four UDHR/FLORES cells trace directly to the released round's
`outputs/tables/external_bench_udhr.md` (baseline 0.8590 / 14.2934e-5, gated
0.8383 / 20.7576e-5) and `external_bench_flores.md` (baseline 0.9317 / 27.7538e-5,
gated 0.9326 / 29.0747e-5). They remain released-model numbers; the corrected
round has produced **no** UDHR or FLORES cell, and none should be quoted until the
eval stage runs.

---

## Summary of state after this session

| step | outcome |
|---|---|
| 1. Verify job 3157817 | PASS -- both apply stages ran, all outputs present, counts independently re-derived; the 27 s runtime matches the released round's own 37 s for the same two stages |
| 2. Verify job 3158825 | PASS -- all five stages, clamp 1,431/1,940 confirmed, corrected cells 0.001-0.005 F1 below the paper's `calibrated_nemo.tex` and 0.06-0.08 (x1e5) above on FPR |
| 3. `build_release_calibration` | PASS -- 1,080 group A rows / 26 excluded, group B the expected four, no override needed |
| 4. `paper_eval` + `paper_breakdowns` | PASS -- both exited 0; two informational published-cell mismatches (Beng, `<500` tier), `.tex` fragments written with provenance comments |
| 5. `release_gates --mode base` | PASS -- exact equality, 250,000/250,000 |
| 5. `release_gates --mode calibrated` | BLOCKED -- no version-2 corrected bundle exists; needs packing from the step-3 calibration JSON |
| 6. `external_bench_eval --stage eval` (UDHR, FLORES) | ABORT -- hardcoded `FLOOR_TARGET = -21.0` check against a `-17.0` sidecar; a second unwaived released-model acceptance gate sits behind it |

### Open decisions for the user

1. Pack a version-2 corrected calibrated `.unilid` from
   `outputs_corrected_round/release/calibration_glotlidc_corrected.json` so
   `release_gates --mode calibrated` can run? (Not done; not in scope this session.)
2. `external_bench_eval.py` needs a `--floor-target`-style override (or a
   fingerprint read) at line 825, **and** a decision on the unwaived acceptance
   gate at line 867, before corrected UDHR/FLORES cells can exist. Both are
   code changes to an adversarially reviewed file; neither was made.
3. `lid_main.tex`'s calibrated GlotLID-C cells (`\camrev{.957}` / `\camrev{1.77e-5}`)
   are still released-model values; the corrected equivalents are .956 / 1.77e-5.

---

# Addendum, 2026-08-24 afternoon: both blockers cleared

Repo commit at run time: dd9c570 plus the working-tree edits described below.
Login node, same SCR and `outputs_corrected_round/` roots as above.

## A. `analysis/external_bench_eval.py` -- the two non-default spots fixed

Three edits, all no-ops for the default (released) model:

1. **Floor target (was line 825).** The eval stage's `meta["floor_target"] !=
   FLOOR_TARGET` is replaced by `_check_sidecar_floor_target(meta["floor_target"],
   meta_path)`, which delegates to `_expected_floor_target()`: the module constant
   for the default model (message byte-identical to before), the `floor_target`
   recorded in `<scratch-dir>/fingerprint_floor21.json` for any other -- the same
   file and field the score stage reads. A missing fingerprint or a missing field
   aborts naming the artifact; nothing falls back to the constant. The report's
   "Constants used" line now states the clamp actually in force and its provenance
   (`FLOOR_TARGET = -17.0 (.../full_test_eval_corrected/fingerprint_floor21.json)`),
   which for the default model renders exactly as before.
2. **Acceptance gate (was line 867).** `gates_binding = _ctx().is_default_model`,
   the same rule as `paper_breakdowns`. Default model: unchanged fatal gate,
   failure `.md`, `sys.exit(1)`. Any other model: the same comparison at the same
   `ACCEPTANCE_TOL`, printed and written as `INFORMATIONAL, NOT A GATE` (naming
   the model that was actually scored), no `.md`, exit 0.
3. **Output-root guard (new).** `configure()` now calls
   `model_context.resolve_out_root`. Without it, fix 1 would have opened a path
   where a non-default model with no `--out-dir` writes its baseline per-label CSV
   over `outputs/diagnostic/external_bench/` -- the released E2 record -- before
   the tau-CSV guard fires. `analysis.external_bench_eval` is now listed twice in
   `model_context_selfcheck.ENTRY_POINTS` (once per rule).

Probes: import; `--help`; 17/17 abort-path cases (default-model message verbatim,
sidecar/fingerprint mismatch both signs, missing fingerprint, missing field, all
three out-root refusals, both branches of the informational message);
`python -m analysis.model_context_selfcheck` 42/42, exit 0.

## B. Corrected UDHR and FLORES cells

```
python3 -m analysis.external_bench_eval --stage eval --bench {udhr,flores} \
  --model ${SCR}/corrected/glotlidc_corrected.unilid \
  --scratch-dir ${SCR}/full_test_eval_corrected --out-dir outputs_corrected_round
```

Both exited 0, reading `${SCR}/external_bench/scored_glotlidc_corrected/`.

| bench | config | corrected F1 | corrected FPR | released F1 | released FPR |
|---|---|---|---|---|---|
| UDHR (366 labels, 24,115 rows) | baseline | 0.8560 | 1.52e-4 | 0.8590 | 1.43e-4 |
| | floor21 | 0.8512 | 1.69e-4 | 0.8474 | 1.75e-4 |
| | gated | 0.8419 | 2.03e-4 | 0.8383 | 2.08e-4 |
| FLORES (190 labels, 192,280 rows) | baseline | 0.9313 | 2.83e-4 | 0.9317 | 2.78e-4 |
| | floor21 | 0.9320 | 2.86e-4 | 0.9323 | 2.85e-4 |
| | gated | 0.9324 | 2.91e-4 | 0.9326 | 2.91e-4 |

Against `paper/tables/lid_main.tex` (3 dp / 3 sf, as printed):

| row | cell | paper | corrected |
|---|---|---|---|
| `\unilid` | UDHR | .859 / 1.43e-4 | **.856 / 1.52e-4** |
| `\unilid` | FLORES | .932 / 2.78e-4 | **.931 / 2.83e-4** |
| `\unilid (calibrated)` | UDHR | .838 / 2.08e-4 | **.842 / 2.03e-4** |
| `\unilid (calibrated)` | FLORES | .933 / 2.91e-4 | **.932 / 2.91e-4** |

Direction: on UDHR the correction lowers the baseline (-0.003 F1, +0.09e-4 FPR)
and *raises* the calibrated row (+0.004 F1, -0.05e-4 FPR) -- the opposite sign to
the internal GlotLID-C pool, where the correction lifted the baseline and left the
gate flat. On FLORES every cell moves by at most 0.001 F1. The re-examination
accounting shifts with it (corrected, then released):

| bench | group | examined | moved | blocked | no_cand |
|---|---|---|---|---|---|
| UDHR | A corrected | 1,169 | 454 | 263 | 452 |
| UDHR | A released | 1,117 | 425 | 322 | 370 |
| UDHR | B corrected | 7 | 7 | 0 | 0 |
| UDHR | B released | 10 | 10 | 0 | 0 |
| FLORES | A corrected | 1,177 | 597 | 224 | 356 |
| FLORES | A released | 1,313 | 636 | 394 | 283 |
| FLORES | B corrected | 160 | 152 | 1 | 7 |
| FLORES | B released | 171 | 163 | 2 | 6 |

The corrected model sends fewer FLORES rows into re-examination in both groups and
is blocked by the proximity condition less often on both benchmarks.

Reports: `outputs_corrected_round/tables/external_bench_{udhr,flores}.md` and
`outputs_corrected_round/diagnostic/external_bench/*.csv`. The released tree's own
E2 record (Aug 7) was not touched.

## C. Corrected version-2 calibrated bundle, and the calibrated gate

The packing step exists as a runnable CLI: `UNILID/unilid/calibrate_cli.py`
subcommand `bundle` (`cmd_bundle`, lines 41-55; console script `unilid-calibrate`,
`UNILID/pyproject.toml:58`), documented at `OPEN_SOURCE_STATUS.md:224` as the step
that produced the released `unilid-1940-calibrated.unilid`. It refuses a model that
already bundles a calibration, loads the JSON through `Calibration.from_json_file`,
validates it against the model's own language list with `cal.runtime_for(langs)`
before writing, and refuses to write over its input.

```
cd UNILID && python3 -m unilid.calibrate_cli bundle \
  ${SCR}/corrected/glotlidc_corrected.unilid \
  ../outputs_corrected_round/release/calibration_glotlidc_corrected.json \
  -o ${SCR}/corrected/glotlidc_corrected_calibrated.unilid
```

`Wrote ... (version 2, calibration bundled, 1,940 languages)`. Verified after
writing: header magic `UNILID\0\0`, **version 2**, 1,940 languages, vocab 100,000,
779,663,677 bytes; `read_calibration` returns c = **-17.0**, head_n 18,000,
proximity 21.0, topk 5, group A 1,080, group B `{arg_Latn, bjn_Latn, sco_Latn,
vls_Latn}`; base tokenizer bytes and language list identical to the version-1
input; weight matrix bit-identical, sha256 `a4aeff1994640322...` (the corrected
base_W of `fingerprint_floor21.json`); container sha256
`135404c834e9e07435b99551c1c3a570cf3b2ac94cff6c26691e90796381dc91`. Nothing was
overwritten: the filename is new and the target was checked absent first.

```
RAYON_NUM_THREADS=32 python3 -m analysis.release_gates --mode calibrated \
  --model ${SCR}/corrected/glotlidc_corrected_calibrated.unilid \
  --ref ${SCR}/full_test_eval_corrected/pred_gate_flat4_prox21.npy \
  --out-dir outputs_corrected_round/release --label corrected
```

```
Applied unseen-token constant -17.0 (1655/1940 languages modified)
calibrated gate: agreement 1.000000 (250,000/250,000) -> PASS;
  wrote outputs_corrected_round/release/gate_calibrated_corrected.json
```

**PASS, and on exact equality**: `n_disagree` 0, `n_unexplained` 0, so the
>= 0.999-agreement-plus-boundary-classification allowance was not needed. The
runtime's own 1,655-of-1,940 clamp count matches `fingerprint_floor21.json`'s
`n_modified`. Both release gates now pass for the corrected generation, each
against a reference recorded from that generation.
