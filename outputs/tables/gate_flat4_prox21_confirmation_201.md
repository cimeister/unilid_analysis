# gate_flat4_prox21 balanced-test confirmation (draw 201)

Instrument: balanced test draw, seed 201, 185,204 lines, disjoint from the working balanced-val draw (seed 101). Memmap subset over the already-scored full-pool prediction memmaps in `/capstor/scratch/cscs/cmeister747/unilid_analysis/full_test_eval`; no scoring runs in this script. Promotion round 2026-08-06 (user decision): gate_flat4_prox21 promoted on the natural track, superseding floor21_gate; the pre-registered composed step of Experiment 50 (rebuilding the gate on the pooled-frequency floor matrix) is skipped by the same decision. This artifact is the required draw-201 confirmation.

## Within-stratum macro-F1, draw-201

Rows for baseline and the promoted configuration gate_flat4_prox21. Stratum definitions: tail is N<1,000, magnets is category==flat_magnet, twins is category==twin, head is N>=18,000, all from `outputs/diagnostic/full_test_per_lang_prf.csv`.

| config | overall | tail | magnets | twins | head |
|---|---|---|---|---|---|
| gate_flat4_prox21 | 0.9781 | 0.8763 | 0.8811 | 0.9437 | 0.9814 |
| baseline | 0.9809 | 0.9086 | 0.9121 | 0.9435 | 0.9817 |

## Sanity gate: baseline overall row vs the recorded headline

Comparable recorded number: `outputs/tables/two_sided_selection.md`, "Headline: balanced test draw, within-stratum macro-F1" table, baseline row (same draw-201 instrument, same within-stratum macro-F1 construction). Recomputed here vs recorded, tolerance 0.001 (recorded table rounds to 4 decimals):

| stratum | recomputed | recorded | abs diff |
|---|---|---|---|
| overall | 0.9809 | 0.9809 | 0.00003 |
| tail | 0.9086 | 0.9086 | 0.00001 |
| magnets | 0.9121 | 0.9121 | 0.00001 |
| twins | 0.9435 | 0.9435 | 0.00004 |
| head | 0.9817 | 0.9817 | 0.00002 |

Max abs diff 0.00004, gate PASS.


## Per-language collapse check, draw-201 global F1

Global per-language F1 (`_per_lang_stats`, full confusion, not within-stratum) for both configs on the draw-201 subset, with per-language support (`np.bincount` of true labels on the draw). Collapse: support >= MIN_COLLAPSE_SUPPORT (10) and (baseline F1 minus gate_flat4_prox21 F1) > LANG_COLLAPSE_BOUND (0.1).

Supported collapses: 3.

This check runs on the balanced draw, where each language has at most 100 true lines (the draw's construction), so a handful of flipped predictions moves a language's F1 by more than 0.1. This is a different instrument from the clause (C) promotion gate, which runs on the far larger judge part (millions of lines per language band) and recorded zero supported collapses for gate_flat4_prox21 (Exp 49, EXPERIMENTS_RESULTS.md). The draw's construction caps support at 100 lines per language, so its collapse count here is a noisier instrument than the judge-part clause of record, which was zero. The counts below describe the draw-201 subset only, not a re-evaluation of the promotion clause.


| lang | F1 gate_flat4_prox21 | F1 baseline | support |
|---|---|---|---|
| npi_Latn | 0.8571 | 0.9677 | 16 |
| sbs_Latn | 0.3000 | 0.4571 | 12 |
| thv_Latn | 0.4000 | 0.5455 | 16 |

Sub-support languages (support < 10) exceeding the bound, reported informationally, exempt from the collapse count: 4.

| lang | F1 gate_flat4_prox21 | F1 baseline | support |
|---|---|---|---|
| bor_Latn | 0.4444 | 0.6000 | 7 |
| kei_Latn | 0.8000 | 0.9412 | 9 |
| sby_Latn | 0.2500 | 0.6000 | 7 |
| tzl_Latn | 0.5455 | 0.8571 | 8 |

## Input memmap provenance

sha256 prefix (first 16 hex characters) of each input memmap as it existed at run time.

| file | sha256 (first 16) |
|---|---|
| y_true.npy | 9d62ce57eb2ea07a |
| pred_baseline.npy | 235380aa759b35fc |
| pred_gate_flat4_prox21.npy | 9b0ad2ccb670d836 |
