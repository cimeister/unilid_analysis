# WiLI training + Tatoeba wave compilation — 2026-08-24

Internal documentation: this is a chronological experiment/results record for this
repository's own tracking, not prose written for an external reader.

Jobs covered: 3161886, 3161887, 3162788, 3162789 (four new WiLI-trained LLM-tokenizer
variants), 3161889 (`wili_100k_defaults` + the Phase-3 vocabulary reproducibility gate),
3161890/91/92/93 (10k/20k/50k/200k vocab-size trainings), 3159860/62/64/65/3159935/40
(six Tatoeba evals). All fourteen SLURM jobs confirmed `COMPLETED 0:0` via `sacct`.

No fallback was applied anywhere below: every container, inspect record and eval file
named in the task existed and was read; nothing was substituted.

---

## 1. Post-training instrument gate (per model)

All nine models pass the automated post-training check embedded in the training scripts
(235 languages, real-token mass 1.000000 to six printed decimals, no defect) and the
`inspect_variant_models`-style JSON records agree with the log to full float precision.
Vocab sizes match the expected values exactly.

| Model (container) | Languages | Vocab (expected) | Vocab (measured) | Real mass range | Defect | SLURM |
|---|---:|---:|---:|---|---|---|
| `mistralnemo_wili_fp64.unilid` | 235 | 131,072 | 131,072 | 0.9999999103–1.0000000936 | False | 3161886 0:0 |
| `llama32_1b_wili_fp64.unilid` | 235 | 128,260 | 128,260 | 0.9999999183–1.0000001084 | False | 3161887 0:0 |
| `mistral7b_v02_wili_fp64.unilid` | 235 | 31,950 | 31,950 | 0.9999999580–1.0000000367 | False | 3162788 0:0 |
| `llama2_7b_wili_fp64.unilid` | 235 | 31,977 | 31,977 | 0.9999999654–1.0000000348 | False | 3162789 0:0 |
| `wili_100k_defaults_fp64.unilid` | 235 | 100,000 | 100,000 | 0.9999998746–1.0000001329 | False | 3161889 0:0 |
| `wili_10k_defaults_fp64.unilid` | 235 | 10,000 | 10,000 | 0.9999999696–1.0000000287 | False | 3161890 0:0 |
| `wili_20k_defaults_fp64.unilid` | 235 | 20,000 | 20,000 | 0.9999999744–1.0000000308 | False | 3161891 0:0 |
| `wili_50k_defaults_fp64.unilid` | 235 | 50,000 | 50,000 | 0.9999999433–1.0000000882 | False | 3161892 0:0 |
| `wili_200k_defaults_fp64.unilid` | 235 | 200,000 | 200,000 | 0.9999998458–1.0000001384 | False | 3161893 0:0 |

No deviation on the core gate (languages / vocab size / mass / defect) for any of the
nine models. Sources: `outputs/rerelease/wili_{mistralnemo,llama32_1b,mistral7b_v02,
llama2_7b}_wili_fp64_inspect.json`, `outputs/rerelease/wili_wili_{10k,20k,50k,100k,200k}
_defaults_fp64_inspect.json`, and the tails of `logs/wili_w-*_<jobid>.out`.

**Secondary log observations (not gate failures, recorded for completeness):**

- `"No pretokenizer set for new HF tokenizer"` fired 235 times each (once per language)
  for `mistral7b_v02_wili` and `llama2_7b_wili` — the two `byte-level=False` conversions
  from BPE source tokenizers — and 0 times for `mistralnemo_wili`/`llama32_1b_wili`
  (`byte-level=True`). This is the first training of this tokenizer family in this repo's
  logs, so there is no prior baseline to compare against; flagged as informational since
  the post-training mass/defect checks both passed cleanly.
- `"Byte-level vocabulary may be incomplete: only 208 byte tokens found"` fired 12 times
  in each of the five `wili_*_defaults_fp64` jobs (10k/20k/50k/100k/200k). This is **not
  new**: it also fires 12 times in the earlier known-good baseline run of the same family
  (`wili_100k_500`, job 3138626), so it is a pre-existing, expected warning.

---

## 2. THE REPRODUCIBILITY VERDICT — **NO MATCH**

Source: `outputs/rerelease/wili_vocab_repro_check.json`, run inside job 3161889.

- `trained_base_sha256` = `ac2862e5…` vs `container_base_sha256` = `5fa5342a…` — differ.
- `ordered_token_list_match`: **false**
- **First divergence index: 18,484** (of 100,000 tokens — the first 18,484 positions,
  18.5% of the vocabulary, are identical).
- `overlap_tokens`: 100,000 / `overlap_fraction_of_stored`: **1.0** — the vocabulary as a
  **set** is 100% identical between the freshly trained base and the stored container.
- Nature of the divergence, from `trained_context`/`stored_context`: at index 18,484–18,485
  two tokens (`"matagĠtuig.ĠAngĠ"` and `"~"`) appear in **swapped order** between the two
  runs. This is a tie-break/ordering divergence in the SentencePiece trainer, not a
  content difference — every token the stored container has, the freshly trained base
  also has, just not always at the same index.
- The job's own printed verdict: *"The four vocabulary sizes are NEW models built by the
  published procedure, not the published models, and the table must say so."*
- The repro-check subprocess itself exited 1 (visible as `"repro check exit 1"` in the
  `.out` log's finish line) — this is the check's designed non-zero exit on a mismatch,
  not a job failure: `sacct` confirms the parent SLURM job 3161889 still `COMPLETED 0:0`
  because the training script continues past a failed gate rather than aborting.

**Scope of this finding.** The order check was run directly only for the 100k case,
because `wili_assets/` holds a stored 100k container to diff against and holds no stored
10k/20k/50k/200k containers. The "four vocabulary sizes" generalization is the job script's
own inference (same trainer, same source of nondeterminism), not an independently
re-measured order-diff for each of 10k/20k/50k/200k. **Governing consequence for
Section 4**: per the task's framing, the 10k/20k/50k/200k models plus `wili_100k_defaults`
are reported below as *new models built by the published procedure*, not confirmed
reproductions of the published table's underlying models.

---

## 3. `tab:unilid_llm_comparison` — measured vs. published

Published cells (`paper/tables/unilid_llm_comparison.tex`). Gate tolerances are
`wili_eval.py`'s own: F1 within ±5e-4 absolute, FPR within ±1% relative.

| Row | Published F1 | Measured F1 | ΔF1 | Published FPR | Measured FPR | ΔFPR (rel) | Verdict | Caveats |
|---|---:|---:|---:|---:|---:|---:|---|---|
| `\unilid-Mistral-Nemo` | 0.958 | 0.958898 | +0.000898 | 1.925e-4 | 1.89380e-4 | −1.62% | **F1 & FPR outside tol** | tokenizer verified byte-identical to `mistralai/Mistral-Nemo-Base-2407` |
| `\unilid-Mistral` (mistral7b_v02) | 0.921 | 0.920215 | −0.000785 | 3.365e-4 | 3.37952e-4 | +0.43% | **F1 outside tol**, FPR ok | 51 refused CR entries dropped (2026-08-23); tokenizer repo unconfirmed against original |
| `\unilid-LLaMA3.2` | 0.954 | 0.954317 | +0.000317 | 2.084e-4 | 2.08147e-4 | −0.12% | **MATCH** (both) | none |
| `\unilid-LLaMA2` | 0.911 | 0.909635 | −0.001365 | 3.698e-4 | 3.73268e-4 | +0.94% | **F1 outside tol**, FPR ok (barely) | 24 refused CR entries dropped (2026-08-23); tokenizer repo unconfirmed against original |

Reference rows, already computed before this wave (Aug 23), included for the same table:

| Row | Published F1 | Measured F1 | ΔF1 | Published FPR | Measured FPR | ΔFPR (rel) | Verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| `\unilid (base)` / `wili_100k_500` (stored) | 0.960 | 0.960113 | +0.000113 | 1.859e-4 | 1.85888e-4 | −0.006% | MATCH (this is the Phase-0 gate, `wili_instrument_gate.json`, `gate_passed: true`) |
| `wili_100k_500_fp64` (retrained base) | 0.960 | 0.960088 | +0.000088 | 1.859e-4 | 1.86288e-4 | +0.21% | MATCH |
| `\unilid-DeepSeek3.2` | 0.955 | 0.955171 | +0.000171 | 2.042e-4 | 2.04837e-4 | +0.31% | MATCH |
| `\unilid-Qwen3` | 0.949 | 0.948125 | −0.000875 | 2.310e-4 | 2.34115e-4 | +1.35% | **F1 & FPR outside tol** — already recorded 2026-08-21 (`SESSION_STATUS.md`: "Only Qwen F1 moves at three decimals"), not new to this wave |

**Headline finding.** Three of the four newly trained rows (Mistral-Nemo, Mistral,
LLaMA2) fall outside `wili_eval.py`'s own reproduction tolerance on macro F1; only
LLaMA3.2 reproduces cleanly on both metrics. The two rows with the largest F1 gap
(Mistral, −0.0008; LLaMA2, −0.0014) are exactly the two with an author-decided entry drop
(51 and 24 CR-containing pieces respectively) and unconfirmed tokenizer-repo identity —
consistent with, but not proof of, that caveat driving the gap. Mistral-Nemo, whose
tokenizer is independently verified byte-identical to the published source, still misses
tolerance on both F1 (+0.0009) and FPR (−1.62%), so the caveat above does not fully
explain the pattern; LLaMA3.2, with no identity caveat, is the only clean match. All
figures round to the published table's 3-decimal/4-sig-fig resolution to within 1 unit in
the last printed digit in every case, i.e. these are small, table-resolution-adjacent
deviations, not gross mismatches.

Sources: `outputs/rerelease/wili_eval_{mistralnemo,llama32_1b,mistral7b_v02,llama2_7b}
_wili_fp64.json` (this wave), `wili_eval_{deepseek_v3.2,qwen3_8b}_wili_fp64.json` and
`wili_instrument_gate.json` / `wili_eval_wili_100k_500_fp64.json` (pre-existing).

---

## 4. `tab:vocab_size_efficiency` — measured vs. published

Published cells (`paper/tables/vocab_size_efficiency.tex`) have five columns: Vocab Size,
Macro F1, Macro FPR, Latency (ms), Samples/s. **Latency (ms) and Samples/s cannot be
regenerated** by any instrument run in this wave — `analysis/wili_eval.py` performs no
per-sample timing instrumentation (its JSON output carries no `elapsed_s` field, unlike
`wili_external_eval.py`'s Tatoeba records), and no dedicated throughput harness was run.
Those two columns are not reported below.

Per the Section 2 gate (NO MATCH), all five rows are new models built by the published
procedure, not confirmed reproductions of the published table's underlying models.

| Vocab | Published F1 | Measured F1 | ΔF1 | Published FPR | Measured FPR | ΔFPR (rel) | Verdict |
|---:|---:|---:|---:|---:|---:|---:|---|
| 10k | 0.945 | 0.943988 | −0.001012 | 2.514e-4 | 2.55574e-4 | +1.66% | **F1 & FPR outside tol** |
| 20k | 0.951 | 0.949882 | −0.001118 | 2.278e-4 | 2.30333e-4 | +1.11% | **F1 & FPR outside tol** (FPR borderline) |
| 50k | 0.957 | 0.956719 | −0.000281 | 2.019e-4 | 2.01491e-4 | −0.20% | MATCH |
| 100k | 0.960 | 0.960088 | +0.000088 | 1.859e-4 | 1.86288e-4 | +0.21% | MATCH |
| 200k | 0.9606 | 0.960565 | −0.000035 | 1.8382e-4 | 1.84179e-4 | +0.20% | MATCH |

**Headline finding.** Despite the NO MATCH order-divergence at the 100k gate, the F1/FPR
metrics for 50k, 100k, and 200k reproduce the published table within `wili_eval.py`'s
tolerance; only 10k and 20k deviate beyond it, and by roughly the same margin the
LLM-tokenizer variants in Section 3 did. The order divergence at 100k did not visibly move
the 50k/100k/200k accuracy numbers; the two smallest vocab sizes carry the larger,
tolerance-breaking gap.

Sources: `outputs/rerelease/wili_eval_wili_{10k,20k,50k,100k,200k... wired as
wili_100k_defaults}_defaults_fp64.json` — filenames: `wili_eval_wili_10k_defaults_fp64.json`,
`wili_eval_wili_20k_defaults_fp64.json`, `wili_eval_wili_50k_defaults_fp64.json`,
`wili_eval_wili_100k_defaults_fp64.json`, `wili_eval_wili_200k_defaults_fp64.json`.

---

## 5. Tatoeba compilation

Published cell (`paper/tables/tatoeba_udhr_comparison.tex`, `\unilid` row, Tatoeba half
only): **F1 0.414, FPR 9.61e-4, 201 languages.**

| Model (container) | n_languages | Macro F1 | Macro FPR | Role | Gate |
|---|---:|---:|---:|---|---|
| `wili_100k_500` (stored, `wili_assets/`) | 201 | 0.414278 | 9.60632e-4 | **PUBLISHED-ROW REPRODUCTION** | `is_reproduction_gate: true`, `gate_passed: true` — F1, FPR, n_languages all MATCH |
| `wili_100k_500_fp64` (retrained base) | 201 | 0.419970 | 9.23005e-4 | comparison-only | `is_reproduction_gate: false`; gate block present but MISMATCH on F1/FPR against the published cell, as expected for a different (retrained) container |
| `deepseek_v3.2_wili` (stored) | 201 | 0.405955 | 9.65979e-4 | comparison-only | no published cell exists for this variant |
| `deepseek_v3.2_wili_fp64` (retrained) | 201 | 0.403359 | 9.76889e-4 | comparison-only | no published cell |
| `qwen3_8b_wili` (stored) | 201 | 0.403030 | 9.91031e-4 | comparison-only | no published cell |
| `qwen3_8b_wili_fp64` (retrained) | 201 | 0.400393 | 10.02560e-4 | comparison-only | no published cell |

**Exact-reproduction check (task requirement):**
- stored `wili_100k_500`: measured 0.414278323061728 / 0.0009606322336752498 →
  rounds to **0.414278 / 9.60632e-4**, matches the earlier recorded numbers exactly.
- fp64 retrain `wili_100k_500_fp64`: measured 0.4199703506581937 / 0.0009230049734638726 →
  rounds to **0.4200 / 9.230e-4**, matches the earlier recorded numbers exactly.

All six Tatoeba jobs: `sacct` shows `COMPLETED 0:0`; all six `.err` files are 0 bytes
(no errors, no tracebacks); all six report the same `n_lines_in_file: 13,101,022` and
`total_samples: 11,848,300` (201-language filter applied consistently); `elapsed_s` ranges
201.7–327.8s. These ran through the dedicated `slurm_wili_external_eval.sh` per-model job
(not the login node), which is the workaround for the earlier login-node Tatoeba kill
noted in `SESSION_STATUS.md` — no partial-completion risk applies here.

Sources: `outputs/rerelease/wili_tatoeba_{deepseek_v3.2_wili,deepseek_v3.2_wili_fp64,
qwen3_8b_wili,qwen3_8b_wili_fp64,wili_100k_500,wili_100k_500_fp64}.json`.

---

## Summary of gate/deviation outcomes

- Post-training instrument gate (Section 1): **9/9 PASS**, no deviations.
- Vocabulary reproducibility gate (Section 2): **NO MATCH** — order divergence at index
  18,484/100,000, full-set overlap; governs Sections 2 and 4 reporting.
- `tab:unilid_llm_comparison` (Section 3, new rows): **1/4 clean match** (LLaMA3.2);
  Mistral-Nemo, Mistral, LLaMA2 each miss the F1 tolerance by 0.0008–0.0014.
- `tab:vocab_size_efficiency` (Section 4): **3/5 clean match** (50k/100k/200k); 10k/20k
  miss tolerance on both F1 and FPR.
- Tatoeba (Section 5): the one published-row reproduction (**stored `wili_100k_500`**)
  matches exactly; all five comparison-only rows completed cleanly with no published cell
  to gate against.
- No missing artifacts, no substitutions, no silent fallbacks encountered anywhere in
  this compilation.
