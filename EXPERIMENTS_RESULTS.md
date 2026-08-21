# UniLID Analysis — Results Summary

> **Reconstruction provenance.** This file was reconstructed on 2026-05-27 after the
> original Claude session transcript (session `9729f7f3-3af8-42d5-818a-1f032a9f6f25`,
> 91 prompts, 2026-03-26 → 2026-04-08) was lost. It is a reorganization of content that
> already existed in `EXPERIMENTS.md` (the session's own write-up, last modified
> 2026-04-08) cross-checked against the analysis code, the generated tables in
> `outputs/tables/`, and the recovered prompt history. Every number below traces to a
> table file under `outputs/tables/` and the corresponding section of `EXPERIMENTS.md`.
> No results were regenerated; if a number cannot be reproduced from the cited artifact,
> treat it as unverified. Inferences are marked **[inferred]**.

This summarizes the most important findings. The detailed narrative (per-experiment
prose, by-axis breakdowns, full tables) lives in `EXPERIMENTS.md`. Experiment design and
search spaces are in `EXPERIMENTAL_SETUP.md`. Status of planned work is in
`EXPERIMENTS_PLAN.md`. Chronology and SLURM job records are in
`EXPERIMENTS_CHRONOLOGICAL.md`.

The system under study, **UniLID**, is a language-ID model: a single shared Unigram
tokenizer (100k vocabulary) with 1,940 per-language log-probability weight vectors.
A text is scored per language by `score(lang) = Σ log p(token_i | lang)` under that
language's own Viterbi segmentation; the argmax over 1,940 languages is the prediction.
Evaluation is on the GlotLID test set (45,627,279 samples). Most sweeps run on a 500k
uniform sample (`seed=42`, without replacement).

---

## Both LLM-tokenizer variants carry the EM corruption; the plateau scan missed one (2026-08-18)

**Correction of my own conclusion earlier the same day.** I wrote that "the
DeepSeek3.2 model shows no sign of this failure", on the strength of the plateau
diagnostic alone. **That was wrong.** The plateau diagnostic is insensitive to the
milder form of the corruption; the retrain gate is not.

**How it was found.** `analysis/gate_correction.py` retrains a language from its
own corpus under the patched trainer and compares against the stored row. Run on
the corrected DeepSeek3.2 model with `azj_Latn` named explicitly (a `--langs`
option added for the purpose), against a size-spread selection:

| language | N_L | signed mean, nats | correlation | verdict |
|---|---|---|---|---|
| `kdr_Latn` | 85 | -5.9e-07 | 1.00000000 | pass |
| `wib_Latn` | 9,431 | -4.3e-05 | 1.00000000 | pass |
| `qxh_Latn` | 17,446 | +2.1e-05 | 1.00000000 | pass |
| `adh_Latn` | 23,111 | +9.6e-08 | 1.00000000 | pass |
| `ace_Latn` | 47,937 | +1.5e-05 | 1.00000000 | pass |
| `zul_Latn` | 100,000 | +6.7e-05 | 1.00000000 | pass |
| **`azj_Latn`** | **100,000** | **+1.0002** | **0.7057** | **FAIL** |
| `bod_Tibt` | 45,476 | +3.5365 | 0.9963 | FAIL |

`zul_Latn` is the control: same corpus size, same 100,000-line cap, same draw
question, and it reproduces to correlation 1.00000000. `azj_Latn` differs by a
signed mean of one full nat at correlation 0.71. That is not a sampling
difference.

**Why the plateau scan missed it.** In the DeepSeek3.2 model `azj_Latn`'s plateau
sits 1.7 sd below expectation, inside the normal range, while in the Qwen3 model
the equivalent row was driven to the hard training floor at 20.1 sd. The
corruption is present in both and its effect on the plateau differs by an order
of magnitude. **The retrain gate is the instrument that catches both; neither the
plateau scan nor the degeneracy scan does.** Recorded because the natural
inference from a clean plateau scan is that a model is sound, and that inference
is not safe.

**The timeline makes this expected rather than surprising.** Both LLM-tokenizer
variants were built 2026-03-27. The fp64 EM bug was fixed 2026-07-27 (Exp 42),
four months later. The base model and the Mistral-Nemo variant were retrained
after the fix; these two were not. So both are expected to carry it, and both do.

**The base model is not in the same category.** Gated with the same three hard
languages named explicitly:

| language | N_L | base model signed mean | correlation | DeepSeek3.2, same language |
|---|---|---|---|---|
| `zul_Latn` | 100,000 | -1.03e-03 | 0.999954 | +6.7e-05 |
| `azj_Latn` | 100,000 | -1.08e-02 | 0.999563 | **+1.0002** |
| `bod_Tibt` | 45,476 | -9.83e-02 | 0.996955 | +3.5365 |
| `mya_Mymr` | 100,000 | -1.74e-01 | 0.946718 | not gated |

The base model's `azj_Latn` discrepancy is **93 times smaller** than the
DeepSeek3.2 one, and only 1.08x the gate's `MAX_ABS_SIGNED_MEAN` threshold. Those
three languages are the hard cases (a capped corpus so a different draw, minority
scripts), and the gate's thresholds were calibrated on a size-spread sample that
contained none of them. **Read as threshold calibration on hard languages, not as
corruption**, which is consistent with the base model having zero plateau
outliers and having been retrained under the patched trainer. It is worth a
closer look and it is not the same phenomenon.

**Action, extending the author's decision of 2026-08-18 for Qwen3 to the
identical defect in its sibling: retrain DeepSeek3.2 too.** Job 3112879,
`slurm_deepseek_train_fp64.sh`. Qwen3 is job 3112846.

**Artifacts:** `outputs/rerelease/gate_correction_deepseek.json`,
`outputs/rerelease/gate_correction_base_azj.json`.

## The Qwen3 variant's Azerbaijani row is corrupted, independently of the special-token defect (2026-08-18)

**How this was found.** Inspecting the two variant models from the co-author's
Drive folder, the Qwen3 model's unseen-token plateau reached exactly
log(1e-12) = -27.631, the training floor, which the base model never reaches.
Two rows sit there: `azj_Latn` and `bod_Tibt`.

**The diagnostic, and it is B0's result used as an instrument.** B0 established
that a row's plateau is near-deterministic in corpus size. Fitting that relation
*within* each model gives a per-language expectation, and a row far below its own
expectation has something wrong that corpus size does not explain
(`analysis/variant_plateau_outliers.py`, robust fit, outliers at 5 sd).

The relation reproduces on both variant vocabularies, which is independent
confirmation of B0 on models B0 never saw:

| model | vocabulary | fitted slope, nats per decade of lines | residual sd | rows beyond 5 sd |
|---|---|---|---|---|
| `glotlidc` (base) | 100,000 | -2.068 | 0.298 | **0** |
| `deepseek_v3.2` | 128,819 | -2.016 | 0.424 | 3 |
| `qwen3_8b` | 151,670 | -2.010 | 0.409 | 5 |

**Finding: `azj_Latn` in the Qwen3 model is 20.1 sd below expectation**, plateau
-27.631 against an expected -19.413, and it appears in no other model. The
project's own degeneracy scan does not flag it, because it retains more than 100
tokens above its plateau.

**Why this is the fixed-vocabulary EM bug and not vocabulary coverage.** The two
causes are separable and the record already distinguishes them:

- **Coverage** shows up wherever the vocabulary lacks a script's pieces.
  `bod_Tibt` and `got_Goth` are flagged in both variants and are minority-script
  languages; the degeneracy scan flags them too. That is the benign cause.
- **`azj_Latn` is Latin script with N_L = 100,000**, fully covered by a
  general-purpose LLM tokenizer, and it is flagged in the Qwen3 model **only**.
  It is also the exact language `EXPERIMENTAL_SETUP.md` records as the fp64 EM
  bug's trigger: "one 142,136-byte line in the Azerbaijani corpus, the longest
  line in all 1,940 corpora". That bug overflowed 32-bit expected counts and the
  fork's guard mapped non-finite counts to zero, **"deleting exactly the most
  frequent tokens and leaving a plausible-looking but collapsed model"**, and it
  was recorded as producing partial collapse "without ever crossing the
  degeneracy threshold". That is precisely this row's signature.

The base model has **zero** rows beyond 5 sd, which both confirms the diagnostic
is not firing indiscriminately and is consistent with the record that the 100k
production model was retrained with the patched trainer.

**Conclusion: the Qwen3 model was very likely trained with the pre-fix (fp32)
trainer.** `mya_Mymr` at -8.5 sd, also unique to Qwen3 and also at N_L = 100,000,
may be a second casualty; Burmese script coverage is the competing explanation
and has not been separated. The DeepSeek3.2 model shows no sign of this failure.

**Consequences.**

1. **The special-token correction does not fix this.** It is a separate,
   pre-existing defect in that model's training run, and correcting the
   special-token mass leaves the corrupted row corrupted.
2. **The published Qwen3 row of `tab:lid_main` rests on this model.** Whatever is
   done about it, the row cannot be presented as an ordinary training run without
   either repairing the model or stating the defect.
3. **Decision required from the author** before the Qwen3 cells are regenerated:
   retrain the variant with the patched trainer, report it with the defect
   stated, or drop the row. Recorded as open in `EXPERIMENTS_PLAN.md`.

**Artifacts:** `analysis/variant_plateau_outliers.py`,
`analysis/inspect_variant_models.py`,
`outputs/rerelease/variant_plateau_outliers.json`,
`outputs/rerelease/variant_models_inspect.json`.

## The tail under both views, corrected model, and a correction to my own framing (2026-08-19)

**Framing I got wrong.** I reported the clamp's tail cost as "smaller than the
released model's, -0.0171 against -0.0204". Those are deltas measured from
different baselines, and the endpoint is what matters. On the within-stratum view
the corrected clamped configuration ends at 0.8875 against the released clamped
configuration's 0.8928, so it is **0.0053 worse**, not better.

**Both views, 96 tail languages (N_L < 1,000), 7,735 true examples:**

| config | within-stratum tail F1 | global tail F1 | precision | recall | FPs into tail labels |
|---|---|---|---|---|---|
| released baseline | 0.9132 | 0.5618 | 0.459 | 0.874 | 22,522 |
| released floor-21 | 0.8928 | 0.7655 | 0.763 | 0.842 | 9,103 |
| corrected baseline | 0.9045 | 0.6292 | 0.544 | 0.862 | 18,162 |
| **corrected floor-17** | **0.8875** | **0.7743** | **0.784** | **0.836** | **8,727** |

**The two views disagree in opposite directions, as Exp 24 established they
must.** Under the within-stratum view the corrected clamped configuration is the
weakest of the four. Under global per-language F1, which counts the false
positives *into* tail labels that the within-stratum view excludes by
construction, it is the strongest on record: 0.7743, and 8,727 false positives
into tail labels against 22,522 at the released baseline.

**The special-token correction on its own is tail-positive under the global
view**: 0.5618 to 0.6292 (+0.067) with 4,360 fewer false positives into tail
labels, before any clamp is applied.

**The trade the clamp makes, stated plainly:** tail recall 0.874 to 0.836
(-3.8pp), tail precision 0.459 to 0.784 (+32.5pp). Which of those matters is a
deployment question, and the paper already argues the precision side at
`submission.tex:824` ("a poor FPR for a low-resource language can lead to a
training corpus dominated by noise").

**Not the shipped configuration.** Exp 20 declined to adopt floor-21 alone
precisely on the within-stratum tail. The promoted configuration adds the
re-examination gate, which targets the low-margin decisions this cost is
concentrated in. Those numbers do not exist yet: job 3123324 (group-A thresholds)
then `gate_variants`.

**A lever, if the tail is to be weighted more heavily.** c is selected on
validation overall macro F1 under the all-strata guard. c = -15 clamps only 317
of 1,940 rows and would cost the tail less; -19 and -21 clamp all 1,940. Changing
the selection criterion to weight the tail is legitimate, but it is a change of
objective and would have to be stated and pre-registered, not picked after seeing
these numbers.

**Artifacts:** `analysis/tail_views_corrected.py`,
`outputs/rerelease/tail_views_corrected.json`.

## Both variant retrains completed; every corrupted row disappeared (2026-08-21, jobs 3112879, 3112846)

**Hypothesis under test:** whether retraining under the patched fp64 trainer
removes the EM corruption, and whether the rows flagged by the plateau diagnostic
were corruption or vocabulary coverage.

**Both retrained models are clean of the special-token defect**: real-token mass
1.000000 in every row, special mass 4.0e-12.

**The outlier lists resolve the classification question:**

| model | flagged before | flagged after |
|---|---|---|
| DeepSeek3.2 | `bod_Tibt` -11.0, `got_Goth` -9.9, `nqo_Nkoo` -5.2 | `got_Goth` -8.4, `nqo_Nkoo` -5.2 |
| Qwen3-8B | `bod_Tibt` -21.8, `azj_Latn` -20.1, `mya_Mymr` -8.5, `got_Goth` -6.9, `kyu_Kali` -5.1 | `got_Goth` -6.9, `kyu_Kali` -5.2 |

**`azj_Latn`, `bod_Tibt` and `mya_Mymr` all disappear after retraining; `got_Goth`,
`nqo_Nkoo` and `kyu_Kali` persist.** A row whose anomaly survives a retrain on the
same corpus is a vocabulary-coverage effect; one that vanishes was corruption in
the training run.

**This settles `mya_Mymr`, which was recorded as unresolved.** It was a second EM
casualty in the Qwen3 model, not Burmese script coverage. **It also corrects my
earlier classification of `bod_Tibt`**, which I had called a coverage effect
because it appeared in both variants; it was corruption in both.

**Artifacts:** `outputs/rerelease/{deepseek,qwen3}_fp64_inspect.json`,
`outputs/rerelease/{deepseek,qwen3}_fp64_plateau_outliers.json`.

## tab:lenbias-norm regenerated on the golden subset, with the implementation check restored (2026-08-21, job 3129778)

The first corrected run omitted the Original column, because half the 500,000-line
draw is the validation half the full-pool runs exclude. Rebuilt on the golden
subset (the test half, 250,000 lines, inside the scored pool) with Original taken
from the corrected model's own `pred_baseline.npy`.

| length | N | Original | Raw rescore | Normalized | published Normalized |
|---|---|---|---|---|---|
| <30 | 13,708 | 0.795 | 0.795 | 0.494 | 0.566 |
| 30-75 | 88,503 | 0.951 | 0.951 | 0.776 | 0.842 |
| 75-150 | 97,861 | 0.977 | 0.977 | 0.883 | 0.925 |
| 150-300 | 43,566 | 0.988 | 0.988 | 0.946 | 0.966 |
| 300+ | 6,362 | 0.994 | 0.994 | 0.986 | 0.991 |
| Overall | 250,000 | **0.960** | **0.960** | **0.837** | 0.885 |

**Raw rescore reproduces the plain scorer at agreement 1.000000**, which is the
implementation check the published table's caption claims and which the first
corrected run could not perform. Original and Raw both still read 0.960, matching
the published value. Normalization is more damaging on the corrected model, 0.960
to 0.837 against the published 0.960 to 0.885, so the paper's conclusion
strengthens.

## UDHR and FLORES-200 scored on the corrected model (2026-08-21, jobs 3130020, 3130021)

Score stage complete: UDHR 24,115 rows over 366 labels, FLORES-200 192,280 rows
over 190 labels, both with 0 empty after preprocess and 0 rows with fewer than 5
saved candidates, clamped at c = -17 with 1,655 of 1,940 rows modified. **The eval
stage is blocked on the group B thresholds**, which come from `gate_variants`.

**A provenance defect I introduced and fixed the same day.** The first run wrote
`scored_udhr.npz` and `scored_flores.npz` into the shared `external_bench`
directory, overwriting the released model's E2 artifacts from 2026-08-07, and its
sidecar recorded `model_path` as the released model and `floor_target` as -21.0,
both read from module constants rather than from the run.

**The scored data itself was correct**: the sha256 check against the corrected
`fingerprint_floor21.json` passed, which it could not have done had the released
weights been loaded, and the log records the clamp at c = -17 with the corrected
model's 1,655 rows. Only the recorded strings were wrong. The mislabelled files
were deleted rather than left in the chain, `analysis/external_bench_eval.py` now
records the model and constant actually used, and a non-default model writes to
`external_bench/scored_<model stem>/` so it cannot overwrite another model's
arrays. Re-run and verified.

**The released model's own scored npz is now absent and would need regenerating**
if its E2 numbers are ever recomputed. The directory is on scratch, not
store-backed, so no published artifact was lost.

## The WiLI evaluation instrument reproduces the paper exactly (2026-08-21)

**Hypothesis under test:** whether the WiLI numbers can be measured here at all.
No WiLI tooling existed in this repository, and `UNILID/eval.py` reports macro
precision and recall but **no macro FPR**, which every WiLI table quotes.

`analysis/wili_eval.py` takes its metrics from `analysis/metrics.py`, whose FPR is
identical to the convention recorded in `analysis/paper_eval.py`:
`tn = n - tp - fp - fn` with `support = tp + fn` is exactly `n - support - fp`.

**Gated against the published cells, scored from the stored defective model:**

| metric | measured | published |
|---|---|---|
| macro F1 | 0.960113 | .960 |
| accuracy | 0.956502 | 0.9565 |
| macro FPR | 1.8589e-04 | 1.859e-4 |

All 117,500 lines, 0 empty after preprocess, 0 gold labels outside the model's
set. **This is the instrument behind the published WiLI numbers**, so measurements
made with it can be trusted. Had it missed, nothing downstream would have been
usable and the plan stopped there.

**Artifacts:** `analysis/wili_eval.py`,
`outputs/rerelease/wili_instrument_gate.json`.

## The WiLI models carry the defect, and where their base vocabularies come from (2026-08-21)

**All three WiLI models carry 0.800000 special-token mass in every row**, so they
were trained with `sp`. The five WiLI tables need regenerating. This was settled by
measurement rather than by asking the co-author.

**Base vocabularies split two ways, and conflating them is the main hazard:**

| model | WiLI against GlotLID-C base vocabulary |
|---|---|
| `deepseek_v3.2` | **byte-identical**, sha256 `79b4c295...` |
| `qwen3_8b` | **byte-identical**, sha256 `311d4685...` |
| `wili_100k_500` | **different**: 24,357 of 100,000 tokens shared |

A model built on an LLM tokenizer carries that tokenizer unchanged across training
corpora. A model with no supplied tokenizer has its vocabulary trained on the
corpus, so it is corpus-specific. Author confirmation 2026-08-21: with no base
tokenizer provided, UniLID trains one on the entire training set with default
settings, which the code confirms (`train.py:465-492`, HuggingFace UnigramTrainer,
`--max-base-samples-per-lang` default 10,000, which never binds at 500 lines per
language).

**The base vocabulary is untouched by the special-token defect**, which lives in
the per-language path, so extracting a container's vocabulary is exact.

## Round-grid c sweep: c = -17, and the pre-registration hit exactly (2026-08-19, job 3117581)

**Pre-registered before the run** (`3a9c65c`): grid {-15,-17,-19,-21}, chosen by
the rule the published grid follows; predicted rows clamped 317 / 1,655 / 1,940 /
1,940; expected selection -17.

**Result: every prediction correct.** Clamp counts 317 / 1,655 / 1,940 / 1,940,
and the guard selected **c = -17**.

| config | rows clamped | val overall |
|---|---|---|
| baseline | 0 | 0.9453 |
| floor-15 | 317 | 0.9473 |
| **floor-17** | **1,655** | **0.9485** |
| floor-19 | 1,940 | 0.9483 |
| floor-21 | 1,940 | 0.9482 |

-17 beats -19 by 0.0002, the same tie the shifted grid showed. Test half:
overall 0.9460 to 0.9488 (+0.0019, CI [+0.0007, +0.0032]), tail -0.0623,
magnets -0.0218, accuracy 0.9604 to 0.9611.

**This is the constant of record for the re-release.** 1,655 of 1,940 rows are
clamped and 285 already lie at or below it. The shifted-grid sweep (job 3107082,
c = -17.3906) stays in the record as the like-for-like comparison against the
released model.

**Artifacts:** `outputs_corrected_round/tables/floor_equalization.md`.

## Floor-c full-pool pass at c = -17 (2026-08-19, job 3117583)

Ran automatically on a SLURM dependency, reading the constant from the sweep's
own output. 1,655 of 1,940 rows clamped, 285 left unchanged.

| stratum | corrected base | corrected + clamp | delta | released model, floor-21 (Exp 20) |
|---|---|---|---|---|
| overall | 0.9327 | **0.9419** | +0.0092 | 0.9292 to 0.9421 (+0.0129) |
| tail | 0.9045 | 0.8875 | -0.0171 | 0.9132 to 0.8928 (-0.0204) |
| magnets | 0.9067 | 0.8928 | -0.0139 | 0.9138 to 0.8974 (-0.0164) |
| twins | 0.9164 | 0.9166 | +0.0002 | -0.0001 |
| head | 0.9596 | 0.9595 | -0.0001 | -0.0003 |

**The clamped models land in the same place.** Corrected at c = -17 gives overall
0.9419 against the released model's 0.9421 at c = -21, a difference of 0.0002.
The clamp absorbs most of the correction's effect, which is what it should do:
both constants target the same place relative to each row's seen tokens. **The
tail cost is smaller than the released model's** (-0.0171 against -0.0204).

**Artifacts:** `outputs_corrected_round/tables/full_test_floor21.md`,
`full_test_eval_corrected/pred_floor21.npy`, `fingerprint_floor21.json`
(records `floor_target: -17.0`, `n_modified: 1655`).

## Three paper tables regenerated on the corrected model (2026-08-19)

**`tab:viterbi_vs_marginal`** (job 3110925, 4h36m, both decoders over the full
pool):

| decoding | accuracy | macro F1 | published |
|---|---|---|---|
| Viterbi | 0.961 | 0.933 | .961 / .929 |
| Marginalization | 0.961 | 0.935 | .962 / .931 |

Marginalization gains +0.0023 macro F1, so **the paper's claim that it "improves
macro F1 by 0.002" survives unchanged**. The accuracy cells now round to the same
value, so the bolding on the accuracy column has to go. Measured cost:
4h36m for both decoders against 1h42m for Viterbi alone, so marginalization is
about 1.7x Viterbi, against the caption's "approximately 2x".

**`tab:lenbias-norm`** (job 3117582):

| length | N | raw rescore | normalized | published normalized |
|---|---|---|---|---|
| <30 | 27,328 | 0.792 | 0.493 | 0.566 |
| 30-75 | 177,256 | 0.952 | 0.777 | 0.842 |
| 75-150 | 195,267 | 0.978 | 0.883 | 0.925 |
| 150-300 | 87,096 | 0.987 | 0.946 | 0.966 |
| 300+ | 13,053 | 0.995 | 0.987 | 0.991 |
| Overall | 500,000 | 0.961 | 0.838 | 0.885 |

**Length normalization is more damaging on the corrected model, not less**:
overall 0.961 to 0.838 (-0.123) against the published 0.960 to 0.885 (-0.075).
The paper's conclusion strengthens rather than changes. The Original column is
omitted because no recorded prediction column exists for this model; the
implementation check it supported can be partly recovered by comparing the
alpha = 0 rescore against `pred_baseline.npy` on the test half, which is not yet
done.

**Mistral-Nemo variant, corrected, GlotLID-C cells** (job 3117569, 1h47m):
macro F1 **0.9119**, macro FPR **1.858e-05**, accuracy 0.9640, against the
published .912 / 1.84e-5. **The row does not move to three decimals.** The
correction shifts the base model by +0.0035 macro F1 and this variant by
essentially zero, which is worth stating rather than assuming the effect is
uniform across models.

## Drive folder `glotlid_unilid`: the DeepSeek3.2 and Qwen3 models are there (2026-08-18)

**Correction of my own earlier entry.** I first recorded that this folder held no
model weights, based on the Google Drive MCP connector, which returned only a
`full_prob` subfolder with two JSON files under every query I tried
(`parentId =` the folder, with and without a mimeType filter; `title contains
'glotlid'`; `owner =` the co-author; `mimeType = 'application/octet-stream'`).
**That conclusion was wrong.** `gdown` run from the login node lists the folder
correctly, so the connector's search index does not cover its contents. The
lesson for later: a negative result from that connector is not evidence of
absence, and a direct fetch is the check.

**Actual contents**, from `gdown --folder`:

| file | note |
|---|---|
| `deepseek_v3.2_glotlid.unilid` | 1,004,157,190 B. **The DeepSeek3.2 model.** |
| `qwen3_8b_glotlid.unilid` | 1,181,995,016 B. **The Qwen3 model.** |
| `deepseek_v3.2_glotlid_y_pred.txt` | its recorded predictions |
| `qwen3_8b_glotlid_y_pred.txt` | its recorded predictions |
| `glotlidc_y_pred.txt` | base model predictions (also inside `full_prob/`) |
| `glotlid_e100_sanity/` | fastText e100 metrics, per-language, y_pred |
| `full_prob/` | `glotlidc_metrics.json`, `glotlidc_per_language.json`, `glotlidc_y_pred.txt` (411 MB) |
| `glotlid_correct_test.txt.zip`, `train.txt.zip`, `glotlid_train_counts.json` | corpora and counts already held here |

Both models downloaded to
`/capstor/scratch/cscs/cmeister747/unilid_analysis/drive_models/`.

**Both carry the same special-token defect, so both take the same correction.**
Measured on `deepseek_v3.2_glotlid.unilid`: 1,940 languages, vocabulary 128,819,
special-token mass **0.800000 in every row** (min equals max), real-token mass
0.200000 in every row, unseen-token plateau -23.354 to -13.464.

**A structural difference worth carrying forward: its special columns are
128815-128818, at the END of the vocabulary, not 0-3.** The base GlotLID-C model
has them at the start and the Mistral-Nemo variant at 0, 1, 2 and 10. Any code
that assumes columns 0:3 is wrong for two of the four models. Two scripts still
do (`full_test_bgfloor.py:207`, `mixed_matrix.py:204`); neither is on the
re-release path, and the rest of the chain now finds them by name from the
vocabulary.

**Unblocks** the C2 group of `paper/PAPER_EDITS_pending.md`: the 24 Table 1 cells
no longer have to stay on pre-correction weights, and `:975`'s "all within 0.025
macro F1" no longer straddles two model generations.

**Still open:** the WiLI and DSL-ML artifacts (group C1) are not in this folder.

**One discrepancy to raise with the co-author, unchanged.** `full_prob`'s metrics
file reports 1,075.0 samples/s while `tab:latency_glotlid` reports 3,253
samples/s for UniLID over the same 45,627,279 samples, a factor of 3.0. The
folder name suggests a run emitting full probability distributions rather than
the argmax, which would account for it, but the latency table's run
configuration remains an open item and this does not close it.

**One cross-check that passes.** `full_prob/glotlidc_metrics.json` gives macro F1
0.9311 over all 45,627,279 lines. The paper's `tab:lid_main` UniLID row prints
`.929`, the scored-pool figure (45,377,279 lines, macro F1 0.9292, measured
here). The two differ exactly as the table caption states, corroborating that
provenance claim from the co-author's own artifact.

## Pre-registration: round-grid c sweep for the corrected model (2026-08-18, recorded before the run)

**Why a second sweep, stated plainly.** The first sweep (job 3107082) used the
published grid shifted by log 5, `{-15.3906, -17.3906, -19.3906, -21.3906}`,
because that is the like-for-like comparison against the released model, and it
selected -17.3906. **That result is already in hand, so this second sweep is run
after seeing an outcome, and it has to be justified as something other than grid
shopping.**

The justification is that the grid is fixed by a rule applied to this model, not
chosen to move the answer. The published grid `{-17,-19,-21,-23}` stands in a
specific relation to the released model's unseen-token plateau range (-19.939 to
-13.216): two values inside the range, two below all of it. The corrected model's
range is -18.329 to -11.606, and `{-15,-17,-19,-21}` stands in exactly the same
relation to it: -15 and -17 inside, -19 and -21 below all of it. The shifted grid
has that structure too, so both grids are defensible; the round one is chosen
because the paper has no released predecessor for a reader to relate the shifted
values to.

**Fixed before the run:**

- Grid: {-15, -17, -19, -21}.
- Selection procedure: unchanged. All-strata guard on the validation half of the
  seed-42 500k draw, test half scored once.
- Predicted rows clamped, computed from the row minima before scoring:
  **317 / 1,655 / 1,940 / 1,940**. (Released, for reference: 452 / 1,821 / 1,940
  / 1,940.)
- Expected selection: **-17**, since the shifted grid selected -17.3906 and -17
  lies 0.39 nats from it. **If the guard selects something else, that is the
  result and it is recorded as such.**
- Both sweeps' full validation tables are reported. Neither is discarded.
- This sweep, if it completes, supplies the constant of record for the paper. The
  shifted-grid sweep remains the record of the like-for-like comparison against
  the released model and the evidence that the correction is a uniform shift.

**Artifacts to be written:** `outputs_corrected_round/tables/floor_equalization.md`.

## Corrected model, full-pool baseline: overall +0.0035, tail -0.0087 (2026-08-18, job 3107045)

**Hypothesis under test:** the effect of the special-token correction, measured on
the full evaluation pool rather than the 250,000-line golden subset.

**Finding: not a wash at full-pool scale, and the strata move in opposite
directions.** Base mode, 45,377,279 lines, `--configs baseline` only.

| stratum | released | corrected | delta |
|---|---|---|---|
| overall macro F1 | 0.9292 | 0.9327 | **+0.0035** |
| overall accuracy | 0.9608 | 0.9609 | +0.0001 |
| tail (7,735 examples, 96 languages) | 0.9132 | 0.9045 | **-0.0087** |
| magnets (64,657 examples, 118 languages) | 0.9138 | 0.9067 | **-0.0071** |
| twins (9,156,023 examples) | 0.9167 | 0.9164 | -0.0003 |
| head (43,665,835 examples) | 0.9602 | 0.9596 | -0.0006 |

**This qualifies the earlier "essentially a wash".** That statement was measured
on the golden subset (macro F1 0.9454 to 0.9460) and remains true there. On the
full pool the overall figure moves by +0.0035 and two strata move by roughly
-0.008. Both statements are correct about their own instrument; the golden-subset
one must not be quoted as though it covered the full pool.

The direction repeats the pattern Exp 20 recorded for the floor-21 clamp: an
overall gain that is a global-precision effect sitting alongside a recall-side
loss on the tail and magnet strata. The stratum rows are the within-stratum
recall view and exclude false positives into tail labels, so a tail figure falling
means examples truly written in tail languages are misclassified more often.

**This is not an adoption decision.** The released weights carry a training
defect and the corrected ones do not; the re-release is not a choice between two
candidate methods on their metrics. The tail figure is recorded because the paper
quotes per-stratum numbers, not because it bears on whether to correct.

**Paper consequence:** `submission.tex:344` quotes macro F1 .929, which becomes
.933.

**Artifacts:** `outputs_corrected/tables/full_test_eval.md`,
`outputs_corrected/diagnostic/full_test_per_lang_f1.csv`, scratch root
`full_test_eval_corrected/`.

## The unseen-token constant did not move; a 0.0001 tie flipped (2026-08-18, job 3107082)

**Hypothesis under test:** where the published Exp 20 selection procedure lands
for the corrected model. Pre-registered expectation, recorded in
`slurm_floor_sweep_corrected.sh` before the run: near -19.3906 = -21 + log 5,
which would reproduce the released clamped matrix up to the uniform shift. **A
result far from that was to be recorded as a finding, not treated as a tuning
problem.**

**The procedure selected -17.3906**, that is -19 + log 5, one grid step above the
expectation. Aligning the two sweeps by grid position (released F against
corrected F + log 5) shows why, and shows that the constant itself did not move:

| position | released F | corrected F | rows clamped | val overall released | val overall corrected |
|---|---|---|---|---|---|
| baseline | | | 0 | 0.9451 | 0.9453 |
| 1 | -17 | -15.3906 | 452 / 452 | 0.9475 | 0.9474 |
| 2 | -19 | -17.3906 | 1,821 / 1,821 | 0.9488 | **0.9486** |
| 3 | -21 | -19.3906 | 1,940 / 1,940 | **0.9489** | 0.9484 |
| 4 | -23 | -21.3906 | 1,940 / 1,940 | 0.9486 | 0.9481 |

**The number of rows clamped is identical at every position**, and every grid step
is exactly log 5, which is the check that the shifted grid asks the same question
of both models. Validation macro F1 differs by at most 0.0005 at any position, so
the correction behaves as the uniform shift it is claimed to be.

**Positions 2 and 3 are tied in both models.** The released model selects position
3 over position 2 **by 0.0001**; the corrected model selects position 2 over
position 3 **by 0.0002**. So the published c = -21 was never resolved against
-19: it won by one ten-thousandth of a macro F1 point, and under the correction
the same procedure picks the other member of the tie. The constant did not shift
by more than log 5; a tie broke the other way.

All four grid values passed the all-strata guard in both models, so the selection
was made on validation overall macro F1 alone.

**Consequence:** the choice of c for the re-release is not determined by the data,
and it propagates. The thresholds tau are percentiles of margins measured on the
clamped matrix, so a different c gives 1,084 different thresholds.

**Artifacts:** `outputs_corrected/tables/floor_equalization.md`,
`analysis/c_selection_comparison.py`,
`outputs/rerelease/c_selection_comparison.json`.

## Special-token defect: per-language training gave four unusable tokens 0.8 of every row's mass (2026-08-17, login node, no SLURM job)

**Hypothesis under test:** a setup report of large score differences between
`--method sp` and `--method em` in `add_language` was assumed to be a
configuration error on the reporter's side.

**Finding, and it is a defect in this project's own trained models.**
`unilid/trainers/language_specific_trainer.py` gave each special token the base
tokenizer's stored score. HuggingFace's Unigram stores specials with score `0.0`,
read there as a log-probability, that is probability 1.0. Four of them dominate
the normalization, each lands at exactly 1/5, and every real token is depressed by
log 5 = 1.6094 nats. All four stored GlotLID-scale models carry exactly 0.800000
special-token mass in every one of their 1,940 rows, independent of the language's
corpus size.

The mass is unusable, not merely inert: no special token's stored weight is ever
read when scoring. Confirmed by perturbation rather than by reading the Rust, so
the conclusion does not depend on my reading of `model.rs`: setting all four
entries of every row to -500 changes predicted scores by exactly 0.000000.

**Reproduction of the reported symptom.** On a toy base model whose rows come from
the pure-Python soft EM (specials at the 1e-12 floor), adding a real-text language
with the default `sp` mixes two scales. Held-out accuracy 0.24 (sp) against 0.90
(em) on identical data. Causal test: repairing only the three unemittable specials
in the sp row and renormalizing, changing nothing else, moves 0.24 to 0.74. So the
user-visible failure is scale mixing, not a broken estimator.

**Artifacts:** `EXPERIMENTAL_SETUP.md` "The special-token defect and the corrected
artifact"; package fix in version 0.3.0 (PR #3, commits 9f7c1cf, 56e7fd4, 2d5f62d).

## The correction changes predictions because it is not a constant offset (2026-08-17, login node, no SLURM job)

**Hypothesis under test:** the recorded claim that the special-token mass is
"uniform across languages so argmax-neutral" (`EXPERIMENTAL_SETUP.md:217`,
`EXPERIMENTS_CHRONOLOGICAL.md:921`, `EXPERIMENTS_PLAN.md:950`).

**Finding: false, for two compounding reasons.** A language scores under its own
Viterbi segmentation, so a language segmenting a text into n_L tokens takes the
1.6094-nat depression n_L times, and n_L differs across candidates for the same
text. The correction also moves the segmentation itself, because the max-plus DP
maximizes `sum(log p_i) + n * log 5` and a positive per-token constant favors more
tokens. Measured on 3,000 pool lines: 1,140 of 3,000 re-segment, all 1,140 toward
more tokens, mean token count 39.369 to 39.920, predictions change on 14 of 3,000.
On the 1,860 lines where prediction and segmentation are both unchanged the score
delta equals `n * log 5` to within 5.5e-4, which is the check that the
transformation is what it claims to be.

**Artifacts:** `analysis/segmentation_shift.py`,
`outputs/rerelease/segmentation_shift.json`.

## The corrected artifact is a transformation of the released weights, and it gates 8/8 (2026-08-17, login node, no SLURM job)

**Hypothesis under test:** whether correcting the released model requires
retraining all 1,940 languages or can be done in closed form.

**Finding: closed form.** Renormalizing each row over its real tokens is exactly
`+ log 5` on every real token, with the specials parked at the training floor.
Corroborated by retraining `aai_Latn` (24,580 lines) under the fixed code and
comparing against (released row + log 5) over 99,996 real tokens: correlation
1.00000000, median absolute difference 1.7e-5, 99.69% within 1e-4.

**Gate:** eight languages spanning N_L 85 to 100,000, retrained from their own
corpus and compared against the transformed row. 8/8 pass. The criterion bounds
the signed mean difference (max 0.01; a wrong constant would show about 1.6), the
mass-weighted difference (max 0.02) and the correlation (min 0.9999). It does not
require exact row reproduction: two retrains of `zul_Latn` proved bit-identical to
each other but not to the released row, because languages above the 100,000-line
cap were subsampled and the corpus on store is the Apertus draw. **The threshold
was chosen after seeing that failure**, which is recorded explicitly because
changing a criterion after seeing a failure is the move that most needs to be
visible.

**Artifacts:** `analysis/correct_special_token_mass.py`,
`analysis/gate_correction.py`, `outputs/rerelease/gate_correction.json`, four
corrected files on scratch under `corrected/`.

## Effect of the correction on the released model: a wash on metrics (2026-08-17, login node, no SLURM job)

**Hypothesis under test:** whether correcting the released weights improves
accuracy, which would make the re-release a metrics story rather than a
correctness one.

**Finding: it does not.** On the 250,000-line golden subset (the test half of the
seed-42 500k draw) against the recorded gold labels, base mode: macro F1 0.9454 to
0.9460, macro FPR 2.083e-05 to 2.081e-05, accuracy 0.9603 to 0.9604. 1,807 of
250,000 predictions change (0.72%), 699 fixed and 669 broken. **The case for
re-releasing is correctness, not metrics.**

This supersedes an earlier estimate of 0.9494 to 0.9509 with 63 fixed and 32
broken, which was accuracy on a 20,000-line every-149th-line sample rather than
macro F1 on the golden subset, and was not adequately powered.

**Artifacts:** `analysis/correction_effect.py`,
`outputs/rerelease/correction_effect.json`.

## The 0.3.0 fix silently disabled the unseen-token constant (regression found and fixed, 2026-08-17)

**Hypothesis under test:** how far the unseen-token constant c moves under the
correction. The probe answered a different question first.

**Finding: a regression I had shipped.** The probe returned `modified 0` of 1,940
rows at every c for the corrected model, against 1,940 for the released one.
Parking the specials at `MIN_TOKEN_LOG_PROB` makes them each row's minimum, and
`apply_unseen_token_constant` defines a row's unseen tokens as its exact
minimum-value plateau, so the plateau of unseen real tokens is never located and
the clamp does nothing. Every model trained by 0.3.0 as first shipped had the
calibration's first correction disabled with no message. It also re-explains the
"row minimum -27.631 is at or below c=-21.0; the row is left unchanged" lines in
the post-fix `add_language` runs, which I had read at the time as the training
floor acting on real tokens.

**Fix:** the clamp takes the special columns and excludes them from the minimum
(`unilid/calibration.py`, `analysis/floor_equalization.py`), with both callers
finding those columns by name from the vocabulary. Pre-0.3.0 files are unaffected,
their specials sitting at -1.6094 and never being the minimum, which is asserted
in a test alongside one for the broken case. Both release gates re-run at
250,000/250,000 because this is an inference-path change.

**Artifacts:** package commit 2d5f62d; `EXPERIMENTAL_SETUP.md` "The 0.3.0 clamp
regression".

## Probe: c is carryable by addition, the thresholds are not (2026-08-17, login node, no SLURM job)

**Hypothesis under test:** whether the calibration's fixed numbers can be carried
to the corrected model by adding log 5, or must be re-derived. Both probes select
on the validation half of the seed-42 draw and never touch the golden subset.

**c: consistent with carrying.** 60,000 validation lines, nine-value sweep, macro
F1. Released optimum at c = -19.5 (0.95686) against -21 (0.95671); corrected
optimum at c = -17.5 (0.95726). The shift is +2.0 against log 5 = 1.609, and the
optimum is flat enough on this subsample that the two are not distinguishable
here. Note the published c = -21 already sits just off the optimum on the released
model.

**tau: not carryable, all 1,084 must be re-estimated.** Six group-A languages,
released clamped at c = -21 and corrected at -21 + log 5 = -19.3906, which is the
like-for-like comparison. Of the six, two are excluded in both models
(`kdr_Latn`, `chq_Latn`, both `low_calibration`). The four with thresholds move
`tul_Latn` 6.7418 to 5.8984 (-12.5%), `bkv_Latn` 12.0420 to 11.3768 (-5.5%),
`mpm_Latn` 0.15009 to 0.04226 (-71.8%), `cmo_Latn` 0.011712 to 0.026153
(+123.3%); mean delta -0.40 nats. **The moves go in both directions and differ by
two orders of magnitude in relative size, so no shift or scaling carries them.**
The own-won counts move too (for example `mpm_Latn` 1,485 to 1,383), which is the
mechanism: the margin is a difference between two languages that segment the line
into different numbers of tokens, so log 5 does not cancel in it.

**Artifacts:** `analysis/probe_calibration_shift.py`, `analysis/probe_tau_shift.py`,
`outputs/rerelease/probe_c.json`, `outputs/rerelease/probe_tau.json`.

## The unseen-token plateau is set by corpus size, not by the training floor (2026-08-17)

**Hypothesis under test:** the paper's stated cause for each row's unseen-token
plateau sitting above c, namely the training-time probability floor.

**Finding: the stated cause is wrong, and the floor is never reached.** Every
observed plateau (-19.94 to -13.22) sits 7.6 to 14.4 nats above the floor's
-27.631. What sets the value is the per-language fit, near-deterministic in corpus
size: `corr(plateau, log10 N_L) = -0.9659` over all 1,940 rows, median plateau
-13.914 below 1,000 training lines rising to -18.766 above 50,000. Against
Exp 27's Viterbi token counts, `corr = -0.9924`,
`plateau = -5.539 - 2.039 * log10(T)`, R-squared 0.985. This reproduces the
project's own Exp 10 figure of -0.966.

**Ruled out:** the missing-token assembly fill (the base tokenizer's scores are
99,997/100,000 distinct and cannot produce a 92,407-entry block of identical
values), a fixed SentencePiece constant (the value varies by 6.7 nats across
languages), and the 1e-12 floor. The special-token defect is real but uniform at
1.609 nats and explains none of the spread: removing it moves the median plateau
only from -17.66 to -16.05.

**Artifacts:** `analysis/plateau_reference_fit.py`,
`outputs/rerelease/plateau_reference_fit.json`, derived from the committed
`outputs/diagnostic/gt_counts.csv`.

## B0: corpus size alone sets the plateau, with language identity held fixed (2026-08-17, login node, no SLURM job)

**Hypothesis under test:** the cross-language relation above is confounded, since
each of its 1,940 points is a different language, so corpus size and language
identity vary together. Does corpus size account for it on its own?

**Design.** One language's corpus is shuffled once (seed 20260817) and nested
prefixes of 1,000 / 3,000 / 10,000 / 30,000 / 100,000 lines are retrained against
the same unmodified base tokenizer, so the only quantity that changes between runs
is how much text the estimator saw. Nested prefixes rather than independent draws,
so the smaller corpora are subsets of the larger ones. Three languages chosen
deterministically from the 282 that reach the 100,000-line cap: `abk_Cyrl`,
`mam_Latn`, `zul_Latn`. Pre-registered pass criterion: the within-language slope
lands within 50% of the cross-language slope and `real_missing` stays near zero.

**Finding: PASS, 3/3, and far more tightly than the criterion required.**

| language | slope, nats per decade of tokens | R-squared |
|---|---|---|
| `abk_Cyrl` | -2.196 | 0.999 |
| `mam_Latn` | -2.196 | 0.999 |
| `zul_Latn` | -2.184 | 0.999 |
| across 1,940 languages | -2.039 | 0.985 |

Three languages across two scripts agree on the slope to three significant
figures, at R-squared 0.999. On a common scale (retrained rows are normalized over
real tokens and sit log 5 above released rows) the within-language fit is
`plateau = -4.628 - 2.192 * log10(T)` against the cross-language
`-5.539 - 2.039 * log10(T)`; the two agree to 0.006 nats at log10 T = 6, near the
median, and to within 0.30 nats across the whole observed range log10 T = 4 to 7.

**The mechanism as a scaling law.** -2.192 nats per decade is an exponent of -0.95
in natural log, so the plateau probability scales as `T^-0.95`, approximately one
count in T. That is what an EM fit assigns a token type it never effectively
observed: a property of the estimator, not of any floor or fallback.

**The one mechanism that could have faked this is ruled out.** `real_missing`, the
count of base-vocabulary tokens absent from the SentencePiece model and filled
from the base tokenizer's score, is **0 in all 15 runs**. The plateau block also
shrinks as the corpus grows (`zul_Latn`: 92,261 entries at 1,000 lines to 82,264
at 100,000), consistent with more token types being observed rather than with a
constant being written in.

**Consequence for the paper.** `submission.tex:629-631` currently attributes the
above-c values to "the training-time probability floor of 10^{-12} and
renormalization". That can now be replaced with a measured causal statement rather
than another guess.

**Artifacts:** `analysis/plateau_vs_corpus_size.py`,
`outputs/rerelease/plateau_vs_corpus_size.json`.

## Camera-ready E1: common reporting set (2026-08-07, login node, no SLURM job)

**Question:** on one reporting instrument per role (amendment 9), what are the
camera-ready numbers for baseline UniLID, the promoted configuration
gate_flat4_prox21, and fastText, with held-out uncertainty for the contrasts?
Pre-registered in `EXPERIMENTS_PLAN.md` "Camera-ready evaluation program"; all
wiring gates passed (carried CSV at 1e-9; judge-part per-language F1 vs
`mixed_eval_judge_f1_gate_flat4_prox21.csv` at 1e-9 with pre-registered anchors
0.9117/0.9498; full-pool baseline reproduced 0.929190 / 2.0263162e-5).

| config | full pool F1 | full pool FPR (x1e5) | judge F1 | judge FPR (x1e5) |
|---|---|---|---|---|
| baseline | 0.9292 | 2.0263 | 0.9117 | 2.0373 |
| gate_flat4_prox21 | 0.9569 | 1.7665 | 0.9498 | 1.7743 |
| fasttext | 0.9443 | 2.7063 | 0.9332 | 2.7165 |

Full pool = the 45,377,279 kept lines (the Table 1 instrument; the submission's
UniLID cell .929 / 2.03e-5 and fastText cell .944 / 2.71e-5 both reproduce).
Judge part = the 27,002,441 held-out lines. Paired bootstrap on the judge part
(B=10,000, seed 0, percentile 95%, over the 1,940 languages):
gate_flat4_prox21 minus baseline +0.0380 [+0.0328, +0.0434]; gate_flat4_prox21
minus fasttext +0.0166 [+0.0112, +0.0223]. Both intervals are entirely above
zero, so on both reporting instruments the promoted configuration's macro F1 and
macro FPR are ahead of both comparators.

**fastText import verification:** the recovered Drive file passed the blocking
gate (100% agreement with `sample_500k_all.pkl` on the 500,000 re-derived
seed-42 lines) and the comparability measurement (kept-pool macro F1 0.944339
vs the paper's .944, difference 3.4e-4; the paper's value was computed on all
45,627,279 lines per the paper team's metrics JSON, ours on the kept pool).

**Finding on the recovered UniLID prediction file:** the Drive folder-1
`glotlidc_y_pred.txt` FAILED its blocking gate against the pickle (2,430 of
500,000 sampled lines disagree, 99.514% agreement, disagreements concentrated
on near-tie pairs), and the follow-up measurement classified it: the file
agrees with our `pred_baseline.npy` at exactly 1.000000 on the kept pool
(macro F1 0.929190, identical). So the Drive file matches our current scoring
run, and the pickle's `pred_UniLID` is the paper-era scoring run (the recorded
0.9951 self-agreement lineage, Exp 16). Consequence: `pred_glotlidc_file.npy`
adds no information beyond `pred_baseline.npy` and is not used anywhere; Table
1 UniLID numbers come from `pred_baseline.npy` as planned.

**Artifacts:** `outputs/tables/paper_eval.md` (+ `paper_eval_table1_row.tex`,
`paper_eval_appendix.tex`), `outputs/diagnostic/paper_eval_per_lang_f1_{fullpool,judge}.csv`,
`outputs/tables/import_fasttext.md`. Scripts `analysis/import_external_pred.py`,
`analysis/paper_eval.py` (Sonnet implementation, Opus pre-run review with 11
findings applied, run at commit 02a346e).

---

## Camera-ready E3: the calibration mechanisms transfer to the Mistral-Nemo variant (2026-08-09, jobs 3028465/3032625/3037165/3038358 + login-node stages)

**Question:** do the promoted configuration's mechanisms generalize to a
different base vocabulary? The variant was retrained from scratch with the
recorded pipeline (pinned Mistral-Nemo-Base-2407 tokenizer, 131,072 vocab,
fp64 trainer), and every calibration component was re-derived for it by the
recorded rules: its own flat set (bjn_Latn, sco_Latn, srp_Latn), its own tau
CSVs, its own floor-21 matrix (two rows with natural floors below -21
correctly left unclamped).

| config | full pool F1 | full pool FPR (x1e5) | judge F1 | judge FPR (x1e5) |
|---|---|---|---|---|
| nemo_baseline | 0.9132 | 1.7927 | 0.8968 | 1.7993 |
| nemo_floor21 | 0.9396 | 1.7139 | 0.9278 | 1.7199 |
| nemo_gated | 0.9538 | 1.5588 | 0.9473 | 1.5627 |

Paired bootstrap (judge part, B=10,000, seed 0): nemo_gated minus
nemo_baseline +0.0504 [+0.0438, +0.0573]. Comparability (recorded
measurement, not a gate): our retrain's full-pool baseline 0.9132 sits
+0.0012 from the paper's printed UniLID-Mistral-Nemo cell (.912, computed on
all 45,627,279 lines vs our 45,377,279 kept lines), so the retrain
reproduces the published variant closely.

**Verdict: the mechanisms transfer, with a larger gain than on the base
model** (+0.0406 full pool vs +0.0277): the calibrated variant (0.9538)
exceeds fastText (0.9443) and approaches the calibrated dedicated-tokenizer
configuration (0.9569). All wiring/sentinel/identity gates passed; the 32
degeneracy-flagged minority-script rows are carried as a listed caveat with
per-language values in the CSVs. Three startup failures during the chain
were each a base-model-invariant gate misfiring on a genuine variant
property, diagnosed before any gate was relaxed (chronological log,
2026-08-08/09).

**Artifacts:** `outputs/tables/mistralnemo_eval.md` (+ `.tex`),
`outputs/diagnostic/mistralnemo_per_lang_f1_{fullpool,judge}.csv`,
`outputs/diagnostic/{mistralnemo_flat_set,tau_mistralnemo_floor21_gate,tau_mistralnemo_flat}.csv`,
model + memmaps + fingerprints in scratch `full_test_eval_mistralnemo/` and
`glotlid_mistralnemo_fp64.unilid` (store migration pending). Pipeline
`analysis/mistralnemo_eval.py` at commit d034cce.

---

## Camera-ready E5: CommonLID out-of-domain evaluation of the calibrated configuration (2026-08-07, job 3031609 + login-node eval)

**Question (user request):** CommonLID performance for the submission. All
wiring gates passed: the score stage's baseline predictions are EXACTLY equal
to Exp 39's persisted arrays (agreement 1.000000, both arms), and the eval
stage reproduced the recorded baseline 0.8452 accuracy / 0.7228 tag-level
macro F1 and floor-21 0.8491 / 0.7181 through the new code path before any
new number.

| config | macro-aware accuracy | tag-level macro F1 |
|---|---|---|
| baseline | 0.8452 | 0.7228 |
| floor-21 only | 0.8491 | 0.7181 |
| gated (promoted mechanisms) | 0.8604 | 0.7149 |

**Verdict:** on web-domain out-of-domain data the promoted configuration
raises macrolanguage-aware accuracy by +0.0152 (four times floor-21's
+0.0040) while tag-level macro F1 falls 0.0079. Mechanism, measured:
predictions carrying a tag-space label outside the 109-tag set fall from
32,901 rows (1,089 distinct labels) to 25,884 (782); the re-examination
returns lines from under-resourced languages (mostly absent from the tag
set) to tag-set languages, which the accuracy counts and the tag-level view
partly does not. Re-examination accounting: group A 9,086 examined / 7,844
moved; group B 3,971 / 3,844. This extends the Exp 39 scoping: floor-21
solo was slightly negative on the tag-level view, and the full promoted
configuration is more positive on accuracy and comparably slightly negative
on the tag-level view.

**Paper integration:** `paper/tables/commonlid.tex` (tab:commonlid) in the
appendix protocol section, with one sentence in the Results calibration
paragraph; the caption states both conventions and the mechanism.

**Artifacts:** `outputs/tables/commonlid_calibrated.md`,
`outputs/diagnostic/commonlid_calibrated_per_tag.csv`, banked npz + meta in
scratch `commonlid/`. Script `analysis/commonlid_calibrated.py` (Sonnet
implementation, Opus review with 10 findings applied, run at commit f5c271a).

---

## Camera-ready E2: UDHR and FLORES-200 transfer test (2026-08-07, job 3028291 + login-node eval)

**Question:** do the promoted configuration's mechanisms, transferred with
every constant unchanged, help or hurt on the paper's two external
benchmarks? Acceptance gates first: our rebuilt eval sets reproduce the
paper's baseline cells (UDHR macro F1 0.858977 vs printed .859, diff 2.3e-5;
FLORES 0.931741 vs .932, diff 2.6e-4; baseline macro FPR reproduces both
printed cells exactly: 1.43e-4 and 2.78e-4), and the paper team's label lists
match our reconstructions exactly, so the instrument is the paper's own.

| config | UDHR F1 | UDHR FPR (x1e5) | FLORES F1 | FLORES FPR (x1e5) |
|---|---|---|---|---|
| baseline | 0.8590 | 14.29 | 0.9317 | 27.75 |
| floor-21 only | 0.8474 | 17.52 | 0.9323 | 28.52 |
| gated (promoted mechanisms) | 0.8383 | 20.76 | 0.9326 | 29.07 |

**Verdict: the pre-registered balanced-set reversal, measured.** FLORES-200
(natural-ish sentence sets, 190 mostly mid-to-high-resource labels) improves
slightly (+0.0009 gated vs baseline); UDHR (parallel, 366 labels with small
equal supports) regresses (-0.0207). On balanced parallel data the
false-positive volume the mechanisms remove is largely absent, so the
own-recall cost dominates; this is the same mechanism as the draw-201
confirmation pattern and is stated in the paper's subsection and Table 1
caption. Re-examination accounting: UDHR group A 1,117 examined / 425 moved,
group B 10/10; FLORES group A 1,313 / 636, group B 171/163.

**CLD3-subset cells: the consolidated discrepancy record (updated
2026-08-09, reported to the user).** Everything on the left side of Table 1
and in the appendix breakdowns reproduces from our pipeline exactly or to
printed precision, and the paper team's per-language fastText JSON agrees
with our imported fastText predictions to fifteen decimal places. Every
unresolved discrepancy sits in the right-side (CLD3-subset) columns:

| cell | printed | our reproduction | status |
|---|---|---|---|
| UniLID GlotLID-C subset F1 | .971 | 0.9719 | reproduces |
| UniLID UDHR subset F1 | .992 | 0.9873 | borderline (0.0047 inside the 0.005 gate) |
| UniLID FLORES subset F1 | .997 | 0.9907 | does not reproduce |
| fastText GlotLID-C subset F1 | .990 | 0.9767 (restricted) / 0.9719 (global = his own JSON) | does not reproduce |
| UniLID GlotLID-C subset FPR | 1.63e-4 | 9.71e-5 (restricted) / 7.77e-5 (global) | no tested convention |

Interpretation of record: since the paper team's own per-language data
agrees with ours, the printed subset cells came from a different computation
than the full-set cells, most plausibly a closed-set evaluation (label space
restricted to the subset) applied to some systems; resolving which requires
their eval script.

Addendum (2026-08-09, after the paper team's four eval scripts arrived in
`unilid_resources/`): the scripts confirm the full-set conventions (their
union-of-gold-and-predicted macro denominator coincides with the gold set on
these benchmarks) but contain NO subset logic, so the right-side columns
were produced by a procedure not in the drop. Their `--lang-only` mode
(bare-code comparison) was tested as a hypothesis and refuted by
measurement: with row restriction and union averaging it yields macro F1
0.0468 (UniLID) / 0.0592 (fastText), nowhere near the printed cells. The
closed-set-for-fastText interpretation is now the only one consistent with
all measurements: open-set restricted-lines reproduces the UniLID cell
exactly (0.9719 -> .971) and a closed-set run can only exceed it, so the
UniLID cell is open-set, while fastText's .990 exceeds every open-set value
we can construct (0.9767 restricted, 0.9719 global) and matches the
closed-set effect of removing out-of-subset errors. Remaining ask: the
specific script or command behind the CLD3-subset columns. Consequences implemented (user decisions 2026-08-09): the
calibrated row's subset F1 cells are filled under the restricted-lines
convention that reproduces the UniLID GlotLID-C cell, with the FLORES
mismatch stated in the Table 1 caption; the subset FPR cells stay dashed;
the printed fastText cells stay untouched. Calibrated subset values:
GlotLID-C 0.9751, UDHR 0.9856, FLORES 0.9920.

**Artifacts:** `outputs/tables/external_bench_{udhr,flores}.md`,
`outputs/diagnostic/external_bench/{udhr,flores}_per_label.csv`,
`outputs/tables/paper_eval_cld3_subset.md`,
`outputs/tables/paper_eval_cld3_subset_external.md`, banked npz + meta in
`external_bench/` (scratch). Gate machinery verified by the bit-exact
self-check before use.

---

## Camera-ready E4: breakdowns in both metric views; residual re-measured on the promoted configuration (2026-08-07, login node, no job)

**Provenance result first (chronological log, same date):** both of the paper's
appendix breakdown tables are the within-stratum view; under it every printed
cell reproduces from `pred_baseline.npy`, including the previously unexplained
Cyrl 0.8774 -> .877 and Hebr 0.7401 -> .740, and the script table's "Other 82"
basis is our 84 minus jpn_Jpan and kor_Hang (0.9374 -> .937). Both E4
reproduction gates, re-pointed at the within-stratum references, PASSED.

**Both-views resource table (full kept pool; global = all false positives
counted; within-stratum = cross-group false positives excluded):** the two
views rank the methods oppositely in the small bins. Under-500 bin: global
baseline 0.5145 -> promoted 0.7796 (fastText 0.7497); within-stratum baseline
0.8709 -> promoted 0.8272 (fastText 0.9150). The promoted configuration's
gains are a global-view fact (false positives into small languages fall); its
within-stratum values in the small bins decrease. Full tables:
`outputs/tables/paper_breakdowns.md` (+ `_script.tex`, `_resource.tex`, both
views, view stated in every caption).

**Residual re-measurement (judge part, 27,002,441 lines), replacing the
floor21_gate-era numbers for paper item 5:** gate_flat4_prox21 has 926,299
wrong predictions (floor21_gate: 962,633, recomputed exactly); 99.15% have a
true language with at least 18,000 training lines; 88.64% of those are
confused with another such language. Top pair ind_Latn -> zsm_Latn (31,113
lines); the floor21_gate-era eng_Latn -> sco_Latn pair (29,779 lines) is
absent from the promoted configuration's top-20, the visible effect of the
flat-language re-examination. Note: the recomputed floor21_gate head-head
share is 0.8929 against the recorded 88.2% from the 2026-07-30 mining session
(+0.0109); the recomputation with full provenance is the number of record.
Artifacts: `outputs/tables/promoted_residual.md`,
`outputs/diagnostic/promoted_residual_pairs.csv`.

**E2 gate machinery verified before any external scoring:** the new
`external_bench_eval.py` self-check replayed the 2,236,864 banked re-examination
rows through its own masking/walk/merge code and reproduced
`pred_gate_flat4_prox21.npy` on all 45,627,279 lines exactly (0 differ).
Scripts Sonnet-implemented, Opus-reviewed (16 findings, all applied), run at
commit 9b1ed20.

---

## Current state (2026-08-06)

Read this block first; it supersedes the 2026-07-29 block below, which is
retained for the decision trail and marked superseded there. Terminology
used throughout (repeated from the superseded block): a **row** is one
language's vector of natural-log token probabilities in the 1,940 x 100,000
weight matrix; the **primary quantity** is per-language F1 on
natural-distribution test data with all false positives counted, averaged
unweighted over the 1,940 languages.

**Promoted configuration: gate_flat4_prox21 (user decision 2026-08-06).**
gate_flat4_prox21 is promoted on the natural track, superseding floor21_gate,
which remains in the pool. It has three mechanisms, computed on a line's
prediction in this order.
1. Floor-21 unseen-token clamp on all 1,940 rows: every entry at a row's
   minimum, the tokens never seen in that language's training data, is set
   to -21 natural-log units (Exp 20).
2. Re-examination of lines predicted into a language with fewer than 18,000
   training lines, and separately of lines predicted into one of four flat
   large-corpus languages: sco_Latn, bjn_Latn, arg_Latn, vls_Latn. Each of
   these four has more than 18,000 training lines but a token distribution
   unusually flat for its script, identified in Exp 48 from the zH flatness
   column of `outputs/diagnostic/lang_diagnostic.csv`. Each group is
   re-examined against its own per-language threshold, a percentile of the
   score margins that language achieves on its own training lines, computed
   under the floor-21 weight matrix. For the under-18,000 group the
   percentile decays with corpus size, q_L = 5 * (1 - N_L / 18,000) for a
   language with N_L training lines (`tau_floor21_gate.csv`); for the four
   flat languages it is fixed at the 5th percentile (`tau_flat4.csv`). A
   line is re-examined when its winning score exceeds the runner-up score
   by less than its predicted language's threshold.
3. A re-examined line is walked through the candidates ranked 2 to 5 of its
   saved top-five candidate list, in rank order. A candidate is accepted
   only if it has at least 100,000 training lines (RES_CAP, the resource
   cap established in Exp 33/34) and its saved score is within
   D3_PROX = 21.0 natural-log units of the top-1
   saved score (Exp 49; chosen from a derivation-part grid search from 0.5
   to 100 in steps of 1, where the optimum plateau spans roughly 15 to 35).
   A line with no accepted candidate keeps its pre-move prediction.

Judge-part overall F1 is 0.9498, +0.0018 [+0.0010, +0.0026] over floor21_gate
(paired bootstrap, B=10,000, seed 0, percentile, 95%, Exp 49), zero supported
collapses (clause C). Confirmation on the balanced test draw (seed 201,
185,204 lines) is in `outputs/tables/gate_flat4_prox21_confirmation_201.md`:
overall 0.9781, tail 0.8763, magnets 0.8811.

**Promotion lineage.** Baseline, the unmodified weight matrix, has
judge-part overall F1 0.9117. floor21_gate (mechanisms 1 and 2 above
restricted to the under-18,000 group only, mechanism 3 without the D3_PROX
condition) has 0.9480 and was promoted 2026-07-30 (user decision, recorded
in the superseded block below). gate_flat4_prox21 has 0.9498 and was
promoted 2026-08-06, superseding floor21_gate.

**The pool, with named mechanisms.** Experiment 47's shared re-examination
threshold (one shared value of 9.0 in place of the per-language thresholds
of mechanism 2, and the replacement bar of mechanism 3 lowered to 18,000
training lines) has judge-part overall F1 0.9534, the highest aggregate on
record, but fails clause C at class level: 9 languages with judge-part
support 15 to 2,947 lose more than 0.10 F1 against baseline, because a
per-language threshold set at a percentile of that language's own margins
bounds its own recall loss by that percentile and a shared value does not.
It is in the pool, not promotable in current form. Experiment 50's
pooled-frequency floor (bgfloor: unseen-token entries set to a shared
constant plus the token's log frequency in the pooled training data,
instead of one flat value) has judge-part overall F1 +0.000412 [+0.000043,
+0.000837] over floor-21 solo (paired bootstrap, B=10,000, seed 0), a real
but small gain at the edge of resolution. The user declined the
pre-registered composed step of rebuilding the Exp 49 gate on this matrix
(2026-08-06); bgfloor stays in the pool at this gate-less result.

**Evaluation instruments.** Selection: the balanced validation set, draw
seed 101, 188,061 lines. Confirmation of an ordinary candidate: the
held-out remainder, 45,004,014 lines outside draws 101 and 201. Final
reporting: the balanced test set, draw seed 201, 185,204 lines. For a
candidate whose rule or constants are chosen using remainder data (Exp 47
through 50, and the mixed matrix before them), the remainder is further
split by a seeded 40/60 partition (RULE_SPLIT_SEED=301,
RULE_SPLIT_FRACTION=0.40, Exp 44): an 18.0M-line derivation part
(18,001,573 lines) where rules and constants may be chosen, and a
27.0M-line judge part (27,002,441 lines) used for the one confirming
measurement per candidate, with comparators recomputed there.

**Open items.** (1) The objective interpretation question is unchanged: the
primary quantity averages per-language F1 over natural-distribution test
data rather than equal-volume-per-language test data, a choice recorded in
Exp 38 and open to correction. (2) Recorded follow-ups, none started: a
hybrid re-examination threshold (the smaller of Exp 47's shared 9.0 and a
per-language cap from own-train margins, to keep Exp 47's aggregate gain
while restoring the per-language collapse bound); the declined Exp 50
composed step (rebuilding the Exp 49 gate on the bgfloor matrix); direction
5 of the candidate-directions list in `EXPERIMENTS_PLAN.md` (raising the
unseen-token floor above -21 for the 12 languages where -21 makes them lose
on their own text), never tried. (3) The largest remaining error category,
measured on floor21_gate's residual as of 2026-07-30 (not re-measured
against gate_flat4_prox21's smaller residual set): of the 962,633 wrong
predictions remaining after floor21_gate on the judge part, 98.7% are lines
whose true language has at least 18,000 training lines, and 88.2% of those
are confused with another such language, concentrated in close pairs
(Indonesian and Malay, English and Scots, Mandarin and Wu Chinese;
`EXPERIMENTS_PLAN.md`, "The boundary this family cannot cross"). No
unseen-token treatment changes such a pair's score comparison materially
(at most 0.14 natural-log units per token). Whether a pairwise mechanism
can close this gap while keeping the add-a-language property, a new
language's parameters depending only on that language's own data, is open
and undesigned.

## Patterns established by Experiments 44 to 50 (2026-08-06)

(a) Balanced draws cannot rank methods for the primary quantity. On the
balanced validation draw (seed 101), the per-group leader disagreed with
the held-out-remainder leader in all six groups tested, and gt_min led
every group on the draw, because the draw's 100-line-per-language cap
removes the false-positive volume the primary quantity is defined on.
Per-language leader counts on the draw are noise: 1,623 of the 1,940
languages tied, and leader agreement between draws 101 and 102 was 58.1%
against a 25% chance level. Measured 2026-07-29, recorded in the decisions
block of the superseded "Current state (2026-07-29)" block below.

(b) Shared calibration constants score higher in aggregate than
per-language estimates in three measured cases: the floor level (-21 for
all languages, close to the pooled Good-Turing derivation of -20.60 in
`outputs/diagnostic/gt_counts.csv`), the refutation of per-script or
per-size floor levels, and the Exp 47 shared re-examination threshold
(SHARED_TAU=9.0, judge-part overall 0.9534 against floor21_gate's 0.9480).
The one measured exception: a shared re-examination threshold with no
per-language bound produces class-level collapses (Exp 47, 9 languages
losing more than 0.10 F1 against baseline), because a per-language
threshold set at a percentile of that language's own margins bounds its own
recall loss by that percentile and a shared value does not.
gate_flat4_prox21 therefore keeps per-language thresholds for own-recall
protection (`tau_floor21_gate.csv`, `tau_flat4.csv`) while using shared
constants for the floor level (-21),
the replacement-candidate resource bar (RES_CAP=100,000), and the proximity
condition (D3_PROX=21.0).

(c) Blocking a re-examination move relocates the false positive it would
have caused from the large target language back onto the small source
language. Under unweighted per-language averaging that relocation costs
more aggregate F1 than it saves, so an acceptance condition based on the
replacement candidate's identity costs aggregate F1, while a condition
based on score proximity to the top-1 candidate does not. Every
target-identity form of the direction-3 acceptance condition that repaired
the motivating Maltese example was measured on the derivation part and cost
aggregate F1 (Exp 49 pre-registration; `EXPERIMENTS_PLAN.md`, "Candidate
directions from the post-promotion error analysis", direction 3); the
score-proximity form (D3_PROX=21.0) does not have this cost and is the form
in the promoted configuration.

(d) A shallower unseen-token floor absorbs contested lines rather than
returning them to their true language. Exp 46 pre-registered two
predictions for a per-language mixed weight matrix (languages at or above
18,000 training lines keep the unmodified row, all others get the floor-21
row): that a head carve-out would recover most of floor21_gate's head loss,
and that non-head groups would gain from the per-language assignment. Both
predictions were refuted in sign: judge-part head group mean F1 fell
further under the carve-out (0.9580 against floor21_gate's 0.9586) while
every non-head group's judge-part mean F1 rose slightly (tail 0.7323
against 0.7306, lowmid 0.9606 against 0.9599, flat_magnet 0.6464 against
0.6435). The restored, shallower head floor absorbed more contested lines
instead of returning them to the head languages.

(e) Most of the remaining wrong predictions are confusions between two
large languages. `EXPERIMENTS_PLAN.md` ("The boundary this family cannot
cross") measured, on floor21_gate's residual as of 2026-07-30 (the promoted
configuration at that date; not re-measured against gate_flat4_prox21's
smaller residual set): of the 962,633 wrong predictions remaining on the
judge part, 98.7% are lines whose true language has at least 18,000
training lines, and 88.2% of those are confused with another language that
also has at least 18,000 training lines, concentrated in close pairs
(Indonesian and Malay, English and Scots, Mandarin and Wu Chinese). No
unseen-token treatment changes such a pair's score comparison materially
(at most 0.14 natural-log units per token); a mechanism reaching past this
point would need to separate specific language pairs directly, and whether
such a mechanism can keep the add-a-language property (a new language's
parameters depend only on that language's own data) is open.

## Current state (2026-07-29) (superseded by the 2026-08-06 block above; retained for the decision trail)

This block is kept as originally written, including its own claim below to
supersede earlier state descriptions; where it conflicts with the
2026-08-06 block above, the 2026-08-06 block is authoritative.

Read this block first; it supersedes any earlier state description in this file.
Terminology used throughout (defined once here): a **row** is one language's
vector of natural-log token probabilities in the 1,940 x 100,000 weight matrix;
the **primary quantity** is per-language F1 on natural-distribution test data with
all false positives counted, averaged unweighted over the 1,940 languages.

**Decisions and protocol changes of 2026-07-29 (user decisions; full text in
`EXPERIMENTAL_SETUP.md` amendment 7 and standing constraint 5).**
1. Status vocabulary made precise (amendment 7). Three statuses: **in the pool**
   (default: improves at least one recorded instrument or group by more than
   0.001 and is not hard-rejected; clause failures are recorded weaknesses with
   their required dig-ins, not removal), **promoted** (at most one per track;
   requires clauses (A), (B), (C) on that track plus an explicit user decision),
   **hard-rejected** (only when worse or equal on every recorded instrument and
   group with at least one strict loss, or the improvement traces to a bug or
   measurement artifact). Evaluation stays unconditional on all instruments of
   both tracks for every candidate. Verdicts below are re-read under this
   vocabulary with numbers unchanged: where an entry says "rejected for
   adoption", read "in the pool, not promotable in current form" unless the
   entry establishes a loss on every instrument. So gt_min, learned_bias
   (reg 5.0), and gt_margin are in the pool with their recorded mechanisms;
   floor-21's provisional adoption and gt_margin_adaptive's
   configuration-to-beat status are unchanged.
2. Plan consistency review (standing constraint 5): every plan is reviewed
   against all prior recorded decisions before execution; any divergence from
   committed text must be labeled as an amendment with its own user decision.
3. The combined-method pre-registration is amended in two places, both user
   decisions taken after an adversarial review measured the defects. (a) The
   assignment rule is derived from a seeded 40/60 split of the held-out
   remainder (RULE_SPLIT_SEED=301, RULE_SPLIT_FRACTION=0.40; derivation part
   about 18.0M lines, judge part about 27.0M lines; comparators recomputed on
   the judge part), replacing the committed draw-101 derivation. Measured
   reason: on draw 101 the per-group leader disagrees with the held-out-remainder
   leader in all six groups, and gt_min leads every group there, because the
   100-line-per-language cap removes the false-positive volume the primary
   quantity is defined on; per-language leader counts on the draw are noise
   (1,623 of 1,940 languages tie; 58.1% leader agreement between draws 101 and
   102 against 25% chance). This is a result about the instruments, independent
   of the combined method's outcome: balanced draws cannot rank methods for the
   primary quantity. (b) The bare aggregate success threshold is replaced by a
   paired bootstrap over the 1,940 languages (BOOT_B=10,000, BOOT_SEED=0,
   percentile, 95% level) of mixed minus gt_margin_adaptive on the judge part;
   promotion additionally requires the full adoption rule on both tracks.

**Objective (decided by the user 2026-07-25).** The primary quantity above. Every
language is weighted equally in the average, with an allowance to exempt an
extreme low-resource subset as unworkable and report it separately. The earlier
open question of which test distribution headlines the paper is therefore closed
for the average; the balanced draws remain selection and confirmation instruments,
not the primary metric. One interpretation choice is on the record and open to
correction: the average is taken over natural-distribution test data rather than
equal-volume-per-language test data (Exp 38 states the reasoning).

**Results under the primary quantity (Exp 38, held-out remainder, 45,004,014
lines).** The configuration to beat is **gt_margin_adaptive at 0.9334**, +0.0213
over the unmodified baseline. It is the leader on the aggregate but not dominant:
floor-21 is substantially better on the two groups this project targets
(languages under 1,000 documents 0.6337 against 0.4620; flat-confusion languages
0.5345 against 0.4206), and the learned bias is uniquely best on more individual
languages (602 against 587). Any new candidate should be compared against 0.9334
on the aggregate and against floor-21 on the small-language groups. No
configuration is best on every group: gt_margin_adaptive 0.9334 overall (best,
driven by the 1,000-18,000-document band at 0.9567), floor-21 0.9309 (best on
languages under 1,000 documents at 0.6337 and on flat-confusion languages at
0.5345), freq_prior 0.9264, learned_bias 0.9254 (best on the largest languages at
0.9696, on twin pairs at 0.9079, and the unique best on 602 individual
languages), margin_q5_head 0.9215, margin_q5 0.9201, baseline 0.9121. An oracle
that picked the best configuration per language would reach 0.9525 (Exp 40), with
the headroom concentrated in the tail (+0.0724) and flat-confusion (+0.0998)
groups.

**Live candidate set (near-tie co-selection, user decision 2026-07-25).** The
balanced validation set cannot separate six eligible configurations (macro-F1
0.9794-0.9800), so all six are carried forward rather than narrowed to one:
freq_prior, learned_bias, floor21, margin_q5, margin_q5_head, gt_margin_adaptive.
floor-21 is top-ranked on that selection set; gt_margin_adaptive leads on the
primary quantity. gt_min is the champion of the equal-volume (uniform-prior)
track, flagged with two per-language dig-ins that were completed (Exp 32).

**Evaluation machinery (three datasets, never interchanged).** Selection: the
balanced validation set, draw seed 101, 188,061 lines, up to 100 lines per
language. Confirmation of a ranked candidate: the held-out remainder, 45,004,014
lines outside draws 101 and 201, where per-language F1 counts every false
positive. Final reporting: the balanced test set, draw seed 201, 185,204 lines,
disjoint from draw 101. The adoption rule and its six amendments are in
`EXPERIMENTAL_SETUP.md`; the most consequential are the outlier-tolerant collapse
clause (up to two per-language collapses trigger a required investigation rather
than rejection) and the dual-track verdicts.

**Out-of-domain (Exp 39, CommonLID, 373,230 web lines).** Line-weighted
macro-aware accuracy improves for both carried leaders (floor-21 +0.0040,
gt_margin_adaptive +0.0070) and the margin gate transfers without any refitting.
Under the objective-consistent per-tag macro-averaged F1 both are slightly
negative (-0.0046, -0.0061). CommonLID's 109 tags are predominantly larger
languages and barely contain the tail labels these methods repair, so it
structurally cannot show their benefit; out-of-domain claims should be scoped
accordingly.

**Infrastructure state.** A numerical bug in the fixed-vocabulary EM trainer was
found, diagnosed, fixed, and the fix adopted (Exp 41, 42): the trainer's E-step
accumulated in 32-bit floats, which breaks on very long training lines, and the
fork silently zeroed the resulting non-finite counts. The fix is in the
sentencepiece fork (commits d0208d9, c5921a2), the installed binary is the
patched build (previous binary kept as `~/.local/bin/spm_train.pre_fp64`), and
both Apertus models were retrained: `glotlid_apertus131k_fp64.unilid` and
`glotlid_apertus200k_fp64.unilid`. The 100k production model that carries every
main-line result was never trained through this path and is unaffected.

**Open decisions and pending measurements.**
1. The per-language combined method (the strongest open direction) is in
   execution under the amended pre-registration (see `EXPERIMENTS_PLAN.md`,
   "Plan: per-language combined method", with the 2026-07-29 amendments). The
   honest ceiling for its six-combination treatment space, measured over the
   four combinations with existing solo runs on the full remainder, is 0.9449
   overall, tail 0.6731, flat_magnet 0.5939 (headroom +0.0115 / +0.0394 /
   +0.0594), smaller than the Exp 40 seven-configuration oracle.
2. Whether the primary quantity should instead average over equal-volume test
   data (see the interpretation note above).

The Apertus 131k branch question is now closed on clean evidence (Exp 43): the
retrained model is still negative against the 100k production model on both
metric views, so the vocabulary-size regression is real and was not an artifact
of the training bug, though the bug had overstated its magnitude by 10 to 20%.

**Refuted method families (do not revisit without a new mechanism).** Moving
probability mass toward group typicality in any form (Exp 9, 13, 18, 19), length
normalization (Exp 2, 5), floor clamps in both directions as a family (Exp 6, 20,
23a; the calibrated descendants floor-21 and the Good-Turing rescale are the
survivors), heuristic variance reweighting (Exp 8a), entropy sharpening. The
macrolanguage hierarchy is a null result with a useful ceiling measurement
(Exp 21). Prior-style additive biases remain measured and available but are
disfavored on modularity grounds and were rejected for adoption on per-language
harm (Exp 16, 25).

**Promotion (2026-07-30, user decision).** floor21_gate is PROMOTED on the
natural track after Exp 44-46 and amendment 8. It supersedes floor-21's
provisional adoption and gt_margin_adaptive's configuration-to-beat status;
both remain in the pool. Confirmation numbers on the balanced test draw (seed
201, 185,204 lines) are in `outputs/tables/floor21_gate_confirmation_201.md`:
floor21_gate overall 0.9741, tail 0.8685, magnets 0.8758.

**Promotion (2026-08-06, user decision).** gate_flat4_prox21 is PROMOTED on
the natural track after Exp 47-50. It supersedes floor21_gate, which remains
in the pool. Judge-part overall F1 is 0.9498, +0.0018 [+0.0010, +0.0026]
over floor21_gate (Exp 49), zero supported collapses. Confirmation numbers
on the balanced test draw (seed 201, 185,204 lines) are in
`outputs/tables/gate_flat4_prox21_confirmation_201.md`: gate_flat4_prox21
overall 0.9781, tail 0.8763, magnets 0.8811. The user declined the Exp 50
composed step (rebuilding the Exp 49 gate on the pooled-frequency floor
matrix); the frequency-shaped floor (bgfloor) stays in the pool as a
recorded small positive over floor-21 solo on the judge part, +0.000412
[+0.000043, +0.000837] (paired bootstrap, B=10,000, seed 0).

## Experiment 50: pooled-frequency unseen-token values; eligible, gain at the edge of resolution (2026-08-06, job 3016380)

The scoring pass completed clean (2:03:04; c = -8.4740, assigned plateau range
-27.61 to -12.31, all build gates passed). Judge part
(`outputs/tables/mixed_eval_bgfloor.md`): overall 0.9304 against floor-21
solo's 0.9300; the pre-registered gate-less contrast is +0.000412 [+0.000043,
+0.000837] (paired bootstrap, B=10,000, seed 0), an interval whose lower end
sits essentially at zero (Correction 2026-08-06: this entry's first committed version stated the interval as [+0.000030, +0.000801], figures written before the computation had run; the values here are the ones that reproduce from the recorded CSV under the documented constants, independently recomputed during the promotion close-out.). Two-sided ELIGIBLE, clause (C) clean. Tail 0.6196
against floor-21's 0.6161. Reading: shaping the unseen-token values by the
pooled token frequency is a real but very small improvement over one flat
value, an order of magnitude below the gate-side gains of Experiments 48 and
49. The pre-registered conditional (rebuild the Experiment 49 gate on this
matrix if the interval is above zero) is technically triggered at the
boundary; whether to spend the composed step is put to the user together with
the promotion decision. User decision (2026-08-06): the composed step is
skipped; bgfloor stays in the pool at this gate-less result.

## Experiment 50 pre-registration: unseen-token values from the pooled token frequency (direction 4; 2026-08-06, before any run)

The rule under test, in full: in every language's row, each entry at the row
minimum (the tokens never seen in that language's training data) is set to
c + log p_base(token), where p_base is the base Unigram tokenizer distribution
already stored in the model file (the pooled training-data token
distribution), and c is one shared constant fixed so that the mean assigned
value over the non-special vocabulary equals -21.0 (c = -21.0 minus the mean
of log p_base). A token common in the pooled data but absent from one
language's data thus scores higher in that language than a token rare
everywhere; the promoted family's single value -21 is the special case of a
flat p_base. Seen entries and the four special tokens are untouched; the
special columns must be bit-identical to the unmodified matrix. No per-language
fitting; a new language inherits both the constant and p_base.

Procedure: matrix build plus one full-pool scoring pass cloned from the
recorded full_test_floor21.py pattern (chunked, resumable, fingerprinted, all
alignment gates), output pred_bgfloor.npy. First evaluation is the
like-for-like gate-less comparison: judge-part per-language F1 against floor21
solo (0.9300) and baseline (0.9117), paired bootstrap per the standard
constants. If the interval against floor21 is above zero, a second step
rebuilds the Experiment 49 gate rule on top (fresh candidate arrays under the
new matrix) for the composed comparison against gate_flat4_prox21 (0.9498).
Derivation-side prior evidence (80,000-line sample): line accuracy 96.138%
against floor-21's 96.097%, no increase in wrong predictions onto small
labels; the sign on per-language F1 is what this experiment establishes.

## Experiment 49: score-proximity condition on the replacement; eligible, the strongest eligible configuration (2026-08-06, job 3016337)

The in-job reproduction gate passed (the reimplemented two-step walk with the
condition disabled matched pred_gate_flat4_tau5.npy bit-identically), then the
conditioned build moved 314,314 lines. Judge part
(`outputs/tables/mixed_eval_gate_flat4_prox21.md`): overall 0.9498, +0.0018
[+0.0010, +0.0026] over the promoted floor21_gate; the refinement contrast
against gate_flat4_tau5 is +0.0012 [+0.0007, +0.0016] (paired bootstrap,
B=10,000, seed 0, from the two per-language CSVs). Two-sided ELIGIBLE, clause
(C) clean, zero supported collapses. The three languages flagged by the
derivation-part design are confirmed as real costs relative to gate_flat4_tau5
(tly_Latn 0.5242 to 0.3502, shu_Arab 0.9713 to 0.8319, las_Latn 0.9155 to
0.7580; all remain far above baseline, which is why clause (C) is clean):
blocking a far-fetched replacement keeps the false positive on the small
source language, and these three absorb some of those kept lines. Maltese is
unchanged (0.9357 to 0.9362), as pre-registered. Status: the strongest
eligible configuration on record (the eligible arc is floor21_gate 0.9480,
gate_flat4_tau5 0.9486, gate_flat4_prox21 0.9498; Exp 47's 0.9534 remains in
the pool with its class-level collapse failure). Promotion proposal still
deferred until direction 4 is tried.

## Experiment 49 pre-registration: score-proximity condition on the replacement candidate (direction 3; 2026-08-06, before the run)

The rule under test, in full: gate_flat4_tau5's two re-examination steps stay
exactly as built (per-language thresholds from tau_floor21_gate.csv for lines
predicted into languages under 18,000 training lines, and from tau_flat4.csv
for the four flat large-corpus languages; replacement candidates walked in
rank order 2 to 5; each must have at least 100,000 training lines). One added
condition: a replacement candidate must also have a saved score within
D3_PROX = 21.0 natural-log units of the top-1 saved score; a candidate
failing either check is skipped and the walk continues; a line with no
acceptable candidate keeps its pre-move prediction. D3_PROX was chosen on the
derivation part (grid 0.5 to 100 in steps of 1; the optimum plateau spans
roughly 15 to 35 within 0.0003, so 21.0 is representative, not finely tuned).
The condition needs only the saved per-line candidate scores; no calibration.

Implementation gate, required before the conditioned build is trusted: with
the condition disabled, the reimplementation of the two-step rule must
reproduce pred_gate_flat4_tau5.npy bit-identically (the design analysis
already reproduced it with zero mismatches on all 814,787 derivation-part
affected lines).

Derivation-part predictions: aggregate 0.9484 to 0.9499 (+0.0015), every
group at or above gate_flat4_tau5. Recorded caveats: the motivating Maltese
example is mostly not repaired by this form (its wrong moves sit close to the
top-1 score); every target-identity form that does repair it was measured and
costs aggregate F1, because blocking a move relocates the false positive from
a large language back onto a small one under unweighted per-language
averaging. Three languages lose more than 0.10 against gate_flat4_tau5 on the
derivation part (tly_Latn, shu_Arab, las_Latn) and are checked on the judge
part. Decision criteria: paired bootstrap (B=10,000, seed 0, percentile, 95%)
against floor21_gate via the standard evaluation, plus the same bootstrap
against gate_flat4_tau5 computed from the two per-language CSVs; statuses per
amendment 7.

## Experiment 48: re-examining the four flat large-corpus languages; eligible, improves the promoted configuration everywhere (2026-08-06, job 3015805)

Build clean (thresholds 2.93 to 5.88 nats from 1,866 to 1,971 self-won
calibration lines per language; 84,586 of 133,606 in-set lines moved, 56,491
to the true label, 66.8% move precision;
`outputs/tables/gate_flat4_tau5_build.md`). Judge part
(`outputs/tables/mixed_eval_gate_flat4_tau5.md`): overall 0.9486, +0.0006
[+0.0001, +0.0013] over the promoted floor21_gate, interval above zero;
+0.0157 over gt_margin_adaptive. Two-sided ELIGIBLE; clause (C) clean, zero
supported collapses. The pre-registered refutation condition inverted: the
four re-examined languages gained the most (sco_Latn 0.2870 to 0.6189,
bjn_Latn 0.3276 to 0.6790, arg_Latn 0.4209 to 0.7185, vls_Latn 0.4577 to
0.7746) because their precision was low enough that shedding false positives
outweighed the recall cost, and their large neighbours also gained (eng_Latn
+0.0087, ind_Latn +0.0051, spa_Latn +0.0028, nld_Latn +0.0030). The
large-language group is 0.9600, above the unmodified baseline's 0.9593, so
the promoted configuration's one recorded weakness (its head loss) is
repaired. Status: the strongest eligible configuration on record; the
promotion proposal is deferred until directions 3 and 4 are tried, since they
build on this configuration.

## Experiment 48 pre-registration: re-examining the four flat large-corpus languages (direction 2; 2026-08-06, before the run)

The rule under test, in full: start from the promoted configuration's
predictions (floor21_gate). Additionally re-examine every line whose predicted
language is one of the four languages that are in the flat-distribution
category while having at least 18,000 training lines (sco_Latn, bjn_Latn,
arg_Latn, vls_Latn; the promoted configuration exempts them because its
re-examined set is defined by corpus size alone, and they receive 54.1% of the
remaining wrong predictions into small-language or flat-distribution labels).
For each of the four, the threshold is the 5th percentile of the score margins
that language achieves on its own training lines, computed under the floor-21
weight matrix (the fixed-percentile form, because the promoted configuration's
size-adaptive percentile is zero at or above 18,000 lines). A re-examined line
moves to the highest-scoring candidate ranked 2 to 5 with at least 100,000
training lines (the promoted configuration's own replacement bar; the four
languages' large neighbours, English, Indonesian, Spanish, and Dutch, all
qualify). Constants: MARGIN_Q = 5, CALIB_MAX = 2,000, CALIB_SEED = 0, all the
recorded calibration constants; no new constants. Modularity caveat, recorded:
the flat-distribution category uses one validation-derived input, so a new
language's membership needs either a validation scoring pass or the
weights-only flatness score alone; which rule generalizes is left to the
write-up if the direction succeeds.

Decision criteria: paired bootstrap (B=10,000, seed 0, percentile, 95%) of
per-language F1, candidate minus floor21_gate, judge part; statuses per
amendment 7. Bounds from the analysis: at most +0.0012 overall and +0.0204 on
the flat-distribution group if every removed wrong prediction were removed;
refuted if the four languages lose more from re-examination of their genuine
lines than their neighbours gain in precision.

## Experiment 47: shared re-examination threshold; best aggregate on record, but a class-level per-language failure (2026-08-06, job 3014614)

Build clean (2,236,864 candidate lines saved, one top-1 disagreement; 387,039
predictions moved, 213,146 to the true label;
`outputs/tables/gate_shared9_bar18k_build.md`). Judge part
(`outputs/tables/mixed_eval_gate_shared9_bar18k.md`): overall 0.9534, +0.0054
[+0.0031, +0.0078] over the promoted floor21_gate, the best aggregate on
record; languages under 1,000 training lines 0.7579; the 1,000-to-18,000 band
0.9714; the flat-distribution group 0.6869; languages at or above 18,000
training lines 0.9547 (floor21_gate 0.9586). The derivation-part predictions
were confirmed (predicted +0.0052 overall, measured +0.0054).

But clause (C) fails at class level: 9 languages with judge-part support 15 to
2,947 lose more than 0.10 F1 against baseline (worst llb_Latn 0.9244 to
0.5716 at support 2,523; also zgh_Tfng 0.9292 to 0.6148, bpr_Latn, bps_Latn,
cya_Latn, mrj_Cyrl, nhe_Latn, twx_Latn, thv_Latn), and clause (A) fails on
every stratum (balanced-val tail drop 0.0964, magnets 0.1117, twins 0.0209).
Mechanism: the single shared threshold removes the bound on own-recall loss
that the per-language calibrated thresholds provided by construction (a
threshold at the 5th percentile of a language's own margins caps its own
recall loss near 5%); the nine victims are languages whose genuine lines have
small margins against a much larger neighbour, so a 9.0 threshold moves their
true lines away wholesale. Status under amendment 7: in the pool, not
promotable in current form; floor21_gate remains promoted. Recorded follow-up
candidate, not pre-registered: a hybrid threshold, the smaller of 9.0 and a
per-language cap from own-train margins, to keep the aggregate gain while
restoring the per-language bound.

## Experiment 47 pre-registration: shared re-examination threshold (direction 1; 2026-08-05, before any run)

The user directed trying candidate directions 1 through 4 in order (2026-08-05),
which also resolves the recorded concern for direction 1: the shared threshold
is equivalent to a corpus-size-indexed score adjustment, a family previously
set aside; the user approved trying it with that equivalence on the record.

The rule under test, in full: keep the promoted configuration's weight matrix
(every unseen token at -21). For every line whose predicted language has fewer
than 18,000 training lines, move the prediction to the highest-scoring
alternative among the top five candidates whose training corpus has at least
18,000 lines, whenever the winning score exceeds the second-place score by
less than a single shared threshold of 9.0 (natural-log units). No
per-language calibration anywhere. Constants, both chosen on the derivation
part of the seed-301 split from the recorded sweep
(`outputs/diagnostic/gate_threshold_sweep_20260730.csv`, optimum flat between
7 and 12): SHARED_TAU = 9.0, replacement-candidate minimum 18,000 (the
existing head boundary, not a new constant).

Decision criteria: paired bootstrap (B=10,000, seed 0, percentile, 95%) of
per-language F1, candidate minus floor21_gate, on the judge part. Interval
above zero plus the adoption-rule clauses: promotion proposed to the user.
Interval containing zero: in the pool. Hard reject only if worse or equal on
every recorded instrument and group. Derivation-part predictions to be checked
against the judge part: overall +0.0052, languages under 1,000 training lines
+0.0103, flat-distribution group +0.0329, languages at or above 18,000
training lines -0.0033.

## Experiment 46 pre-registration: the mixed matrix under rule v1 (2026-07-30, before scoring)

Rule v1 (user sign-off 2026-07-30, materialized by `analysis/mixed_assign.py`,
`outputs/diagnostic/mixed_assignments.csv`): languages with N >= 18,000 keep the
unmodified row, gate off (860 languages); every other language gets the floor-21
row treatment with the adaptive gate on (1,080 languages). The only feature is N
(training-side). Head is assigned unmodified rather than gt_min (derivation-part
head means 0.9596 against 0.9594) to keep production head rows bit-identical.

Decision criteria (amendment-7 vocabulary; anchor switched to floor21_gate per
the user's conditional decision, now confirmed by Exp 45): paired bootstrap
(B=10,000, seed 0, percentile, 95%) over the 1,940 languages of per-language F1,
mixed minus floor21_gate, on the judge part of the seed-301 split, with the
gt_margin_adaptive contrast reported alongside. Interval above zero plus clauses
(A), (B), (C): promotion proposed to the user. Interval containing zero: in the
pool, explored more deeply. Hard reject only if worse or equal on every recorded
instrument and group. The clause-(A) cap question raised by Exp 45 is DEFERRED
by user decision until this run's results are in.

Pre-registered predictions: (1) the mixed configuration recovers most of
floor21_gate's 0.0007 judge-part head loss; (2) it loses ground to floor21_gate
concentrated in lowmid, correlating negatively (Spearman) with the per-language
floor-gap shift recorded in `mixed_assignments.csv` (555 non-zero shifts, all
negative, quartiles -3.24/-2.73/-2.23 nats, 549 of 555 in lowmid); (3) FPs into
tail labels stay far below baseline's judge-part 13,483 (the Exp 28 falsifier).

## Experiment 46: the mixed matrix under rule v1 is indistinguishable from uniform floor21_gate (2026-07-30, job 2932154)

All four stages clean (no-op gate bit-identical on 497,188 chunk-0 lines; gate
agreement 1.0000; `outputs/tables/mixed_matrix_build.md`,
`outputs/tables/mixed_eval_mixed_mixed_solotau.md`). Judge-part results: mixed
0.9482 overall against floor21_gate 0.9480, paired bootstrap +0.0002
[-0.0003, +0.0006]; the interval contains zero, so under the pre-registered
criteria the mixed configuration is in the pool as a near-tie, and the
per-language assignment adds nothing measurable over the uniform composition.
Both pre-registered mechanism predictions were refuted in sign: the head
carve-out lowered head further (0.9580 against floor21_gate's 0.9586; the
restored shallower head floors absorb more contested lines, the reassignment
law operating through floors) while raising every non-head group slightly
(tail 0.7323 vs 0.7306, lowmid 0.9606 vs 0.9599, flat_magnet 0.6464 vs
0.6435); the floor-gap-shift correlation is absent within the shifted subset
(Spearman r=0.037, p=0.39, n=555; shifted languages averaged +0.00074 against
-0.00030 for unshifted; computed from
`outputs/diagnostic/mixed_eval_judge_f1_mixed_mixed_solotau.csv` and
`outputs/diagnostic/mixed_assignments.csv`). The tau-recalibration component
is +0.0003 (mixed 0.9482 against mixed_solotau 0.9479). Clause (A) fails as
predicted (balanced-val tail drop 0.0457, magnets 0.0416, against the 0.03
cap); both tracks otherwise as for floor21_gate. Conclusion: the naive
per-language-assignment direction is closed for this codomain by a null
interaction, not by harm; floor21_gate, the simpler uniform configuration,
stands as the strongest configuration on the judging instrument, in the pool,
with promotion blocked only by the clause-(A) cap question (deferred to now).

**Exp 45 addendum (2026-07-30, amendment 8).** Under the conditional widening
of the clause-(A) cap (amendment 8, `EXPERIMENTAL_SETUP.md`; WIDE_CAP=0.05
when the veto-side stratum gain exceeds GAIN_RATIO_MIN=5 times the
within-stratum loss), floor21_gate re-evaluates to ELIGIBLE with every clause
passing: tail gain-to-loss ratio 9.36 (0.3986/0.0426), magnets 9.35
(0.3589/0.0384), both losses under 0.05. The replay harness reproduced the
recorded pre-amendment verdicts exactly before the re-evaluation, and gt_min's
rejection is unchanged (its failures are clause-B/C side). Both verdicts are
on the record per the amendment's prospective-application rule. Promotion and
the draw-201 confirmation were both decided by the user on 2026-07-30
(recorded above under "Promotion"). The draw-201 confirmation ran 2026-07-30
(`outputs/tables/floor21_gate_confirmation_201.md`): floor21_gate
overall 0.9741, tail 0.8685, magnets 0.8758, against baseline overall 0.9809,
tail 0.9086, magnets 0.9121. The per-language collapse check on that same draw
found 8 supported collapses (support >= 10, F1 drop > 0.10), all attributable
to the draw's own per-language support cap of at most 100 lines, where a
handful of flipped predictions moves F1 by more than 0.10; the promotion-gate
clause (C), computed on the far larger judge part in Exp 45/46, remains clean
at zero supported collapses.

## Experiment 45: solo-gate references; floor21_gate is the strongest configuration on record (2026-07-30, jobs 2930701/2930702)

The two missing codomain references built (`analysis/solo_gates.py`, top-1
agreement 1.0000 on both; `outputs/tables/{unmod,floor21}_gate_build.md`) and
evaluated on the judge part (`analysis/mixed_eval.py`,
`outputs/tables/mixed_eval_floor21_gate_unmod_gate.md`). **floor21_gate
(floor-21 rows for all 1,940 languages plus the adaptive gate, target bar
100k) is the strongest configuration on the judging instrument**: judge-part
overall 0.9480 against gt_margin_adaptive 0.9329, paired bootstrap +0.0151
[+0.0112, +0.0191]; tail 0.7306 (floor-21 alone 0.6161); flat_magnet 0.6435;
lowmid 0.9599 (also the leader); head 0.9586 against baseline 0.9593. FPs into
tail labels 1,912 against baseline 13,483; clause (C) clean (zero supported
collapses). Clause (A) FAILS: balanced-val tail 0.9170 -> 0.8744 (drop 0.0426)
and magnets 0.9174 -> 0.8790 (0.0383), both beyond the 0.03 widened cap, so
under amendment 7 floor21_gate is IN THE POOL, not promotable in current form;
the cap decision is deferred (user, 2026-07-30) until Exp 46. unmod_gate (gate
on unmodified rows): two-sided eligible, judge-part overall 0.9328,
indistinguishable from gt_margin_adaptive (-0.0001 [-0.0021, +0.0018]).

## Experiment 44: evidence base and the seed-301 rule split (2026-07-30)

`analysis/combined_evidence.py` (`outputs/tables/combined_evidence.md`): all
wiring gates passed (Exp 38 reproduced to 1.1e-16; gt_min veto anchor exact on
FPs). The seed-301 split partitions the 45,004,014-line remainder into
18,001,573 derivation and 27,002,441 judge lines. On the derivation part the
codomain oracle is 0.9514 against floor21_gate's 0.9478 (gain +0.0036 overall,
+0.0236 tail, +0.0031 lowmid), so one uniform configuration captures nearly all
of the per-language assignment headroom that motivated the combined method.
Derivation-part support: tail median 8 true lines, 37 of 96 tail languages at
10 or more, so per-language leader evidence is group-level only there.

## Headline observations (historical, Exp 1-9 era)

1. All evaluated scoring modifications reduce 500k-sample accuracy relative to the 0.960
   UniLID baseline: full length normalization (α=1) → 0.885; partial normalization at any
   α > 0.1 (best α=0.1 → 0.961, +0.001); floor=−10 → 0.916; heuristic discriminative
   weighting in all three setups (best A α=0.5 → 0.866). The accuracy reduction concentrates
   on the <30 char bin in every case.

2. Token-count delta (pred − true) on full-test-set misclassifications has mean −0.17
   (Cohen's d −0.092; one-sample t = −122.5, p ≈ 0). Mean magnitude grows monotonically with
   input length (<30: −0.11; 300+: −2.71). Median is 0 across all length bins except 300+
   (−1.00); 61.1% of misclassifications have equal token count under the predicted and true
   languages.

3. Mean KL(lang‖base) by resource bin: <500 → 0.32; 500–1k → 0.50; 1k–12k → 0.70; 35k+
   → 0.68. Saturates above ~10k samples; log(training count) vs KL r = 0.261.

Possible (unconfirmed) conclusions:
- The existing UniLID parameterization is near-optimal under sum-of-log-prob scoring;
  modifications that reduce per-token discrimination reduce accuracy most on short inputs.
- Low-resource (<5k sample) per-language distributions are under-fit (close to the base)
  rather than over-fit.
- Post-hoc probability-space blending alone does not jointly improve overall and very-low-
  resource accuracy in the configurations tested.

---

## Experiment 1 — Multi-system comparison

**Question:** How does UniLID compare to four alternatives (UniLID-DeepSeek, UniLID-Qwen,
UniLID-Marg, fastText) across text length, training-resource level, and script?

**Findings** (500k sample; full-dataset tables also generated):
- Overall accuracy is tight across systems: UniLID 0.960, Marg 0.961, DeepSeek 0.959,
  Qwen 0.951, fastText 0.947. fastText has the highest macro-F1 (0.947).
- Accuracy is strongly length-dependent: <30 chars ≈ 79% for UniLID, >300 chars ≈ 99.5%.
- Resource sweet spot at 12k–18k training samples (99.5%); the 35k+ bin is lower (95.8%)
  but is 92.8% of the data.
- Latin script (1,659 languages): 96.3%; Devanagari 89.6%, Arabic 90.7%; unique-script languages reach ~100%.
- UniLID's worst relative weakness: `lzh_Hani` (Literary Chinese), 25.0% error vs
  DeepSeek 3.9%. Its standout strength: `azj_Latn`, 1.1% error vs DeepSeek 61.2%.

**Artifacts:** `outputs/tables/table1_overall`, `table2_by_length`, `table3_by_resource`,
`table4_by_script`, `table5_error_overlap`, `table6_per_script_winner`,
`table7_divergences` (`.md`/`.tex`); confusion-matrix PNGs/TeX for 7 clusters.
**Detail:** `EXPERIMENTS.md` §1.

---

## Experiment 2 — Tokenization length bias and normalization

**Hypothesis:** Because UniLID sums per-token log-probs with no length prior, it may favor
languages that produce fewer tokens for the same text.

**Findings:**
- The bias exists and scales with length. Full-dataset misclassifications (1,789,423):
  mean token delta (pred − true) = −0.17, Cohen's d = −0.092. At 300+ chars the mean delta
  is −2.71.
- Pairwise counterfactual: normalizing by token count would flip 18.6% of errors toward
  the true language. When the predicted language uses ≥2 fewer tokens, ~75% of those errors
  are pairwise-correctable.
- **But full re-classification with normalized scores made accuracy worse: 0.960 → 0.885.**
  Raw rescore reproduced the original predictions exactly (100% agreement, validating the
  Rust implementation). Normalization broke far more predictions than it fixed (net
  −37,740), with the damage concentrated on short texts (<30 chars: 0.792 → 0.566).

**Possible (unconfirmed) conclusion:** the unnormalized sum-of-log-probs scoring carries
signal that simple length normalization removes; the per-token average is noisiest for
short texts.

**Artifacts:** `outputs/tables/length_bias`, `normalized_comparison`;
`outputs/figures/length_bias_histogram.png`. **Detail:** `EXPERIMENTS.md` §2.

---

## Experiment 5 — Partial length normalization (alpha sweep)

**Question:** Does a partial correction `score / n_tokens^alpha` for `alpha ∈ {0.0,…,1.0}`
help, even if full normalization (alpha=1) hurts?

**Observations:** Best is `alpha=0.1` at 0.961 accuracy (+0.001 over `alpha=0.0`),
net +114 corrections out of 1,749 changed predictions. Accuracy decreases monotonically for
`alpha > 0.1`; the <30 char bin drops fastest (0.792 → 0.566 at `alpha=1.0`).

**Artifacts:** `outputs/tables/alpha_sweep`; `outputs/figures/alpha_sweep.png`.
**Detail:** `EXPERIMENTS.md` §5.

---

## Experiment 6 — Log-probability floor sweep

**Question:** Does clamping all per-language weights at a higher floor (giving OOV tokens a
finite low probability) help?

**Observations:** `floor=-22` clamps 0 elements; predictions identical to baseline.
`floor=-15` clamps 90.7% of the matrix and changes 3,372/500k predictions, net −109.
`floor=-10` clamps 99.5%, accuracy 0.960 → 0.916 with the <30 char bin going 0.792 → 0.608;
25,485 predictions broken vs 3,390 corrected.

**Note:** A code comment in `analysis/floor_sweep.py` refers to "OOV at -1e30". The user
flagged (recovered prompt, 2026-04-06) that `-1e30` is never used in the repo and the
actual clamp value differs. The comment is inaccurate; trust the runtime weight matrix.

**Artifacts:** `outputs/tables/floor_sweep`; `outputs/figures/floor_sweep.png`.
**Detail:** `EXPERIMENTS.md` §6.

---

## Experiment 3 — Per-language distribution analysis

**Question:** How do the EM-estimated per-language distributions differ from the base
distribution and from each other, and where does EM noise appear?

**Findings:**
- KL(lang‖base) is highest for unique-script languages (~1.2–1.3) and lowest for
  low-resource Latin-script languages (~0.20–0.24), which barely moved from the base.
- Mean KL rises with resource level then saturates above ~10k samples
  (log-count vs KL: r=0.261).
- Pair with one data-poor language: Hindi/Angika (anp 4,499 samples) has correlation 0.884
  (Indonesian/Malay 0.896 at 100k each) but MAD 2.218 (Indonesian/Malay 0.364), with many
  tokens near the probability floor.
- Within-cluster pairwise KL vs min training size: r ≈ −0.03 (≈ zero).

**Resource-level structure:** mean KL from base 0.32 at <500 samples vs 0.68–0.71 at 5k+;
mean MAD 1.17 vs 4.84–5.90. EM uses no explicit regularization (only a probability floor
and convergence early stopping).

**Possible (unconfirmed) conclusion:** low-resource (<500 sample) distributions are
under-fit (close to the base) rather than over-fit; for mixed-resource pairs the lower-
resource distribution shows distinct EM noise.

**Artifacts:** `outputs/tables/distribution_analysis`;
`outputs/figures/kl_vs_training_size.png`, `pairwise_logprob_scatter.png`,
`pairwise_kl_vs_training.png`. **Detail:** `EXPERIMENTS.md` §3.

---

## Experiment 4 — Token classification for confused pairs

**Question:** Are the tokens that drive within-pair divergence linguistically meaningful or
artifacts (punctuation, encoding, domain markers)?

**Observations:** Across 300 tokens (15 pairs × 20), morphological affixes contribute 32.6%
of KL and content words 22.8% (combined 55.4%); function words 15.7%; punctuation 10.5%;
character/phonotactic 10.2%; multi-word units 7.9%; script/encoding 0.0%; domain/religious
0.3%. The one mixed-resource pair (Hindi/Angika) is 75% content words, with the Angika
distribution showing many tokens at or near the probability floor (Exp 3.5). Indonesian/
Malay shows 7.3% domain KL from `ĠYehuwa` (JW.org marker).

**Possible (unconfirmed) conclusions:** discriminative features in the top 20 KL tokens per
pair are dominated by linguistic units rather than tokenization or encoding artifacts; the
Indonesian/Malay distinction is partially dependent on religious-text domain markers.

**Artifacts:** `outputs/tables/token_classification`;
`outputs/figures/token_categories_stacked.png`. **Detail:** `EXPERIMENTS.md` §4.

---

## Experiment 7 — Training-data analysis

**Question:** Does the training corpus have domain skew, quality problems, or script
mislabeling that explains confusions?

**Findings** (full corpus, 60,683,151 lines):
- 98.1% of training data classifies as "other" (not religious, not Wikipedia); 1.9%
  religious; Wikipedia markers 0.002%. By resource bin religious share ranges 0.3%–2.7%.
  Confusable cluster examples: Indonesian (ind_Latn) 2.2% religious vs Malay (zsm_Latn)
  0.2%; Bokmål (nob_Latn) 6.2%. **Caveat:** the heuristics are conservative (keyword/
  pattern matching); true religious fraction is likely higher.
- Low-resource languages (<500) have shorter texts (~84 chars) and tiny vocabularies
  (~1.4k tokens) vs ~100k for the 35k+ bin.
- Script labels: 20 languages have >5% off-script characters. 6 "Canadian Aboriginal
  Syllabics" languages are actually 100% Latin romanization; Japanese `jpn_Jpan` is mostly
  Hiragana (a script-code mapping artifact, not a data problem). 99.0% of languages are
  >95% in-script.

**Scope note:** sub-analyses 7.2 (mislabeling) and 7.3 (overlap) were deferred. See
`EXPERIMENTS_PLAN.md`.

**Artifacts:** `outputs/tables/train_data_analysis.md`;
`outputs/figures/train_domain_stacked.png`, `train_quality_scatter.png`,
`train_script_purity.png`. **Detail:** `EXPERIMENTS.md` §7.

---

## Experiment 8a — Heuristic discriminative weighting

**Hypothesis:** Variance-based up-weighting of discriminative tokens within a confusion
cluster could improve within-cluster separation.

**Observations:** All three setups reduce accuracy at every parameter setting tested. At
the mildest settings: A(α=0.5) 0.866, B(α=0.5) 0.889, C(β=1.0) 0.899, vs 0.960 baseline.
Per-cluster accuracy is 0 across all seven clusters for A and B at α ≥ 1.0.

**Possible (unconfirmed) conclusion:** variance-based token re-weighting at the granularity
used here does not improve within-cluster discrimination; any improvement, if it exists,
requires a mechanism other than per-token additive/multiplicative adjustments to the EM-
trained weights.

**Artifacts:** `outputs/tables/discriminative_heuristic.md`. **Detail:** `EXPERIMENTS.md`
§8a.

---

## Experiment 9 — Distribution transfer for low-resource languages

**Hypothesis:** Interpolating an under-fit low-resource distribution toward a related
high-resource language (9a) or a script-average (9b) could raise low-resource accuracy.

**Observations** (probability-space interpolation, `lambda=1` is baseline):
- **9a related-language transfer:** <500 accuracy 0.789 → 0.895 at λ=0.3 (+10.6pp);
  500–5k peaks at 0.968 at λ=0.7 (+1.0pp); overall accuracy 0.960 → 0.947 at λ=0.3. For
  λ ≤ 0.3, accuracy on all three groups drops sharply (e.g. <500: 0.053 at λ=0.0).
- **9b script-average transfer:** overall accuracy stays 0.960–0.961 across λ ∈ [0.1, 1.0];
  <500 accuracy does not exceed 0.789 (baseline) at any λ<1.0 and falls to 0.526 at λ=0.1.
- Neither approach increases both overall and <500-group accuracy simultaneously in the
  tested range.

**Possible (unconfirmed) conclusion:** probability-space interpolation of EM-trained
per-language distributions toward a related-language or script-average distribution does not
jointly improve overall and very-low-resource accuracy in the configurations tested.

**Artifacts:** `outputs/tables/transfer_sweep.md`;
`outputs/figures/transfer_sweep.png`. **Detail:** `EXPERIMENTS.md` §9.

---

## Experiment 10 — Error analysis (signal for the pooling direction)

**Question:** Is there a qualitative trend in UniLID's errors that points to a focus for
improvement? Run 2026-06-24 via a 7-cut agent workflow over a 28,527-error stratified
sample of the full test set plus a per-token score decomposition and a weight-matrix audit
(scorer validated: `score(pred) >= score(true)` for 97.3% of recorded errors).

**Central finding:** UniLID's errors are dominated by the **under-fit low-resource tail
acting as false-positive attractors that steal predictions from the high-resource head.**
In every dominant confusion pair the low-resource sibling wins (eng->sco 48,620 forward vs
151 reverse, 322x). 84.7% of short-text errors route a high-resource truth to a
non-high-resource sink (true high-resource 64.5%, pred high-resource 13.2%). 86% of errors
predict a language ~30x rarer than the truth. The direct cause is the absent language prior
plus under-regularized EM vectors. The per-language smoothing floor is resource-tied
(`corr(floor, log10 count) = -0.966`, range -13.2 to -19.9), so small languages
under-penalize the unseen; 86.4% of the score gap toward the wrong language comes from short
non-content tokens (<=3-char subwords 51%, punctuation 20%), content words only 13.6%.
Whitespace/digit columns are the cleanest part of the matrix, not noisy magnets.

**Two attractor types with opposite pooling prescriptions:** (a) FLAT promiscuous magnets
(kzn, tly, vol, ido, mlt, qus) pulling from 25-111 unrelated languages, 0 true-label
appearances, 40% all-caps; all-caps is 62x over-represented in errors. Shrinking these is a
strict win. (b) TIGHT sibling sinks (glk<-fas, anp<-hin, wuu<-cmn); symmetric pooling would
blur them, so shrink only the low-resource member one-directionally. 62% of top-confusion
volume is equal-resource near-twins (ind/zsm, kin/run) where pooling is a blur risk.

**Ceiling:** ~4% clean-text mislabels, ~14% unverifiable content, ~20% arbitrary
macrolanguage splits (hbs/srp; arb->arz is 91% diacritized scripture, a diacritization
domain shift). fastText recovers 63.7% of UniLID's errors but is no better on the genuine
twins. So ~80-85% of the error budget is recoverable, and pooling must NOT be credited with
the twin share.

**Implication:** pooling must be gated by flatness and distance-to-confuser and evaluated
stratified (tail vs twins vs head) or aggregate macro-F1 can move the wrong way. Detailed
findings are in the project memory (`unilid-error-analysis-findings`).

**Artifacts:** `scratchpad/error_stats.json`, `scratchpad/errors_sample.jsonl` (analysis
inputs). **Status:** drives the hierarchical-pooling program (`EXPERIMENTS_PLAN.md` Exp 11-15).

---

## Experiment 43: clean re-measurement of the Apertus 131k branch; the verdict holds, the magnitude was overstated (2026-07-27, job 2911700)

Full-test baseline evaluation of `glotlid_apertus131k_fp64.unilid`, the model
retrained under the corrected trainer, against the 100k production model
(`outputs/tables/full_test_eval_131k_fp64.md`, 45,377,279 lines, 2h06m). This
replaces the Exp 29 comparison, which used a model containing one collapsed row.

| quantity | 100k | 131k corrupted (Exp 29) | 131k clean (this run) |
|---|---|---|---|
| within-stratum overall | 0.9292 | -0.0113 | **-0.0090** |
| within-stratum tail | 0.9132 | -0.0437 | **-0.0395** [-0.0476, -0.0322] |
| within-stratum magnets | 0.9138 | -0.0352 | **-0.0318** [-0.0386, -0.0251] |
| global per-language, all 1,940 | 0.9292 | 0.9179 | **0.9202** |
| global per-language, tail | 0.5618 | 0.4046 | **0.4269** |
| false positives into tail labels | 22,522 | 51,926 | **32,211** |
| accuracy | 0.9608 | +0.0004 | **+0.0055** |

**The branch verdict holds.** The 131k vocabulary is negative against the 100k
model on both metric views and on the balanced validation set (overall 0.9776
against 0.9811), with every stratum at or below the 100k model. Removing the
single collapsed row does not change the conclusion, so the Exp 29 reading stands:
with training data fixed, a larger shared vocabulary means more parameters per
language, flatter low-resource distributions, and more false-positive absorption.
Per-language vocabulary truncation remains the recorded counterfactual for any
future vocabulary work.

**The magnitude was overstated by the bug, as predicted.** Every gap narrows: the
tail deficit by about 10% relative, the overall deficit by 20%, and accuracy moves
from parity to +0.0055. The clearest number is the false-positive count into tail
labels: 51,926 corrupted, 32,211 clean. Exp 30's counterfactual, computed by
deleting the collapsed language's lines from the corrupted predictions, predicted
32,161. The retrained model measured 32,211, within 0.2% of that estimate, which
is a strong independent check on the whole diagnostic chain from Exp 30 through
Exp 42.

**Status of the branch.** Closed as a drop-in replacement for the production
model, now on clean evidence rather than on a measurement contaminated by a
training bug. The retrained models remain available for hybrid analyses (the
Exp 30 finding that the 131k model fixes 42% of the 100k model's errors, including
35% of its Indic-script errors, was never in question and is unaffected by the
bug).

## Experiment 42: both Apertus models retrained under the corrected trainer; the fix works and is correctly scoped (2026-07-27, jobs 2903767, 2903768)

The double-precision E-step and the hard non-finite-count check were adopted into
the sentencepiece fork (commits d0208d9, c5921a2) and the installed binary
replaced. Both Apertus models were retrained in full rather than repaired per
language, so each model has single-provenance weights. Row-level comparison of
each model against its retrain (`outputs/tables/fp64_retrain_check.md`,
`analysis/fp64_retrain_check.py`; read from the packed matrices, no scoring):

| model | degenerate rows before | after | repaired | rows moved >1 nat | >5 nats |
|---|---|---|---|---|---|
| 131k | 18 | 17 | azj_Latn | 20 of 1,940 | 5 |
| 200k | 17 | 17 | none flagged | 18 of 1,940 | 7 |

**The collapse is repaired.** azj_Latn at 131k goes from 7 entries above the row
minimum (entropy 1.609 nats) to 22,704 (3.025). At 200k it goes from 1,798 to
31,210 (entropy 2.473 to 3.000), which confirms the Exp 41 finding that the
200k row was partially collapsed even though it never crossed the degeneracy
threshold: the corrupted 200k model was carrying a damaged Azerbaijani model
that no gate would have caught.

**The fix is correctly scoped.** The 17 minority-script rows (Syriac, Cherokee,
Coptic, Cree syllabics, Gothic, Kali, Limbu, Lisu, Meetei, Mongolian-script)
remain degenerate in both retrains, with unchanged above-minimum counts. This is
the confirmation that the two phenomena were correctly separated in Exp 35: those
rows are a deterministic vocabulary-coverage property (the Apertus inventories
carry no multi-byte pieces for those scripts), not a numerical failure, and a
precision fix cannot and should not change them. No row became newly degenerate.

**The rest of the model is nearly untouched.** Only about 1% of rows changed by
more than 1 nat at their largest entry, and mean row entropy moves by 0.0006 nats
(131k) and 0.0003 (200k). The languages that moved are the long-line corpora the
diagnosis predicted: quc_Latn (max change 9.43 nats), pcm_Latn, fas_Arab,
mam_Latn, bod_Tibt. Controls behaved as expected: tat_Latn and fra_Latn are
unchanged to three decimals in entropy.

**Consequence for the branch verdict.** Exp 29 concluded the 131k base is
negative on both metric views, and Exp 30 showed that a single collapsed row
(azj_Latn) carried about two thirds of the false-positive increase into tail
labels. That comparison used the corrupted model. The clean comparison is now
possible and is the pending next measurement: a full-test baseline evaluation of
`glotlid_apertus131k_fp64.unilid` against the 100k production model. Until it is
run, the Exp 29 verdict stands as a measurement of the corrupted model, and its
magnitude is known to be overstated.

## Experiment 41: the fixed-vocabulary EM bug diagnosed, verified, and patched in scratch (2026-07-26, two-agent investigation)

**Defect, established by faithful reimplementation and dual builds (an unpatched
rebuild reproduces the installed binary's trained scores bit-for-bit).** Two
interacting causes:
1. Primary, in UPSTREAM code the fork inherited (`src/unigram_model.cc`,
   Lattice::ForwardAlgorithm / BackwardAlgorithm / PopulateMarginal): the E-step's
   forward-backward accumulates in 32-bit floats. On very long sentences the
   rounding error breaks the mathematical identity that a lattice node's
   log-posterior (alpha + score + beta - Z) is at most 0: on azj's trigger line it
   reaches +351, so expected counts blow up (measured 1e157 against the correct
   56,699). Upstream never encounters this because its default
   --max_sentence_length=4192 skips long lines; the pipeline passes
   --max_sentence_length=1000000 (`language_specific_trainer.py:163`).
2. Secondary, in the fork's rewritten M-step (`unigram_model_trainer.cc:435`):
   `if (!isfinite(ni)) ni = 0.0` silently zeroes overflowed counts, which are
   exactly the most frequent pieces (1,881 pieces overflow on the trigger). This
   converts a numerical explosion into a plausible-looking model: one garbage
   surviving piece near probability 1, everything else at the Dirichlet-prior
   floor. Every logged number of the azj run is reproduced arithmetically by this
   account, including the frozen L1 deltas and the identity of 'ĠMun' (an
   arbitrary piece whose overflowed total landed just under float32's maximum;
   unrelated to text frequency, consistent with " Mun" occurring 33 times in
   100k lines).

**Trigger, established by 14-run corpus bisection:** line 81,302 of azj's corpus,
a 142,136-byte casualty-roll line, the longest line in all 1,940 corpora (next:
quc_Latn at ~55k characters).

**The graded-corruption finding (revises Exp 35's reading).** The failure is not
binary. Measured float32-vs-float64 count ratios grow with line length (about 3%
error at 2,000 characters, 7% at 4,000, unbounded beyond); quc_Latn's DELIVERED
131k model has 188 pieces off by more than 5 nats; and azj at 200k, previously
described as healthy, is PARTIALLY collapsed (its EM sat in the trap for eight
iterations and escaped at iteration 18; 1,798 above-minimum entries against
3,000-40,000 for comparable languages). At least 94 corpora contain lines above
the 4,192-character upstream cap (33 above 10,000) and carry measurable
corruption in the Apertus-branch models. The production 100k model is NOT
affected: it was not trained through this code path (recovered from the original
UniLID release), and no main-line experiment (Exp 1-40) consumes fork-trained
weights; the corruption is confined to the Apertus branch, further caveating the
Exp 15/29 magnitudes.

**Fix, verified [ADOPTED 2026-07-27 on user instruction: fork commits d0208d9
and c5921a2, installed binary replaced, both models retrained; see Exp 42. The
paragraph below describes its state at diagnosis time].** Patch preserved at
`patches/sentencepiece_fp64_estep.patch`: compute the trainer's
forward-backward in double precision; leave the float paths in place for
inference so shipped models behave bit-identically. The patch changes only the
precision at which the E-step's defining identity is evaluated: vocabulary,
iteration count, Dirichlet prior, objective, and M-step are untouched, so it
restores the documented method rather than departing from it. Verified: patched
azj-131k converges monotonically to a healthy model (objective 229.8, entropy
3.155, ~22,400 above-floor entries, top pieces ordinary Azerbaijani units);
1,846 of 1,940 corpora change at or below the 1e-3-nat level; the recommended
companion change makes the isfinite guard a hard failure so any future overflow
aborts instead of silently zeroing. If the Apertus branch is ever revived, the
33 long-line corpora need retraining under the patch.

## Experiment 40: oracle upper bound over the carried set (2026-07-25)

Per-language maximum F1 over the seven Exp 38 configurations on the held-out
remainder (`outputs/tables/oracle_bound.md`, `analysis/oracle_bound.py`): a
perfect per-language method chooser would reach 0.9525 overall against 0.9334 for
the best single configuration (+0.0191). The headroom sits almost entirely in the
tail (0.7061 vs 0.6337, +0.0724) and flat_magnets (0.6344 vs 0.5345, +0.0998);
head (+0.0008) and twins (+0.0038) are close to saturated by single methods. An
upper bound on method combination, not an achievable configuration: any real
chooser must pick per language from training-side information only, and its gap
to this bound measures the chooser's quality.

## Experiment 39: CommonLID out-of-domain check of the carried leaders (2026-07-25, job 2898246)

Web-domain evaluation (373,230 lines, Exp 12's macro-aware accuracy convention;
baseline gate reproduced the recorded 0.8452 exactly;
`outputs/tables/commonlid_carried.md`): floor-21 +0.0040, gt_margin_adaptive
+0.0070 over baseline. The adaptive gate ran with its training-calibrated tau
values unchanged and fired on 9,886 lines: the no-refitting portability property
holds mechanically, and the ordering matches the primary quantity (adaptive ahead
of floor-21). Both gains are modest against the learned bias's +0.0427 (Exp 12),
which fits the web domain's natural traffic prior but remains rejected for
adoption on per-language grounds (Exp 25/31).

**Per-tag extension (job 2903415; both baseline gates reproduced: accuracy 0.8452,
tag-level macro-F1 0.7228; per-line predictions persisted to
`outputs/diagnostic/commonlid_carried_preds.npz`).** Under the objective-consistent
metric, tag-level macro-averaged F1, both carried leaders are slightly NEGATIVE out
of domain: floor-21 0.7181 (-0.0046), gt_margin_adaptive 0.7167 (-0.0061), while
their line-weighted accuracies are positive. Reading: CommonLID's 109 tags are
predominantly larger languages, so the tail labels these methods repair are mostly
absent from the tag set, while the methods' recall costs on mid-size tags are
visible. Consequence for the paper: the tail-precision gains do not transfer to
CommonLID because CommonLID cannot see them; out-of-domain claims about the
carried methods should be scoped to what the 109-tag set measures, and the
in-domain primary quantity (Exp 38) remains the evaluation where the methods'
purpose is visible.

## Experiment 38: the carried-forward set under the primary quantity (2026-07-25)

Per-language F1 on the held-out remainder (45,004,014 lines outside draws 101 and
201, natural distribution, all false positives counted), averaged unweighted over
the 1,940 languages (`outputs/tables/carried_set_comparison.md`, per-language
values in `outputs/diagnostic/carried_set_per_lang_f1.csv`):

| config | all 1,940 | tail | lowmid | head | flat_magnet | twin |
|---|---|---|---|---|---|---|
| gt_margin_adaptive | 0.9334 | 0.4620 | 0.9567 | 0.9593 | 0.4206 | 0.8917 |
| floor21 | 0.9309 | 0.6337 | 0.9352 | 0.9590 | 0.5345 | 0.8854 |
| freq_prior | 0.9264 | 0.4816 | 0.9399 | 0.9605 | 0.4168 | 0.8892 |
| learned_bias | 0.9254 | 0.3736 | 0.9405 | 0.9696 | 0.3562 | 0.9079 |
| margin_q5_head | 0.9215 | 0.5321 | 0.9267 | 0.9590 | 0.4185 | 0.8856 |
| margin_q5 | 0.9201 | 0.5125 | 0.9256 | 0.9592 | 0.4038 | 0.8847 |
| baseline | 0.9121 | 0.3382 | 0.9267 | 0.9593 | 0.2890 | 0.8856 |

The set is complementary, not redundant: gt_margin_adaptive leads overall through
the lowmid band (0.9567), floor-21 leads tail (0.6337) and flat_magnets (0.5345),
learned_bias leads head (0.9696) and twins (0.9079). Strict per-language leader
counts (a configuration is counted only where it is the unique best; 326 of 1,940
languages tie at the maximum and are excluded): learned_bias 602, gt_margin_adaptive
587, freq_prior 211, floor21 173, margin_q5_head 21, baseline 10, margin_q5 10. No
configuration dominates any other across all groups, supporting the keep-and-
explore decision. An oracle upper bound (per-language best over the set) is a
natural next measurement if method combination returns to the agenda.

## Experiment 37: the azj_Latn collapse is deterministic and numerically diagnosed (2026-07-25, isolated re-run)

**Procedure:** azj_Latn's per-language training re-run in isolation (same corpus
file, same 131k base tokenizer, same trainer call as job 2883222; output to a
scratch directory, the packed model untouched; recipe in the chronological log).

**Result: byte-identical reproduction.** 7 entries above the row minimum,
entropy 1.609 nats, same EM trace. The collapse is DETERMINISTIC, not stochastic:
across the whole project there is now zero evidence of random EM degeneration.
The same corpus file produced a healthy azj row in the 200k retrain, so the
trigger is the corpus-plus-131k-seed-vocabulary pair specifically.

**Diagnosis.** The 7 surviving vocabulary entries are the four special tokens
plus 'ĠMun' (all at probability 0.2) and 'gin'/'ayar' at trace level; the corpus
is ordinary Azerbaijani prose. The EM objective WORSENS from its start (374 to
2746 at sub-iteration 1, total distribution replacement, then frozen at machine
zero deltas): a likelihood that deteriorates and freezes is a numerical breakdown
in the fixed-vocab EM fork (cimeister/sentencepiece, branch fixed-vocab-em), not
a legitimate optimum. Root cause inside the C++ E/M steps is an open item for the
fork; the one-command reproduction is the handoff.

## Experiment 36: gt_margin_adaptive judged; both pre-run predictions confirmed; floor-21 retains selection 5/5 (2026-07-25, job 2895821)

**Verdict: ELIGIBLE, flagged (ota_Arab only); not selected.** Both on-record
predictions confirmed: reassignments 325,546 (down from round 3's 407,562;
207,241 to the true label) and ota_Arab remains the sole flag (its FPs are
high-margin weight-side flips no quantile catches; mechanism unchanged from
Exp 34). Rows: balanced-val 0.9798 / 0.9580 / 0.9503 / 0.9408 / 0.9808
(overall/tail/magnets/twins/head); veto overall 0.9334, the best of all eleven
configurations tested (floor-21 0.9309).

**Selection.** The natural-track ranking margin over floor-21 is 0.0002 of
balanced-val overall. Stability check across all five val draws (the instrument
built for exactly this): floor-21 leads 5/5 with margins 0.0000-0.0002,
consistently signed, so floor-21 retains selection under the standing rule and
the margin is not draw noise. The two candidates are near-equivalent on the
selection instrument while gt_margin_adaptive is better on the veto and on
balanced-val tail (0.9580 vs 0.8942). Open rule-design question for the user,
recorded not decided: whether near-ties on the selection instrument (margin
below, say, 0.001) should break on the veto overall, which would select
gt_margin_adaptive here.

**Family arc closed (four rounds).** gt_min (rejected, FP explosion) ->
runner-up gate (szy mechanism) -> head-targeted (lowmid class) -> 100k bar
(barely-head class closed, ota residue) -> N-adaptive strength (boundary cliff
removed, val cost recovered from 0.9744 to 0.9798). Each rejection produced the
next candidate's mechanism; the family ends with two eligible compositions and
the reassignment law as a documented result.

## Experiment 35: the EM-degeneracy question bounded (2026-07-25, analysis)

**User concern:** if per-language EM occasionally degenerates, previous "failures"
are in question. **Finding: the concern is bounded to one language-run.** Scan of
all three packed models (`analysis/degeneracy_scan.py`,
`outputs/tables/degenerate_rows.md`, threshold 100 estimated tokens):
- 100k model: 0 degenerate rows. Every adoption verdict and every method
  experiment (Exp 1-28, 31-34) rests on this model; none is affected.
- 200k: 17 flagged rows; 131k: 18. The two sets are near-identical (Syriac,
  Cherokee, Coptic, Cree syllabics, Gothic, Kali, Limbu, Lisu, Meetei, Mongolian):
  this class is DETERMINISTIC vocabulary coverage, not EM instability. The Apertus
  BPE inventory holds no multi-byte merges for these scripts, so byte-level
  pretokenized text exposes only the ~60-90 single-byte pieces and the fixed-vocab
  EM correctly estimates only those (csw's EM log converges normally, L1 delta
  0.4 to 0.03). Unique-script members are harmless (cop/lis/chr at F1 ~1.0); the
  six Cree-syllabics languages share the script and byte-only pieces cannot
  discriminate within it (csw 0.088, cwd 0.542). The same class was present,
  undetected, in the Exp 15 200k retrain: its tail deficit includes this coverage
  effect, a recorded caveat on the Exp 15 magnitude (direction unchanged).
- azj_Latn at 131k is the ONLY anomaly outside that class in 3,880 Apertus
  per-language EM runs (absent from the 200k and 100k flagged sets); the adjacent
  EM trace freezes at machine-zero deltas from sub-iteration 2, a genuine collapse.
  Open item: re-run azj's EM in isolation to test determinism, and attribute the
  trace conclusively (batched logs interleave languages).
  [Superseded in part by Exp 41 (2026-07-26): the isolated re-run confirmed
  determinism (Exp 37) and the full diagnosis found the failure is GRADED, not
  binary: azj-at-200k is partially collapsed, and at least 94 Apertus-branch
  corpora with lines above 4,192 characters carry measurable corruption below the
  degeneracy scan's threshold. The "single anomaly" framing understated the
  exposure; the class framing (deterministic, input-specific, absent from the
  100k production model) stands.]

**Process change:** `degeneracy_scan.py` is the post-training gate for any future
retrain: run it on every new .unilid before evaluation; flagged rows outside the
known unique-script exemption block the evaluation.

**Pre-registration: candidate `gt_margin_adaptive` (recorded before any run;
user-requested direction).** Identical to gt_margin_all_100k except the gate
strength adapts to training size: the calibration quantile becomes
q_L = MARGIN_Q * (1 - min(N_L, HEAD_N)/HEAD_N), so suppression strength decays
linearly from q=5 at N=0 to zero at the head boundary. No new constants (reuses
MARGIN_Q and HEAD_N); removes the threshold cliff that created boundary victims;
gated set and the 100k target bar unchanged. One candidate; both tracks.

## Experiment 34: gt_margin_all_100k, the first fully eligible gt-family candidate; floor-21 retains selection (2026-07-25, job 2895683)

**Verdict: ELIGIBLE, flagged (single outlier ota_Arab); not selected.** Build:
407,562 reassignments (251,419 to the true label, the highest recovery of the
family; 94,462 below-tau lines kept for lack of a 100k-bar candidate in the
top-5). Rows: balanced-val 0.9744 / 0.9579 / 0.9479 / 0.9392 / 0.9808
(overall/tail/magnets/twins/head); veto overall 0.9330 (top tier, floor-21
0.9309), tail global F1 0.4621, FPs into tail 28,743. Raising the target bar
eliminated the barely-head collapses entirely (aba/bam/llb/twx all recover), and
the natural-track ranking still selects floor-21 (balanced-val overall 0.9800 vs
0.9744: the gate's suppression cost on balanced data). Not a uniform-track passer
(val overall must improve there).

**ota_Arab dig-in (required by the flag).** ota (N=674, tail, Ottoman Turkish)
receives 395 new false positives, 295 from fas_Arab, 0 recall loss. It is NOT a
reassignment receiver (below the target bar): these are gt-weight-side flips whose
margins exceed tau_ota, i.e. confidently wrong under the GT floors. A quantile
gate cannot catch high-margin flips without destroying recall; the fas/ota pair
(Persian-influenced Ottoman orthography) is the gt-side residue class, distinct
from the reassignment law of Exp 31/33.

**Round and family summary.** Three margin-composition rounds converged: the
reassignment law (burden relocates to the lowest-capacity valid target) is closed
by the 100k bar at the cost of one gt-side residual flag. Final standings under
the amended rule: natural-traffic champion floor-21 (selection margin 0.0056 of
balanced-val overall over the fully-repaired composition); uniform-prior champion
gt_min (flagged, mev/sbs dig-ins on record); gt_margin_all_100k stands as the
eligible composition demonstrating that the two-correction decomposition
(within-language calibration + FP-side repair) can be made per-language-safe.

## Experiment 33: gt_margin_all judged; reassignment burden relocates to barely-head sinks (2026-07-25, job 2895566)

**Verdict: REJECTED on both tracks** (val overall drop > 0.01; 4 supported
collapses, worst -0.3211), with the best natural-veto aggregates of any candidate
tested: overall 0.9121 to 0.9331 (above floor-21's 0.9309), tail global F1 0.5373,
lowmid global F1 0.9267 to 0.9554, FPs into lowmid 451,042 to 139,506, FPs into
tail 19,390. Build: 461,605 reassignments over 1,080 gated labels, 235,421 to the
true label (`outputs/tables/gt_margin_all_build.md`).

**Mechanism (new, third of its kind).** The four collapsed languages (aba_Latn
N=18,107; bam_Latn N=18,697; llb_Latn N=25,228; twx_Latn N=26,573) are all
barely-head languages just above the 18k threshold: valid reassignment targets
that received the relocated burden. Pure precision loss (llb precision 0.878 to
0.437; recall unchanged or improved in all four). Across three rounds the same
law: runner-up targeting moved FP burden to small relatives (szy), tail-only
gating moved it to lowmid sinks (arq/skr/llb/vmk), all-label gating moves it to
barely-head sinks. Reassignment relocates FP burden to the lowest-capacity valid
target near the cluster.

**Pre-registration: round-3 candidate `gt_margin_all_100k` (recorded before any
run).** Identical to gt_margin_all except the reassignment-target bar rises from
HEAD_N=18,000 to RES_CAP=100,000 (the established resource cap; 98.9% of the
original false positives come from sources with median N = 100,000, so the
returned lines belong at top-resource labels). Gated set unchanged (N < 18,000);
if the top-5 holds no candidate at the bar, keep the gt_min prediction. One
candidate; both tracks; round closes on its verdict.

## Experiment 32: victim dig-ins, a degenerate-row finding in the 131k model, and the round-2 pre-registration (2026-07-24)

**Dig-ins required by the outlier-tolerant clause
(`outputs/tables/victim_digins.md`, `analysis/victim_digin.py`).** One mechanism
covers every flagged or collapsed victim: false-positive INFLOW at a non-head
label, never recall loss (recall lost is 0-9 lines in every case). llb_Latn gains
1,356 FPs under the learned bias (many small Bantu sources whose suppression
redirects their lines into the mid-sized Bantu sink) and 2,695 under gt_margin
(ndo/kua/bem/nya/zul); arq_Arab gains 765 from the Arabic cluster (ary/arb/fas);
skr_Arab 636 from pnb/urd; vmk_Latn 463 from vmw/ngl; sbs/mev on the balanced draw
gain scattered small-language FPs. Conclusion: the flagged outliers and the
rejected class share one addressable failure mode, and a margin gate that defends
ALL non-head labels addresses every observed case.

**Degenerate rows in the 131k model.** azj_Latn's 131k row collapsed in EM:
entropy 1.609 nats, 131,065 of 131,072 entries at the floor (about 7 estimated
tokens), recall 0.0000; its 229k test lines scatter to tly (161,886!), crh, tat,
tur. A matrix scan finds 18 rows with fewer than 100 estimated tokens (0 such rows
in the 100k model); most are unique-script languages where this is harmless
(cop/lis/chr at F1 ~1.0), but azj_Latn (head, Latin) and the Cree-syllabics
cluster (csw 0.088, cwd 0.542) are genuine per-language training failures.
Counterfactual with azj-true lines removed: 131k FPs into tail fall from 51,926 to
32,161 (the single failed row explains about two thirds of the FP increase);
tail global F1 gap narrows to 0.5627 vs 0.4258 and overall to 0.9287 vs 0.9196.
The Exp 29 verdict stands (the branch loses every aggregate even without azj) but
its magnitude was overstated by one EM failure. Repair path if the branch is ever
revived: delete the affected langspec files and re-run per-language EM with
--skip-existing-langs (independent per language), then repack; not run now.

**Pre-registration: round-2 candidate `gt_margin_all` (recorded before any run).**
gt_min weights plus the margin gate extended to ALL predicted labels with
N < 18,000 (tail and lowmid), tau recalibrated under gt_min per gated language on
its own train lines, head-targeted reassignment, all constants unchanged
(MARGIN_Q=5, MIN_CALIB_LINES=200 with exclusion logged, CALIB_MAX=2000,
CALIB_SEED=0, TOPK_MARGIN=5, HEAD_N=18,000). Directly motivated by the dig-in
mechanism. One candidate; judged on both tracks under the amended rule.

## Experiment 31: amended gating, dual-track verdicts, and the gt_margin composition (2026-07-24)

**Gating amendments** (user-invited reconsideration; EXPERIMENTAL_SETUP.md
"Amendments"): (B)-overall softened to a bounded drop; a uniform-prior track added
(`passes_uniform` selection on balanced val, collapse-checked confirmation on the
balanced test draw for the single track-selected candidate); ITERATE verdict lane.
Delta-reviewed: verdict-neutral for the first round; select-on-val/confirm-on-test
discipline preserved with the multiplicity count recorded.

**Dual-track outcomes (final, under the outlier-tolerant collapse clause added
the same day, amendment 4).**
- Natural-traffic track: floor-21 remains champion. learned_bias returns to
  eligible, flagged for the llb_Latn dig-in; ranking unchanged (floor-21 0.9800
  vs 0.9799).
- Uniform-prior track: **gt_min is the champion, flagged** with two required
  dig-ins from its balanced-test confirmation (mev_Latn -0.172 n=12, sbs_Latn
  -0.182 n=12). Under the first, outlier-intolerant clause it had been blocked
  outright; the amendment converts that block into targeted investigations, which
  is the intended semantics: the bound catches class-level harm, and two flagged
  outliers are not a class.

**gt_margin (pre-registered composition, built and judged; reviewed, no defects;
`outputs/tables/gt_margin_build.md`).** The recalibrated gate reassigns 60,320 of
gt_min's 86,924 tail predictions (28,533 to the true label; 22 languages under the
calibration floor). It repairs gt_min's headline pathologies: FPs into tail fall
from 79,113 to 19,390 (below the baseline's 22,404) and supported clause-C
collapses fall from 12 to 4. It passes the natural track's stages (A) and (B)
outright. Verdicts:
- Natural track: REJECTED on clause (C) alone: arq_Arab -0.131 (n=271), llb_Latn
  -0.206 (n=4,181), skr_Arab -0.192 (n=157), vmk_Latn -0.187 (n=93). These are
  LOWMID victims of the gt weight side; the gate defends only tail labels, so
  mid-band languages with dominant neighbors inherit gt_min's damage. llb_Latn is
  now a chronic victim across method families (learned bias -0.113, gt family
  -0.206).
- Uniform track: passes selection (0.9827, ranked under gt_min); a recorded second
  look at the test draw (multiplicity noted) shows it would also fail that
  confirmation (knx_Latn -0.111 n=15, sbs_Latn -0.107 n=12, sdc_Latn -0.160 n=30).

**Round closed per pre-registration.** Recorded mechanism for the next round: the
gt-family damage that survives composition is concentrated in lowmid languages
under dominant neighbors (the arq/skr/llb/vmk pattern, the same population as the
tight_lowres category), and the decision-layer defense must cover those labels,
not only tail: either extend the margin gate to all non-head predicted labels
(tau for every language with sufficient train data) or repair the gt mid-band
redistribution at the weight level. Not pre-registered; next round's candidate.

## Pre-registration: composition candidate gt_margin (recorded 2026-07-24 before any run)

Config `gt_margin`: the gt_min weight matrix (Exp 28) combined with the
head-targeted margin gate (Exp 26 rule), with tau_L RECALIBRATED under the gt_min
matrix (margins change when weights change; the Exp 26 tau values do not transfer).
All constants unchanged from their pre-registrations: MARGIN_Q=5,
MIN_CALIB_LINES=200, CALIB_MAX=2000, CALIB_SEED=0, TOPK_MARGIN=5, HEAD_N=18,000,
one-sided-min GT. Rationale: Exp 28 showed gt_min repairs the within-language
ranking (recall) and the margin gate repairs FP inflow (precision); this is the
first candidate that separates the two corrections explicitly. Judged on both
tracks of the amended rule. One candidate; if it fails, the failure mechanism is
recorded and the round closes.

## Experiment 30: the 131k does not repeat the baseline's errors; its regression is concentrated (2026-07-24, analysis)

**Question (user):** does the 131k model repeat exactly the same errors as the 100k
baseline? **No.** Line-exact overlap (`outputs/tables/error_overlap_131k.md`,
`analysis/error_overlap_131k.py`, accuracy gates passed):
- Of the 100k's 1,779,499 errors, 57.7% recur under 131k, and only 66.2% of those
  pick the same wrong label. The 131k fixes 753,463 errors (42.3%) and introduces
  733,388 (net -20,075; accuracy +0.0004).
- The tokenizer's documented strengths show up: 35% of the 100k's Indic-script
  errors are fixed (net positive there), and the improvement list is tail/lowmid
  heavy (syl_Latn +0.381, pwn_Latn +0.258, tig_Ethi +0.248, lad_Hebr +0.241,
  tcy_Knda +0.198). 190 languages improve by more than 0.01.
- But 403 languages regress, and the tail-FP explosion is CONCENTRATED: the single
  pair tat_Latn <- azj_Latn carries 17,603 false positives (azj_Latn, a head
  language and tat's close Turkic relative, collapses to F1 0.006 from 0.999),
  accounting for over half of the FP increase (51,926 vs 22,522). The remaining
  structure mirrors the baseline's known pairs (pnt<-ell 2,810, sbs<-Bantu,
  mrq<-tah).

**Reading:** the 131k base is a different error trade, not a uniform degradation:
real wins where the tokenizer adds coverage, broad small-language losses from
flatter distributions, plus one pathological relative pair (azj/tat) that a repair
layer (floor edit or margin gate on that model) or an EM inspection of tat's 131k
row might address. This softens the Exp 29 discontinuation reading from "the branch
is dead" to "the branch loses on net as a drop-in but contains recoverable
structure"; the discontinuation recommendation stands for the drop-in use, and the
131k memmap remains available for hybrid analyses.

## Experiment 29: Apertus 131k multilingual base does not reverse the vocab-size regression (2026-07-24, jobs 2883222 + 2885941)

**Hypothesis:** the 200k retrain's tail regression (-3.4pp, Exp 15) came from
vocabulary allocation, so `preliminary_mul` (131,072; documented in the tokenizer
project as highest compression on Indic, Chinese, and the low-resource tail) should
reverse it.

**Result: refuted; the regression tracks vocabulary size, not allocation.** Retrain
completed in one 9.8 h window (all 1,940 languages, standard setup, corpus split
reused; `glotlid_apertus131k.unilid`). Full-test b=0 baseline vs the 100k baseline
(`outputs/tables/full_test_eval_131k.md`):
- Within-stratum: overall -0.0113, tail -0.0437 [CI -0.0515, -0.0371], magnets
  -0.0352, twins -0.0044, head -0.0019; accuracy +0.0004.
- Global view: tail mean F1 0.5618 to 0.4046; false positives into tail labels
  22,522 to 51,926 (2.3x); flat_magnet 0.4716 to 0.3551; every group lower.
- Balanced val: overall 0.9766 vs 0.9811, tail 0.8679 vs 0.9170.

Both objectives agree, so this is not metric-conditional. Reading: with the training
data fixed, a larger vocabulary means more parameters per language and flatter
low-resource distributions, so tail models under-fit harder and their labels take
more head traffic (the FP doubling is the Exp 24 absorption mechanism amplified).
Better tail compression in the tokenizer does not compensate; the 131k regression
(-0.0437 within-stratum tail) is in line with the 200k's (-3.4pp on the older eval).
Per the plan's branch criterion the 131k base should not be continued;
recommendation recorded, decision with the user. A retrain-side counterfactual worth
noting for any future vocab work: per-language vocabulary truncation (each language
keeping only its top-k pieces) would decouple shared-vocab size from per-language
parameter count.

**Artifacts:** `outputs/tables/full_test_eval_131k.md`,
`outputs/diagnostic/full_test_131k_per_lang_prf.csv`, model + memmaps on scratch
(`glotlid_apertus131k.unilid`, `full_test_eval_131k/`). Scripts:
`slurm_apertus_train_131k.sh`, `analysis/full_test_eval_131k.py` (both reviewed).

## Experiment 28: gt_min judged; per-language honesty is not the fix for a between-language problem (2026-07-24, job 2884210)

**Verdict: REJECTED by the adoption rule; floor-21 remains selected.** Single
pre-registered candidate, no sweep. Full tables in `outputs/tables/full_test_gt.md`
and `two_sided_selection.md`.

**The two views split harder than for any previous config.**
- Selection view (balanced val): gt_min is the best configuration measured on this
  protocol: overall 0.9841 (baseline 0.9811), tail 0.9769 (+0.0599), magnets 0.9688
  (+0.0514), twins/head flat. Full-test within-stratum: tail +0.0656
  [CI +0.0603, +0.0729], magnets +0.0528 [+0.0478, +0.0588].
- Veto view (natural traffic): false positives into tail labels rise from 22,404 to
  79,113 (3.5x); tail global mean F1 drops 0.3382 to 0.2950 despite tail recall
  rising 0.8664 to 0.9675; overall global macro-F1 drops; 12 languages above the
  support floor lose more than 0.1 (worst -0.2123).

**Mechanism.** Exp 27 showed every floor is individually overstated (~10x), and
correcting each language against its own data fixes the ranking on genuinely
low-resource text: that is the recall/selection-view gain. But the argmax competes
ACROSS languages, and the honest per-language estimates preserve and even widen the
resource-tied floor gap (median tail-to-head plateau-mass ratio grows from ~87x to
~115x), so out-of-model head text flows into tail labels harder than before. The
floor pathology is a between-language externality, not per-language miscalibration.
Floor-21 works on natural traffic precisely because it is NOT per-language-honest:
one shared constant flattens the cross-language gap. gt_min and floor-21 are exact
mirror images: within-language calibration buys recall and pays precision;
cross-language equalization buys precision and pays recall.

**What this sharpens.**
1. The open objective decision now has concrete champions: under the uniform-prior
   view gt_min is the best configuration ever tested here; under the natural-traffic
   view it is disqualified and floor-21 stands. The adoption rule encodes the
   natural-traffic veto, so floor-21 remains the provisionally adopted config.
2. Next-round composition hypothesis (recorded, NOT run, not yet pre-registered):
   combine the two mechanisms explicitly, e.g. cross-language floor equalization at
   the GT-implied level (equalize per-token floor values across languages, with the
   shared level set from the GT counts instead of a swept constant), or gt_min plus
   the head-targeted margin gate (Exp 26) with tau recalibrated under the gt_min
   matrix. Either would be the first candidate family that separates the
   within-language and between-language corrections.

**Artifacts:** `outputs/tables/full_test_gt.md`, updated `two_sided_selection.md`,
`pred_gt_min.npy` + `fingerprint_gt.json` on the full-test scratch dir. Script
`analysis/full_test_gt.py` (reviewed pre-run, no defects).

## Experiment 27: Good-Turing counts, and the floor overstates unseen mass everywhere (2026-07-23, job 2883714)

**Hypothesis:** the emergent floor plateau misstates each language's unseen-token
probability; the Good-Turing plug-in n1/T from the language's own Viterbi counts
gives the calibrated value.

**Result (`outputs/diagnostic/gt_counts.csv`, 1,940 languages, 2.43B tokens
counted; three languages re-verified by hand end-to-end, exact match):** the
emergent plateau OVERSTATES unseen mass for every language without exception.
Exact GT would raise the plateau for 0/1,940 languages, so the pre-registered
one-sided-min rule coincides with exact GT on this model. Tail median: current
plateau mass 8.7e-2 against a GT target of 9.7e-3 (9x); head median 1.0e-3 against
8.4e-5 (12x). The gt_min matrix (built and gate-checked; floor drops -0.67 to
-9.18 nats, mean -3.08) is the per-language-calibrated version of the mechanism
floor-21 applied as one global constant. No language has n1=0; no tuned constant
anywhere (n1, T from own data; 0.2 is the fixed non-special budget).

**Pending:** the full-test scoring pass (`analysis/full_test_gt.py`, in review,
then SLURM) and the adoption verdict via the two-sided report. gt_min must beat
floor-21 (veto tail F1 0.6337) to displace it.

## Experiment 26: margin diagnostic, viable (2026-07-23, job 2883715)

**Hypothesis:** on lines the baseline routes into a tail label, the score gap between
the winning tail language and the runner-up is separable from the gaps on genuine
lines of that language, so a per-language threshold tau_L calibrated on L's OWN
training data (5th percentile of self-won train margins; MIN_CALIB_LINES=200
exclusion) can remove false positives at a bounded recall cost.

**Result: viable.** Aggregate over the 96 tail languages
(`outputs/tables/margin_diagnostic.md`, tau values in
`outputs/diagnostic/tau_per_lang.csv`):
- 17,299 of the 22,522 FP-into-tail lines fall below tau (76.8% catch rate); 5,413 of
  the caught lines (31.3%) have the true label as runner-up and would be recovered
  outright by reassignment.
- Test-side genuine suppression: 474 of 7,084 self-won true-tail lines (6.7%,
  against the 5% train-side bound; the gap is ordinary train-test shift). Only 53
  suppressed lines have another tail language as runner-up, so there is no
  tail-to-tail cascade.
- Per-language margin AUC (genuine train vs FP) is 0.90-0.9998 for the large
  receivers: sbs_Latn 0.9022 (catch 0.588), pnt_Grek 0.9409 (catch 0.738, recovery
  0.741, consistent with the Exp 25 audit finding that the residual is standard
  Greek with ell as runner-up), mrq_Latn 0.9763, pwn_Latn 0.999, arb_Latn 0.9998.
- 26 languages are excluded from gating (under 200 scoreable self-won train lines;
  listed in the report); they keep baseline behavior.

**Candidate pass 1, `margin_q5` (reassign to the runner-up): REJECTED on clause (C),
with the failure mechanism identified.** Build (`analysis/full_test_margin.py`,
login node, top-1 agreement 1.0000, 17,773 reassignments = 17,299 caught FPs + 474
suppressions, exactly matching the diagnostic): passed stage A (val tail drop 0.0281
inside the widened 0.03) and stage B with the largest FP reduction of any config
(veto FPs into tail 22,404 to 6,594; tail global F1 +0.1743), but one language above
the support floor collapses: szy_Latn -0.107 (n=175). Mechanism (verified from the
memmaps): szy_Latn receives 86 new false positives, 82 of them pwn_Latn's caught
lines handed to their runner-up (szy is pwn's Formosan neighbor), with head true
labels (ind/zsm/hbs). Globally 7,582 of the 17,773 reassignments land on languages
with N < 18,000 and 1,465 on tail languages: runner-up reassignment moves FP burden
onto other precision-fragile labels instead of returning it to the head sources that
produced 98.9% of the FPs (Exp 24).

**Pre-registered follow-up (recorded 2026-07-23 before the run), config
`margin_q5_head`:** identical gate, but reassign to the highest-scoring candidate
with N >= 18,000 (the established head threshold) in the top-5; if the top-5 holds
no head candidate, keep the baseline prediction. No new constants. This is the
second and final candidate from the margin family in this round (multiplicity note:
two candidates from this family have been judged against the veto).

**Candidate pass 2, `margin_q5_head`: ELIGIBLE, not selected.** Build: 16,239
reassignments, 6,858 to the true label (against 5,413 under runner-up targeting;
1,534 below-tau lines kept at baseline for lack of a head candidate). All three
stages pass; the szy_Latn collapse is gone (head targeting removes the
small-relative reassignment path). Balanced-val row 0.9799 / 0.8981 / 0.9036 /
0.9406 / 0.9814 (overall/tail/magnets/twins/head); veto row: overall 0.9215, tail
F1 0.5321 (precision 0.4445), FPs into tail 6,560, the lowest of all six configs.
Floor-21 remains selected: it ranks higher on the selection instrument (val overall
0.9800 vs 0.9799) and is also better on the veto (overall 0.9309 vs 0.9215, tail
F1 0.6337 vs 0.5321).

**Reading of the round:** the two eligible mechanisms act at different layers.
Floor-21 (weight-side) removes the unseen-token score advantage at the source and
wins outright; the margin gate (decision-side) reaches the same-script relative
residual that no floor edit can touch (pnt/ell) and achieves the largest FP
reduction, but pays recall for it twice (its own suppression plus the caught lines
it cannot recover). The natural next composition, deliberately NOT run this round
(no pre-registration, and the weight-side component may change when the
Good-Turing candidate lands): weight-side winner plus the margin gate with tau
RECALIBRATED under the composed weight matrix (margins change when weights change,
so Exp 26 tau values do not transfer). Recorded in Open paths.

## Experiment 25: precision-primary adoption rule, first verdicts, and the pnt/ell label audit (2026-07-23)

Implements the adoption rule the user fixed on 2026-07-23 (EXPERIMENTAL_SETUP.md
"Precision-primary adoption rule"; symmetric widening follow-up decision same day) and
applies it to the four finished configurations. Analysis only, login node, no new
scoring; code reviewed pre-run (Opus adversarial pass: no correctness defects, two
flags fixed: `run_bias_refit` now shortlists like the other sweeps, and
`balanced_split.__main__` no longer runs a pipeline that would undo the regenerated
draws).

**Instruments built.**
- Balanced test draw seed 201 (headline): 185,204 lines, all 1,940 languages, tail
  median support 16, 70/96 tail languages with >= 10 examples. Drawn disjoint from the
  working val (draw 101) only: excluding the union of all five val draws would leave
  ~2 of a 66-line tail pool (deviation from the first plan wording, recorded in
  `balanced_split.py` and EXPERIMENTAL_SETUP.md).
- Stability draws 102-105 regenerated to exclude the test draw (they had no consumers;
  each is again 188,061 lines with zero languages at reduced k).
- Veto instrument: pool minus the selection and headline draws, 45,004,014 lines,
  retaining median 17 (minimum 4) true lines per tail language. The first run used
  pool minus all six draws and measured veto tail recall 0.2188: six half-draws
  exhaust small pools, so per-language F1 was recall-broken exactly where the veto
  needs it. Amended same day; a runtime gate now aborts if the veto retains median
  < 10 true tail lines per language. Veto levels are not comparable to the Exp 24
  full-pool numbers (half the tail's true lines are excluded, all false positives
  remain); the rule uses gains and drops only.

**Verdicts (`outputs/tables/two_sided_selection.md`).**
- **floor-21: ELIGIBLE and selected** (highest balanced-val overall among eligible,
  0.9800). This supersedes the Exp 20 "not adopted" verdict, which was conditional on
  the recall-only view (Exp 24). Floor-21 is the provisionally adopted configuration:
  provisional because E3 (Good-Turing) is specified as the principled replacement that
  must beat it, and E2 (margin) targets its residual.
- **freq_prior: ELIGIBLE** (val tail/magnet losses 0.0195/0.0197 inside the widened
  0.03 with veto gains +0.1434/+0.1278; worst per-language drop zsm_Arab -0.085 at 13
  veto lines). Not selected (balanced-val overall 0.9798 < floor-21's 0.9800).
- **learned_bias reg=5.0: REJECTED** by the per-language collapse clause: llb_Latn
  loses 0.113 global F1 on 4,181 veto lines (shu_Arab -0.100, led_Latn -0.079 close
  behind). This is the bias suppression pattern at the individual-language level, now
  formally bounded. The Exp 16 numbers stand as the natural-traffic reference
  measurement; what is rejected is adoption under the precision-primary rule. The
  delta review confirmed the llb_Latn drop on the full pool (0.111, false-positive
  inflation 579 to 1,861) and added a support floor to the collapse clause
  (MIN_COLLAPSE_SUPPORT=10 veto lines; at n=4 a single line flip moves F1 by
  0.11-0.14); verdicts are unchanged under the fixed clause.

**Headline (balanced test draw, within-stratum) for the selected configuration:**
baseline 0.9809 / 0.9086 / 0.9121 / 0.9435 / 0.9817
(overall/tail/magnets/twins/head); floor-21 0.9804 / 0.8924 / 0.8984 / 0.9433 /
0.9817. The tail recall cost of the adopted configuration is on the record here; its
precision gain lives in the veto view (tail global F1 0.3382 -> 0.6337 on the veto
instrument; 0.5618 -> 0.7655 full-pool, Exp 24).

**Label audit (plan B2, `outputs/tables/label_audit_pnt_ell.md`).** 50 of the 2,644
lines labeled ell_Grek that floor-21 predicts as pnt_Grek, deterministic sample
(seed 0): all 50 read as standard Modern Greek, short subtitle-register lines (sample
median ~25 characters), none with Pontic diagnostics (provisional assistant
classification, open to override). 48/50 are also flipped by the baseline. Conclusion:
this residual is model error on short low-evidence lines, not corpus label noise, so
the margin method's recoverable ceiling on the pair is the full 2,644 lines.

**Artifacts:** `outputs/tables/two_sided_selection.md`,
`outputs/tables/label_audit_pnt_ell.md`,
`outputs/diagnostic/balanced_val/val_lines_seed201.npy` (+ regenerated seeds 102-105,
manifest annotations). Code: `analysis/two_sided_report.py`, `analysis/label_audit.py`,
`passes_shortlist`/`passes_two_sided` in `analysis/hierarchical_pool.py`,
`build_test_draw`/`rebuild_stability_draws` in `analysis/balanced_split.py`.

## Experiment 24: within-stratum vs global per-language F1 (metric decomposition, 2026-07-23)

**Question:** do the stratum rows of the full-test tables and global per-language F1
agree about the tail? Analysis of the saved prediction memmaps (Exp 16 job 2784115,
floor-21 job 2791722); no new scoring. Script `analysis/metric_decomposition.py`
(reviewed pre-run, no defects); before reporting it reproduces every recorded
within-stratum value (gate tolerance 6e-5) and the saved per-language F1 exactly.

**Finding 1: the two views disagree about the tail, structurally.** Every stratum row
in `full_test_eval.md` / `full_test_floor21.md` and every guard column restricts truth
and predictions to examples whose true label is in the stratum, so a head-true line
predicted as a tail language is excluded from the tail row. The overall rows are global
per-language F1 and include it. Baseline: tail within-stratum 0.9132; tail global mean
F1 0.5618 over the same 96 languages (mean precision 0.4590, mean recall 0.8741). Tail
labels receive 22,522 false positives against 7,735 true tail examples; 98.9% of them
come from head sources (median source N = 100,000; the head stratum has 43.67M test
lines against the tail's 7,735, so a leak rate near 0.1% from one head language exceeds
a tail language's whole test support; 3,426 lines labeled ell_Grek are predicted as
pnt_Grek, whose true support is 150). With precision fixed at 1.0, tail global mean F1
would be 0.9154:
measured globally, the tail deficit is precision, and the stratum rows cannot show it.
57/96 tail languages have precision below 0.5; 7/96 have recall below 0.5.

**Finding 2: the config ranking on tail inverts between the views.**

| config | tail within-stratum (reported) | tail global F1 | precision | recall | FPs into tail labels |
|---|---|---|---|---|---|
| baseline | 0.9132 | 0.5618 | 0.459 | 0.874 | 22,522 |
| learned bias reg=5.0 | 0.9114 | 0.6003 | 0.502 | 0.871 | 17,496 |
| freq prior gamma=0.5 | 0.8950 | 0.6800 | 0.616 | 0.850 | 12,381 |
| floor-21 | 0.8928 | 0.7655 | 0.763 | 0.842 | 9,103 |

The "not tail-safe" (Exp 16) and "not adopted" (Exp 20) verdicts are conclusions about
the within-stratum view only. Under global per-language F1 each of those configurations
raises tail mean F1 (+0.04 to +0.20) at a tail recall cost of at most 3.3pp, and the
ordering is exactly reversed: floor-21, the configuration with the largest
within-stratum tail drop, raises tail global F1 most and raises flat_magnet mean F1
from 0.4716 to 0.6402.

**Finding 3: the mechanisms are complementary.** Decomposing each overall +delta into
per-category contributions: floor-21's +0.0129 comes mostly from flat_magnets (+0.0103
of it), with head and twins flat; the learned bias reaches the same +0.0129 from head
(+0.0031), mid (+0.0039), and twins (+0.0009), and is the only configuration that
raises twin global F1 (0.8887 to 0.9103). A composition test is proposed (Open paths
E4).

**Finding 4: the selection protocol cannot register this failure mode, twice over.**
(a) The guard's stratum columns are within-stratum, so cross-stratum false positives
are excluded by construction; (b) the balanced val caps every language at K=100 lines,
which removes the volume asymmetry that produces the false positives in the first
place. Selection under the current guard is therefore systematically directed against
tail-precision configurations. The two views answer different questions (is a genuine
tail line recognized, versus is an emitted tail label correct); which one the paper's
tail claims use is now part of the objective decision in Open paths.

**Finding 5: residual structure under floor-21.** 9,103 residual false positives into
tail labels, 100% same-script, concentrated in directed pairs of close relatives:
pnt_Grek from ell_Grek (2,644), sbs_Latn from loz/bem/kng/toi/kqn (about 1,040
combined), mrq_Latn from tah/rar (480), tat_Latn from tur (209), min_Arab from fas
(199), rme_Latn from eng (193), mns_Cyrl from rus (158). Floor manipulation does not
separate close relatives; the proposed follow-ups for that residue are a calibrated
decision margin (E2) and a label audit of the pnt/ell pair (E6).

**Artifacts:** `outputs/tables/metric_decomposition.md`;
`outputs/diagnostic/full_test_per_lang_prf.csv` (per-language precision/recall/F1/FP
for all four configurations, including floor-21, which `full_test_per_lang_f1.csv`
lacks). Script: `analysis/metric_decomposition.py`. **Status:** analysis of record;
proposed follow-ups in `EXPERIMENTS_PLAN.md` Open paths block E; the metric-view
question added to the Decision required item there.

## Experiment 23 — First sweeps under the balanced protocol (2026-07-19)

Three experiments, selection-only on the seed-101 balanced val (job 2794210; baseline
validated against the saved full-test predictions, agreement 1.0000 expected path).

**23a. Floor equalization: rejected at selection time.** Every F drops val tail
(-0.0177 at F=-17 down to -0.0269 at F=-23) and magnets similarly; nothing passes. The
tail-sighted guard reaches in nine minutes the verdict that previously required a
five-hour full-test pass (Exp 20). Plan item 14 closes as a selection outcome.

**23b. Punctuation partial pooling (plan item 15): alpha=300 PASSES the guard.** All
strata non-negative (overall +0.0001, tail +0.0004, magnets +0.0003, twins +0.0001,
head +0.0000); stronger alphas turn twins negative (-0.0031 at alpha=30000), consistent
with the tying result that twin conventions are signal. The effect at alpha=300 is at
the edge of measurability; a full-test pass and a balanced-test evaluation are required
before calling it real. Direction: the only configuration in the program to date with
no negative stratum at selection.

**23c. Learned-bias refit on balanced data (plan item 16): reg=0.3 passes with a
substantial selection-half gain,** overall 0.9818 -> 0.9834 (+0.0016), tail +0.0299,
magnets +0.0252, head +0.0001, twins -0.0016 (within tolerance). Answer to the design
question: attractor suppression survives the uniform-prior objective; the Exp 14 gain
was not purely traffic-prior fitting. The fitted vector is aggressive
(||b||_inf = 11.3) and its most-suppressed languages are NOT the flat magnets but
head/twin sinks (nya_Latn -11.3, por_Latn -8.9, heb_Hebr -7.2, swc_Latn -5.9),
matching the diagnostic finding that 40% of false-positive mass sits on head-level
sinks: under a uniform prior the optimum suppresses dominant cluster members to free
their many satellites. CAUTIONS before any adoption: (1) single draw; refit-per-draw
stability (draws 102-105) is required by protocol caveat 3; (2) the guard's stratum
tolerances do not bound INDIVIDUAL-language harm, and b = -8.9 on Portuguese trades
that language's marginal recall for its satellites, an objective-level question to
decide explicitly; (3) selection-half optimism (fit and selection halves share the
draw's candidate structure); (4) a full-test pass and a balanced-test draw (disjoint
from val) are needed for final numbers.

**Artifacts:** `outputs/tables/balanced_{floor_eq,punct_prior,bias_refit}.md`,
`learned_bias_balanced.npy`; `analysis/balanced_sweeps.py`.

## Experiment 22 — Balanced validation protocol and re-baseline (2026-07-19)

**What:** the split redesign (plan item 10). Language-balanced val drawn from the kept
full-test pool (K=100 per language, fraction cap 0.5; 188,061 lines; all 1,940
languages represented, tail median support 33 vs 0 under the old protocol; five seeds
for split-variance checks; the original 250k val is retired). The four saved full-test
prediction sets were re-scored under the new protocol with no new model scoring.

**Selection view (balanced val, guard verdicts):** frequency prior FAIL (tail -0.0195),
floor-21 FAIL (tail -0.0228, magnets -0.0196): the balanced val catches, at selection
time, both failures that previously required full-test passes to discover. Learned
bias FAIL for a different and structural reason: balanced-val overall drops -0.0012
below baseline. A fitted per-language bias approximates the log prior of natural
traffic; on a language-balanced sample the optimal prior is uniform, so any nonzero
bias loses by construction. **Under the balanced (uniform-prior) objective, the
unmodified baseline is the best configuration tested to date.**

**Final view (pool minus val):** natural-traffic numbers persist (baseline 0.9210,
freq prior 0.9343, learned bias 0.9342, floor-21 0.9372 overall; tail 0.9069 / 0.8908 /
0.9062 / 0.8883). The two views answer different deployment questions (natural traffic
vs every-language-equal); the protocol makes the choice explicit instead of conflating
them, and selection uses the balanced view.

**Artifacts:** `outputs/tables/balanced_split_rebaseline.md`,
`outputs/diagnostic/balanced_val/`; `analysis/balanced_split.py`.

## Experiment 21 — Macrolanguage-hierarchical decision (NULL, 2026-07-18)

**Hypothesis:** treating the variety within a macrolanguage as latent (group score =
logsumexp over members from the top-50 candidates, argmax group, best member within it)
recovers errors inside the 83 multi-member SIL macrolanguage groups (289 languages).
Parameter-free; job 2791444.

**Result: exact null.** The marginal essentially never flips a group decision (test
deltas -0.0000 on every stratum; exact accuracy unchanged at 0.9603). Score gaps
between candidates are large in nats, so the group marginal is dominated by its top
member and argmax-of-groups equals group-of-argmax. The useful output is the ceiling
measurement: macro-aware accuracy on the test half is 0.9680 against exact 0.9603, so
0.77pp of accuracy (and the ~20% Exp 10 error share) is within-macro confusion that no
decision rule can recover; it is an evaluation-convention question, not a modeling one.

**Artifacts:** `outputs/tables/macro_hierarchy.md`; `analysis/macro_hierarchy.py`.

## Experiment 20 — Downward floor equalization (POSITIVE on overall; tail pending
full-test, 2026-07-18)

**Hypothesis:** the resource-tied floor means small languages under-penalize unseen
tokens (Exp 10); every mass-ADDING fix failed (Exp 9/13/18/19), so equalize DOWNWARD:
clamp each language's exact floor plateau to `min(floor_L, F)`, one global constant,
nothing raised, observed tokens and specials bit-identical. F swept over
{-17, -19, -21, -23} (n_modified 452 / 1,821 / 1,940 / 1,940). Job 2791444.

**Result:** val overall rises at every F (+0.0024 to +0.0038, peak at F=-21) with
twins/head/magnets flat on val; the guard selects **floor-21**. Test half:

| stratum | base | equalized | delta | 95% CI |
|---|---|---|---|---|
| overall | 0.9454 | 0.9494 | +0.0030 | [+0.0016, +0.0044] |
| twins | 0.9224 | 0.9228 | +0.0003 | [-0.0004, +0.0014] |
| head | 0.9603 | 0.9600 | -0.0003 | [-0.0009, +0.0003] |
| magnets | 0.8832 | 0.8630 | -0.0108 | [-0.0429, +0.0295] |
| tail | 0.9310 | 0.8621 | -0.0623 | [-0.1111, +0.0000] |

This is the first likelihood-side modification to beat baseline with a CI excluding
zero, it is fully modular (one shared constant, no fitting), and the mechanism is the
subtractive direction the four negatives pointed to. Adoption was blocked on the tail
question pending a full-test pass.

**Full-test verdict (job 2791722, 2026-07-19): the tail cost is real; NOT adopted.**
One scoring pass under the floor-21 matrix against the saved Exp 16 baseline
(45,377,279 lines):

| stratum | base | floor-21 | delta | 95% CI |
|---|---|---|---|---|
| overall | 0.9292 | 0.9421 | +0.0129 | point only (n > cap) |
| head | 0.9602 | 0.9599 | -0.0003 | point only |
| twins | 0.9167 | 0.9166 | -0.0001 | point only |
| tail | 0.9132 | 0.8928 | -0.0204 | [-0.0257, -0.0161] |
| magnets | 0.9138 | 0.8974 | -0.0164 | [-0.0210, -0.0129] |

Mid stratum (1k <= N < 18k, 984 languages): +0.0001. Accuracy +0.0009. Decomposition:
the stratum tables are computed on stratum-masked example subsets, so the overall
+0.0129 is a global-precision effect (languages stop receiving stolen cross-stratum
false positives) while the tail -0.0204 is a recall-side harm (examples truly written
in tail languages are misclassified more; lowering a tail language's floor penalizes
its own out-of-corpus tokens). Under the fairness objective the tail regression is
disqualifying: at equal overall gain (+0.0129 both), the learned bias costs the tail
-0.0018 (Exp 16) versus floor-21's -0.0204, so floor equalization is dominated and is
recorded as an overall-for-tail trade alongside the frequency prior, not adopted.
[Metric-conditional, added 2026-07-23: the -0.0204 and the domination claim hold on
the within-stratum (recall) view only. Under global per-language F1, floor-21 is the
strongest tested configuration for tail (0.5618 to 0.7655) and flat_magnet (0.4716 to
0.6402) mean F1, and the learned bias acts on different strata (head/mid/twins); see
Exp 24.]

Third structural lesson about selection: the val guard passed floor-21 because val is
blind on tail AND magnets; the full test refuted both strata. After the Apertus
gamma=3.0 flaw (Exp 15) and the freq-prior tail artifact (Exp 16), this is the third
val-selected operating point overturned at full scale. No further sweep selections
should be trusted on tail/magnet claims until the split redesign (plan item 10) adds
tail-sensitive validation.

**Artifacts:** `outputs/tables/floor_equalization.md`, `full_test_floor21.md`;
`analysis/floor_equalization.py`, `full_test_floor21.py`.

## Experiment 19 — Group-mean back-off at floor positions (NEGATIVE, 2026-07-18)

**Hypothesis:** the resource-tied unseen-token floor (an exact per-language plateau over
74,617-99,810 of 100k entries; Exp 10: corr(floor, log10 N) = -0.966) is the tail's
under-fitting mechanism, so replacing the flat floor with a group-informed profile
(`lam_L * m_G(t)`, `lam_L = alpha/(N_L+alpha)`, observed tokens bit-identical, no
renormalization) should improve discrimination. Two groupings: script backbone means
(job 2790155) and, at the user's request, WALS genealogical tiers
(genus-within-script 535 languages, family-within-script 360, script fallback 1,012;
source `data/wals_languages.csv`; job 2790174). Modes lift/full x alpha
{300, 3000, 30000}.

**Result: negative at every strength under both groupings, and the grouping barely
matters.** Val overall falls monotonically with alpha (script: -0.0028 to -0.0289; WALS:
-0.0036 to -0.0304); at alpha >= 3000 val tail falls 0.8710 -> 0.8387 and magnets
0.8797 -> 0.8609 under both. Full mode tracks lift mode within 0.0007 everywhere. No
config passes the guard; baseline selected in both runs.

**Mechanism reading:** lifting a language's unseen-token mass toward any group mean makes
it MORE accepting of group-plausible foreign material, increasing exactly the theft that
Exp 10 diagnosed (small languages already under-penalize unseen tokens). Together with
Exp 9 (transfer), Exp 13 (shrinkage/sharpening), and Exp 18 (tying), every intervention
that moves probability mass toward group typicality has now failed in the same
direction. Consequence for the family-initialization idea: this was its post-hoc
surrogate (initialization persists only where data is absent, i.e. at these floor
positions), so a family-initialized or family-MAP retrain is not supported by current
evidence. The direction Exp 10 actually implies remains untried: equalizing the
per-token unseen penalty DOWNWARD (lowering low-resource floors toward high-resource
levels), the opposite of every mass-adding fix tested so far. Note Exp 6 clamped floors
UPWARD (also negative), so downward equalization is genuinely untested.

**Artifacts:** `outputs/tables/family_backoff.md`, `family_backoff_wals.md`,
`outputs/diagnostic/backoff_groups_{script,wals}.csv`; `analysis/family_backoff.py`;
`data/wals_languages.csv` (provenance `data/README.md`).

## Experiment 18 — Non-content token tying (NEGATIVE, 2026-07-18)

**Hypothesis:** tokens with no language identity (digits, whitespace, punctuation)
contribute only estimation noise to score differences (Exp 10: 86.4% of the stolen margin
is short non-content tokens), so tying their probabilities to one shared value across all
languages should help. Pure tying, no renormalization (renormalizing would inject a
per-language per-token offset up to 0.36 nats/token; derived in the pre-run review).
Three tied sets classified on byte-decoded token text: digits_ws (298 tokens),
nonalpha_ascii (479), nonalpha_all (1,291). Job 2790078.

**Result: negative at every scope.** Val overall macro-F1 drops monotonically with tied-set
size: digits_ws -0.0010, nonalpha_ascii -0.0063, nonalpha_all -0.0078; nonalpha_all also
drops val tail -0.0108. No config passes the guard; baseline selected.

**Reading:** cross-language variation in non-content token probabilities is not pure noise;
it carries usable signal. The most likely single cause is whitespace: the tied sets include
the space and newline tokens, and whitespace frequency separates spaced from unspaced
scripts. A refinement (tie digits+punctuation but not whitespace) was not run.

**Curated re-run (2026-07-19, job 2793541): still negative; the hypothesis is refuted at
every curation level.** After the user's critique of the original design (whitespace should
never have been tied), the tied set was rebuilt: 212 tokens of ASCII digits plus neutral
punctuation only, with documented linguistic exclusions (apostrophes, hyphens/dashes,
ampersand, currency, Spanish inverted marks, typographic quotes, all whitespace including
leading-space variants, all non-ASCII punctuation), tied within script groups (primary) and
globally (comparison). Val outcome: dp_global overall -0.0014 with twins -0.0060; dp_script
overall -0.0016 with twins -0.0103 (fails the twin guard on its own); tail and magnets
flat under both. Baseline selected; all test deltas zero by construction.

**Final reading:** the cost concentrates in the twins stratum, so within-pair differences in
digit and punctuation usage rates are genuine discriminative signal for near-twin pairs,
consistent with Exp 4 (punctuation is 10.5% of within-pair KL; that KL is signal, not
estimation noise). The Exp 10 stolen-margin finding needs the sharper interpretation this
gives: short non-content tokens dominate margins because they are frequent and their
per-language rates are informative; the pathological part is only their UNSEEN (floor)
positions, and both floor directions have now been tested directly (raising toward group
means, Exp 19: worse; lowering to a common value, Exp 20: tail-harmful). Full tying
(weight 1 on the shared value) is refuted at every curation level; the constructive
reading is that these experiments LOCALIZED where punctuation/digit statistics are
signal (well-estimated head/twin rates) versus noise (low-N estimates and floor
positions), which motivates partial, N-indexed treatment of exactly these columns
(punctuation back-off / hierarchical prior, plan item 15) rather than the
all-or-nothing tie tested here.

**Artifacts:** `outputs/tables/token_tying.md`, `token_tying_dp.md`;
`analysis/token_tying.py`.

## Experiment 17 — Prior-centered learned bias with corrected gradient (2026-07-18)

**Setup:** the learned-bias penalty generalized to `reg*||b - gamma*log(N+1)||^2`
(gamma in {0, 0.25, 0.5} x reg grid; gamma=0 is plain L2), fit with the corrected NLL
gradient (see the Exp 14 estimator caution). Job 2790077.

**Result:** the guard selects gamma=0.25, reg=10. Test half: overall +0.0117
[CI +0.0104, +0.0130], twins +0.0124, head +0.0089, magnets -0.0052 (crosses 0), tail
-0.0320 (the 250k-half tail, which Exp 16 showed is noise-dominated; a full-test read is
required before interpreting it). Under the corrected gradient the previous operating
point (gamma=0, reg=5) fails the guard (val magnets -0.0119), so the Exp 14 revised
selection does not survive the estimator fix as-is. The centered gamma=0.25 point is
marginally above the old plain-L2 reg=5 on the same half (+0.0117 vs +0.0112,
overlapping CIs). Not adopted as a result of record pending a full-test evaluation;
method status also depends on the modularity concern recorded 2026-07-18 (the
discriminative fit couples all languages through the softmax, so adding a language
requires refitting on global data, unlike the likelihood-side methods).

**Artifacts:** `outputs/tables/learned_prior_centered.md`, `learned_bias_centered.npy`.

## Experiment 16 — Full-test-set evaluation of the fixed configurations (2026-07-18)

**Question:** do the guard-selected results (Exp 14) hold at full test-set scale, where the
tail is measurable? On the 250k test half every one of the 96 tail languages has <= 2
examples, so the Exp 14 tail deltas (freq prior 0.0000; learned bias -0.0320 with CI
touching 0) rested on ~35 items. Job 2784115 (05:06:50) scored the 100k model on all
45,377,279 non-val test lines for three configurations FIXED on val: baseline, frequency
prior gamma=0.5, learned bias reg=5.0. No selection; pure evaluation. Zero-bias
predictions validated against the recorded UniLID predictions (agreement 0.9951, matching
the known baseline self-agreement). All 1,940 languages have test support (tail stratum
7,735 examples, magnets 64,657).

| stratum | base | freq delta | freq 95% CI | learned delta | learned 95% CI |
|---|---|---|---|---|---|
| overall | 0.9292 | +0.0116 | point only | +0.0129 | point only |
| head | 0.9602 | +0.0011 | point only | +0.0101 | point only |
| twins | 0.9167 | +0.0011 | point only | +0.0116 | point only |
| tail | 0.9132 | -0.0182 | [-0.0225, -0.0146] | -0.0018 | [-0.0035, -0.0001] |
| magnets | 0.9138 | -0.0173 | [-0.0207, -0.0141] | -0.0082 | [-0.0099, -0.0066] |

Overall accuracy: baseline 0.9608, freq prior 0.9638, learned bias 0.9751. CIs (B=1000
item bootstrap) are computed for strata under 3M examples; for the others the item-level
CI half-width is below 0.001.

**Conclusions.**
1. The learned bias result is confirmed and its tail scare is resolved: the true tail
   cost is -0.0018 [CI -0.0035, -0.0001], small though nonzero; the -0.0320 point
   estimate on the 250k half was split noise. Magnets cost is real but modest (-0.0082).
   Overall +0.0129, head +0.0101, twins +0.0116, accuracy +0.0143. These are the numbers
   of record for the learned bias.
2. The frequency prior is NOT tail-safe: tail -0.0182 [CI -0.0225, -0.0146]. Its Exp 14
   "tail exactly 0.0000" was an artifact of the test half containing almost no tail
   examples. The Exp 14 claim that the frequency prior is the safer minimal version is
   withdrawn: on the full test set the learned bias has BOTH the larger gain and the
   10x smaller tail cost. [Metric-conditional, added 2026-07-23: these tail deltas are
   within-stratum (recall-view) numbers. Under global per-language F1 the frequency
   prior raises tail mean F1 from 0.5618 to 0.6800 by reducing false positives into
   tail labels (22,522 to 12,381); see Exp 24.]
3. Macro-F1 LEVELS are not comparable between the 250k half and the full set (baseline
   0.9454 vs 0.9292): languages absent from the half's true-label set contributed no term
   there, and the full set adds every hard rare language. Deltas are the comparable
   quantity.

**Artifacts:** `outputs/tables/full_test_eval.md`;
`outputs/diagnostic/full_test_per_lang_f1.csv` (per-language F1 under all three configs,
input for plan items 5-6); memmaps + config fingerprint in
`/capstor/scratch/.../unilid_analysis/full_test_eval/`. Script:
`analysis/full_test_eval.py` (reviewed pre-launch; resumable; fail-loud alignment and
scorer checks).

## Experiment 15 — Apertus 200k retrain + frequency prior on it (MIXED / cautionary)

Tests the orthogonal vocab-coverage hypothesis: does a larger, better-covered vocabulary
(Apertus V2 200k byte-level, seeded into the Unigram and re-estimated per language) improve
macro-F1 on its own? Standard-setup retrain (NOT the refuted MAP-EM prior), all 1,940
languages, SP per-language re-estimation on the recovered GlotLID corpus. Job 2639097 (timed
out at 12 h, 1,690/1,940) + 2641940 (resumed, COMPLETED, all 1,940). Model
`glotlid_apertus200k.unilid` (1.56 GB). Evaluated on the same 500k sample / val-test split /
strata as the 100k model, so baselines are directly comparable.

**Retrain baseline vs the 100k model (test half, gamma=0 in each):**

| stratum | 100k base | Apertus base | delta |
|---|---|---|---|
| overall macroF1 | 0.9454 | 0.9447 | -0.0007 |
| tail | 0.9310 | 0.8966 | **-0.0344** |
| magnets | 0.8832 | 0.8999 | +0.0167 |
| twins | 0.9224 | 0.9219 | -0.0005 |
| head | 0.9603 | 0.9608 | +0.0005 |
| accuracy | 0.9603 | 0.9644 | +0.0041 |

The 200k vocab RAISES overall accuracy (+0.41pp, head-driven) but LOWERS tail macro-F1
(-3.4pp). For a macro-F1 / fairness goal the retrain is a net negative on the tail: bigger
vocab helps common languages and hurts rare-tail discrimination. The vocab-coverage
hypothesis is not supported for the tail.

**Frequency prior on the Apertus model: reveals a guard flaw.** The val guard (no
twin/head regression) selected gamma=3.0: test overall 0.9447 -> 0.9673 (+0.0203
[CI +0.0179,+0.0228]) BUT tail -0.0945 [CI -0.1667,0] and magnets -0.1102 [-0.1598,-0.0570].
The +0.0203 "overall" is an artifact of macro-averaging over 1,940 languages: the many head
languages (+0.0061) outnumber the collapsing tail. On the Apertus model even the mildest
gamma=0.25 already drops val_tail (0.8387 -> 0.8065), so NO frequency-prior strength helps
the Apertus tail. **The guard is insufficient: it protected only twins/head, not tail/magnets,
and so selected a tail-destroying operating point.**

**Resolution (2026-07-10, job 2731803).** Under the fixed all-strata guard (`GUARD_TOL =
0.01`, `EXPERIMENTAL_SETUP.md` "Selection guard") no gamma is eligible on the Apertus model
(every gamma >= 0.25 drops val tail by >= 0.032); the baseline is selected and the frequency
prior is formally rejected on this model. The gamma=3.0 selection above is superseded and
kept as the record of the guard flaw.

**Conclusions.** (1) The frequency prior `gamma*log N_L` is a blunt frequency reweighting
that trades tail for head; it looked tail-safe on the 100k model only because gamma=0.5 was
mild there. (2) The learned per-language bias (Exp 14) is the precise instrument: on the 100k
model it improved overall +0.0180 with the tail FLAT, because a free per-language offset can
down-weight specific attractors without penalizing every rare language. (3) The Apertus
retrain is a mostly-negative branch for the tail. Next: fix the selection guard to protect
all strata, and run the LEARNED bias (not the frequency prior) on the Apertus model.

**Artifacts:** `outputs/tables/prior_sweep_apertus.md`; model `glotlid_apertus200k.unilid`;
`slurm_apertus_train.sh`, `slurm_prior_apertus.sh`.

## Experiment 14 — Per-language frequency prior (POSITIVE, first real improvement)

**Redirect after Exp 13.** The Stage 1 failure (shrinkage removes a magnet's recall together
with its false positives) pointed to a per-language PRIOR instead: a constant offset `b_L`
added to each language's summed score before the argmax (Rust `best_of_cached_weight_sets_biased_batch`,
added + validated). A constant matters most on SHORT text (few tokens, small |score|) where
the rare-attractor problem is worst, and it is selective: a magnet still wins on its own text
(large likelihood margin) but loses the marginal cases it was stealing. Prior family swept:
`b_L = gamma * log(N_L + 1)` (frequency prior P(L) ~ N_L^gamma). Tuned on val, scored once on
test (job 2639127).

**Result: significant improvement.** Val overall macro-F1 rises monotonically with gamma
(0.9451 -> 0.9639 at gamma=5), but twins/tail/magnets regress at high gamma (over-favouring
common languages). The val guard (no twin/head regression) selects **gamma=0.5**, where BOTH
twins and head already improve. On TEST at gamma=0.5:

| stratum | base | prior | delta | 95% CI |
|---|---|---|---|---|
| overall | 0.9454 | 0.9524 | +0.0058 | [+0.0048, +0.0069] |
| head | 0.9603 | 0.9615 | +0.0012 | [+0.0008, +0.0017] |
| twins | 0.9224 | 0.9243 | +0.0019 | [+0.0004, +0.0042] |
| tail | 0.9310 | 0.9310 | +0.0000 | [0, 0] |
| magnets | 0.8832 | 0.8811 | -0.0026 | [-0.0130, +0.0035] |

Overall accuracy 0.9603 -> 0.9634 (+0.0032). The macro-F1, head, and twins CIs all exclude 0;
the tail is untouched and magnets are flat (NOT destroyed as shrinkage destroyed them). This
is the selective effect predicted: the constant offset removes marginal false wins (so the
victim head/twin languages recover) without blunting any language's own-text margin. Modular
(one scalar per language from its train count), no retraining. This is the first modification
in the project to beat baseline with a CI that excludes zero. Re-run 2026-07-10 under the
revised all-strata guard (job 2731804): gamma=0.5 is again selected (val magnets -0.0081 is
within the 0.01 tolerance) and the test numbers are unchanged; the artifact header now
records the guard rule. **Full-test-set update (2026-07-18, Exp 16):** at full scale the
frequency prior costs tail -0.0182 [CI -0.0225, -0.0146]; the "tail +0.0000" in the table
above is an artifact of the 250k half's tail invisibility (every tail language has <= 2
examples there). The tail-safety claim for this prior is withdrawn.

**Learned per-language bias (guard-revised result of record, 2026-07-10).** Generalizing the
1-param frequency prior to a free `b_L` per language, fit on val by L2-regularized softmax over
each example's top-k candidate scores (top-20 recall of the true label 0.9971; Rust
`top_k_of_cached_weight_sets_batch` added; gradient verified to 2e-10). The original run (job
2640065) selected reg=0.3 under the twins/head-only guard and reported overall +0.0180; that
selection is superseded (see Invalidated / superseded results) because reg=0.3 costs val
magnets -0.0318, which the revised all-strata guard disallows. Under the fixed guard
(`GUARD_TOL = 0.01`, see `EXPERIMENTAL_SETUP.md` "Selection guard"; REGS extended with 5 and
7), the re-run (job 2731802) selects **reg=5.0** (val magnets -0.0075). On GlotLID test:

| stratum | base | learned | delta | 95% CI |
|---|---|---|---|---|
| overall | 0.9454 | 0.9567 | +0.0112 | [+0.0099, +0.0124] |
| head | 0.9603 | 0.9696 | +0.0094 | [+0.0086, +0.0101] |
| twins | 0.9224 | 0.9358 | +0.0135 | [+0.0101, +0.0170] |
| tail | 0.9310 | 0.8966 | -0.0320 | [-0.0588, +0.0000] |
| magnets | 0.8832 | 0.8862 | +0.0051 | [-0.0302, +0.0463] |

Overall accuracy 0.9603 -> 0.9749 (+0.0147); overall macro-F1 +0.0112, about 2x the frequency
prior; overall/head/twins CIs exclude zero; magnets crosses zero. This is the project's best
guard-compliant result.

**Estimator caution (found in review, 2026-07-18).** The fit's NLL gradient accumulated
softmax soft counts over ALL examples' top-k candidates while the loss conditions on the
true label being inside the top-k (recall 0.9971), so the fitted b was not exactly the
minimizer of the stated objective (finite-difference verified; the perturbation
concentrates on the confuser languages that populate absent examples' top-k lists). The
test deltas above are valid measurements of the b that was produced; only the estimator
description was wrong. The gradient is fixed in `analysis/learned_prior.py` and the
prior-centered re-run (job 2790077) re-fits the plain-L2 bias with the corrected
gradient as its gamma=0 rows.

**Tail caution and a guard blind spot.** The test tail delta is -0.0320 with CI
[-0.0588, +0.0000] (upper bound exactly 0): not significantly negative at the 95% level, but
the point estimate is large. The val guard could not have seen this: val tail macro-F1 is
0.8710 for every reg (and for every gamma <= 1.5 in the frequency-prior sweep), i.e. the val
half contains too few decision-sensitive tail examples for the guard to register tail
movement at all. Addressing this needs a split-design change (plan item 10: resampled
val/test splits, possibly a tail-weighted val allocation), not a tolerance change.
**Resolved (2026-07-18, Exp 16):** on the full test set (7,735 tail examples, all 96
languages) the learned bias's tail delta is -0.0018 [CI -0.0035, -0.0001]; the -0.0320 was
split noise. The guard blind spot itself remains a val-design problem for future sweeps.

**Out-of-domain validation (CommonLID web text, Exp 12 pipeline + priors).** With the guarded
reg=5.0 bias (job 2731818): baseline macro-aware accuracy 0.8452 -> frequency prior
(gamma=0.5) 0.8518 (+0.0067) -> learned bias 0.8879 (+0.0427). The gain holds out of domain;
the superseded reg=0.3 vector gave 0.8936, so the milder guarded vector keeps most of it.
CommonLID's 109 labels are all common languages, so suppressing the rare attractors there is
nearly pure gain.

**Caveats:** the bias down-weights rare languages, so a deployment whose inputs are genuinely
rare-language-heavy could see tail regression (full-test tail delta -0.0018
[CI -0.0035, -0.0001], Exp 16). The earlier framing of the frequency prior as the safer
minimal version is withdrawn: on the full test set (Exp 16) the frequency prior costs tail
-0.0182 while the learned bias costs -0.0018, so the learned bias has both the larger gain
and the smaller tail cost. Pending: learned bias on the Apertus-200k model (plan item 2);
the prior-centered regularizer (plan item 3) may reduce the residual tail/magnet cost.

**Artifacts:** `outputs/tables/prior_sweep.md`, `learned_prior.md`, `commonlid_eval.md`,
`learned_bias.npy`; `analysis/{prior_sweep,learned_prior,commonlid_eval}.py`; Rust
`best_of_cached_weight_sets_biased_batch` + `top_k_of_cached_weight_sets_batch`.

## Experiment 13 — Stage 1 post-hoc gated shrinkage + sweep (NEGATIVE)

**Hypothesis:** shrinking the diagnosed flat_magnet / tight_lowres / isolated_tail rows
toward a confuser-excluded backbone script mean (gated by category) raises stratified
macro-F1 without regressing twins/head. Run on the 500k sample, tuned on the val half,
scored once on the test half (job 2638804, 16 min). Baseline self-agreement 0.9951.

**Result: shrinkage REDUCES macro-F1.** Val overall fell monotonically with shrinkage
strength (0.9451 baseline -> 0.9429 at mag=0.3 -> 0.9378 at mag=0.9). On test the mildest
config gave overall -0.0013 [CI -0.0025, +0.0001], magnets -0.0298, tail -0.0622, twins
+0.0000, head +0.0003. Every stronger config was worse.

**Why (mechanism):** the flat magnets are recalled well on their own (rare) true examples
(magnet-stratum baseline macro-F1 0.88), so shrinking their distributions toward the mean
destroys that recall. The hoped-for victim recovery did not materialise: the head stratum
barely moved (+0.0003), because (a) magnet false-positives are spread thinly across many
high-support victims and (b) removing a magnet as the argmax winner sends the example to the
2nd-place language, which is often still wrong (another magnet or a sibling), not the true
label. So shrinkage trades away magnet/tail recall for a victim-recovery that does not occur.

**Consequence for Stage 2:** the MAP-EM posterior mean `(N c + alpha m)/(N+alpha)` equals
`(1-lambda) p + lambda m` at the EM fixed point (lambda = alpha/(N+alpha)), i.e. the same
operation Stage 1 applied. The re-segmentation difference is second-order. So a full Apertus
MAP-EM retrain with this script-mean prior would very likely reproduce this negative result;
it is not worth the 1,940-language retrain. The diagnosis (magnets steal) is correct; the
*shrinkage-toward-mean* fix is refuted. Redirect candidates: a per-language prior /
calibration offset (down-weights rare attractors without blunting their own-text margin),
or entropy-sharpening the magnet rows (addresses flatness without pulling toward a foreign
mean). Both are cheap post-hoc tests.

**Artifacts:** `outputs/tables/hierarchical_pool.md`, `analysis/hierarchical_pool.py`,
`outputs/diagnostic/lang_diagnostic.csv`.

## Experiment 12 — CommonLID external validation (trends partially hold)

**Question:** do the GlotLID-test error trends hold on out-of-domain web text? UniLID run on
CommonLID (Common Crawl, 373,230 lines, 109 bare-ISO-639-3 tags, macro-aware scoring via the
SIL macrolanguage table; job 2638803, 2.5 min).

**Findings:** macro-aware accuracy 0.8452 (vs 0.9615 on GlotLID test; web text is harder, as
the CommonLID paper intends). The attractor trends transfer: diagnosed flat_magnets account
for 27.7% of error predictions (uzb->tly/vol/ido, eng->sco/nov, msa->abs/bew are all
magnet thefts), confirming the magnet phenomenon is real and domain-general. Resource
asymmetry is weaker but present: predicted language is rarer than the truth in 61.6% of
errors (vs ~86% on GlotLID test). Top confusions are the same closely-related pairs
(arb->ars/ary/acm, ind->zsm, eng->sco, fas->mzn/glk).

**Artifacts:** `outputs/tables/commonlid_eval.md`, `analysis/commonlid_eval.py`,
`/capstor/scratch/.../commonlid/unilid_preds.npz`.

## Invalidated / superseded results

- **Exp 14 learned bias at reg=0.3 (overall macro-F1 +0.0180; accuracy +0.0169; CommonLID
  0.8936), jobs 2640065 / 2640066.** Superseded 2026-07-10. The reg=0.3 operating point was
  selected under the twins/head-only guard and costs val magnets -0.0318, which the revised
  all-strata guard (tolerance 0.01) disallows; the measurements themselves are valid, but the
  configuration is no longer the selected one. Result of record: reg=5.0 (job 2731802),
  overall +0.0112, CommonLID 0.8879 (job 2731818). See Exp 14.
- **Exp 15 Apertus frequency prior at gamma=3.0 (overall +0.0203), job 2649123.** Superseded
  2026-07-10. Selected by the flawed guard while collapsing tail (-0.0945) and magnets
  (-0.1102). Under the fixed guard no gamma is eligible on the Apertus model (job 2731803);
  the frequency prior is rejected there. The sweep table remains valid as a sweep record.

No results were invalidated during the 2026-05-27 reconstruction.

### Premise now false: the 0.2 special-token budget (2026-08-17)

These entries treat the four special tokens' 0.8 share of every row as a property
of the model. It is a training defect (see the 2026-08-17 entries at the top of
this file). **The measurements stand; the derivations built on top of them do
not.**

- **Exp 27, the Good-Turing target `0.2 * n1/T` and the rescaling
  `(0.2 - target)/(0.2 - current)`.** The 0.2 budget is the defect. Against a
  corrected model the target is `n1/T` and the rescaling `(1 - target)/(1 - current)`.
  The headline finding, that the emergent unseen mass overstates the Good-Turing
  estimate for all 1,940 languages (tail median 9x, head median 12x), is a ratio in
  which the factor of 5 cancels, but n1 and T are counts under each language's own
  Viterbi segmentation and the correction shifts segmentation finer, so both counts
  must be recounted before the ratio can be quoted against a corrected model.
- **Exp 50, the `bgfloor` construction.** Inherits the same 0.2 budget and would
  need re-deriving before it could be run against a corrected model.
- **The "argmax-neutral" reading of the special-token structure**, recorded in
  `EXPERIMENTAL_SETUP.md:217`, `EXPERIMENTS_CHRONOLOGICAL.md:921` and
  `EXPERIMENTS_PLAN.md:950`. Measured false on both counts; all three sites are
  corrected in place.

### Valid for the shipped artifact, superseded pending regeneration (2026-08-17)

These are correct measurements **of the model as released**, which carries the
defect. They are not wrong and must not be deleted. Every one of them has to be
regenerated against the corrected weights before it can be quoted again, because
the correction changes 0.72% of predictions and moves the thresholds by up to
123%.

- **Exp 20, the c = -21 sweep over {-17, -19, -21, -23}.** The probe already shows
  the released optimum sits at -19.5 on validation data, and the corrected optimum
  further up.
- **The threshold families: Exp 26, 31, 33, 34, 36, 47, 48.** All 1,084 group-A
  thresholds must be re-estimated; measured, they move in both directions.
- **Exp 48, the four-member high-entropy group** (`sco_Latn`, `bjn_Latn`,
  `arg_Latn`, `vls_Latn`). Identified from predictions, 0.72% of which change, so
  the membership may change. `analysis/build_release_calibration.py` asserts this
  exact set and will abort until it is re-derived.
- **Camera-ready E1 through E5, and every UniLID cell in the paper's tables.**
  Tracked in `RERELEASE_PLAN.md` and in the `EXPERIMENTS_PLAN.md` re-release
  section.

**Uncommitted-results caution.** The committed `EXPERIMENTS.md` (commit `b7508fd`, the only
commit) contains only Experiments 1–6. The working-tree copy adds Experiments 7, 8a, and 9
(+144 lines, purely additive — no committed content was changed). The summaries above use
the working-tree copy, which is the more complete version. This means the Exp 7 / 8a / 9
results currently exist **only** in the uncommitted working tree plus their
`outputs/tables/` files (`train_data_analysis.md`, `discriminative_heuristic.md`,
`transfer_sweep.md`). They are not in git history and would be lost if the working tree were
reset. Committing them would secure the provenance chain. The table files are the artifacts
of record for those numbers.
