# Session Status

## Camera-ready handoff (read this first when resuming paper work)

Everything experimental is FINISHED (E1-E5, results entries "Camera-ready E1".."E5" in EXPERIMENTS_RESULTS.md). The paper (paper/submission.tex + paper/tables/*.tex, all committed) carries the complete calibrated-UniLID integration:

- Abstract + intro sentence; sec:calibration paragraph in section 4 (principled framing: error analysis -> two identified groups -> two inference-time corrections; explicit no-test-overfitting statement; accuracy-reviewed twice, once after the user's hand edits).
- Terminology of record: "the unseen-token constant c" (never bare "floor"; the only "floor" is "the training-time probability floor of 1e-12"); "the high-entropy group/languages" with the precise criterion in app:protocol (entropy z-scored within script via median/MAD > 1.5 AND misclassified-validation absorption > 2x own support, OR z > 5; restricted to >= 18,000 samples). "Flat" appears nowhere.
- Table 1: calibrated row (left cells .957/1.77e-5, .838/2.08e-4, .933/2.91e-4; subset F1 cells .975/.986/.992 under the reproduced convention; subset FPR cells dashed); caption states the instrument and the FLORES-subset convention mismatch.
- Results: Effect of Calibration paragraph. Appendix app:protocol: mechanism spec, provenance, protocol, held-out + bootstrap tables, both-views table, other instruments incl. tab:commonlid, alternatives, the variant transfer paragraph + tab:calibrated_nemo (added 2026-08-09, user-confirmed), remaining errors.
- Every number traces to an artifact; mappings in the chronological log entries dated 2026-08-07 through 2026-08-09.

## Open items for the camera-ready
- Edit pass APPLIED (2026-08-09, commit pending user check): all review items implemented per the approved plan (~/.claude/plans/steady-finding-abelson.md), every text edit wrapped in \camrev{} (red; strip by redefining to #1 in the preamble). User to review the red text, especially: the two published-row unbolds in Table 1 (fastText .944, Mistral-Nemo UDHR-subset FPR 1.03e-4), the Table 1 caption editorial note on the 1.06e-5 cell, and the C5 fixed-constants provenance sentence. Dispositions in paper/review_notes_2026-08-09.md.
- Ahmetcan ask list (updated): subset-evaluation script/command (standing); UDHR-subset UniLID FPR 1.06e-5 confirmation; fastText WiLI config behind tab:unilid_llm_comparison vs tab:noise_robustness; DSL-ML competitor-score source and split (0.385/0.823/0.752/0.762); CommonLID citation; latency run configs (0.155 vs 0.175 ms/sample WiLI).
- Compile checklist for the user: red text renders; Table 1 row label prints "(calibrated, S4)" as a section ref; \cref{sec:ablations} prints the appendix section; page count (new sentences add length; trimming reserves 9-12 remain).
- Subset FPR cells + the printed fastText GlotLID-C subset cell (.990): the four eval scripts Ahmetcan sent (unilid_resources/) confirm the full-set conventions but contain no subset logic; the lang-only hypothesis is measured and refuted; closed-set-for-fastText is the only consistent explanation (full addendum in the "Camera-ready E2" entry). Remaining ask: the specific subset-evaluation script/command, or an author decision on the convention.
- Co-author reads: the UDHR regression framing (Results + Table 1 caption) and the name "calibrated UniLID".
- The user compiles the PDF themselves (no icml2026.sty in this repo); page count fit exactly as of the trimming commit; reserves 9-12 of the trimming report (chat, 2026-08-08) remain available.

## Pending actions (non-paper)
- Open-source release of calibrated UniLID: SHIPPED AND MERGED UPSTREAM (PR #1 merged by Ahmetcan "lgtm!" 2026-08-11, license thereby published; PR #2 merged by the user; HF repo cmeister/unilid-1940 live; both gates 250,000/250,000 exact). Full state, runbook, decisions ledger, and open items: OPEN_SOURCE_STATUS.md (supersedes OPEN_SOURCE_HANDOFF.md).
- SETUP_FEEDBACK.md follow-up: DONE 2026-08-15. PR #3 open (Ahmetcanyvz/UNILID/pull/3) with 8dd90ec, 3427640, 6ab2201. 104 tests, version 0.2.1, Python 3.9/3.11/3.12/3.13/3.14 measured green, CI added, doctor.py added, tokenizers toolchain pinned; both gates re-passed at 250,000/250,000 after the pinned rebuild. Details and one found-not-fixed item (fork-in-multi-threaded-process warning on 3.12/3.13) in OPEN_SOURCE_STATUS.md.
- E3 store migration: DONE 2026-08-10 (all three artifact sets on store, sha256-verified, scratch symlinks in place; chronological log entry).

## sp-vs-em add-language gap (2026-08-17): FIXED in code (author decision: special tokens must not contribute to a score under any method)

Root cause found and proven causal. `unilid/trainers/language_specific_trainer.py:203-204` gives every special token the base tokenizer's score; HF Unigram stores specials with score `0.0`, read here as a log-probability, i.e. probability 1.0. Four specials at 1.0 dominate `_log_normalize`, so each ends at exactly 1/5 and every real token is depressed by log(5) = 1.6094 nats. Measured, not inferred:

- Released model: all 1,940 rows carry exactly 0.2 on each of `<s>`, `</s>`, `<pad>`, `<unk>` (0.8 total), independent of the language's data. This is why the released rows' unseen-token plateau sits at approximately -19 instead of the trainer floor -27.63, which is the phenomenon the unseen-token constant c = -21 corrects.
- Toy model (base rows from the pure-Python soft EM, specials at the 1e-12 floor): adding a real-text language with the default `sp` mixes the two scales. Held-out accuracy 0.24 (sp) vs 0.90 (em) on identical data, reproducing the report.
- Causal test: repairing only the three unemittable specials in the sp row and renormalizing, changing nothing else, moves 0.24 to 0.74. The residual gap to em is the already-documented flatter-at-small-N effect.
- On the released model both methods are usable (sp 0.72, em 0.88 held out; pool accuracy 0.9477 unchanged for both; 0 and 1 of 3,000 pool lines captured), because there the two mismatches partly offset. So this is a scale-mixing failure, not "sp is broken".

Mechanism, confirmed by perturbation rather than by reading the Rust: no special token's stored weight is ever read when scoring. `model.rs` takes the unknown-token score from a single model-wide constant (`min_score - K_UNK_PENALTY`), and `<s>`/`</s>`/`<pad>` are reachable only by text containing those literal substrings. Setting all four entries of every row to -500 changes predicted scores by exactly 0.000000. Mass on them is therefore mass removed from the tokens that do decide predictions.

Fix, version 0.3.0:
- One enforcement point in `LanguageSpecificUnigramLMTokenizer.train`: whichever method produced the row, it is renormalized over the real tokens and the specials are parked at the floor (`unilid/vocab_io.py::renormalize_over_real_tokens`). The sp path also stops copying the base tokenizer's 0.0 scores for specials, which was the source of probability 1.0.
- `add_language` puts the new row on the model's own scale (`_match_real_token_scale`), so a corrected row does not outscore a pre-0.3.0 model's rows by a constant per token. New named constant `SCALE_SPREAD_REPORT_RATIO = 2.0`, diagnostic only, never fatal.

Measured effect (held-out accuracy on the added language). Every row below is a matched pair: one frozen corpus, both code versions run in this session, the pre-fix side obtained by checking the three changed files out of HEAD~1 and rebuilding. An earlier set of numbers regenerated the Python corpus from this repository's own sources between runs, which I was editing, so those were not comparable across runs and are superseded by these.
| case | pre-fix | post-fix |
|---|---|---|
| example ddd_Latn, sp | 0.60 (186/250 own-won) | 0.98 (250/250) |
| example ddd_Latn, em | 0.98 (250/250) | 0.98 (250/250) |
| toy + 300 lines of Python, sp | 0.26 (85/300) | 0.88 (282/300) |
| toy + 300 lines of Python, em | 0.88 (284/300) | 0.98 (298/300) |
| released 1,940 + same Python, sp | 0.84 | 0.84 |
| released 1,940 + same Python, em | 0.90 | 0.86 |
Pool accuracy on 3,000 labelled test-pool lines is 0.9477 in every case. The released-model em drop is the correction working: pre-fix that row kept its full mass against rows holding a fifth of theirs, a 1.6094-nat-per-token advantage over all 1,940 languages, and it captured a pool line that post-fix it does not.

Released artifact is untouched by the code fix and both golden gates re-passed.

Effect of correcting the released weights, measured on the 250,000-line golden subset against the recorded gold labels, base mode: macro F1 0.9454 -> 0.9460, macro FPR 2.083e-05 -> 2.081e-05, accuracy 0.9603 -> 0.9604, with 1,807 of 250,000 predictions changed (0.72%), 699 fixed and 669 broken. Essentially a wash. This supersedes an earlier estimate of 0.9494 -> 0.9509 with 63 fixed and 32 broken, which was accuracy on a 20,000-line every-149th-line sample rather than macro F1 on the golden subset, and was not well enough powered. The case for re-releasing is correctness, not metrics.

Scripts: scratchpad repro_sp_vs_em.py, released_sp_vs_em.py, measure_correction.py, matched_released.py, toy_matched.py, ddd_matched.py.

Shipped into PR #3 (folded per author decision): branch pushed at 9f7c1cf, PR retitled "Special tokens hold no probability mass, plus the out-of-box setup fixes", description rewritten to lead with the training fix.

## Re-release of the corrected model: IN EXECUTION (RERELEASE_PLAN.md, revised after adversarial review)

Done: all four stored models corrected (`analysis/correct_special_token_mass.py`, outputs on scratch under `corrected/`); retrain corroboration passes 8/8 languages; effect measured (macro F1 0.9454 -> 0.9460 on the golden subset, 699 gained / 669 lost, a wash); the two blocking safety defects fixed and verified (`full_test_eval.py` now fingerprints the model, and refuses to write a non-default model's results into the store-symlinked scratch directory).

Blocked on the author: `lid_main.tex:90,98` carry `\unilid-DeepSeek3.2` and `\unilid-Qwen3`, 24 cells of GlotLID-C-trained UniLID numbers with no artifact on this machine and no identified owner. Step 4 cannot complete without them.

Needs measurement before the paper is touched: the paper's stated cause of the above-c unseen-token values is wrong, and the replacement I had planned would also be wrong (the defect accounts for 1.609 of an 11.58-nat gap). `submission.tex:1383-1384` reverses: released Mistral-Nemo has exactly two rows below c=-21 (khm_Khmr, ory_Orya), corrected has zero.

Not started: calibration re-derivation, full-pool runs, paper edits, HF uploads.

## Calibration probe (2026-08-17): found a regression I shipped, not a shifted optimum

Probing how far c moves under the correction (`analysis/probe_calibration_shift.py`, 60,000 lines from the validation half, sweep over nine values with the recipe of record) returned `modified 0` of 1,940 rows at every c for the corrected model, against 1,940 for the released one.

Cause: the 0.3.0 fix parks the special tokens at the training floor, and `apply_unseen_token_constant` defines a row's unseen tokens as its exact minimum-value plateau. With the specials at -27.631 the minimum is them, the plateau of unseen real tokens is never found, and the clamp does nothing. So every model trained by 0.3.0 as shipped in PR #3 has the calibration's first correction silently disabled. This also explains the "row minimum -27.631 is at or below c=-21.0; the row is left unchanged" lines in the post-fix add_language runs, which I read at the time as the trainer floor on real tokens.

Fixed in `unilid/calibration.py` and `analysis/floor_equalization.py`: the clamp now takes the special columns and leaves them out of the minimum. Pre-0.3.0 files are unaffected because their specials sit at -1.6094, never the minimum, which is asserted in a test alongside one for the broken case. Package commit 2d5f62d. Both release gates re-run because this is an inference-path change.

The c sweep on the released model is still informative: macro F1 peaks at c = -19.5 (0.9569) against -21 (0.9567) and -19.0 (0.9568), so the published constant sits just off a flat optimum on this subsample. The corrected model's sweep has to be repeated now that the clamp works.

## Superseded plan note

RERELEASE_PLAN.md. The load-bearing finding is that the corrected artifact is a deterministic transformation of the released weights, not a retrain: real tokens += log(5), specials to the floor. Verified by retraining aai_Latn (24,580 lines) with the 0.3.0 code and comparing against (released row + log 5) over 99,996 real tokens: correlation 1.00000000, median absolute difference 1.7e-5, 99.69% within 1e-4. All four stored GlotLID-scale models carry the same 0.800000 special mass and take the same transformation.

What still has to be re-derived rather than transformed: the unseen-token constant c (sweep on the new scale; -21 + log 5 = -19.3906 would preserve the old behaviour up to the uniform shift, so the sweep should land near it), the per-language taus (margins move with segmentation), and the four-member high-entropy group (identified from predictions that change). Paper tables are split in the plan into re-runnable here (GlotLID-C family), needing the co-author's WiLI and DSL-ML models, and unaffected (latency, training time, all fastText/GlotLID/CLD3 rows).

## Open decisions
- Whether the Mistral-Nemo variant ships in a v1.1 release (handoff open item, unresolved; artifacts ready on store, recipe in OPEN_SOURCE_STATUS.md open item 2).
- Camera-ready items above unchanged.
