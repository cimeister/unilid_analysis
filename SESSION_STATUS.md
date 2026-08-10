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
- Open-source release of calibrated UniLID: fully specified in OPEN_SOURCE_HANDOFF.md (goal, four binding user decisions of 2026-08-09, mechanism spec + reference code paths, verification gates, checklist, and the verbatim prompt for the new session). Not started; intended for a dedicated session.
- E3 store migration: DONE 2026-08-10 (all three artifact sets on store, sha256-verified, scratch symlinks in place; chronological log entry).

## Open decisions
- None beyond the camera-ready items above.
