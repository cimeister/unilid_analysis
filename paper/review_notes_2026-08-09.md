# Camera-ready review, 2026-08-09 (internal notes)

Two-pass review of paper/submission.tex + paper/tables/*.tex: (a) record-checked pass by the
main session (verified the calibration prose against the repo record; mechanism description is
accurate, all cross-checked numbers trace); (b) cold-read pass by an Opus agent with no project
context, simulating a first-time reviewer. Findings below are merged, verified where checkable,
ranked. No edits applied; awaiting user decisions.

## Verdict
The method sections and the calibration mechanism description are accurate against the record.
The risks are concentrated in: Table 1 (caption language, stale bolding, dash semantics,
possible FPR typo), the evaluation-protocol story (dev-part inside the headline pool; the
unexplained 250k line gap), number/table mismatches (0.332/0.732 sentence; latency; fastText
WiLI), and missing provenance/definitions (six constants, macro FPR formula, margin definition,
CommonLID).

## A. Highest severity (verified)

1. Table 1 caption ships work-in-progress language (tables/lid_main.tex:6): "remain provisional
   pending a convention alignment", "reproduces under no convention we tested";
   "restricted-lines convention" undefined. Also `--` overloaded: no-coverage (CLD3/GlotLID-M
   rows) vs not-computed (calibrated FPR cells). Needs author decision (resolve convention with
   Ahmetcan or reword/drop the calibrated subset block).
2. Stale/wrong bolding in Table 1: .944 bolded but calibrated .957 higher (lid_main.tex:58 vs 75);
   1.84e-5 bolded but calibrated 1.77e-5 lower (83 vs 75); pre-existing: UDHR-subset FPR bolds
   1.03e-4 (Nemo) while UniLID 1.06e-5 is column min (87 vs 71). Caption never says what bold
   denotes. SEPARATELY: 1.06e-5 is 10x below every other cell in its column; possible typo for
   1.06e-4; re-check with Ahmetcan (ties into the subset-FPR open item).
3. Protocol tension: proximity bound selected on the 18.0M-line development part = 40% of the
   45,377,279-line pool behind Table 1's .957; 100k bar selected on the full pool. Disclosed in
   app:protocol + provenance table, held-out confirmation exists (0.950, CI [+0.033,+0.043]),
   but Table 1 caption carries no note, and submission.tex:636 "avoids fitting to the evaluation
   data" sits two lines from the admission. Reviewer-visible contradiction as written.
4. The 250,000-line gap is deducible and unexplained: latency table says 45,627,279 samples;
   resource-tier N_test column sums to exactly 45,627,279 (verified); lenbias-delta says 45.6M;
   Table 1 + calibrated_views say 45,377,279 "scored" lines. Difference = 250,000 = the retired
   validation sample (carved from the test data, then excluded from all reporting; sentinel
   EXCLUDED=-2 in the repo). The paper never states the sample's origin or defines "scored".
   Also internally inconsistent: calibrated_views prints the identical within-stratum F1 values
   as resource-tier while declaring a different line total.
5. submission.tex:815-818 cites tab:calibrated_views for "0.332 to 0.732 (held-out subset)" but
   the table is full-pool with tiers <500 (0.515->0.780) and 500-1k (0.628->0.892); neither
   number is in the table; weighted <1k combination is 0.562->0.827. Instrument and bucketing
   both differ from the citation.
6. Latency numbers contradict the table (verified): body 0.274 ms/sample, "1.65x slower"
   (submission.tex:966-968; repeated 1170) vs latency_glotlid.tex 0.307 (=1.85x vs fastText
   0.166, 1.64x vs CLD3 0.187). WiLI: body 0.158 vs table 0.155; vocab_size_efficiency
   reportedly 0.175 for the same config.
7. fastText WiLI inconsistency across tables (verified): unilid_llm_comparison.tex F1 0.946 /
   FPR 2.331e-4 vs noise_robustness.tex p=0% F1 0.954 / FPR 1.98e-4; samples-accuracy 500/lang
   accuracy 94.55 vs noise table 0.954. UniLID's values agree across the same tables (0.960 /
   1.86e-4), isolating the fastText mismatch. If configs differ, tables must say which.
8. CommonLID never introduced: no entry in app:datasets, no citation, first mention in section 4
   (line 640); "macrolanguage-aware accuracy" defined only in the commonlid.tex caption; body
   text (823-824) reports only the improving accuracy while tag-level macro F1 declines
   (0.723->0.715, appendix + caption only). At minimum: dataset paragraph + cite + both metrics
   in one place.
9. Macro FPR never defined: formula block commented out (submission.tex:717-721). The 1e-5-scale
   values are uninterpretable without the per-language TN convention (TN over the whole pool).
   Uncomment/restore the equation.

## B. Method-description gaps (app:protocol)

10. Six constants have no provenance while line 733 promises it: the 18,000 boundary; entropy
    z-thresholds 1.5 and 5; the absorption factor 2; the top-5 candidate cutoff; the base 5th
    percentile; the functional form q = 5(1 - N/18,000). Provenance table covers only c, the
    proximity bound, and the 100k bar.
11. Proximity grid as stated excludes the chosen value: "grid over 0.5 to 100 in steps of 1"
    read literally is {0.5, 1.5, ..., 99.5}, which does not contain 21. Paper transcribed
    EXPERIMENTAL_SETUP.md:635 faithfully; the actual grid must be read off the sweep artifact
    and restated exactly. Also: why 21 within the stated 15-35 plateau is unexplained, and the
    coincidence with |c|=21 will draw a question.
12. The 100,000-sample bar equals the per-language training cap (app:datasets: cap 100k), so it
    selects exactly the capped languages; "selected from a false-positive profile" implies a
    tunable range that does not exist above the cap; semantics on an uncapped corpus undefined.
13. Margin under-specified: never formally defined; unclear whether a language's training
    margins include lines it does not win (negative margins) or only wins; the calibration
    sample cap (2,000 lines/lang) and the minimum-lines exclusion (26 languages with
    unestimable thresholds are exempt from re-examination) are absent from the paper.
14. "Bounds its own recall loss by construction" (634-635, 1289-1291) overstates twice over:
    percentile bounds the re-examined fraction on training lines; train margins are in-sample
    (computed under distributions fit on those same lines), hence optimistically shifted vs
    test. This claim also carries the choice of per-language thresholds over the shared
    threshold that scored higher held-out (0.953 vs 0.950).
15. Body says unseen-token log-probs are "replaced" with c (616-618); the Nemo transfer
    paragraph reveals clamp semantics ("two languages ... left unchanged", 1299-1301). State the
    min() semantics once in section 4. Also: no statement on renormalization after the
    substitution; strictly the calibrated scores are no longer normalized likelihoods, so the
    Bayes-posterior framing of eq. 3 becomes approximate. One sentence would preempt this.
    (Cold-read agent's claim that the correction "raises unseen-token mass by ~3 orders of
    magnitude" is WRONG for the base model: trained unseen values all sit above -21, so the
    replacement lowers them; do not act on that sub-claim.)
16. Premise "misclassifications concentrate on predictions into two identifiable groups"
    (604-606) has no quantitative support; and calibrated_views' 12k-18k stratum (526 langs,
    global F1 0.979, above 35k+'s 0.958) shows most re-examined languages have no measured
    deficiency; 18,000 also coincides with a stratification bin boundary. State the
    concentration number and the basis for 18,000.

## C. Claim calibration

17. Intro claim unscoped (line 341): "outperforming every system with full GlotLID-C label
    coverage"; GlotLID-M has full coverage and is unmeasured (excluded for possible
    contamination). Sec. 6 scopes correctly to "every system in tab:lid_main" (809). Abstract
    meanwhile says only "competitive performance" and never mentions the calibrated result.
18. Scope of the calibration gain: positive on GlotLID-C full pool, held-out part, the retrained
    Nemo variant, and CommonLID accuracy; flat on FLORES (0.932->0.933); negative on UDHR
    (0.859->0.838), the balanced draw (0.981->0.978 within-stratum), CommonLID tag F1
    (0.723->0.715), and the small strata within-stratum view. The paper documents all of this;
    the intro sentence conveys none of it. (Cold-read's "absent or negative under every
    evaluation except one" overstates; use the scoped statement.)
19. lid_main caption causal clause "gains are specific to natural-distribution data, which is
    why the parallel-text UDHR cell decreases" is undermined by FLORES: also parallel, flat.
    The differentiator is the label set (UDHR's 366 includes many under-resourced languages;
    FLORES's 190 mostly larger). Restate or drop "which is why".
20. Noise robustness: body (941) "neither system is meaningfully more robust than the other" and
    sec. 2 (421) "roughly tied" vs table: fastText ahead by 0.082 accuracy at p=25% and 0.280 at
    p=50%; the appendix's own text says fastText degrades more gracefully. Scope the body claim
    to p<=10%.
21. DSL-ML shared-task comparison (831): UniLID evaluated on dev; shared-task rankings are on
    test; competitor numbers (0.385, 0.823, 0.752, 0.762) uncited with no split statement.
    Headline result (abstract, intro, secs. 2 and 6). Needs cite + split disclosure; co-author
    check (Ahmetcan).
22. fastText DSL-ML baseline tuned on the reported eval split (dev); "best validation
    performance" (687) misleading when no separate validation split exists; also 686-687 says
    "recommended hyperparameter settings" then footnotes deviating to 100 epochs. Biases favor
    the baseline; disclose in one sentence.
23. 9x training-time claim (961-964) conditioned entirely on the 100-epoch fastText choice; at
    the reference 1-2 epoch recipe fastText would be ~1.6-3.3k s, faster than UniLID's 17.8k s.
    State the conditioning.
24. Viterbi-vs-marginal "no statistically significant difference" (535, viterbi_vs_marginal
    caption): no test, interval, or seed count reported. Either add the test or soften to the
    measured difference. More broadly, everything outside the calibration bootstrap is
    single-run point estimates (known: TODO at line 1027).
25. Abstract's "incremental addition ... without retraining existing models" is qualified for
    calibrated UniLID (high-entropy grouping uses cross-language script statistics); sec. 4
    concedes it; abstract unqualified. Minor.
26. "Calibration" overloaded: standard probabilistic sense at 418 vs the variant name. Ties to
    the existing co-author read item on the name "calibrated UniLID".
27. Held-out 0.912 vs full-pool 0.929 for identical systems on a random 60% partition looks like
    two different evaluations without a sentence explaining the macro-F1 shift under
    subsampling.

## D. Wording, symbols, mechanics

28. Symbol collisions: N_l = corpus character length (590) vs training-sample count (732,
    q formula); N = EM round index (533) vs input string length (548) while string length was
    introduced as T (351).
29. Two datasets share the name "held-out": "held-out validation subset of GlotLID-C" (1229,
    = the retired sample) vs the 27.0M-line held-out part. Rename the first to "the retired
    validation sample".
30. Jargon/agentive leaks: "rows" for lines (commonlid.tex:18); "retired" never explained;
    "post-draw pool" (1238, 1280); "100,000-sample bar" / "replacement-candidate bar";
    "Measurements on other instruments" heading (1262); "the same trade" (822); "discounts
    unreliable wins" (624); "attract misclassified samples" (610); "honest in isolation"
    (1281); "lets 9 languages lose" (1289); "natural-distribution data" used without a one-line
    definition; "INTO" in caps (calibrated_views caption).
31. Wrong cross-ref: Efficiency paragraph (960) cites app:datasets for the latency tables;
    correct target app:efficiency.
32. Label placement check on next compile (downgraded from cold-read's claim):
    \label{sec:calibration} on an unnumbered \paragraph and \label{sec:ablations} after a float
    most likely resolve to the enclosing section (counter assignments in equation/footnote/float
    groups are local), rendering "S4" as intended; the user has compiled since without reporting
    broken refs. Verify the Table 1 row label prints "(calibrated, S4)" and \cref{sec:ablations}
    prints the appendix section; fix only if wrong.
33. unilid_llm_comparison caption "latter 4 systems" vs 6 rows below the dashed line (verified).
34. Typos/grammar: "is slightly exceeds" (1188); "We lastly look potential length biases"
    (1195); "performace" (1196); "\Cref{...,...} reports" plural (1170); "within 2.5 F1" units
    (944) when tables are 0-1 scale; hardcoded "Table 1"/Table~\ref mixed with \cref (1252,
    lenbias-norm, resource-tier captions).
35. Leftover author notes are disabled via todonotes [disable] but remain in source, including
    an unresolved EOS-cancellation question (452) and the experiments TODO list (1019-1031).
    User decision 2026-08-08: leave them. Compile without [disable] would print all of it.
36. script-breakdown attribution (1182) says the vocabulary corpus is Latin-data-dominated, but
    training_time caption says the base tokenizer used 1k samples/language (balanced), so
    dominance is via language count (1,700/1,940 Latin); the proposed script-upsampling fix is
    still coherent, but the wording and the 1k detail's location (caption only) should be fixed.

## Suggested fix order (user to confirm)
Mechanical, no decisions needed: 2 (bolding), 5, 6, 7 (number/table mismatches), 9 (FPR
formula), 28-31, 33, 34. Author decisions: 1 (subset cells/caption), 3+4 (protocol/250k
disclosure), 8 (CommonLID cite), 11-13 (constants provenance), 17-23 (claim scoping), 21
(Ahmetcan), 26 (name). Verify-only: 32, and the 1.06e-5 cell in item 2.
