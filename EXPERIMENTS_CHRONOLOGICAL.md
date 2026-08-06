# UniLID Analysis — Chronological Experiment Log

> **Reconstruction provenance.** Rebuilt on 2026-05-27 after the original session
> transcript (`9729f7f3-3af8-42d5-818a-1f032a9f6f25`, 2026-03-26 → 2026-04-08) was lost.
> SLURM **job IDs, states, durations, and memory** are taken verbatim from the job-history
> tables in `EXPERIMENTS.md` and are reliable. **Absolute calendar dates** for each run are
> **[inferred]** from the recovered prompt timestamps and source-file modification times,
> not from SLURM records, so treat them as approximate. There is only one git commit for
> the whole project (`b7508fd`, 2026-04-08); per-experiment code versions are not separately
> tracked, so the "code version" for every entry below is "working tree as of `b7508fd`".
> Shared configuration is in `EXPERIMENTAL_SETUP.md`; outcomes are in
> `EXPERIMENTS_RESULTS.md`.

Reverse-chronological, grouped by experiment family. Within each family, jobs are listed
most-recent-first. All jobs ran on CSCS Clariden (account `a139`, partition `normal`),
Python `/users/cmeister747/.pyenv/versions/3.11.5/bin/python3`, working dir
`/users/cmeister747/unilid_analysis`, data on scratch (`config.DATA_DIR`). See `SETUP.md`
for the infrastructure record.

---

## Family: Error analysis, calibration, and the balanced protocol (Exp 10–24)

**Window:** 2026-06-24 to present. Plan items: `EXPERIMENTS_PLAN.md` Exp 10–16 and the
"Next set of methods" items. Setup:
`EXPERIMENTAL_SETUP.md` (hierarchical pooling). Full plan:
`~/.claude/plans/yes-do-both-then-giggly-sprout.md`.

### 2026-08-06: gate_flat4_prox21 promoted; draw-201 confirmation recorded (no job)

- **No SLURM job.** User decision 2026-08-06: gate_flat4_prox21 promoted on
  the natural track after Exp 47-50, superseding floor21_gate (remains in
  the pool). Judge-part overall F1 0.9498, +0.0018 [+0.0010, +0.0026] over
  floor21_gate, zero supported collapses (Exp 49). The pre-registered
  composed step of Exp 50 (rebuilding the gate on the pooled-frequency floor
  matrix) is skipped by the same decision. Confirmation script ran as a
  memmap subset over the already-scored full-pool prediction memmaps
  (`/capstor/scratch/cscs/cmeister747/unilid_analysis/full_test_eval`); no
  scoring pass, no job submitted.
- Artifact: `outputs/tables/gate_flat4_prox21_confirmation_201.md`.
  Instrument: balanced test draw, seed 201, 185,204 lines. Within-stratum
  macro-F1: gate_flat4_prox21 overall 0.9781, tail 0.8763, magnets 0.8811,
  twins 0.9437, head 0.9814; baseline overall 0.9809, tail 0.9086, magnets
  0.9121, twins 0.9435, head 0.9817. Sanity gate (baseline overall row vs
  the recorded headline in `outputs/tables/two_sided_selection.md`): PASS,
  max abs diff 0.00004. Per-language collapse check on the draw-201 subset:
  3 supported collapses (support >= 10, F1 drop > 0.10), attributable to the
  draw's own per-language support cap of at most 100 lines; the
  promotion-gate clause (C), computed on the far larger judge part in Exp
  49, remains clean at zero supported collapses.

### 2026-08-06: Exp 47-50, the four candidate directions in order (jobs 3014614, 3015805, 3016337, 3016380)

- **3014614** gate_variants topk plus apply shared9_bar18k (Exp 47, 07:06):
  candidate arrays banked, shared-threshold variant built; verdict in the pool
  (best aggregate 0.9534, class-level clause-C fail, 9 collapses).
- **3015805** apply flat4_tau5 (Exp 48, 00:33): eligible, 0.9486, zero
  collapses; four flat large-corpus languages and their neighbours all gained.
- **3016337** apply flat4_prox21 (Exp 49, 00:18): in-job reproduction gate
  passed; eligible, 0.9498, the strongest eligible configuration; refinement
  contrast +0.0012 [+0.0007, +0.0016] over gate_flat4_tau5.
- **3016380** `slurm_full_test_bgfloor.sh` (Exp 50, running): full-pool scoring
  under the pooled-frequency unseen-token values (c = -8.4740; assigned plateau
  range -27.61 to -12.31; alignment of the base distribution verified four
  independent ways in review). Output pred_bgfloor.npy; first verdict is the
  gate-less judge-part comparison against floor21 solo.
- All four scripts Opus-reviewed before their runs; every verdict recorded in
  EXPERIMENTS_RESULTS.md Exp 47-50. Checkpoint hygiene: no deletions; new
  artifacts are three candidate arrays, four prediction memmaps, tau_flat4.csv.

### 2026-08-05: Exp 47 submitted (job 3014614): shared re-examination threshold

- **3014614** `slurm_gate_variants.sh`: 64 CPU, 100G, 03:00:00. Stage one saves
  the top five candidate languages and scores under the floor-21 matrix for the
  2,236,864 kept lines whose floor-21 prediction is a language with under
  18,000 training lines or in the flat-distribution category (the saved arrays
  also serve Experiments 48 and 49 with no further scoring). Stage two builds
  the Experiment 47 variant: one shared threshold of 9.0, replacement-candidate
  minimum 18,000 training lines. Pre-registration: EXPERIMENTS_RESULTS.md
  "Experiment 47 pre-registration". Code `analysis/gate_variants.py`, Opus
  review found two blockers (a label-set inflation to 12.0M lines and a false
  timing claim), both fixed with nine hardening items before submission.
- Directions 1 through 4 are being tried in order (user 2026-08-05).
  Checkpoint hygiene: no deletions; new artifacts are the three candidate
  arrays (about 130 MB total) and one prediction memmap per variant.

### 2026-07-30: Exp 46 mixed-matrix scoring submitted (job 2932154); Exp 44/45 completed

- **2932154** `unilid-mixed-matrix`: 64 CPU, 100G, 06:00:00. Four fail-fast
  stages: the pre-registered no-op scorer check (chunk 0 under W must be
  bit-identical to pred_baseline.npy), stage A full-pool scoring under the
  rule-v1 mixed matrix (sha 0c31f143..., 860 unmod rows + 1,080 floor-21 rows,
  fingerprint_mixed.json), stage B adaptive gate (tau recalibrated under the
  mixed matrix, tau_mixed.csv+json provenance binding), stage B_solotau (tau
  read from tau_floor21_gate.csv, isolating the tau-recalibration component).
  Outputs pred_mixed_nogate.npy, pred_mixed.npy, pred_mixed_solotau.npy.
  Hypotheses and criteria: EXPERIMENTS_RESULTS.md "Experiment 46
  pre-registration". Code reviewed (Opus, no blockers) with four hardening
  fixes applied before submission; `analysis/mixed_matrix.py`.
- Exp 44 (evidence base, seed-301 split) and Exp 45 (solo-gate references,
  jobs 2930701/2930702, 15:23 and 13:48; floor21_gate judge-part evaluation)
  recorded in EXPERIMENTS_RESULTS.md. User decisions 2026-07-30: rule v1
  signed off; bootstrap anchor switched to floor21_gate (condition met);
  clause-(A) cap question deferred to Exp 46 results; amendment scope
  confirmed (judge part is the confirmation instrument for
  derivation-informed candidates).
- Checkpoint hygiene: no deletions; three new ~91 MB prediction memmaps
  expected on scratch.

### 2026-07-30: floor21_gate promoted; draw-201 confirmation recorded (no job)

- **No SLURM job.** User decision 2026-07-30: floor21_gate promoted on the
  natural track after Exp 44-46 and amendment 8, superseding floor-21's
  provisional adoption and gt_margin_adaptive's configuration-to-beat status
  (both remain in the pool). Confirmation script ran as a memmap subset over
  the already-scored full-pool prediction memmaps
  (`/capstor/scratch/cscs/cmeister747/unilid_analysis/full_test_eval`); no
  scoring pass, no job submitted.
- Artifact: `outputs/tables/floor21_gate_confirmation_201.md`. Instrument:
  balanced test draw, seed 201, 185,204 lines. Within-stratum macro-F1:
  floor21_gate overall 0.9741, tail 0.8685, magnets 0.8758, twins 0.9425, head
  0.9813; baseline overall 0.9809, tail 0.9086, magnets 0.9121, twins 0.9435,
  head 0.9817. Sanity gate (baseline overall row vs the recorded headline in
  `outputs/tables/two_sided_selection.md`): PASS, max abs diff 0.00004.
  Per-language collapse check on the draw-201 subset: 8 supported collapses
  (support >= 10, F1 drop > 0.10), attributable to the draw's own
  per-language support cap of at most 100 lines; the promotion-gate clause
  (C), computed on the far larger judge part in Exp 45/46, remains clean at
  zero supported collapses.

### 2026-07-29: combined-method plan reviewed twice, amended, and started (no jobs yet)

- **No SLURM jobs.** Two adversarial Opus reviews of the implementation plan for
  the combined method (plan file
  `~/.claude/plans/steady-finding-abelson.md`): a mechanics review and a
  design-consistency review against all prior recorded decisions. Findings and
  the resulting user decisions are recorded in `EXPERIMENTS_RESULTS.md` "Current
  state (2026-07-29)" and as amendment 7 plus standing constraint 5 in
  `EXPERIMENTAL_SETUP.md`; the pre-registration amendments are recorded at the
  top of the combined-method section in `EXPERIMENTS_PLAN.md`.
- Key measured finding recorded as a result about the instruments: balanced
  draw 101 reverses the method ranking for the primary quantity (six of six
  group leaders disagree with the held-out remainder; gt_min leads every group
  on the draw), so the assignment rule derives from the seed-301 40/60
  remainder split instead (new pre-registered constants RULE_SPLIT_SEED=301,
  RULE_SPLIT_FRACTION=0.40, BOOT_B=10,000, BOOT_SEED=0).
- Execution started with the no-scoring steps: the feature-provenance artifact
  (`outputs/tables/combined_feature_provenance.md`), the two solo-gate
  reference builds (`analysis/solo_gates.py`: adaptive gate, 100k target bar,
  over `pred_baseline.npy` and `pred_floor21.npy`), and the evidence base
  (`analysis/combined_evidence.py`, derivation part only). Both scripts passed
  an Opus pre-run review (no blockers; seven should-fixes applied before any
  run, including the sha256_base_W provenance gate on the unmod branch, the
  memmap provenance table, the fixed contrasts against gt_margin_adaptive, and
  the carried-set oracle). No scoring before the step-2 user checkpoint (rule
  sign-off).
- **Solo-gate builds: jobs 2930701 (unmod) and 2930702 (floor21)**, each 1
  node, 64 CPU, 100G, 04:00:00, submitted 2026-07-29 after a login-node
  attempt was killed at exit 137 (memory) during the 2,175,310-line affected
  scoring stage; the affected count matched the review's predicted volume, so
  the failure is resource-side, not code-side. Expected run time about 15
  minutes each once scheduled (precedent: job 2895821, 00:14:11).
- Checkpoint hygiene: no deletions proposed. Upcoming artifacts: two solo-gate
  prediction memmaps (~91 MB each), later three mixed-configuration memmaps,
  and the split index file on scratch
  (`full_test_eval/rule_split_seed301.npz`, ~360 MB, regenerable from the
  seed). The `results_apertus200k/corpus` calibration files and the
  `full_test_eval/` memmaps must be re-touched before the scoring job (14-day
  scratch purge).

### 2026-07-27: Exp 43, clean re-measurement of the Apertus 131k branch (job 2911700)

- **2911700** `unilid-ft-131k-fp64`: COMPLETED, 02:06:08. Full-test baseline
  evaluation of `glotlid_apertus131k_fp64.unilid` against the 100k model, into the
  separate scratch dir `full_test_eval_131k_fp64`. Report
  `outputs/tables/full_test_eval_131k_fp64.md`, per-language values in
  `outputs/diagnostic/full_test_131k_fp64_per_lang_prf.csv`.
- Code: `analysis/full_test_eval_131k.py` parameterized over model path, scratch
  directory, outputs, and report label (reviewed: no writes reach the earlier
  scratch directories, no variable shadowing, fingerprint gate prevents mixing two
  models' predictions; one shadowing defect was caught and fixed before submission
  because the chunk loop already binds `label`).
- Outcome (`EXPERIMENTS_RESULTS.md` Exp 43): branch verdict HOLDS on clean
  evidence (within-stratum overall -0.0090, tail -0.0395 [CI -0.0476, -0.0322];
  global tail 0.4269 against the 100k model's 0.5618), with the magnitude
  overstated by the bug (false positives into tail 32,211 clean against 51,926
  corrupted, within 0.2% of the Exp 30 counterfactual prediction of 32,161).

### 2026-07-27: Exp 42, both fp64 retrains completed and verified

- **2903767** `unilid-apertus131k-fp64`: COMPLETED, 04:36:18. **2903768**
  `unilid-apertus200k-fp64`: COMPLETED, 07:28:27. Artifacts
  `glotlid_apertus131k_fp64.unilid` (1,021,914,972 bytes) and
  `glotlid_apertus200k_fp64.unilid` (1,559,144,154 bytes) on scratch; results
  dirs `results_apertus{131k,200k}_fp64`. Both under the patched trainer
  (fork commits d0208d9, c5921a2). Note: the 131k retrain took 4.6 h against
  9.8 h for the corrupted run, because the collapsed Azerbaijani row no longer
  wastes iterations.
- Post-training gate and effect measurement (`analysis/fp64_retrain_check.py`,
  `outputs/tables/fp64_retrain_check.md`): azj_Latn repaired in both models
  (131k: 7 -> 22,704 entries above the row minimum, entropy 1.609 -> 3.025;
  200k: 1,798 -> 31,210, entropy 2.473 -> 3.000, confirming the partial collapse
  diagnosed in Exp 41); the 17 minority-script rows unchanged in kind, which
  confirms the vocabulary-coverage class is not a numerical failure; no new
  degenerate rows; only 20 of 1,940 (131k) and 18 of 1,940 (200k) rows moved by
  more than 1 nat, concentrated in the long-line corpora the diagnosis predicted
  (quc_Latn 9.43 nats, pcm_Latn, fas_Arab, mam_Latn).
- Pending and specified, not run: a full-test baseline evaluation of the
  retrained 131k model to re-measure the Exp 29 branch verdict without the
  corrupted row (`EXPERIMENTS_PLAN.md` open item 1).

### 2026-07-26: fp64 trainer adopted; both Apertus models retraining (jobs 2903767, 2903768)

- Patch adopted into the sentencepiece fork (user directive): commits d0208d9
  (double-precision forward-backward in the trainer's E-step; inference paths
  untouched) and c5921a2 (hard CHECK on non-finite expected counts instead of
  silent zeroing), pushed to cimeister/sentencepiece branch fixed-vocab-em.
- Installed binary replaced (~/.local/bin/spm_train; backup spm_train.pre_fp64).
  Acceptance test: isolated azj retrain under the new binary yields 22,704
  above-minimum entries, entropy 3.025, floor -18.93 (was 7 / 1.609 / -27.63).
- Full retrains of BOTH Apertus models launched under the corrected trainer
  (2903767 = 131k, 2903768 = 200k; new results dirs and artifacts with the _fp64
  suffix; corpus split reused; originals kept as the Exp 15/29 record). Full
  rather than per-language retrains so each model has single-provenance weights.
  Post-completion gates: degeneracy scan on both new models, then a full-test
  baseline evaluation of the 131k_fp64 model to re-examine the Exp 29 branch
  verdict with the bug removed.

### 2026-07-26: Exp 40-41, oracle bound, per-tag CommonLID, EM bug diagnosed (multi-agent)

- Exp 40 (login node): oracle upper bound over the carried set, 0.9525 overall
  (+0.0191 over the best single configuration); headroom concentrated in tail
  (+0.0724) and flat_magnets (+0.0998). `analysis/oracle_bound.py`.
- Exp 39 extension (job 2903415): per-tag macro-averaged CommonLID F1 (agent-
  authored, reviewed; both baseline gates reproduced: 0.8452 accuracy, 0.7228 tag
  F1; per-line predictions persisted). Both carried leaders slightly NEGATIVE on
  the objective-consistent metric (floor-21 -0.0046, adaptive -0.0061): CommonLID's
  109 tags barely contain the repaired tail labels.
- Exp 41 (two agents: source analysis on a scratch clone of the sentencepiece
  fork; 14-run corpus bisection): the fixed-vocabulary EM bug fully diagnosed:
  float32 forward-backward breaks the log-posterior identity on very long lines
  (upstream's 4,192-byte cap masked it; the pipeline passes 1,000,000); the fork's
  isfinite-to-zero M-step guard turns the overflow into a silent collapse. Trigger:
  azj line 81,302 (142,136 bytes, longest in all corpora). Graded corruption in at
  least 94 Apertus-branch corpora; azj-200k partially collapsed; 100k production
  model unaffected (not trained through this path). Double-precision patch built
  and VERIFIED in scratch (unpatched rebuild bit-identical to the installed
  binary; patched azj healthy at objective 229.8); NOT applied anywhere
  (user decision). Artifacts: session scratchpad `em_debug/` (fix.patch, fb_sim.py,
  both build trees) and `em_bisect/` (minimal 390-line trigger file, run log).

### 2026-07-25: Exp 36-39, adaptive verdict, azj re-run, user decisions, carried-set and CommonLID checks

- Exp 36 (job 2895821): gt_margin_adaptive ELIGIBLE flagged (ota only); both
  pre-run predictions confirmed; floor-21 retains top rank 5/5 draws at a 0.0002
  margin. Exp 37 (login node): azj collapse reproduced byte-identically in
  isolation; deterministic numerical breakdown in the fixed-vocab EM fork.
- User decisions: near-tie co-selection (NEAR_TIE_BAND=0.001; six configurations
  carried forward), primary quantity = macro-averaged per-language F1 on
  natural-distribution test data (equal language weighting, extreme low-resource
  exemptions allowed); precise terminology definitions committed to memory.
- Exp 38 (login node): carried-set per-language comparison on the held-out
  remainder; complementary structure (adaptive leads overall 0.9334 via lowmid,
  floor-21 leads tail/magnets, learned_bias leads head/twins with 602 strict
  per-language wins). `analysis/carried_set_comparison.py`.
- Exp 39 (job 2898246; login-node attempt OOM-killed at exit 137, resubmitted via
  SLURM): CommonLID check of the carried leaders, floor-21 +0.0040 and
  gt_margin_adaptive +0.0070 over the reproduced 0.8452 baseline; gate portable
  without refitting (9,886 firings). `analysis/commonlid_carried.py`, reviewed
  with fingerprint and batch-length hardenings applied pre-run.

### 2026-07-25: Exp 34-35, round-3 verdict, EM-degeneracy bounded, adaptive variant launched

- **2895683** `unilid-gt-margin-100k`: COMPLETED 08:37. gt_margin_all_100k
  ELIGIBLE, flagged (single outlier ota_Arab); not selected (balanced-val overall
  0.9744 vs floor-21 0.9800); veto overall 0.9330 top-tier. ota dig-in: 395 new
  FPs are gt-weight-side high-margin flips (fas->ota 295), not reassignment;
  quantile gates cannot catch them. `EXPERIMENTS_RESULTS.md` Exp 34.
- EM-degeneracy investigation (user-raised): `analysis/degeneracy_scan.py`,
  `outputs/tables/degenerate_rows.md`: 0 flagged rows at 100k (main results
  clean), 17/18 near-identical sets at 200k/131k = deterministic vocabulary
  coverage (no multi-byte merges for those scripts; csw EM converges normally);
  azj at 131k is the single anomalous run of 3,880. Scan adopted as the
  post-training gate. Exp 15 magnitude caveat recorded. `EXPERIMENTS_RESULTS.md`
  Exp 35.
- Pre-registered `gt_margin_adaptive` (user-requested N-adaptive gate strength:
  q_L = MARGIN_Q * (1 - min(N,HEAD_N)/HEAD_N), target bar unchanged), built via
  `run(gate='nonhead', target_n=100_000, adaptive_q=True)`,
  `slurm_gt_margin_adaptive.sh`.

### 2026-07-25: Exp 33, gt_margin_all judged; round-3 launched (jobs 2895566, 2895683)

- **2895566** `unilid-gt-margin-all`: COMPLETED (2026-07-25 08:13). Verdict REJECTED
  on both tracks (4 barely-head collapses, worst llb_Latn -0.3211 precision-side)
  despite the best natural-veto aggregates of any candidate (overall 0.9331, lowmid
  FPs 451k -> 140k). Mechanism and the three-round reassignment law in
  `EXPERIMENTS_RESULTS.md` Exp 33.
- **2895683** `unilid-gt-margin-100k`: round-3 candidate `gt_margin_all_100k`
  (pre-registered in Exp 33 before the run; reassignment-target bar raised to
  RES_CAP=100,000, gated set unchanged; `run(gate='nonhead', target_n=100_000)`,
  `slurm_gt_margin_all_100k.sh`). Verdict via `two_sided_report` when complete.

### 2026-07-24: Exp 32, outlier-tolerant clause + victim dig-ins + round-2 launch (login node + SLURM)

- Clause (C) revised per user decision (MAX_LANG_COLLAPSE_OUTLIERS=2; dig-in
  instead of rejection for 1-2 outliers). Re-judged: learned_bias flagged-eligible
  (llb), gt_min = uniform-track champion flagged (mev/sbs), gt_margin still
  rejected (4 = class pattern). Commit series through Exp 31 pushed
  (a3ff2c2..046e147, 13 commits).
- Dig-ins (`analysis/victim_digin.py`, `outputs/tables/victim_digins.md`): every
  victim is FP inflow at a non-head label, never recall loss. 131k pathology: azj's
  row is a degenerate EM outcome (7 estimated tokens); 18 degenerate 131k rows vs 0
  at 100k; azj alone explains ~2/3 of the 131k FP-into-tail increase
  (counterfactual in Exp 32).
- Round-2 candidate `gt_margin_all` pre-registered (gate all N<18k labels) and
  built via `analysis/gt_margin.py run(gate='nonhead')`, `slurm_gt_margin_all.sh`.

### 2026-07-24: Exp 31, gating reconsideration + 131k error overlap + gt_margin build (login node)

- User-directed gating review. Amendments (EXPERIMENTAL_SETUP.md, pending final
  user confirmation): (B)-overall softened to a bounded drop; dual-track verdicts
  (natural-traffic + uniform-prior `passes_uniform`); ITERATE lane. Delta review:
  amendments verdict-neutral for the first round; seed-201 discipline preserved
  (one confirmed candidate per track; multiplicity note recorded).
- Dual-track run: floor-21 unchanged as natural-track champion; gt_min wins
  uniform-track selection but FAILS the balanced-test collapse confirmation
  (2 supported languages beyond 0.10, worst sbs_Latn -0.182), so the uniform track
  has no champion and the baseline holds there.
- Exp 30 (131k error overlap, `analysis/error_overlap_131k.py`): the 131k does not
  repeat the baseline's errors (57.7% shared; 42.3% fixed incl. Indic wins; 403
  languages regress; tat<-azj alone carries 17,603 FPs).
- Pre-registered composition `gt_margin` (gt_min weights + head-targeted margin
  gate, tau recalibrated under gt_min; `analysis/gt_margin.py`, reviewed, no
  defects) built on the login node; judged in the same dual-track report.

### 2026-07-24: Exp 28, gt_min full-test pass judged (job 2884210)

- **2884210** `unilid-full-test-gt`: COMPLETED (2026-07-24 01:37, ~2.2 h scoring).
  Verdict via `two_sided_report`: REJECTED (veto overall and tail/magnet global F1
  drop; FPs into tail 22,404 -> 79,113; 12 supported languages beyond the collapse
  bound) despite the best selection-view numbers ever measured (balanced-val overall
  0.9841, tail 0.9769; full-test within-stratum tail +0.0656 CI [+0.0603, +0.0729]).
  Floor-21 remains selected. Mechanism and the next-round composition hypothesis in
  `EXPERIMENTS_RESULTS.md` Exp 28. Artifacts: `outputs/tables/full_test_gt.md`,
  `pred_gt_min.npy`, `fingerprint_gt.json`.

### 2026-07-23: Good-Turing counting pass + margin diagnostic launched (plan B4/B3)

- **2883714** `unilid-gt-counts`: COMPLETED (2026-07-23 23:17, ~35 min run).
  Outcome (`EXPERIMENTS_RESULTS.md` Exp 27): plateau overstates unseen mass for
  all 1,940 languages (exact GT raises 0 rows; tail 9x, head 12x median
  overstatement); spot-checks exact. gt_min matrix built and gate-checked; scoring
  pass `analysis/full_test_gt.py` reviewed (no defects) and submitted as
  **2884210** `unilid-full-test-gt` (64 CPU, 100G, 6 h; pred_gt_min.npy +
  fingerprint_gt.json on the full-test scratch dir; verdict via
  `two_sided_report`, gt_min added to CONFIGS). Original submission record: plan B4
  prerequisite: per-language T, n1, plateau mass under each language's own Viterbi
  segmentation of its training corpus (`analysis/gt_counts.py`, resumable per
  language, review-fixed torn-line resume handling). Output
  `outputs/diagnostic/gt_counts.csv`; feeds the one-sided-min GT candidate
  (`full_test_gt.py`, to be written when counts exist). 64 CPU, 100G, 8 h.
- **2883715** `unilid-margin-diag`: COMPLETED (2026-07-23 22:39). Outcome
  (`EXPERIMENTS_RESULTS.md` Exp 26): VIABLE; FP catch 76.8%, test-side suppression
  6.7%, cascade 53 lines, per-language AUC 0.90-0.9998, 26 languages excluded.
  Follow-up candidate `margin_q5` (reassign to runner-up), built login-node by
  `analysis/full_test_margin.py` (reviewed; agreement 1.0000; 17,773 reassignments):
  REJECTED on clause (C), szy_Latn -0.107 via 82 reassigned pwn_Latn FPs (see
  Exp 26 addendum in EXPERIMENTS_RESULTS.md, mechanism verified from memmaps).
  Pre-registered final variant `margin_q5_head` (reassign to highest-scoring head
  candidate) recorded before its run, then built and judged same evening: ELIGIBLE
  (all stages pass, szy collapse gone) but not selected; floor-21 ranks higher on
  both instruments (val overall 0.9800 vs 0.9799; veto tail F1 0.6337 vs 0.5321).
  Margin family closed this round; composition path recorded in the plan. Original
  submission record: plan B3:
  margins on the 22,522 FP-into-tail lines, the 7,735 true-tail lines, and up to
  2,000 train lines per tail language; tau_L at the 5th percentile of self-won train
  margins; MIN_CALIB_LINES=200 exclusion, logged (`analysis/margin_diagnostic.py`).
  Outputs `outputs/tables/margin_diagnostic.md`,
  `outputs/diagnostic/tau_per_lang.csv`. 64 CPU, 100G, 2 h. Constants MARGIN_Q=5,
  MIN_CALIB_LINES=200, CALIB_MAX=2000, TOPK_MARGIN=5, CALIB_SEED=0 pre-registered in
  the approved plan.
- Both modules reviewed pre-launch (Opus adversarial pass: no defects that would
  produce a wrong headline number; fixed before submission: gt_counts torn-line
  resume gap, a diagnostic-column double-count, a hardcoded stratum size; the
  encode_batch-vs-scorer segmentation choice documented as deliberate).

### 2026-07-23: Exp 25, adoption-rule instruments + first verdicts + pnt/ell audit (plan B1/B2, no SLURM job)

- Login-node analysis, no new scoring. Code: `passes_shortlist`/`passes_two_sided`
  (`analysis/hierarchical_pool.py`), `build_test_draw`/`rebuild_stability_draws`
  (`analysis/balanced_split.py`), `analysis/two_sided_report.py`,
  `analysis/label_audit.py`; `balanced_sweeps.py` sweeps now shortlist. Reviewed
  pre-run (Opus adversarial pass: no correctness defects; two flags fixed:
  `run_bias_refit` guard -> shortlist, `balanced_split.__main__` de-pipelined).
- Instrument amendment at first run: the veto originally excluded all six balanced
  draws, leaving median ~1 true tail line per language (veto tail recall 0.2188);
  amended to exclude draws 101/201 only, with a conditional exclusion for candidates
  fit on stability draws and a median>=10 runtime gate. Delta review of the amendment
  (same agent): exclusion set sound for the four current configs, llb_Latn rejection
  confirmed genuine on the full pool (drop 0.111), freq_prior trace confirmed; one
  defect found and fixed: clause (C) now judges only languages with
  MIN_COLLAPSE_SUPPORT=10 true veto lines (at n=4 one line flip moves F1 by
  0.11-0.14 and false-trips the 0.10 bound), and the fit-draw conditional is
  enforced via `CONFIG_FIT_DRAWS`. Verdicts unchanged under the fixed clause.
- Outcomes (`EXPERIMENTS_RESULTS.md` Exp 25): floor-21 ELIGIBLE and selected
  (provisional adopted configuration; supersedes the Exp 20 recall-view verdict);
  freq_prior ELIGIBLE not selected; learned_bias reg=5.0 REJECTED on the
  per-language collapse clause (llb_Latn -0.113, n=4,181). Label audit: 50/50
  sampled pnt_Grek<-ell_Grek residual lines are standard Modern Greek (provisional),
  so that residual is model error, not label noise.
- Artifacts: `outputs/tables/two_sided_selection.md`,
  `outputs/tables/label_audit_pnt_ell.md`, `outputs/diagnostic/balanced_val/`
  (val_lines_seed201.npy new; seeds 102-105 regenerated, manifest annotated).

### 2026-07-23: Apertus 131k (preliminary_mul) retrain launched (plan Track A)

- **2883222** `unilid-apertus131k-train`: COMPLETED 2026-07-24 02:47 (~9.8 h total,
  inside the 12 h window; no resume needed, unlike the 200k run). All 1,940
  languages trained; `glotlid_apertus131k.unilid` packed (1,021,914,972 bytes).
  Evaluation: `analysis/full_test_eval_131k.py` (reviewed, no defects; one optional
  hardening applied) submitted as **2885941** `unilid-full-test-131k` (64 CPU,
  100G, 8 h): scores the b=0 baseline over the full pool into the separate scratch
  dir `full_test_eval_131k/`, reusing y_true read-only after a language-list gate;
  report `outputs/tables/full_test_eval_131k.md` +
  `outputs/diagnostic/full_test_131k_per_lang_prf.csv`. **2885941** COMPLETED
  2026-07-24 (~3.5 h): NEGATIVE on both views (within-stratum tail -0.0437
  [CI -0.0515, -0.0371], overall -0.0113; FPs into tail 22,522 -> 51,926; balanced
  val also lower). `EXPERIMENTS_RESULTS.md` Exp 29; branch discontinuation
  recommended, user decision pending. Original submission record: PENDING
  at submission (Resources), RUNNING on nid007559 within 8 minutes. Purpose:
  test whether a multilingual-focus 131k vocabulary reverses the Apertus 200k tail
  regression (-3.4pp, Exp 15); the tokenizer is documented in
  `~/apertus-tokenizer-development/README.md` as the balanced-multilingual candidate
  with the highest compression on Indic, Chinese, and the low-resource tail.
- Plan item: approved plan `~/.claude/plans/steady-finding-abelson.md` Track A;
  tokenizer choice (`preliminary_mul` over the stock Apertus tokenizer) fixed by the
  user 2026-07-23.
- Init: vocab seeding from
  `/users/cmeister747/apertus-tokenizer-development/preliminary_mul/tokenizer.json`
  (sha256 6f8c5ca267c94975081045a46686ae68f8a1335b70a104810904389272117d41, vocab
  131,072, BPE, NFC; specials `<unk>/<s>/</s>/<pad>` at ids 0-3), uniform Unigram
  init, per-language fixed-vocab EM (forked spm_train), standard setup as the 200k
  retrain (`EXPERIMENTAL_SETUP.md` Apertus retrain entry).
- Data: `train.txt` (60,683,151 lines) via the 200k run's per-language corpus split,
  reused read-only with `--corpus-dir` (preflight `analysis/preflight_131k.py`
  verified 1,940 files, line total exact, all checks passed pre-submission).
- Script: `slurm_apertus_train_131k.sh` (12 h, 64 CPU, 400G, infra01/normal;
  auto-adds `--reuse-base` on resume). Expected: 12 h timeout at ~1,700/1,940
  languages plus one resume, matching the 200k run; then `convert.py` packs
  `$SCR/glotlid_apertus131k.unilid` (~1.02 GB).
- Artifacts: `$SCR/results_apertus131k/` (per-language tokenizers),
  `$SCR/glotlid_apertus131k.unilid`, logs `apertus131k_train_2883222.{out,err}`.
- Checkpoint hygiene: no deletions; `results_apertus131k/` is new and the 200k corpus
  dir is reused read-only.

### 2026-07-23: Exp 24, metric decomposition of the saved full-test predictions (analysis only, no SLURM job)

- Login-node analysis, no new scoring. Script `analysis/metric_decomposition.py`,
  reviewed pre-run by an adversarial agent (no defects). Inputs: the Exp 16 prediction
  memmaps (job 2784115) and the floor-21 memmap (job 2791722) on scratch, plus
  `outputs/diagnostic/full_test_per_lang_f1.csv` and `lang_diagnostic.csv`.
  Consistency gates, all passed: kept-line count 45,377,279; every recorded
  within-stratum table value reproduced to 6e-5; saved per-language F1 reproduced
  exactly.
- Purpose: decompose the stratum rows (within-stratum macro-F1) against global
  per-language F1/precision/recall. Outcome (`EXPERIMENTS_RESULTS.md` Exp 24): the
  tail deficit is precision (0.459), not recall (0.874); the rejected configurations'
  tail ranking reverses under the global view (floor-21 reaches tail mean F1 0.7655
  versus baseline 0.5618); neither the guard nor the balanced val can register this
  failure mode. Follow-ups proposed as `EXPERIMENTS_PLAN.md` Open paths block E; the
  metric-view question added to the Decision required item.
- Artifacts: `outputs/tables/metric_decomposition.md`,
  `outputs/diagnostic/full_test_per_lang_prf.csv`.

### 2026-07-18 — Prior-centered regularizer + non-content token tying (plan items 3, 11)

Code changes before these runs, both reviewed pre-launch by an adversarial agent:
- `analysis/learned_prior.py`: (a) prior-centered penalty `reg*||b - gamma*log(N+1)||^2`,
  grid `PRIOR_GAMMAS = {0, 0.25, 0.5} x REGS`; outputs to `learned_prior_centered.md` /
  `learned_bias_centered.npy` so the Exp 14 artifacts stay intact. (b) GRADIENT BUG FIX
  (found by the review, present since Exp 14): the softmax-NLL gradient accumulated soft
  counts over ALL examples' top-k candidates while the loss conditions on the true label
  being in the top-k (recall 0.9971); the fitted b was therefore not the minimizer of the
  stated objective. Fixed by restricting the soft counts to present examples; verified by
  finite differences (max error 3e-8 with 33/40 absent examples). The Exp 14 measured
  deltas remain valid measurements of the b that was produced; the gamma=0 rows of this
  run give the corrected plain-L2 fit for comparison. Caution note added to
  `EXPERIMENTS_RESULTS.md` Exp 14.
- `analysis/token_tying.py` (new): pure non-content token tying; tied sets digits_ws
  (298 tokens), nonalpha_ascii (479), nonalpha_all (1,291) classified on byte-decoded
  token text. No renormalization: the review derived that renormalizing injects a
  per-language per-token offset `-log Z_L` up to 0.36 nats/token concentrated on flat
  confusers, conflating mechanisms; pure tying leaves untied columns bit-identical
  (unit-verified). Special tokens (each exactly p=0.2 per row, the peak-probability
  artifact investigated 2026-07-18: HF Unigram score-0 specials normalized into every
  row, 0.8 of all mass, uniform across languages so argmax-neutral) are asserted and
  never touched.

- **2794210** `unilid-bal-sweeps` — COMPLETED 00:08:52 (2026-07-19). Outcomes
  (`EXPERIMENTS_RESULTS.md` Exp 23): floor equalization rejected at selection (tail
  -0.0177 to -0.0269 now visible); punctuation partial pooling alpha=300 PASSES (all
  strata non-negative, effect at measurability edge); learned-bias refit on balanced
  data reg=0.3 PASSES (sel overall +0.0016, tail +0.0299, magnets +0.0252; suppressed
  list = head/twin sinks nya/por/heb, not flat magnets; ||b||_inf 11.3). Pending before
  adoption of either: refit-per-draw stability, balanced-test draw, full-test passes,
  and the explicit objective decision on individual-language suppression. First sweeps
  under the
  balanced protocol (Exp 22), three experiments in one job
  (`analysis/balanced_sweeps.py`): (a) floor-equalization re-selection (plan item 14
  follow-up, F grid {-17,-19,-21,-23}); (b) punctuation partial pooling (plan item 15,
  212 neutral dp columns toward within-script means, lam = alpha/(N+alpha), alpha
  {300, 3000, 30000}); (c) learned-bias refit on balanced data (plan item 16,
  per-language alternating fit/selection halves, plain L2, corrected gradient,
  interpretability table of most-negative offsets). Selection only, no test scoring;
  baseline validated against the saved full-test predictions at the balanced-val lines
  (expected agreement ~1.0; gate 0.99). Reviewed pre-launch (adversarial agent: no
  blocking defects; empties provably absent from the pool; all 1,940 languages in both
  refit halves, fit >= 4 / sel >= 3 examples each; one overclaiming conclusion sentence
  rewritten to single-draw wording per review). Artifacts:
  `outputs/tables/balanced_{floor_eq,punct_prior,bias_refit}.md`,
  `learned_bias_balanced.npy` (only on a guard pass). Script:
  `slurm_balanced_sweeps.sh`.
- **2793541** `unilid-tying-dp` — COMPLETED 00:09:08 (2026-07-19). NEGATIVE: dp_global
  val overall -0.0014 (twins -0.0060), dp_script -0.0016 (twins -0.0103, failing the
  twin guard alone); tail/magnets flat; baseline selected. The cost concentrating in
  twins shows digit/punctuation usage rates are within-pair discriminative signal
  (consistent with Exp 4's 10.5% punctuation share of within-pair KL). Tying is closed
  at every curation level; see the Exp 18 final reading. Curated re-run of the token
  tying after the user's critique of the Exp 18 design (whitespace/newlines should
  never have been tied; their frequencies encode spacing conventions). Tied set: 212
  tokens whose decoded text is entirely ASCII digits + neutral punctuation
  (`.,:;!?()[]{}/\|@#*+=<>~`_"%^`), with documented linguistic exclusions (apostrophes,
  hyphens/dashes, ampersand, currency, Spanish inverted marks, typographic quotes, all
  whitespace including leading-space Ġ-variants, all non-ASCII punctuation). Two
  configs: dp_script (primary, tie within script groups so writing-system conventions
  never cross scripts; single-script languages unchanged by construction) and dp_global
  (comparison). Pure tying, no renormalization; same guard. Reviewed pre-launch
  (adversarial agent: no defects; character inventory of the tied set audited
  linguistically; single-language-script invariance verified bit-exact; Exp 18 default
  path confirmed byte-identical). Note two in-scope caveats from the review: tied
  tokens like `,000`/`.000` smooth the decimal-separator locale convention
  (intentional), and ASCII click-letter risk is nil (orthographic clicks are alphabetic
  Unicode). Init-from: recovered `glotlidc.unilid`. Artifact:
  `outputs/tables/token_tying_dp.md`. Script: `slurm_token_tying_dp.sh`.
- **2791722** `unilid-ft-floor21` — COMPLETED 01:41:04 (2026-07-19). Resubmission of
  2791583 after fixing an over-strict startup gate. Full-test verdict on floor-21:
  overall +0.0129 (point), head -0.0003, twins -0.0001, mid +0.0001, but tail -0.0204
  [CI -0.0257, -0.0161] and magnets -0.0164 [CI -0.0210, -0.0129]; accuracy +0.0009.
  The tail cost is real (unlike the learned bias's test-half scare); floor-21 is a
  global-precision-for-tail-recall trade, dominated by the learned bias at equal
  overall gain, and is NOT adopted. Third val-selected point overturned at full scale.
  Results: `EXPERIMENTS_RESULTS.md` Exp 20. Artifact:
  `outputs/tables/full_test_floor21.md`.
- **2791583** `unilid-ft-floor21` — FAILED 00:00:23 (2026-07-19 00:13, exit 1:0). The
  startup completeness gate demanded no UNSEEN anywhere in `pred_baseline.npy`, but the
  Exp 16 run wrote predictions only for kept lines, so the 250,000 val positions are
  legitimately UNSEEN there (verified: 250,000 UNSEEN total, 0 on kept lines). Failed
  before any scoring or state changes; the gate now checks kept lines only. The pre-run
  review had asserted this check was verified against the memmaps; that verification
  claim was wrong. Original entry follows. Full-test evaluation of the
  Exp 20 guard-selected floor-21 matrix (plan item 14 follow-up): one scoring pass over
  the 45,377,279 non-val lines under the clamped matrix, compared against the SAVED
  Exp 16 baseline memmaps (job 2784115; opened read-only). Deterministic matrix rebuild
  + sha256 fingerprint; per-line label gate against the saved y_true; bidirectional
  val-partition cross-check; resumable. Decides whether floor-21 (test-half overall
  +0.0030) becomes a result of record or joins the tail-risk record: the test-half tail
  point is -0.0623 on ~35 items. Reviewed pre-launch (adversarial agent: no defects;
  walltime bound <= 5h10m vs 8h request; special-column assert added from its
  suggestion). Note for interpretation: absolute levels here use the zero-bias scorer
  path (like Exp 16), whereas the Exp 20 sweep used the unbiased predict path; the
  reported quantity is the internally consistent baseline-relative delta. Script:
  `analysis/full_test_floor21.py`, `slurm_full_test_floor21.sh`. Artifact:
  `outputs/tables/full_test_floor21.md`.
- **2791444** `unilid-flooreq-hier` — COMPLETED (2026-07-18). Outcomes: (a) floor
  equalization POSITIVE on overall, guard selects floor-21, test-half overall +0.0030
  [CI +0.0016, +0.0044], twins/head flat, magnets -0.0108 (crosses 0), tail -0.0623 (CI
  touching 0 on ~35 items; full-test check required before adoption; see Exp 20).
  (b) macrolanguage hierarchy NULL: deltas -0.0000 everywhere; macro-aware accuracy
  0.9680 vs exact 0.9603 measures the within-macro ceiling (Exp 21). Two experiments in
  one job
  (plan items 14 and 13, run in that order). (a) Downward floor equalization
  (`analysis/floor_equalization.py`): plateau clamped to min(floor_L, F), F in
  {-17, -19, -21, -23}; measured n_modified per F: 452 / 1,821 / 1,940 / 1,940; premise
  gate added (abort if corr(floor, log10 N) > -0.5). (b) Macrolanguage-hierarchical
  decision (`analysis/macro_hierarchy.py`): parameter-free logsumexp group marginal over
  SIL macrolanguages from top-50 candidates; guard is accept/reject; table reports
  hierarchical-vs-baseline deltas unconditionally (no tuning). Both reviewed pre-launch
  (adversarial agent: no correctness or crash bugs; Rust empty-string top-k early-return
  and K=50 marginal truncation verified; two reporting items fixed before submission;
  known cosmetic caveat: the hierarchy module's baseline uses top-k tie-breaking, which
  can differ from best_of on exact float32 ties). Init-from: recovered
  `glotlidc.unilid`. Artifacts: `outputs/tables/floor_equalization.md`,
  `macro_hierarchy.md`. Script: `slurm_floor_eq_hierarchy.sh`.
- **2790174** `unilid-backoff-wals` — COMPLETED 00:17:03 (2026-07-18). NEGATIVE, same
  monotone pattern as 2790155 within 0.0016 at every config (val overall -0.0036 at
  wals_lift_a300 down to -0.0304 at alpha=30000; tail/magnets drop at alpha >= 3000);
  nothing passes the guard; baseline selected. Genealogical grouping fidelity is
  immaterial to the outcome; the mass-lifting operation is the refuted element. Results:
  `EXPERIMENTS_RESULTS.md` Exp 19. Original entry follows. WALS genealogical grouping
  for the back-off (plan item 12, user-requested true families): tiered per-language
  fallback genus-within-script -> family-within-script (each requires >=
  `MIN_BACKBONE_GROUP = 3` backbone members) -> script. Source:
  `data/wals_languages.csv` (WALS export copied from `~/tokenizer-lm/data`, provenance
  in `data/README.md`; covers 1,159/1,940 languages; the parity-aware grouped config was
  evaluated as an alternative and rejected as primary source at 207/1,940 coverage).
  Tier assignment: genus 535, family 360, script 1,012, none 33; 37/96 tail languages
  get a genealogical tier (`outputs/diagnostic/backoff_groups_wals.csv`). Same six
  mode x alpha configs and guard as 2790155. Reviewed pre-launch (focused adversarial
  agent: no number-corrupting defects; nested-group semantics and None cascade verified
  on the real arrays). CAVEAT for interpretation: in genus groups with exactly 3
  backbone members (21 of 50 eligible genus groups), the EXCLUDE_K=3 confuser exclusion
  empties and falls back to all-but-self, so the confuser-excluded property is weaker at
  the genus tier. Artifact: `outputs/tables/family_backoff_wals.md`. Script:
  `slurm_family_backoff_wals.sh`.
- **2790155** `unilid-backoff` — COMPLETED 00:17:42 (2026-07-18). NEGATIVE: every config
  reduces val overall (lift_a300 -0.0028 ... lift_a30000 -0.0289; full mode within
  0.0007 of lift throughout); at alpha >= 3000 val tail drops 0.8710 -> 0.8387 and
  magnets 0.8797 -> 0.8609. Nothing passes the guard; baseline selected (test deltas
  zero by construction). Mechanism reading: lifting unseen-token mass toward the script
  mean makes languages MORE accepting of group-plausible foreign material, increasing
  theft; this is the direction Exp 10 warned about (small languages already
  under-penalize unseen tokens). The untried direction implied by Exp 10 is floor
  EQUALIZATION downward, not group-informed lifting. Original entry follows. Script-mean
  back-off at floor
  positions (plan item 12): each language's exact floor plateau (74,617-99,810 entries
  per row, measured; the emergent resource-tied unseen-token constant) is replaced by
  `lam_L * m_G(t)` with `lam_L = alpha/(N_L+alpha)`, m_G = confuser-excluded resource-
  weighted script backbone mean; modes lift/full x alpha {300, 3000, 30000}; observed
  tokens and specials bit-identical; no renormalization. 33 languages without a
  same-script backbone stay unmodified. Guard-selected on val, test half once. Reviewed
  pre-launch (adversarial agent: no number-corrupting defects; units of prior vs
  log-weights confirmed consistent because all rows are exactly normalized). Init-from:
  recovered `glotlidc.unilid`. Script: `analysis/family_backoff.py`,
  `slurm_family_backoff.sh`. Artifact: `outputs/tables/family_backoff.md`.
- **2790077** `unilid-learnprior` — COMPLETED (2026-07-18). Prior-centered learned bias
  sweep (18 fits, corrected gradient), val-guarded selection, test half once. Init-from:
  recovered `glotlidc.unilid`. Selected gamma=0.25, reg=10: test-half overall +0.0117
  [CI +0.0104, +0.0130], twins +0.0124, head +0.0089, magnets -0.0052 (crosses 0), tail
  -0.0320 (the noisy 250k-half tail; full-test read pending). Note: under the corrected
  gradient the previous operating point (gamma=0, reg=5) now FAILS the guard (val
  magnets -0.0119), so the gradient fix changed the fit materially. Marginal gain over
  plain-L2 reg=5 (+0.0112 on the same half); needs full-test confirmation before any
  supersession. Artifacts: `outputs/tables/learned_prior_centered.md`,
  `learned_bias_centered.npy`.
- **2790078** `unilid-tying` — COMPLETED 00:10:42 (2026-07-18). Non-content tying sweep
  (3 tied sets), val-guarded selection. NEGATIVE: every tied set reduces val overall
  (digits_ws -0.0010, nonalpha_ascii -0.0063, nonalpha_all -0.0078); nothing passes the
  guard; baseline selected (all test deltas zero by construction). Refinement hypothesis
  for a possible follow-up: the tied sets include the whitespace tokens (Ġ, Ċ), and
  whitespace frequency is genuinely language-discriminative (spaced vs unspaced
  scripts), so the negative may be dominated by tying whitespace; a digits+punctuation
  set that excludes whitespace was not run. Artifact: `outputs/tables/token_tying.md`.

### 2026-07-16 — Full-test-set evaluation (plan "Next methods" item 10, part 1)

- **2784115** `unilid-fulltest` — COMPLETED 05:06:50 (2026-07-18 00:02). Scores the 100k
  model on the full GlotLID test set minus the 250k val lines (45,377,279 lines) for three
  FIXED configurations: baseline, frequency prior gamma=0.5, learned bias reg=5.0
  (`outputs/tables/learned_bias.npy` from job 2731802). No selection: pure evaluation to
  tighten the stratified deltas; on the 250k test half every one of the 96 tail languages
  has <= 2 examples (67.7% have zero VAL examples), so the open question was whether the
  learned bias's tail delta (-0.0320, CI touching 0) is real. Outcome
  (`EXPERIMENTS_RESULTS.md` Exp 16): learned bias overall +0.0129, tail -0.0018
  [CI -0.0035, -0.0001] (the -0.0320 was split noise), magnets -0.0082, accuracy
  0.9608 -> 0.9751; frequency prior tail -0.0182 [CI -0.0225, -0.0146], i.e. NOT
  tail-safe (its Exp 14 tail 0.0000 was a tail-invisibility artifact). Baseline
  agreement with recorded predictions 0.9951 (check passed). Script:
  `analysis/full_test_eval.py` + `slurm_full_test_eval.sh`.
  Safety: seed-42 val-line reconstruction cross-checked against `val_mask.npy`; every
  sampled test-half line's label validated against the sample pickle (abort on first
  mismatch); zero-bias predictions validated against recorded UniLID predictions
  (abort if agreement < 0.99); resumable chunked memmaps on scratch guarded by a config
  fingerprint (sha256 of all three bias vectors + language list + chunking) so a resume
  with changed inputs aborts instead of mixing configurations. Reviewed before launch by
  an adversarial agent: numeric path confirmed correct; the fingerprint, atomic progress
  writes, and the baseline-agreement check were added from its findings. Bootstrap CIs
  (B=1000) for strata <= 3M examples (tail ~6k, magnets ~61k); point deltas only for
  twins/head/overall (n > 3M, item-level CI half-width < 0.001).
  Artifacts: `outputs/tables/full_test_eval.md`,
  `outputs/diagnostic/full_test_per_lang_f1.csv` (per-language F1 for plan items 5-6),
  memmaps + fingerprint in `/capstor/scratch/.../unilid_analysis/full_test_eval/`.

### 2026-07-10 — Selection-guard fix + re-selection re-runs (plan "Next methods" item 1)

Code change before these runs: the val-based selection guard in
`analysis/{hierarchical_pool,prior_sweep,learned_prior}.py` was unified into
`passes_guard` (`hierarchical_pool.py`) with `GUARD_STRATA = (tail, magnets, twins, head)`
and `GUARD_TOL = 0.01`; a config is eligible only if val overall macro-F1 improves and no
guarded stratum drops more than 0.01. Rule provenance and the tolerance decision (user
choice among {0.002, 0.01, 0.02} on 2026-07-10) are in `EXPERIMENTAL_SETUP.md`
("Selection guard"). `learned_prior.REGS` extended to {0.3, 1, 3, 5, 7, 10}.
`learned_prior.py`'s no-eligible-reg fallback now selects the baseline b=0 (the old code
fitted `max(REGS)` while printing "smallest reg"). A review agent found, and we fixed, a
rounding defect: `prior_sweep.py` had guarded on 4dp-rounded values; the guard and the
argmax now use unrounded values. Expected selections, precomputed from the saved val
tables (session scratchpad `test_guard.py`, all confirmed): 100k freq prior keeps
gamma=0.5; Apertus freq prior rejects all gammas (baseline selected, negative result);
learned bias moves off reg=0.3 (which costs val magnets -0.0318) to reg in {5, 7, 10}.

All three jobs: 500k seed-42 sample, val/test split `outputs/diagnostic/val_mask.npy`,
strata `outputs/diagnostic/lang_diagnostic.csv`, account `infra01`, 400 GB, expected
runtime 10-30 min each once scheduled. No checkpoints deleted for these runs (post-hoc
analyses; nothing to clean).

- **2731818** `unilid-commonlid` — COMPLETED (2026-07-10 22:22). CommonLID out-of-domain
  re-evaluation with the reg=5.0 bias (`learned_bias.npy` was overwritten by 2731802, so
  the recorded 0.8936 from job 2640066 belongs to the de-selected reg=0.3 vector and is
  superseded). Macro-aware accuracy: baseline 0.8452 -> freq prior gamma=0.5 0.8518
  (+0.0067, unchanged) -> learned bias reg=5.0 **0.8879** (+0.0427). Artifact:
  `outputs/tables/commonlid_eval.md` (overwritten).
- **2731802** `unilid-learnprior` — COMPLETED 00:09:05 (2026-07-10 22:15). Learned
  per-language bias re-run under the fixed guard + extended REGS. Init-from: recovered
  `glotlidc.unilid` (100k). Guard selected reg=5.0 (val magnets -0.0075; reg<=3 fail on
  magnets). TEST: overall +0.0112 [CI +0.0099,+0.0124], head +0.0094, twins +0.0135,
  magnets +0.0051 (CI crosses 0), tail -0.0320 [CI -0.0588,+0.0000], accuracy
  0.9603 -> 0.9749 (+0.0147). Supersedes job 2640065's selection (reg=0.3, +0.0180).
  NOTE: val tail macro-F1 is 0.8710 for every reg (and every gamma in the prior sweeps),
  so the val guard has no sensitivity on the tail stratum; the test tail movement was
  invisible to selection. Recorded as a guard limitation in `EXPERIMENTS_RESULTS.md`.
  Artifacts: `outputs/tables/learned_prior.md`, `learned_bias.npy` (both overwritten;
  npy is now the reg=5.0 fit).
- **2731803** `unilid-prior-apertus` — COMPLETED 00:15:25 (2026-07-10 22:21).
  Frequency-prior sweep on the Apertus 200k model under the fixed guard. Init-from:
  `glotlid_apertus200k.unilid`. Outcome as precomputed: NO gamma eligible (every gamma
  >= 0.25 drops val tail by >= 0.032), baseline selected, all test deltas zero. Replaces
  job 2649123's flawed gamma=3.0 selection (Exp 15 guard flaw); the frequency prior is
  rejected on the Apertus model. Val table identical to the 2649123 run (deterministic
  rescore). Note: the in-run "agreement with recorded UniLID preds" print (0.9608) checks
  against the 100k model's stored predictions, so below 1.0 is expected when model_path
  is overridden; the check is meaningful only for the default model. Artifact:
  `outputs/tables/prior_sweep_apertus.md` (overwritten).
- **2731804** `unilid-prior` — COMPLETED 00:16:36 (2026-07-10 22:23). Frequency-prior
  sweep on the 100k model under the fixed guard. Init-from: recovered `glotlidc.unilid`.
  Outcome as precomputed: gamma=0.5 re-selected (val magnets -0.0081 within tolerance);
  val and test tables identical to job 2639127 (deterministic rescore; gamma=0 agreement
  with recorded preds 0.9951, matching the known baseline self-agreement). The Exp 14
  frequency-prior result stands unchanged under the fixed guard; artifact header now
  records the guard rule. Artifact: `outputs/tables/prior_sweep.md` (overwritten).

### 2026-06-28 to 2026-06-29 — Prior redirect + Apertus retrain

> **Account change:** these runs used `--account=infra01` (not `a139`; the repo `SETUP.md`
> and older scripts are wrong — see memory `unilid-slurm-account`). All on CSCS Clariden,
> Python `/users/cmeister747/.pyenv/versions/3.11.5/bin/python3`, 500k uniform sample
> (`seed=42`), val/test split `outputs/diagnostic/val_mask.npy`, strata from
> `outputs/diagnostic/lang_diagnostic.csv`. Init-from: recovered `glotlidc.unilid`
> (100k model) unless noted.

Infra done this window: installed CPU `torch` (the default `pip install torch` pulled the
multi-GB CUDA build and blew the home quota); made the `transformers` import lazy in
`UNILID/unilid/api.py` so training doesn't clobber the custom Rust `tokenizers` build; added
Rust methods `best_of_cached_weight_sets_biased_batch` (per-language bias before argmax) and
`top_k_of_cached_weight_sets_batch` (top-k candidate scores for the learned-bias fit), both
rebuilt with `maturin develop --release` and validated (biased: zero-bias == unbiased;
top-k: gradient of the softmax fit checked to 2e-10).

- **2649123** `unilid-prior-apertus` — COMPLETED 00:16:19 (2026-06-29 23:11). Frequency-prior
  sweep on the Apertus 200k model. Plan: Exp 15. Init-from: `glotlid_apertus200k.unilid`.
  Guard selected gamma=3.0; overall +0.0203 but tail -0.0945, magnets -0.1102 (guard flaw).
  Artifact: `outputs/tables/prior_sweep_apertus.md`. Results: `EXPERIMENTS_RESULTS.md` Exp 15.
- **2641940** `unilid-apertus-train` — COMPLETED 01:41:24 (2026-06-29 13:45). Resume of 2639097
  (`--reuse-corpus --reuse-base --skip-existing-langs`); trained the last 250 languages and
  packed `glotlid_apertus200k.unilid` (1.56 GB, 1,940 langs, 200k vocab) via `convert.py`.
- **2639097** `unilid-apertus-train` — TIMEOUT 12:00:25 (2026-06-29 05:55). Standard-setup
  Apertus 200k retrain (no MAP prior): Apertus V2 200k byte-level BPE seeded into Unigram,
  SP per-language re-estimation on recovered `train.txt` (60,683,151 lines),
  `--max-base-samples-per-lang 10000 --lang-batch-size 20`. Reached 1,690/1,940 at the wall.
  Plan: Exp 15 (replaces the abandoned MAP-EM Exp 14 plan item). Script: `slurm_apertus_train.sh`.
- **2640066** `unilid-commonlid` — COMPLETED 00:06:13 (2026-06-28 21:05). CommonLID out-of-domain
  eval with the priors (baseline / freq gamma=0.5 / learned bias). 373,230 web lines, 109 tags.
  Macro-aware acc 0.8452 -> 0.8518 -> 0.8936. Artifact: `outputs/tables/commonlid_eval.md`.
- **2640065** `unilid-learnprior` — COMPLETED 00:07:56 (2026-06-28 20:56). Learned per-language
  bias: top-20 candidate extraction on val, L2-regularized softmax fit (reg swept {0.3,1,3,10},
  reg=0.3 selected on val), exact test eval via biased scorer. Test macro-F1 0.9454 -> 0.9638
  (+0.0180). Artifacts: `outputs/tables/learned_prior.md`, `learned_bias.npy`. Plan: Exp 14.
- **2639127** `unilid-prior` — COMPLETED 00:27:28 (2026-06-28 18:27). Frequency-prior sweep
  `b_L=gamma*log N_L` on the 100k model, gamma in {0..5}. gamma=0.5 selected: macro-F1 +0.0058.
  Artifact: `outputs/tables/prior_sweep.md`. Plan: Exp 14.
- **2639065** `unilid-hpool` — COMPLETED 00:28:16 (2026-06-28 18:19). Expanded Stage 1 sweep
  (uniform shrink, liability-scaled shrink, entropy sharpening; 11 configs). NEGATIVE:
  sharpening collapses magnets to ~0, shrink neutral at best. Artifact:
  `outputs/tables/hierarchical_pool.md`. Plan: Exp 13.
- **2638804 / 2638803** `unilid-hpool` / `unilid-commonlid` — COMPLETED 2026-06-28 ~17:28.
  Original Stage 1 shrinkage prototype + first CommonLID eval (Exp 13, Exp 12).
- **2626411 / 2626402** `unilid-hpool` / `unilid-commonlid` — FAILED 00:00:11 (2026-06-27).
  Early submissions before the CPU-torch / lazy-transformers infra fixes; import errors.

### 2026-06-26 — Artifact recovery + infra
- Recovered from Google Drive (folder `19sRPRiFHX8Lk3vZWlNGl0zzA88eAZ3Yx`) to scratch:
  `train.txt` (60,683,151 lines), `glotlid_correct_test.txt`, `glotlid_train_counts.json`,
  and all 5 prediction files (UniLID, DeepSeek, Qwen, Marg, fastText). Full 744 MB model
  recovered from the polybox link in `UNILID/README.md` to
  `/capstor/scratch/cscs/cmeister747/unilid_analysis/glotlidc.unilid` (verified full
  1940x100k matrix; the in-repo copy is truncated to ~7%).
- Rebuilt the custom Rust tokenizers (`maturin develop --release`); verified
  `set_weight_sets` + scoring on eng/deu/fra.
- Staged Apertus V2 200k tokenizer (`swiss-ai/apertus-tokenizer-development`,
  `preliminary_mul_200k`, byte-level BPE) to
  `/capstor/scratch/cscs/cmeister747/unilid_analysis/apertus_v2_200k/tokenizer.json`.

### 2026-06-24 — Exp 10 error analysis
- 7-cut agent workflow over a 28,527-error stratified sample + per-token decomposition +
  weight-matrix audit. Outcome: under-fit-tail attractor mechanism; two attractor types;
  ~80-85% recoverable ceiling. Drives the pooling direction. See `EXPERIMENTS_RESULTS.md`
  Exp 10 and project memory `unilid-error-analysis-findings`.

### Exp 11 — Per-language diagnostic (ongoing)
- `analysis/diagnostic.py`: structural flatness/entropy + full pairwise symmetric-KL +
  empirical magnet ratio (val half only) -> per-language category classifier.
  Output `outputs/diagnostic/lang_diagnostic.csv`.

---

## Family: Discriminative re-weighting and low-resource transfer (Exp 8–9)

**Window [inferred]:** ~2026-04-06 to 2026-04-07. Plan items: `EXPERIMENTS_PLAN.md` Exp 8,
Exp 9. Designed during the `/ultraplan` discussion on 2026-04-06 (recovered prompts 77–88).

### Exp 9 — Distribution transfer (related-language 9a + script-average 9b)
- **Job:** `1808399` (`unilid-transfer`), COMPLETED, 25 min, 400 GB.
- **Submission script:** `slurm_transfer_sweep.sh` → `analysis.transfer_sweep.generate_transfer_sweep(sample_size=500_000)`.
- **Hypothesis:** interpolating under-fit low-resource distributions toward a related
  high-resource language / script-average raises low-resource accuracy.
- **Search space:** `lambda ∈ {0.0,0.1,…,1.0}` (11), two approaches → 22 configs;
  223 related-language transfer pairs.
- **Init from:** scratch (analysis over the existing `glotlidc.unilid` weights; no model
  training).
- **Outcome:** 9a: <500 +10.6pp at λ=0.3, overall −1.3pp at the same λ; 9b: overall stays 0.960–0.961 across
  λ∈[0.1, 1.0], <500 does not exceed the 0.789 baseline at any λ<1.0. See `EXPERIMENTS_RESULTS.md` Exp 9.
- **Artifacts:** `outputs/tables/transfer_sweep.md`, `outputs/figures/transfer_sweep.png`.

### Exp 8a — Heuristic discriminative weighting
- **Job:** `1808414` (`unilid-disc8a`), COMPLETED, 13 min, 400 GB.
- **Submission script:** `slurm_discriminative_heuristic.sh` → `analysis.discriminative_finetune.generate_heuristic_discriminative(sample_size=500_000)`.
- **Hypothesis:** variance-based token up-weighting improves within-cluster separation.
- **Search space:** Setup A & B at `α ∈ {0.0,0.5,1.0,2.0,5.0}`, Setup C at
  `β ∈ {1.0,5.0,10.0}`, across the 7 confusion clusters.
- **Outcome:** all three setups reduce accuracy at every tested parameter; per-cluster accuracy 0 across all 7
  clusters for A and B at α≥1.0. See `EXPERIMENTS_RESULTS.md` Exp 8a.
- **Artifacts:** `outputs/tables/discriminative_heuristic.md`.

### Exp 8b — MMI discriminative fine-tuning (NOT RUN)
- **Status:** designed but not implemented. `analysis/discriminative_finetune.py` marks it
  `TODO` in the module docstring; no submission script exists. Carried in
  `EXPERIMENTS_PLAN.md` as not-started. This is where the session ended.

---

## Family: Training-data analysis (Exp 7)

**Window [inferred]:** ~2026-04-06. Plan item: `EXPERIMENTS_PLAN.md` Exp 7.
- **Run location:** login node, single pass, ~30 min (no SLURM job ID recorded).
- **Code:** `analysis/train_data_analysis.py`. Input: full training corpus
  (`config.TRAIN_FILE = $SCRATCH/train.txt`, 60,683,151 lines), downloaded from Google
  Drive on 2026-04-06 (recovered prompts 67–68).
- **Sub-analyses run:** 7.1 domain distribution, 7.4 per-language corpus quality, 7.5 script
  verification. **Deferred:** 7.2 mislabeling, 7.3 overlap (recovered prompt 82, "Skip
  analysis 7.2 and 7.3 for now").
- **Outcome:** 98.1% "other" domain; low-resource corpora short and small-vocab; 20
  off-script languages. See `EXPERIMENTS_RESULTS.md` Exp 7.
- **Artifacts:** `outputs/tables/train_data_analysis.md`,
  `outputs/figures/train_{domain_stacked,quality_scatter,script_purity}.png`.

---

## Family: Floor sweep (Exp 6)

**Window [inferred]:** ~2026-04-06 (recovered prompt 73).
- **Job:** `1806690` (`unilid-floor`), COMPLETED, 4.5 min, 400 GB.
- **Submission script:** `slurm_floor_sweep.sh` → `analysis.floor_sweep.generate_floor_sweep(sample_size=500_000)`.
- **Hypothesis:** clamping per-language weights at a higher floor (finite OOV probability)
  improves accuracy.
- **Search space:** `floor ∈ {None, -22.0, -15.0, -10.0}`.
- **Outcome:** floor=−22: 0/500k predictions changed; floor=−15: 3,372 changed, net −109; floor=−10: accuracy
  0.960 → 0.916. See `EXPERIMENTS_RESULTS.md` Exp 6.
- **Artifacts:** `outputs/tables/floor_sweep.{md,tex}`, `outputs/figures/floor_sweep.png`.
- **Note:** code comment in `floor_sweep.py` cites "OOV at -1e30"; user flagged this as not
  present in the repo (recovered prompt 75). Comment is inaccurate.

---

## Family: Normalization (Exp 2 re-classification + Exp 5 alpha sweep)

### Exp 5 — Alpha sweep (partial normalization)
**Window [inferred]:** ~2026-04-05/06 (recovered prompt 66, "did the alpha sweep finish").
- **Job:** `1804584` (`unilid-alpha`), COMPLETED, 12 min, 400 GB.
- **Submission script:** `slurm_alpha_sweep.sh` → `analysis.normalized_predict.generate_alpha_sweep(sample_size=500_000)`.
- **Search space:** `alpha ∈ {0.0,0.1,…,1.0}` (11), scoring `score / n_tokens^alpha`.
- **Outcome:** best `alpha=0.1` accuracy 0.961 (+0.001 over `alpha=0.0`); accuracy decreases monotonically for
  α>0.1. See `EXPERIMENTS_RESULTS.md` Exp 5.
- **Artifacts:** `outputs/tables/alpha_sweep.{md,tex}`, `outputs/figures/alpha_sweep.png`.

### Exp 2.5 — Full re-classification with normalized scores
**Window [inferred]:** ~2026-04-04 (recovered prompt 55).
- **Job:** `1795556` (`unilid-norm`), COMPLETED, 2.5 min, 400 GB.
- **Submission script:** `slurm_normalized.sh` → `analysis.normalized_predict.generate_normalized_analysis(sample_size=500_000)`.
- **Implementation:** added `best_of_cached_weight_sets_normalized` to the Rust Unigram
  tokenizer fork (`UNILID/tokenizers/`); PyO3 bindings; `predict_normalized` wrappers.
- **Outcome:** normalization drops accuracy 0.960 → 0.885; raw rescore reproduces originals
  exactly (100% agreement, validates implementation). See `EXPERIMENTS_RESULTS.md` Exp 2.
- **Artifacts:** `outputs/tables/normalized_comparison.{md,tex}`.

---

## Family: Tokenization length bias (Exp 2 token-delta + counterfactual)

**Window [inferred]:** ~2026-03-28 to 2026-04-03. This family hit repeated OOM/timeout
failures before succeeding; failures are recorded with the same rigor as the success.

- **Job `1791511`** (`unilid-lenbias`), COMPLETED, 5h, 400 GB. Full run: token deltas +
  scores + pairwise normalization counterfactual over 1,789,423 misclassifications
  (12h walltime). **The result of record for Exp 2.3–2.4.**
- **Job `1790440`**, TIMEOUT, 6h, 400 GB. Score computation roughly doubled runtime; hit
  the 6h walltime. Fixed by requesting 12h.
- **Job `1789048`**, OOM, 12 min, 128 GB. After a code refactor; still needed 400 GB.
- **Job `1752234`**, COMPLETED, 3h, 400 GB. Token-delta only (older code, no scores).
- **Job `1750406`**, OOM, 1.75h, 128 GB. Streaming fix applied; tokenizer cache still OOM
  at 128 GB.
- **Job `1747559`**, OOM, 2.5h, 128 GB. First attempt; storing texts in the pickle caused
  the OOM. Led to the decision to keep raw texts out of the sample pickle (see `SETUP.md`).
- **Memory lesson:** the per-language tokenizer cache (~1,895 tokenizers × 100k vocab)
  needs ~250 GB; 400 GB requests are required for any tokenizer-heavy job.
- **Code:** `analysis/length_bias.py`. **Artifacts:** `outputs/tables/length_bias.{md,tex}`,
  `outputs/figures/length_bias_histogram.png`.

---

## Family: Per-language distribution + token classification (Exp 3, Exp 4)

**Window [inferred]:** ~2026-04-05 to 2026-04-06 (recovered prompts 60–64).
- **Run location:** login node (no SLURM job IDs recorded).
- **Exp 3** (`analysis/distribution_analysis.py`): KL(lang‖base) for 1,940 languages;
  15 related-pair comparisons (symmetric KL, correlation, MAD); top divergent tokens.
- **Exp 4** (`analysis/token_classification.py`): heuristic 8-category classifier over the
  300 top divergent tokens (15 pairs × 20).
- **Outcomes:** low-resource (<500 sample) mean KL from base 0.32 vs 0.68–0.71 at 5k+; in token-category
  classification of top KL-contributors, morphological affixes 32.6% + content words 22.8% + function words 15.7%
  = 71.1% of category share; script/encoding 0%. See `EXPERIMENTS_RESULTS.md` Exp 3, Exp 4.
- **Artifacts:** `outputs/tables/distribution_analysis.{md,tex}`,
  `token_classification.{md,tex}`; figures `kl_vs_training_size.png`,
  `pairwise_logprob_scatter.png`, `pairwise_kl_vs_training.png`,
  `token_categories_stacked.png`.

---

## Family: Multi-system comparison + tables (Exp 1)

**Window [inferred]:** ~2026-03-26 to 2026-03-28. The project's first work.
- **Job `1747558`** (`unilid-tables`), COMPLETED, 16 min, 64 GB. Full-dataset tables
  (45.6M samples) via `analysis.run_all --sample-size 45627279 --format both`
  (`slurm_tables.sh`).
- **Sampling:** 500k uniform, `seed=42`, without replacement (`analysis/sample_data.py`).
- **Models added incrementally** (recovered prompts): UniLID + DeepSeek + Qwen first
  (prompt 10, 2026-03-26), then UniLID-Marg (prompt 18, 2026-03-27), then fastText
  (prompt 19, 2026-03-27).
- **Macro-F1 bug found and fixed** after a code-review agent (recovered prompt 20,
  2026-03-27): metrics had averaged F1 over `set(y_true) | set(y_pred)`; corrected to
  average over `set(y_true)` only (sklearn convention). See `SETUP.md` gotcha 5.
- **Artifacts:** `outputs/tables/table{1-7}_*.{md,tex}`, confusion-matrix PNGs/TeX for 7
  clusters; raw prediction outputs in `full_prob/` and `glotlid_e100_sanity/`.

---

## Infrastructure events (recovered from prompt history)

- **2026-04-02:** moved the UniLID model and data from home to scratch to free disk space
  (recovered prompts 38–39); `config.DATA_DIR` repointed to scratch.
- **2026-03-29/30:** bumped SLURM memory request to 400 GB after repeated 128 GB OOMs
  (recovered prompt 33).
- **2026-04-03:** permission-denied on a shared dataset path
  (`/capstor/store/.../stackv2-edu`) (recovered prompt 46); not central to UniLID analysis.

## Checkpoint / artifact-deletion assessment (updated 2026-08-06; deletions user-approved)

The 2026-05-27 version of this section stated that no model checkpoints are produced;
that stopped being true with the Apertus retrains (2026-07). Current assessment, with
the deletions the user approved on 2026-08-06 executed:

Deleted from scratch (about 62 GB freed):
- `glotlid_apertus131k.unilid` and `glotlid_apertus200k.unilid` (the two models trained
  through the float32 EM bug; superseded by the `_fp64` retrains for every recorded
  purpose; the bug itself remains reproducible from the documented azj_Latn recipe, the
  retained pre-fix binary `~/.local/bin/spm_train.pre_fp64`, and the trigger-line
  artifact `outputs/diagnostic/em_trigger_azj_81251_81640.txt`).
- `results_apertus131k/`, `results_apertus131k_fp64/`, `results_apertus200k_fp64/`
  (per-language training outputs behind those models; the retained packed `.unilid`
  files carry the results of record, and the training dirs are regenerable from the
  corpus plus the patched trainer and the recorded recipes). Consequence accepted with
  the approval: re-running the superseded corrupted-branch evaluation (Exp 29/30) is no
  longer possible; its prediction memmap and recorded entries remain.

Kept, with reasons:
- `glotlidc.unilid` (the production model behind every main-line result),
  `glotlid_apertus131k_fp64.unilid`, `glotlid_apertus200k_fp64.unilid` (the clean
  records of the closed vocabulary branch, Exp 42/43).
- `results_apertus200k/` in full: its `corpus/` subdirectory is load-bearing (the
  per-language calibration corpora read by the gate machinery and `gt_counts.py`), and
  its per-language tokenizer outputs back `gt_counts.csv` regeneration.
- Everything in `full_test_eval/` (y_true, all prediction memmaps including the
  superseded margin-family ones that the recorded two_sided_report wiring gates read,
  the seed-301 split record, the banked candidate arrays), plus
  `full_test_eval_131k/` and `full_test_eval_131k_fp64/` (the branch measurements of
  record, 88 MB each).
- All kept scratch artifacts re-touched 2026-08-06 against the 14-day purge.

Durable-storage migration (2026-08-06, user-directed): the artifacts of record
were moved from scratch to /capstor/store/cscs/swissai/a0229/cmeister/unilid_analysis
(5,975,047,573 bytes, 50 files, each byte-verified with cmp before the scratch
original was removed) and replaced on scratch by absolute symlinks, so every
path recorded in code and in these documents keeps working unchanged. Moved:
the three models of record (glotlidc.unilid and both _fp64 retrains),
sample_500k_all.pkl, and every file in full_test_eval/, full_test_eval_131k/,
and full_test_eval_131k_fp64/. Still on scratch under the re-touch policy:
results_apertus200k/ (the calibration corpus, regenerable from Drive) and
glotlid_unilid/ (the test file, regenerable from its zip). The user designated
the store path for anything that must not risk the 14-day purge.

