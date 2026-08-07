"""Camera-ready E4: breakdowns and residual re-measurement (EXPERIMENTS_PLAN.md,
"Camera-ready evaluation program (2026-08-06)", E4 bullet). Conventions:
EXPERIMENTAL_SETUP.md, "Camera-ready reporting conventions".

Two independent parts, selected by --part; each can run and write its outputs
without the other.

Part "breakdowns" (waiting on E1, analysis/paper_eval.py, for the global view;
reads the SCRATCH_DIR prediction memmaps directly for the within-stratum view):
  1. Script breakdown: group the 1,940 languages into the paper's script rows
     (Latn, Cyrl, Arab, Deva, Beng, Grek, Hebr, Armn, Other) and report, for
     each of {baseline, gate_flat4_prox21, fastText}, BOTH the global
     per-language F1 (all false positives counted) and the within-stratum F1
     (examples restricted to true labels in the group, cross-group false
     positives excluded), on the full kept pool.
  2. REPRODUCTION GATE: both paper appendix tables turned out to be the
     within-stratum view (EXPERIMENTS_CHRONOLOGICAL.md, "provenance of the
     paper's appendix breakdown tables resolved", 2026-08-07), so the gate
     compares the recomputed within-stratum baseline column against the
     paper's published script table (paper/submission.tex,
     tab:script-breakdown); the Other row uses the paper's own basis
     (PAPER_OTHER_EXCLUDES: our Other minus jpn_Jpan and kor_Hang). The old
     global-view comparison is kept as a RECORDED, non-gating diagnostic
     table (an expected cross-view mismatch, not a regression). Non-fatal: on
     any MISMATCH the affected .tex is not written and the script exits
     nonzero at the end, after writing every other output.
  3. Label-basis diagnostic: cross-checks the paper team's own fastText
     per-language file against our 1,940-language set, to see whether it
     explains the paper's stated 1,938-language basis and the Hebr-row
     mismatch. Runs AFTER every other report above is written, wrapped so a
     failure inside it is recorded (into its own report) rather than
     crashing before the main/gate reports are written.
  4. Resource-tier breakdown: the same both-views/gate treatment against
     paper/submission.tex, tab:resource-tier (no PAPER_OTHER_EXCLUDES-style
     basis adjustment needed there: the paper's six resource bins already sum
     to the full 1,940).
  5. Outputs: outputs/tables/paper_breakdowns.md (always, both views), the
     reproduction gate detail in outputs/tables/paper_breakdowns_gate.md
     (always, within-stratum gates plus the RECORDED global-view diagnostic
     table), the label-basis diagnostic in
     outputs/tables/paper_breakdowns_basis.md (always, including on
     diagnostic failure), and outputs/tables/paper_breakdowns_script.tex /
     _resource.tex (both views) only for whichever table's gate passed.

Part "residual" (no dependency on E1; reads saved prediction memmaps
directly):
  Re-measures the promoted configuration's (gate_flat4_prox21) judge-part
  residual: how many predictions are wrong, what share of the wrong lines
  have a well-resourced (N >= HEAD_N) true language, what share of those are
  confused with another well-resourced language, and the top 20 confused
  (true, predicted) pairs by line count. floor21_gate is recomputed alongside
  for comparison against the EXPERIMENTS_RESULTS.md "Current state
  (2026-08-06)" open item 3 figures (962,633 / 98.7% / 88.2%), which this
  script re-derives rather than trusts. The judge part itself is re-derived
  from RULE_SPLIT_SEED/RULE_SPLIT_FRACTION (analysis.combined_evidence's own
  pattern) and required to bit-match the split recorded at SPLIT_PATH, rather
  than trusting that npz alone.
  Outputs: outputs/tables/promoted_residual.md and
  outputs/diagnostic/promoted_residual_pairs.csv (gate_flat4_prox21's top 20
  pairs).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import Counter

import numpy as np
import pandas as pd

from analysis.balanced_split import OUT_DIR as DRAW_DIR, SEEDS, TEST_SEED
from analysis.carried_set_comparison import EXPECTED_REMAINDER
from analysis.combined_evidence import (EXPECTED_DERIVATION, EXPECTED_JUDGE,
                                        RULE_SPLIT_FRACTION, RULE_SPLIT_SEED,
                                        SPLIT_PATH)
from analysis.config import RESOURCE_BINS, RESOURCE_LABELS, TOTAL_LINES
from analysis.format_utils import to_latex, to_markdown
from analysis.full_test_eval import EMPTY, SCRATCH_DIR
from analysis.full_test_margin import HEAD_N
from analysis.margin_diagnostic import PRF_CSV
from analysis.metric_decomposition import EXPECTED_KEPT, _per_lang_stats
from analysis.paper_eval import CONFIGS, DISPLAY, OUT_CSV_FULLPOOL as PAPER_EVAL_FULLPOOL_CSV
from analysis.transfer_sweep import _load_model_data

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

FULLPOOL_VIEW = ("global per-language F1, all false positives counted, "
                 "averaged over languages in the group")

# Part 2 (EXPERIMENTS_CHRONOLOGICAL.md, "provenance of the paper's appendix
# breakdown tables resolved", 2026-08-07): the second view the published tables now
# carry alongside the global view above. within-stratum restricts to examples whose
# TRUE label is in the group, so cross-group false positives (a prediction landing
# on a group language from a true label outside the group) are excluded; this is the
# view analysis/metric_decomposition.py's tail/head/magnets/twins stratum rows
# already use, and the view both of the paper's appendix tables turned out to be.
STRATUM_VIEW = ("within-stratum F1: examples restricted to true labels inside "
               "the group, cross-group false positives excluded, averaged over "
               "languages in the group")

# Caption text for tables that carry both views side by side (Part 2b, the exact
# wording the task spec pre-registered).
BOTH_VIEWS_CAPTION = (
    "global = per-language F1 with all false positives counted; "
    "within-stratum = examples restricted to true labels in the group, "
    f"cross-group false positives excluded; full kept pool, {EXPECTED_KEPT:,} "
    "lines")


def _fullpool_caption(prefix: str) -> str:
    """Every table caption names its instrument: line set + view."""
    return f"{prefix}, full kept pool, {EXPECTED_KEPT:,} lines, {FULLPOOL_VIEW}"


# ---------------------------------------------------------------------------
# Part "breakdowns": script grouping
# ---------------------------------------------------------------------------

# The paper's script table rows (paper/submission.tex, tab:script-breakdown,
# \begin{table} at line 1483, data rows 1490-1498). "Other" is everything not
# in the eight named scripts. analysis.config.TOP_SCRIPTS is deliberately not
# imported here: the paper's table uses a different 8-script partition
# (TOP_SCRIPTS adds Hang and Jpan, which fall into Other below).
SCRIPT_ROWS = ["Latn", "Cyrl", "Arab", "Deva", "Beng", "Grek", "Hebr", "Armn"]
OTHER_LABEL = "Other"
SCRIPT_ORDER = SCRIPT_ROWS + [OTHER_LABEL]

# Published paper/submission.tex tab:script-breakdown, read 2026-08-07:
# script -> (# Langs, UniLID F1, fastText F1). The paper's stated basis is
# 1,938 languages against our 1,940 (EXPERIMENTS_PLAN.md, "Camera-ready
# evaluation program", E4 bullet); this is a pre-registered EXPECTED
# mismatch source, not assumed to be the only one.
PAPER_SCRIPT_TABLE = {
    "Latn":  (1700, 0.940, 0.946),
    "Cyrl":  (70,   0.877, 0.970),
    "Arab":  (38,   0.691, 0.747),
    "Deva":  (32,   0.811, 0.932),
    "Beng":  (6,    0.885, 0.985),
    "Grek":  (4,    0.677, 0.925),
    "Hebr":  (4,    0.740, 0.967),
    "Armn":  (2,    0.974, 0.986),
    "Other": (82,   0.937, 0.973),
}
# Reproduction gate tolerance (magic number, pre-registered in the E4 task
# spec: "at tolerance SCRIPT_GATE_TOL = 0.005").
SCRIPT_GATE_TOL = 0.005

# Part 2a (EXPERIMENTS_CHRONOLOGICAL.md, "provenance of the paper's appendix
# breakdown tables resolved", 2026-08-07): the paper's script-table "Other" row
# (82 languages) is our own 84-language Other group minus these two. Used only by
# the script-table reproduction gate below; the published breakdown table (Part 2b)
# keeps the full 1,940-language / 84-language Other basis.
PAPER_OTHER_EXCLUDES = ("jpn_Jpan", "kor_Hang")


def _script_of(label: str) -> str:
    """Script code: the part of the label after the underscore, e.g.
    eng_Latn -> Latn."""
    parts = label.rsplit("_", 1)
    if len(parts) != 2:
        raise RuntimeError(f"label {label!r} has no script suffix")
    return parts[1]


def _script_group(label: str) -> str:
    s = _script_of(label)
    return s if s in SCRIPT_ROWS else OTHER_LABEL


# ---------------------------------------------------------------------------
# Part "breakdowns": resource-tier grouping
# ---------------------------------------------------------------------------

# RESOURCE_BINS/RESOURCE_LABELS (analysis.config) already match
# paper/submission.tex tab:resource-tier (\begin{table*} at line 1511, data
# rows 1520-1525) exactly: [0, 500, 1_000, 12_000, 18_000, 35_000, inf) with
# left-inclusive/right-exclusive bins, labels "<500", "500--1k", "1k--12k",
# "12k--18k", "18k--35k", "35k+". Imported rather than redefined (repo
# convention: analysis/tables.py, analysis/comparison.py, etc.).

# Published paper/submission.tex tab:resource-tier, read 2026-08-07:
# bin label -> (# Langs, UniLID F1). The paper's FPR columns are not
# reproduced here (not part of this task's spec).
PAPER_RESOURCE_TABLE = {
    "<500":     (56,  0.871),
    "500--1k":  (40,  0.975),
    "1k--12k":  (458, 0.990),
    "12k--18k": (526, 0.997),
    "18k--35k": (398, 0.992),
    "35k+":     (462, 0.958),
}
RESOURCE_GATE_TOL = 0.005


def _resource_bin(n: int) -> str:
    for i in range(len(RESOURCE_LABELS)):
        lo, hi = RESOURCE_BINS[i], RESOURCE_BINS[i + 1]
        if lo <= n < hi:
            return RESOURCE_LABELS[i]
    raise RuntimeError(f"N={n} did not match any RESOURCE_BINS bucket")


# ---------------------------------------------------------------------------
# Part "breakdowns": generic group-means + reproduction-gate helpers
# ---------------------------------------------------------------------------

def _grouped_means(df: pd.DataFrame, group_col: str, order: list[str]) -> dict:
    """group -> {"n": int, config: mean F1 for config in CONFIGS}. Raises, naming
    the empty group and the grouping column, if any group in `order` has zero rows
    in df[group_col]: an empty group is a wiring error (e.g. a stale group name
    that no longer matches any language), not a case to paper over with n=0/NaN."""
    g = df.groupby(group_col)
    means = g[[f"f1_{c}" for c in CONFIGS]].mean()
    counts = g.size()
    out = {}
    for k in order:
        if k not in counts.index:
            raise RuntimeError(
                f"group {k!r} of column {group_col!r} has zero rows in the "
                "input DataFrame; refusing to report it as n=0/NaN")
        out[k] = {"n": int(counts[k])}
        for c in CONFIGS:
            out[k][c] = float(means.loc[k, f"f1_{c}"])
    return out


def _within_stratum_f1(pred: np.ndarray, y: np.ndarray, group_lang_idx: np.ndarray,
                       n_lang: int) -> float:
    """Within-stratum F1 for one group (Part 2, STRATUM_VIEW above): restricts to
    lines whose TRUE label is in `group_lang_idx`, runs
    analysis.metric_decomposition._per_lang_stats on that restricted subset (so a
    prediction landing on a group language from a true label OUTSIDE the group
    contributes no false positive, by construction of the restricted subset), then
    averages F1 over the group's own languages. Mirrors the within-stratum
    convention analysis/metric_decomposition.py already uses for its tail/head/
    magnets/twins stratum rows."""
    m = np.isin(y, group_lang_idx)
    _prec, _rec, f1, _tp, _fp, _fn = _per_lang_stats(pred[m], y[m], n_lang)
    return float(f1[group_lang_idx].mean())


def _gate_rows(computed: dict, paper: dict, order: list[str], tol: float):
    """computed, paper: group -> (n_langs, f1). Returns (rows, mismatched_groups)
    where rows is the full comparison table (every group, OK or MISMATCH)."""
    rows, mismatches = [], []
    for g in order:
        our_n, our_f1 = computed[g]
        paper_n, paper_f1 = paper[g]
        diff = our_f1 - paper_f1
        ok = (our_n == paper_n) and (abs(diff) <= tol)
        status = "OK" if ok else "MISMATCH"
        if not ok:
            mismatches.append(g)
        rows.append([g, our_n, paper_n, f"{our_f1:.4f}", f"{paper_f1:.4f}",
                     f"{diff:+.4f}", status])
    return rows, mismatches


def _blocked_message(table_name: str, mismatches: list[str]) -> str:
    return (f"BLOCKED: {table_name} reproduction gate failed for rows "
            f"{mismatches}; discrepancy goes to the user (Ahmetcan ask item "
            "3); do not publish.")


GATE_HEADERS = ["group", "our # langs", "paper # langs", "our F1", "paper F1",
                "diff", "status"]

OUT_MD = "outputs/tables/paper_breakdowns.md"
OUT_MD_GATE = "outputs/tables/paper_breakdowns_gate.md"
OUT_MD_BASIS = "outputs/tables/paper_breakdowns_basis.md"
OUT_TEX_SCRIPT = "outputs/tables/paper_breakdowns_script.tex"
OUT_TEX_RESOURCE = "outputs/tables/paper_breakdowns_resource.tex"

# ---------------------------------------------------------------------------
# Part "breakdowns": label-basis diagnostic (item 3)
# ---------------------------------------------------------------------------

FASTTEXT_TEAM_JSON = ("outputs/diagnostic/paper_team/fasttext_folder/"
                      "glotlid_fasttext_e100_sanity_per_language.json")
FASTTEXT_TEAM_METRICS = ("outputs/diagnostic/paper_team/fasttext_folder/"
                         "glotlid_fasttext_e100_sanity_metrics.json")

# Hypothesis under test (EXPERIMENTS_PLAN.md E4 bullet, "steady-finding-abelson.md"
# plan): the paper's script table covers 1,938 languages against our 1,940. If the
# paper team's own fastText per-language file has exactly this many labels, the two
# missing ones would explain the basis mismatch.
PAPER_SCRIPT_BASIS_LANGS = 1938

# Sibling metrics.json macro_f1, recorded in the task spec as a self-check
# reference for the JSON's own unweighted mean F1 over its own labels.
SIBLING_MACRO_F1_RECORDED = 0.9443269
SIBLING_MACRO_F1_TOL = 1e-6


def _script_means_from_labelf1(label_f1: dict[str, float]) -> dict[str, tuple[int, float]]:
    """label -> f1 dict, grouped into SCRIPT_ORDER. Returns group -> (n, mean f1)."""
    buckets: dict[str, list[float]] = {g: [] for g in SCRIPT_ORDER}
    for label, f1 in label_f1.items():
        buckets[_script_group(label)].append(f1)
    return {g: (len(vals), float(np.mean(vals)) if vals else float("nan"))
            for g, vals in buckets.items()}


def _label_basis_diagnostic(langs: list[str], df: pd.DataFrame) -> list[str]:
    """Item 3. Writes OUT_MD_BASIS. Returns the report lines (for logging only;
    the file is the record)."""
    for p in (FASTTEXT_TEAM_JSON, FASTTEXT_TEAM_METRICS):
        if not os.path.exists(p):
            raise FileNotFoundError(f"required artifact missing: {p}")

    with open(FASTTEXT_TEAM_JSON) as f:
        team = json.load(f)
    if not isinstance(team, dict):
        raise RuntimeError(f"{FASTTEXT_TEAM_JSON}: expected a {{label: stats}} "
                           f"dict, got {type(team).__name__}")
    with open(FASTTEXT_TEAM_METRICS) as f:
        team_metrics = json.load(f)
    if "macro_f1" not in team_metrics:
        raise RuntimeError(f"{FASTTEXT_TEAM_METRICS} has no 'macro_f1' key")

    team_labels = set(team.keys())
    our_labels = set(langs)
    n_team = len(team_labels)
    missing_from_team = sorted(our_labels - team_labels)
    extra_in_team = sorted(team_labels - our_labels)

    L = ["# Camera-ready E4: label-basis diagnostic\n",
         "Pre-registration: EXPERIMENTS_PLAN.md, \"Camera-ready evaluation "
         f"program (2026-08-06)\", E4 bullet. Inspects {FASTTEXT_TEAM_JSON} "
         "(the paper team's own per-language fastText results) to see "
         "whether it explains the paper's stated 1,938-language script-table "
         "basis against our 1,940 languages and the Hebr-row mismatch.\n",
         "## Label inventory\n"]
    L.append(f"- {FASTTEXT_TEAM_JSON}: {n_team:,} labels.")
    L.append(f"- Our canonical language list: {len(our_labels):,} labels.")
    L.append(f"- Absent from the JSON, present in ours: {len(missing_from_team)}"
             + (f" ({', '.join(missing_from_team)})" if missing_from_team else "") + ".")
    L.append(f"- Present in the JSON, absent from ours: {len(extra_in_team)}"
             + (f" ({', '.join(extra_in_team)})" if extra_in_team else "") + ".")

    # Self-check: the JSON's own unweighted mean f1 over its own labels
    # against the sibling metrics.json macro_f1, and against the recorded
    # 0.9443269 reference.
    missing_f1 = [l for l, v in team.items() if "f1" not in v]
    if missing_f1:
        shown = missing_f1[:10]
        raise RuntimeError(
            f"{FASTTEXT_TEAM_JSON}: {len(missing_f1)} entries lack an 'f1' "
            f"key (first 10: {shown})")
    team_f1_by_label = {l: v["f1"] for l, v in team.items()}
    team_macro = float(np.mean(list(team_f1_by_label.values())))
    diff_sibling = abs(team_macro - team_metrics["macro_f1"])
    diff_recorded = abs(team_macro - SIBLING_MACRO_F1_RECORDED)
    L.append("\n## Self-check\n")
    L.append(f"Unweighted mean of the JSON's own {n_team:,} f1 values: "
             f"{team_macro:.7f}.")
    L.append(f"- vs sibling {FASTTEXT_TEAM_METRICS} macro_f1 "
             f"{team_metrics['macro_f1']:.7f} (diff {diff_sibling:.2e}).")
    L.append(f"- vs recorded reference {SIBLING_MACRO_F1_RECORDED} (diff "
             f"{diff_recorded:.2e}, tol {SIBLING_MACRO_F1_TOL:.0e}): "
             + ("OK" if diff_recorded <= SIBLING_MACRO_F1_TOL else "WARNING: exceeds tolerance") + ".")
    if diff_recorded > SIBLING_MACRO_F1_TOL:
        print("WARNING: label-basis diagnostic self-check exceeds tolerance: "
              f"{team_macro:.7f} vs {SIBLING_MACRO_F1_RECORDED} "
              f"(diff {diff_recorded:.2e}).")

    L.append("\n## Hypothesis test: does the JSON explain the 1,938-vs-1,940 basis?\n")
    if n_team == PAPER_SCRIPT_BASIS_LANGS:
        if len(missing_from_team) != 2:
            raise RuntimeError(
                f"{FASTTEXT_TEAM_JSON} has {n_team} labels, matching the "
                f"hypothesized {PAPER_SCRIPT_BASIS_LANGS}, but "
                f"{len(missing_from_team)} of our labels are absent from it, "
                "not the 2 the pre-registration hypothesized; the exclusion "
                "recompute below assumes exactly 2")
        excluded = missing_from_team
        L.append(f"CONFIRMED: the JSON has exactly {PAPER_SCRIPT_BASIS_LANGS:,} "
                 f"labels, missing {excluded} from our set of "
                 f"{len(our_labels):,}. Recomputing baseline and fastText "
                 "script-group means excluding these labels:\n")
        df_excl = df[~df["lang"].isin(excluded)]
        excl_baseline = _grouped_means(df_excl, "script_group", SCRIPT_ORDER)
        hebr_f1 = excl_baseline["Hebr"]["baseline"]
        hebr_paper = PAPER_SCRIPT_TABLE["Hebr"][1]
        hebr_diff = hebr_f1 - hebr_paper
        L.append(f"- Hebr row (baseline, excluding {excluded}): {hebr_f1:.4f} "
                 f"vs paper {hebr_paper} (diff {hebr_diff:+.4f}, tol "
                 f"{SCRIPT_GATE_TOL}): "
                 + ("REPRODUCES" if abs(hebr_diff) <= SCRIPT_GATE_TOL else "still MISMATCH") + ".")

        excl_fasttext_paper = {g: (excl_baseline[g]["n"], excl_baseline[g]["fasttext"])
                               for g in SCRIPT_ORDER}
        paper_fasttext = {g: (PAPER_SCRIPT_TABLE[g][0], PAPER_SCRIPT_TABLE[g][2])
                          for g in SCRIPT_ORDER}
        rows_ft_paper, _mism_ft_paper = _gate_rows(
            excl_fasttext_paper, paper_fasttext, SCRIPT_ORDER, SCRIPT_GATE_TOL)
        L.append("\nfastText script-group means (excluding the 2 labels) vs "
                 "the paper's fastText column:\n")
        L.append(to_markdown(rows_ft_paper, GATE_HEADERS,
                             caption="Excluded-label recompute vs paper fastText column"))

        team_script_means = _script_means_from_labelf1(team_f1_by_label)
        rows_ft_json, _mism_ft_json = _gate_rows(
            excl_fasttext_paper, team_script_means, SCRIPT_ORDER, SCRIPT_GATE_TOL)
        L.append("fastText script-group means (excluding the 2 labels) vs "
                 "the JSON's own script-group means:\n")
        L.append(to_markdown(rows_ft_json, GATE_HEADERS,
                             caption="Excluded-label recompute vs JSON-derived means"))
    else:
        L.append(f"NOT CONFIRMED: the JSON has {n_team:,} labels, not the "
                 f"hypothesized {PAPER_SCRIPT_BASIS_LANGS:,}. "
                 f"{len(missing_from_team)} of our {len(our_labels):,} labels "
                 f"are missing from it and {len(extra_in_team)} labels in it "
                 "are not in ours. This diagnostic does not resolve the "
                 "1,938-vs-1,940 basis or the Hebr-row discrepancy; both "
                 "remain open (ask list item 3, "
                 "~/.claude/plans/steady-finding-abelson.md, \"What to ask "
                 "Ahmetcan for\").")

    # Independent of the hypothesis outcome above: the JSON's own script-group
    # means (all n_team labels) against the paper's printed fastText column.
    team_script_means = _script_means_from_labelf1(team_f1_by_label)
    paper_fasttext_full = {g: (PAPER_SCRIPT_TABLE[g][0], PAPER_SCRIPT_TABLE[g][2])
                           for g in SCRIPT_ORDER}
    rows_json_vs_paper, _mism_json_vs_paper = _gate_rows(
        team_script_means, paper_fasttext_full, SCRIPT_ORDER, SCRIPT_GATE_TOL)
    L.append("\n## JSON's own script-group means vs the paper's fastText column "
             f"(all {n_team:,} JSON labels, no exclusion)\n")
    L.append(to_markdown(rows_json_vs_paper, GATE_HEADERS,
                         caption="JSON per-language f1, grouped by script, vs "
                                 "paper/submission.tex fastText column"))

    sup = [v.get("support") for v in team.values()]
    n_missing_support = sum(1 for s in sup if s is None)
    if n_missing_support == 0:
        # total_samples formatted OUTSIDE the f-string's format spec: team_metrics
        # is untrusted external JSON, and a comma ("," thousands) format spec on a
        # non-numeric fallback (the old code used .get(..., 'n/a')) raises
        # ValueError at format time, not at the .get() call, which crashed this
        # diagnostic whenever the key was absent.
        if "total_samples" in team_metrics:
            total_samples_str = f"{team_metrics['total_samples']:,}"
        else:
            total_samples_str = "n/a"
        L.append(f"\nSum of the JSON's own support values: {sum(sup):,} "
                 f"(total_samples in {FASTTEXT_TEAM_METRICS}: "
                 f"{total_samples_str}). Note this "
                 f"may differ from our full kept pool ({EXPECTED_KEPT:,} "
                 "lines) if the JSON's run scored a different line set.")
    else:
        L.append(f"\n{n_missing_support:,} of {n_team:,} entries in "
                 f"{FASTTEXT_TEAM_JSON} lack a \"support\" key; the "
                 "sum-of-support cross-check against total_samples was "
                 "skipped.")

    os.makedirs(os.path.dirname(OUT_MD_BASIS), exist_ok=True)
    with open(OUT_MD_BASIS, "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\nWrote {OUT_MD_BASIS}")
    return L


# ---------------------------------------------------------------------------
# Part "breakdowns": run
# ---------------------------------------------------------------------------

def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, OSError) as e:
        raise RuntimeError(f"git rev-parse HEAD failed: {e}") from e


def _write_basis_diagnostic_failure(exc: Exception) -> None:
    """Finding 11: run_breakdowns calls _label_basis_diagnostic AFTER the main and
    gate reports are already written, wrapped in try/except, so a failure inside
    the diagnostic never leaves those two reports unwritten. This records the
    failure into OUT_MD_BASIS itself (the diagnostic's own report path) rather
    than only printing a stack trace and stopping."""
    L = ["# Camera-ready E4: label-basis diagnostic\n",
        "FAILED. The label-basis diagnostic raised before completing. The main "
        f"({OUT_MD}) and gate ({OUT_MD_GATE}) reports were written successfully "
        "before this diagnostic ran and are unaffected by this failure.\n",
        f"- Exception: {type(exc).__name__}: {exc}",
    ]
    os.makedirs(os.path.dirname(OUT_MD_BASIS), exist_ok=True)
    with open(OUT_MD_BASIS, "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\nWrote {OUT_MD_BASIS} (FAILURE record).")


def run_breakdowns() -> int:
    """Returns the process exit code: 0 if both reproduction gates AND the
    label-basis diagnostic passed, 1 if any failed. Writes every output
    regardless (non-fatal MISMATCH protocol).

    Part 2 (EXPERIMENTS_CHRONOLOGICAL.md, "provenance of the paper's appendix
    breakdown tables resolved", 2026-08-07): both paper appendix tables turned out
    to be the WITHIN-STRATUM view (STRATUM_VIEW above), not the global per-language
    view (FULLPOOL_VIEW) this script used to gate against. The reproduction gates
    below now compare within-stratum F1 (computed from the SCRATCH_DIR prediction
    memmaps, not the global per-language CSV); the published tables carry both
    views for all three configs; the old global-vs-paper comparison is kept as a
    RECORDED, non-gating diagnostic table (an expected mismatch, not a regression)."""
    if not os.path.exists(PAPER_EVAL_FULLPOOL_CSV):
        raise FileNotFoundError(
            f"required artifact missing: {PAPER_EVAL_FULLPOOL_CSV} "
            "(produced by analysis/paper_eval.py; run it first)")
    df = pd.read_csv(PAPER_EVAL_FULLPOOL_CSV)
    required_cols = {"lang", "N", "f1_baseline", "fp_baseline",
                     "f1_gate_flat4_prox21", "fp_gate_flat4_prox21",
                     "f1_fasttext", "fp_fasttext", "support"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise RuntimeError(f"{PAPER_EVAL_FULLPOOL_CSV} is missing expected "
                           f"columns: {sorted(missing_cols)}")

    weights, langs, _m = _load_model_data()
    del weights
    n_lang = len(langs)
    if df.lang.tolist() != langs:
        raise RuntimeError(
            f"language order gate failed: {PAPER_EVAL_FULLPOOL_CSV} lang "
            "column does not match _load_model_data's canonical language list")
    lang_to_pos = {l: i for i, l in enumerate(langs)}

    df = df.copy()
    df["script_group"] = df["lang"].map(_script_group)
    df["resource_bin"] = df["N"].map(_resource_bin)

    # --- Part 2d: memmaps for the within-stratum view. y_true.npy plus the three
    # prediction files, same sentinel guard as run_residual's _residual_stats
    # (abort on any value below -1, UNSEEN/EXCLUDED, on the kept pool) ---
    y_path = os.path.join(SCRATCH_DIR, "y_true.npy")
    if not os.path.exists(y_path):
        raise FileNotFoundError(f"required artifact missing: {y_path}")
    y_full = np.asarray(np.lib.format.open_memmap(y_path, mode="r"))
    if y_full.shape != (TOTAL_LINES,):
        raise RuntimeError(f"y_true memmap has shape {y_full.shape}, expected "
                           f"({TOTAL_LINES},)")
    kept = y_full >= 0
    n_kept = int(kept.sum())
    if n_kept != EXPECTED_KEPT:
        raise RuntimeError(f"full kept pool {n_kept:,} != EXPECTED_KEPT "
                           f"({EXPECTED_KEPT:,})")
    yk = y_full[kept]

    preds_kept = {}
    for cfg in CONFIGS:
        p_path = os.path.join(SCRATCH_DIR, f"pred_{cfg}.npy")
        if not os.path.exists(p_path):
            raise FileNotFoundError(f"required artifact missing: {p_path}")
        pred_full = np.asarray(np.lib.format.open_memmap(p_path, mode="r"))
        if pred_full.shape != (TOTAL_LINES,):
            raise RuntimeError(f"{p_path} has shape {pred_full.shape}, expected "
                               f"({TOTAL_LINES},)")
        pred_kept = pred_full[kept]
        bad = pred_kept < -1
        if bad.any():
            raise RuntimeError(
                f"{p_path}: {int(bad.sum()):,} predictions on the kept pool "
                "have a sentinel value below -1 (UNSEEN or EXCLUDED); the "
                "scoring pass that produced this memmap is incomplete")
        preds_kept[cfg] = pred_kept.astype(np.int64)

    def _within_stratum(cfg: str, group_lang_idx: np.ndarray) -> float:
        return _within_stratum_f1(preds_kept[cfg], yk, group_lang_idx, n_lang)

    # --- group -> language-index arrays: membership does not depend on the view,
    # shared by the global and within-stratum computations below ---
    script_group_idx = {g: np.where(df["script_group"].values == g)[0]
                        for g in SCRIPT_ORDER}
    resource_bin_idx = {b: np.where(df["resource_bin"].values == b)[0]
                        for b in RESOURCE_LABELS}
    for lang in PAPER_OTHER_EXCLUDES:
        if lang not in lang_to_pos:
            raise RuntimeError(f"{lang} (PAPER_OTHER_EXCLUDES) not found in the "
                               "canonical language list")
    other_excl_idx = np.array(
        [i for i in script_group_idx[OTHER_LABEL]
         if langs[i] not in PAPER_OTHER_EXCLUDES], dtype=np.int64)
    expected_other_excl_n = len(script_group_idx[OTHER_LABEL]) - len(PAPER_OTHER_EXCLUDES)
    if len(other_excl_idx) != expected_other_excl_n:
        raise RuntimeError(
            f"excluding {PAPER_OTHER_EXCLUDES} from the Other group left "
            f"{len(other_excl_idx)} languages, expected {expected_other_excl_n} "
            "(one or both languages may not actually be in the Other group)")

    # LaTeX-safe (underscore-free) display forms of PAPER_OTHER_EXCLUDES: to_latex
    # does not escape underscores (analysis/paper_eval.py's DISPLAY dict follows
    # the same convention, "gate-flat4-prox21" not "gate_flat4_prox21"), so a raw
    # "jpn_Jpan" in a .tex caption would break compilation.
    paper_other_excludes_tex = tuple(l.replace("_", "-") for l in PAPER_OTHER_EXCLUDES)

    git_commit = _git_commit()

    md = ["# Camera-ready E4: script and resource-tier breakdowns\n",
          "Pre-registration: EXPERIMENTS_PLAN.md, \"Camera-ready evaluation "
          "program (2026-08-06)\", E4 bullet. Conventions: "
          "EXPERIMENTAL_SETUP.md, \"Camera-ready reporting conventions\". "
          f"Input: {PAPER_EVAL_FULLPOOL_CSV} (analysis/paper_eval.py) for the "
          f"global view; {SCRATCH_DIR} prediction memmaps for the "
          "within-stratum view.\n",
          f"Both views below: {BOTH_VIEWS_CAPTION}.\n"]

    gate_md = ["# Camera-ready E4: script and resource-tier reproduction "
              "gates\n",
              "Full comparison tables (every group, OK or MISMATCH), "
              f"tolerance {SCRIPT_GATE_TOL} on F1 and exact match on language "
              "count. Non-fatal: a MISMATCH blocks only the affected .tex "
              "output, not this report. The gates below compare the "
              "WITHIN-STRATUM view (EXPERIMENTS_CHRONOLOGICAL.md, \"provenance "
              "of the paper's appendix breakdown tables resolved\", "
              "2026-08-07): both paper appendix tables turned out to be "
              "within-stratum, not the global per-language view this script "
              "used to gate against previously.\n",
              f"Our columns below are the full kept pool ({EXPECTED_KEPT:,} "
              "lines), in the view stated per table (global or "
              "within-stratum). The paper team's own computation used a "
              f"different basis: their own metrics JSON "
              f"({FASTTEXT_TEAM_METRICS}) states total_samples "
              f"{TOTAL_LINES:,} (the full raw test-file line count, not "
              "restricted to our kept pool).\n"]

    # --- 1/2: script breakdown (both views, all three configs) + within-stratum
    # reproduction gate (Part 2a/2b) ---
    script_stats_global = _grouped_means(df, "script_group", SCRIPT_ORDER)
    script_stats_stratum = {
        g: {"n": len(script_group_idx[g]),
           **{c: _within_stratum(c, script_group_idx[g]) for c in CONFIGS}}
        for g in SCRIPT_ORDER}

    both_view_headers = (["Script", "# Langs"]
                         + [f"{DISPLAY[c]} (global)" for c in CONFIGS]
                         + [f"{DISPLAY[c]} (within-stratum)" for c in CONFIGS])
    script_rows_md = [
        [g, script_stats_global[g]["n"]]
        + [f"{script_stats_global[g][c]:.4f}" for c in CONFIGS]
        + [f"{script_stats_stratum[g][c]:.4f}" for c in CONFIGS]
        for g in SCRIPT_ORDER]
    md.append(to_markdown(
        script_rows_md, both_view_headers,
        caption=f"Script breakdown, both views ({BOTH_VIEWS_CAPTION}). Uses "
                "the full 1,940-language basis (all 84 Other languages); the "
                "paper's original basis excluded "
                f"{' and '.join(PAPER_OTHER_EXCLUDES)} from Other."))

    # Reproduction gate: within-stratum baseline vs the paper's script table.
    # Other's gate row uses the paper's own basis (PAPER_OTHER_EXCLUDES excluded);
    # every other row uses the full group.
    script_gate_computed = {
        g: (script_stats_stratum[g]["n"], script_stats_stratum[g]["baseline"])
        for g in SCRIPT_ROWS}
    script_gate_computed[OTHER_LABEL] = (
        len(other_excl_idx), _within_stratum("baseline", other_excl_idx))
    script_paper = {g: (PAPER_SCRIPT_TABLE[g][0], PAPER_SCRIPT_TABLE[g][1])
                    for g in SCRIPT_ORDER}
    script_gate_rows, script_mismatches = _gate_rows(
        script_gate_computed, script_paper, SCRIPT_ORDER, SCRIPT_GATE_TOL)
    script_gate_passed = not script_mismatches

    gate_md.append("## Script-table reproduction gate (within-stratum)\n")
    gate_md.append("Our recomputed within-stratum baseline column vs "
                   "paper/submission.tex, tab:script-breakdown (UniLID "
                   f"column). Other's row uses the paper's own basis: our "
                   f"Other group minus {' and '.join(PAPER_OTHER_EXCLUDES)}.\n")
    gate_md.append(to_markdown(script_gate_rows, GATE_HEADERS,
                               caption="Script-table reproduction gate "
                                       "(within-stratum)"))
    if script_gate_passed:
        md.append("Script-table reproduction gate (within-stratum): PASSED.\n")
    else:
        msg = _blocked_message("script-table", script_mismatches)
        md.append(f"Script-table reproduction gate (within-stratum): {msg} See "
                 f"{OUT_MD_GATE} for the full comparison.\n")
        print(msg)

    # --- RECORDED (Part 2c, not a gate): global-view comparison against the
    # paper's script table -- an expected mismatch, kept for the record ---
    script_global_vs_paper_computed = {
        g: (script_stats_global[g]["n"], script_stats_global[g]["baseline"])
        for g in SCRIPT_ORDER}
    script_global_vs_paper_rows, _sgm = _gate_rows(
        script_global_vs_paper_computed, script_paper, SCRIPT_ORDER, SCRIPT_GATE_TOL)
    gate_md.append("\n## RECORDED: global-view comparison against the paper's "
                   "script table (expected-mismatch cross-view comparison, "
                   "NOT a gate)\n")
    gate_md.append(to_markdown(
        script_global_vs_paper_rows, GATE_HEADERS,
        caption=f"Global baseline F1 ({FULLPOOL_VIEW}) vs paper/submission.tex, "
                "tab:script-breakdown; expected mismatch, the paper is "
                "within-stratum"))

    # --- 4: resource-tier breakdown (both views, all three configs) + within-
    # stratum reproduction gate (Part 2a/2b) ---
    resource_stats_global = _grouped_means(df, "resource_bin", RESOURCE_LABELS)
    resource_stats_stratum = {
        b: {"n": len(resource_bin_idx[b]),
           **{c: _within_stratum(c, resource_bin_idx[b]) for c in CONFIGS}}
        for b in RESOURCE_LABELS}

    resource_rows_md = [
        [b, resource_stats_global[b]["n"]]
        + [f"{resource_stats_global[b][c]:.4f}" for c in CONFIGS]
        + [f"{resource_stats_stratum[b][c]:.4f}" for c in CONFIGS]
        for b in RESOURCE_LABELS]
    md.append(to_markdown(
        resource_rows_md,
        (["Resource", "# Langs"] + [f"{DISPLAY[c]} (global)" for c in CONFIGS]
         + [f"{DISPLAY[c]} (within-stratum)" for c in CONFIGS]),
        caption=f"Resource-tier breakdown, both views ({BOTH_VIEWS_CAPTION})"))

    resource_gate_computed = {b: (resource_stats_stratum[b]["n"],
                                  resource_stats_stratum[b]["baseline"])
                             for b in RESOURCE_LABELS}
    resource_paper = dict(PAPER_RESOURCE_TABLE)
    resource_gate_rows, resource_mismatches = _gate_rows(
        resource_gate_computed, resource_paper, RESOURCE_LABELS, RESOURCE_GATE_TOL)
    resource_gate_passed = not resource_mismatches

    gate_md.append("\n## Resource-tier reproduction gate (within-stratum)\n")
    gate_md.append("Our recomputed within-stratum baseline column vs "
                   "paper/submission.tex, tab:resource-tier (UniLID F1 "
                   "column).\n")
    gate_md.append(to_markdown(resource_gate_rows, GATE_HEADERS,
                               caption="Resource-tier reproduction gate "
                                       "(within-stratum)"))
    if resource_gate_passed:
        md.append("Resource-tier reproduction gate (within-stratum): PASSED.\n")
    else:
        msg = _blocked_message("resource-tier", resource_mismatches)
        md.append(f"Resource-tier reproduction gate (within-stratum): {msg} "
                 f"See {OUT_MD_GATE} for the full comparison.\n")
        print(msg)

    # --- RECORDED (Part 2c, not a gate): global-view comparison against the
    # paper's resource-tier table -- an expected mismatch, kept for the record ---
    resource_global_vs_paper_computed = {
        b: (resource_stats_global[b]["n"], resource_stats_global[b]["baseline"])
        for b in RESOURCE_LABELS}
    resource_global_vs_paper_rows, _rgm = _gate_rows(
        resource_global_vs_paper_computed, resource_paper, RESOURCE_LABELS,
        RESOURCE_GATE_TOL)
    gate_md.append("\n## RECORDED: global-view comparison against the paper's "
                   "resource-tier table (expected-mismatch cross-view "
                   "comparison, NOT a gate)\n")
    gate_md.append(to_markdown(
        resource_global_vs_paper_rows, GATE_HEADERS,
        caption=f"Global baseline F1 ({FULLPOOL_VIEW}) vs paper/submission.tex, "
                "tab:resource-tier; expected mismatch, the paper is "
                "within-stratum"))

    md.append(f"\nGit commit: {git_commit}.")

    os.makedirs(os.path.dirname(OUT_MD), exist_ok=True)
    with open(OUT_MD, "w") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))

    os.makedirs(os.path.dirname(OUT_MD_GATE), exist_ok=True)
    with open(OUT_MD_GATE, "w") as f:
        f.write("\n".join(gate_md) + "\n")
    print("\n".join(gate_md))
    print(f"\nWrote {OUT_MD}, {OUT_MD_GATE}")

    # --- 5: publishable .tex, only for the tables whose gate passed. Both views,
    # all three configs (Part 2b). The camera-ready script table uses the full
    # 1,940-language basis (all 84 Other languages), with a note that the paper's
    # original basis excluded the two languages in PAPER_OTHER_EXCLUDES ---
    def _tex_rows(stats_global, stats_stratum, order):
        return [[g, stats_global[g]["n"]]
               + [stats_global[g][c] for c in CONFIGS]
               + [stats_stratum[g][c] for c in CONFIGS]
               for g in order]

    tex_headers = (["Group", "# Langs"]
                  + [f"{DISPLAY[c]} (global)" for c in CONFIGS]
                  + [f"{DISPLAY[c]} (within-stratum)" for c in CONFIGS])
    tex_col_formats = ["str", "int"] + ["metric"] * (2 * len(CONFIGS))

    if script_gate_passed:
        tex_script = to_latex(
            _tex_rows(script_stats_global, script_stats_stratum, SCRIPT_ORDER),
            ["Script"] + tex_headers[1:],
            caption=(f"Camera-ready script breakdown, both views "
                    f"({BOTH_VIEWS_CAPTION}). Uses the full 1,940-language "
                    "basis (all 84 Other languages); the paper's original "
                    f"basis excluded {' and '.join(paper_other_excludes_tex)} "
                    "from Other."),
            label="tab:paper_breakdowns_script", col_formats=tex_col_formats)
        with open(OUT_TEX_SCRIPT, "w") as f:
            f.write(tex_script + "\n")
        print(f"Wrote {OUT_TEX_SCRIPT}")
    else:
        print(f"Did NOT write {OUT_TEX_SCRIPT} (script-table reproduction "
              "gate failed). If a copy from an earlier passing run exists on "
              "disk, it no longer reflects the current gate status and must "
              "not be published.")

    if resource_gate_passed:
        tex_resource = to_latex(
            _tex_rows(resource_stats_global, resource_stats_stratum, RESOURCE_LABELS),
            ["Resource"] + tex_headers[1:],
            caption=f"Camera-ready resource-tier breakdown, both views "
                    f"({BOTH_VIEWS_CAPTION}).",
            label="tab:paper_breakdowns_resource", col_formats=tex_col_formats)
        with open(OUT_TEX_RESOURCE, "w") as f:
            f.write(tex_resource + "\n")
        print(f"Wrote {OUT_TEX_RESOURCE}")
    else:
        print(f"Did NOT write {OUT_TEX_RESOURCE} (resource-tier reproduction "
              "gate failed). If a copy from an earlier passing run exists on "
              "disk, it no longer reflects the current gate status and must "
              "not be published.")

    # --- 3 (finding 11): label-basis diagnostic, moved to AFTER every other
    # report above is written, and wrapped so a failure inside it is recorded
    # (into its own OUT_MD_BASIS report) plus a nonzero exit, never an
    # unwritten-reports crash ---
    basis_ok = True
    try:
        _label_basis_diagnostic(langs, df)
    except Exception as e:
        basis_ok = False
        _write_basis_diagnostic_failure(e)

    return 0 if (script_gate_passed and resource_gate_passed and basis_ok) else 1


# ---------------------------------------------------------------------------
# Part "residual"
# ---------------------------------------------------------------------------

# "Top 20 confused ordered pairs" (pre-registered in the E4 task spec).
TOP_PAIRS = 20

RESIDUAL_PRED_FILES = {
    "gate_flat4_prox21": os.path.join(SCRATCH_DIR, "pred_gate_flat4_prox21.npy"),
    "floor21_gate": os.path.join(SCRATCH_DIR, "pred_floor21_gate.npy"),
}
Y_TRUE_PATH = os.path.join(SCRATCH_DIR, "y_true.npy")

# EXPERIMENTS_RESULTS.md, "Current state (2026-08-06)", open item 3: floor21_gate's
# judge-part residual as of 2026-07-30, not re-measured against gate_flat4_prox21's
# smaller residual set at the time. This script recomputes rather than trusts these.
RECORDED_FLOOR21_GATE_N_WRONG = 962_633
RECORDED_FLOOR21_GATE_HEAD_TRUE_SHARE = 0.987
RECORDED_FLOOR21_GATE_HEAD_HEAD_SHARE = 0.882

OUT_MD_RESIDUAL = "outputs/tables/promoted_residual.md"
OUT_CSV_PAIRS = "outputs/diagnostic/promoted_residual_pairs.csv"

JUDGE_INSTRUMENT = f"judge part, {EXPECTED_JUDGE:,} lines"


def _load_draw(seed: int) -> np.ndarray:
    """Mirrors analysis.paper_eval._load_draw / analysis.combined_evidence._load_draw."""
    path = os.path.join(DRAW_DIR, f"val_lines_seed{seed}.npy")
    if not os.path.exists(path):
        raise FileNotFoundError(f"draw file missing: {path}")
    return np.load(path)


def _residual_stats(pred_judge: np.ndarray, y_judge: np.ndarray, N: np.ndarray,
                    langs: list[str], top_pairs: int = TOP_PAIRS) -> dict:
    """pred_judge, y_judge: int arrays over the judge part, positionally aligned.
    N: per-language training count, indexed by label id (same order as langs)."""
    bad = pred_judge < -1
    if bad.any():
        raise RuntimeError(
            f"{int(bad.sum()):,} predictions on the judge part have a sentinel "
            "value below -1 (UNSEEN or EXCLUDED); the scoring pass that "
            "produced this memmap is incomplete")

    n_empty = int((pred_judge == EMPTY).sum())
    wrong = pred_judge != y_judge
    n_wrong = int(wrong.sum())
    if n_wrong == 0:
        raise RuntimeError("zero wrong predictions on the judge part; refusing "
                           "to report shares with a zero denominator")

    wrong_true = y_judge[wrong]
    wrong_pred = pred_judge[wrong]

    head_true_mask = N[wrong_true] >= HEAD_N
    n_head_true = int(head_true_mask.sum())
    head_true_share = n_head_true / n_wrong

    valid_pred = wrong_pred >= 0
    pred_is_head = np.zeros(wrong_pred.shape, dtype=bool)
    pred_is_head[valid_pred] = N[wrong_pred[valid_pred]] >= HEAD_N
    head_head_mask = head_true_mask & pred_is_head
    n_head_head = int(head_head_mask.sum())
    head_head_share = (n_head_head / n_head_true) if n_head_true else float("nan")

    # Top confused pairs: EMPTY predictions have no specific predicted language, so
    # they are excluded here (matching analysis.metric_decomposition._per_lang_stats'
    # treatment of EMPTY as a false negative with no false positive), and reported
    # separately via n_empty above.
    pair_true = wrong_true[valid_pred]
    pair_pred = wrong_pred[valid_pred]
    counts = Counter(zip(pair_true.tolist(), pair_pred.tolist()))
    top = counts.most_common(top_pairs)
    pairs = [{"true_lang": langs[t], "pred_lang": langs[p], "n_lines": c,
             "N_true": int(N[t]), "N_pred": int(N[p])}
            for (t, p), c in top]

    return {"n_wrong": n_wrong, "n_empty": n_empty,
           "n_head_true": n_head_true, "head_true_share": head_true_share,
           "n_head_head": n_head_head, "head_head_share": head_head_share,
           "pairs": pairs}


def run_residual() -> None:
    weights, langs, _m = _load_model_data()
    del weights

    if not os.path.exists(PRF_CSV):
        raise FileNotFoundError(f"required artifact missing: {PRF_CSV}")
    prf = pd.read_csv(PRF_CSV)
    if prf.lang.tolist() != langs:
        raise RuntimeError(
            f"language order gate failed: {PRF_CSV} lang column does not "
            "match _load_model_data's canonical language list")
    N = prf.N.values

    if not os.path.exists(Y_TRUE_PATH):
        raise FileNotFoundError(f"required artifact missing: {Y_TRUE_PATH}")
    y = np.asarray(np.lib.format.open_memmap(Y_TRUE_PATH, mode="r"))
    if y.shape != (TOTAL_LINES,):
        raise RuntimeError(f"y_true memmap has shape {y.shape}, expected "
                           f"({TOTAL_LINES},)")
    kept = y >= 0

    # --- finding 15: re-derive the seed-301 split from RULE_SPLIT_SEED/
    # RULE_SPLIT_FRACTION (the same pattern analysis/paper_eval.py and
    # analysis/combined_evidence.py use) and require bit-equality with the split
    # stored at SPLIT_PATH, instead of trusting the npz alone ---
    val101 = _load_draw(SEEDS[0])
    test201 = _load_draw(TEST_SEED)
    if np.intersect1d(val101, test201).size:
        raise RuntimeError("test draw overlaps the working val draw")
    excl = np.zeros(TOTAL_LINES, bool)
    excl[val101] = True
    excl[test201] = True
    remainder_mask = kept & ~excl
    if int(remainder_mask.sum()) != EXPECTED_REMAINDER:
        raise RuntimeError(f"remainder {int(remainder_mask.sum()):,} != "
                           f"EXPECTED_REMAINDER {EXPECTED_REMAINDER:,}")

    remainder_idx = np.where(remainder_mask)[0]
    u = np.random.default_rng(RULE_SPLIT_SEED).random(remainder_idx.size)
    derive_idx = remainder_idx[u < RULE_SPLIT_FRACTION]
    judge_idx = remainder_idx[u >= RULE_SPLIT_FRACTION]
    if len(derive_idx) != EXPECTED_DERIVATION:
        raise RuntimeError(f"derivation part {len(derive_idx):,} != "
                           f"EXPECTED_DERIVATION {EXPECTED_DERIVATION:,}")
    if len(judge_idx) != EXPECTED_JUDGE:
        raise RuntimeError(f"judge part {len(judge_idx):,} != EXPECTED_JUDGE "
                           f"{EXPECTED_JUDGE:,}")

    if not os.path.exists(SPLIT_PATH):
        raise FileNotFoundError(f"required artifact missing: {SPLIT_PATH}")
    with np.load(SPLIT_PATH) as stored:
        if "derive_idx" not in stored or "judge_idx" not in stored:
            raise RuntimeError(f"{SPLIT_PATH} is missing 'derive_idx' or "
                               "'judge_idx'")
        if not (np.array_equal(stored["derive_idx"], derive_idx)
                and np.array_equal(stored["judge_idx"], judge_idx)):
            raise RuntimeError(
                f"the seed-{RULE_SPLIT_SEED} split recomputed here does not "
                f"bit-match the split stored at {SPLIT_PATH}; the stored npz "
                "may be stale or from a different draw/seed")

    y_judge = y[judge_idx]
    if (y_judge < 0).any():
        raise RuntimeError(
            "the recomputed judge part includes lines with a non-kept "
            "y_true value; the remainder/kept-pool derivation above is broken")

    print(f"Gates passed: language order matches {PRF_CSV}; seed-{RULE_SPLIT_SEED} "
          f"split re-derived from RULE_SPLIT_SEED/RULE_SPLIT_FRACTION and matches "
          f"{SPLIT_PATH} bit-for-bit ({len(derive_idx):,} derivation / "
          f"{len(judge_idx):,} judge, matches EXPECTED_JUDGE {EXPECTED_JUDGE:,}).")

    results = {}
    for cfg, path in RESIDUAL_PRED_FILES.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"required artifact missing: {path}")
        pred = np.asarray(np.lib.format.open_memmap(path, mode="r"))
        if pred.shape != (TOTAL_LINES,):
            raise RuntimeError(f"{path} has shape {pred.shape}, expected "
                               f"({TOTAL_LINES},)")
        pred_judge = pred[judge_idx]
        results[cfg] = _residual_stats(pred_judge, y_judge, N, langs)
        r = results[cfg]
        print(f"{cfg}: n_wrong={r['n_wrong']:,} (EMPTY {r['n_empty']:,}), "
              f"head-true share={r['head_true_share']:.4f} "
              f"({r['n_head_true']:,}/{r['n_wrong']:,}), head-head share="
              f"{r['head_head_share']:.4f} ({r['n_head_head']:,}/{r['n_head_true']:,})")

    promoted = results["gate_flat4_prox21"]
    f21g = results["floor21_gate"]

    L = ["# Camera-ready E4: promoted-configuration residual re-measurement\n",
        "Pre-registration: EXPERIMENTS_PLAN.md, \"Camera-ready evaluation "
        "program (2026-08-06)\", E4 bullet. Instrument: "
        f"{JUDGE_INSTRUMENT} (analysis.combined_evidence's seed-301 rule "
        f"split, {SPLIT_PATH}). HEAD_N = {HEAD_N:,} training lines "
        "(analysis.full_test_margin). A prediction is \"wrong\" when it "
        "differs from the gold label; EMPTY (pred = -1, no specific "
        "predicted language) counts as wrong and is reported separately.\n",
        "## n_wrong / head-true / head-head, per configuration\n"]
    summary_rows = [[cfg, f"{results[cfg]['n_wrong']:,}",
                     f"{results[cfg]['n_empty']:,}",
                     f"{results[cfg]['head_true_share']:.4f}",
                     f"{results[cfg]['head_head_share']:.4f}"]
                    for cfg in RESIDUAL_PRED_FILES]
    L.append(to_markdown(
        summary_rows,
        ["config", "n_wrong", "n_empty (of n_wrong)", "head-true share",
        "head-head share (of head-true)"],
        caption=f"Residual summary, {JUDGE_INSTRUMENT}"))

    diff_n = f21g["n_wrong"] - RECORDED_FLOOR21_GATE_N_WRONG
    diff_ht = f21g["head_true_share"] - RECORDED_FLOOR21_GATE_HEAD_TRUE_SHARE
    diff_hh = f21g["head_head_share"] - RECORDED_FLOOR21_GATE_HEAD_HEAD_SHARE
    L.append("\n## floor21_gate vs the EXPERIMENTS_RESULTS.md recorded figures\n")
    L.append("EXPERIMENTS_RESULTS.md, \"Current state (2026-08-06)\", open "
            "item 3, measured 2026-07-30 (for the record; not a pass/fail "
            "gate, printed for comparison as pre-registered).\n")
    L.append(to_markdown(
        [["n_wrong", f"{f21g['n_wrong']:,}", f"{RECORDED_FLOOR21_GATE_N_WRONG:,}",
          f"{diff_n:+,}"],
         ["head-true share", f"{f21g['head_true_share']:.4f}",
          f"{RECORDED_FLOOR21_GATE_HEAD_TRUE_SHARE}", f"{diff_ht:+.4f}"],
         ["head-head share", f"{f21g['head_head_share']:.4f}",
          f"{RECORDED_FLOOR21_GATE_HEAD_HEAD_SHARE}", f"{diff_hh:+.4f}"]],
        ["quantity", "recomputed", "recorded", "diff"],
        caption=f"floor21_gate recomputed vs recorded, {JUDGE_INSTRUMENT}"))

    L.append(f"\n## Top {TOP_PAIRS} confused (true, predicted) pairs, "
            "gate_flat4_prox21 (the promoted configuration)\n")
    pair_rows = [[p["true_lang"], p["pred_lang"], f"{p['n_lines']:,}",
                 f"{p['N_true']:,}", f"{p['N_pred']:,}"]
                for p in promoted["pairs"]]
    L.append(to_markdown(pair_rows, ["true_lang", "pred_lang", "n_lines",
                                     "N_true", "N_pred"],
                         caption=f"Top {TOP_PAIRS} confused pairs, "
                                 f"gate_flat4_prox21, {JUDGE_INSTRUMENT}"))

    L.append(f"\n## Top {TOP_PAIRS} confused (true, predicted) pairs, "
            "floor21_gate (for comparison)\n")
    pair_rows_f21g = [[p["true_lang"], p["pred_lang"], f"{p['n_lines']:,}",
                       f"{p['N_true']:,}", f"{p['N_pred']:,}"]
                      for p in f21g["pairs"]]
    L.append(to_markdown(pair_rows_f21g, ["true_lang", "pred_lang", "n_lines",
                                          "N_true", "N_pred"],
                         caption=f"Top {TOP_PAIRS} confused pairs, "
                                 f"floor21_gate, {JUDGE_INSTRUMENT}"))

    git_commit = _git_commit()
    L.append(f"\nGit commit: {git_commit}.")

    os.makedirs(os.path.dirname(OUT_MD_RESIDUAL), exist_ok=True)
    with open(OUT_MD_RESIDUAL, "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))

    pairs_df = pd.DataFrame(promoted["pairs"])[
        ["true_lang", "pred_lang", "n_lines", "N_true", "N_pred"]]
    os.makedirs(os.path.dirname(OUT_CSV_PAIRS), exist_ok=True)
    pairs_df.to_csv(OUT_CSV_PAIRS, index=False)

    print(f"\nWrote {OUT_MD_RESIDUAL}, {OUT_CSV_PAIRS}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Camera-ready E4: script/resource-tier breakdowns "
                    "(with reproduction gates against the paper's published "
                    "tables) and the promoted configuration's judge-part "
                    "residual re-measurement.")
    parser.add_argument("--part", choices=["breakdowns", "residual", "all"],
                        required=True)
    args = parser.parse_args()

    exit_code = 0
    if args.part in ("breakdowns", "all"):
        exit_code = run_breakdowns()
    if args.part in ("residual", "all"):
        run_residual()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
