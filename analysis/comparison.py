"""Tables 5-7: Cross-system comparative analysis."""

from __future__ import annotations

import numpy as np
from collections import Counter

from analysis.config import (
    MODEL_NAMES,
    PRIMARY_MODEL,
    TOP_SCRIPTS,
    RESOURCE_BINS,
    RESOURCE_LABELS,
)
from analysis.metrics import compute_metrics
from analysis.format_utils import to_markdown, to_latex


def _extract_arrays(data: dict):
    y_true = np.array(data["y_true"])
    preds = {name: np.array(data[f"pred_{name}"]) for name in MODEL_NAMES}
    scripts = np.array([lab.rsplit("_", 1)[-1] for lab in y_true])
    train_counts = data["train_counts"]
    true_train_cts = np.array([train_counts.get(lab, 0) for lab in y_true])
    return y_true, preds, scripts, true_train_cts


def _resource_bin_label(count: int) -> str:
    for i in range(len(RESOURCE_LABELS)):
        lo = RESOURCE_BINS[i]
        hi = RESOURCE_BINS[i + 1]
        if lo <= count < hi:
            return RESOURCE_LABELS[i]
    return RESOURCE_LABELS[-1]


# ---------------------------------------------------------------------------
# Table 5: Error overlap
# ---------------------------------------------------------------------------

def table_error_overlap(data: dict) -> tuple[str, str]:
    y_true, preds, _, _ = _extract_arrays(data)
    errs = {name: (y_true != preds[name]) for name in MODEL_NAMES}

    any_wrong = np.zeros(len(y_true), dtype=bool)
    for e in errs.values():
        any_wrong |= e
    all_wrong = np.ones(len(y_true), dtype=bool)
    for e in errs.values():
        all_wrong &= e

    n_models = len(MODEL_NAMES)

    # When all models wrong, do they predict the same (wrong) label?
    if all_wrong.sum() > 0:
        all_same_pred = np.ones(all_wrong.sum(), dtype=bool)
        pred_vals = [preds[name][all_wrong] for name in MODEL_NAMES]
        for i in range(1, len(pred_vals)):
            all_same_pred &= (pred_vals[0] == pred_vals[i])
        n_all_same = int(all_same_pred.sum())
    else:
        n_all_same = 0

    headers = ["Category", "Count", "% of any-wrong"]
    rows = []
    n_any = int(any_wrong.sum())
    for name in MODEL_NAMES:
        n_err = int(errs[name].sum())
        rows.append([f"{name} errors", f"{n_err:,}", f"{n_err / n_any * 100:.1f}%"])

    rows.append(["", "", ""])

    # Only-X-wrong
    for name in MODEL_NAMES:
        only_mask = errs[name].copy()
        for other in MODEL_NAMES:
            if other != name:
                only_mask &= ~errs[other]
        n_only = int(only_mask.sum())
        rows.append([f"Only {name} wrong", f"{n_only:,}", f"{n_only / n_any * 100:.1f}%"])

    rows.append(["", "", ""])
    rows.append([f"All {n_models} wrong", f"{int(all_wrong.sum()):,}", f"{all_wrong.sum() / n_any * 100:.1f}%"])
    rows.append([f"All {n_models} wrong, same pred", f"{n_all_same:,}", f"{n_all_same / n_any * 100:.1f}%"])
    rows.append(["Any wrong", f"{n_any:,}", "100.0%"])

    md = to_markdown(rows, headers, caption="Error Overlap Across Models")
    latex_rows = []
    for r in rows:
        if r[0] == "":
            continue
        count_val = r[1].replace(",", "")
        if count_val.isdigit():
            latex_rows.append([r[0], int(count_val), r[2]])
        else:
            latex_rows.append(r)
    latex = to_latex(latex_rows, headers, caption="Error overlap across models.", label="tab:error-overlap")
    return md, latex


# ---------------------------------------------------------------------------
# Table 6: Per-script winner
# ---------------------------------------------------------------------------

def table_per_script_winner(data: dict) -> tuple[str, str]:
    y_true, preds, scripts, _ = _extract_arrays(data)

    headers = [
        "Script", "# Langs", "N",
        "Best (Acc)", "Acc", "2nd Acc", "Delta",
        "Best (Ma-F1)", "Ma-F1", "2nd Ma-F1", "Delta",
    ]

    script_order = list(TOP_SCRIPTS) + ["Other"]
    rows_md = []
    rows_latex = []

    for s in script_order:
        if s == "Other":
            mask = ~np.isin(scripts, TOP_SCRIPTS)
        else:
            mask = scripts == s
        n = int(mask.sum())
        if n == 0:
            continue

        n_langs = len(set(y_true[mask]))
        model_metrics = {}
        for name in MODEL_NAMES:
            model_metrics[name] = compute_metrics(y_true[mask], preds[name][mask])

        # Best accuracy
        accs = [(name, model_metrics[name]["accuracy"]) for name in MODEL_NAMES]
        accs.sort(key=lambda x: -x[1])
        best_acc_name, best_acc = accs[0]
        second_acc = accs[1][1]
        delta_acc = best_acc - second_acc

        # Best macro F1
        mf1s = [(name, model_metrics[name]["macro_f1"]) for name in MODEL_NAMES]
        mf1s.sort(key=lambda x: -x[1])
        best_mf1_name, best_mf1 = mf1s[0]
        second_mf1 = mf1s[1][1]
        delta_mf1 = best_mf1 - second_mf1

        row_md = [
            s, f"{n_langs:,}", f"{n:,}",
            best_acc_name, f"{best_acc:.3f}", f"{second_acc:.3f}", f"+{delta_acc:.3f}",
            best_mf1_name, f"{best_mf1:.3f}", f"{second_mf1:.3f}", f"+{delta_mf1:.3f}",
        ]
        row_latex = [
            s, n_langs, n,
            best_acc_name, best_acc, second_acc, delta_acc,
            best_mf1_name, best_mf1, second_mf1, delta_mf1,
        ]
        rows_md.append(row_md)
        rows_latex.append(row_latex)

    col_fmts = ["str", "int", "int", "str", "metric", "metric", "metric", "str", "metric", "metric", "metric"]
    md = to_markdown(rows_md, headers, caption="Per-Script Best Model")
    latex = to_latex(rows_latex, headers, caption="Per-script best model comparison.", label="tab:per-script-winner", col_formats=col_fmts)
    return md, latex


# ---------------------------------------------------------------------------
# Table 7: Biggest per-language divergences
# ---------------------------------------------------------------------------

MIN_SAMPLES = 20
TOP_N = 15


def table_divergences(data: dict) -> tuple[str, str]:
    y_true, preds, scripts, true_train_cts = _extract_arrays(data)
    train_counts = data["train_counts"]

    lang_list = sorted(set(y_true))
    gaps = []

    for lang in lang_list:
        mask = y_true == lang
        n = int(mask.sum())
        if n < MIN_SAMPLES:
            continue

        err_rates = {}
        for name in MODEL_NAMES:
            err_rates[name] = float((preds[name][mask] != y_true[mask]).mean())

        primary_err = err_rates[PRIMARY_MODEL]
        best_other_err = min(err_rates[name] for name in MODEL_NAMES if name != PRIMARY_MODEL)
        gap = primary_err - best_other_err

        script = lang.rsplit("_", 1)[-1]
        res_bin = _resource_bin_label(train_counts.get(lang, 0))

        gaps.append({
            "lang": lang,
            "script": script,
            "n": n,
            "resource": res_bin,
            "gap": gap,
            **{f"err_{name}": err_rates[name] for name in MODEL_NAMES},
        })

    headers = ["Language", "Script", "N", "Resource"]
    for name in MODEL_NAMES:
        headers.append(f"Err ({name})")
    headers.append("Gap")

    def _make_rows(items):
        rows_md = []
        rows_latex = []
        for g in items:
            row_md = [
                g["lang"], g["script"], f"{g['n']:,}", g["resource"],
            ]
            row_latex = [g["lang"], g["script"], g["n"], g["resource"]]
            for name in MODEL_NAMES:
                row_md.append(f"{g[f'err_{name}']:.3f}")
                row_latex.append(g[f"err_{name}"])
            row_md.append(f"{g['gap']:+.3f}")
            row_latex.append(g["gap"])
            rows_md.append(row_md)
            rows_latex.append(row_latex)
        return rows_md, rows_latex

    col_fmts = ["str", "str", "int", "str"] + ["metric"] * len(MODEL_NAMES) + ["metric"]

    # UniLID worse
    gaps_worse = sorted(gaps, key=lambda x: -x["gap"])[:TOP_N]
    md_worse_rows, latex_worse_rows = _make_rows(gaps_worse)
    md_worse = to_markdown(md_worse_rows, headers, caption=f"Languages where {PRIMARY_MODEL} is most worse than alternatives")
    latex_worse = to_latex(latex_worse_rows, headers,
                           caption=f"Languages where {PRIMARY_MODEL} has higher error rate than the best alternative.",
                           label="tab:diverge-worse", col_formats=col_fmts)

    # UniLID better
    gaps_better = sorted(gaps, key=lambda x: x["gap"])[:TOP_N]
    md_better_rows, latex_better_rows = _make_rows(gaps_better)
    md_better = to_markdown(md_better_rows, headers, caption=f"Languages where {PRIMARY_MODEL} is most better than alternatives")
    latex_better = to_latex(latex_better_rows, headers,
                            caption=f"Languages where {PRIMARY_MODEL} has lower error rate than the best alternative.",
                            label="tab:diverge-better", col_formats=col_fmts)

    md = md_worse + "\n" + md_better
    latex = latex_worse + "\n\n" + latex_better
    return md, latex


def generate_all_comparisons(data: dict) -> dict[str, tuple[str, str]]:
    return {
        "table5_error_overlap": table_error_overlap(data),
        "table6_per_script_winner": table_per_script_winner(data),
        "table7_divergences": table_divergences(data),
    }
