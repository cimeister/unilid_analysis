"""Tables 1-4: Overall, by text length, by resource level, by script."""

from __future__ import annotations

import numpy as np

from analysis.config import (
    LENGTH_BINS,
    LENGTH_LABELS,
    MODEL_NAMES,
    RESOURCE_BINS,
    RESOURCE_LABELS,
    TOP_SCRIPTS,
)
from analysis.metrics import compute_metrics
from analysis.format_utils import to_markdown, to_latex


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_arrays(data: dict) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    """Unpack the sample dict into arrays used by all table functions."""
    y_true = np.array(data["y_true"])
    preds = {}
    for name in MODEL_NAMES:
        key = f"pred_{name}"
        preds[name] = np.array(data[key])
    text_lengths = np.array(data["text_lengths"])
    train_counts = data["train_counts"]
    true_train_cts = np.array([train_counts.get(lab, 0) for lab in y_true])
    scripts = np.array([lab.rsplit("_", 1)[-1] for lab in y_true])
    return y_true, preds, text_lengths, true_train_cts, scripts


# ---------------------------------------------------------------------------
# Table 1: Overall comparison
# ---------------------------------------------------------------------------

def table_overall(data: dict) -> tuple[str, str]:
    y_true, preds, *_ = _extract_arrays(data)
    FPR_SCALE = 100_000
    fpr_header = "Ma-FPR (x1e5)"
    headers = ["Model", "N", "Accuracy", "Macro F1", fpr_header]
    rows = []
    all_metrics = {}
    for name in MODEL_NAMES:
        m = compute_metrics(y_true, preds[name])
        all_metrics[name] = m
        rows.append([name, f"{len(y_true):,}", f"{m['accuracy']:.3f}", f"{m['macro_f1']:.3f}", f"{m['macro_fpr'] * FPR_SCALE:.1f}"])

    md = to_markdown(rows, headers, caption="Overall Model Comparison")
    latex = to_latex(
        [[name, len(y_true), all_metrics[name]["accuracy"], all_metrics[name]["macro_f1"], all_metrics[name]["macro_fpr"]]
         for name in MODEL_NAMES],
        headers,
        caption="Overall model comparison on GlotLID test set.",
        label="tab:overall",
        col_formats=["str", "int", "metric", "metric", "fpr"],
    )
    return md, latex


# ---------------------------------------------------------------------------
# Table 2-4: Generic binned table
# ---------------------------------------------------------------------------

def _binned_table(
    data: dict,
    bin_values: np.ndarray,
    bins: list[float],
    bin_labels: list[str],
    y_true: np.ndarray,
    preds: dict[str, np.ndarray],
    caption: str,
    label: str,
    extra_col_name: str = "Bin",
    show_n_langs: bool = True,
) -> tuple[str, str]:
    """Generate markdown and LaTeX for a binned breakdown."""
    left_cols = [extra_col_name]
    if show_n_langs:
        left_cols.append("# Langs")
    left_cols.append("N")

    md_headers = list(left_cols)
    for name in MODEL_NAMES:
        md_headers.extend([f"Acc ({name})", f"Ma-F1 ({name})", f"Ma-FPR x1e5 ({name})"])

    rows_md = []
    rows_latex = []

    for i in range(len(bin_labels)):
        lo, hi = bins[i], bins[i + 1]
        mask = (bin_values >= lo) & (bin_values < hi)
        n = int(mask.sum())
        if n == 0:
            continue

        n_langs = len(set(y_true[mask]))
        row_left_md = [bin_labels[i]]
        row_left_latex = [bin_labels[i]]
        if show_n_langs:
            row_left_md.append(f"{n_langs:,}")
            row_left_latex.append(n_langs)
        row_left_md.append(f"{n:,}")
        row_left_latex.append(n)

        row_metrics_md = []
        row_metrics_latex = []
        for name in MODEL_NAMES:
            m = compute_metrics(y_true[mask], preds[name][mask])
            row_metrics_md.extend([f"{m['accuracy']:.3f}", f"{m['macro_f1']:.3f}", f"{m['macro_fpr'] * 100_000:.1f}"])
            row_metrics_latex.extend([m["accuracy"], m["macro_f1"], m["macro_fpr"]])

        rows_md.append(row_left_md + row_metrics_md)
        rows_latex.append(row_left_latex + row_metrics_latex)

    col_fmts = ["str"]
    if show_n_langs:
        col_fmts.append("int")
    col_fmts.append("int")
    for _ in MODEL_NAMES:
        col_fmts.extend(["metric", "metric", "fpr"])

    md = to_markdown(rows_md, md_headers, caption=caption)
    latex = to_latex(rows_latex, md_headers, caption=caption, label=label, col_formats=col_fmts)
    return md, latex


def table_by_length(data: dict) -> tuple[str, str]:
    y_true, preds, text_lengths, _, _ = _extract_arrays(data)
    return _binned_table(
        data, text_lengths, LENGTH_BINS, LENGTH_LABELS,
        y_true, preds,
        caption="Accuracy by text length (characters)",
        label="tab:by-length",
        extra_col_name="Length",
        show_n_langs=True,
    )


def table_by_resource(data: dict) -> tuple[str, str]:
    y_true, preds, _, true_train_cts, _ = _extract_arrays(data)
    return _binned_table(
        data, true_train_cts, RESOURCE_BINS, RESOURCE_LABELS,
        y_true, preds,
        caption="Accuracy by training resource level",
        label="tab:by-resource",
        extra_col_name="Resource",
        show_n_langs=True,
    )


def table_by_script(data: dict) -> tuple[str, str]:
    y_true, preds, _, _, scripts = _extract_arrays(data)

    left_cols = ["Script", "# Langs", "N"]
    md_headers = list(left_cols)
    for name in MODEL_NAMES:
        md_headers.extend([f"Acc ({name})", f"Ma-F1 ({name})", f"Ma-FPR x1e5 ({name})"])

    rows_md = []
    rows_latex = []

    script_order = list(TOP_SCRIPTS) + ["Other"]
    for s in script_order:
        if s == "Other":
            mask = ~np.isin(scripts, TOP_SCRIPTS)
        else:
            mask = scripts == s
        n = int(mask.sum())
        if n == 0:
            continue

        n_langs = len(set(y_true[mask]))
        row_left_md = [s, f"{n_langs:,}", f"{n:,}"]
        row_left_latex = [s, n_langs, n]

        row_metrics_md = []
        row_metrics_latex = []
        for name in MODEL_NAMES:
            m = compute_metrics(y_true[mask], preds[name][mask])
            row_metrics_md.extend([f"{m['accuracy']:.3f}", f"{m['macro_f1']:.3f}", f"{m['macro_fpr'] * 100_000:.1f}"])
            row_metrics_latex.extend([m["accuracy"], m["macro_f1"], m["macro_fpr"]])

        rows_md.append(row_left_md + row_metrics_md)
        rows_latex.append(row_left_latex + row_metrics_latex)

    col_fmts = ["str", "int", "int"]
    for _ in MODEL_NAMES:
        col_fmts.extend(["metric", "metric", "fpr"])

    md = to_markdown(rows_md, md_headers, caption="Accuracy by script")
    latex = to_latex(rows_latex, md_headers, caption="Accuracy by script.", label="tab:by-script", col_formats=col_fmts)
    return md, latex


def generate_all_tables(data: dict) -> dict[str, tuple[str, str]]:
    """Return dict mapping table name -> (markdown, latex)."""
    return {
        "table1_overall": table_overall(data),
        "table2_by_length": table_by_length(data),
        "table3_by_resource": table_by_resource(data),
        "table4_by_script": table_by_script(data),
    }
