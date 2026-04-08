"""Output formatting utilities: markdown tables, LaTeX tables, confusion matrix plots."""

from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


# ---------------------------------------------------------------------------
# Number formatting helpers
# ---------------------------------------------------------------------------

def _fmt_int(v) -> str:
    return f"{int(v):,}"


def _fmt_pct(v) -> str:
    return f"{v:.1f}\\%"


def _fmt_metric(v) -> str:
    return f"{v:.3f}"


def _fmt_fpr(v) -> str:
    """Format FPR as value × 10^5 (i.e., multiply by 100,000)."""
    return f"{v * 100_000:.1f}"


def _fmt_cell(v, fmt: str) -> str:
    if fmt == "int":
        return _fmt_int(v)
    if fmt == "pct":
        return _fmt_pct(v)
    if fmt == "metric":
        return _fmt_metric(v)
    if fmt == "fpr":
        return _fmt_fpr(v)
    return str(v)


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------

def to_markdown(rows: list[list], headers: list[str], caption: str = "") -> str:
    """Render a table as a GitHub-flavoured markdown string.

    Parameters
    ----------
    rows : list of lists (one per table row, same length as headers)
    headers : column headers
    caption : optional caption printed above the table
    """
    col_widths = [len(h) for h in headers]
    str_rows = []
    for row in rows:
        sr = [str(c) for c in row]
        str_rows.append(sr)
        for j, c in enumerate(sr):
            col_widths[j] = max(col_widths[j], len(c))

    def _pad(cells):
        return "| " + " | ".join(c.rjust(w) for c, w in zip(cells, col_widths)) + " |"

    lines = []
    if caption:
        lines.append(f"### {caption}")
        lines.append("")
    lines.append(_pad(headers))
    lines.append("|" + "|".join("-" * (w + 2) for w in col_widths) + "|")
    for sr in str_rows:
        lines.append(_pad(sr))
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# LaTeX
# ---------------------------------------------------------------------------

def to_latex(
    rows: list[list],
    headers: list[str],
    caption: str = "",
    label: str = "",
    col_formats: list[str] | None = None,
) -> str:
    """Render a table as a LaTeX booktabs table.

    Parameters
    ----------
    rows : list of lists
    headers : column headers
    caption, label : LaTeX caption and label
    col_formats : per-column format hints ("str", "int", "pct", "metric")
        If None, all cells are passed through as-is.
    """
    n_cols = len(headers)
    alignment = "l" + "r" * (n_cols - 1)

    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\small",
        f"\\begin{{tabular}}{{{alignment}}}",
        "\\toprule",
        " & ".join(f"\\textbf{{{h}}}" for h in headers) + " \\\\",
        "\\midrule",
    ]

    for row in rows:
        cells = []
        for j, v in enumerate(row):
            if col_formats and j < len(col_formats):
                cells.append(_fmt_cell(v, col_formats[j]))
            else:
                cells.append(str(v))
        lines.append(" & ".join(cells) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    if caption:
        lines.append(f"\\caption{{{caption}}}")
    if label:
        lines.append(f"\\label{{{label}}}")
    lines.append("\\end{table}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Confusion matrix – LaTeX
# ---------------------------------------------------------------------------

def confusion_matrix_to_latex(
    cm: np.ndarray,
    labels: list[str],
    caption: str = "",
    label: str = "",
    normalize: bool = True,
) -> str:
    """Render a confusion matrix as a LaTeX table with cellcolor shading.

    Parameters
    ----------
    cm : 2-D array of counts (true x predicted)
    labels : class labels (same order as rows/cols)
    normalize : if True, row-normalize and shade cells by recall
    """
    n = len(labels)
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1, row_sums)
        cm_norm = cm / row_sums
    else:
        cm_norm = cm.astype(float)

    alignment = "l" + "r" * n
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\small",
        f"\\begin{{tabular}}{{{alignment}}}",
        "\\toprule",
        " & " + " & ".join(f"\\textbf{{{_short_label(l)}}}" for l in labels) + " \\\\",
        "\\midrule",
    ]

    for i, row_label in enumerate(labels):
        cells = [f"\\textbf{{{_short_label(row_label)}}}"]
        for j in range(n):
            val = cm_norm[i, j]
            count = int(cm[i, j])
            intensity = int(val * 100)
            color_cmd = f"\\cellcolor{{blue!{intensity}}}"
            text_color = "\\color{white}" if intensity > 50 else ""
            if normalize:
                cells.append(f"{color_cmd}{text_color}{val:.2f}")
            else:
                cells.append(f"{color_cmd}{text_color}{count:,}")
        lines.append(" & ".join(cells) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    if caption:
        lines.append(f"\\caption{{{caption}}}")
    if label:
        lines.append(f"\\label{{{label}}}")
    lines.append("\\end{table}")
    return "\n".join(lines)


def _short_label(label: str) -> str:
    """Shorten lang_Script to just the ISO 639-3 code."""
    return label.split("_")[0]


# ---------------------------------------------------------------------------
# Confusion matrix – matplotlib image
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    cm: np.ndarray,
    labels: list[str],
    title: str,
    path: str,
    normalize: bool = True,
    figsize: tuple[float, float] | None = None,
) -> None:
    """Save a confusion matrix heatmap as a PNG.

    Parameters
    ----------
    cm : 2-D array of counts
    labels : class labels
    title : plot title
    path : output file path
    normalize : row-normalize before plotting
    """
    n = len(labels)
    if figsize is None:
        side = max(4, n * 0.8)
        figsize = (side, side)

    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1, row_sums)
        cm_plot = cm / row_sums
    else:
        cm_plot = cm.astype(float)

    short_labels = [_short_label(l) for l in labels]

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm_plot, cmap="Blues", vmin=0, vmax=1 if normalize else None, aspect="equal")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(short_labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(short_labels, fontsize=9)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

    # Annotate cells
    for i in range(n):
        for j in range(n):
            val = cm_plot[i, j]
            count = int(cm[i, j])
            text_color = "white" if val > 0.5 else "black"
            if normalize:
                txt = f"{val:.2f}\n({count:,})"
            else:
                txt = f"{count:,}"
            ax.text(j, i, txt, ha="center", va="center", color=text_color, fontsize=7)

    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
