"""Confusion matrices for identified language clusters."""

from __future__ import annotations

import os

import numpy as np

from analysis.config import CONFUSION_CLUSTERS, MODEL_NAMES
from analysis.format_utils import confusion_matrix_to_latex, plot_confusion_matrix


def _build_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str],
) -> np.ndarray:
    """Build a confusion matrix restricted to the given labels.

    Rows = true, columns = predicted.  Predictions that fall outside the
    label set are collected in an implicit "Other" column which is NOT
    included here — only in-cluster confusions are shown.
    """
    label_to_idx = {l: i for i, l in enumerate(labels)}
    n = len(labels)
    cm = np.zeros((n, n), dtype=int)

    for t, p in zip(y_true, y_pred):
        ti = label_to_idx.get(t)
        pi = label_to_idx.get(p)
        if ti is not None and pi is not None:
            cm[ti, pi] += 1

    return cm


def generate_confusion_matrices(
    data: dict,
    output_dir: str = "outputs",
) -> dict[str, dict[str, str]]:
    """Generate confusion matrices for all clusters and all models.

    Returns
    -------
    dict mapping cluster_key -> {"latex": full_latex_string, "figures": [paths]}
    """
    y_true = np.array(data["y_true"])
    preds = {name: np.array(data[f"pred_{name}"]) for name in MODEL_NAMES}

    figures_dir = os.path.join(output_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    results = {}

    for cluster_key, cluster_info in CONFUSION_CLUSTERS.items():
        title = cluster_info["title"]
        languages = cluster_info["languages"]

        # Filter to samples whose true label is in this cluster
        mask = np.isin(y_true, languages)
        n_samples = int(mask.sum())
        if n_samples == 0:
            print(f"  WARNING: no samples for cluster '{cluster_key}', skipping")
            continue

        yt_cluster = y_true[mask]

        # Only include languages that actually appear in the sample
        present_langs = [l for l in languages if (yt_cluster == l).sum() > 0]
        if len(present_langs) < 2:
            print(f"  WARNING: <2 languages present for cluster '{cluster_key}', skipping")
            continue

        latex_parts = []
        figure_paths = []

        for model_name in MODEL_NAMES:
            yp_cluster = preds[model_name][mask]
            cm = _build_confusion_matrix(yt_cluster, yp_cluster, present_langs)

            full_title = f"{title} — {model_name}"
            tex_label = f"tab:cm-{cluster_key}-{model_name.lower().replace(' ', '-')}"
            safe_fname = f"cm_{cluster_key}_{model_name.lower().replace(' ', '_').replace('-', '_')}.png"

            latex_str = confusion_matrix_to_latex(
                cm, present_langs,
                caption=f"Confusion matrix: {full_title} (row-normalized).",
                label=tex_label,
                normalize=True,
            )
            latex_parts.append(latex_str)

            fig_path = os.path.join(figures_dir, safe_fname)
            plot_confusion_matrix(
                cm, present_langs,
                title=full_title,
                path=fig_path,
                normalize=True,
            )
            figure_paths.append(fig_path)

        results[cluster_key] = {
            "latex": "\n\n".join(latex_parts),
            "figures": figure_paths,
            "n_samples": n_samples,
            "n_languages": len(present_langs),
        }

    return results
