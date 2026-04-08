"""Generate predictions with length-normalized scoring and compare against original UniLID.

Supports sweeping the normalization exponent alpha in score / n_tokens^alpha.
"""

from __future__ import annotations

import importlib
import os
import random
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analysis.config import (
    CACHE_DIR,
    DATA_DIR,
    DEFAULT_SAMPLE_SIZE,
    LENGTH_BINS,
    LENGTH_LABELS,
    RESOURCE_BINS,
    RESOURCE_LABELS,
    SAMPLE_SEED,
    TOP_SCRIPTS,
    TOTAL_LINES,
)
from analysis.metrics import compute_metrics
from analysis.format_utils import to_markdown, to_latex
from analysis.sample_data import load_sample


UNILID_MODEL_PATH = "/capstor/scratch/cscs/cmeister747/unilid_analysis/glotlidc.unilid"
BATCH_SIZE = 10_000
ALPHA_VALUES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


def _load_unilid_model(model_path: str = UNILID_MODEL_PATH):
    """Load UnilidModel without triggering torch import from __init__.py."""
    spec = importlib.util.spec_from_file_location(
        "model_io",
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "UNILID", "unilid", "model_io.py"),
    )
    model_io = importlib.util.module_from_spec(spec)
    sys.modules["unilid.model_io"] = model_io
    spec.loader.exec_module(model_io)
    return model_io.UnilidModel(model_path)


def _stream_sampled_texts(sample_size: int = DEFAULT_SAMPLE_SIZE) -> list[str]:
    """Stream the test file and collect texts for the sampled indices."""
    random.seed(SAMPLE_SEED)
    sample_indices = set(sorted(random.sample(range(TOTAL_LINES), sample_size)))

    test_path = os.path.join(DATA_DIR, "glotlid_correct_test.txt")
    texts = []

    with open(test_path, "r") as f:
        for i, line in enumerate(f):
            if i in sample_indices:
                parts = line.split(" ", 1)
                text = parts[1].rstrip("\n") if len(parts) > 1 else ""
                texts.append(text)
            if i % 10_000_000 == 0:
                print(f"    Scanned {i:,} / {TOTAL_LINES:,}", flush=True)

    print(f"    Collected {len(texts):,} texts")
    return texts


def predict_all(texts: list[str], model, alpha: float = 0.0) -> list[str]:
    """Run prediction on all texts in batches, return list of predicted labels.

    alpha=0.0 uses raw scoring (no normalization).
    alpha>0 uses score / n_tokens^alpha.
    """
    preds = []
    n = len(texts)

    for start in range(0, n, BATCH_SIZE):
        end = min(start + BATCH_SIZE, n)
        batch = texts[start:end]
        if alpha == 0.0:
            results = model.predict_batch(batch)
        else:
            results = model.predict_normalized_batch(batch, alpha=alpha)
        preds.extend(lang for lang, _, _ in results)
        if (start // BATCH_SIZE) % 10 == 0:
            print(f"    {start + len(batch):,} / {n:,}", flush=True)

    return preds


def generate_alpha_sweep(
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    output_dir: str = "outputs",
    model_path: str = UNILID_MODEL_PATH,
    alpha_values: list[float] = ALPHA_VALUES,
):
    """Sweep alpha values and generate comparison tables + figures."""
    print("  Loading model...")
    model = _load_unilid_model(model_path)

    print("  Streaming sampled texts...")
    texts = _stream_sampled_texts(sample_size)

    print("  Loading existing sample for y_true and metadata...")
    data = load_sample(sample_size)
    y_true = np.array(data["y_true"])
    text_lengths = np.array(data["text_lengths"])
    train_counts = data["train_counts"]
    true_train_cts = np.array([train_counts.get(lab, 0) for lab in y_true])
    scripts = np.array([lab.rsplit("_", 1)[-1] for lab in y_true])

    assert len(texts) == len(y_true), f"Text count {len(texts)} != y_true count {len(y_true)}"

    tables_dir = os.path.join(output_dir, "tables")
    figures_dir = os.path.join(output_dir, "figures")
    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    # ===================================================================
    # Run predictions for each alpha
    # ===================================================================
    all_preds = {}
    all_metrics = {}

    for alpha in alpha_values:
        print(f"  Running alpha={alpha:.1f}...")
        preds = np.array(predict_all(texts, model, alpha=alpha))
        all_preds[alpha] = preds
        all_metrics[alpha] = compute_metrics(y_true, preds)
        print(f"    Acc={all_metrics[alpha]['accuracy']:.4f}, Ma-F1={all_metrics[alpha]['macro_f1']:.4f}")

    # Free texts
    del texts

    # ===================================================================
    # Table: alpha sweep summary
    # ===================================================================
    headers_sweep = ["Alpha", "Accuracy", "Macro F1"]
    rows_sweep_md = []
    rows_sweep_latex = []
    for alpha in alpha_values:
        m = all_metrics[alpha]
        rows_sweep_md.append([f"{alpha:.1f}", f"{m['accuracy']:.3f}", f"{m['macro_f1']:.3f}"])
        rows_sweep_latex.append([f"{alpha:.1f}", m["accuracy"], m["macro_f1"]])

    md_sweep = to_markdown(rows_sweep_md, headers_sweep, caption="Alpha sweep: accuracy vs normalization strength")
    tex_sweep = to_latex(rows_sweep_latex, headers_sweep,
                          caption="Alpha sweep: accuracy vs normalization strength.",
                          label="tab:alpha-sweep", col_formats=["str", "metric", "metric"])

    # ===================================================================
    # Table: alpha sweep by text length
    # ===================================================================
    headers_len = ["Alpha"] + LENGTH_LABELS
    rows_len_md = []
    for alpha in alpha_values:
        row = [f"{alpha:.1f}"]
        for i in range(len(LENGTH_LABELS)):
            lo, hi = LENGTH_BINS[i], LENGTH_BINS[i + 1]
            mask = (text_lengths >= lo) & (text_lengths < hi)
            if mask.sum() == 0:
                row.append("--")
            else:
                acc = (all_preds[alpha][mask] == y_true[mask]).mean()
                row.append(f"{acc:.3f}")
        rows_len_md.append(row)

    md_len = to_markdown(rows_len_md, headers_len, caption="Alpha sweep: accuracy by text length")

    # ===================================================================
    # Figure: accuracy vs alpha (overall + per length bin)
    # ===================================================================
    fig, ax = plt.subplots(figsize=(10, 6))

    # Overall
    accs = [all_metrics[a]["accuracy"] for a in alpha_values]
    ax.plot(alpha_values, accs, "k-o", linewidth=2, markersize=6, label="Overall")

    # Per length bin
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(LENGTH_LABELS)))
    for j, (lbl, color) in enumerate(zip(LENGTH_LABELS, colors)):
        lo, hi = LENGTH_BINS[j], LENGTH_BINS[j + 1]
        mask = (text_lengths >= lo) & (text_lengths < hi)
        if mask.sum() == 0:
            continue
        bin_accs = [(all_preds[a][mask] == y_true[mask]).mean() for a in alpha_values]
        ax.plot(alpha_values, bin_accs, "-s", color=color, markersize=4, label=lbl)

    ax.set_xlabel("Alpha (normalization exponent)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy vs normalization strength (score / n_tokens^alpha)")
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(figures_dir, "alpha_sweep.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ===================================================================
    # Find best alpha and detailed breakdown
    # ===================================================================
    best_alpha = max(alpha_values, key=lambda a: all_metrics[a]["accuracy"])
    best_m = all_metrics[best_alpha]
    print(f"\n  Best alpha: {best_alpha:.1f} (Acc={best_m['accuracy']:.4f}, Ma-F1={best_m['macro_f1']:.4f})")

    # Detailed breakdown at best alpha (if != 0)
    if best_alpha != 0.0:
        best_preds = all_preds[best_alpha]
        raw_preds = all_preds[0.0]

        changed = raw_preds != best_preds
        corrected = changed & (best_preds == y_true) & (raw_preds != y_true)
        broken = changed & (best_preds != y_true) & (raw_preds == y_true)

        print(f"  At alpha={best_alpha:.1f}:")
        print(f"    Changed: {changed.sum():,} / {len(y_true):,}")
        print(f"    Corrected: {corrected.sum():,}, Broken: {broken.sum():,}, Net: {corrected.sum() - broken.sum():+,}")

    # ===================================================================
    # Write outputs
    # ===================================================================
    full_md = "\n".join([md_sweep, md_len])
    with open(os.path.join(tables_dir, "alpha_sweep.md"), "w") as f:
        f.write(full_md)
    with open(os.path.join(tables_dir, "alpha_sweep.tex"), "w") as f:
        f.write(tex_sweep)

    print(f"  Outputs: {tables_dir}/alpha_sweep.{{md,tex}}, {figures_dir}/alpha_sweep.png")


# Keep the original function for backward compatibility
def generate_normalized_analysis(
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    output_dir: str = "outputs",
    model_path: str = UNILID_MODEL_PATH,
):
    """Run the alpha sweep (replaces the old single-alpha comparison)."""
    generate_alpha_sweep(sample_size=sample_size, output_dir=output_dir, model_path=model_path)
