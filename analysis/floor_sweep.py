"""Sweep the log-probability floor parameter and measure prediction accuracy.

Clamps all per-language log-prob weights (including OOV tokens at -1e30) to
different floor values and re-evaluates on a 500k sample.
"""

from __future__ import annotations

import gc
import importlib
import os
import random
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analysis.config import (
    DATA_DIR,
    DEFAULT_SAMPLE_SIZE,
    LENGTH_BINS,
    LENGTH_LABELS,
    SAMPLE_SEED,
    TOTAL_LINES,
)
from analysis.metrics import compute_metrics
from analysis.format_utils import to_markdown, to_latex
from analysis.sample_data import load_sample


UNILID_MODEL_PATH = "/capstor/scratch/cscs/cmeister747/unilid_analysis/glotlidc.unilid"
BATCH_SIZE = 10_000
FLOOR_VALUES = [None, -22.0, -15.0, -10.0]


def _load_model_and_raw_weights(model_path: str = UNILID_MODEL_PATH):
    """Load UnilidModel and return it alongside the raw weight memmap.

    The model gets the original (unclamped) weights pushed to Rust on init.
    The memmap is kept for repeated clamping.
    """
    # Import model_io without triggering torch from __init__.py
    spec = importlib.util.spec_from_file_location(
        "model_io",
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "UNILID", "unilid", "model_io.py"),
    )
    model_io = importlib.util.module_from_spec(spec)
    sys.modules["unilid.model_io"] = model_io
    spec.loader.exec_module(model_io)

    # Load raw weights (memmap) separately so we can clamp from them
    base_tok, weights_mmap, langs = model_io.load_unilid(model_path)

    # Build the model (pushes original weights to Rust)
    model = model_io.UnilidModel(model_path)

    return model, weights_mmap


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


def clamp_weights(weights_mmap: np.ndarray, floor: float) -> np.ndarray:
    """Copy weights and clamp all values below floor (including OOV at -1e30)."""
    w = np.array(weights_mmap, dtype=np.float32)
    w[w < floor] = np.float32(floor)
    return w


def predict_all(texts: list[str], model, batch_size: int = BATCH_SIZE) -> list[str]:
    """Run raw-score prediction on all texts in batches."""
    preds = []
    n = len(texts)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch = texts[start:end]
        results = model.predict_batch(batch)
        preds.extend(lang for lang, _, _ in results)
        if (start // batch_size) % 10 == 0:
            print(f"    {start + len(batch):,} / {n:,}", flush=True)

    return preds


def generate_floor_sweep(
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    output_dir: str = "outputs",
    model_path: str = UNILID_MODEL_PATH,
    floor_values: list[float | None] = FLOOR_VALUES,
):
    """Sweep floor values and generate comparison tables + figures."""
    print("  Loading model and raw weights...")
    model, weights_mmap = _load_model_and_raw_weights(model_path)

    # Diagnostics: count affected tokens per floor value
    total_elements = weights_mmap.size
    print(f"\n  Weight matrix: {weights_mmap.shape[0]} langs x {weights_mmap.shape[1]} vocab = {total_elements:,} elements")
    for floor in floor_values:
        if floor is None:
            continue
        affected = (weights_mmap < floor).sum()
        pct = affected / total_elements * 100
        print(f"  Floor {floor}: {affected:,} elements clamped ({pct:.1f}%)")
    print()

    print("  Streaming sampled texts...")
    texts = _stream_sampled_texts(sample_size)

    print("  Loading existing sample for y_true and metadata...")
    data = load_sample(sample_size)
    y_true = np.array(data["y_true"])
    text_lengths = np.array(data["text_lengths"])

    if len(texts) != len(y_true):
        raise RuntimeError(f"Text count {len(texts)} != y_true count {len(y_true)}")

    tables_dir = os.path.join(output_dir, "tables")
    figures_dir = os.path.join(output_dir, "figures")
    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    # ===================================================================
    # Run predictions for each floor value
    # ===================================================================
    all_preds = {}
    all_metrics = {}

    for fi, floor in enumerate(floor_values):
        label = "None" if floor is None else f"{floor}"
        print(f"  Running floor={label}...")

        if floor is None and fi > 0:
            # Re-push original weights if baseline isn't first
            print(f"    Re-pushing original weights to Rust...")
            model.model.set_weight_sets(np.array(weights_mmap, dtype=np.float32).tolist())
            gc.collect()
        elif floor is not None:
            clamped = clamp_weights(weights_mmap, floor)
            print(f"    Pushing clamped weights to Rust...")
            model.model.set_weight_sets(clamped.tolist())
            del clamped
            gc.collect()

        preds = np.array(predict_all(texts, model))
        all_preds[floor] = preds
        all_metrics[floor] = compute_metrics(y_true, preds)
        m = all_metrics[floor]
        print(f"    Acc={m['accuracy']:.4f}, Ma-F1={m['macro_f1']:.4f}")

    del texts
    gc.collect()

    # ===================================================================
    # Table: floor sweep summary
    # ===================================================================
    headers_sweep = ["Floor", "Accuracy", "Macro F1"]
    rows_sweep_md = []
    rows_sweep_tex = []
    for floor in floor_values:
        m = all_metrics[floor]
        label = "None" if floor is None else f"{floor}"
        rows_sweep_md.append([label, f"{m['accuracy']:.3f}", f"{m['macro_f1']:.3f}"])
        rows_sweep_tex.append([label, m["accuracy"], m["macro_f1"]])

    md_sweep = to_markdown(rows_sweep_md, headers_sweep,
                           caption="Floor sweep: accuracy vs log-prob floor")
    tex_sweep = to_latex(rows_sweep_tex, headers_sweep,
                         caption="Floor sweep: accuracy vs log-prob floor.",
                         label="tab:floor-sweep",
                         col_formats=["str", "metric", "metric"])

    # ===================================================================
    # Table: floor sweep by text length
    # ===================================================================
    headers_len = ["Floor"] + LENGTH_LABELS
    rows_len_md = []
    for floor in floor_values:
        label = "None" if floor is None else f"{floor}"
        row = [label]
        for i in range(len(LENGTH_LABELS)):
            lo, hi = LENGTH_BINS[i], LENGTH_BINS[i + 1]
            mask = (text_lengths >= lo) & (text_lengths < hi)
            if mask.sum() == 0:
                row.append("--")
            else:
                acc = (all_preds[floor][mask] == y_true[mask]).mean()
                row.append(f"{acc:.3f}")
        rows_len_md.append(row)

    md_len = to_markdown(rows_len_md, headers_len,
                         caption="Floor sweep: accuracy by text length")

    # ===================================================================
    # Figure: accuracy vs floor
    # ===================================================================
    # Use only the numeric floor values for the x-axis
    numeric_floors = [f for f in floor_values if f is not None]
    fig, ax = plt.subplots(figsize=(10, 6))

    # Overall
    accs = [all_metrics[f]["accuracy"] for f in numeric_floors]
    baseline_acc = all_metrics[None]["accuracy"]
    ax.axhline(baseline_acc, color="gray", linestyle="--", linewidth=1, label=f"Baseline ({baseline_acc:.3f})")
    ax.plot(numeric_floors, accs, "k-o", linewidth=2, markersize=6, label="Overall")

    # Per length bin
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(LENGTH_LABELS)))
    for j, (lbl, color) in enumerate(zip(LENGTH_LABELS, colors)):
        lo, hi = LENGTH_BINS[j], LENGTH_BINS[j + 1]
        mask = (text_lengths >= lo) & (text_lengths < hi)
        if mask.sum() == 0:
            continue
        bin_accs = [(all_preds[f][mask] == y_true[mask]).mean() for f in numeric_floors]
        ax.plot(numeric_floors, bin_accs, "-s", color=color, markersize=4, label=lbl)

    ax.set_xlabel("Log-probability floor")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy vs log-prob floor (including OOV clamping)")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()  # more negative (less aggressive) on the right
    fig.tight_layout()
    fig.savefig(os.path.join(figures_dir, "floor_sweep.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ===================================================================
    # Detailed breakdown at each floor vs baseline
    # ===================================================================
    baseline_preds = all_preds[None]
    for floor in numeric_floors:
        preds = all_preds[floor]
        changed = baseline_preds != preds
        corrected = changed & (preds == y_true) & (baseline_preds != y_true)
        broken = changed & (preds != y_true) & (baseline_preds == y_true)
        print(f"\n  Floor={floor}:")
        print(f"    Changed: {changed.sum():,} / {len(y_true):,}")
        print(f"    Corrected: {corrected.sum():,}, Broken: {broken.sum():,}, Net: {corrected.sum() - broken.sum():+,}")

    # ===================================================================
    # Write outputs
    # ===================================================================
    full_md = "\n".join([md_sweep, md_len])
    with open(os.path.join(tables_dir, "floor_sweep.md"), "w") as f:
        f.write(full_md)
    with open(os.path.join(tables_dir, "floor_sweep.tex"), "w") as f:
        f.write(tex_sweep)

    print(f"\n  Outputs: {tables_dir}/floor_sweep.{{md,tex}}, {figures_dir}/floor_sweep.png")
