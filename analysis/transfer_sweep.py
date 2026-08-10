"""Experiment 9: Distribution transfer for low-resource languages.

Two approaches:
  9a: Interpolate low-resource distributions with closest high-resource language
  9b: Interpolate low-resource distributions with script-average distribution

All interpolation in probability space (arithmetic mean), producing valid distributions.
"""

from __future__ import annotations

import gc
import importlib
import json
import os
import random
import struct
import sys

import numpy as np
from scipy.special import logsumexp as scipy_logsumexp

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
    TOTAL_LINES,
)
from analysis.metrics import compute_metrics
from analysis.format_utils import to_markdown, to_latex
from analysis.sample_data import load_sample


UNILID_MODEL_PATH = "/capstor/scratch/cscs/cmeister747/unilid_analysis/glotlidc.unilid"
BATCH_SIZE = 10_000
LAMBDA_VALUES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
LOW_RESOURCE_THRESHOLD = 5_000
HIGH_RESOURCE_THRESHOLD = 20_000


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_model_data(model_path: str = UNILID_MODEL_PATH):
    """Load base tokenizer scores, per-language weights, language list."""
    HEADER_FMT = "<8sIIIII4x"
    HEADER_SIZE = struct.calcsize(HEADER_FMT)

    with open(model_path, "rb") as f:
        header = f.read(HEADER_SIZE)
        _, _, num_langs, vocab_size, base_tok_len, langs_len = struct.unpack(HEADER_FMT, header)
        f.read(base_tok_len)  # skip base tokenizer
        langs = json.loads(f.read(langs_len).decode("utf-8"))
        weights_offset = f.tell()

    weights = np.memmap(
        model_path, dtype=np.float32, mode="r",
        offset=weights_offset, shape=(num_langs, vocab_size),
    )
    lang_to_idx = {l: i for i, l in enumerate(langs)}

    print(f"  Loaded {num_langs} languages, vocab size {vocab_size}")
    return weights, langs, lang_to_idx


def _load_train_counts():
    with open(os.path.join(DATA_DIR, "glotlid_train_counts.json")) as f:
        return json.load(f)


def _load_unilid_model(model_path: str = UNILID_MODEL_PATH):
    """Load UnilidModel (base mode: the analysis chain is defined on the base
    matrix and applies the unseen-token constant itself via
    floor_equalization.build_equalized_weights). The former
    spec_from_file_location single-file load broke once model_io.py gained
    intra-package imports (calibration-release branch), so the package is
    imported normally off sys.path."""
    unilid_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "UNILID")
    if unilid_dir not in sys.path:
        sys.path.insert(0, unilid_dir)
    from unilid.model_io import UnilidModel
    return UnilidModel(model_path, calibrated=False)


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


# ---------------------------------------------------------------------------
# Probability-space operations
# ---------------------------------------------------------------------------

def _logprobs_to_probs(log_w: np.ndarray) -> np.ndarray:
    """Convert log-probabilities to probabilities using numerically stable softmax."""
    # log_w: (vocab_size,) float64
    shifted = log_w - log_w.max()
    p = np.exp(shifted)
    p /= p.sum()
    return p


def _probs_to_logprobs(p: np.ndarray) -> np.ndarray:
    """Convert probabilities back to log-probabilities."""
    return np.log(np.maximum(p, 1e-30))


# ---------------------------------------------------------------------------
# Transfer pair identification (9a)
# ---------------------------------------------------------------------------

def identify_transfer_pairs(
    weights: np.ndarray,
    langs: list[str],
    lang_to_idx: dict[str, int],
    train_counts: dict[str, int],
) -> dict[str, tuple[str, float]]:
    """For each low-resource language, find the closest high-resource same-script language.

    Returns dict: low_resource_lang -> (high_resource_lang, symmetric_kl)
    """
    # Group languages by script
    script_groups = {}
    for lang in langs:
        script = lang.rsplit("_", 1)[-1] if "_" in lang else "Unknown"
        script_groups.setdefault(script, []).append(lang)

    # Identify low-resource and high-resource languages
    low_resource = {l for l in langs if train_counts.get(l, 0) < LOW_RESOURCE_THRESHOLD}
    high_resource = {l for l in langs if train_counts.get(l, 0) >= HIGH_RESOURCE_THRESHOLD}

    print(f"  Transfer pairs: {len(low_resource)} low-resource, {len(high_resource)} high-resource")

    # For each low-resource language, find closest high-resource same-script
    pairs = {}
    for lr_lang in sorted(low_resource):
        script = lr_lang.rsplit("_", 1)[-1] if "_" in lr_lang else None
        if not script:
            continue

        # Candidates: same-script high-resource languages
        candidates = [l for l in script_groups.get(script, []) if l in high_resource and l != lr_lang]
        if not candidates:
            continue

        # Compute symmetric KL to each candidate
        lr_idx = lang_to_idx[lr_lang]
        lr_probs = _logprobs_to_probs(weights[lr_idx].astype(np.float64))

        best_kl = float('inf')
        best_lang = None
        for cand in candidates:
            c_idx = lang_to_idx[cand]
            c_probs = _logprobs_to_probs(weights[c_idx].astype(np.float64))
            # Symmetric KL
            eps = 1e-30
            kl_ab = float(np.sum(lr_probs * np.log(np.maximum(lr_probs, eps) / np.maximum(c_probs, eps))))
            kl_ba = float(np.sum(c_probs * np.log(np.maximum(c_probs, eps) / np.maximum(lr_probs, eps))))
            sym_kl = (kl_ab + kl_ba) / 2
            if sym_kl < best_kl:
                best_kl = sym_kl
                best_lang = cand

        if best_lang:
            pairs[lr_lang] = (best_lang, best_kl)

    print(f"  Found {len(pairs)} transfer pairs")
    return pairs


# ---------------------------------------------------------------------------
# Script-average computation (9b)
# ---------------------------------------------------------------------------

def compute_script_averages(
    weights: np.ndarray,
    langs: list[str],
    lang_to_idx: dict[str, int],
) -> dict[str, np.ndarray]:
    """Compute per-script average distribution in probability space.

    Returns dict: script_code -> log_probs (vocab_size,) float32
    """
    script_probs = {}  # script -> running sum of probabilities
    script_counts = {}  # script -> count

    for lang in langs:
        script = lang.rsplit("_", 1)[-1] if "_" in lang else "Unknown"
        idx = lang_to_idx[lang]
        probs = _logprobs_to_probs(weights[idx].astype(np.float64))

        if script not in script_probs:
            script_probs[script] = np.zeros_like(probs)
            script_counts[script] = 0
        script_probs[script] += probs
        script_counts[script] += 1

    # Normalize to get averages, convert back to log-probs
    script_logprobs = {}
    for script, prob_sum in script_probs.items():
        avg_probs = prob_sum / script_counts[script]
        script_logprobs[script] = _probs_to_logprobs(avg_probs).astype(np.float32)
        print(f"    Script {script}: {script_counts[script]} languages")

    return script_logprobs


# ---------------------------------------------------------------------------
# Weight interpolation
# ---------------------------------------------------------------------------

def interpolate_weights_related(
    weights: np.ndarray,
    pairs: dict[str, tuple[str, float]],
    lang_to_idx: dict[str, int],
    lam: float,
) -> np.ndarray:
    """Interpolate low-resource weights with related high-resource in probability space."""
    w = np.array(weights, dtype=np.float32)

    for lr_lang, (hr_lang, _kl) in pairs.items():
        lr_idx = lang_to_idx[lr_lang]
        hr_idx = lang_to_idx[hr_lang]

        p_em = _logprobs_to_probs(weights[lr_idx].astype(np.float64))
        p_ref = _logprobs_to_probs(weights[hr_idx].astype(np.float64))
        p_new = lam * p_em + (1 - lam) * p_ref
        w[lr_idx] = _probs_to_logprobs(p_new).astype(np.float32)

    return w


def interpolate_weights_script_avg(
    weights: np.ndarray,
    script_logprobs: dict[str, np.ndarray],
    langs: list[str],
    lang_to_idx: dict[str, int],
    train_counts: dict[str, int],
    lam: float,
) -> np.ndarray:
    """Interpolate low-resource weights with script average in probability space."""
    w = np.array(weights, dtype=np.float32)

    for lang in langs:
        tc = train_counts.get(lang, 0)
        if tc >= LOW_RESOURCE_THRESHOLD:
            continue

        script = lang.rsplit("_", 1)[-1] if "_" in lang else None
        if not script or script not in script_logprobs:
            continue

        idx = lang_to_idx[lang]
        p_em = _logprobs_to_probs(weights[idx].astype(np.float64))
        p_avg = _logprobs_to_probs(script_logprobs[script].astype(np.float64))
        p_new = lam * p_em + (1 - lam) * p_avg
        w[idx] = _probs_to_logprobs(p_new).astype(np.float32)

    return w


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def predict_all(texts: list[str], model) -> list[str]:
    """Run raw-score prediction on all texts in batches."""
    preds = []
    n = len(texts)
    for start in range(0, n, BATCH_SIZE):
        end = min(start + BATCH_SIZE, n)
        batch = texts[start:end]
        results = model.predict_batch(batch)
        preds.extend(lang for lang, _, _ in results)
        if (start // BATCH_SIZE) % 10 == 0:
            print(f"    {start + len(batch):,} / {n:,}", flush=True)
    return preds


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

def generate_transfer_sweep(
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    output_dir: str = "outputs",
    model_path: str = UNILID_MODEL_PATH,
    lambda_values: list[float] = LAMBDA_VALUES,
):
    """Run transfer sweep: 9a (related language) and 9b (script average)."""
    print("  Loading model data...")
    weights, langs, lang_to_idx = _load_model_data(model_path)
    train_counts = _load_train_counts()

    print("  Loading UniLID model for prediction...")
    model = _load_unilid_model(model_path)

    print("  Identifying transfer pairs (9a)...")
    pairs = identify_transfer_pairs(weights, langs, lang_to_idx, train_counts)

    print("  Computing script averages (9b)...")
    script_logprobs = compute_script_averages(weights, langs, lang_to_idx)

    print("  Streaming sampled texts...")
    texts = _stream_sampled_texts(sample_size)

    print("  Loading sample for y_true and metadata...")
    data = load_sample(sample_size)
    y_true = np.array(data["y_true"])
    text_lengths = np.array(data["text_lengths"])
    true_train_cts = np.array([train_counts.get(lab, 0) for lab in y_true])

    if len(texts) != len(y_true):
        raise RuntimeError(f"Text count {len(texts)} != y_true count {len(y_true)}")

    tables_dir = os.path.join(output_dir, "tables")
    figures_dir = os.path.join(output_dir, "figures")
    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    # Resource masks for evaluation
    low_mask = true_train_cts < 500
    mid_mask = (true_train_cts >= 500) & (true_train_cts < LOW_RESOURCE_THRESHOLD)
    high_mask = true_train_cts >= LOW_RESOURCE_THRESHOLD

    # ===================================================================
    # Baseline prediction
    # ===================================================================
    print("  Running baseline (lambda=1.0, no transfer)...")
    baseline_preds = np.array(predict_all(texts, model))
    baseline_acc = (baseline_preds == y_true).mean()
    baseline_acc_low = (baseline_preds[low_mask] == y_true[low_mask]).mean() if low_mask.sum() > 0 else float('nan')
    baseline_acc_mid = (baseline_preds[mid_mask] == y_true[mid_mask]).mean() if mid_mask.sum() > 0 else float('nan')
    print(f"    Baseline: acc={baseline_acc:.4f}, low={baseline_acc_low:.4f}, mid={baseline_acc_mid:.4f}")

    # ===================================================================
    # Sweep both approaches
    # ===================================================================
    results_9a = {}
    results_9b = {}

    for lam in lambda_values:
        if lam == 1.0:
            # lambda=1 means 100% EM, 0% transfer = baseline
            results_9a[lam] = {"preds": baseline_preds}
            results_9b[lam] = {"preds": baseline_preds}
            continue

        # 9a: related language
        print(f"  9a: lambda={lam:.1f} (related language)...")
        w_9a = interpolate_weights_related(weights, pairs, lang_to_idx, lam)
        model.model.set_weight_sets(w_9a.tolist())
        del w_9a
        gc.collect()
        preds_9a = np.array(predict_all(texts, model))
        results_9a[lam] = {"preds": preds_9a}
        acc = (preds_9a == y_true).mean()
        acc_low = (preds_9a[low_mask] == y_true[low_mask]).mean() if low_mask.sum() > 0 else float('nan')
        print(f"    acc={acc:.4f}, low={acc_low:.4f}")

        # 9b: script average
        print(f"  9b: lambda={lam:.1f} (script average)...")
        w_9b = interpolate_weights_script_avg(weights, script_logprobs, langs, lang_to_idx, train_counts, lam)
        model.model.set_weight_sets(w_9b.tolist())
        del w_9b
        gc.collect()
        preds_9b = np.array(predict_all(texts, model))
        results_9b[lam] = {"preds": preds_9b}
        acc = (preds_9b == y_true).mean()
        acc_low = (preds_9b[low_mask] == y_true[low_mask]).mean() if low_mask.sum() > 0 else float('nan')
        print(f"    acc={acc:.4f}, low={acc_low:.4f}")

    del texts
    gc.collect()

    # ===================================================================
    # Generate tables
    # ===================================================================
    headers = ["Lambda", "9a Overall", "9a <500", "9a 500-5k", "9b Overall", "9b <500", "9b 500-5k"]
    rows = []
    for lam in lambda_values:
        p9a = results_9a[lam]["preds"]
        p9b = results_9b[lam]["preds"]
        rows.append([
            f"{lam:.1f}",
            f"{(p9a == y_true).mean():.3f}",
            f"{(p9a[low_mask] == y_true[low_mask]).mean():.3f}" if low_mask.sum() else "--",
            f"{(p9a[mid_mask] == y_true[mid_mask]).mean():.3f}" if mid_mask.sum() else "--",
            f"{(p9b == y_true).mean():.3f}",
            f"{(p9b[low_mask] == y_true[low_mask]).mean():.3f}" if low_mask.sum() else "--",
            f"{(p9b[mid_mask] == y_true[mid_mask]).mean():.3f}" if mid_mask.sum() else "--",
        ])

    md = to_markdown(rows, headers, caption="Transfer sweep: accuracy vs interpolation weight (lambda=1 is baseline)")

    # High-resource regression check
    headers_hr = ["Lambda", "9a High-res", "9b High-res"]
    rows_hr = []
    for lam in lambda_values:
        p9a = results_9a[lam]["preds"]
        p9b = results_9b[lam]["preds"]
        rows_hr.append([
            f"{lam:.1f}",
            f"{(p9a[high_mask] == y_true[high_mask]).mean():.3f}" if high_mask.sum() else "--",
            f"{(p9b[high_mask] == y_true[high_mask]).mean():.3f}" if high_mask.sum() else "--",
        ])
    md += "\n" + to_markdown(rows_hr, headers_hr, caption="Transfer sweep: high-resource accuracy (regression check)")

    with open(os.path.join(tables_dir, "transfer_sweep.md"), "w") as f:
        f.write(md)

    # ===================================================================
    # Figure
    # ===================================================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, (results, title) in zip(axes, [(results_9a, "9a: Related language"), (results_9b, "9b: Script average")]):
        accs_overall = [(results[l]["preds"] == y_true).mean() for l in lambda_values]
        ax.plot(lambda_values, accs_overall, "k-o", linewidth=2, label="Overall")

        if low_mask.sum():
            accs_low = [(results[l]["preds"][low_mask] == y_true[low_mask]).mean() for l in lambda_values]
            ax.plot(lambda_values, accs_low, "r-s", linewidth=1.5, label="<500 samples")
        if mid_mask.sum():
            accs_mid = [(results[l]["preds"][mid_mask] == y_true[mid_mask]).mean() for l in lambda_values]
            ax.plot(lambda_values, accs_mid, "b-^", linewidth=1.5, label="500-5k samples")

        ax.axhline(baseline_acc, color="gray", linestyle="--", alpha=0.5)
        ax.set_xlabel("Lambda (1=EM only, 0=full transfer)")
        ax.set_ylabel("Accuracy")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(figures_dir, "transfer_sweep.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\n  Outputs: {tables_dir}/transfer_sweep.md, {figures_dir}/transfer_sweep.png")
