"""Experiment 8: Discriminative optimization for confusion clusters.

8a: Variance-based token weighting (heuristic, fast)
    Three setups: upweight-only, multiplicative rescaling, sigmoid gate
8b: MMI discriminative fine-tuning (principled, gradient-based) — TODO
"""

from __future__ import annotations

import gc
import importlib
import json
import math
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
    CONFUSION_CLUSTERS,
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


# ---------------------------------------------------------------------------
# Data loading (same pattern as transfer_sweep.py)
# ---------------------------------------------------------------------------

def _load_model_data(model_path: str = UNILID_MODEL_PATH):
    """Load per-language weights, language list."""
    HEADER_FMT = "<8sIIIII4x"
    HEADER_SIZE = struct.calcsize(HEADER_FMT)

    with open(model_path, "rb") as f:
        header = f.read(HEADER_SIZE)
        _, _, num_langs, vocab_size, base_tok_len, langs_len = struct.unpack(HEADER_FMT, header)
        base_tok_json = f.read(base_tok_len)
        langs = json.loads(f.read(langs_len).decode("utf-8"))
        weights_offset = f.tell()

    from tokenizers import Tokenizer
    base_tok = Tokenizer.from_str(base_tok_json.decode("utf-8"))
    state = json.loads(base_tok.model.__getstate__().decode("utf-8"))
    base_scores = np.array([s for _, s in state["vocab"]], dtype=np.float64)

    weights = np.memmap(
        model_path, dtype=np.float32, mode="r",
        offset=weights_offset, shape=(num_langs, vocab_size),
    )
    lang_to_idx = {l: i for i, l in enumerate(langs)}

    print(f"  Loaded {num_langs} languages, vocab size {vocab_size}")
    return base_scores, weights, langs, lang_to_idx


def _load_unilid_model(model_path: str = UNILID_MODEL_PATH):
    """Load UnilidModel."""
    spec = importlib.util.spec_from_file_location(
        "model_io",
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "UNILID", "unilid", "model_io.py"),
    )
    model_io = importlib.util.module_from_spec(spec)
    sys.modules["unilid.model_io"] = model_io
    spec.loader.exec_module(model_io)
    return model_io.UnilidModel(model_path)


def _stream_sampled_texts(sample_size: int = DEFAULT_SAMPLE_SIZE) -> list[str]:
    """Stream test file and collect sampled texts."""
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


def predict_all(texts: list[str], model) -> list[str]:
    """Run raw-score prediction in batches."""
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
# Experiment 8a: Variance-based token weighting
# ---------------------------------------------------------------------------

def _compute_cluster_variance(weights: np.ndarray, cluster_indices: list[int]) -> np.ndarray:
    """Compute per-token variance of log-probs across cluster languages.

    Returns (vocab_size,) float64 array.
    """
    cluster_weights = np.array([weights[i] for i in cluster_indices], dtype=np.float64)
    return np.var(cluster_weights, axis=0)


def _renormalize_logprobs(w: np.ndarray) -> np.ndarray:
    """Renormalize a weight vector to valid log-probabilities: w - logsumexp(w)."""
    return w - scipy_logsumexp(w)


def apply_setup_a(weights: np.ndarray, cluster_indices: list[int],
                  var_t: np.ndarray, alpha: float) -> np.ndarray:
    """Setup A: Upweight only — boost discriminative tokens above median variance."""
    w = np.array(weights, dtype=np.float64)
    threshold = np.median(var_t[var_t > 0]) if (var_t > 0).any() else 0
    mask = var_t > threshold

    for idx in cluster_indices:
        w[idx][mask] += alpha * var_t[mask]
        w[idx] = _renormalize_logprobs(w[idx])

    return w.astype(np.float32)


def apply_setup_b(weights: np.ndarray, cluster_indices: list[int],
                  var_t: np.ndarray, alpha: float) -> np.ndarray:
    """Setup B: Additive rescaling in log-space with z-scored variance.

    Adds alpha * z_var to log-probs. Positive z_var (discriminative tokens) get
    boosted (less negative log-prob = higher probability), negative z_var
    (non-discriminative) get dampened (more negative = lower probability).
    """
    w = np.array(weights, dtype=np.float64)
    mean_var = var_t.mean()
    std_var = var_t.std()
    if std_var < 1e-10:
        return w.astype(np.float32)

    z_var = (var_t - mean_var) / std_var

    for idx in cluster_indices:
        w[idx] += alpha * z_var
        w[idx] = _renormalize_logprobs(w[idx])

    return w.astype(np.float32)


def apply_setup_c(weights: np.ndarray, base_scores: np.ndarray,
                  cluster_indices: list[int],
                  var_t: np.ndarray, beta: float) -> np.ndarray:
    """Setup C: Sigmoid gate — replace non-discriminative tokens with base."""
    w = np.array(weights, dtype=np.float64)
    threshold = np.median(var_t[var_t > 0]) if (var_t > 0).any() else 0

    # Sigmoid gate: σ(β * (var - threshold))
    gate = 1.0 / (1.0 + np.exp(-beta * (var_t - threshold)))

    for idx in cluster_indices:
        w[idx] = w[idx] * gate + base_scores * (1 - gate)
        w[idx] = _renormalize_logprobs(w[idx])

    return w.astype(np.float32)


def generate_heuristic_discriminative(
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    output_dir: str = "outputs",
    model_path: str = UNILID_MODEL_PATH,
):
    """Run Experiment 8a: variance-based token weighting for all clusters."""
    print("  Loading model data...")
    base_scores, weights, langs, lang_to_idx = _load_model_data(model_path)

    print("  Loading UniLID model for prediction...")
    model = _load_unilid_model(model_path)

    print("  Streaming sampled texts...")
    texts = _stream_sampled_texts(sample_size)

    print("  Loading sample for y_true...")
    data = load_sample(sample_size)
    y_true = np.array(data["y_true"])

    if len(texts) != len(y_true):
        raise RuntimeError(f"Text count {len(texts)} != y_true count {len(y_true)}")

    tables_dir = os.path.join(output_dir, "tables")
    figures_dir = os.path.join(output_dir, "figures")
    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    # Baseline
    print("  Running baseline...")
    baseline_preds = np.array(predict_all(texts, model))
    baseline_metrics = compute_metrics(y_true, baseline_preds)
    print(f"    Baseline: acc={baseline_metrics['accuracy']:.4f}")

    # Per-cluster masks for evaluation
    cluster_masks = {}
    for ckey, cinfo in CONFUSION_CLUSTERS.items():
        mask = np.isin(y_true, cinfo["languages"])
        if mask.sum() > 0:
            cluster_masks[ckey] = mask

    # Collect all cluster language indices
    all_cluster_indices = {}
    all_cluster_variances = {}
    for ckey, cinfo in CONFUSION_CLUSTERS.items():
        indices = [lang_to_idx[l] for l in cinfo["languages"] if l in lang_to_idx]
        all_cluster_indices[ckey] = indices
        all_cluster_variances[ckey] = _compute_cluster_variance(weights, indices)

    # ===================================================================
    # Setup A: Upweight only
    # ===================================================================
    alpha_values_ab = [0.0, 0.5, 1.0, 2.0, 5.0]
    beta_values_c = [1.0, 5.0, 10.0]

    def _apply_all_clusters(setup_fn, *args):
        """Apply a setup function to all clusters, starting from fresh original weights."""
        w = np.array(weights, dtype=np.float32)
        for ckey in CONFUSION_CLUSTERS:
            # Always apply from original weights to avoid cross-cluster contamination
            w_mod = setup_fn(weights, all_cluster_indices[ckey],
                             all_cluster_variances[ckey], *args)
            for idx in all_cluster_indices[ckey]:
                w[idx] = w_mod[idx]
        return w

    results_a = {}
    print("\n  === Setup A: Upweight only ===")
    for alpha in alpha_values_ab:
        label = f"A_alpha={alpha}"
        print(f"  {label}...")

        if alpha == 0.0:
            results_a[alpha] = {"preds": baseline_preds, "metrics": baseline_metrics}
            continue

        w = _apply_all_clusters(apply_setup_a, alpha)
        model.model.set_weight_sets(w.tolist())
        del w
        gc.collect()
        preds = np.array(predict_all(texts, model))
        metrics = compute_metrics(y_true, preds)
        results_a[alpha] = {"preds": preds, "metrics": metrics}
        print(f"    acc={metrics['accuracy']:.4f}")

    # ===================================================================
    # Setup B: Additive rescaling (z-scored variance)
    # ===================================================================
    results_b = {}
    print("\n  === Setup B: Additive rescaling ===")
    for alpha in alpha_values_ab:
        label = f"B_alpha={alpha}"
        print(f"  {label}...")

        if alpha == 0.0:
            results_b[alpha] = {"preds": baseline_preds, "metrics": baseline_metrics}
            continue

        w = _apply_all_clusters(apply_setup_b, alpha)
        model.model.set_weight_sets(w.tolist())
        del w
        gc.collect()
        preds = np.array(predict_all(texts, model))
        metrics = compute_metrics(y_true, preds)
        results_b[alpha] = {"preds": preds, "metrics": metrics}
        print(f"    acc={metrics['accuracy']:.4f}")

    # ===================================================================
    # Setup C: Sigmoid gate
    # ===================================================================
    results_c = {}
    print("\n  === Setup C: Sigmoid gate ===")

    def _apply_setup_c_wrapper(weights_in, cluster_indices, cluster_var, beta):
        return apply_setup_c(weights_in, base_scores, cluster_indices, cluster_var, beta)

    for beta in beta_values_c:
        label = f"C_beta={beta}"
        print(f"  {label}...")

        # Setup C has different signature (needs base_scores), use wrapper
        w = np.array(weights, dtype=np.float32)
        for ckey in CONFUSION_CLUSTERS:
            w_mod = apply_setup_c(weights, base_scores, all_cluster_indices[ckey],
                                  all_cluster_variances[ckey], beta)
            for idx in all_cluster_indices[ckey]:
                w[idx] = w_mod[idx]

        model.model.set_weight_sets(w.tolist())
        del w
        gc.collect()
        preds = np.array(predict_all(texts, model))
        metrics = compute_metrics(y_true, preds)
        results_c[beta] = {"preds": preds, "metrics": metrics}
        print(f"    acc={metrics['accuracy']:.4f}")

    del texts
    gc.collect()

    # ===================================================================
    # Generate tables
    # ===================================================================

    # Overall results table
    headers = ["Setup", "Param", "Overall Acc", "Overall Ma-F1"]
    rows = []
    rows.append(["Baseline", "--", f"{baseline_metrics['accuracy']:.3f}", f"{baseline_metrics['macro_f1']:.3f}"])
    for alpha in alpha_values_ab:
        if alpha == 0.0:
            continue
        m = results_a[alpha]["metrics"]
        rows.append([f"A (upweight)", f"α={alpha}", f"{m['accuracy']:.3f}", f"{m['macro_f1']:.3f}"])
    for alpha in alpha_values_ab:
        if alpha == 0.0:
            continue
        m = results_b[alpha]["metrics"]
        rows.append([f"B (rescale)", f"α={alpha}", f"{m['accuracy']:.3f}", f"{m['macro_f1']:.3f}"])
    for beta in beta_values_c:
        m = results_c[beta]["metrics"]
        rows.append([f"C (gate)", f"β={beta}", f"{m['accuracy']:.3f}", f"{m['macro_f1']:.3f}"])

    md = to_markdown(rows, headers, caption="Experiment 8a: Heuristic discriminative weighting")

    # Per-cluster accuracy table
    headers_cl = ["Setup", "Param"] + [cinfo["title"][:15] for cinfo in CONFUSION_CLUSTERS.values()]
    rows_cl = []

    def _cluster_acc(preds):
        accs = []
        for ckey in CONFUSION_CLUSTERS:
            mask = cluster_masks.get(ckey)
            if mask is not None and mask.sum() > 0:
                accs.append(f"{(preds[mask] == y_true[mask]).mean():.3f}")
            else:
                accs.append("--")
        return accs

    rows_cl.append(["Baseline", "--"] + _cluster_acc(baseline_preds))
    for alpha in alpha_values_ab:
        if alpha == 0.0:
            continue
        rows_cl.append([f"A", f"α={alpha}"] + _cluster_acc(results_a[alpha]["preds"]))
    for alpha in alpha_values_ab:
        if alpha == 0.0:
            continue
        rows_cl.append([f"B", f"α={alpha}"] + _cluster_acc(results_b[alpha]["preds"]))
    for beta in beta_values_c:
        rows_cl.append([f"C", f"β={beta}"] + _cluster_acc(results_c[beta]["preds"]))

    md += "\n" + to_markdown(rows_cl, headers_cl, caption="Experiment 8a: Per-cluster accuracy")

    # Detailed breakdown at best overall config
    all_configs = []
    for alpha in alpha_values_ab:
        if alpha > 0:
            all_configs.append(("A", alpha, results_a[alpha]))
            all_configs.append(("B", alpha, results_b[alpha]))
    for beta in beta_values_c:
        all_configs.append(("C", beta, results_c[beta]))

    best_setup, best_param, best_result = max(all_configs, key=lambda x: x[2]["metrics"]["accuracy"])
    best_preds = best_result["preds"]
    changed = baseline_preds != best_preds
    corrected = changed & (best_preds == y_true) & (baseline_preds != y_true)
    broken = changed & (best_preds != y_true) & (baseline_preds == y_true)

    print(f"\n  Best config: Setup {best_setup}, param={best_param}")
    print(f"    Changed: {changed.sum():,}, Corrected: {corrected.sum():,}, Broken: {broken.sum():,}, Net: {corrected.sum() - broken.sum():+,}")

    with open(os.path.join(tables_dir, "discriminative_heuristic.md"), "w") as f:
        f.write(md)

    print(f"\n  Outputs: {tables_dir}/discriminative_heuristic.md")
