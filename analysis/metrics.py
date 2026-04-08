"""Shared metrics computation for UniLID analysis."""

from __future__ import annotations

import numpy as np
from collections import Counter


def _encode_labels(y_true: np.ndarray, y_pred: np.ndarray):
    """Map string labels to integer indices. Returns (true_idx, pred_idx, labels)."""
    all_labels = sorted(set(y_true) | set(y_pred))
    label_to_idx = {l: i for i, l in enumerate(all_labels)}
    true_idx = np.array([label_to_idx[l] for l in y_true], dtype=np.int32)
    pred_idx = np.array([label_to_idx[l] for l in y_pred], dtype=np.int32)
    return true_idx, pred_idx, all_labels


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute accuracy, micro F1, and macro F1.

    Uses integer encoding + Counter for O(n) performance instead of
    O(n * n_labels) per-label array scans.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)

    accuracy = float((y_true == y_pred).mean())

    # Count TP, support (true count), and predicted count per class
    # using Counter on the paired arrays
    true_counts = Counter(y_true.tolist())
    pred_counts = Counter(y_pred.tolist())

    # TP per class: count where true == pred, grouped by label
    match_mask = y_true == y_pred
    tp_counts = Counter(y_true[match_mask].tolist())

    # Macro F1 averages only over true labels (matching sklearn convention).
    # Micro F1 uses the union so spurious predictions are still penalized.
    true_labels = set(true_counts)
    all_labels = true_labels | set(pred_counts)

    micro_tp = 0
    micro_fp = 0
    micro_fn = 0
    f1s = []
    fprs = []

    for lab in all_labels:
        tp = tp_counts.get(lab, 0)
        support = true_counts.get(lab, 0)
        predicted = pred_counts.get(lab, 0)
        fp = predicted - tp
        fn = support - tp
        tn = n - tp - fp - fn

        micro_tp += tp
        micro_fp += fp
        micro_fn += fn

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        if lab in true_labels:
            f1s.append(f1)
            fprs.append(fpr)

    micro_prec = micro_tp / (micro_tp + micro_fp) if (micro_tp + micro_fp) > 0 else 0.0
    micro_rec = micro_tp / (micro_tp + micro_fn) if (micro_tp + micro_fn) > 0 else 0.0
    micro_f1 = (
        2 * micro_prec * micro_rec / (micro_prec + micro_rec)
        if (micro_prec + micro_rec) > 0
        else 0.0
    )
    macro_f1 = float(np.mean(f1s)) if f1s else 0.0
    macro_fpr = float(np.mean(fprs)) if fprs else 0.0

    return {
        "accuracy": accuracy,
        "micro_f1": float(micro_f1),
        "macro_f1": float(macro_f1),
        "macro_fpr": macro_fpr,
    }


def compute_per_class_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute per-class precision, recall, F1, and support."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    true_counts = Counter(y_true.tolist())
    pred_counts = Counter(y_pred.tolist())
    match_mask = y_true == y_pred
    tp_counts = Counter(y_true[match_mask].tolist())

    all_labels = sorted(set(true_counts) | set(pred_counts))
    results = {}

    for lab in all_labels:
        tp = tp_counts.get(lab, 0)
        support = true_counts.get(lab, 0)
        predicted = pred_counts.get(lab, 0)
        fp = predicted - tp
        fn = support - tp

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

        results[lab] = {"precision": prec, "recall": rec, "f1": f1, "support": support}

    return results
