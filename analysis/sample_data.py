"""Uniform sampling of the GlotLID test set with all model predictions."""

from __future__ import annotations

import json
import os
import pickle
import random

from analysis.config import (
    CACHE_DIR,
    DATA_DIR,
    DEFAULT_SAMPLE_SIZE,
    PRED_FILES,
    SAMPLE_SEED,
    TEST_FILE,
    TOTAL_LINES,
    TRAIN_COUNTS_FILE,
)


def sample_data(
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    output_path: str | None = None,
) -> dict:
    """Read test data and all prediction files in a single pass, sampling uniformly.

    Returns the sample dict and also saves it as a pickle.
    """
    if output_path is None:
        os.makedirs(CACHE_DIR, exist_ok=True)
        output_path = os.path.join(CACHE_DIR, f"sample_{sample_size // 1000}k_all.pkl")

    random.seed(SAMPLE_SEED)
    sample_indices = set(sorted(random.sample(range(TOTAL_LINES), sample_size)))

    with open(TRAIN_COUNTS_FILE) as f:
        train_counts = json.load(f)

    y_true_labels: list[str] = []
    preds: dict[str, list[str]] = {name: [] for name in PRED_FILES}
    text_lengths: list[int] = []

    handles = {name: open(path) for name, path in PRED_FILES.items()}
    test_handle = open(TEST_FILE)

    try:
        for i in range(TOTAL_LINES):
            test_line = test_handle.readline()
            pred_lines = {name: h.readline().strip() for name, h in handles.items()}

            if i in sample_indices:
                parts = test_line.split(" ", 1)
                label = parts[0].replace("__label__", "")
                text = parts[1].rstrip("\n") if len(parts) > 1 else ""

                y_true_labels.append(label)
                text_lengths.append(len(text))
                for name, pred_line in pred_lines.items():
                    preds[name].append(pred_line)

            if i % 10_000_000 == 0:
                print(f"  {i:,} / {TOTAL_LINES:,}", flush=True)
    finally:
        test_handle.close()
        for h in handles.values():
            h.close()

    result = {
        "y_true": y_true_labels,
        "text_lengths": text_lengths,
        "train_counts": train_counts,
        "sample_size": sample_size,
        "seed": SAMPLE_SEED,
    }
    for name in PRED_FILES:
        result[f"pred_{name}"] = preds[name]

    with open(output_path, "wb") as f:
        pickle.dump(result, f)

    print(f"Saved {len(y_true_labels):,} samples to {output_path}")
    return result


def load_sample(sample_size: int = DEFAULT_SAMPLE_SIZE) -> dict:
    """Load a previously saved sample, or create it if missing."""
    path = os.path.join(CACHE_DIR, f"sample_{sample_size // 1000}k_all.pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    print(f"Sample not found at {path}, generating...")
    return sample_data(sample_size=sample_size, output_path=path)
