"""Tokenization length bias analysis for UniLID.

Tests whether misclassifications are systematically biased toward languages
whose tokenizer produces fewer tokens (fewer negative log-prob terms),
and whether length-normalizing scores would correct those errors.

This module streams misclassified texts directly from disk to avoid storing
all 45.6M texts in memory.
"""

from __future__ import annotations

import json
import os
import struct
from collections import defaultdict

import numpy as np
from tokenizers import Tokenizer
from scipy import stats

from analysis.config import (
    CONFUSION_CLUSTERS,
    DATA_DIR,
    LENGTH_BINS,
    LENGTH_LABELS,
    PRED_FILES,
    TOTAL_LINES,
)
from analysis.format_utils import to_markdown, to_latex


UNILID_MODEL_PATH = "/capstor/scratch/cscs/cmeister747/unilid_analysis/glotlidc.unilid"
VERY_NEG = -1e30


# ---------------------------------------------------------------------------
# Tokenizer loading
# ---------------------------------------------------------------------------

def _load_model_components(model_path: str = UNILID_MODEL_PATH):
    """Load base tokenizer, weights (memmap), and language list from .unilid file."""
    HEADER_FMT = "<8sIIIII4x"
    HEADER_SIZE = struct.calcsize(HEADER_FMT)

    with open(model_path, "rb") as f:
        header = f.read(HEADER_SIZE)
        magic, version, num_langs, vocab_size, base_tok_len, langs_len = struct.unpack(
            HEADER_FMT, header
        )
        base_tok_bytes = f.read(base_tok_len)
        langs_bytes = f.read(langs_len)
        weights_offset = f.tell()

    base_tok = Tokenizer.from_str(base_tok_bytes.decode("utf-8"))
    langs = json.loads(langs_bytes.decode("utf-8"))
    weights = np.memmap(
        model_path,
        dtype=np.float32,
        mode="r",
        offset=weights_offset,
        shape=(num_langs, vocab_size),
    )
    return base_tok, weights, langs


def _build_lang_tokenizer(base_tok: Tokenizer, weights: np.ndarray, lang_idx: int, vocab_list):
    """Reconstruct a per-language tokenizer by injecting language-specific weights."""
    base_state = json.loads(base_tok.model.__getstate__().decode("utf-8"))
    base_state.pop("type", None)

    new_vocab = [(token, float(weights[lang_idx, tok_id])) for token, tok_id in vocab_list]
    state = base_state.copy()
    state["vocab"] = new_vocab

    lang_tok = Tokenizer.from_str(base_tok.to_str())
    lang_tok.model = lang_tok.model.__class__(**state)
    return lang_tok


def _extract_vocab_scores(tok: Tokenizer) -> dict[str, float]:
    """Extract {token_string: log_probability} from a Unigram tokenizer."""
    state = json.loads(tok.model.__getstate__().decode("utf-8"))
    return {token: score for token, score in state["vocab"]}


def build_tokenizer_cache(needed_langs: set[str], model_path: str = UNILID_MODEL_PATH):
    """Build tokenizers only for the languages we need."""
    base_tok, weights, langs = _load_model_components(model_path)
    lang_to_idx = {lang: i for i, lang in enumerate(langs)}
    ref_vocab = base_tok.get_vocab()
    vocab_list = sorted(ref_vocab.items(), key=lambda x: x[1])

    cache = {}
    for lang in needed_langs:
        idx = lang_to_idx.get(lang)
        if idx is None:
            continue
        cache[lang] = _build_lang_tokenizer(base_tok, weights, idx, vocab_list)

    print(f"  Built {len(cache)} per-language tokenizers")
    return cache


# ---------------------------------------------------------------------------
# Streaming misclassified text extraction
# ---------------------------------------------------------------------------

def stream_misclassified_texts():
    """Stream through test file and UniLID predictions, collecting only misclassified samples.

    Returns list of dicts with keys: text, true_label, pred_label, text_length
    """
    test_path = os.path.join(DATA_DIR, "glotlid_correct_test.txt")
    pred_path = PRED_FILES["UniLID"]

    misclassified = []
    n_total = 0

    with open(test_path, "r") as f_test, open(pred_path, "r") as f_pred:
        for test_line, pred_line in zip(f_test, f_pred):
            pred_label = pred_line.strip()
            parts = test_line.split(" ", 1)
            true_label = parts[0].replace("__label__", "")

            if true_label != pred_label:
                text = parts[1].rstrip("\n") if len(parts) > 1 else ""
                misclassified.append({
                    "text": text,
                    "true_label": true_label,
                    "pred_label": pred_label,
                    "text_length": len(text),
                })

            n_total += 1
            if n_total % 10_000_000 == 0:
                print(f"    Scanned {n_total:,} / {TOTAL_LINES:,} "
                      f"({len(misclassified):,} misclassified)", flush=True)

    print(f"    Scanned {n_total:,} total, {len(misclassified):,} misclassified")
    return misclassified


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------

def _compute_score(tokens: list[str], vocab_scores: dict[str, float]) -> float:
    """Sum log-probabilities for a token sequence."""
    return sum(vocab_scores.get(t, VERY_NEG) for t in tokens)


def compute_token_deltas_and_scores(misclassified: list[dict], model_path: str = UNILID_MODEL_PATH):
    """For misclassified samples, compute token count deltas and scores.

    For each sample computes:
    - Token count delta (pred - true)
    - Raw scores for true and predicted language
    - Length-normalized scores
    - Whether normalization flips the ranking
    """
    n_wrong = len(misclassified)
    print(f"  {n_wrong:,} misclassified samples to tokenize and score")

    # Determine which languages we need tokenizers for
    needed_langs = set()
    for item in misclassified:
        needed_langs.add(item["true_label"])
        needed_langs.add(item["pred_label"])
    tok_cache = build_tokenizer_cache(needed_langs, model_path)

    # Group by (true_lang, pred_lang) for batched encoding
    pair_groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for i, item in enumerate(misclassified):
        pair_groups[(item["true_label"], item["pred_label"])].append(i)

    deltas = np.zeros(n_wrong, dtype=int)
    n_tokens_true_arr = np.zeros(n_wrong, dtype=int)
    n_tokens_pred_arr = np.zeros(n_wrong, dtype=int)
    score_true_arr = np.full(n_wrong, np.nan, dtype=np.float64)
    score_pred_arr = np.full(n_wrong, np.nan, dtype=np.float64)
    norm_score_true_arr = np.full(n_wrong, np.nan, dtype=np.float64)
    norm_score_pred_arr = np.full(n_wrong, np.nan, dtype=np.float64)
    flipped_arr = np.zeros(n_wrong, dtype=bool)

    text_lengths = np.array([item["text_length"] for item in misclassified])
    y_true = np.array([item["true_label"] for item in misclassified])
    y_pred = np.array([item["pred_label"] for item in misclassified])

    processed = 0
    for (true_lang, pred_lang), indices in pair_groups.items():
        true_tok = tok_cache.get(true_lang)
        pred_tok = tok_cache.get(pred_lang)
        if true_tok is None or pred_tok is None:
            processed += len(indices)
            continue

        # Extract vocab scores lazily — only 2 dicts alive at a time
        true_vocab = _extract_vocab_scores(true_tok)
        pred_vocab = _extract_vocab_scores(pred_tok)

        batch_texts = [misclassified[i]["text"] for i in indices]

        true_encodings = true_tok.encode_batch(batch_texts)
        pred_encodings = pred_tok.encode_batch(batch_texts)

        for j, (te, pe) in enumerate(zip(true_encodings, pred_encodings)):
            idx = indices[j]
            n_true = len(te.tokens)
            n_pred = len(pe.tokens)
            n_tokens_true_arr[idx] = n_true
            n_tokens_pred_arr[idx] = n_pred
            deltas[idx] = n_pred - n_true

            s_true = _compute_score(te.tokens, true_vocab)
            s_pred = _compute_score(pe.tokens, pred_vocab)
            score_true_arr[idx] = s_true
            score_pred_arr[idx] = s_pred

            ns_true = s_true / n_true if n_true > 0 else VERY_NEG
            ns_pred = s_pred / n_pred if n_pred > 0 else VERY_NEG
            norm_score_true_arr[idx] = ns_true
            norm_score_pred_arr[idx] = ns_pred

            flipped_arr[idx] = ns_true > ns_pred

        # Free the vocab dicts for this pair
        del true_vocab, pred_vocab

        processed += len(indices)
        if processed % 50_000 < len(indices):
            print(f"    Processed {processed:,} / {n_wrong:,}", flush=True)

    print(f"    Processed {processed:,} / {n_wrong:,} (done)")

    return {
        "deltas": deltas,
        "n_tokens_true": n_tokens_true_arr,
        "n_tokens_pred": n_tokens_pred_arr,
        "score_true": score_true_arr,
        "score_pred": score_pred_arr,
        "norm_score_true": norm_score_true_arr,
        "norm_score_pred": norm_score_pred_arr,
        "flipped": flipped_arr,
        "text_lengths": text_lengths,
        "y_true": y_true,
        "y_pred": y_pred,
        "n_total_wrong": n_wrong,
        "n_processed": processed,
    }


# ---------------------------------------------------------------------------
# Tables and figures
# ---------------------------------------------------------------------------

def _bias_summary_row(deltas):
    """Compute summary stats for a set of deltas."""
    n = len(deltas)
    if n == 0:
        return [0, 0, 0, 0, 0, 0]
    mean_d = float(np.mean(deltas))
    median_d = float(np.median(deltas))
    pct_fewer = float((deltas < 0).sum() / n * 100)
    pct_same = float((deltas == 0).sum() / n * 100)
    pct_more = float((deltas > 0).sum() / n * 100)
    return [n, mean_d, median_d, pct_fewer, pct_same, pct_more]


def _counterfactual_row(flipped, label="All"):
    """Compute counterfactual summary: how many misclassifications would normalization fix."""
    n = len(flipped)
    if n == 0:
        return [label, 0, 0, "0.0"]
    n_flipped = int(flipped.sum())
    pct = n_flipped / n * 100
    return [label, f"{n:,}", f"{n_flipped:,}", f"{pct:.1f}"]


def generate_length_bias_analysis(output_dir: str = "outputs", model_path: str = UNILID_MODEL_PATH):
    """Run the full length bias + normalization counterfactual analysis."""
    print("  Streaming misclassified texts from disk...")
    misclassified = stream_misclassified_texts()

    results = compute_token_deltas_and_scores(misclassified, model_path)

    # Free the raw text data now that tokenization is done
    del misclassified

    deltas = results["deltas"]
    text_lens = results["text_lengths"]
    y_true = results["y_true"]
    y_pred = results["y_pred"]
    flipped = results["flipped"]
    n = results["n_processed"]

    tables_dir = os.path.join(output_dir, "tables")
    figures_dir = os.path.join(output_dir, "figures")
    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    # ===================================================================
    # Part 1: Token delta tables (existing)
    # ===================================================================
    headers = ["Category", "N", "Mean delta", "Median delta", "% fewer", "% same", "% more"]
    overall_row = ["All misclassified"] + [
        f"{v:.2f}" if isinstance(v, float) else f"{v:,}" for v in _bias_summary_row(deltas)
    ]
    rows = [overall_row]

    for i in range(len(LENGTH_LABELS)):
        lo, hi = LENGTH_BINS[i], LENGTH_BINS[i + 1]
        mask = (text_lens >= lo) & (text_lens < hi)
        if mask.sum() == 0:
            continue
        row = [LENGTH_LABELS[i]] + [
            f"{v:.2f}" if isinstance(v, float) else f"{v:,}" for v in _bias_summary_row(deltas[mask])
        ]
        rows.append(row)

    md_delta = to_markdown(rows, headers, caption="Token count delta (pred - true) for UniLID misclassifications")

    cluster_rows = []
    for cluster_key, cluster_info in CONFUSION_CLUSTERS.items():
        cluster_langs = set(cluster_info["languages"])
        mask = np.array([t in cluster_langs for t in y_true])
        if mask.sum() == 0:
            continue
        row = [cluster_info["title"]] + [
            f"{v:.2f}" if isinstance(v, float) else f"{v:,}" for v in _bias_summary_row(deltas[mask])
        ]
        cluster_rows.append(row)

    md_clusters_delta = to_markdown(cluster_rows, headers, caption="Token delta by confusion cluster")

    # Statistical test
    if n > 1:
        t_stat, t_pval = stats.ttest_1samp(deltas, 0)
        nonzero = deltas[deltas != 0]
        w_stat, w_pval = stats.wilcoxon(nonzero) if len(nonzero) > 10 else (float("nan"), float("nan"))
        std = np.std(deltas, ddof=1)
        cohens_d = float(np.mean(deltas) / std) if std > 0 else 0.0
    else:
        t_stat = t_pval = w_stat = w_pval = cohens_d = float("nan")

    test_headers = ["Test", "Statistic", "p-value"]
    test_rows = [
        ["One-sample t-test (H0: mean delta = 0)", f"{t_stat:.4f}", f"{t_pval:.2e}"],
        ["Wilcoxon signed-rank (excl. zeros)", f"{w_stat:.4f}" if not np.isnan(w_stat) else "N/A", f"{w_pval:.2e}" if not np.isnan(w_pval) else "N/A"],
        ["Cohen's d", f"{cohens_d:.4f}", ""],
    ]
    md_tests = to_markdown(test_rows, test_headers, caption="Statistical tests for systematic bias")

    # ===================================================================
    # Part 2: Length normalization counterfactual
    # ===================================================================
    cf_headers = ["Category", "N", "N corrected", "% correctable"]

    cf_rows = [_counterfactual_row(flipped, "All misclassified")]

    # By text length bin
    for i in range(len(LENGTH_LABELS)):
        lo, hi = LENGTH_BINS[i], LENGTH_BINS[i + 1]
        mask = (text_lens >= lo) & (text_lens < hi)
        if mask.sum() == 0:
            continue
        cf_rows.append(_counterfactual_row(flipped[mask], LENGTH_LABELS[i]))

    md_counterfactual = to_markdown(cf_rows, cf_headers,
                                     caption="Length normalization counterfactual: misclassifications corrected by normalizing scores by token count")

    # By confusion cluster
    cf_cluster_rows = []
    for cluster_key, cluster_info in CONFUSION_CLUSTERS.items():
        cluster_langs = set(cluster_info["languages"])
        mask = np.array([t in cluster_langs for t in y_true])
        if mask.sum() == 0:
            continue
        cf_cluster_rows.append(_counterfactual_row(flipped[mask], cluster_info["title"]))

    md_cf_clusters = to_markdown(cf_cluster_rows, cf_headers,
                                  caption="Length normalization counterfactual by confusion cluster")

    # By token delta bin
    delta_bins = [(-np.inf, -2), (-2, -1), (-1, 0), (0, 0), (0, 1), (1, 2), (2, np.inf)]
    delta_labels = ["<-2", "-2 to -1", "-1", "0", "+1", "+1 to +2", ">+2"]
    cf_delta_rows = []
    for (lo, hi), label in zip(delta_bins, delta_labels):
        if lo == hi == 0:
            mask = deltas == 0
        elif lo == 0 and hi == 1:
            mask = (deltas > 0) & (deltas <= 1)
        else:
            mask = (deltas > lo) & (deltas <= hi) if lo != -np.inf else (deltas <= hi)
            if hi == np.inf:
                mask = deltas > lo
        if mask.sum() == 0:
            continue
        cf_delta_rows.append(_counterfactual_row(flipped[mask], label))

    md_cf_delta = to_markdown(cf_delta_rows, cf_headers,
                               caption="Length normalization counterfactual by token delta")

    # ===================================================================
    # Figures
    # ===================================================================
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Histogram of deltas
    fig, ax = plt.subplots(figsize=(8, 4))
    clip_range = np.percentile(deltas, [1, 99])
    clipped = deltas[(deltas >= clip_range[0]) & (deltas <= clip_range[1])]
    ax.hist(clipped, bins=50, edgecolor="black", alpha=0.7)
    ax.axvline(0, color="red", linestyle="--", linewidth=1.5, label="zero")
    ax.axvline(np.mean(deltas), color="blue", linestyle="-", linewidth=1.5, label=f"mean={np.mean(deltas):.2f}")
    ax.set_xlabel("Token delta (pred - true)")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of token count delta for UniLID misclassifications")
    ax.legend()
    fig.tight_layout()
    hist_path = os.path.join(figures_dir, "length_bias_histogram.png")
    fig.savefig(hist_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ===================================================================
    # Write outputs
    # ===================================================================
    full_md = "\n".join([
        md_delta, md_clusters_delta, md_tests,
        md_counterfactual, md_cf_clusters, md_cf_delta,
    ])
    with open(os.path.join(tables_dir, "length_bias.md"), "w") as f:
        f.write(full_md)

    # LaTeX
    latex_parts = []
    latex_parts.append(to_latex(
        [[r[0]] + [r[i] for i in range(1, len(r))] for r in rows],
        headers,
        caption="Token count delta (predicted $-$ true) for UniLID misclassifications.",
        label="tab:length-bias-overall",
    ))
    if cluster_rows:
        latex_parts.append(to_latex(
            [[r[0]] + [r[i] for i in range(1, len(r))] for r in cluster_rows],
            headers,
            caption="Token delta by confusion cluster.",
            label="tab:length-bias-clusters",
        ))
    latex_parts.append(to_latex(
        cf_rows, cf_headers,
        caption="Misclassifications correctable by length-normalizing scores.",
        label="tab:counterfactual-overall",
    ))
    if cf_cluster_rows:
        latex_parts.append(to_latex(
            cf_cluster_rows, cf_headers,
            caption="Counterfactual correctability by confusion cluster.",
            label="tab:counterfactual-clusters",
        ))
    latex_parts.append(to_latex(
        cf_delta_rows, cf_headers,
        caption="Counterfactual correctability by token delta.",
        label="tab:counterfactual-delta",
    ))

    with open(os.path.join(tables_dir, "length_bias.tex"), "w") as f:
        f.write("\n\n".join(latex_parts))

    # ===================================================================
    # Summary to stdout
    # ===================================================================
    n_flipped = int(flipped.sum())
    print(f"  --- Token delta ---")
    print(f"  Mean delta: {np.mean(deltas):.3f}, Median: {np.median(deltas):.1f}")
    print(f"  Pred uses fewer tokens: {(deltas < 0).sum() / n * 100:.1f}%")
    print(f"  Pred uses more tokens:  {(deltas > 0).sum() / n * 100:.1f}%")
    print(f"  t-test p={t_pval:.2e}, Cohen's d={cohens_d:.4f}")
    print(f"  --- Length normalization counterfactual ---")
    print(f"  Correctable: {n_flipped:,} / {n:,} ({n_flipped / n * 100:.1f}%)")
    print(f"  Outputs: {tables_dir}/length_bias.{{md,tex}}, {hist_path}")

    return {
        "deltas": deltas,
        "flipped": flipped,
        "mean_delta": float(np.mean(deltas)),
        "median_delta": float(np.median(deltas)),
        "cohens_d": cohens_d,
        "t_pval": float(t_pval),
        "n_correctable": n_flipped,
        "pct_correctable": n_flipped / n * 100,
        "n_processed": n,
    }
