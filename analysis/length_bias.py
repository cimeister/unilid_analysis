"""Tokenization length bias analysis for UniLID.

Tests whether misclassifications are systematically biased toward languages
whose tokenizer produces fewer tokens (fewer negative log-prob terms),
and whether length-normalizing scores would correct those errors.

This module streams misclassified texts directly from disk to avoid storing
all 45.6M texts in memory.

Which model, and over which lines (2026-08-25). The published tab:lenbias-delta
was computed over all 45,627,279 test lines from the released model's recorded
prediction file. A corrected model has no prediction for the 250,000 validation
lines its full-pool run excludes, so the table needed a basis; the author's PD-4
ruling (2026-08-24) fixes it as the GOLDEN SUBSET, the 250,000-line test half of
the seed-42 500,000-line draw, the same subset tab:lenbias-norm was rebuilt on
(analysis/lenbias_norm_table.py). The command line below resolves the model
through analysis.model_context, exactly as the rest of the chain does:

  python -m analysis.length_bias                       # unchanged: released
                                                       # model, all 45.6M lines
  python -m analysis.length_bias --subset golden       # released model, subset
  python -m analysis.length_bias --subset golden \\
      --model CORRECTED.unilid --scratch-dir SCRATCH_CORRECTED \\
      --out-dir outputs_corrected_round

With no flags nothing changes: the released model, its recorded prediction file,
all 45.6M lines, and outputs/tables/length_bias.{md,tex} byte for byte as before.
A subset run writes to suffixed artifact names (length_bias_golden.*) so it can
never overwrite the full-pool artifacts that back the published table.
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys
from collections import defaultdict

import numpy as np
from tokenizers import Tokenizer
from scipy import stats

from analysis.config import (
    CONFUSION_CLUSTERS,
    DATA_DIR,
    DEFAULT_SAMPLE_SIZE,
    LENGTH_BINS,
    LENGTH_LABELS,
    PRED_FILES,
    TOTAL_LINES,
)
from analysis.format_utils import to_markdown, to_latex
from analysis.model_context import (DEFAULT_OUT_ROOT, add_arguments, resolve,
                                    resolve_out_root)


UNILID_MODEL_PATH = "/capstor/scratch/cscs/cmeister747/unilid_analysis/glotlidc.unilid"
VERY_NEG = -1e30

# PD-4 (author ruling 2026-08-24, "Do the obvious match"): the basis for a
# corrected tab:lenbias-delta is the same golden subset tab:lenbias-norm was
# rebuilt on. Selected here exactly as analysis/lenbias_norm_table.py selects it:
# odd positions of the seed-42 DEFAULT_SAMPLE_SIZE draw are the test half, the
# same split analysis/full_test_eval.py and both release gates use.
GOLDEN_SUBSET = "golden"
SUBSET_SUFFIX = {None: "", GOLDEN_SUBSET: "_golden"}


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
    """Build tokenizers only for the languages we need.

    A language that this model does not carry aborts the run naming it. It used
    to be skipped silently, and compute_token_deltas_and_scores then skipped every
    misclassification touching it while still counting those rows as processed and
    leaving their delta at 0, which reads in the tables as "predicted and true
    segmentations agree" -- the substitution the project's rules forbid, invisible
    in every output.
    """
    base_tok, weights, langs = _load_model_components(model_path)
    lang_to_idx = {lang: i for i, lang in enumerate(langs)}
    missing = sorted(l for l in needed_langs if l not in lang_to_idx)
    if missing:
        raise RuntimeError(
            f"{len(missing)} label(s) in the misclassified set are absent from "
            f"{model_path}'s {len(langs)}-language list: {missing[:10]}"
            f"{'...' if len(missing) > 10 else ''}. Their token deltas cannot be "
            "computed under these weights, and scoring the rest would report a "
            "table over a silently smaller line set than its N column claims.")
    ref_vocab = base_tok.get_vocab()
    vocab_list = sorted(ref_vocab.items(), key=lambda x: x[1])

    cache = {}
    for lang in needed_langs:
        cache[lang] = _build_lang_tokenizer(base_tok, weights, lang_to_idx[lang],
                                            vocab_list)

    print(f"  Built {len(cache)} per-language tokenizers")
    return cache


# ---------------------------------------------------------------------------
# Streaming misclassified text extraction
# ---------------------------------------------------------------------------

def stream_misclassified_texts(pred_path: str = None, line_indices=None,
                               pred_labels=None, true_labels=None):
    """Stream through test file and UniLID predictions, collecting only misclassified samples.

    Returns list of dicts with keys: text, true_label, pred_label, text_length

    ``line_indices``: absolute test-file line numbers to restrict the scan to
    (strictly increasing, no duplicates). Every one of them must be found, or the
    run aborts: a subset table whose N silently counts fewer lines than the subset
    holds answers a different question than the one it is labelled with.

    ``pred_labels``: predicted labels aligned positionally to ``line_indices``,
    for a model whose predictions live in a per-line code array rather than a
    recorded text file. Mutually exclusive with ``pred_path``.

    ``true_labels``: the labels the caller believes sit on ``line_indices``, also
    aligned positionally. Every one is checked against the label actually parsed
    from the test file, and a single disagreement aborts. This is the alignment
    gate: predictions read out of a per-line memmap by line index are silently
    plausible when they are off by one, and the resulting table looks like an
    ordinary analysis of a slightly worse model. It is the same assertion
    analysis/full_test_eval.py applies to its own memmaps.

    With all four None this is the original whole-file scan against
    ``PRED_FILES["UniLID"]``, unchanged.
    """
    if pred_labels is not None:
        if line_indices is None:
            raise RuntimeError("pred_labels requires line_indices: without it "
                               "there is nothing to align the labels to")
        if pred_path is not None:
            raise RuntimeError("pred_labels and pred_path are two different "
                               "prediction sources; pass exactly one")
        if len(pred_labels) != len(line_indices):
            raise RuntimeError(f"pred_labels has {len(pred_labels):,} entries "
                               f"against {len(line_indices):,} line_indices")
    if true_labels is not None:
        if line_indices is None:
            raise RuntimeError("true_labels requires line_indices")
        if len(true_labels) != len(line_indices):
            raise RuntimeError(f"true_labels has {len(true_labels):,} entries "
                               f"against {len(line_indices):,} line_indices")

    wanted = None
    pos_of = None
    if line_indices is not None:
        idx = np.asarray(line_indices)
        if not np.issubdtype(idx.dtype, np.integer):
            raise RuntimeError(f"line_indices has dtype {idx.dtype}; it must be an "
                               "integer array (a float array would be truncated "
                               "silently into a different set of lines)")
        idx = idx.astype(np.int64)
        if idx.ndim != 1 or idx.size == 0:
            raise RuntimeError(f"line_indices must be a non-empty 1-D array, got "
                               f"shape {idx.shape}")
        if not np.all(np.diff(idx) > 0):
            raise RuntimeError("line_indices must be strictly increasing and free "
                               "of duplicates")
        if idx[0] < 0 or idx[-1] >= TOTAL_LINES:
            raise RuntimeError(f"line_indices fall outside [0, {TOTAL_LINES:,})")
        wanted = set(idx.tolist())
        pos_of = {int(v): k for k, v in enumerate(idx.tolist())}

    test_path = os.path.join(DATA_DIR, "glotlid_correct_test.txt")
    use_pred_file = pred_labels is None
    if use_pred_file:
        pred_path = pred_path or PRED_FILES["UniLID"]

    misclassified = []
    n_total = 0
    n_seen = 0

    with open(test_path, "r") as f_test:
        f_pred = open(pred_path, "r") if use_pred_file else None
        try:
            for i, test_line in enumerate(f_test):
                if use_pred_file:
                    pred_line = f_pred.readline()
                    if pred_line == "":
                        raise RuntimeError(
                            f"{pred_path} ended at line {i:,} while {test_path} "
                            "still has lines; the two files do not describe the "
                            "same test set")
                n_total += 1
                if n_total % 10_000_000 == 0:
                    print(f"    Scanned {n_total:,} / {TOTAL_LINES:,} "
                          f"({len(misclassified):,} misclassified)", flush=True)
                if wanted is not None and i not in wanted:
                    continue
                n_seen += 1

                k = pos_of[i] if pos_of is not None else None
                pred_label = (pred_line.strip() if use_pred_file
                              else str(pred_labels[k]))
                parts = test_line.split(" ", 1)
                true_label = parts[0].replace("__label__", "")

                if true_labels is not None and str(true_labels[k]) != true_label:
                    raise RuntimeError(
                        f"alignment gate failed at test-file line {i:,}: the "
                        f"caller's arrays put {str(true_labels[k])!r} there, the "
                        f"file has {true_label!r}. The per-line arrays this run "
                        "reads its predictions from are not aligned to the test "
                        "file, so every prediction below would belong to a "
                        "different line than the text scored against it.")

                if true_label != pred_label:
                    text = parts[1].rstrip("\n") if len(parts) > 1 else ""
                    misclassified.append({
                        "text": text,
                        "true_label": true_label,
                        "pred_label": pred_label,
                        "text_length": len(text),
                    })
            if use_pred_file and f_pred.readline() != "":
                raise RuntimeError(
                    f"{pred_path} has more lines than {test_path}; the two files "
                    "do not describe the same test set")
        finally:
            if f_pred is not None:
                f_pred.close()

    if wanted is not None and n_seen != len(wanted):
        raise RuntimeError(f"only {n_seen:,} of the {len(wanted):,} requested "
                           f"line_indices were found in {test_path} "
                           f"({n_total:,} lines read)")
    print(f"    Scanned {n_total:,} total, {len(misclassified):,} misclassified"
          + ("" if wanted is None
             else f" out of {n_seen:,} lines in the requested subset"))
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
            # Unreachable: build_tokenizer_cache is given exactly this label set
            # and aborts on any language it cannot build. It used to `continue`
            # here, which left those rows' delta at 0 while still counting them in
            # `processed` -- the denominator of every percentage in the tables --
            # so a skipped pair read as "the two segmentations agree".
            missing = [l for l, t in ((true_lang, true_tok), (pred_lang, pred_tok))
                       if t is None]
            raise RuntimeError(
                f"no tokenizer for {missing} while scoring {len(indices):,} "
                f"misclassifications of ({true_lang} -> {pred_lang}); "
                "build_tokenizer_cache should have aborted before this point")

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


def generate_length_bias_analysis(output_dir: str = "outputs",
                                  model_path: str = UNILID_MODEL_PATH,
                                  pred_path: str = None,
                                  line_indices=None,
                                  pred_labels=None,
                                  true_labels=None,
                                  basis_note: str = None,
                                  basis_caption: str = None,
                                  artifact_suffix: str = ""):
    """Run the full length bias + normalization counterfactual analysis.

    The misclassified set comes from a recorded prediction source and the token
    deltas are computed from ``model_path``. Those two have to describe the same
    model: pairing a different model with the default prediction file produces an
    analysis of one model's errors scored under another's weights, and it runs to
    completion without complaining. The guard below makes that combination an
    error instead.

    ``line_indices`` / ``pred_labels`` / ``true_labels`` restrict the analysis to
    a subset of the test file, supply that subset's predictions, and gate the
    alignment of both (see stream_misclassified_texts). ``basis_note`` is
    prepended to the markdown report, ``basis_caption`` is appended to every
    LaTeX caption and its slug to every LaTeX label, and ``artifact_suffix`` is
    appended to every artifact's basename, so a subset run never overwrites the
    full-pool artifacts and no fragment of it can be read as a full-pool one.
    Defaults leave all of them unset and reproduce the original behaviour,
    filenames and captions exactly.
    """
    if os.path.abspath(model_path) != os.path.abspath(UNILID_MODEL_PATH) \
            and pred_path is None and pred_labels is None:
        raise RuntimeError(
            f"model_path is {model_path}, but the misclassified set would come "
            f"from {PRED_FILES['UniLID']}, which records the released model's "
            f"predictions. Pass pred_path or pred_labels for the same model, or "
            f"the analysis would attribute one model's errors to another's weights")
    print("  Streaming misclassified texts from disk...")
    misclassified = stream_misclassified_texts(pred_path, line_indices, pred_labels,
                                               true_labels)

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
    ax.set_title("Distribution of token count delta for UniLID misclassifications"
                 + (f"\n{basis_caption}" if basis_caption else ""))
    ax.legend()
    fig.tight_layout()
    hist_path = os.path.join(figures_dir,
                             f"length_bias_histogram{artifact_suffix}.png")
    fig.savefig(hist_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ===================================================================
    # Write outputs
    # ===================================================================
    full_md = "\n".join([
        md_delta, md_clusters_delta, md_tests,
        md_counterfactual, md_cf_clusters, md_cf_delta,
    ])
    if basis_note:
        full_md = basis_note.rstrip("\n") + "\n\n" + full_md
    md_path = os.path.join(tables_dir, f"length_bias{artifact_suffix}.md")
    with open(md_path, "w") as f:
        f.write(full_md)

    # LaTeX. A subset run's fragments carry the basis in every caption and a
    # suffixed label: without that they are indistinguishable from a full-pool
    # fragment, and the caption is exactly where the published table asserts the
    # 45.6M basis that a subset run no longer has.
    def _cap(text):
        return text if not basis_caption else f"{text} {basis_caption}"

    def _lab(text):
        return text + artifact_suffix.replace("_", "-")

    latex_parts = []
    latex_parts.append(to_latex(
        [[r[0]] + [r[i] for i in range(1, len(r))] for r in rows],
        headers,
        caption=_cap("Token count delta (predicted $-$ true) for UniLID "
                     "misclassifications."),
        label=_lab("tab:length-bias-overall"),
    ))
    if cluster_rows:
        latex_parts.append(to_latex(
            [[r[0]] + [r[i] for i in range(1, len(r))] for r in cluster_rows],
            headers,
            caption=_cap("Token delta by confusion cluster."),
            label=_lab("tab:length-bias-clusters"),
        ))
    latex_parts.append(to_latex(
        cf_rows, cf_headers,
        caption=_cap("Misclassifications correctable by length-normalizing scores."),
        label=_lab("tab:counterfactual-overall"),
    ))
    if cf_cluster_rows:
        latex_parts.append(to_latex(
            cf_cluster_rows, cf_headers,
            caption=_cap("Counterfactual correctability by confusion cluster."),
            label=_lab("tab:counterfactual-clusters"),
        ))
    latex_parts.append(to_latex(
        cf_delta_rows, cf_headers,
        caption=_cap("Counterfactual correctability by token delta."),
        label=_lab("tab:counterfactual-delta"),
    ))

    tex_path = os.path.join(tables_dir, f"length_bias{artifact_suffix}.tex")
    with open(tex_path, "w") as f:
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
    print(f"  Outputs: {md_path}, {tex_path}, {hist_path}")

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
        "md_path": md_path,
        "tex_path": tex_path,
        "hist_path": hist_path,
        "rows": rows,
        "cf_rows": cf_rows,
    }


# ---------------------------------------------------------------------------
# Command line: which model, which lines, which output root
# ---------------------------------------------------------------------------

def _latex_escape(s: str) -> str:
    """Escape the LaTeX specials that occur in the strings this module puts into
    captions (file basenames). analysis.format_utils.to_latex escapes only percent
    signs, so a caption has to arrive already escaped."""
    for ch in ("\\", "&", "%", "$", "#", "_", "{", "}"):
        s = s.replace(ch, "\\" + ch)
    return s.replace("~", "\\textasciitilde{}").replace("^", "\\textasciicircum{}")


def _golden_subset_indices() -> np.ndarray:
    """Absolute test-file line indices of the golden subset (PD-4).

    Selected exactly as analysis/lenbias_norm_table.py selects it: the odd
    positions of the seed-42 DEFAULT_SAMPLE_SIZE draw are the test half, the same
    split analysis/full_test_eval.py and both release gates use. Imported here
    rather than reimplemented so the two tables cannot drift onto different lines.
    """
    from analysis.full_test_eval import _sample_line_indices
    test_pos = (np.arange(DEFAULT_SAMPLE_SIZE) % 2) == 1
    return _sample_line_indices()[test_pos]


def _scratch_true_labels(ctx, line_indices: np.ndarray):
    """The true labels ``ctx.scratch_dir``'s memmaps put on ``line_indices``.

    Fed to stream_misclassified_texts as its alignment gate: these are checked
    against the labels parsed from the test file itself, which is what proves the
    per-line arrays in that scratch root are indexed by test-file line number.
    Returns None when the scratch root has no y_true.npy, so the released
    whole-file path (which reads both files in lockstep and needs no such gate)
    is unaffected."""
    from analysis.transfer_sweep import _load_model_data
    y_path = os.path.join(ctx.scratch_dir, "y_true.npy")
    if not os.path.exists(y_path):
        return None
    _w, langs, _m = _load_model_data(ctx.model_path)
    del _w
    codes = np.asarray(np.lib.format.open_memmap(y_path, mode="r"))[line_indices]
    if (codes < 0).any():
        raise RuntimeError(
            f"{int((codes < 0).sum()):,} of {len(codes):,} subset lines are not "
            f"in the kept pool according to {y_path}; the subset was chosen to "
            "sit inside the scored pool, so this means the two disagree")
    return np.array([langs[c] for c in codes], dtype=object)


def _subset_pred_labels(ctx, line_indices: np.ndarray) -> np.ndarray:
    """This model's own plain-scorer predictions on ``line_indices``, read from
    pred_baseline.npy under the run's own scratch root and mapped through the
    model's own language list.

    Gated two ways. (1) The scratch root's fingerprint.json must record this
    model's sha256: the root is chosen by a flag, and several corrected variants
    sit in one directory, so nothing else stops --model X --scratch-dir Y from
    attributing one model's errors to another's weights -- the exact failure
    generate_length_bias_analysis's own guard exists to prevent, one directory
    over. (2) A negative code aborts: numpy fancy indexing with a negative int16
    would silently wrap to the end of the language array."""
    from analysis.full_test_eval import EMPTY, EXCLUDED, UNSEEN
    from analysis.model_context import model_sha256
    from analysis.transfer_sweep import _load_model_data
    pred_path = os.path.join(ctx.scratch_dir, "pred_baseline.npy")
    if not os.path.exists(pred_path):
        raise FileNotFoundError(
            f"required artifact missing: {pred_path}. A non-default model's "
            "misclassified set must come from that model's own full-pool run.")
    fp_path = os.path.join(ctx.scratch_dir, "fingerprint.json")
    if not os.path.exists(fp_path):
        raise FileNotFoundError(
            f"required artifact missing: {fp_path}. Without it there is nothing "
            f"tying {pred_path} to {ctx.model_path}.")
    with open(fp_path) as f:
        fp = json.load(f)
    recorded = fp.get("model_sha256")
    if not recorded:
        raise RuntimeError(f"{fp_path} records no model_sha256; it predates the "
                           "weights being fingerprinted and cannot pair this "
                           "scratch root with a model")
    actual = model_sha256(ctx.model_path)
    if recorded != actual:
        raise RuntimeError(
            f"{pred_path} was produced from a different model than the one being "
            f"scored.\n  --model      {ctx.model_path} (sha256 {actual})\n"
            f"  {fp_path} records sha256 {recorded}\n"
            "Scoring one model's errors under another's weights runs to "
            "completion and reports a plausible table for neither.")
    _w, langs, _m = _load_model_data(ctx.model_path)
    del _w
    codes = np.asarray(np.load(pred_path, mmap_mode="r"))[line_indices]
    if (codes < 0).any():
        kinds = {UNSEEN: "UNSEEN (never scored)",
                 EXCLUDED: "EXCLUDED (validation line)",
                 EMPTY: "EMPTY (empty after preprocessing)"}
        seen = {kinds.get(int(v), str(int(v))): int((codes == v).sum())
                for v in np.unique(codes[codes < 0])}
        raise RuntimeError(
            f"{int((codes < 0).sum()):,} of {len(codes):,} subset lines carry a "
            f"sentinel in {pred_path}: {seen}. None of them can be given a "
            "predicted label, so the table would silently cover fewer lines than "
            "its N column claims.")
    return np.array([langs[c] for c in codes], dtype=object)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subset", choices=[GOLDEN_SUBSET], default=None,
                    help="restrict the analysis to a named subset of the test "
                         "file (default: all lines, the published basis)")
    ap.add_argument("--out-dir", default=None,
                    help=f"root for tables/ and figures/ (default: "
                         f"{DEFAULT_OUT_ROOT}); required, and required to be "
                         "outside the default root, when --model is not the "
                         "released model")
    add_arguments(ap)
    a = ap.parse_args(argv)

    ctx = resolve(a.model_path, a.scratch_dir,
                  purpose="length-bias / normalization counterfactual")
    out_root = resolve_out_root(ctx, a.out_dir,
                                purpose="length-bias / normalization counterfactual")
    print(f"length bias against {ctx.describe()}\n  reports {out_root}", flush=True)

    line_indices = pred_labels = true_labels = basis_note = basis_caption = None
    if a.subset == GOLDEN_SUBSET:
        line_indices = _golden_subset_indices()
        source = PRED_FILES["UniLID"]
        source_note = (" (the released model's recorded prediction file, the "
                       "published source for this table)")
        if not ctx.is_default_model:
            pred_labels = _subset_pred_labels(ctx, line_indices)
            source = os.path.join(ctx.scratch_dir, "pred_baseline.npy")
            source_note = " (this model's own full-pool run)"
        true_labels = _scratch_true_labels(ctx, line_indices)
        basis_note = (
            f"Model: `{ctx.model_path}`. Predictions: `{source}`{source_note}.\n\n"
            f"BASIS: the golden subset, {len(line_indices):,} lines -- the test "
            f"half of the seed-42 {DEFAULT_SAMPLE_SIZE:,}-line draw, the same "
            f"subset `tab:lenbias-norm` was rebuilt on (PD-4 ruling 2026-08-24). "
            f"The published `tab:lenbias-delta` was computed over all "
            f"{TOTAL_LINES:,} test lines, so the N columns below are NOT "
            f"comparable term-by-term with the published table's.\n\n"
            f"Alignment gate: "
            + ("the true label of every one of these lines was checked against "
               f"`{os.path.join(ctx.scratch_dir, 'y_true.npy')}`."
               if true_labels is not None else
               "no y_true.npy in this scratch root; predictions and texts were "
               "read from the two files in lockstep, which needs no gate."))
        # analysis.format_utils.to_latex escapes only percent signs, so anything
        # put into a caption has to arrive already escaped. A model basename with
        # an underscore (every model in this project has one) would otherwise
        # produce a fragment that does not compile.
        model_name = _latex_escape(os.path.basename(ctx.model_path))
        basis_caption = (
            f"Computed over the golden subset ({len(line_indices):,} lines, the "
            f"test half of the seed-42 {DEFAULT_SAMPLE_SIZE:,}-line draw), NOT "
            f"the full {TOTAL_LINES:,}-line test set, and from the model "
            f"\\texttt{{{model_name}}}.")
    elif not ctx.is_default_model:
        raise SystemExit(
            "a non-default --model has no recorded whole-test-file prediction "
            "file; run it with --subset golden, whose predictions come from that "
            "model's own pred_baseline.npy")

    generate_length_bias_analysis(
        output_dir=out_root, model_path=ctx.model_path,
        line_indices=line_indices, pred_labels=pred_labels,
        true_labels=true_labels, basis_note=basis_note,
        basis_caption=basis_caption, artifact_suffix=SUBSET_SUFFIX[a.subset])
    return 0


if __name__ == "__main__":
    sys.exit(main())
