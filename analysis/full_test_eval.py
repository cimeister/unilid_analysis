"""Full-test-set evaluation of the finished configurations (plan item 10, first part).

Scores the 100k model on the ENTIRE GlotLID test set minus the 250k val lines (the
val half of the seed-42 500k sample) for three FIXED configurations:
  baseline (b = 0), frequency prior (b = 0.5 * log(N+1), Exp 14), and the learned
  per-language bias (outputs/tables/learned_bias.npy, the reg=5.0 fit from job 2731802).
No selection happens here: all three configurations were fixed on val. The purpose is
to tighten the stratified deltas: on the 250k test half every tail language has <= 2
examples, so the tail deltas rest on a handful of items; the full test set has ~90x more.

Alignment safety: the sampled val/test line indices are reconstructed with the same
seed-42 draw as analysis/sample_data.py, cross-checked against the saved val mask, and
every kept line that belongs to the sample's test half must match the pickle's stored
y_true label exactly (abort on the first mismatch).

Resumable: predictions and true labels are written to int16 memmaps on scratch keyed by
absolute line index, with completed chunks tracked in a progress file; a restart skips
completed chunks.
"""
from __future__ import annotations

import hashlib
import json
import os
import random

import numpy as np

from analysis.config import DEFAULT_SAMPLE_SIZE, SAMPLE_SEED, TEST_FILE, TOTAL_LINES
from analysis.metrics import compute_metrics, compute_per_class_metrics
from analysis.sample_data import load_sample
from analysis.transfer_sweep import _load_model_data, _load_train_counts, _load_unilid_model
from analysis.hierarchical_pool import _bootstrap_delta, VAL_MASK, DIAG_CSV

CHUNK_LINES = 500_000
FREQ_GAMMA = 0.5                    # the Exp 14 guard-selected frequency prior
LEARNED_BIAS_NPY = "outputs/tables/learned_bias.npy"
SCRATCH_DIR = "/capstor/scratch/cscs/cmeister747/unilid_analysis/full_test_eval"
OUT_DIR = "outputs"
UNSEEN = -3                         # memmap sentinel: line not processed yet
EXCLUDED = -2                       # memmap sentinel: val line, excluded
EMPTY = -1                          # sentinel: empty after preprocess (scored as wrong)
# Bootstrap CIs are computed only for strata up to this size; for larger strata the
# item-level CI half-width is below ~1e-3 and the point delta is reported with a note.
BOOTSTRAP_MAX_N = 3_000_000
CONFIGS = ["baseline", "freq_prior", "learned_bias"]


def _sample_line_indices():
    """Reproduce the seed-42 sample draw from analysis/sample_data.py exactly."""
    random.seed(SAMPLE_SEED)
    idx = sorted(random.sample(range(TOTAL_LINES), DEFAULT_SAMPLE_SIZE))
    return np.array(idx, dtype=np.int64)


def _parse_line(line: str):
    parts = line.split(" ", 1)
    label = parts[0].replace("__label__", "")
    text = parts[1].rstrip("\n") if len(parts) > 1 else ""
    return label, text


def _fingerprint(biases: dict, langs: list) -> dict:
    """Identity of everything that determines the memmap contents. A resume with ANY
    mismatch (edited bias, refit learned_bias.npy, changed chunking, different language
    set) must abort rather than mix configurations across chunks."""
    fp = {f"sha256_{c}": hashlib.sha256(np.ascontiguousarray(b).tobytes()).hexdigest()
          for c, b in biases.items()}
    fp["langs_sha256"] = hashlib.sha256("|".join(langs).encode()).hexdigest()
    fp["chunk_lines"] = CHUNK_LINES
    fp["total_lines"] = TOTAL_LINES
    return fp


def run(out_dir: str = OUT_DIR):
    import pandas as pd
    os.makedirs(SCRATCH_DIR, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "tables"), exist_ok=True)

    weights, langs, lang_to_idx = _load_model_data()
    del weights
    train_counts = _load_train_counts()
    n_lang = len(langs)
    N = np.array([train_counts.get(l, 0) for l in langs], dtype=np.float64)

    biases = {
        "baseline": np.zeros(n_lang, dtype=np.float32),
        "freq_prior": (FREQ_GAMMA * np.log(N + 1.0)).astype(np.float32),
        "learned_bias": np.load(LEARNED_BIAS_NPY).astype(np.float32),
    }
    if biases["learned_bias"].shape != (n_lang,):
        raise RuntimeError(f"learned bias shape {biases['learned_bias'].shape} != ({n_lang},)")

    fp = _fingerprint(biases, langs)
    fp_path = os.path.join(SCRATCH_DIR, "fingerprint.json")
    if os.path.exists(fp_path):
        with open(fp_path) as f:
            prev = json.load(f)
        if prev != fp:
            bad = sorted(k for k in fp if prev.get(k) != fp[k])
            raise RuntimeError(
                f"scratch state in {SCRATCH_DIR} was produced under a different "
                f"configuration (mismatched: {bad}); clear the directory or restore "
                f"the original inputs before resuming")
    else:
        with open(fp_path + ".tmp", "w") as f:
            json.dump(fp, f)
        os.replace(fp_path + ".tmp", fp_path)

    # --- val/test line bookkeeping + alignment data ---
    sample_idx = _sample_line_indices()
    val_mask_saved = np.load(VAL_MASK)
    parity_val = (np.arange(DEFAULT_SAMPLE_SIZE) % 2) == 0
    if not np.array_equal(val_mask_saved, parity_val):
        raise RuntimeError("saved val_mask.npy does not match position-parity split")
    val_lines = set(sample_idx[parity_val].tolist())
    sample_test_lines = sample_idx[~parity_val]
    sample_pkl = load_sample(DEFAULT_SAMPLE_SIZE)
    pickle_y = np.array(sample_pkl["y_true"])[~parity_val]
    expect_label = dict(zip(sample_test_lines.tolist(), pickle_y.tolist()))
    pickle_pred = np.array(sample_pkl["pred_UniLID"])[~parity_val]
    expect_pred = dict(zip(sample_test_lines.tolist(), pickle_pred.tolist()))

    # --- resumable memmaps keyed by absolute line index ---
    paths = {c: os.path.join(SCRATCH_DIR, f"pred_{c}.npy") for c in CONFIGS}
    paths["y_true"] = os.path.join(SCRATCH_DIR, "y_true.npy")
    mm = {}
    for name, p in paths.items():
        if os.path.exists(p):
            mm[name] = np.lib.format.open_memmap(p, mode="r+")
            if mm[name].shape != (TOTAL_LINES,):
                raise RuntimeError(f"existing memmap {p} has shape {mm[name].shape}")
        else:
            mm[name] = np.lib.format.open_memmap(p, mode="w+", dtype=np.int16,
                                                 shape=(TOTAL_LINES,))
            mm[name][:] = UNSEEN
            mm[name].flush()
    progress_path = os.path.join(SCRATCH_DIR, "progress.json")
    done_chunks = set()
    if os.path.exists(progress_path):
        with open(progress_path) as f:
            done_chunks = set(json.load(f))
    agree_hit, agree_total = 0, 0

    n_chunks = (TOTAL_LINES + CHUNK_LINES - 1) // CHUNK_LINES
    model = None
    with open(TEST_FILE) as fh:
        for chunk in range(n_chunks):
            lo = chunk * CHUNK_LINES
            hi = min(lo + CHUNK_LINES, TOTAL_LINES)
            if chunk in done_chunks:
                for _ in range(hi - lo):
                    fh.readline()
                continue
            if model is None:
                print("Loading model + caching weight sets...", flush=True)
                model = _load_unilid_model()
            lines = [fh.readline() for _ in range(hi - lo)]

            keep_pos, texts, yt = [], [], []
            for j, line in enumerate(lines):
                i = lo + j
                if i in val_lines:
                    mm["y_true"][i] = EXCLUDED
                    continue
                label, text = _parse_line(line)
                exp = expect_label.get(i)
                if exp is not None and label != exp:
                    raise RuntimeError(f"alignment mismatch at line {i}: parsed "
                                       f"{label!r}, sample pickle has {exp!r}")
                li = lang_to_idx.get(label)
                if li is None:
                    raise RuntimeError(f"line {i}: test label {label!r} not in the model's "
                                       f"{n_lang} languages")
                keep_pos.append(i)
                texts.append(text)
                yt.append(li)

            pre, valid = [], []
            for k, t in enumerate(texts):
                p = model.preprocess(t)
                if p:
                    pre.append(p)
                    valid.append(k)
            keep_pos = np.asarray(keep_pos, dtype=np.int64)
            mm["y_true"][keep_pos] = np.asarray(yt, dtype=np.int16)
            outs = {}
            for c in CONFIGS:
                out = np.full(len(texts), EMPTY, dtype=np.int16)
                if pre:
                    batch = model.model.best_of_cached_weight_sets_biased_batch(
                        pre, biases[c].tolist())
                    if len(batch) != len(pre):
                        raise RuntimeError(f"chunk {chunk} config {c}: scorer returned "
                                           f"{len(batch)} results for {len(pre)} inputs")
                    for k, (idx, _t, _s) in zip(valid, batch):
                        out[k] = idx
                outs[c] = out
                mm[c][keep_pos] = out
            # baseline sanity: the zero-bias scorer must reproduce the recorded UniLID
            # predictions on the sampled test-half lines (same check as the sweeps)
            for k, i in enumerate(keep_pos.tolist()):
                exp_p = expect_pred.get(i)
                if exp_p is not None and outs["baseline"][k] != EMPTY:
                    agree_total += 1
                    agree_hit += int(langs[outs["baseline"][k]] == exp_p)
            for name in mm:
                mm[name].flush()
            done_chunks.add(chunk)
            with open(progress_path + ".tmp", "w") as f:
                json.dump(sorted(done_chunks), f)
            os.replace(progress_path + ".tmp", progress_path)
            print(f"chunk {chunk + 1}/{n_chunks} done ({hi - lo} lines)", flush=True)

    if agree_total >= 1000:
        agreement = agree_hit / agree_total
        print(f"baseline agreement with recorded UniLID preds: {agreement:.4f} "
              f"({agree_total:,} sampled lines)", flush=True)
        if agreement < 0.99:
            raise RuntimeError(f"baseline scorer agreement {agreement:.4f} < 0.99 with "
                               f"the recorded UniLID predictions; model/scorer changed")
    elif agree_total > 0:
        print(f"baseline agreement checked on only {agree_total} lines (resume run); "
              f"earlier chunks were validated by the run that computed them", flush=True)

    # --- metrics ---
    print("Computing metrics...", flush=True)
    y = np.asarray(mm["y_true"])
    kept = y >= 0
    if int((y == UNSEEN).sum()) != 0:
        raise RuntimeError("unprocessed lines remain; progress file inconsistent")
    n_kept = int(kept.sum())
    if n_kept != TOTAL_LINES - len(val_lines):
        raise RuntimeError(f"kept {n_kept} lines, expected {TOTAL_LINES - len(val_lines)}")

    diag = pd.read_csv(DIAG_CSV)
    cat = dict(zip(diag["lang"], diag["category"]))
    lang_flags = {
        "overall": np.ones(n_lang, bool),
        "tail": N < 1_000,
        "magnets": np.array([cat.get(l) == "flat_magnet" for l in langs]),
        "twins": np.array([cat.get(l) == "twin" for l in langs]),
        "head": N >= 18_000,
    }
    yk = y[kept]
    preds = {c: np.asarray(mm[c])[kept] for c in CONFIGS}

    lines_out = ["# Full-test-set evaluation (test set minus the 250k val lines)\n",
                 f"Lines evaluated: {n_kept:,} of {TOTAL_LINES:,}. Configurations fixed on "
                 f"val: frequency prior gamma={FREQ_GAMMA}, learned bias = reg=5.0 fit "
                 f"(job 2731802). Empty-preprocess lines are scored as wrong, matching the "
                 f"500k-sample sweeps.",
                 f"Bootstrap CIs (B=1000) are computed for strata with at most "
                 f"{BOOTSTRAP_MAX_N:,} examples; for larger strata the item-level CI "
                 f"half-width is below ~0.001 and only the point delta is reported.\n"]
    for st, flags in lang_flags.items():
        m = flags[yk]
        n_st = int(m.sum())
        n_langs_st = int(flags.sum())
        sup = np.bincount(yk[m], minlength=n_lang) if n_st else np.zeros(n_lang)
        covered = int(((sup > 0) & flags).sum())
        lines_out.append(f"## {st}: {n_st:,} examples, {covered}/{n_langs_st} languages "
                         f"with test support\n")
        base_m = compute_metrics(yk[m], preds["baseline"][m])
        rows = [f"| config | macroF1 | delta | 95% CI | accuracy |", "|---|---|---|---|---|",
                f"| baseline | {base_m['macro_f1']:.4f} | | | {base_m['accuracy']:.4f} |"]
        for c in CONFIGS[1:]:
            cm = compute_metrics(yk[m], preds[c][m])
            d = cm["macro_f1"] - base_m["macro_f1"]
            if 0 < n_st <= BOOTSTRAP_MAX_N:
                _, lo_ci, hi_ci = _bootstrap_delta(yk, preds["baseline"], preds[c], m)
                ci = f"[{lo_ci:+.4f}, {hi_ci:+.4f}]"
            else:
                ci = "point only (n > cap)"
            rows.append(f"| {c} | {cm['macro_f1']:.4f} | {d:+.4f} | {ci} | "
                        f"{cm['accuracy']:.4f} |")
        lines_out.extend(rows + [""])

    md_path = os.path.join(out_dir, "tables", "full_test_eval.md")
    with open(md_path, "w") as f:
        f.write("\n".join(lines_out))
    print("\n".join(lines_out))

    # per-language F1 artifact for downstream analyses (plan items 5-6)
    per_lang = {"lang": langs, "N": N.astype(int),
                "category": [cat.get(l, "mid") for l in langs],
                "test_support": np.bincount(yk, minlength=n_lang)}
    for c in CONFIGS:
        pc = compute_per_class_metrics(yk, preds[c])
        per_lang[f"f1_{c}"] = [pc.get(i, {}).get("f1", 0.0) for i in range(n_lang)]
    csv_path = os.path.join(out_dir, "diagnostic", "full_test_per_lang_f1.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    pd.DataFrame(per_lang).to_csv(csv_path, index=False)
    print(f"\nWrote {md_path} and {csv_path}")


if __name__ == "__main__":
    run()
