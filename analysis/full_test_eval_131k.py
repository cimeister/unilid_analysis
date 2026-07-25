"""Full-test baseline evaluation of the Apertus 131k (preliminary_mul) retrain
(plan Track A step 3; model glotlid_apertus131k.unilid, jobs 2883222).

One scoring pass (b = 0 only) over the full test pool, modeled on full_test_eval.py:
- Language-list gate: the 131k model's language list must equal the 100k model's
  list in order; the saved y_true memmap (label indices) is then valid for both and
  is reused READ-ONLY. Abort otherwise (a label-string join is not implemented).
- No recorded-predictions agreement gate (this model legitimately differs); the
  per-line y_true alignment gate carries that role, as in full_test_floor21.py.
- Separate scratch dir for all writable state (pred memmap, fingerprint, progress);
  the Exp 16 scratch dir is never written.

Report (one artifact, the Track A branch-decision numbers):
1. within-stratum macro-F1, 100k baseline vs 131k baseline, bootstrap CIs on small
   strata;
2. global per-language mean F1 by group (both models, full pool);
3. balanced-val draw-101 within-stratum rows (selection view under the current
   protocol);
plus a per-language PRF CSV for the 131k baseline.
"""
from __future__ import annotations

import hashlib
import json
import os

import numpy as np

from analysis.config import DEFAULT_SAMPLE_SIZE, TEST_FILE, TOTAL_LINES
from analysis.metrics import compute_metrics
from analysis.sample_data import load_sample
from analysis.transfer_sweep import (_load_model_data, _load_train_counts,
                                     _load_unilid_model)
from analysis.hierarchical_pool import _bootstrap_delta, VAL_MASK, DIAG_CSV
from analysis.full_test_eval import (
    CHUNK_LINES, SCRATCH_DIR as SCRATCH_100K, UNSEEN, EXCLUDED, EMPTY,
    BOOTSTRAP_MAX_N, _sample_line_indices, _parse_line,
)
from analysis.metric_decomposition import _per_lang_stats
from analysis.balanced_split import OUT_DIR as DRAW_DIR, SEEDS

MODEL_131K = "/capstor/scratch/cscs/cmeister747/unilid_analysis/glotlid_apertus131k.unilid"
SCRATCH_DIR = "/capstor/scratch/cscs/cmeister747/unilid_analysis/full_test_eval_131k"
OUT_MD = "outputs/tables/full_test_eval_131k.md"
OUT_CSV = "outputs/diagnostic/full_test_131k_per_lang_prf.csv"


def run():
    import pandas as pd
    os.makedirs(SCRATCH_DIR, exist_ok=True)
    os.makedirs("outputs/tables", exist_ok=True)

    _w100, langs100, _m100 = _load_model_data()
    w131, langs131, lang_to_idx = _load_model_data(MODEL_131K)
    del w131
    if langs131 != langs100:
        raise RuntimeError("131k language list differs from the 100k list; the "
                           "shared y_true indexing is invalid and a label-string "
                           "join is required (not implemented)")
    langs = langs131
    n_lang = len(langs)
    train_counts = _load_train_counts()
    N = np.array([train_counts.get(l, 0) for l in langs], dtype=np.float64)

    fp = {"model_sha256": hashlib.sha256(open(MODEL_131K, "rb").read()).hexdigest(),
          "langs_sha256": hashlib.sha256("|".join(langs).encode()).hexdigest(),
          "chunk_lines": CHUNK_LINES, "total_lines": TOTAL_LINES}
    fp_path = os.path.join(SCRATCH_DIR, "fingerprint.json")
    if os.path.exists(fp_path):
        with open(fp_path) as f:
            prev = json.load(f)
        if prev != fp:
            bad = sorted(k for k in fp if prev.get(k) != fp[k])
            raise RuntimeError(f"131k scratch state mismatch ({bad}); clear "
                               f"{SCRATCH_DIR} or restore inputs")
    else:
        with open(fp_path + ".tmp", "w") as f:
            json.dump(fp, f)
        os.replace(fp_path + ".tmp", fp_path)

    y_mm = np.lib.format.open_memmap(os.path.join(SCRATCH_100K, "y_true.npy"),
                                     mode="r")
    base100_mm = np.lib.format.open_memmap(
        os.path.join(SCRATCH_100K, "pred_baseline.npy"), mode="r")
    for name, arr in (("y_true", y_mm), ("pred_baseline", base100_mm)):
        if arr.shape != (TOTAL_LINES,):
            raise RuntimeError(f"{name} memmap has shape {arr.shape}")
    y_all = np.asarray(y_mm)
    if int((y_all == UNSEEN).sum()) != 0:
        raise RuntimeError("y_true memmap contains UNSEEN lines")

    sample_idx = _sample_line_indices()
    parity_val = (np.arange(DEFAULT_SAMPLE_SIZE) % 2) == 0
    if not np.array_equal(np.load(VAL_MASK), parity_val):
        raise RuntimeError("saved val_mask.npy does not match position-parity split")
    val_lines = set(sample_idx[parity_val].tolist())
    sample_test_lines = sample_idx[~parity_val]
    pickle_y = np.array(load_sample(DEFAULT_SAMPLE_SIZE)["y_true"])[~parity_val]
    expect_label = dict(zip(sample_test_lines.tolist(), pickle_y.tolist()))

    pred_path = os.path.join(SCRATCH_DIR, "pred_baseline131k.npy")
    if os.path.exists(pred_path):
        pred_mm = np.lib.format.open_memmap(pred_path, mode="r+")
        if pred_mm.shape != (TOTAL_LINES,):
            raise RuntimeError(f"pred memmap has shape {pred_mm.shape}")
    else:
        pred_mm = np.lib.format.open_memmap(pred_path, mode="w+", dtype=np.int16,
                                            shape=(TOTAL_LINES,))
        pred_mm[:] = UNSEEN
        pred_mm.flush()
    progress_path = os.path.join(SCRATCH_DIR, "progress.json")
    done_chunks = set()
    if os.path.exists(progress_path):
        with open(progress_path) as f:
            done_chunks = set(json.load(f))

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
                print("Loading the 131k model (weights cached by the loader)...",
                      flush=True)
                model = _load_unilid_model(MODEL_131K)
            lines = [fh.readline() for _ in range(hi - lo)]

            keep_pos, texts = [], []
            for j, line in enumerate(lines):
                i = lo + j
                if i in val_lines:
                    if y_mm[i] != EXCLUDED:
                        raise RuntimeError(f"line {i}: val line not EXCLUDED")
                    continue
                label, text = _parse_line(line)
                exp = expect_label.get(i)
                if exp is not None and label != exp:
                    raise RuntimeError(f"alignment mismatch at line {i}: parsed "
                                       f"{label!r}, pickle has {exp!r}")
                li = lang_to_idx.get(label)
                if li is None or y_mm[i] != li:
                    raise RuntimeError(f"line {i}: label {label!r} (idx {li}) does "
                                       f"not match saved y_true {int(y_mm[i])}")
                keep_pos.append(i)
                texts.append(text)

            pre, valid = [], []
            for k, t in enumerate(texts):
                p = model.preprocess(t)
                if p:
                    pre.append(p)
                    valid.append(k)
            out = np.full(len(texts), EMPTY, dtype=np.int16)
            if pre:
                batch = model.model.best_of_cached_weight_sets_batch(pre)
                if len(batch) != len(pre):
                    raise RuntimeError(f"chunk {chunk}: scorer returned "
                                       f"{len(batch)} results for {len(pre)} inputs")
                for k, (idx, _t, _s) in zip(valid, batch):
                    out[k] = idx
            pred_mm[np.asarray(keep_pos, dtype=np.int64)] = out
            pred_mm.flush()
            done_chunks.add(chunk)
            with open(progress_path + ".tmp", "w") as f:
                json.dump(sorted(done_chunks), f)
            os.replace(progress_path + ".tmp", progress_path)
            print(f"chunk {chunk + 1}/{n_chunks} done", flush=True)

    print("Computing metrics...", flush=True)
    y = np.asarray(y_mm)
    kept = y >= 0
    pred131 = np.asarray(pred_mm)
    if int((pred131[kept] == UNSEEN).sum()) != 0:
        raise RuntimeError("unprocessed kept lines remain")
    yk = y[kept]
    base = np.asarray(base100_mm)[kept]
    new = pred131[kept]

    diag = pd.read_csv(DIAG_CSV)
    cat = dict(zip(diag["lang"], diag["category"]))
    category = np.array([cat.get(l, "mid") for l in langs])
    lang_flags = {
        "overall": np.ones(n_lang, bool),
        "tail": N < 1_000,
        "magnets": category == "flat_magnet",
        "twins": category == "twin",
        "head": N >= 18_000,
    }

    L = ["# Apertus 131k (preliminary_mul) baseline vs the 100k baseline\n",
         f"One scoring pass, b = 0, {int(kept.sum()):,} lines; 100k baseline and "
         "y_true reused read-only from the Exp 16 memmaps. Branch-decision numbers "
         "for plan Track A.\n",
         "## Within-stratum macro-F1 (full test)\n",
         "| stratum | 100k | 131k | delta | 95% CI |", "|---|---|---|---|---|"]
    for st, flags in lang_flags.items():
        m = flags[yk]
        n_st = int(m.sum())
        bm = compute_metrics(yk[m], base[m])["macro_f1"]
        nm = compute_metrics(yk[m], new[m])["macro_f1"]
        if 0 < n_st <= BOOTSTRAP_MAX_N:
            _, lo_ci, hi_ci = _bootstrap_delta(yk, base, new, m)
            ci = f"[{lo_ci:+.4f}, {hi_ci:+.4f}]"
        else:
            ci = "point only (n > cap)"
        L.append(f"| {st} ({n_st:,}) | {bm:.4f} | {nm:.4f} | {nm - bm:+.4f} | {ci} |")
    acc_b = float((base == yk).mean())
    acc_n = float((new == yk).mean())
    L.append(f"\nOverall accuracy: {acc_b:.4f} -> {acc_n:.4f} ({acc_n - acc_b:+.4f}).")

    stats100 = _per_lang_stats(base, yk, n_lang)
    stats131 = _per_lang_stats(new, yk, n_lang)
    groups = {"tail (N<1k)": N < 1_000, "lowmid (1k<=N<18k)": (N >= 1_000) & (N < 18_000),
              "head (N>=18k)": N >= 18_000, "flat_magnet": category == "flat_magnet",
              "twin": category == "twin", "all 1,940": np.ones(n_lang, bool)}
    L += ["\n## Global per-language mean F1 by group (full pool)\n",
          "| model | " + " | ".join(groups) + " |", "|" + "---|" * (len(groups) + 1)]
    for name, st in [("100k", stats100), ("131k", stats131)]:
        L.append(f"| {name} | " + " | ".join(f"{st[2][m].mean():.4f}"
                                             for m in groups.values()) + " |")
    fps100 = stats100[4][N < 1_000].sum()
    fps131 = stats131[4][N < 1_000].sum()
    L.append(f"\nFPs into tail labels: 100k {fps100:,.0f} -> 131k {fps131:,.0f}.")

    val101 = np.load(os.path.join(DRAW_DIR, f"val_lines_seed{SEEDS[0]}.npy"))
    vmask = np.zeros(TOTAL_LINES, bool)
    vmask[val101] = True
    if not (y[vmask] >= 0).all():
        raise RuntimeError("balanced-val draw contains non-kept lines")
    L += ["\n## Balanced-val draw 101, within-stratum macro-F1 (selection view)\n",
          "| model | overall | tail | magnets | twins | head |", "|" + "---|" * 6]
    for name, parr in [("100k", np.asarray(base100_mm)), ("131k", pred131)]:
        ym, pm = y[vmask], parr[vmask]
        row = {st: compute_metrics(ym[flags[ym]], pm[flags[ym]])["macro_f1"]
               for st, flags in lang_flags.items()}
        L.append(f"| {name} | " + " | ".join(
            f"{row[st]:.4f}" for st in ["overall", "tail", "magnets", "twins",
                                        "head"]) + " |")

    per = {"lang": langs, "N": N.astype(int), "category": category,
           "test_support": np.bincount(yk, minlength=n_lang)}
    for nm_, st in [("100k", stats100), ("131k", stats131)]:
        per[f"prec_{nm_}"] = st[0]
        per[f"rec_{nm_}"] = st[1]
        per[f"f1_{nm_}"] = st[2]
        per[f"fp_{nm_}"] = st[4].astype(int)
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    pd.DataFrame(per).to_csv(OUT_CSV, index=False)

    with open(OUT_MD, "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\nWrote {OUT_MD} and {OUT_CSV}")


if __name__ == "__main__":
    run()
