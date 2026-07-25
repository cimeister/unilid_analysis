"""Full-test-set evaluation of floor equalization at F=-21 (Exp 20 follow-up).

The Exp 20 sweep selected floor-21 on val (test-half overall +0.0030 [CI +0.0016,
+0.0044]) but its tail delta (-0.0623, CI touching 0) rests on ~35 noisy test-half
items, the same situation Exp 16 resolved for the learned bias. This script scores the
floor-21 matrix on the full test set minus the 250k val lines, ONE new scoring pass:
the baseline predictions and y_true come from the saved Exp 16 memmaps (job 2784115)
in the same scratch directory, which this script never writes.

Safety: the floor-21 matrix is rebuilt deterministically (build_equalized_weights,
F = FLOOR_TARGET) and fingerprinted (sha256 of the base and modified matrices +
chunking + language list); a resume with any mismatch aborts. Per chunk, every parsed
label must equal the stored y_true memmap value (cross-run alignment gate), and sampled
test-half lines are additionally checked against the sample pickle. There is no
recorded-predictions agreement gate for the modified matrix (its predictions are
supposed to differ); the y_true gate plus the deterministic rebuild carry that role.
"""
from __future__ import annotations

import hashlib
import json
import os

import numpy as np

from analysis.config import DEFAULT_SAMPLE_SIZE, TEST_FILE, TOTAL_LINES
from analysis.metrics import compute_metrics
from analysis.sample_data import load_sample
from analysis.transfer_sweep import _load_model_data, _load_train_counts, _load_unilid_model
from analysis.hierarchical_pool import _bootstrap_delta, VAL_MASK, DIAG_CSV
from analysis.full_test_eval import (
    CHUNK_LINES, SCRATCH_DIR, UNSEEN, EXCLUDED, EMPTY, BOOTSTRAP_MAX_N,
    _sample_line_indices, _parse_line,
)
from analysis.floor_equalization import build_equalized_weights

FLOOR_TARGET = -21.0            # the Exp 20 guard-selected constant
OUT_DIR = "outputs"


def run(out_dir: str = OUT_DIR):
    import pandas as pd
    os.makedirs(os.path.join(out_dir, "tables"), exist_ok=True)

    weights, langs, lang_to_idx = _load_model_data()
    train_counts = _load_train_counts()
    n_lang = len(langs)
    N = np.array([train_counts.get(l, 0) for l in langs], dtype=np.float64)
    W = np.array(weights, dtype=np.float32)
    del weights
    w21, n_mod = build_equalized_weights(W, FLOOR_TARGET)
    if n_mod != n_lang:
        raise RuntimeError(f"floor {FLOOR_TARGET} modified {n_mod} rows, expected all "
                           f"{n_lang} (row floors all exceed it)")
    if not np.array_equal(w21[:, :4], W[:, :4]):
        raise RuntimeError("special-token columns (0-3) were modified by the clamp")

    fp = {"sha256_base_W": hashlib.sha256(W.tobytes()).hexdigest(),
          "sha256_w21": hashlib.sha256(w21.tobytes()).hexdigest(),
          "floor_target": FLOOR_TARGET,
          "langs_sha256": hashlib.sha256("|".join(langs).encode()).hexdigest(),
          "chunk_lines": CHUNK_LINES, "total_lines": TOTAL_LINES}
    fp_path = os.path.join(SCRATCH_DIR, "fingerprint_floor21.json")
    if os.path.exists(fp_path):
        with open(fp_path) as f:
            prev = json.load(f)
        if prev != fp:
            bad = sorted(k for k in fp if prev.get(k) != fp[k])
            raise RuntimeError(f"floor21 scratch state mismatch ({bad}); clear the "
                               f"floor21 files in {SCRATCH_DIR} or restore inputs")
    else:
        with open(fp_path + ".tmp", "w") as f:
            json.dump(fp, f)
        os.replace(fp_path + ".tmp", fp_path)

    # saved Exp 16 memmaps (read-only inputs)
    y_mm = np.lib.format.open_memmap(os.path.join(SCRATCH_DIR, "y_true.npy"), mode="r")
    base_mm = np.lib.format.open_memmap(os.path.join(SCRATCH_DIR, "pred_baseline.npy"),
                                        mode="r")
    for name, arr in (("y_true", y_mm), ("pred_baseline", base_mm)):
        if arr.shape != (TOTAL_LINES,):
            raise RuntimeError(f"{name} memmap has shape {arr.shape}")
    y_all = np.asarray(y_mm)
    if int((y_all == UNSEEN).sum()) != 0:
        raise RuntimeError("y_true memmap contains UNSEEN lines; Exp 16 run incomplete")
    # the parent wrote predictions only for kept lines; val positions are legitimately
    # UNSEEN in the pred memmaps, so completeness is checked on kept lines only
    if int((np.asarray(base_mm)[y_all >= 0] == UNSEEN).sum()) != 0:
        raise RuntimeError("pred_baseline has UNSEEN entries on kept lines; Exp 16 run "
                           "incomplete")

    sample_idx = _sample_line_indices()
    parity_val = (np.arange(DEFAULT_SAMPLE_SIZE) % 2) == 0
    if not np.array_equal(np.load(VAL_MASK), parity_val):
        raise RuntimeError("saved val_mask.npy does not match position-parity split")
    val_lines = set(sample_idx[parity_val].tolist())
    sample_test_lines = sample_idx[~parity_val]
    pickle_y = np.array(load_sample(DEFAULT_SAMPLE_SIZE)["y_true"])[~parity_val]
    expect_label = dict(zip(sample_test_lines.tolist(), pickle_y.tolist()))

    pred_path = os.path.join(SCRATCH_DIR, "pred_floor21.npy")
    if os.path.exists(pred_path):
        pred_mm = np.lib.format.open_memmap(pred_path, mode="r+")
        if pred_mm.shape != (TOTAL_LINES,):
            raise RuntimeError(f"pred_floor21 memmap has shape {pred_mm.shape}")
    else:
        pred_mm = np.lib.format.open_memmap(pred_path, mode="w+", dtype=np.int16,
                                            shape=(TOTAL_LINES,))
        pred_mm[:] = UNSEEN
        pred_mm.flush()
    progress_path = os.path.join(SCRATCH_DIR, "progress_floor21.json")
    done_chunks = set()
    if os.path.exists(progress_path):
        with open(progress_path) as f:
            done_chunks = set(json.load(f))

    n_chunks = (TOTAL_LINES + CHUNK_LINES - 1) // CHUNK_LINES
    model = None
    zero_bias = [0.0] * n_lang
    with open(TEST_FILE) as fh:
        for chunk in range(n_chunks):
            lo = chunk * CHUNK_LINES
            hi = min(lo + CHUNK_LINES, TOTAL_LINES)
            if chunk in done_chunks:
                for _ in range(hi - lo):
                    fh.readline()
                continue
            if model is None:
                print("Loading model + caching floor-21 weights...", flush=True)
                model = _load_unilid_model()
                model.model.set_weight_sets(w21.tolist())
            lines = [fh.readline() for _ in range(hi - lo)]

            keep_pos, texts = [], []
            for j, line in enumerate(lines):
                i = lo + j
                if i in val_lines:
                    if y_mm[i] != EXCLUDED:
                        raise RuntimeError(f"line {i}: val line not EXCLUDED in the "
                                           f"saved y_true memmap")
                    continue
                label, text = _parse_line(line)
                exp = expect_label.get(i)
                if exp is not None and label != exp:
                    raise RuntimeError(f"alignment mismatch at line {i}: parsed "
                                       f"{label!r}, pickle has {exp!r}")
                li = lang_to_idx.get(label)
                if li is None or y_mm[i] != li:
                    raise RuntimeError(f"line {i}: label {label!r} (idx {li}) does not "
                                       f"match saved y_true {int(y_mm[i])}")
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
                batch = model.model.best_of_cached_weight_sets_biased_batch(pre, zero_bias)
                if len(batch) != len(pre):
                    raise RuntimeError(f"chunk {chunk}: scorer returned {len(batch)} "
                                       f"results for {len(pre)} inputs")
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
    pred21 = np.asarray(pred_mm)
    if int((pred21[kept] == UNSEEN).sum()) != 0:
        raise RuntimeError("unprocessed kept lines remain in pred_floor21")
    yk = y[kept]
    base = np.asarray(base_mm)[kept]
    new = pred21[kept]

    diag = pd.read_csv(DIAG_CSV)
    cat = dict(zip(diag["lang"], diag["category"]))
    lang_flags = {
        "overall": np.ones(n_lang, bool),
        "tail": N < 1_000,
        "magnets": np.array([cat.get(l) == "flat_magnet" for l in langs]),
        "twins": np.array([cat.get(l) == "twin" for l in langs]),
        "head": N >= 18_000,
    }
    lines_out = ["# Full-test-set evaluation of floor equalization (F=-21)\n",
                 f"One new scoring pass under the floor-21 matrix ({int(kept.sum()):,} "
                 f"lines); baseline predictions and y_true reused from the Exp 16 run "
                 f"(job 2784115). Config fixed on val by the Exp 20 guard; no selection "
                 f"here.",
                 f"Bootstrap CIs (B=1000) for strata with at most {BOOTSTRAP_MAX_N:,} "
                 f"examples; larger strata report the point delta only.\n",
                 "| stratum | base macroF1 | floor-21 macroF1 | delta | 95% CI |",
                 "|---|---|---|---|---|"]
    for st, flags in lang_flags.items():
        m = flags[yk]
        n_st = int(m.sum())
        bm = compute_metrics(yk[m], base[m])
        nm = compute_metrics(yk[m], new[m])
        d = nm["macro_f1"] - bm["macro_f1"]
        if 0 < n_st <= BOOTSTRAP_MAX_N:
            _, lo_ci, hi_ci = _bootstrap_delta(yk, base, new, m)
            ci = f"[{lo_ci:+.4f}, {hi_ci:+.4f}]"
        else:
            ci = "point only (n > cap)"
        lines_out.append(f"| {st} ({n_st:,}) | {bm['macro_f1']:.4f} | "
                         f"{nm['macro_f1']:.4f} | {d:+.4f} | {ci} |")
    acc_b = float((base == yk).mean())
    acc_n = float((new == yk).mean())
    lines_out.append(f"\nOverall accuracy: {acc_b:.4f} -> {acc_n:.4f} "
                     f"({acc_n-acc_b:+.4f}).")

    out_path = os.path.join(out_dir, "tables", "full_test_floor21.md")
    with open(out_path, "w") as f:
        f.write("\n".join(lines_out))
    print("\n".join(lines_out))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    run()
