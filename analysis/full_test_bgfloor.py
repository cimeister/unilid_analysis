"""Full-test-set evaluation of the pooled-token-frequency unseen-token rule
(Exp 50, direction 4; pre-registered 2026-08-06, EXPERIMENTS_RESULTS.md
"Experiment 50 pre-registration").

The rule under test: in every language's row, each entry at the row minimum (the
tokens never seen in that language's training data, i.e. the row's exact-minimum
plateau) is replaced by c + log_p_base(token), where log_p_base is the base Unigram
tokenizer's per-token log-probability (the pooled training-data token distribution,
already stored in the model file) and c is one shared constant fixed so the mean
assigned value over the non-special vocabulary equals FLOOR_TARGET (c = FLOOR_TARGET
minus the mean of log_p_base over the non-special vocabulary). Seen entries and the
four special-token columns are untouched; the promoted floor-21 family's single flat
value is the special case where p_base is uniform. No per-language fitting: a new
language inherits both c and log_p_base.

CAUTION: unlike floor-21's clamp (which only ever lowers a plateau, never raises it
past a seen entry), this rule can leave a plateau entry above some seen entries after
assignment: a token common in the pooled data but unseen in one language can
legitimately outscore a rare seen token in that language. This is expected under the
rule as pre-registered; no monotonicity gate is applied.

Procedure and safety cloned from analysis/full_test_floor21.py: the matrix build is
deterministic and fingerprinted (sha256 of the base matrix, the built matrix, the
p_base vector, the language list, plus chunking/total-lines), a resume with any
mismatch aborts; baseline predictions and y_true are reused read-only from the Exp 16
memmaps (job 2784115) in the same scratch directory, which this script never writes;
per chunk, every parsed label must equal the stored y_true memmap value. There is no
recorded-predictions agreement gate for the new matrix (its predictions are supposed
to differ from baseline); the y_true gate plus the deterministic rebuild carry that
role. This script produces one full-pool scoring pass (pred_bgfloor.npy) and the same
cheap within-stratum report floor-21 records; the judge-part per-language F1 paired
bootstrap against floor21 solo and against baseline (per the pre-registration) is a
separate downstream step, not part of this script.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys

import numpy as np

from analysis.config import DEFAULT_SAMPLE_SIZE, TEST_FILE, TOTAL_LINES
from analysis.metrics import compute_metrics
from analysis.sample_data import load_sample
from analysis.transfer_sweep import (
    _load_model_data, _load_train_counts, _load_unilid_model, UNILID_MODEL_PATH,
)
from analysis.hierarchical_pool import _bootstrap_delta, VAL_MASK, DIAG_CSV
from analysis.full_test_eval import (
    CHUNK_LINES, SCRATCH_DIR, UNSEEN, EXCLUDED, EMPTY, BOOTSTRAP_MAX_N,
    _sample_line_indices, _parse_line,
)
from analysis.full_test_floor21 import FLOOR_TARGET
from analysis.floor_equalization import SPECIAL_P

OUT_DIR = "outputs"
# Sanity bounds on the assigned plateau values (c + log_p_base(token), over every
# (row, col) position actually replaced). A prior analysis of this model's base
# Unigram distribution measured std 0.99 nats and p1-to-p99 spread 4.90 nats over the
# non-special vocabulary, giving assigned plateau levels from -27.61 to -12.31 under
# this formula. These bounds are a loose sanity gate on that prior measurement, not a
# tight equality check: abort only if the built range differs grossly.
PLATEAU_MIN_SANE = -30.0
# The measured maximum assigned value is -12.3143, so -10.0 is a real gate.
PLATEAU_MAX_SANE = -10.0


def _model_io_module():
    """Dynamically load UNILID/unilid/model_io.py, the same way
    analysis.transfer_sweep._load_unilid_model does, so this script has no import-path
    dependency on the UNILID package being installed."""
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    spec = importlib.util.spec_from_file_location(
        "model_io", os.path.join(repo_root, "UNILID", "unilid", "model_io.py"))
    model_io = importlib.util.module_from_spec(spec)
    sys.modules["unilid.model_io"] = model_io
    spec.loader.exec_module(model_io)
    return model_io


def _load_base_log_probs(model_path: str, langs: list, vocab_size: int) -> np.ndarray:
    """Extract the base Unigram tokenizer's per-token log-probabilities, aligned to
    the same vocab-id columns as the per-language weight matrix W.

    analysis.transfer_sweep._load_model_data skips the base-tokenizer block entirely
    (it does `f.read(base_tok_len)  # skip base tokenizer`, UNILID/unilid/model_io.py
    header layout documented at lines 1-17 and mirrored at
    analysis/transfer_sweep.py:62), so the base scores are not available from that
    path. This goes through UNILID/unilid/model_io.py:load_unilid (lines 141-182),
    which decodes the base tokenizer into an HF Tokenizer object, and reuses
    UNILID/unilid/model_io.py:_get_vocab_with_scores (lines 33-37), the same
    token -> log-prob extraction save_unilid uses to build each language's row
    (UNILID/unilid/model_io.py:91-99), applied here to the base tokenizer itself so
    every vocab id is filled with the base distribution's own log-probability."""
    model_io = _model_io_module()
    base_tok, weights_chk, langs_chk = model_io.load_unilid(model_path)
    if list(langs_chk) != list(langs):
        raise RuntimeError("model_io.load_unilid language list does not match "
                           "_load_model_data's language list; model file changed "
                           "between reads")
    if weights_chk.shape[1] != vocab_size:
        raise RuntimeError(f"model_io.load_unilid vocab_size {weights_chk.shape[1]} "
                           f"!= {vocab_size}")
    ref_vocab = base_tok.get_vocab()
    if len(ref_vocab) != vocab_size:
        raise RuntimeError(f"base tokenizer vocab size {len(ref_vocab)} != "
                           f"{vocab_size}")
    vocab_scores = model_io._get_vocab_with_scores(base_tok)
    scores = {token: score for token, score in vocab_scores}
    log_p_base = np.full(vocab_size, np.nan, dtype=np.float64)
    for token, rid in ref_vocab.items():
        lp = scores.get(token)
        if lp is None:
            raise RuntimeError(f"base tokenizer token {token!r} has no score in its "
                               f"own vocab state (id {rid})")
        log_p_base[rid] = lp
    if not np.isfinite(log_p_base).all():
        n_bad = int((~np.isfinite(log_p_base)).sum())
        raise RuntimeError(f"log_p_base has {n_bad} unfilled/non-finite entries after "
                           f"extraction")
    return log_p_base


def build_bgfloor_weights(W: np.ndarray, log_p_base: np.ndarray) -> tuple[np.ndarray, dict]:
    """The bgfloor matrix (Exp 50). Per row, entries at the exact row minimum (the
    plateau) are set to c + log_p_base[v]; seen entries and the four special columns
    are untouched. c is one shared constant, fixed so the mean assigned value over the
    non-special vocabulary equals FLOOR_TARGET.

    Returns (out, stats), where stats holds c, the non-special log_p_base std, the p1
    and p99 percentiles, and the min/max assigned plateau value, for reporting."""
    if log_p_base.ndim != 1 or log_p_base.shape[0] != W.shape[1]:
        raise RuntimeError(f"log_p_base shape {log_p_base.shape} does not match "
                           f"vocab size {W.shape[1]}")
    if not np.isfinite(log_p_base).all():
        raise RuntimeError("log_p_base contains non-finite entries")

    special_cols = np.where(np.all(np.abs(np.exp(W.astype(np.float64))
                                          - SPECIAL_P) < 1e-4, axis=0))[0]
    if len(special_cols) != 4:
        raise RuntimeError(f"expected 4 special columns at p={SPECIAL_P}, "
                           f"found {len(special_cols)}")
    sp_mask = np.zeros(W.shape[1], dtype=bool)
    sp_mask[special_cols] = True

    lp_ns = log_p_base[~sp_mask]
    mean_lp_ns = float(lp_ns.mean())
    c = FLOOR_TARGET - mean_lp_ns
    p1, p99 = float(np.percentile(lp_ns, 1)), float(np.percentile(lp_ns, 99))
    print(f"log_p_base over non-special vocab: mean={mean_lp_ns:.4f} "
          f"std={float(lp_ns.std()):.4f} p1={p1:.4f} p99={p99:.4f} "
          f"(p99-p1 spread={p99 - p1:.4f}); prior-analysis reference: std 0.99, "
          f"p1-p99 spread 4.90")
    print(f"bgfloor constant c = {c:.4f} (FLOOR_TARGET={FLOOR_TARGET} minus mean "
          f"log_p_base over non-special vocab)")

    out = np.array(W, dtype=np.float32)
    tmin, tmax = np.inf, -np.inf
    for i in range(W.shape[0]):
        row = W[i]
        floor = row.min()
        plateau = row == floor
        if not plateau.any():
            raise RuntimeError(f"row {i}: empty plateau mask")
        if plateau[special_cols].any():
            raise RuntimeError(f"row {i}: a special column sits on the floor plateau")
        new_vals = (c + log_p_base[plateau]).astype(np.float32)
        out[i, plateau] = new_vals
        tmin = min(tmin, float(new_vals.min()))
        tmax = max(tmax, float(new_vals.max()))

    if not np.array_equal(out[:, special_cols], W[:, special_cols]):
        raise RuntimeError("special-token columns were modified")
    if not np.isfinite(out).all():
        raise RuntimeError("non-finite weights after bgfloor assignment")

    print(f"bgfloor assigned-value range over all replaced entries: "
          f"[{tmin:.4f}, {tmax:.4f}] (prior-analysis reference: -27.61 to -12.31)")
    if tmin < PLATEAU_MIN_SANE or tmax > PLATEAU_MAX_SANE:
        raise RuntimeError(f"bgfloor assigned-value range [{tmin:.4f}, {tmax:.4f}] "
                           f"outside sanity bounds [{PLATEAU_MIN_SANE}, "
                           f"{PLATEAU_MAX_SANE}]; matrix build likely wrong")
    stats = {"c": c, "std_lp_ns": float(lp_ns.std()), "p1": p1, "p99": p99,
              "plateau_min": tmin, "plateau_max": tmax}
    return out, stats


def run(out_dir: str = OUT_DIR):
    import pandas as pd
    os.makedirs(os.path.join(out_dir, "tables"), exist_ok=True)

    weights, langs, lang_to_idx = _load_model_data()
    train_counts = _load_train_counts()
    n_lang = len(langs)
    N = np.array([train_counts.get(l, 0) for l in langs], dtype=np.float64)
    W = np.array(weights, dtype=np.float32)
    del weights
    log_p_base = _load_base_log_probs(UNILID_MODEL_PATH, langs, W.shape[1])
    w_bgfloor, bg_stats = build_bgfloor_weights(W, log_p_base)
    changed = (w_bgfloor != W).any(axis=1)
    if not changed.all():
        raise RuntimeError(f"bgfloor build left {int((~changed).sum())} of {n_lang} "
                           f"rows unmodified; expected every row to change")
    if not np.array_equal(w_bgfloor[:, :4], W[:, :4]):
        raise RuntimeError("special-token columns (0-3) were modified by the build "
                           "(index-convention cross-check against the value-based "
                           "gate already applied)")

    fp = {"sha256_base_W": hashlib.sha256(W.tobytes()).hexdigest(),
          "sha256_w_bgfloor": hashlib.sha256(w_bgfloor.tobytes()).hexdigest(),
          "sha256_log_p_base": hashlib.sha256(log_p_base.tobytes()).hexdigest(),
          "langs_sha256": hashlib.sha256("|".join(langs).encode()).hexdigest(),
          "chunk_lines": CHUNK_LINES, "total_lines": TOTAL_LINES}
    fp_path = os.path.join(SCRATCH_DIR, "fingerprint_bgfloor.json")
    if os.path.exists(fp_path):
        with open(fp_path) as f:
            prev = json.load(f)
        if prev != fp:
            bad = sorted(k for k in fp if prev.get(k) != fp[k])
            raise RuntimeError(f"bgfloor scratch state mismatch ({bad}); clear the "
                               f"bgfloor files in {SCRATCH_DIR} or restore inputs")
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

    pred_path = os.path.join(SCRATCH_DIR, "pred_bgfloor.npy")
    if os.path.exists(pred_path):
        pred_mm = np.lib.format.open_memmap(pred_path, mode="r+")
        if pred_mm.shape != (TOTAL_LINES,):
            raise RuntimeError(f"pred_bgfloor memmap has shape {pred_mm.shape}")
    else:
        pred_mm = np.lib.format.open_memmap(pred_path, mode="w+", dtype=np.int16,
                                            shape=(TOTAL_LINES,))
        pred_mm[:] = UNSEEN
        pred_mm.flush()
    progress_path = os.path.join(SCRATCH_DIR, "progress_bgfloor.json")
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
                print("Loading model + caching bgfloor weights...", flush=True)
                model = _load_unilid_model()
                model.model.set_weight_sets(w_bgfloor.tolist())
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
    pred_bg = np.asarray(pred_mm)
    if int((pred_bg[kept] == UNSEEN).sum()) != 0:
        raise RuntimeError("unprocessed kept lines remain in pred_bgfloor")
    yk = y[kept]
    base = np.asarray(base_mm)[kept]
    new = pred_bg[kept]

    diag = pd.read_csv(DIAG_CSV)
    cat = dict(zip(diag["lang"], diag["category"]))
    lang_flags = {
        "overall": np.ones(n_lang, bool),
        "tail": N < 1_000,
        "magnets": np.array([cat.get(l) == "flat_magnet" for l in langs]),
        "twins": np.array([cat.get(l) == "twin" for l in langs]),
        "head": N >= 18_000,
    }
    lines_out = ["# Full-test-set evaluation of the pooled-token-frequency unseen-"
                 "token rule (bgfloor, Exp 50)\n",
                 f"One new scoring pass under the bgfloor matrix ({int(kept.sum()):,} "
                 f"lines); baseline predictions and y_true reused from the Exp 16 run "
                 f"(job 2784115). c={bg_stats['c']:.4f} ({FLOOR_TARGET} minus the mean "
                 f"of log_p_base over the non-special vocabulary); no per-language "
                 f"fitting; no selection here.",
                 f"Bootstrap CIs (B=1000) for strata with at most {BOOTSTRAP_MAX_N:,} "
                 f"examples; larger strata report the point delta only.\n",
                 "| stratum | base macroF1 | bgfloor macroF1 | delta | 95% CI |",
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
    lines_out.append(
        f"\nThe computed constant is c = {bg_stats['c']:.4f}. "
        f"The standard deviation of log_p_base over the non-special vocabulary is "
        f"{bg_stats['std_lp_ns']:.4f} nats. "
        f"The p1-to-p99 spread of log_p_base over the non-special vocabulary is "
        f"{bg_stats['p99'] - bg_stats['p1']:.4f} nats. "
        f"The minimum assigned plateau value is {bg_stats['plateau_min']:.4f}. "
        f"The maximum assigned plateau value is {bg_stats['plateau_max']:.4f}.")

    out_path = os.path.join(out_dir, "tables", "full_test_bgfloor.md")
    with open(out_path, "w") as f:
        f.write("\n".join(lines_out))
    print("\n".join(lines_out))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    run()
