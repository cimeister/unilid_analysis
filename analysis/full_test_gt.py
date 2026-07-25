"""Full-test evaluation of the Good-Turing unseen-mass candidate (config gt_min,
plan B4; one-sided-min rule fixed by the user 2026-07-23).

Weight construction per language row (specials frozen at 0.8 total; the seen+floor
budget is exactly 0.2): current plateau mass M_L = sum(exp) over the exact row-min
plateau; target M' = min(M_L, 0.2 * n1/T) with n1, T from the language's own Viterbi
token counts (outputs/diagnostic/gt_counts.csv, job 2883714); plateau entries set
uniformly to log(M'/plateau_size); seen non-special entries rescaled by
(0.2 - M')/(0.2 - M_L) so the row stays normalized. Per Exp 27's counting result the
min never binds upward (exact GT lowers every language's plateau), so gt_min and
exact GT coincide on this model; the rule is still applied as pre-registered.

No swept constant: a single candidate, judged afterwards by the precision-primary
rule (analysis/two_sided_report.py). Scoring machinery, fingerprints, alignment
gates, and resume protocol are cloned from full_test_floor21.py (job 2791722
pattern); baseline predictions and y_true are reused read-only from the Exp 16
memmaps.
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
from analysis.floor_equalization import SPECIAL_P

GT_CSV = "outputs/diagnostic/gt_counts.csv"
NONSPECIAL_BUDGET = 0.2         # 1.0 minus the four specials at p=0.2 each
ROW_BUDGET_TOL = 1e-4           # |S_L + M_L - 0.2| gate per row (float32 rounding)
OUT_DIR = "outputs"


def build_gt_weights(W: np.ndarray, gt: "pd.DataFrame", langs: list) -> np.ndarray:
    """The gt_min matrix. Gates: gt CSV covers every language and reproduces the
    plateau stats recomputed from W; specials untouched; every plateau lowered or
    kept (one-sided); rows renormalized within tolerance."""
    special_cols = np.where(np.all(np.abs(np.exp(W.astype(np.float64))
                                          - SPECIAL_P) < 1e-4, axis=0))[0]
    if len(special_cols) != 4:
        raise RuntimeError(f"expected 4 special columns at p={SPECIAL_P}, "
                           f"found {len(special_cols)}")
    sp_mask = np.zeros(W.shape[1], bool)
    sp_mask[special_cols] = True

    by_lang = gt.set_index("lang")
    missing = [l for l in langs if l not in by_lang.index]
    if missing:
        raise RuntimeError(f"gt_counts missing {len(missing)} languages, first: "
                           f"{missing[0]}")

    out = np.array(W, dtype=np.float32)
    n_raise_capped = 0
    for i, lang in enumerate(langs):
        row = W[i].astype(np.float64)
        floor = row.min()
        plateau = row == floor
        if plateau[special_cols].any():
            raise RuntimeError(f"{lang}: a special column sits on the floor plateau")
        M = float(np.exp(row[plateau]).sum())
        g = by_lang.loc[lang]
        if int(g["plateau_size"]) != int(plateau.sum()) or \
                abs(float(g["plateau_mass"]) - M) > 1e-6:
            raise RuntimeError(f"{lang}: gt_counts plateau stats do not match W "
                               f"(size {int(g['plateau_size'])} vs "
                               f"{int(plateau.sum())}, mass {g['plateau_mass']:.3e} "
                               f"vs {M:.3e})")
        T, n1 = int(g["T"]), int(g["n1"])
        if T <= 0:
            raise RuntimeError(f"{lang}: T={T}")
        seen = ~plateau & ~sp_mask
        S = float(np.exp(row[seen]).sum())
        if abs(S + M - NONSPECIAL_BUDGET) > ROW_BUDGET_TOL:
            raise RuntimeError(f"{lang}: seen+plateau mass {S + M:.6f} != "
                               f"{NONSPECIAL_BUDGET}")
        target = min(M, NONSPECIAL_BUDGET * n1 / T)
        if NONSPECIAL_BUDGET * n1 / T > M:
            n_raise_capped += 1
        if target <= 0.0:
            # n1 == 0 (every type repeated): the GT plug-in says zero unseen mass,
            # which -inf log-weights cannot represent; abort rather than substitute
            raise RuntimeError(f"{lang}: GT target mass is zero (n1=0); no "
                               "pre-registered handling exists for this case")
        out[i, plateau] = np.float32(np.log(target / plateau.sum()))
        out[i, seen] = (row[seen]
                        + np.log((NONSPECIAL_BUDGET - target)
                                 / (NONSPECIAL_BUDGET - M))).astype(np.float32)
    if not np.array_equal(out[:, special_cols], W[:, special_cols]):
        raise RuntimeError("special-token columns were modified")
    if not np.isfinite(out).all():
        raise RuntimeError("non-finite weights after GT rescale")
    lowered = (out.min(axis=1).astype(np.float64)
               <= W.min(axis=1).astype(np.float64) + 1e-6)
    if not lowered.all():
        raise RuntimeError(f"{int((~lowered).sum())} rows raised their floor; "
                           "one-sided rule violated")
    sums = np.exp(out.astype(np.float64)).sum(axis=1)
    if np.abs(sums - 1.0).max() > 5e-4:
        raise RuntimeError(f"row normalization broken after rescale "
                           f"(max |sum-1| = {np.abs(sums - 1.0).max():.2e})")
    print(f"gt_min matrix built; exact-GT would have raised {n_raise_capped} rows "
          f"(min binds there)")
    return out


def run(out_dir: str = OUT_DIR):
    import pandas as pd
    os.makedirs(os.path.join(out_dir, "tables"), exist_ok=True)

    weights, langs, lang_to_idx = _load_model_data()
    train_counts = _load_train_counts()
    n_lang = len(langs)
    N = np.array([train_counts.get(l, 0) for l in langs], dtype=np.float64)
    W = np.array(weights, dtype=np.float32)
    del weights
    gt = pd.read_csv(GT_CSV)
    wgt = build_gt_weights(W, gt, langs)

    fp = {"sha256_base_W": hashlib.sha256(W.tobytes()).hexdigest(),
          "sha256_wgt": hashlib.sha256(wgt.tobytes()).hexdigest(),
          "gt_csv_sha256": hashlib.sha256(open(GT_CSV, "rb").read()).hexdigest(),
          "langs_sha256": hashlib.sha256("|".join(langs).encode()).hexdigest(),
          "chunk_lines": CHUNK_LINES, "total_lines": TOTAL_LINES}
    fp_path = os.path.join(SCRATCH_DIR, "fingerprint_gt.json")
    if os.path.exists(fp_path):
        with open(fp_path) as f:
            prev = json.load(f)
        if prev != fp:
            bad = sorted(k for k in fp if prev.get(k) != fp[k])
            raise RuntimeError(f"gt scratch state mismatch ({bad}); clear the gt "
                               f"files in {SCRATCH_DIR} or restore inputs")
    else:
        with open(fp_path + ".tmp", "w") as f:
            json.dump(fp, f)
        os.replace(fp_path + ".tmp", fp_path)

    y_mm = np.lib.format.open_memmap(os.path.join(SCRATCH_DIR, "y_true.npy"), mode="r")
    base_mm = np.lib.format.open_memmap(os.path.join(SCRATCH_DIR, "pred_baseline.npy"),
                                        mode="r")
    for name, arr in (("y_true", y_mm), ("pred_baseline", base_mm)):
        if arr.shape != (TOTAL_LINES,):
            raise RuntimeError(f"{name} memmap has shape {arr.shape}")
    y_all = np.asarray(y_mm)
    if int((y_all == UNSEEN).sum()) != 0:
        raise RuntimeError("y_true memmap contains UNSEEN lines; Exp 16 run incomplete")
    if int((np.asarray(base_mm)[y_all >= 0] == UNSEEN).sum()) != 0:
        raise RuntimeError("pred_baseline has UNSEEN entries on kept lines")

    sample_idx = _sample_line_indices()
    parity_val = (np.arange(DEFAULT_SAMPLE_SIZE) % 2) == 0
    if not np.array_equal(np.load(VAL_MASK), parity_val):
        raise RuntimeError("saved val_mask.npy does not match position-parity split")
    val_lines = set(sample_idx[parity_val].tolist())
    sample_test_lines = sample_idx[~parity_val]
    pickle_y = np.array(load_sample(DEFAULT_SAMPLE_SIZE)["y_true"])[~parity_val]
    expect_label = dict(zip(sample_test_lines.tolist(), pickle_y.tolist()))

    pred_path = os.path.join(SCRATCH_DIR, "pred_gt_min.npy")
    if os.path.exists(pred_path):
        pred_mm = np.lib.format.open_memmap(pred_path, mode="r+")
        if pred_mm.shape != (TOTAL_LINES,):
            raise RuntimeError(f"pred_gt_min memmap has shape {pred_mm.shape}")
    else:
        pred_mm = np.lib.format.open_memmap(pred_path, mode="w+", dtype=np.int16,
                                            shape=(TOTAL_LINES,))
        pred_mm[:] = UNSEEN
        pred_mm.flush()
    progress_path = os.path.join(SCRATCH_DIR, "progress_gt.json")
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
                print("Loading model + caching gt_min weights...", flush=True)
                model = _load_unilid_model()
                model.model.set_weight_sets(wgt.tolist())
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
                batch = model.model.best_of_cached_weight_sets_biased_batch(pre,
                                                                            zero_bias)
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
    predg = np.asarray(pred_mm)
    if int((predg[kept] == UNSEEN).sum()) != 0:
        raise RuntimeError("unprocessed kept lines remain in pred_gt_min")
    yk = y[kept]
    base = np.asarray(base_mm)[kept]
    new = predg[kept]

    diag = pd.read_csv(DIAG_CSV)
    cat = dict(zip(diag["lang"], diag["category"]))
    lang_flags = {
        "overall": np.ones(n_lang, bool),
        "tail": N < 1_000,
        "magnets": np.array([cat.get(l) == "flat_magnet" for l in langs]),
        "twins": np.array([cat.get(l) == "twin" for l in langs]),
        "head": N >= 18_000,
    }
    lines_out = ["# Full-test-set evaluation of the Good-Turing unseen-mass "
                 "candidate (gt_min)\n",
                 f"One new scoring pass under the gt_min matrix ({int(kept.sum()):,} "
                 f"lines); baseline predictions and y_true reused from the Exp 16 "
                 f"run. Single pre-registered candidate, no sweep; adoption judged "
                 f"by analysis/two_sided_report.py.",
                 f"Bootstrap CIs (B=1000) for strata with at most "
                 f"{BOOTSTRAP_MAX_N:,} examples; larger strata report the point "
                 f"delta only.\n",
                 "| stratum | base macroF1 | gt_min macroF1 | delta | 95% CI |",
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
                     f"({acc_n - acc_b:+.4f}).")

    out_path = os.path.join(out_dir, "tables", "full_test_gt.md")
    with open(out_path, "w") as f:
        f.write("\n".join(lines_out))
    print("\n".join(lines_out))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    run()
