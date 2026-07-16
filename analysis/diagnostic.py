"""Per-language diagnostic for hierarchical-pooling design.

Classifies all 1,940 languages into {flat_magnet, tight_lowres, twin,
isolated_tail, head, mid} from structural properties of the weight matrix
(entropy/flatness, full pairwise symmetric-KL, train count) plus an empirical
magnet signal computed on a held-out validation half (never the test half).

The output table (outputs/diagnostic/lang_diagnostic.csv) drives the gated
shrinkage in analysis/hierarchical_pool.py:
  - flat_magnet   : abnormally flat, promiscuous, low-resource, net false-positive
                    liability -> shrink HARD toward the group mean (strict win).
  - tight_lowres  : low-resource member sitting very close to a much higher-
                    resource same-script sibling -> shrink GENTLY toward that
                    sibling only (one-directional).
  - twin          : very close to a comparable-resource neighbour -> DO NOT pool
                    (blur risk).
  - isolated_tail : low-resource, far from any neighbour -> MILD shrink toward
                    the script mean (fixes the resource-tied floor).
  - head / mid    : protected (lambda = 0).

Structural features need no predictions and no test data. Only the empirical
magnet ratio touches predictions, and it uses the validation half exclusively.
"""
from __future__ import annotations

import os

import numpy as np

from analysis.sample_data import load_sample
from analysis.transfer_sweep import _load_model_data, _load_train_counts

# --- constants (documented, not magic) -------------------------------------
HEAD_N = 18_000          # top two RESOURCE_BINS define the well-fit "head"/backbone
TAIL_N = 1_000           # "tail" resource ceiling
TIGHT_N = 5_000          # tight-pair low-resource member ceiling
TIGHT_RESRATIO = 0.1     # sibling must be >=10x richer
TWIN_LOGN_GAP = 0.3      # |log10 N_L - log10 N_nn| < 0.3 ~ within 2x resource
KCLOSE_QUANTILE = 0.05   # "close" radius = 5th pct of off-diagonal symKL
DNN_LO_Q = 0.10          # twin / tight nearest-neighbour distance low cutoff
ZH_MAGNET = 1.5          # flatness: >1.5 robust SDs above the within-script median entropy
ZH_EXTREME = 5.0         # extreme flatness -> magnet regardless of val activity
MAGNET_RATIO_MIN = 2.0   # net liability: >2x as many false positives as true support
OUT_DIR = "outputs/diagnostic"


def _probs_and_logprobs(weights: np.ndarray):
    """Row-wise stable softmax of the log-score matrix -> (P, logP) float32."""
    W = np.asarray(weights, dtype=np.float32)
    rowmax = W.max(axis=1, keepdims=True)
    shifted = W - rowmax
    e = np.exp(shifted)
    Z = e.sum(axis=1, keepdims=True)
    P = e / Z
    logP = shifted - np.log(Z)
    del e, shifted, W
    return P.astype(np.float32), logP.astype(np.float32)


def _sym_kl_matrix(P: np.ndarray, logP: np.ndarray) -> np.ndarray:
    """Full pairwise symmetric KL via one matmul. symKL[i,j] = 0.5(KL(i||j)+KL(j||i))."""
    neg_entropy = (P * logP).sum(axis=1)             # = -H_i  (n,)
    cross = P @ logP.T                                # cross[i,j] = sum_t P_i logP_j
    kl = neg_entropy[:, None] - cross                # KL(i||j) >= 0
    del cross
    sym = 0.5 * (kl + kl.T)
    np.fill_diagonal(sym, 0.0)
    return sym, -neg_entropy                          # (symKL, entropy H)


def _val_test_masks(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic 50/50 split of the seed-42 500k sample by position parity."""
    pos = np.arange(n)
    val = (pos % 2) == 0
    test = ~val
    return val, test


def _empirical_magnet(langs, val_y_true, val_pred):
    """Per-language false-positive count, support, and magnet ratio on val half."""
    from collections import Counter
    yt = np.asarray(val_y_true)
    yp = np.asarray(val_pred)
    true_c = Counter(yt.tolist())
    pred_c = Counter(yp.tolist())
    tp_c = Counter(yt[yt == yp].tolist())
    S = {l: true_c.get(l, 0) for l in langs}
    FP = {l: pred_c.get(l, 0) - tp_c.get(l, 0) for l in langs}
    ratio = {l: FP[l] / (S[l] + 1.0) for l in langs}
    return S, FP, ratio


def build_diagnostic(sample_size: int = 500_000, out_dir: str = OUT_DIR) -> str:
    os.makedirs(out_dir, exist_ok=True)
    print("Loading model weights...")
    weights, langs, lang_to_idx = _load_model_data()
    train_counts = _load_train_counts()
    n = len(langs)
    scripts = np.array([l.rsplit("_", 1)[-1] if "_" in l else "Unknown" for l in langs])
    N = np.array([train_counts.get(l, 0) for l in langs], dtype=np.float64)
    logN = np.log10(N + 1.0)

    print("Computing probabilities / entropy / pairwise symmetric KL...")
    P, logP = _probs_and_logprobs(weights)
    del weights
    sym, H = _sym_kl_matrix(P, logP)
    peak = P.max(axis=1)
    del P, logP

    # within-script flatness z-score (robust: median / MAD)
    zH = np.zeros(n)
    for s in np.unique(scripts):
        m = scripts == s
        if m.sum() < 3:
            continue
        med = np.median(H[m])
        mad = np.median(np.abs(H[m] - med)) * 1.4826 + 1e-9
        zH[m] = (H[m] - med) / mad

    # nearest-neighbour distance, promiscuity radius, higher-resource same-script target
    off = sym.copy()
    np.fill_diagonal(off, np.inf)
    nn_idx = off.argmin(axis=1)
    d_nn = off[np.arange(n), nn_idx]
    tau = np.quantile(off[np.isfinite(off)], KCLOSE_QUANTILE)
    k_close = (off < tau).sum(axis=1)

    d_up = np.full(n, np.inf)
    up_idx = np.full(n, -1)
    for i in range(n):
        same = (scripts == scripts[i]) & (N > N[i])
        same[i] = False
        if same.any():
            cand = np.where(same)[0]
            j = cand[off[i, cand].argmin()]
            d_up[i] = off[i, j]
            up_idx[i] = j
    up_resratio = np.where(up_idx >= 0, N / np.maximum(N[up_idx], 1), np.nan)

    # empirical magnet signal on the validation half only
    print(f"Building {sample_size//1000}k sample for empirical magnet signal...")
    data = load_sample(sample_size)
    yt = data["y_true"]
    yp = data["pred_UniLID"]
    val, _test = _val_test_masks(len(yt))
    S, FP, ratio = _empirical_magnet(langs, np.array(yt)[val], np.array(yp)[val])
    support = np.array([S[l] for l in langs])
    fp = np.array([FP[l] for l in langs])
    magnet_ratio = np.array([ratio[l] for l in langs])
    np.save(os.path.join(out_dir, "val_mask.npy"), val)

    # data-driven thresholds for twin / tight detection
    dnn_lo = np.quantile(d_nn, DNN_LO_Q)
    dup_lo = np.quantile(d_up[np.isfinite(d_up)], DNN_LO_Q)

    # classification (priority order)
    category = np.array(["mid"] * n, dtype=object)
    logN_nn = logN[nn_idx]
    for i in range(n):
        # A flat magnet is FLAT (high within-script entropy z-score) AND an empirical
        # net liability (magnet_ratio: more false positives than true support, from
        # the val half only), or structurally extreme-flat regardless of val activity.
        # Flatness is what separates a magnet from a twin: twins (ind/zsm) are sharp
        # distributions sitting close to one sibling (zH ~ 1), magnets (kzn/tly/qus)
        # are diffuse (zH 2-12). Resource tier is intentionally NOT a gate, so
        # high-resource flat magnets (e.g. qus) are caught. k_close is reported but
        # not used here (it is high for any dense-cluster member, including twins).
        is_magnet = ((zH[i] > ZH_MAGNET) and (magnet_ratio[i] > MAGNET_RATIO_MIN)) or (zH[i] > ZH_EXTREME)
        if is_magnet:
            category[i] = "flat_magnet"
        elif (d_nn[i] < dnn_lo) and (abs(logN[i] - logN_nn[i]) < TWIN_LOGN_GAP):
            category[i] = "twin"
        elif (N[i] < TIGHT_N) and (d_up[i] < dup_lo) and (up_resratio[i] < TIGHT_RESRATIO):
            category[i] = "tight_lowres"
        elif N[i] < TAIL_N:
            category[i] = "isolated_tail"
        elif N[i] >= HEAD_N:
            category[i] = "head"

    rows = []
    for i, l in enumerate(langs):
        rows.append(dict(
            lang=l, script=scripts[i], N=int(N[i]), logN=round(float(logN[i]), 3),
            H=round(float(H[i]), 4), zH=round(float(zH[i]), 3), peak=round(float(peak[i]), 4),
            k_close=int(k_close[i]), d_nn=round(float(d_nn[i]), 4), nn_lang=langs[nn_idx[i]],
            d_up=round(float(d_up[i]), 4) if np.isfinite(d_up[i]) else None,
            target_lang=(langs[up_idx[i]] if up_idx[i] >= 0 else None),
            up_resratio=round(float(up_resratio[i]), 4) if up_idx[i] >= 0 else None,
            support_val=int(support[i]), fp_val=int(fp[i]),
            magnet_ratio=round(float(magnet_ratio[i]), 3), category=str(category[i]),
        ))
    import pandas as pd
    df = pd.DataFrame(rows)
    csv_path = os.path.join(out_dir, "lang_diagnostic.csv")
    df.to_csv(csv_path, index=False)

    # markdown summary
    cat_counts = df["category"].value_counts().to_dict()
    lines = ["# Per-language diagnostic\n",
             f"Languages: {n}. Magnet: zH>{ZH_MAGNET} & magnet_ratio>{MAGNET_RATIO_MIN} "
             f"(or zH>{ZH_EXTREME}). Twin/tight: dnn_lo={dnn_lo:.3f}, dup_lo={dup_lo:.3f}, "
             f"tau(kclose radius)={tau:.3f}.\n",
             "## Category counts\n",
             *[f"- {k}: {v}" for k, v in sorted(cat_counts.items(), key=lambda x: -x[1])],
             "\n## Top 25 flat magnets by val false-positives\n"]
    mag = df[df.category == "flat_magnet"].sort_values("fp_val", ascending=False).head(25)
    lines.append(mag[["lang", "N", "zH", "k_close", "support_val", "fp_val", "magnet_ratio"]].to_markdown(index=False))
    lines.append("\n## Sample tight_lowres (shrink toward target)\n")
    tig = df[df.category == "tight_lowres"].sort_values("up_resratio").head(20)
    lines.append(tig[["lang", "N", "d_up", "target_lang", "up_resratio"]].to_markdown(index=False))
    lines.append("\n## Sample twins (do NOT pool)\n")
    tw = df[df.category == "twin"].sort_values("d_nn").head(20)
    lines.append(tw[["lang", "N", "d_nn", "nn_lang"]].to_markdown(index=False))
    md_path = os.path.join(out_dir, "lang_diagnostic.md")
    with open(md_path, "w") as f:
        f.write("\n".join(lines))

    # sanity check against known cases from the error analysis
    print("\n=== category counts ===")
    for k, v in sorted(cat_counts.items(), key=lambda x: -x[1]):
        print(f"  {k:14} {v}")
    print("\n=== sanity: known flat magnets ===")
    for code in ["kzn_Latn", "tly_Latn", "vol_Latn", "ido_Latn", "mlt_Latn", "qus_Latn", "sbs_Latn"]:
        r = df[df.lang == code]
        if len(r):
            r = r.iloc[0]
            print(f"  {code:10} cat={r.category:13} zH={r.zH:6.2f} k_close={r.k_close:4d} fp_val={r.fp_val:4d} magnet_ratio={r.magnet_ratio:.2f}")
    print("\n=== sanity: known twins / siblings ===")
    for code in ["ind_Latn", "zsm_Latn", "kin_Latn", "run_Latn", "dan_Latn", "nob_Latn", "anp_Deva", "glk_Arab"]:
        r = df[df.lang == code]
        if len(r):
            r = r.iloc[0]
            print(f"  {code:10} cat={r.category:13} N={r.N:7d} d_nn={r.d_nn:6.3f} nn={r.nn_lang:10} d_up={str(r.d_up):>7} target={r.target_lang}")
    print(f"\nWrote {csv_path} and {md_path}")
    return csv_path


if __name__ == "__main__":
    build_diagnostic()
