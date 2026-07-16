"""Stage 1: post-hoc gated hierarchical shrinkage on the existing 100k model.

Shrinks each per-language distribution toward a prior, with strength gated by the
diagnostic category (analysis/diagnostic.py):
  - flat_magnet   : toward the confuser-excluded, backbone-weighted SCRIPT mean,
                    strength lambda_mag (large; these are net precision liabilities).
  - tight_lowres  : toward its specific high-resource sibling (target_lang) only,
                    strength lambda_tight (small; one-directional).
  - isolated_tail : toward the script mean, strength lambda_tail (mild; fixes floor).
  - twin / head / mid : untouched (lambda = 0).

Group means are built from a frozen backbone (N >= BACKBONE_N) so the same prior
would serve a newly-added language (modularity). A language's near-confusers
(symKL < tau) are excluded from its prior so a magnet is not pulled toward the
very language it steals from (e.g. sco is not pulled toward eng).

Tuning is on the validation half of the seed-42 500k sample; the test half is
scored once. Reports stratified macro-F1 (overall / tail / twins / head / magnets)
with item-level bootstrap CIs on the delta vs baseline.

Needs the custom Rust tokenizer + ~250 GB to cache all 1,940 weight sets -> SLURM.
"""
from __future__ import annotations

import gc
import json
import os

import numpy as np
from scipy.special import logsumexp as sp_logsumexp

from analysis.config import DEFAULT_SAMPLE_SIZE
from analysis.metrics import compute_metrics
from analysis.transfer_sweep import (
    _load_model_data, _load_train_counts, _load_unilid_model,
    _stream_sampled_texts, _probs_to_logprobs, predict_all,
)
from analysis.diagnostic import _probs_and_logprobs, _sym_kl_matrix

DIAG_CSV = "outputs/diagnostic/lang_diagnostic.csv"
VAL_MASK = "outputs/diagnostic/val_mask.npy"
BACKBONE_N = 18_000        # frozen group-mean backbone (top two RESOURCE_BINS)
RES_CAP = 100_000          # resource weight cap min(N, cap)
EXCLUDE_K = 3              # exclude a language's K nearest backbone members (its likely
                           # victims, e.g. sco's nearest is eng) from its own prior, so a
                           # magnet is sharpened AWAY from what it steals from, without
                           # gutting the backbone (the old tau-percentile rule emptied it).
TAIL_N = 1_000
OUT_DIR = "outputs"

# Selection guard (plan item 1, fixed 2026-07-10): a candidate config is eligible only if
# its val overall macro-F1 beats the baseline AND no guarded stratum drops by more than
# GUARD_TOL. The original twins/head-only guard at 0.002 selected a tail-collapsing gamma
# on the Apertus model (Exp 15). A CI-only rule was rejected: the small strata have
# bootstrap CIs wide enough to pass even that collapse. Rule + tolerance choice recorded
# in EXPERIMENTAL_SETUP.md; shared by prior_sweep.py and learned_prior.py.
GUARD_STRATA = ("tail", "magnets", "twins", "head")
GUARD_TOL = 0.01

# Sweep over three method families (all tuned on val, scored once on test):
#   shrink         : magnet/tail rows mixed toward the group mean / target (Stage 1 form)
#   shrink_scaled  : magnet lambda = mag_max * min(1, magnet_ratio/M0) so pure liabilities
#                    (kzn, ratio 67) shrink hard while supported magnets (sco, ratio 4.5) barely move
#   sharpen        : magnet rows raised to power beta (p^beta) to cut flatness and concentrate
#                    each magnet on its OWN peaks, without pulling it toward a foreign mean
SWEEP = [
    {"name": "baseline", "method": "none"},
    {"name": "shrink_mag0.1", "method": "shrink", "mag": 0.1, "tight": 0.0, "tail": 0.0},
    {"name": "shrink_mag0.2", "method": "shrink", "mag": 0.2, "tight": 0.0, "tail": 0.0},
    {"name": "scaled_max0.7_M10", "method": "shrink_scaled", "mag_max": 0.7, "M0": 10.0, "tight": 0.0, "tail": 0.0},
    {"name": "scaled_max0.9_M10", "method": "shrink_scaled", "mag_max": 0.9, "M0": 10.0, "tight": 0.0, "tail": 0.0},
    {"name": "scaled_max0.9_M5",  "method": "shrink_scaled", "mag_max": 0.9, "M0": 5.0,  "tight": 0.0, "tail": 0.0},
    {"name": "sharpen_b1.5",  "method": "sharpen", "beta": 1.5, "ratio_thr": 0.0},
    {"name": "sharpen_b2",    "method": "sharpen", "beta": 2.0, "ratio_thr": 0.0},
    {"name": "sharpen_b3",    "method": "sharpen", "beta": 3.0, "ratio_thr": 0.0},
    {"name": "sharpen_b2_r5", "method": "sharpen", "beta": 2.0, "ratio_thr": 5.0},
    {"name": "sharpen_b3_r5", "method": "sharpen", "beta": 3.0, "ratio_thr": 5.0},
    {"name": "scaled_max0.9_M5+sharpen_off", "method": "shrink_scaled", "mag_max": 0.9, "M0": 5.0, "tight": 0.2, "tail": 0.2},
]


def _build_group_means(N, scripts):
    """Frozen per-script backbone: member indices (N >= BACKBONE_N) and resource
    weights min(N, cap). Returns (group: script -> member_idx array, w: weights)."""
    backbone = N >= BACKBONE_N
    w = np.minimum(N, RES_CAP) * backbone  # 0 weight for non-backbone
    group = {s: np.where((scripts == s) & backbone)[0] for s in np.unique(scripts)}
    return group, w


def _prior_for(L_idx, P, scripts, sym, group, w):
    """Backbone-weighted script mean for language L, excluding L itself (LOO) and its
    EXCLUDE_K nearest backbone members (likely victims). Falls back to all-but-self if
    that empties the backbone; returns None only if the script has no backbone."""
    members = group.get(scripts[L_idx])
    if members is None or len(members) == 0:
        return None
    excl = {L_idx}
    order = members[np.argsort(sym[L_idx, members])]
    for j in order:
        if j == L_idx:
            continue
        excl.add(int(j))
        if len(excl) > EXCLUDE_K:  # self + K victims
            break
    keep = np.array([j for j in members if j not in excl])
    if len(keep) == 0:  # tiny backbone -> fall back to all-but-self
        keep = np.array([j for j in members if j != L_idx])
    if len(keep) == 0:
        return None
    ww = w[keep]
    if ww.sum() <= 0:
        return None
    return (ww[:, None] * P[keep]).sum(axis=0) / ww.sum()


def build_modified_weights(weights, langs, lang_to_idx, N, scripts, P, sym, diag, cfg):
    """Return a modified float32 weight matrix for one sweep config (dispatch on method)."""
    out = np.array(weights, dtype=np.float32)
    if cfg["method"] == "none":
        return out
    cat = dict(zip(diag["lang"], diag["category"]))
    tgt = dict(zip(diag["lang"], diag["target_lang"]))
    mratio = dict(zip(diag["lang"], diag["magnet_ratio"]))
    group, w = _build_group_means(N, scripts)
    n_mod, skipped = 0, []
    for L in langs:
        i = lang_to_idx[L]
        c = cat.get(L, "mid")
        if cfg["method"] == "sharpen":
            # concentrate a magnet on its own peaks: log p' = beta*log p - logsumexp(beta*log p)
            if c != "flat_magnet" or mratio.get(L, 0.0) < cfg.get("ratio_thr", 0.0):
                continue
            row = out[i].astype(np.float64) * cfg["beta"]
            out[i] = (row - sp_logsumexp(row)).astype(np.float32)
            n_mod += 1
            continue
        # shrink / shrink_scaled: mix toward a prior in probability space
        if c in ("flat_magnet", "isolated_tail"):
            if c == "flat_magnet":
                lam_L = (cfg["mag"] if cfg["method"] == "shrink"
                         else cfg["mag_max"] * min(1.0, mratio.get(L, 0.0) / cfg["M0"]))
            else:
                lam_L = cfg.get("tail", 0.0)
            prior = _prior_for(i, P, scripts, sym, group, w)
        elif c == "tight_lowres":
            lam_L = cfg.get("tight", 0.0)
            t = tgt.get(L)
            prior = P[lang_to_idx[t]] if (isinstance(t, str) and t in lang_to_idx) else None
        else:
            continue
        if lam_L <= 0:
            continue
        if prior is None:
            if c == "flat_magnet":
                skipped.append(L)
            continue
        p_new = (1.0 - lam_L) * P[i] + lam_L * prior
        p_new /= p_new.sum()
        out[i] = _probs_to_logprobs(p_new).astype(np.float32)
        n_mod += 1
    msg = f"    [{cfg['name']}] modified {n_mod} rows"
    if skipped:
        msg += f"; skipped {len(skipped)} magnets (no same-script backbone)"
    print(msg)
    return out


def _strata(y_true, langs, lang_to_idx, N, diag):
    cat = dict(zip(diag["lang"], diag["category"]))
    Narr = {l: N[lang_to_idx[l]] for l in langs}
    yt = np.asarray(y_true)
    twin_langs = set(diag.loc[diag.category == "twin", "lang"])
    magnet_langs = set(diag.loc[diag.category == "flat_magnet", "lang"])
    tail = np.array([Narr.get(l, 0) < TAIL_N for l in yt])
    head = np.array([Narr.get(l, 0) >= BACKBONE_N for l in yt])
    twins = np.array([l in twin_langs for l in yt])
    magnets = np.array([l in magnet_langs for l in yt])
    return {"overall": np.ones(len(yt), bool), "tail": tail, "head": head,
            "twins": twins, "magnets": magnets}


def _macro_f1(y_true, y_pred, mask):
    if mask.sum() == 0:
        return float("nan")
    return compute_metrics(np.asarray(y_true)[mask], np.asarray(y_pred)[mask])["macro_f1"]


def passes_guard(cand, base, prefix=""):
    """Guard for val-based selection: cand and base are per-stratum macro-F1 dicts keyed
    f"{prefix}{stratum}". Eligible iff val overall improves and no guarded stratum drops
    by more than GUARD_TOL."""
    if cand[f"{prefix}overall"] <= base[f"{prefix}overall"]:
        return False
    return all(cand[f"{prefix}{st}"] >= base[f"{prefix}{st}"] - GUARD_TOL
               for st in GUARD_STRATA)


def _bootstrap_delta(y_true, y_base, y_new, mask, B=1000, seed=0):
    """Bootstrap 95% CI of macro-F1(new) - macro-F1(base) within mask."""
    rng = np.random.default_rng(seed)
    yt = np.asarray(y_true)[mask]; yb = np.asarray(y_base)[mask]; yn = np.asarray(y_new)[mask]
    n = len(yt)
    if n == 0:
        return (float("nan"), float("nan"), float("nan"))
    deltas = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, n, n)
        deltas[b] = (compute_metrics(yt[idx], yn[idx])["macro_f1"]
                     - compute_metrics(yt[idx], yb[idx])["macro_f1"])
    return (float(np.mean(deltas)), float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5)))


def run(sample_size: int = DEFAULT_SAMPLE_SIZE, out_dir: str = OUT_DIR):
    import pandas as pd
    os.makedirs(os.path.join(out_dir, "tables"), exist_ok=True)
    print("Loading model + diagnostic...")
    weights, langs, lang_to_idx = _load_model_data()
    train_counts = _load_train_counts()
    N = np.array([train_counts.get(l, 0) for l in langs], dtype=np.float64)
    scripts = np.array([l.rsplit("_", 1)[-1] if "_" in l else "Unknown" for l in langs])
    diag = pd.read_csv(DIAG_CSV)

    print("Computing probabilities + symmetric KL...")
    P, logP = _probs_and_logprobs(weights)
    sym, _H = _sym_kl_matrix(P, logP)
    del logP; gc.collect()

    print("Streaming sample texts + loading split...")
    texts = _stream_sampled_texts(sample_size)
    from analysis.sample_data import load_sample
    data = load_sample(sample_size)
    y_true = np.array(data["y_true"])
    if len(texts) != len(y_true):
        raise RuntimeError(f"text/y_true mismatch {len(texts)} vs {len(y_true)}")
    val = np.load(VAL_MASK)
    test = ~val
    strata = _strata(y_true, langs, lang_to_idx, N, diag)

    model = _load_unilid_model()

    # baseline predictions (full sample once)
    print("Baseline prediction...")
    base_pred = np.array(predict_all(texts, model))
    # sanity: raw rescore must reproduce the recorded UniLID predictions closely
    agree = (base_pred == np.array(data["pred_UniLID"])).mean()
    print(f"  baseline agreement with recorded UniLID preds: {agree:.4f}")

    def strat_row(pred, mask_split):
        return {st: _macro_f1(y_true[mask_split], pred[mask_split], strata[st][mask_split])
                for st in strata}

    val_base = strat_row(base_pred, val)
    rows = [{"config": "baseline", **{f"val_{k}": v for k, v in val_base.items()}}]
    preds_by_cfg = {"baseline": base_pred}

    for cfg in SWEEP:
        if cfg["method"] == "none":
            continue
        print(f"Config {cfg['name']}...")
        w = build_modified_weights(weights, langs, lang_to_idx, N, scripts, P, sym, diag, cfg)
        model.model.set_weight_sets(w.tolist())
        del w; gc.collect()
        pred = np.array(predict_all(texts, model))
        preds_by_cfg[cfg["name"]] = pred
        vr = strat_row(pred, val)
        rows.append({"config": cfg["name"], **{f"val_{k}": v for k, v in vr.items()}})
        print("    val macro-F1:", {k: round(v, 4) for k, v in vr.items()})

    df = pd.DataFrame(rows)
    # select best config by val overall macro-F1, guarded on all strata (passes_guard)
    eligible = [r for r in rows if r["config"] != "baseline"
                and passes_guard(r, rows[0], prefix="val_")]
    best = max(eligible, key=lambda r: r["val_overall"])["config"] if eligible else "baseline"
    print(f"\nBest config (val, guarded): {best}")

    # final TEST evaluation, once, with bootstrap CIs vs baseline
    best_pred = preds_by_cfg[best]
    test_lines = ["# Stage 1 hierarchical shrinkage — TEST evaluation\n",
                  f"Best config selected on val: **{best}**. Baseline agreement {agree:.4f}.",
                  f"Selection guard: val overall must improve and no stratum "
                  f"({'/'.join(GUARD_STRATA)}) may drop more than {GUARD_TOL} vs baseline.\n",
                  "| stratum | base macroF1 | new macroF1 | delta | 95% CI |",
                  "|---|---|---|---|---|"]
    for st in ["overall", "magnets", "tail", "twins", "head"]:
        b = _macro_f1(y_true[test], base_pred[test], strata[st][test])
        nw = _macro_f1(y_true[test], best_pred[test], strata[st][test])
        d, lo, hi = _bootstrap_delta(y_true[test], base_pred[test], best_pred[test], strata[st][test])
        test_lines.append(f"| {st} | {b:.4f} | {nw:.4f} | {d:+.4f} | [{lo:+.4f}, {hi:+.4f}] |")
    md = df.to_markdown(index=False) + "\n\n" + "\n".join(test_lines)
    out_path = os.path.join(out_dir, "tables", "hierarchical_pool.md")
    with open(out_path, "w") as f:
        f.write(md)
    print("\n" + "\n".join(test_lines))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    run()
