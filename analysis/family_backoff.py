"""Group-mean back-off at unseen (floor) positions (plan item 12, added 2026-07-18).

Measured structure: every language's weight row is an exact floor plateau over 82-97% of
the vocabulary (one per-language constant at the row minimum; e.g. eng 82,112 entries at
exactly -18.9728, anp 96,774 at exactly -16.7217), and the floor level is resource-tied
(Exp 10: corr(floor, log10 N) = -0.966). This is the mechanism by which low-resource
languages under-penalize unseen material. Back-off replaces the flat floor with a
group-informed profile, touching ONLY floor positions; every observed-token estimate is
bit-identical, which is the difference from the refuted Exp 13 shrinkage (that mixed
observed evidence toward the mean).

Group prior: the confuser-excluded, resource-weighted script backbone mean from
hierarchical_pool (`_prior_for`), i.e. a script-level proxy for a language family; a
genealogical grouping (e.g. Glottolog families) is a possible refinement, not wired in.
Back-off weight is data-dependent in the standard Jelinek-Mercer form
`lam_L = alpha / (N_L + alpha)` with one global alpha (swept).

Two modes at floor positions t of language L (m_G = group mean, f_L = floor value):
  lift : p' = max(f_L, lam_L * m_G(t))   raises plausible-in-group tokens only
  full : p' = lam_L * m_G(t)             also lowers out-of-group tokens below the floor

No renormalization (the scorer sums unnormalized log-weights; renormalizing injects a
per-language per-token offset, see the token-tying review note 2026-07-18). Special
tokens are never floor entries (each sits at p=0.2) and are asserted unchanged.
Languages whose script has no backbone are left unmodified and counted. Selection on the
val half via the all-strata guard; test half scored once.
"""
from __future__ import annotations

import csv
import gc
import os
import re

import numpy as np

from analysis.config import DEFAULT_SAMPLE_SIZE
from analysis.transfer_sweep import (
    _load_model_data, _load_train_counts, _load_unilid_model, _stream_sampled_texts,
    predict_all,
)
from analysis.sample_data import load_sample
from analysis.diagnostic import _probs_and_logprobs, _sym_kl_matrix
from analysis.hierarchical_pool import (
    _build_group_means, _prior_for, _strata, _macro_f1, _bootstrap_delta, passes_guard,
    GUARD_STRATA, GUARD_TOL, VAL_MASK, DIAG_CSV,
)

ALPHAS = [300.0, 3000.0, 30000.0]   # lam_L = alpha/(N_L+alpha); N=100k -> lam 0.003-0.23,
                                    # N=500 -> lam 0.375-0.98
MODES = ["lift", "full"]
SPECIAL_P = 0.2
OUT_DIR = "outputs"
# Genealogical grouping (added 2026-07-18, user request for true families): WALS export
# copied from ~/tokenizer-lm/data (provenance in data/README.md). Groups are always
# WITHIN script (a raw family mean would mix scripts and lift a language's unseen mass
# on other scripts' tokens). Tiered fallback per language: genus-within-script, then
# family-within-script (each only if the group has >= MIN_BACKBONE_GROUP backbone
# members), then script. WALS covers 1,159/1,940 languages; the rest use script.
WALS_CSV = "data/wals_languages.csv"
MIN_BACKBONE_GROUP = 3
GROUPINGS = ["script", "wals"]


def build_backoff_weights(W: np.ndarray, priors: list, N: np.ndarray,
                          alpha: float, mode: str) -> tuple[np.ndarray, int]:
    """Apply back-off at each language's exact floor plateau. priors[i] is the group
    mean for language i (None = no backbone; row left unmodified)."""
    out = np.array(W, dtype=np.float32)
    n_skipped = 0
    for i in range(W.shape[0]):
        m_g = priors[i]
        if m_g is None:
            n_skipped += 1
            continue
        row = W[i]
        floor_mask = row == row.min()
        lam = alpha / (N[i] + alpha)
        backed = lam * m_g[floor_mask].astype(np.float64)
        if mode == "lift":
            backed = np.maximum(backed, np.exp(np.float64(row.min())))
        elif mode != "full":
            raise ValueError(f"unknown mode {mode!r}")
        out[i, floor_mask] = np.log(backed).astype(np.float32)
    if not np.isfinite(out).all():
        raise RuntimeError("non-finite weights after back-off")
    return out, n_skipped


def _load_wals_tables(path: str = WALS_CSV) -> tuple[dict, dict]:
    """ISO 639-3 -> WALS Family and Genus (first entry wins on duplicates)."""
    fam, gen = {}, {}
    with open(path) as f:
        for row in csv.DictReader(f):
            codes = set(filter(None, re.split(r"[ ;,]+", row.get("ISO_codes") or "")))
            if row.get("ISO639P3code"):
                codes.add(row["ISO639P3code"])
            for c in codes:
                if row.get("Family"):
                    fam.setdefault(c, row["Family"])
                if row.get("Genus"):
                    gen.setdefault(c, row["Genus"])
    if not fam:
        raise RuntimeError(f"no Family entries parsed from {path}")
    return fam, gen


def _tiered_priors(langs, scripts, N, P, sym, grouping):
    """Per-language back-off prior and the tier it came from. 'script' is the
    single-level script grouping; 'wals' tries genus-within-script, then
    family-within-script (each only if the group has >= MIN_BACKBONE_GROUP backbone
    members), then script. Membership for a language's group is evaluated at that
    language's own tier, so a script-tier language still gets the FULL script backbone
    mean (groups are nested/overlapping, as in hierarchical smoothing)."""
    n = len(langs)
    levels = []
    if grouping == "wals":
        fam, gen = _load_wals_tables()
        iso = [l.rsplit("_", 1)[0] for l in langs]
        gen_lab = np.array([f"{gen[i]}|{s}" if i in gen else "__none__"
                            for i, s in zip(iso, scripts)], dtype=object)
        fam_lab = np.array([f"{fam[i]}|{s}" if i in fam else "__none__"
                            for i, s in zip(iso, scripts)], dtype=object)
        levels = [("genus_script", gen_lab), ("family_script", fam_lab)]
    elif grouping != "script":
        raise ValueError(f"unknown grouping {grouping!r}")
    levels.append(("script", np.asarray(scripts, dtype=object)))

    built = [(name, labels, *_build_group_means(N, labels)) for name, labels in levels]

    priors, tiers = [], []
    for i in range(n):
        prior, tier = None, "none"
        for name, labels, group, w in built:
            lab = labels[i]
            if lab == "__none__":
                continue
            members = group.get(lab)
            if members is None or (name != "script" and len(members) < MIN_BACKBONE_GROUP):
                continue
            p_i = _prior_for(i, P, labels, sym, group, w)
            if p_i is not None:
                prior, tier = p_i, name
                break
        priors.append(prior)
        tiers.append(tier)
    return priors, tiers


def run(sample_size: int = DEFAULT_SAMPLE_SIZE, out_dir: str = OUT_DIR,
        groupings: tuple = ("script",), tag: str = ""):
    import pandas as pd
    os.makedirs(os.path.join(out_dir, "tables"), exist_ok=True)
    weights, langs, lang_to_idx = _load_model_data()
    train_counts = _load_train_counts()
    N = np.array([train_counts.get(l, 0) for l in langs], dtype=np.float64)
    scripts = np.array([l.rsplit("_", 1)[-1] if "_" in l else "Unknown" for l in langs])
    W = np.array(weights, dtype=np.float32)
    diag = pd.read_csv(DIAG_CSV)

    print("Computing probabilities + symmetric KL for the group priors...")
    P, logP = _probs_and_logprobs(W)
    sym, _H = _sym_kl_matrix(P, logP)
    del logP
    gc.collect()
    from collections import Counter
    os.makedirs(os.path.join(out_dir, "diagnostic"), exist_ok=True)
    priors_by_grouping, tier_lines = {}, []
    for grouping in groupings:
        priors, tiers = _tiered_priors(langs, scripts, N, P, sym, grouping)
        n_none = sum(p is None for p in priors)
        tc = dict(Counter(tiers))
        line = f"grouping={grouping}: tiers {tc}; {n_none} languages unmodified"
        print(line)
        tier_lines.append(line)
        priors_by_grouping[grouping] = (priors, n_none)
        pd.DataFrame({"lang": langs, "tier": tiers}).to_csv(
            os.path.join(out_dir, "diagnostic", f"backoff_groups_{grouping}.csv"),
            index=False)

    # special tokens sit at p=0.2 in every row and must never be floor entries
    special_cols = np.where(np.all(np.abs(np.exp(W.astype(np.float64))
                                          - SPECIAL_P) < 1e-4, axis=0))[0]
    if len(special_cols) != 4:
        raise RuntimeError(f"expected 4 special columns at p={SPECIAL_P}, "
                           f"found {len(special_cols)}")

    texts = _stream_sampled_texts(sample_size)
    data = load_sample(sample_size)
    y_true = np.array(data["y_true"])
    if len(texts) != len(y_true):
        raise RuntimeError(f"text/y_true mismatch {len(texts)} vs {len(y_true)}")
    val = np.load(VAL_MASK)
    test = ~val
    strata = _strata(y_true, langs, lang_to_idx, N, diag)

    model = _load_unilid_model()
    print("Baseline prediction...")
    base_pred = np.array(predict_all(texts, model))
    agree = (base_pred == np.array(data["pred_UniLID"])).mean()
    print(f"  baseline agreement with recorded UniLID preds: {agree:.4f}")
    if agree < 0.99:
        raise RuntimeError(f"baseline agreement {agree:.4f} < 0.99; model/scorer changed")

    def strat_row(pred, mask_split):
        return {st: _macro_f1(y_true[mask_split], pred[mask_split], strata[st][mask_split])
                for st in strata}

    val_base = strat_row(base_pred, val)
    rows = [{"config": "baseline", **{f"val_{k}": v for k, v in val_base.items()}}]
    preds_by_cfg = {"baseline": base_pred}
    for grouping in groupings:
        priors, n_none = priors_by_grouping[grouping]
        for mode in MODES:
            for alpha in ALPHAS:
                cfg = f"{grouping}_{mode}_a{alpha:g}"
                print(f"Config {cfg}...")
                w_new, n_skip = build_backoff_weights(W, priors, N, alpha, mode)
                if n_skip != n_none:
                    raise RuntimeError(f"skipped {n_skip} rows, expected {n_none}")
                if not np.array_equal(w_new[:, special_cols], W[:, special_cols]):
                    raise RuntimeError("special-token columns were modified by back-off")
                model.model.set_weight_sets(w_new.tolist())
                del w_new
                gc.collect()
                pred = np.array(predict_all(texts, model))
                preds_by_cfg[cfg] = pred
                vr = strat_row(pred, val)
                rows.append({"config": cfg, **{f"val_{k}": v for k, v in vr.items()}})
                print("    val macro-F1:", {k: round(v, 4) for k, v in vr.items()})

    eligible = [r for r in rows if r["config"] != "baseline"
                and passes_guard(r, rows[0], prefix="val_")]
    best = max(eligible, key=lambda r: r["val_overall"])["config"] if eligible else "baseline"
    print(f"\nBest config (val, guarded): {best}")

    best_pred = preds_by_cfg[best]
    lines = ["# Group-mean back-off at floor positions — TEST evaluation\n",
             f"Best config selected on val: **{best}** (baseline means nothing passed the "
             f"guard). Baseline agreement {agree:.4f}.",
             *tier_lines,
             f"lam_L = alpha/(N_L+alpha); modes: lift = raise floor entries only, "
             f"full = replace floor entries in both directions.",
             f"Selection guard: val overall must improve and no stratum "
             f"({'/'.join(GUARD_STRATA)}) may drop more than {GUARD_TOL} vs baseline.\n",
             "| stratum | base macroF1 | backoff macroF1 | delta | 95% CI |",
             "|---|---|---|---|---|"]
    for st in ["overall", "magnets", "tail", "twins", "head"]:
        b = _macro_f1(y_true[test], base_pred[test], strata[st][test])
        nw = _macro_f1(y_true[test], best_pred[test], strata[st][test])
        d, lo, hi = _bootstrap_delta(y_true[test], base_pred[test], best_pred[test],
                                     strata[st][test])
        lines.append(f"| {st} | {b:.4f} | {nw:.4f} | {d:+.4f} | [{lo:+.4f}, {hi:+.4f}] |")
    acc_b = (base_pred[test] == y_true[test]).mean()
    acc_n = (best_pred[test] == y_true[test]).mean()
    lines.append(f"\nOverall accuracy: {acc_b:.4f} -> {acc_n:.4f} ({acc_n-acc_b:+.4f}).")

    md = pd.DataFrame(rows).round(4).to_markdown(index=False) + "\n\n" + "\n".join(lines)
    out_path = os.path.join(out_dir, "tables", f"family_backoff{tag}.md")
    with open(out_path, "w") as f:
        f.write(md)
    print("\n" + "\n".join(lines))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    run()
