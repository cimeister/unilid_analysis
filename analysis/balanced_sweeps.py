"""First sweeps under the balanced validation protocol (Exp 22): three experiments.

Shared harness: selection metrics are stratified macro-F1 on the seed-101 balanced val
(188,061 texts, all 1,940 languages; `analysis/balanced_split.py`), guarded by
`passes_guard`. The zero-modification baseline is validated against the saved
`pred_baseline.npy` memmap at the balanced-val lines (agreement gate >= 0.99). Sweeps
only select; any guard-passing config still requires its own full-test pass before
becoming a result of record.

1. run_floor: floor equalization re-selection (Exp 20 follow-up). Same F grid; the
   balanced val can now see the tail, so the guard verdicts are meaningful at
   selection time.
2. run_punct_prior: plan item 15. Partial pooling of ONLY the 212 neutral
   digit/punctuation columns toward the within-script mean:
       p' = (1 - lam_L) * p + lam_L * m_script,   lam_L = alpha / (N_L + alpha).
   Head/twin conventions (N=100k, lam ~ 0) keep their signal in probability space (the
   refuted full tie is lam = 1); low-N estimates are stabilized. Note the blend is in
   probability space, so even at lam ~ 0 a head language's floor-level dp columns can
   move by a few nats in log space where the script mean is much larger; that is the
   intended neutralization of rarely-used punctuation, not leakage. Equivalent to a
   script-level hierarchical prior on punctuation rates, MAP-estimated per language.
   Modular.
3. run_bias_refit: learned per-language bias refit ON BALANCED DATA. Fitting on a
   language-balanced sample removes the traffic-prior component of the objective, so
   the fitted b isolates attractor suppression. Per-language alternating split of the
   balanced val into a fit half and a selection half (both language-balanced); plain
   L2 (centering on a frequency prior would contradict the uniform-prior objective);
   corrected NLL gradient from learned_prior. Reports the most negative offsets
   against the diagnosed magnet list. Refit-per-draw stability (protocol caveat 3) is
   a follow-up once the draw-101 outcome is known.
"""
from __future__ import annotations

import os
import pickle

import numpy as np

from analysis.metrics import compute_metrics
from analysis.transfer_sweep import (
    _load_model_data, _load_train_counts, _load_unilid_model, predict_all,
)
from analysis.hierarchical_pool import (
    RES_CAP, passes_guard, passes_shortlist, GUARD_STRATA, GUARD_TOL,
    TAIL_RECALL_TOL, DIAG_CSV,
)
from analysis.full_test_eval import SCRATCH_DIR
from analysis.balanced_split import TEXT_CACHE
from analysis.floor_equalization import build_equalized_weights, FLOORS
from analysis.token_tying import (
    _load_vocab, _gpt2_byte_decoder, _token_classes,
)
from analysis.learned_prior import _make_loss, _flatten_topk, TOPK, REGS

PUNCT_ALPHAS = [300.0, 3000.0, 30000.0]   # lam_L = alpha/(N_L+alpha), same grid as Exp 19
OUT_DIR = "outputs"


# ---------------------------------------------------------------- shared harness
def _load_harness():
    import pandas as pd
    weights, langs, lang_to_idx = _load_model_data()
    train_counts = _load_train_counts()
    N = np.array([train_counts.get(l, 0) for l in langs], dtype=np.float64)
    scripts = np.array([l.rsplit("_", 1)[-1] if "_" in l else "Unknown" for l in langs])
    W = np.array(weights, dtype=np.float32)
    diag = pd.read_csv(DIAG_CSV)
    cat = dict(zip(diag["lang"], diag["category"]))
    lang_flags = {
        "overall": np.ones(len(langs), bool),
        "tail": N < 1_000,
        "magnets": np.array([cat.get(l) == "flat_magnet" for l in langs]),
        "twins": np.array([cat.get(l) == "twin" for l in langs]),
        "head": N >= 18_000,
    }
    with open(TEXT_CACHE, "rb") as f:
        cache = pickle.load(f)
    texts = cache["texts"]
    y_lab = np.array(cache["y_true"], dtype=object)
    line_idx = np.asarray(cache["line_idx"])
    y_idx = np.array([lang_to_idx[l] for l in y_lab], dtype=np.int32)
    return W, langs, lang_to_idx, N, scripts, lang_flags, texts, y_lab, y_idx, line_idx


def _strat_row(y_idx, pred_idx, lang_flags):
    """Stratified macro-F1 over the balanced val; pred_idx = -1 counts as wrong."""
    return {st: compute_metrics(y_idx[flags[y_idx]], pred_idx[flags[y_idx]])["macro_f1"]
            for st, flags in lang_flags.items()}


def _pred_to_idx(pred_labels, lang_to_idx):
    return np.array([lang_to_idx.get(p, -1) for p in pred_labels], dtype=np.int32)


def _validate_baseline(base_idx, line_idx, langs):
    """Zero-modification predictions must match the saved full-test baseline memmap."""
    mm = np.asarray(np.lib.format.open_memmap(
        os.path.join(SCRATCH_DIR, "pred_baseline.npy"), mode="r"))[line_idx]
    ok = mm >= 0
    agree = float((base_idx[ok] == mm[ok]).mean())
    print(f"  baseline agreement with saved full-test predictions: {agree:.4f} "
          f"({int(ok.sum()):,} lines)")
    if agree < 0.99:
        raise RuntimeError(f"baseline agreement {agree:.4f} < 0.99; harness misaligned")


def _sweep_and_report(name, configs, build_fn, out_path):
    """Generic weight-modification sweep on the balanced val: predict baseline once,
    then each config; guard-select; write the table. No test scoring here."""
    import pandas as pd
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    (W, langs, lang_to_idx, N, scripts, lang_flags,
     texts, _y_lab, y_idx, line_idx) = _load_harness()
    model = _load_unilid_model()
    print(f"[{name}] baseline prediction on {len(texts):,} balanced-val texts...")
    base_idx = _pred_to_idx(predict_all(texts, model), lang_to_idx)
    _validate_baseline(base_idx, line_idx, langs)
    base_row = _strat_row(y_idx, base_idx, lang_flags)
    rows = [{"config": "baseline", **{f"val_{k}": v for k, v in base_row.items()}}]
    for cfg_name, cfg_args in configs:
        print(f"[{name}] config {cfg_name}...")
        w_new = build_fn(W, N, scripts, *cfg_args)
        model.model.set_weight_sets(w_new.tolist())
        del w_new
        pred = _pred_to_idx(predict_all(texts, model), lang_to_idx)
        r = _strat_row(y_idx, pred, lang_flags)
        rows.append({"config": cfg_name, **{f"val_{k}": v for k, v in r.items()}})
        print("   ", {k: round(v, 4) for k, v in r.items()})
    eligible = [r for r in rows if r["config"] != "baseline"
                and passes_shortlist(r, rows[0], prefix="val_")]
    best = max(eligible, key=lambda r: r["val_overall"])["config"] if eligible else "baseline"
    lines = [f"# {name} — balanced-val shortlist (no test scoring)\n",
             f"Shortlisted config: **{best}** (baseline means nothing passed).",
             f"Shortlist rule (`passes_shortlist`): val overall may drop at most "
             f"{GUARD_TOL}; tail/magnets may drop at most {TAIL_RECALL_TOL}, "
             f"twins/head at most {GUARD_TOL}. The balanced val cannot show the "
             f"global (precision) view, so a shortlisted config is NOT adopted here: "
             f"adoption requires `passes_two_sided` on its full-pool pass "
             f"(see EXPERIMENTAL_SETUP.md, precision-primary adoption rule).\n",
             pd.DataFrame(rows).round(4).to_markdown(index=False)]
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"[{name}] best (guarded): {best}; wrote {out_path}")
    return best


# ---------------------------------------------------------------- experiment 1: floor
def run_floor(out_dir: str = OUT_DIR):
    def build(W, N, scripts, F):
        w, _n = build_equalized_weights(W, F)
        return w
    configs = [(f"floor{F:g}", (F,)) for F in FLOORS]
    return _sweep_and_report(
        "Floor equalization under the balanced protocol", configs, build,
        os.path.join(out_dir, "tables", "balanced_floor_eq.md"))


# ---------------------------------------------------------- experiment 2: punct prior
def _dp_ids(vocab):
    dec = _gpt2_byte_decoder()
    ids = np.array([i for i, t in enumerate(vocab) if "dp" in _token_classes(t, dec)])
    if len(ids) == 0:
        raise RuntimeError("dp token set is empty")
    return ids


def build_punct_prior_weights(W, N, scripts, dp_ids, alpha):
    """Partial pooling of the dp columns toward the within-script mean; every other
    entry bit-identical; no renormalization."""
    out = np.array(W, dtype=np.float32)
    cols = np.exp(W[:, dp_ids].astype(np.float64))
    lam = (alpha / (N + alpha))[:, None]
    for s in np.unique(scripts):
        m = scripts == s
        w = np.minimum(N[m], float(RES_CAP))
        tot = w.sum()
        if tot <= 0:
            continue
        mean_s = ((w / tot)[:, None] * cols[m]).sum(axis=0)
        blended = (1.0 - lam[m]) * cols[m] + lam[m] * mean_s[None, :]
        out[np.ix_(np.where(m)[0], dp_ids)] = np.log(blended).astype(np.float32)
    if not np.isfinite(out).all():
        raise RuntimeError("non-finite weights after punctuation partial pooling")
    return out


def run_punct_prior(out_dir: str = OUT_DIR):
    vocab = _load_vocab()
    dp = _dp_ids(vocab)
    print(f"punctuation prior: {len(dp)} neutral digit/punctuation columns")
    def build(W, N, scripts, alpha):
        return build_punct_prior_weights(W, N, scripts, dp, alpha)
    configs = [(f"punct_a{a:g}", (a,)) for a in PUNCT_ALPHAS]
    return _sweep_and_report(
        "Punctuation partial pooling (script-level hierarchical prior)", configs, build,
        os.path.join(out_dir, "tables", "balanced_punct_prior.md"))


# ---------------------------------------------------------- experiment 3: bias refit
def run_bias_refit(out_dir: str = OUT_DIR):
    """Fit b on the fit half of the balanced val (per-language alternating split),
    select reg on the selection half via the guard, using the top-k candidate argmax."""
    import pandas as pd
    from scipy.optimize import minimize
    os.makedirs(os.path.join(out_dir, "tables"), exist_ok=True)
    (W, langs, lang_to_idx, N, scripts, lang_flags,
     texts, _y_lab, y_idx, line_idx) = _load_harness()
    n_lang = len(langs)
    model = _load_unilid_model()

    # per-language alternating split (both halves language-balanced)
    order = np.lexsort((line_idx, y_idx))
    rank_within = np.empty(len(y_idx), dtype=np.int64)
    seen = {}
    for pos in order:
        li = int(y_idx[pos])
        rank_within[pos] = seen.get(li, 0)
        seen[li] = seen.get(li, 0) + 1
    fit_mask = (rank_within % 2) == 0
    sel_mask = ~fit_mask
    print(f"bias refit: fit half {int(fit_mask.sum()):,}, "
          f"selection half {int(sel_mask.sum()):,}")

    print("preprocess + top-k extraction...")
    pre = [model.preprocess(t) for t in texts]
    nonempty = [p if p else "" for p in pre]
    topk = model.model.top_k_of_cached_weight_sets_batch(nonempty, TOPK)
    if len(topk) != len(texts):
        raise RuntimeError(f"top-k returned {len(topk)} for {len(texts)} texts")

    def flat_for(mask):
        pos = [i for i in np.where(mask)[0] if topk[i]]
        ci, cs, seg, tfp = _flatten_topk([topk[i] for i in pos],
                                         y_idx[np.array(pos)], n_lang)
        return pos, ci, cs, seg, tfp

    fit_pos, fci, fcs, fseg, ftfp = flat_for(fit_mask)
    recall = float((ftfp >= 0).mean())
    print(f"  fit-half top-{TOPK} recall of true label: {recall:.4f}")

    sel_pos, sci, scs, sseg, _stfp = flat_for(sel_mask)
    sel_y = y_idx[np.array(sel_pos)]
    sel_flags = {st: flags[sel_y] for st, flags in lang_flags.items()}

    def sel_row_for_b(b):
        logit = scs + b[sci]
        order2 = np.lexsort((logit, sseg))
        seg_sorted, idx_sorted = sseg[order2], sci[order2]
        lasts = np.searchsorted(seg_sorted, np.arange(len(sel_pos)), side="right") - 1
        pred = idx_sorted[lasts].astype(np.int32)
        return {st: compute_metrics(sel_y[m], pred[m])["macro_f1"]
                for st, m in sel_flags.items()}

    base_row = sel_row_for_b(np.zeros(n_lang))
    rows = [{"reg": 0.0, **{f"sel_{k}": round(v, 4) for k, v in base_row.items()}}]
    best = {"reg": None, "b": None, "sel_overall": -1}
    for reg in REGS:
        lg = _make_loss(fci, fcs, fseg, ftfp, len(fit_pos), n_lang, reg,
                        np.zeros(n_lang))
        res = minimize(lg, np.zeros(n_lang), jac=True, method="L-BFGS-B",
                       options={"maxiter": 200, "maxfun": 300})
        b = res.x.astype(np.float64)
        r = sel_row_for_b(b)
        rows.append({"reg": reg, **{f"sel_{k}": round(v, 4) for k, v in r.items()}})
        if passes_shortlist(r, base_row) and r["overall"] > best["sel_overall"]:
            best = {"reg": reg, "b": b, "sel_overall": r["overall"]}
        print(f"  reg={reg}: " + " ".join(f"{k}={r[k]:.4f}"
                                          for k in ("overall", *GUARD_STRATA)))

    lines = ["# Learned bias refit on balanced data — selection-half results\n",
             f"Fit on the language-balanced fit half (top-{TOPK} recall {recall:.4f}); "
             f"plain L2 (uniform-prior objective); corrected NLL gradient.",
             f"Shortlist rule (`passes_shortlist`): selection-half overall may drop "
             f"at most {GUARD_TOL}; tail/magnets at most {TAIL_RECALL_TOL}, "
             f"twins/head at most {GUARD_TOL}. Adoption additionally requires "
             f"`passes_two_sided` on a full-pool pass."]
    if best["b"] is None:
        lines.append("\n**On draw 101, no reg passes the balanced-val guard:** fitting "
                     "b on a language-balanced half and selecting on a language-"
                     "balanced half yields no per-language bias that improves "
                     "stratified macro-F1 over b = 0. This is consistent with the "
                     "Exp 14 gain depending on the head/frequency signal the balanced "
                     "protocol removes, but one draw does not establish it. Confirm "
                     "across draws 102-105; the reg=5.0 point's full-test numbers "
                     "stand on their own.")
    else:
        b = best["b"]
        np.save(os.path.join(out_dir, "tables", "learned_bias_balanced.npy"),
                b.astype(np.float32))
        neg = np.argsort(b)[:10]
        lines.append(f"\nSelected reg={best['reg']}. ||b||_inf={np.abs(b).max():.3f}; "
                     f"nonzero mass concentrates on:")
        lines.append("| lang | b | category |")
        lines.append("|---|---|---|")
        import pandas as _pd
        diag = _pd.read_csv(DIAG_CSV)
        cat = dict(zip(diag["lang"], diag["category"]))
        for i in neg:
            lines.append(f"| {langs[i]} | {b[i]:+.3f} | {cat.get(langs[i], 'mid')} |")
        lines.append("\nA guard-passing bias still needs a full-test pass (exact "
                     "biased scorer) before adoption.")
    md = pd.DataFrame(rows).to_markdown(index=False) + "\n\n" + "\n".join(lines)
    out_path = os.path.join(out_dir, "tables", "balanced_bias_refit.md")
    with open(out_path, "w") as f:
        f.write(md)
    print("\n".join(lines))
    print(f"wrote {out_path}")
    return best["reg"]


if __name__ == "__main__":
    run_floor()
    run_punct_prior()
    run_bias_refit()
