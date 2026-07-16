"""Phase 2: evaluate UniLID on CommonLID (Common Crawl web-domain LID benchmark)
and check whether the GlotLID-test error trends hold out of domain.

CommonLID labels are bare ISO 639-3 (incl. macrolanguages: uzb, msa, ara, zho,
...). UniLID predicts <iso639-3>_<Script>. Scoring strips the script and counts a
prediction correct if it equals the gold tag OR is a member of the gold
macrolanguage (authoritative SIL iso-639-3-macrolanguages.tab), plus one
documented bridge tgl<->fil (Filipino is the standardized register of Tagalog).
Mapping is explicit and logged (no fuzzy matching).

Needs the custom Rust tokenizer + ~250 GB to cache all 1,940 weight sets -> SLURM.
"""
from __future__ import annotations

import gzip
import json
import os
from collections import Counter

import numpy as np

from analysis.metrics import compute_metrics
from analysis.transfer_sweep import _load_model_data, _load_train_counts, _load_unilid_model, predict_all

CL_DIR = "/capstor/scratch/cscs/cmeister747/unilid_analysis/commonlid"
CL_TSV = f"{CL_DIR}/commonlid_20251209.tsv.gz"
MACRO_TAB = f"{CL_DIR}/macro_iso639-3.tab"
DIAG_CSV = "outputs/diagnostic/lang_diagnostic.csv"
MANUAL_EQUIV = {"tgl": "fil"}  # Filipino is the standardized register of Tagalog
OUT_DIR = "outputs"


def _load_macro_member_to_macro():
    """member iso639-3 -> macrolanguage iso639-3 (e.g. uzn -> uzb, zsm -> msa)."""
    m2M = {}
    with open(MACRO_TAB) as f:
        next(f)
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2:
                m2M[parts[1]] = parts[0]
    return m2M


def _read_commonlid():
    texts, tags = [], []
    with gzip.open(CL_TSV, "rt", encoding="utf-8") as f:
        next(f)
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) >= 2:
                texts.append(p[0])
                tags.append(p[1].strip())
    return texts, tags


def _canonical_correct(pred_iso, gold, m2M):
    """True if a UniLID prediction iso `pred_iso` matches CommonLID gold tag."""
    if pred_iso == gold:
        return True
    if m2M.get(pred_iso) == gold:                 # pred is a member of gold macro
        return True
    if MANUAL_EQUIV.get(gold) == pred_iso:        # tgl <-> fil bridge
        return True
    return False


def run(out_dir: str = OUT_DIR):
    os.makedirs(os.path.join(out_dir, "tables"), exist_ok=True)
    m2M = _load_macro_member_to_macro()
    weights, langs, lang_to_idx = _load_model_data()
    train_counts = _load_train_counts()
    import pandas as pd
    diag = pd.read_csv(DIAG_CSV)
    magnet_langs = set(diag.loc[diag.category == "flat_magnet", "lang"])

    print("Reading CommonLID...")
    texts, tags = _read_commonlid()
    print(f"  {len(texts):,} lines, {len(set(tags))} tags")

    print("Loading UniLID model (caches 1940 weight sets ~250GB)...")
    model = _load_unilid_model()
    N = np.array([train_counts.get(l, 0) for l in langs], dtype=np.float64)
    print("Preprocessing + predicting (baseline, frequency prior, learned bias if available)...")
    pre, vidx = [], []
    for i, t in enumerate(texts):
        p = model.preprocess(t)
        if p:
            pre.append(p); vidx.append(i)

    def predict_biased(biases):
        batch = model.model.best_of_cached_weight_sets_biased_batch(pre, biases)
        out = np.array([""] * len(texts), dtype=object)
        for j, (idx, _t, _s) in zip(vidx, batch):
            out[j] = langs[idx]
        return out

    tags = np.array(tags)

    def macro_aware_acc(preds_arr):
        pi = np.array([p.split("_")[0] if p else "" for p in preds_arr])
        corr = np.array([_canonical_correct(a, g, m2M) for a, g in zip(pi, tags)])
        return corr, pi, corr.mean()

    preds = predict_biased([0.0] * len(langs))            # baseline (b=0)
    correct, pred_iso, acc = macro_aware_acc(preds)
    freq_b = (0.5 * np.log(N + 1.0)).astype(np.float32).tolist()
    _, _, acc_freq = macro_aware_acc(predict_biased(freq_b))
    lb_path = os.path.join(out_dir, "tables", "learned_bias.npy")
    acc_learned = None
    if os.path.exists(lb_path):
        _, _, acc_learned = macro_aware_acc(predict_biased(np.load(lb_path).astype(np.float32).tolist()))
    # gold tag -> a representative UniLID label set for macro-F1 at the tag level:
    # collapse each prediction to its "scored tag" (gold if correct, else its own macro-or-iso).
    def to_scored_tag(pi, g):
        if _canonical_correct(pi, g, m2M):
            return g
        return m2M.get(pi, pi)
    y_pred_tag = np.array([to_scored_tag(pi, g) for pi, g in zip(pred_iso, tags)])
    metrics = compute_metrics(tags, y_pred_tag)

    # --- trend 1: resource asymmetry (does UniLID predict a rarer language than truth?) ---
    err = ~correct
    iso_N: dict[str, int] = {}
    for l in langs:
        iso_N[l.split("_")[0]] = iso_N.get(l.split("_")[0], 0) + train_counts.get(l, 0)
    isoN = lambda iso: iso_N.get(iso, 0)
    true_N = np.array([isoN(g) for g in tags[err]]) if err.any() else np.array([])
    pred_N = np.array([isoN(p) for p in pred_iso[err]]) if err.any() else np.array([])
    rarer = (pred_N < true_N).mean() if len(true_N) else float("nan")

    # --- trend 2: flat-magnet activity among errors ---
    pred_is_magnet = np.array([p in magnet_langs for p in preds])
    magnet_err_share = pred_is_magnet[err].mean() if err.any() else float("nan")
    magnet_pred_share = pred_is_magnet.mean()

    lines = [
        "# CommonLID (web-domain) evaluation of UniLID\n",
        f"Lines: {len(texts):,}. Tags: {len(set(tags.tolist()))}.",
        f"Macro-aware accuracy (baseline): **{acc:.4f}** (vs GlotLID-test 0.9615).",
        f"Macro-aware accuracy (+ frequency prior gamma=0.5): **{acc_freq:.4f}** ({acc_freq-acc:+.4f}).",
        (f"Macro-aware accuracy (+ learned bias): **{acc_learned:.4f}** ({acc_learned-acc:+.4f})."
         if acc_learned is not None else "Learned bias not yet available (run learned_prior first)."),
        f"Tag-level macro-F1 (baseline): {metrics['macro_f1']:.4f}.\n",
        "## Trend checks (do GlotLID-test findings hold on web data?)",
        f"- Resource asymmetry: among errors, predicted language is RARER than the true "
        f"language {rarer:.3f} of the time (GlotLID-test: ~0.86).",
        f"- Flat-magnet activity: diagnosed flat_magnets account for {magnet_err_share:.3f} of "
        f"error predictions and {magnet_pred_share:.4f} of all predictions.\n",
        "## Top confusions (true_tag -> pred_iso)",
    ]
    conf = Counter((g, p) for g, p, e in zip(tags, pred_iso, err) if e)
    for (g, p), c in conf.most_common(25):
        lines.append(f"- {g} -> {p}: {c}")

    out = os.path.join(out_dir, "tables", "commonlid_eval.md")
    with open(out, "w") as f:
        f.write("\n".join(lines))
    # also persist raw predictions for re-scoring with the pooled model later
    np.savez(os.path.join(CL_DIR, "unilid_preds.npz"), preds=preds, tags=tags)
    print("\n".join(lines))
    print(f"\nWrote {out} and {CL_DIR}/unilid_preds.npz")


if __name__ == "__main__":
    run()
