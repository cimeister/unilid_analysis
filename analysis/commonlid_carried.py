"""CommonLID out-of-domain check for the top carried configurations (Exp 39).

Configurations: baseline (b=0), floor21 (the top-ranked carried configuration on
the balanced validation set), gt_margin_adaptive (the leader on the primary
quantity). The adaptive gate applies its training-data-calibrated tau values
unchanged; portability to new domains without refitting is the property under
test. Reuses commonlid_eval's readers and macro-aware scoring convention.

Gates: the recomputed baseline macro-aware accuracy must equal the recorded 0.8452
and the baseline tag-level macro-F1 the recorded 0.7228 (both Exp 12), at 4
decimals, else abort before the expensive passes.

Outputs: outputs/tables/commonlid_carried.md and per-line predictions in
outputs/diagnostic/commonlid_carried_preds.npz.
"""
from __future__ import annotations

import hashlib
import json
import os

import numpy as np
import pandas as pd

from analysis.commonlid_eval import (_read_commonlid, _load_macro_member_to_macro,
                                     _canonical_correct)
from analysis.full_test_eval import SCRATCH_DIR as FT_SCRATCH
from analysis.transfer_sweep import _load_model_data, _load_unilid_model
from analysis.floor_equalization import build_equalized_weights
from analysis.full_test_gt import build_gt_weights, GT_CSV
from analysis.full_test_margin import HEAD_N
from analysis.margin_diagnostic import _gap, TOPK_MARGIN
from analysis.metrics import compute_metrics

TAU_CSV = "outputs/diagnostic/tau_gt_margin_adaptive.csv"
TARGET_N = 100_000                  # the Exp 33/35 reassignment-target bar
EXPECTED_BASELINE_ACC = 0.8452      # recorded in commonlid_eval.md (Exp 12)
EXPECTED_BASELINE_TAG_F1 = 0.7228   # tag-level macro-F1 recorded in commonlid_eval.md (Exp 12)
FLOOR_TARGET = -21.0
OUT_MD = "outputs/tables/commonlid_carried.md"
SCORE_CHUNK = 20_000


def run(out_md: str = OUT_MD) -> str:
    m2M = _load_macro_member_to_macro()
    weights, langs, _m = _load_model_data()
    n_lang = len(langs)
    W = np.array(weights, dtype=np.float32)
    del weights
    prf = pd.read_csv("outputs/diagnostic/full_test_per_lang_prf.csv")
    if prf.lang.tolist() != langs:
        raise RuntimeError("PRF language order differs from the model")
    N = prf.N.values
    gated = set(int(i) for i in np.where(N < HEAD_N)[0])

    taus = pd.read_csv(TAU_CSV)
    tau_by_lang = dict(zip(taus.lang, taus.tau))
    missing = [langs[i] for i in gated if langs[i] not in tau_by_lang]
    if missing:
        raise RuntimeError(f"tau missing for {len(missing)} gated languages")
    tau = {i: float(tau_by_lang[langs[i]]) for i in gated}

    texts, tags = _read_commonlid()
    tags = np.array(tags)
    print(f"{len(texts):,} CommonLID lines")
    model = _load_unilid_model()
    pre, vidx = [], []
    for i, t in enumerate(texts):
        p = model.preprocess(t)
        if p:
            pre.append(p)
            vidx.append(i)

    def predict_idx():
        out = np.full(len(texts), -1, dtype=np.int32)
        for lo in range(0, len(pre), SCORE_CHUNK):
            chunk = pre[lo:lo + SCORE_CHUNK]
            batch = model.model.best_of_cached_weight_sets_batch(chunk)
            if len(batch) != len(chunk):
                raise RuntimeError(f"scorer returned {len(batch)} results for "
                                   f"{len(chunk)} inputs")
            for k, (idx, _t, _s) in enumerate(batch):
                out[vidx[lo + k]] = idx
        return out

    def macro_aware_acc(pred_idx):
        pi = np.array([langs[i].split("_")[0] if i >= 0 else ""
                       for i in pred_idx])
        corr = np.array([_canonical_correct(a, g, m2M) for a, g in zip(pi, tags)])
        return float(corr.mean())

    # Tag-level macro-F1: collapse each prediction to its "scored tag" (gold tag
    # if canonically correct, else its own macrolanguage-or-iso), then macro-F1
    # over the tag set. Identical convention to commonlid_eval.to_scored_tag.
    def macro_f1_tag(pred_idx):
        pi = np.array([langs[i].split("_")[0] if i >= 0 else ""
                       for i in pred_idx])
        y_pred_tag = np.array([g if _canonical_correct(p, g, m2M) else m2M.get(p, p)
                               for p, g in zip(pi, tags)])
        return float(compute_metrics(tags, y_pred_tag)["macro_f1"])

    results = {}
    print("baseline pass...")
    results["baseline"] = base_idx = predict_idx()
    acc0 = macro_aware_acc(base_idx)
    if round(acc0, 4) != EXPECTED_BASELINE_ACC:
        raise RuntimeError(f"baseline CommonLID accuracy {acc0:.4f} != recorded "
                           f"{EXPECTED_BASELINE_ACC}")
    tag_f1_0 = macro_f1_tag(base_idx)
    if round(tag_f1_0, 4) != EXPECTED_BASELINE_TAG_F1:
        raise RuntimeError(f"baseline CommonLID tag-level macro-F1 {tag_f1_0:.4f} != "
                           f"recorded {EXPECTED_BASELINE_TAG_F1}")

    print("floor21 pass...")
    w21, n_mod = build_equalized_weights(W, FLOOR_TARGET)
    if n_mod != n_lang:
        raise RuntimeError(f"floor {FLOOR_TARGET} modified {n_mod} rows, expected "
                           f"all {n_lang}")
    scr = FT_SCRATCH
    with open(os.path.join(scr, "fingerprint_floor21.json")) as f:
        if hashlib.sha256(w21.tobytes()).hexdigest() != json.load(f)["sha256_w21"]:
            raise RuntimeError("rebuilt floor-21 matrix does not match the recorded "
                               "fingerprint")
    model.model.set_weight_sets(w21.tolist())
    del w21
    results["floor21"] = predict_idx()

    print("gt_margin_adaptive pass...")
    gt = pd.read_csv(GT_CSV)
    wgt = build_gt_weights(W, gt, langs)
    with open(os.path.join(scr, "fingerprint_gt.json")) as f:
        if hashlib.sha256(wgt.tobytes()).hexdigest() != json.load(f)["sha256_wgt"]:
            raise RuntimeError("rebuilt gt_min matrix does not match the recorded "
                               "fingerprint")
    model.model.set_weight_sets(wgt.tolist())
    del wgt, W
    gt_idx = predict_idx()
    gate_pos = [k for k, i in enumerate(vidx) if int(gt_idx[i]) in gated]
    pre_gate = [pre[k] for k in gate_pos]
    reassigned = kept_no_target = 0
    for lo in range(0, len(pre_gate), SCORE_CHUNK):
        chunk = pre_gate[lo:lo + SCORE_CHUNK]
        topk = model.model.top_k_of_cached_weight_sets_batch(chunk, TOPK_MARGIN)
        if len(topk) != len(chunk):
            raise RuntimeError(f"top-k returned {len(topk)} results for "
                               f"{len(chunk)} inputs")
        for k, cands in enumerate(topk):
            line = vidx[gate_pos[lo + k]]
            top1 = int(cands[0][0])
            if top1 != int(gt_idx[line]):
                continue
            if not (len(cands) >= 2 and _gap(cands) < tau[top1]):
                continue
            head_cands = [int(c[0]) for c in cands[1:] if N[int(c[0])] >= TARGET_N]
            if not head_cands:
                kept_no_target += 1
                continue
            gt_idx[line] = head_cands[0]
            reassigned += 1
    results["gt_margin_adaptive"] = gt_idx

    os.makedirs("outputs/diagnostic", exist_ok=True)
    np.savez_compressed("outputs/diagnostic/commonlid_carried_preds.npz",
                        tags=tags, **results)

    L = ["# CommonLID (web-domain) check of the top carried configurations "
         "(Exp 39)\n",
         f"{len(texts):,} lines; macro-aware accuracy convention of Exp 12 "
         "(prediction correct if its iso code, or its macrolanguage, matches the "
         "gold tag). The adaptive gate runs with its training-calibrated tau "
         "values unchanged (no refitting on this domain).\n",
         "| config | macro-aware accuracy | delta | tag-level macro-F1 | delta |",
         "|---|---|---|---|---|"]
    for c, pred in results.items():
        acc = macro_aware_acc(pred)
        d = "" if c == "baseline" else f"{acc - acc0:+.4f}"
        tag_f1 = macro_f1_tag(pred)
        dt = "" if c == "baseline" else f"{tag_f1 - tag_f1_0:+.4f}"
        L.append(f"| {c} | {acc:.4f} | {d} | {tag_f1:.4f} | {dt} |")
    L.append(f"\nReference points from Exp 12: frequency prior +0.0067, learned "
             f"bias +0.0427.")
    L.append(f"Adaptive-gate activity on this domain: {reassigned:,} reassignments; "
             f"{kept_no_target:,} below-tau lines kept for lack of a top-resource "
             "candidate in the top-5.")
    os.makedirs(os.path.dirname(out_md), exist_ok=True)
    with open(out_md, "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))
    return out_md


if __name__ == "__main__":
    run()
