"""Solo-gate reference builds for the per-language combined-method experiment
(pre-registered 2026-07-29, EXPERIMENTS_PLAN.md "Plan: per-language combined
method", amendment 3).

The committed treatment set pairs three row treatments (unmodified, floor-21
clamp, Good-Turing rescale) with the adaptive margin gate off or on. Four of
the six combinations already have a recorded build (the three row-only
predictions, and analysis/gt_margin.py's gt_margin_adaptive for the
Good-Turing row with the gate on). This module supplies the remaining two:
the adaptive margin gate applied over (a) the unmodified weight matrix and
pred_baseline.npy (base="unmod"), and (b) the floor-21 matrix and
pred_floor21.npy (base="floor21"). Both are reference builds for the
combined-method comparison, not carried-set candidates in their own right.

Gate parameters are fixed, mirroring the gt_margin_adaptive invocation
recorded in slurm_gt_margin_adaptive.sh
(run(gate="nonhead", target_n=100_000, adaptive_q=True)):
- gated set: languages with N < HEAD_N (the nonhead gate);
- per-language quantile q_L = MARGIN_Q * (1.0 - min(N_L, HEAD_N) / HEAD_N),
  decaying linearly to zero at the head boundary; a language is excluded from
  gating if q_L <= 0 (zero strength at the boundary);
- reassignment target bar: among the top-TOPK_MARGIN runner-up candidates,
  reassign to the first with N >= RES_CAP (100,000), else keep the base
  prediction.

No new constants: HEAD_N comes from analysis.full_test_margin; MARGIN_Q,
MIN_CALIB_LINES, CALIB_MAX, CALIB_SEED, TOPK_MARGIN, PRF_CSV, and CORPUS_DIR
from analysis.margin_diagnostic; RES_CAP from analysis.hierarchical_pool; the
floor-21 target from analysis.full_test_floor21.FLOOR_TARGET;
build_equalized_weights from analysis.floor_equalization.

One addition over analysis/gt_margin.py: the tau CSV carries a `cause`
column recording why an excluded language was excluded ("low_calibration":
fewer than MIN_CALIB_LINES self-won train lines; "zero_strength": q_L <= 0;
empty string when the language is not excluded; "low_calibration" if both
reasons apply). Under the nonhead gate q_L > 0 for every gated language, so
"zero_strength" is unreachable here; the branch is kept for parity with
gt_margin.py and for reuse with other gate sets.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd

from analysis.config import TEST_FILE, TOTAL_LINES
from analysis.transfer_sweep import _load_model_data, _load_unilid_model
from analysis.full_test_eval import SCRATCH_DIR, _parse_line
from analysis.full_test_margin import HEAD_N
from analysis.full_test_floor21 import FLOOR_TARGET
from analysis.floor_equalization import build_equalized_weights
from analysis.hierarchical_pool import RES_CAP
from analysis.margin_diagnostic import (_topk_batch, _gap, PRF_CSV, TOPK_MARGIN,
                                        MARGIN_Q, MIN_CALIB_LINES, CALIB_MAX,
                                        CALIB_SEED, CORPUS_DIR)


def run(base: str) -> str:
    """base="unmod": adaptive margin gate over the unmodified weight matrix and
    pred_baseline.npy. base="floor21": the same gate over the floor-21 matrix
    (build_equalized_weights at FLOOR_TARGET, fingerprint-verified against
    fingerprint_floor21.json) and pred_floor21.npy. Both use the fixed
    nonhead/adaptive-quantile/target_n=100,000 gate described in the module
    docstring."""
    if base not in ("unmod", "floor21"):
        raise ValueError(f"unknown base {base!r}; must be 'unmod' or 'floor21'")

    if base == "unmod":
        base_pred_path = os.path.join(SCRATCH_DIR, "pred_baseline.npy")
        out_pred = os.path.join(SCRATCH_DIR, "pred_unmod_gate.npy")
        out_tau = "outputs/diagnostic/tau_unmod_gate.csv"
        out_md = "outputs/tables/unmod_gate_build.md"
    else:
        base_pred_path = os.path.join(SCRATCH_DIR, "pred_floor21.npy")
        out_pred = os.path.join(SCRATCH_DIR, "pred_floor21_gate.npy")
        out_tau = "outputs/diagnostic/tau_floor21_gate.csv"
        out_md = "outputs/tables/floor21_gate_build.md"

    prf = pd.read_csv(PRF_CSV)
    langs = prf.lang.tolist()
    n_lang = len(langs)
    N = prf.N.values
    tail_idx = np.where(N < HEAD_N)[0]   # nonhead gate: every language below the head

    weights, langs_m, _m = _load_model_data()
    if langs_m != langs:
        raise RuntimeError("model language order differs from the PRF CSV")
    W = np.array(weights, dtype=np.float32)
    del weights

    fp_path = os.path.join(SCRATCH_DIR, "fingerprint_floor21.json")
    with open(fp_path) as f:
        fp = json.load(f)
    if base == "unmod":
        # provenance gate: the loaded weights must be the matrix that produced
        # pred_baseline.npy (fingerprint_floor21.json records its sha as sha256_base_W)
        sha = hashlib.sha256(W.tobytes()).hexdigest()
        if sha != fp["sha256_base_W"]:
            raise RuntimeError(f"loaded weight matrix does not match sha256_base_W in "
                               f"{fp_path}; it is not the matrix that produced "
                               f"pred_baseline.npy")
        matrix = None
    else:
        matrix, n_mod = build_equalized_weights(W, FLOOR_TARGET)
        if n_mod != n_lang:
            raise RuntimeError(f"floor {FLOOR_TARGET} modified {n_mod} rows, expected "
                               f"all {n_lang} (row floors all exceed it)")
        if not np.array_equal(matrix[:, :4], W[:, :4]):
            raise RuntimeError("special-token columns (0-3) were modified by the clamp")
        sha = hashlib.sha256(matrix.tobytes()).hexdigest()
        if sha != fp["sha256_w21"]:
            raise RuntimeError(f"rebuilt floor-21 matrix does not match sha256_w21 in "
                               f"{fp_path}; it would not be bit-identical to the matrix "
                               f"that produced pred_floor21.npy")
    del W

    y = np.asarray(np.lib.format.open_memmap(
        os.path.join(SCRATCH_DIR, "y_true.npy"), mode="r"))
    if y.shape != (TOTAL_LINES,):
        raise RuntimeError(f"y_true.npy shape {y.shape} != ({TOTAL_LINES},)")
    pg = np.asarray(np.lib.format.open_memmap(base_pred_path, mode="r"))
    if pg.shape != (TOTAL_LINES,):
        raise RuntimeError(f"{base_pred_path} shape {pg.shape} != ({TOTAL_LINES},)")
    affected = np.where((pg >= 0) & np.isin(pg, tail_idx))[0]
    print(f"{len(affected):,} lines carry a gated {base} prediction")

    model = _load_unilid_model()
    if base == "floor21":
        print("Caching floor-21 weights...", flush=True)
        model.model.set_weight_sets(matrix.tolist())
        del matrix
    # base="unmod" uses the model's own cached weights, the same code path that
    # produced pred_baseline.npy (precedent: analysis/full_test_margin.py, which
    # never calls set_weight_sets and recorded top-1 agreement 1.0000)

    # --- tau recalibration under the cached matrix ---
    rng = np.random.default_rng(CALIB_SEED)
    tau = {}
    calib_rows = []
    for li in tail_idx:
        lang = langs[li]
        path = os.path.join(CORPUS_DIR, f"{lang}_train.txt")
        if not os.path.exists(path):
            raise FileNotFoundError(f"corpus file missing: {path}")
        with open(path) as f:
            lines = [l.rstrip("\n") for l in f if l.rstrip("\n")]
        if len(lines) > CALIB_MAX:
            lines = [lines[k] for k in
                     sorted(rng.choice(len(lines), CALIB_MAX, replace=False))]
        cdrop = []
        _cpos, ctopk = _topk_batch(model, lines, cdrop)
        wins = [c for c in ctopk if c and c[0][0] == li]
        gaps = np.array([_gap(c) for c in wins])
        gaps = gaps[np.isfinite(gaps)]
        low_calib = len(gaps) < MIN_CALIB_LINES
        q_l = MARGIN_Q * (1.0 - min(float(N[li]), float(HEAD_N)) / HEAD_N)
        zero_strength = q_l <= 0.0
        excluded = low_calib or zero_strength
        cause = ("low_calibration" if low_calib
                 else "zero_strength" if zero_strength else "")
        tau[int(li)] = (float("-inf") if excluded
                        else float(np.percentile(gaps, q_l)))
        calib_rows.append({"lang": lang, "n_scoreable": len(ctopk),
                           "n_self_won": len(wins), "tau": tau[int(li)],
                           "excluded": excluded, "cause": cause})
    os.makedirs(os.path.dirname(out_tau), exist_ok=True)
    pd.DataFrame(calib_rows).to_csv(out_tau, index=False)
    n_excluded = sum(r["excluded"] for r in calib_rows)

    # --- score affected lines and apply the head-targeted gate ---
    want = set(affected.tolist())
    texts = {}
    with open(TEST_FILE) as fh:
        for i in range(TOTAL_LINES):
            line = fh.readline()
            if i not in want:
                continue
            _label, text = _parse_line(line)
            texts[i] = text
    if len(texts) != len(affected):
        raise RuntimeError(f"collected {len(texts)} texts for {len(affected)} lines")
    drop = []
    pos, topk = _topk_batch(model, [texts[i] for i in affected], drop)
    if drop:
        raise RuntimeError(f"{len(drop)} affected lines empty after preprocess")
    agree = np.mean([topk[k][0][0] == int(pg[affected[pos[k]]])
                     for k in range(len(topk))])
    # unmod rescoring uses the identical weights that produced the base memmap, so
    # the bar follows full_test_margin's recorded 1.0000 (0.9999 tolerates tie flips);
    # floor21 keeps the gt_margin precedent of 0.99 for a rebuilt modified matrix
    agree_min = 0.9999 if base == "unmod" else 0.99
    if agree < agree_min:
        raise RuntimeError(f"top-1 agreement with {base_pred_path} {agree:.4f} < "
                           f"{agree_min}")

    pred = np.array(pg, dtype=np.int16)
    n_disagree = n_gated = n_reassigned = n_to_true = n_no_target = 0
    for k, cands in enumerate(topk):
        line = int(affected[pos[k]])
        top1 = int(cands[0][0])
        if top1 != int(pg[line]):
            n_disagree += 1
            continue
        t = tau[top1]
        n_gated += 1
        if not (len(cands) >= 2 and _gap(cands) < t):
            continue
        head_cands = [int(c[0]) for c in cands[1:] if N[int(c[0])] >= RES_CAP]
        if not head_cands:
            n_no_target += 1
            continue
        pred[line] = np.int16(head_cands[0])
        n_reassigned += 1
        n_to_true += int(head_cands[0] == int(y[line]))
    np.save(out_pred, pred)

    L = [f"# {base} solo-gate build (combined-method plan amendment 3 reference "
         f"build)\n",
         (f"- unmodified weight matrix, sha256-verified against sha256_base_W "
          f"(sha {sha[:16]}...)." if base == "unmod" else
          f"- floor-21 matrix rebuilt and fingerprint-verified (sha {sha[:16]}...)."),
         f"- tau recalibrated under the {base} matrix: {n_excluded} of {len(tail_idx)} "
         f"gated languages excluded (cause recorded per language); values in "
         f"{out_tau}.",
         f"- Gated {base}-predicted lines scored: {len(affected):,} (top-1 agreement "
         f"{agree:.4f}; {n_disagree} disagreeing lines left ungated).",
         f"- Reassigned to a head candidate: {n_reassigned:,} of {n_gated:,} gated "
         f"lines; {n_to_true:,} land on the true label; {n_no_target:,} below-tau "
         f"lines kept for lack of a head candidate in the top-{TOPK_MARGIN}.",
         f"- All other lines bit-identical to {os.path.basename(base_pred_path)}. "
         f"Output: {out_pred}."]
    os.makedirs(os.path.dirname(out_md), exist_ok=True)
    with open(out_md, "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))
    return out_pred


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("unmod", "floor21"):
        raise SystemExit("usage: python -m analysis.solo_gates {unmod|floor21}")
    run(sys.argv[1])
