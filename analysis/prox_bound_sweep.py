"""Sweep the re-examination gate's score-proximity bound over the banked candidates.

The promoted configuration accepts a replacement candidate only if its saved score
is within D3_PROX natural-log units of the saved top-1 score (Experiment 49,
analysis/gate_variants.py). D3_PROX = 21.0 was chosen on the derivation part of the
seed-301 split from a grid search over 0.5 to 100, and the paper states
(submission.tex:1288-1290) that "overall macro F1 on that part varies by less than
0.0003 across bounds from roughly 15 to 35". That selection was made on the
RELEASED model. This script re-measures the statement on any model whose topk stage
has run, which is what the corrected re-release needed.

Varying the bound is post-processing: the topk stage banked every candidate score
in gate_topk_{lines,ids,scores}.npy, so no test-pool scoring happens here.

analysis/gate_variants.py is NOT modified and its walk is NOT re-derived. This
script rebuilds exactly the state _run_apply_flat4_prox21 builds (banked topk
arrays, N, category, tau1/tau2, below1/below2 masks) and then calls that module's
own _flat4_prox21_two_step_pred with a varying prox_limit. The only new code here
is the loop over bounds and the scoring of each resulting prediction array.

  python -m analysis.prox_bound_sweep --json-out outputs/diagnostic/prox_sweep.json
  python -m analysis.prox_bound_sweep --model CORRECTED.unilid \\
      --scratch-dir SCRATCH_CORRECTED --out-dir outputs_corrected_round \\
      --json-out outputs_corrected_round/diagnostic/prox_sweep.json

Gates, each aborting:
  G1  every consistency check _run_apply_flat4_prox21 runs before its walk
      (fingerprint head_n, language list sha, expanded-mask sha, step disjointness,
      exactly 4 step-2 languages, in-set partition, non-negative base predictions),
      plus the canonical language-order gate against the run's own model.
  G2  prox_limit=None reproduces pred_gate_flat4_tau5.npy bit-identically
      (the module's own mandatory self-check).
  G3  prox_limit=D3_PROX reproduces the saved pred_gate_flat4_prox21.npy
      bit-identically. Without G3 a sweep could be measuring a rule that is not the
      promoted one, so D3_PROX is kept in the bound list even when a caller passes
      its own --bounds.
  G4  the seed-301 split is re-derived from this run's own y_true and required to
      match the stored rule_split_seed301.npz bit-for-bit (analysis/paper_eval.py
      gate 2).

The evaluation set is the DERIVATION part (18,001,573 lines), the set the original
selection used (EXPERIMENTAL_SETUP.md, Exp 49). The judge-part figure is reported
alongside for context and must not be used to choose a bound.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

import analysis.gate_variants as gv
from analysis.balanced_split import SEEDS, TEST_SEED
from analysis.carried_set_comparison import EXPECTED_REMAINDER
from analysis.combined_evidence import (EXPECTED_DERIVATION, EXPECTED_JUDGE,
                                        RULE_SPLIT_FRACTION, RULE_SPLIT_SEED,
                                        SPLIT_PATH)
from analysis.config import TOTAL_LINES
from analysis.hierarchical_pool import DIAG_CSV
from analysis.margin_diagnostic import PRF_CSV
from analysis.metric_decomposition import EXPECTED_KEPT, _per_lang_stats
from analysis.paper_eval import _load_draw
from analysis.transfer_sweep import _load_model_data

import hashlib

# The grid the Experiment 49 selection used: "0.5 to 100 in steps of 1"
# (EXPERIMENTAL_SETUP.md), which is 0.5, 1.5, ..., 99.5. That grid does not contain
# 21.0, so D3_PROX is unioned in below and by _bounds(): gate G3 needs the source
# value present, and a sweep that cannot reproduce the promoted predictions at the
# promoted bound is not measuring the promoted rule.
SELECTION_GRID = [0.5 + i for i in range(100)]

# The interval and the tolerance the paper's sentence states (submission.tex:1288-
# 1290: "overall macro F1 on that part varies by less than 0.0003 across bounds
# from roughly 15 to 35"). Both are properties of the published claim being
# checked, not tuning knobs.
CLAIM_LO = 15.0
CLAIM_HI = 35.0
CLAIM_TOL = 0.0003


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def _bounds(requested):
    """The bounds to sweep, always including D3_PROX so gate G3 can run."""
    return sorted(set(list(requested or SELECTION_GRID) + [gv.D3_PROX]))


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=None,
                    help="path to the .unilid model (default: the released model)")
    ap.add_argument("--scratch-dir", default=None,
                    help="root holding this model's banked topk arrays, y_true and "
                         "prediction memmaps; required when --model is not the "
                         "released model")
    ap.add_argument("--out-dir", default="outputs",
                    help="root holding this model's tau CSVs (default: outputs)")
    ap.add_argument("--bounds", nargs="+", type=float, default=None,
                    help=f"bounds in nats (default: the Experiment 49 selection "
                         f"grid, {SELECTION_GRID[0]} to {SELECTION_GRID[-1]} in "
                         f"steps of 1). D3_PROX is always added.")
    ap.add_argument("--json-out", default=None,
                    help="write the full per-bound record here")
    a = ap.parse_args(argv)
    a.bounds = _bounds(a.bounds)

    ctx = gv.configure(a.model, a.scratch_dir, a.out_dir)
    report = {"model": ctx.model_path, "scratch_dir": ctx.scratch_dir,
              "out_dir": a.out_dir, "bounds": a.bounds,
              "d3_prox_in_source": gv.D3_PROX}

    # ---- rebuild _run_apply_flat4_prox21's state, check for check (G1) ----
    entry = gv.VARIANTS["flat4_prox21"]
    lines, ids, scores, fp = gv._load_topk_arrays()
    if fp["head_n"] != gv.HEAD_N:
        raise RuntimeError(f"gate_topk_fingerprint head_n={fp['head_n']} != "
                           f"HEAD_N={gv.HEAD_N}")
    prf = pd.read_csv(PRF_CSV)
    langs = prf.lang.tolist()
    N = prf.N.values
    if len(langs) != fp["n_lang"]:
        raise RuntimeError(f"{PRF_CSV} has {len(langs)} languages, fingerprint "
                           f"records {fp['n_lang']}")
    langs_sha = hashlib.sha256("|".join(langs).encode()).hexdigest()
    if langs_sha != fp["langs_sha256"]:
        raise RuntimeError(f"{PRF_CSV} language list != fingerprint langs_sha256")
    # The canonical language order gate every scoring script runs: the model's own
    # language list must be the one these positional arrays are aligned to.
    _w, model_langs, _m = _load_model_data(ctx.model_path)
    del _w
    if model_langs != langs:
        raise RuntimeError(f"{ctx.model_path} language list does not match {PRF_CSV}")

    diag = pd.read_csv(DIAG_CSV)
    if diag.lang.tolist() != langs:
        raise RuntimeError(f"{DIAG_CSV} language order != {PRF_CSV}")
    category = diag.category.values

    low_n_mask = N < gv.HEAD_N
    flat_mask = (category == "flat_magnet")
    expanded_mask = low_n_mask | flat_mask
    expanded_sha = hashlib.sha256(expanded_mask.tobytes()).hexdigest()
    if expanded_sha != fp["sha256_expanded_mask"]:
        raise RuntimeError("expanded mask sha does not match the fingerprint")

    step1_mask = low_n_mask
    step2_mask = flat_mask & (N >= gv.HEAD_N)
    if (step1_mask & step2_mask).any():
        raise RuntimeError("step 1 and step 2 language masks overlap")
    if int(step2_mask.sum()) != 4:
        raise RuntimeError(f"step 2 set has {int(step2_mask.sum())} languages, "
                           "expected 4")

    y = np.asarray(np.lib.format.open_memmap(gv._scratch("y_true.npy"), mode="r"))
    if y.shape != (TOTAL_LINES,):
        raise RuntimeError(f"y_true shape {y.shape}")
    base_pred_path = gv._scratch(entry["base_pred"])
    base_pred_arr = np.asarray(np.lib.format.open_memmap(base_pred_path, mode="r"))
    if base_pred_arr.shape != (TOTAL_LINES,):
        raise RuntimeError(f"{base_pred_path} shape {base_pred_arr.shape}")

    n_affected = len(lines)
    base_line_pred = base_pred_arr[lines]
    if (base_line_pred < 0).any():
        raise RuntimeError(f"{base_pred_path} has a negative prediction at a "
                           "gate_topk line index")
    in_set1 = step1_mask[base_line_pred]
    in_set2 = step2_mask[base_line_pred]
    if (in_set1 & in_set2).any():
        raise RuntimeError("a line falls in both steps' in-set masks")
    if int(in_set1.sum()) + int(in_set2.sum()) != n_affected:
        raise RuntimeError("in-set masks do not partition the banked expanded set")

    step1_lang_set = {langs[i] for i in np.where(step1_mask)[0]}
    step2_lang_set = {langs[i] for i in np.where(step2_mask)[0]}
    tau1, sha_tau1 = gv._load_tau_csv(gv._out(gv.TAU_FLOOR21_GATE_CSV_NAME),
                                      langs, step1_lang_set)
    tau2, sha_tau2 = gv._load_tau_csv(gv._out(gv.TAU_FLAT4_CSV_NAME),
                                      langs, step2_lang_set)

    top1 = ids[:, 0]
    agree_mask = top1 == base_line_pred
    gap = scores[:, 0] - scores[:, 1]

    def _step_below(in_set, tau_arr):
        gated = in_set & agree_mask
        return gated & (gap < tau_arr[base_line_pred])

    below1 = _step_below(in_set1, tau1)
    below2 = _step_below(in_set2, tau2)
    n_below1, n_below2 = int(below1.sum()), int(below2.sum())
    report["setup"] = {
        "n_affected": n_affected, "n_in_set1": int(in_set1.sum()),
        "n_in_set2": int(in_set2.sum()), "n_below1": n_below1,
        "n_below2": n_below2, "tau1_sha256": sha_tau1, "tau2_sha256": sha_tau2,
        "prf_csv": PRF_CSV, "diag_csv": DIAG_CSV,
        "tau1_csv": gv._out(gv.TAU_FLOOR21_GATE_CSV_NAME),
        "tau2_csv": gv._out(gv.TAU_FLAT4_CSV_NAME),
    }
    print(f"G1 passed: {n_affected:,} banked rows; re-examined "
          f"{n_below1:,} (step 1) + {n_below2:,} (step 2)", flush=True)

    # ---- G2: prox disabled must reproduce pred_gate_flat4_tau5.npy ----
    ref_path = gv._scratch("pred_gate_flat4_tau5.npy")
    reference = np.asarray(np.lib.format.open_memmap(ref_path, mode="r"))
    none_res = gv._flat4_prox21_two_step_pred(
        None, entry["replacement_min_n"], below1, below2, lines, ids, scores, N, y,
        base_pred_arr)
    if not np.array_equal(none_res["pred"], reference):
        raise RuntimeError(f"G2 FAILED: prox_limit=None differs from {ref_path} on "
                           f"{int((none_res['pred'] != reference).sum()):,} lines")
    print(f"G2 passed: prox_limit=None is bit-identical to {ref_path}", flush=True)
    report["G2"] = {"reference": ref_path, "bit_identical": True,
                    "stats": none_res["combined"]}

    # ---- G4: seed-301 derivation part, re-derived (paper_eval.py gate 2) ----
    kept = y >= 0
    if int(kept.sum()) != EXPECTED_KEPT:
        raise RuntimeError(f"kept {int(kept.sum()):,} != {EXPECTED_KEPT:,}")
    val101, test201 = _load_draw(SEEDS[0]), _load_draw(TEST_SEED)
    if np.intersect1d(val101, test201).size:
        raise RuntimeError("test draw overlaps the working val draw")
    excl = np.zeros(TOTAL_LINES, bool)
    excl[val101] = True
    excl[test201] = True
    if int((kept & excl).sum()) != int(excl.sum()):
        raise RuntimeError("draw lines fall outside the kept pool")
    remainder_mask = kept & ~excl
    if int(remainder_mask.sum()) != EXPECTED_REMAINDER:
        raise RuntimeError(f"remainder {int(remainder_mask.sum()):,}")
    remainder_idx = np.where(remainder_mask)[0]
    u = np.random.default_rng(RULE_SPLIT_SEED).random(remainder_idx.size)
    derive_idx = remainder_idx[u < RULE_SPLIT_FRACTION]
    judge_idx = remainder_idx[u >= RULE_SPLIT_FRACTION]
    if len(derive_idx) != EXPECTED_DERIVATION or len(judge_idx) != EXPECTED_JUDGE:
        raise RuntimeError("split sizes do not match the pre-registered constants")
    with np.load(SPLIT_PATH) as stored:
        if not (np.array_equal(stored["derive_idx"], derive_idx)
                and np.array_equal(stored["judge_idx"], judge_idx)):
            raise RuntimeError(f"recomputed split != {SPLIT_PATH}")
    print(f"G4 passed: derivation part {len(derive_idx):,} lines, judge part "
          f"{len(judge_idx):,} lines, matches {SPLIT_PATH}", flush=True)
    yd = y[derive_idx]
    yj = y[judge_idx]
    n_lang = len(langs)

    # ---- the sweep ----
    saved_path = gv._scratch("pred_gate_flat4_prox21.npy")
    saved = np.asarray(np.lib.format.open_memmap(saved_path, mode="r"))
    rows = []
    for b in a.bounds:
        res = gv._flat4_prox21_two_step_pred(
            b, entry["replacement_min_n"], below1, below2, lines, ids, scores, N, y,
            base_pred_arr)
        pred = res["pred"]
        for k, nb in (("step1", n_below1), ("step2", n_below2)):
            s = res[k]
            if nb != s["n_moved"] + s["n_no_cand"] + s["n_blocked_prox"]:
                raise RuntimeError(f"bound {b}: {k} outcomes do not partition")
        f1d = _per_lang_stats(pred[derive_idx], yd, n_lang)[2]
        f1j = _per_lang_stats(pred[judge_idx], yj, n_lang)[2]
        row = {"bound": b,
               "derivation_macro_f1": float(f1d.mean()),
               "judge_macro_f1": float(f1j.mean()),
               "n_moved": res["combined"]["n_moved"],
               "n_to_true": res["combined"]["n_to_true"],
               "n_no_cand": res["combined"]["n_no_cand"],
               "n_blocked_prox": res["combined"]["n_blocked_prox"],
               "step1": res["step1"], "step2": res["step2"]}
        if b == gv.D3_PROX:
            # G3: the source bound must reproduce the saved promoted predictions.
            if not np.array_equal(pred, saved):
                raise RuntimeError(
                    f"G3 FAILED: bound {b} differs from {saved_path} on "
                    f"{int((pred != saved).sum()):,} lines")
            row["reproduces_saved_pred"] = True
            print(f"G3 passed: bound {b} is bit-identical to {saved_path}",
                  flush=True)
        rows.append(row)
        print(f"  bound {b:>5}: derivation macro F1 {row['derivation_macro_f1']:.6f}  "
              f"judge macro F1 {row['judge_macro_f1']:.6f}  moved "
              f"{row['n_moved']:,}  blocked {row['n_blocked_prox']:,}", flush=True)

    f1s = [r["derivation_macro_f1"] for r in rows]
    best = max(rows, key=lambda r: r["derivation_macro_f1"])
    at21 = [r for r in rows if r["bound"] == gv.D3_PROX][0]
    report["rows"] = rows
    report["derivation_f1_range_all_bounds"] = max(f1s) - min(f1s)
    report["argmax_bound"] = best["bound"]
    report["argmax_derivation_f1"] = best["derivation_macro_f1"]
    report["d3_prox_derivation_f1"] = at21["derivation_macro_f1"]
    report["gap_from_optimum"] = best["derivation_macro_f1"] - at21["derivation_macro_f1"]

    # The paper's claim is about the interval 15 to 35, not about whatever the
    # caller swept, so it is evaluated on that interval and named for it. Reporting
    # the range over the whole grid under this name would read as a refutation of a
    # claim the paper never made.
    window = [r for r in rows if CLAIM_LO <= r["bound"] <= CLAIM_HI]
    wf1 = [r["derivation_macro_f1"] for r in window]
    w_range = (max(wf1) - min(wf1)) if len(wf1) > 1 else 0.0
    report["claim_window"] = [CLAIM_LO, CLAIM_HI]
    report["claim_window_n_bounds"] = len(window)
    report["claim_window_derivation_f1_range"] = w_range
    report["claim_window_holds"] = bool(len(wf1) > 1 and w_range < CLAIM_TOL)

    # Where the plateau actually is: the contiguous run of bounds within CLAIM_TOL
    # of the grid optimum.
    near = [r["bound"] for r in rows
            if best["derivation_macro_f1"] - r["derivation_macro_f1"] < CLAIM_TOL]
    contiguous = len(near) == sum(1 for r in rows
                                  if min(near) <= r["bound"] <= max(near))
    report["plateau_lo"] = min(near)
    report["plateau_hi"] = max(near)
    report["plateau_contiguous"] = bool(contiguous)

    print(f"\nDerivation-part macro F1 over the claim window [{CLAIM_LO}, "
          f"{CLAIM_HI}] ({len(window)} bounds): range {w_range:.6f} "
          f"(< {CLAIM_TOL}: {report['claim_window_holds']})")
    print(f"Bounds within {CLAIM_TOL} of the grid optimum: {min(near)} to "
          f"{max(near)} (contiguous: {contiguous})")
    print(f"argmax bound {best['bound']} at {best['derivation_macro_f1']:.6f}; "
          f"bound {gv.D3_PROX} at {at21['derivation_macro_f1']:.6f} "
          f"(shortfall {best['derivation_macro_f1'] - at21['derivation_macro_f1']:.6f})")

    if a.json_out:
        os.makedirs(os.path.dirname(a.json_out), exist_ok=True)
        with open(a.json_out, "w") as f:
            json.dump(report, f, indent=2)
        print(f"Wrote {a.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
