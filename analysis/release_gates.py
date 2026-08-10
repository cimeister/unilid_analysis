"""Release verification gates for the open-source calibrated UniLID package
(OPEN_SOURCE_DESIGN.md section 5; blocking).

Golden subset: the TEST HALF of the seed-42 500,000-line draw from the GlotLID-C
test pool (odd positions of the sorted draw; the even half is the excluded
250,000-line validation sample, masked to negative values in the reference
arrays). On these 250,000 lines:

- base gate: package base predictions (calibrated=False on the version-1 model)
  must EQUAL pred_baseline.npy exactly;
- calibrated gate: package calibrated predictions (the version-2 release bundle)
  must agree with pred_gate_flat4_prox21.npy on >= 99.9% of lines, and every
  disagreeing line must be classified as a boundary case (exact rank-1/rank-2
  tie, |gap - tau| <= eps, or |(s1 - s_j) - proximity_bound| <= eps, eps
  recorded); an unclassified disagreement fails the gate.

Run on a login node (the package's Rust batch inference; cap threads):
  RAYON_NUM_THREADS=32 python -m analysis.release_gates --mode base
  RAYON_NUM_THREADS=32 python -m analysis.release_gates --mode calibrated

Outputs: outputs/release/gate_<mode>.json (+ forensics for disagreements).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys

import numpy as np

from analysis.config import (DEFAULT_SAMPLE_SIZE, SAMPLE_SEED, TEST_FILE,
                             TOTAL_LINES)

UNILID_DIR = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "UNILID")
if UNILID_DIR not in sys.path:
    sys.path.insert(0, UNILID_DIR)

STORE = "/capstor/store/cscs/swissai/a0229/cmeister/unilid_analysis"
BASE_MODEL = f"{STORE}/glotlidc.unilid"
CAL_MODEL = f"{STORE}/release/unilid-1940-calibrated.unilid"
REF = {"base": f"{STORE}/full_test_eval/pred_baseline.npy",
       "calibrated": f"{STORE}/full_test_eval/pred_gate_flat4_prox21.npy"}
OUT_DIR = "outputs/release"
EMPTY = -1                    # reference sentinel: empty after preprocess
CHUNK = 10_000
EPS = 1e-3                    # nats; the recorded boundary epsilon
CAL_AGREEMENT_MIN = 0.999


def _golden_indices() -> np.ndarray:
    random.seed(SAMPLE_SEED)
    idx = np.array(sorted(random.sample(range(TOTAL_LINES),
                                        DEFAULT_SAMPLE_SIZE)), dtype=np.int64)
    test_half = idx[(np.arange(DEFAULT_SAMPLE_SIZE) % 2) == 1]
    if len(test_half) != DEFAULT_SAMPLE_SIZE // 2:
        raise RuntimeError(f"test half has {len(test_half)} lines")
    return test_half


def _collect_texts(indices: np.ndarray) -> list[str]:
    wanted = set(indices.tolist())
    texts = {}
    with open(TEST_FILE, encoding="utf-8", errors="strict") as f:
        for i, line in enumerate(f):
            if i in wanted:
                parts = line.split(" ", 1)
                texts[i] = parts[1].rstrip("\n") if len(parts) > 1 else ""
        n_seen = i + 1
    if n_seen != TOTAL_LINES:
        raise RuntimeError(f"{TEST_FILE} has {n_seen:,} lines, expected "
                           f"{TOTAL_LINES:,}")
    missing = wanted - set(texts)
    if missing:
        raise RuntimeError(f"{len(missing)} golden lines missing from the "
                           f"test file")
    return [texts[i] for i in indices.tolist()]


def _predict(model, texts: list[str], lang_to_idx: dict) -> np.ndarray:
    out = np.empty(len(texts), dtype=np.int16)
    for start in range(0, len(texts), CHUNK):
        batch = texts[start:start + CHUNK]
        for j, (lang, _tokens, _score) in enumerate(model.predict_batch(batch)):
            out[start + j] = EMPTY if lang is None else lang_to_idx[lang]
        done = min(start + CHUNK, len(texts))
        if done % 50_000 == 0 or done == len(texts):
            print(f"  scored {done:,}/{len(texts):,}")
    return out


def _forensics(model, texts, indices, pkg, ref, langs):
    """Classify every disagreeing line; return (rows, n_unexplained)."""
    cal = model.calibration
    runtime = model._runtime
    rows = []
    n_unexplained = 0
    diff_pos = np.where(pkg != ref)[0]
    for p in diff_pos.tolist():
        pt = model.preprocess(texts[p])
        entry = {"line": int(indices[p]),
                 "package": langs[pkg[p]] if pkg[p] >= 0 else None,
                 "reference": langs[ref[p]] if ref[p] >= 0 else None}
        if pt is None:
            entry["cause"] = "unexplained:preprocess_empty_mismatch"
            n_unexplained += 1
            rows.append(entry)
            continue
        cands = model.model.top_k_of_cached_weight_sets_batch([pt], cal.topk)[0]
        scores = np.array([s for _i, s in cands], dtype=np.float32)
        ids = [int(i) for i, _s in cands]
        s1 = scores[0]
        gap32 = np.float32(scores[0] - scores[1]) if len(scores) > 1 else np.float32(np.inf)
        top1 = ids[0]
        tau = (runtime.tau_a[top1] if runtime.in_a[top1]
               else runtime.tau_b[top1] if runtime.in_b[top1] else None)
        entry["top1"] = langs[top1]
        entry["gap"] = float(gap32)
        entry["tau"] = None if tau is None else float(tau)
        causes = []
        if len(scores) > 1 and scores[0] == scores[1]:
            causes.append("rank1_rank2_exact_tie")
        if tau is not None and np.isfinite(tau) and abs(float(gap32) - tau) <= EPS:
            causes.append("gate_boundary")
        for j in range(1, len(ids)):
            prox = float(np.float32(s1 - scores[j]))
            if abs(prox - cal.proximity_bound) <= EPS:
                causes.append(f"proximity_boundary_rank{j + 1}")
        if len(scores) > 1 and float(gap32) <= EPS:
            causes.append("near_tie")
        entry["causes"] = causes
        if not causes:
            entry["cause"] = "UNEXPLAINED"
            n_unexplained += 1
        rows.append(entry)
    return rows, n_unexplained


def run(mode: str):
    from unilid.model_io import UnilidModel

    os.makedirs(OUT_DIR, exist_ok=True)
    indices = _golden_indices()
    ref_all = np.load(REF[mode], mmap_mode="r")
    ref = np.asarray(ref_all[indices])
    if (ref == -3).any() or (ref == -2).any():
        raise RuntimeError(
            "reference predictions carry unseen/excluded sentinels on the "
            "golden subset; the subset derivation is wrong")

    print(f"Collecting {len(indices):,} golden lines from the test pool...")
    texts = _collect_texts(indices)

    if mode == "base":
        model = UnilidModel(BASE_MODEL, calibrated=False)
    else:
        model = UnilidModel(CAL_MODEL, calibrated=True)
    lang_to_idx = {lang: i for i, lang in enumerate(model.langs)}

    print(f"Scoring ({mode})...")
    pkg = _predict(model, texts, lang_to_idx)

    n = len(indices)
    n_agree = int((pkg == ref).sum())
    agreement = n_agree / n
    result = {"mode": mode, "n_lines": n, "n_agree": n_agree,
              "agreement": agreement, "eps_nats": EPS,
              "model": BASE_MODEL if mode == "base" else CAL_MODEL,
              "reference": REF[mode]}

    if mode == "base":
        result["gate"] = "exact_equality"
        passed = n_agree == n
        if not passed:
            diff_pos = np.where(pkg != ref)[0][:200]
            result["disagreements_first200"] = [
                {"line": int(indices[p]), "package": int(pkg[p]),
                 "reference": int(ref[p])} for p in diff_pos.tolist()]
    else:
        result["gate"] = f"agreement >= {CAL_AGREEMENT_MIN} and every " \
                          f"disagreement a classified boundary case"
        rows, n_unexplained = _forensics(model, texts, indices, pkg, ref,
                                         model.langs)
        result["n_disagree"] = len(rows)
        result["n_unexplained"] = n_unexplained
        result["disagreements"] = rows
        passed = agreement >= CAL_AGREEMENT_MIN and n_unexplained == 0

    result["passed"] = passed
    out_path = os.path.join(OUT_DIR, f"gate_{mode}.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=1)
    print(f"{mode} gate: agreement {agreement:.6f} ({n_agree:,}/{n:,}) -> "
          f"{'PASS' if passed else 'FAIL'}; wrote {out_path}")
    if not passed:
        raise SystemExit(f"{mode} gate FAILED (see {out_path})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["base", "calibrated"], required=True)
    args = parser.parse_args()
    run(args.mode)


if __name__ == "__main__":
    main()
