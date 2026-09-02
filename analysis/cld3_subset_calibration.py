"""The calibration PROCEDURE applied unchanged to the three subset-fitted models
(author ruling, 2026-09-02).

    "A calibrated row for the subset should still exist. Perform the calibration
     procedure on the subset-fitted UniLID model. Do not due any hyperparameter
     sweeps. This is a test to see the generalizability of that approach."

WHAT THIS IS, AND WHAT IT IS NOT
--------------------------------
This is a TRANSFER-OF-PROCEDURE test, the same shape as E3 (the Mistral-Nemo
variant, analysis/mistralnemo_eval.py): every selected constant of the promoted
configuration `gate_flat4_prox21` is CARRIED from the full corrected model and
applied unchanged; only the quantities the procedure DEFINES as model-derived
are recomputed, because they are functions of the model's own weight matrix and
its own training distribution:

  carried, never re-selected      recomputed, because the procedure defines them
  --------------------------      ----------------------------------------------
  unseen_token_constant  -17.0    group A membership   (N_L < head_n)
  head_n               18,000     group B membership   (the flat-magnet rule)
  replacement_min_n   100,000     per-language tau     (the size-adaptive recipe)
  proximity_bound        21.0
  topk                      5
  margin_q                5.0
  group_b_percentile      5.0
  calib_max             2,000
  min_calib_lines         200
  calib_seed                0
  ZH_MAGNET/ZH_EXTREME/MAGNET_RATIO_MIN (the flat rule's own thresholds)

NO sweep is run and no constant is re-selected. That is the test: carrying c and
the gate constants unswept is exactly what makes this a generalizability
measurement rather than a refit.

WHERE THE PROCEDURE DEGENERATES ON A SUBSET, THAT IS A RESULT
-------------------------------------------------------------
outputs/rerelease/cld3_regenerated_2026-09-01.md section 6.3 predicted group B
would be empty on all three subsets (none of the full model's flat four --
sco_Latn, bjn_Latn, arg_Latn, vls_Latn -- is a CLD3 language). This module does
NOT assume that: it re-measures the flat-magnet rule on each subset model's own
weight matrix and its own validation-half predictions, exactly as
analysis/mistralnemo_eval.py's "flatrule" stage does for the Nemo variant, whose
flat set is likewise "whatever the recomputed rule yields for this weight matrix,
including possibly zero". An empty group B is recorded as a finding of the
generalizability test, not worked around; the tau and bundle stages handle it
without special-casing, as they do in the Nemo chain.

Likewise `low_calibration`: a group A language with fewer than MIN_CALIB_LINES
finite winning margins is EXCLUDED with that cause, which is the recipe's own
rule (analysis/solo_gates.py's run("floor21"), ported verbatim into
analysis/mistralnemo_eval.py::_calibrate_group and again into
unilid.calibration.estimate_tau). Group B aborts instead of excluding, for the
recorded reason: its members have N >= head_n, so a shortfall there is a wiring
error.

WHERE tau COMES FROM (the code, not the prose)
-----------------------------------------------
The recipe of record estimates tau from each language's OWN TRAINING LINES under
the clamped matrix -- CORPUS_DIR/{lang}_train.txt, at most CALIB_MAX of them
drawn with one np.random.default_rng(CALIB_SEED) shared sequentially across
languages, keeping the lines the language itself wins, taking the q_L-th
percentile of the finite winning margins. The seed-42 250,000-line validation
half enters the procedure at ONE point only: the flat-magnet rule's
`magnet_ratio` (false positives into a language on the held-out half, over its
true support there plus one). Both readings are transcribed from the code
(analysis/margin_diagnostic.py, analysis/mistralnemo_eval.py::_calibrate_group
and ::run_flatrule), which is the authority here.

THE STAGES
----------
Four are CLI stages of THIS module (--stage calibval / flatrule / tau / bundle).
The last two are separate scripts, named here so the chain reads end to end:
slurm_cld3_subset_calibrated_eval.sh (or its login-node twin
run_cld3_subset_calibrated_eval_loginnode.sh) for "eval", and
analysis/cld3_subset_calibrated_report.py for "report".

STAGE "calibval": streams the seed-42 500,000-line sample once, takes
its position-parity validation half (250,000 lines, cross-checked against the
saved val_mask.npy), and for EACH subset model scores the val lines whose gold
`lang_Script` label is in that model's own label set, under the model's
UNMODIFIED matrix. The restriction is the direct analogue of the procedure's own
situation on the full model, where every val gold is already a model label, and
of the `only_model_langs` filter every subset evaluation in this repository uses.
Without it, val lines whose gold is outside the subset would contribute false
positives to subset languages that the procedure never intended to count, and
every magnet_ratio would be inflated by the 1,841 languages the model cannot
express. Output: one .npz per subset under SCRATCH_ROOT, plus a fingerprint
binding the weight-matrix sha, the langs sha and the val line count.

STAGE "flatrule" (login node): recomputes zH (within-script entropy z-score) on
each subset model's own weight matrix via analysis.diagnostic._probs_and_logprobs
(entropy taken directly as -(P*logP).sum(axis=1), the same shortcut
mistralnemo_eval documents and verifies), and support_val/fp_val/magnet_ratio via
analysis.diagnostic._empirical_magnet on that model's calibval predictions.
Applies the rule verbatim with the three thresholds imported from
analysis.diagnostic. Flat set = is_magnet AND N >= HEAD_N.

STAGE "tau" (SLURM or login node -- 14/12/12 languages, cheap): builds each
model's clamped matrix with analysis.floor_equalization.build_equalized_weights
at c = -17.0, passing the special columns read from the container's own
vocabulary (the 0.3.0 containers park the specials at the training floor, so
omitting them would make the clamp silently do nothing), verifies the one-sided
rule with verify_one_sided_clamp, and gates that the analysis-side clamp and the
package-side unilid.calibration.apply_unseen_token_constant agree bit for bit --
because the analysis side is what tau is calibrated under and the package side is
what runs at evaluation time. Then per-language tau, group A with the
size-adaptive quantile and exclude-and-log, group B with the fixed MARGIN_Q-th
percentile and abort-on-shortfall.

STAGE "bundle" (login node): assembles the Calibration artifact from the two tau
tables, the subset's own training counts and the carried constants, and writes a
version-2 container beside the version-1 one. The weight matrix is copied
unchanged (the clamp is applied at load time by the package), which is asserted
by sha256 rather than assumed.

STAGE "eval" (about an hour for glotlidc, seconds for udhr and flores):
analysis.cld_subset_eval --mode subset on the version-2 container, once with
--calibrated and once without, so the change the calibration makes is measured
on one container by one code path.

STAGE "report": outputs/rerelease/cld3_subset_calibrated_2026-09-02.{md,json}.

NO SILENT FALLBACKS
-------------------
Every input path is required. Row counts, group A sizes, the val line count, the
weight-matrix shas, the clamp equivalence and the special-column agreement all
abort rather than degrade. Nothing here writes into paper/ or into any shared
record.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd

from analysis.build_release_calibration import CONSTANTS as FULL_MODEL_CONSTANTS
from analysis.config import DEFAULT_SAMPLE_SIZE, SCRATCH_DIR
from analysis.diagnostic import (MAGNET_RATIO_MIN, ZH_EXTREME, ZH_MAGNET,
                                 _empirical_magnet, _probs_and_logprobs)
from analysis.floor_equalization import (_special_columns,
                                         build_equalized_weights,
                                         verify_one_sided_clamp)
from analysis.full_test_eval import EMPTY
from analysis.full_test_margin import HEAD_N
from analysis.gate_variants import SCORE_BATCH_MAX
from analysis.hierarchical_pool import VAL_MASK
from analysis.margin_diagnostic import (CALIB_MAX, CALIB_SEED, CORPUS_DIR,
                                        MARGIN_Q, MIN_CALIB_LINES, _gap,
                                        _topk_batch)
from analysis.sample_data import load_sample
from analysis.transfer_sweep import (_load_model_data, _load_unilid_model,
                                     _stream_sampled_texts)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# Constants introduced by this module (flagged per the project's magic-number
# rule). Everything with a precedent value is imported above, never re-typed.
# ---------------------------------------------------------------------------

# The corrected generation's own selected unseen-token constant. NOT a new
# selection: the round-grid sweep on the corrected base model chose -17.0 (job
# 3117581, EXPERIMENTS_RESULTS.md "Round-grid c sweep"), it is the value bundled
# in outputs_corrected_round/release/calibration_glotlidc_corrected.json, and
# carrying it here unswept is the point of this test. FLOOR_TARGET in
# analysis/full_test_floor21.py is still the RELEASED chain's -21.0, so it is
# deliberately not imported.
UNSEEN_TOKEN_CONSTANT_CORRECTED = -17.0

# The validation half of the seed-42 sample, by the position-parity split
# (derived, not chosen -- the same expression analysis/mistralnemo_eval.py uses).
EXPECTED_VAL_LINES = DEFAULT_SAMPLE_SIZE // 2

# Assertions carried from outputs/rerelease/cld3_regenerated_2026-09-01.md
# sections 3 and 6.3, so a container that is not the model that record describes
# is caught here rather than producing a plausible number.
EXPECTED_ROWS = {83: 99, 80: 94, 77: 93}
EXPECTED_GROUP_A = {83: 14, 80: 12, 77: 12}

# The four special-token columns every 0.3.0-packed container in this family
# carries (outputs/rerelease/cld3sub{83,80,77}_inspect.json). Asserted, not
# assumed: the columns are located by reading each container's own vocabulary,
# and this is only the count the read must return.
EXPECTED_N_SPECIAL_COLUMNS = 4

SCRATCH_ROOT = os.path.join(SCRATCH_DIR, "cld3sub_calib")
OUT_ROOT = os.path.join(REPO_ROOT, "outputs", "rerelease",
                        "cld3_subset_calibration")

SUBSETS = {
    83: {"bench": "glotlidc",
         "model": os.path.join(SCRATCH_DIR, "cld3sub83.unilid"),
         "manifest": "outputs/rerelease/cld3_subset_corpus_manifest_83.json",
         "baseline_json": ("outputs/rerelease/cld3_subset_models/"
                           "cld3sub83_glotlidc_subset.json")},
    80: {"bench": "udhr",
         "model": os.path.join(SCRATCH_DIR, "cld3sub80.unilid"),
         "manifest": "outputs/rerelease/cld3_subset_corpus_manifest_80.json",
         "baseline_json": ("outputs/rerelease/cld3_subset_models/"
                           "cld3sub80_udhr_subset.json")},
    77: {"bench": "flores",
         "model": os.path.join(SCRATCH_DIR, "cld3sub77.unilid"),
         "manifest": "outputs/rerelease/cld3_subset_corpus_manifest_77.json",
         "baseline_json": ("outputs/rerelease/cld3_subset_models/"
                           "cld3sub77_flores_subset.json")},
}

CARRIED_CONSTANTS = dict(FULL_MODEL_CONSTANTS,
                         unseen_token_constant=UNSEEN_TOKEN_CONSTANT_CORRECTED)


# ---------------------------------------------------------------------------
# Paths and small helpers
# ---------------------------------------------------------------------------

def _p(*parts) -> str:
    return os.path.join(*parts)


def calibval_npz(subset: int, root: str = None) -> str:
    return _p(root or SCRATCH_ROOT, f"cld3sub{subset}_calibval.npz")


def calibval_fp(subset: int, root: str = None) -> str:
    return _p(root or SCRATCH_ROOT, f"cld3sub{subset}_calibval_fp.json")


def flat_set_csv(subset: int, out_root: str = None) -> str:
    return _p(out_root or OUT_ROOT, f"cld3sub{subset}_flat_set.csv")


def tau_group_a_csv(subset: int, out_root: str = None) -> str:
    return _p(out_root or OUT_ROOT, f"cld3sub{subset}_tau_group_a.csv")


def tau_group_b_csv(subset: int, out_root: str = None) -> str:
    return _p(out_root or OUT_ROOT, f"cld3sub{subset}_tau_group_b.csv")


def clamp_fp(subset: int, out_root: str = None) -> str:
    return _p(out_root or OUT_ROOT, f"cld3sub{subset}_clamp_fingerprint.json")


def calibration_json(subset: int, out_root: str = None) -> str:
    return _p(out_root or OUT_ROOT, f"cld3sub{subset}_calibration.json")


def calibrated_model(subset: int, root: str = None) -> str:
    return _p(root or SCRATCH_DIR, f"cld3sub{subset}_calibrated.unilid")


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", REPO_ROOT, "rev-parse", "HEAD"],
            text=True, stderr=subprocess.DEVNULL).strip()
    except Exception as exc:                       # provenance only
        return f"unavailable: {exc}"


def _add_unilid_to_path():
    unilid_dir = os.path.join(REPO_ROOT, "UNILID")
    if unilid_dir not in sys.path:
        sys.path.insert(0, unilid_dir)


def _require(path: str, what: str) -> str:
    if not os.path.exists(path):
        raise FileNotFoundError(f"FATAL: {what} missing at {path}")
    return path


def _subset_conf(subset: int) -> dict:
    if subset not in SUBSETS:
        raise SystemExit(f"FATAL: unknown subset {subset!r}; expected one of "
                         f"{sorted(SUBSETS)}")
    return SUBSETS[subset]


def load_subset_model_data(subset: int):
    """(weights memmap, langs, N array, train_counts) for one subset model.

    Training counts come from the subset's own corpus manifest -- the file list
    and per-label line counts the corpus builder actually wrote -- and are
    cross-checked against the shared 1,940-language count file, because the
    subset corpora are drawn from the same per-language corpus directory. A
    divergence means the manifest and the corpus have parted company and aborts.
    """
    conf = _subset_conf(subset)
    model_path = _require(conf["model"], f"subset-{subset} model")
    weights, langs, _lang_to_idx = _load_model_data(model_path)
    if len(langs) != EXPECTED_ROWS[subset]:
        raise RuntimeError(
            f"subset-{subset} container carries {len(langs)} rows, expected "
            f"{EXPECTED_ROWS[subset]} (outputs/rerelease/"
            f"cld3_regenerated_2026-09-01.md section 3)")

    with open(_require(_p(REPO_ROOT, conf["manifest"]),
                       f"subset-{subset} corpus manifest")) as f:
        manifest = json.load(f)
    lpl = manifest["lines_per_label"]
    missing = [l for l in langs if l not in lpl]
    if missing:
        raise RuntimeError(
            f"{len(missing)} subset-{subset} model language(s) absent from "
            f"{conf['manifest']}: {missing[:5]}")

    from analysis.transfer_sweep import _load_train_counts
    shared = _load_train_counts()
    disagree = [(l, lpl[l], shared.get(l)) for l in langs
                if shared.get(l) != lpl[l]]
    if disagree:
        raise RuntimeError(
            f"subset-{subset} manifest line counts disagree with the shared "
            f"1,940-language count file for {len(disagree)} language(s): "
            f"{disagree[:5]}")

    N = np.array([lpl[l] for l in langs], dtype=np.float64)
    n_group_a = int((N < HEAD_N).sum())
    if n_group_a != EXPECTED_GROUP_A[subset]:
        raise RuntimeError(
            f"subset-{subset} has {n_group_a} languages with N < HEAD_N "
            f"({HEAD_N:,}), expected {EXPECTED_GROUP_A[subset]} "
            f"(outputs/rerelease/cld3_regenerated_2026-09-01.md section 6.3)")
    return weights, langs, N, {l: int(lpl[l]) for l in langs}


def _weight_matrix_sha(weights) -> str:
    return _sha256_bytes(np.array(weights, dtype=np.float32).tobytes())


def _langs_sha(langs) -> str:
    return _sha256_bytes("|".join(langs).encode())


def _special_columns_checked(model_path: str) -> list[int]:
    """The container's special-token columns, read from its own vocabulary by
    the analysis-side reader, cross-checked against the package-side reader.

    Both readers are used downstream -- the analysis one builds the matrix tau is
    calibrated under, the package one builds the matrix inference runs on -- so
    a disagreement between them would put the calibration and the evaluation on
    different matrices. That is exactly the failure the special_idx lesson is on
    record for, so it is gated here rather than trusted.
    """
    cols = _special_columns(model_path)
    if len(cols) != EXPECTED_N_SPECIAL_COLUMNS:
        raise RuntimeError(
            f"{model_path}: read {len(cols)} special-token columns from the "
            f"vocabulary, expected {EXPECTED_N_SPECIAL_COLUMNS}")
    _add_unilid_to_path()
    from unilid.model_io import load_unilid, special_columns_of
    base_tok, _w, _langs = load_unilid(model_path)
    pkg_cols = special_columns_of(base_tok)
    del _w
    if sorted(pkg_cols) != sorted(cols):
        raise RuntimeError(
            f"{model_path}: analysis-side special columns {sorted(cols)} != "
            f"package-side {sorted(pkg_cols)}; the clamp used for threshold "
            f"estimation and the clamp used at inference would differ")
    return cols


# ---------------------------------------------------------------------------
# STAGE "calibval"
# ---------------------------------------------------------------------------

def _val_half(max_lines: int | None = None):
    """(val_texts, val_gold) for the position-parity validation half of the
    seed-42 sample, in ascending line order.

    `max_lines` is a SMOKE-TEST ONLY cap on how much of the sample is used; it
    changes every downstream number and so is refused by every non-smoke path.
    """
    data = load_sample(DEFAULT_SAMPLE_SIZE)
    y_true = np.array(data["y_true"], dtype=object)
    if len(y_true) != DEFAULT_SAMPLE_SIZE:
        raise RuntimeError(f"sample carries {len(y_true):,} labels, expected "
                           f"{DEFAULT_SAMPLE_SIZE:,}")
    texts = _stream_sampled_texts(DEFAULT_SAMPLE_SIZE)
    if len(texts) != DEFAULT_SAMPLE_SIZE:
        raise RuntimeError(f"streamed {len(texts):,} texts, expected "
                           f"{DEFAULT_SAMPLE_SIZE:,}")
    parity_val = (np.arange(DEFAULT_SAMPLE_SIZE) % 2) == 0
    saved = np.load(_require(VAL_MASK, "val_mask.npy"))
    if not np.array_equal(saved, parity_val):
        raise RuntimeError(f"{VAL_MASK} does not match the position-parity "
                           "split of the seed-42 sample")
    val_pos = np.flatnonzero(parity_val)
    if len(val_pos) != EXPECTED_VAL_LINES:
        raise RuntimeError(f"validation half has {len(val_pos):,} lines, "
                           f"expected {EXPECTED_VAL_LINES:,}")
    if max_lines is not None:
        val_pos = val_pos[:max_lines]
    val_texts = [texts[i] for i in val_pos.tolist()]
    val_gold = y_true[val_pos]
    return val_pos, val_texts, val_gold


def run_calibval(subsets: list[int], scratch_root: str, max_lines=None) -> dict:
    os.makedirs(scratch_root, exist_ok=True)
    report = {}
    print(f"Reading the seed-42 {DEFAULT_SAMPLE_SIZE:,}-line sample and its "
          f"validation half...", flush=True)
    val_pos, val_texts, val_gold = _val_half(max_lines)
    print(f"validation half: {len(val_pos):,} lines"
          f"{' (SMOKE CAP)' if max_lines else ''}", flush=True)

    for subset in subsets:
        conf = _subset_conf(subset)
        weights, langs, _N, _tc = load_subset_model_data(subset)
        fp = {"weight_matrix_sha256": _weight_matrix_sha(weights),
              "langs_sha256": _langs_sha(langs),
              "n_val_lines": int(len(val_pos)),
              "smoke_max_lines": max_lines}
        del weights

        lang_set = set(langs)
        lang_to_idx = {l: i for i, l in enumerate(langs)}
        keep = np.array([g in lang_set for g in val_gold.tolist()], dtype=bool)
        n_keep = int(keep.sum())
        if not n_keep:
            raise RuntimeError(
                f"subset-{subset}: no validation-half line has a gold label in "
                f"this model's {len(langs)} labels")
        texts = [t for t, k in zip(val_texts, keep.tolist()) if k]
        gold = val_gold[keep]
        print(f"\nsubset-{subset}: {n_keep:,} of {len(val_pos):,} validation "
              f"lines have a gold label among the model's {len(langs)} "
              f"({100.0 * n_keep / len(val_pos):.1f}%)", flush=True)

        model = _load_unilid_model(conf["model"])
        if list(model.langs) != list(langs):
            raise RuntimeError(f"subset-{subset}: UnilidModel language order "
                               "differs from the header order")
        pre, valid = [], []
        for k, t in enumerate(texts):
            p = model.preprocess(t)
            if p:
                pre.append(p)
                valid.append(k)
        out = np.full(len(texts), EMPTY, dtype=np.int16)
        for lo in range(0, len(pre), SCORE_BATCH_MAX):
            hi = min(lo + SCORE_BATCH_MAX, len(pre))
            batch = model.model.best_of_cached_weight_sets_batch(pre[lo:hi])
            if len(batch) != hi - lo:
                raise RuntimeError(f"subset-{subset} calibval scorer returned "
                                   f"{len(batch)} results for {hi - lo} inputs")
            for k, (idx, _t, _s) in zip(valid[lo:hi], batch):
                out[k] = idx
            print(f"  scored [{lo:,}:{hi:,}) of {len(pre):,}", flush=True)
        del model

        gold_idx = np.array([lang_to_idx[g] for g in gold.tolist()],
                            dtype=np.int32)
        np.savez_compressed(
            calibval_npz(subset, scratch_root),
            sample_positions=val_pos[keep].astype(np.int64),
            gold_idx=gold_idx, pred_idx=out,
            langs=np.array(langs, dtype=object))
        with open(calibval_fp(subset, scratch_root), "w") as f:
            json.dump(fp, f, indent=2)
        n_empty = int((out == EMPTY).sum())
        print(f"subset-{subset}: wrote {calibval_npz(subset, scratch_root)} "
              f"({n_keep:,} lines, {n_empty:,} empty after preprocess)",
              flush=True)
        report[subset] = {
            "n_val_lines_total": int(len(val_pos)),
            "n_val_lines_with_gold_in_model": n_keep,
            "fraction_of_val_half": n_keep / len(val_pos),
            "n_empty_after_preprocess": n_empty,
            "n_model_rows": len(langs),
            "npz": calibval_npz(subset, scratch_root),
            "npz_sha256": _sha256_file(calibval_npz(subset, scratch_root)),
            "fingerprint": fp,
            "restriction": (
                "validation-half lines whose gold lang_Script label is one of "
                "this model's rows. On the full model this restriction is the "
                "identity; on a subset model it is the only_model_langs filter "
                "every subset evaluation in this repository applies."),
        }
    return report


# ---------------------------------------------------------------------------
# STAGE "flatrule"
# ---------------------------------------------------------------------------

def run_flatrule(subsets: list[int], scratch_root: str,
                 out_root: str) -> dict:
    os.makedirs(out_root, exist_ok=True)
    report = {}
    for subset in subsets:
        conf = _subset_conf(subset)
        weights, langs, N, _tc = load_subset_model_data(subset)
        n_lang = len(langs)
        sha_w = _weight_matrix_sha(weights)
        with open(_require(calibval_fp(subset, scratch_root),
                           f"subset-{subset} calibval fingerprint")) as f:
            fp = json.load(f)
        if fp["weight_matrix_sha256"] != sha_w:
            raise RuntimeError(
                f"subset-{subset}: the matrix loaded here for zH (sha256 "
                f"{sha_w[:16]}...) is not the matrix the calibval pass scored "
                f"under ({fp['weight_matrix_sha256'][:16]}...)")

        scripts = np.array([l.rsplit("_", 1)[-1] if "_" in l else "Unknown"
                            for l in langs])
        P, logP = _probs_and_logprobs(weights)
        del weights
        H = -(P * logP).sum(axis=1)
        del P, logP

        zH = np.zeros(n_lang)
        n_scripts_scored = 0
        for s in np.unique(scripts):
            m = scripts == s
            if m.sum() < 3:
                continue
            n_scripts_scored += 1
            med = np.median(H[m])
            mad = np.median(np.abs(H[m] - med)) * 1.4826 + 1e-9
            zH[m] = (H[m] - med) / mad
        n_zh_scored = int(sum((scripts == s).sum() for s in np.unique(scripts)
                              if (scripts == s).sum() >= 3))

        d = np.load(calibval_npz(subset, scratch_root), allow_pickle=True)
        if list(d["langs"]) != list(langs):
            raise RuntimeError(f"subset-{subset}: calibval langs differ from "
                               "the container's")
        gold_idx = d["gold_idx"]
        pred_idx = d["pred_idx"]
        true_labels = np.array([langs[i] for i in gold_idx.tolist()],
                               dtype=object)
        pred_labels = np.array(
            [langs[i] if i >= 0 else "<EMPTY>" for i in pred_idx.tolist()],
            dtype=object)
        S, FP, ratio = _empirical_magnet(langs, true_labels, pred_labels)
        support_val = np.array([S[l] for l in langs])
        fp_val = np.array([FP[l] for l in langs])
        magnet_ratio = np.array([ratio[l] for l in langs])

        is_magnet = ((zH > ZH_MAGNET) & (magnet_ratio > MAGNET_RATIO_MIN)) | \
            (zH > ZH_EXTREME)
        flat_mask = is_magnet & (N >= HEAD_N)
        # The zH ceiling over the languages that are eligible for group B at all
        # (N >= HEAD_N). Group B membership needs zH > ZH_MAGNET on top of the
        # magnet_ratio test, so if this ceiling is below ZH_MAGNET then no head
        # language can be flagged at ANY magnet_ratio. That matters because
        # magnet_ratio is the one quantity the validation-half restriction can
        # move, and it can only move it down. Recording the ceiling here shows
        # the empty group B is fixed by the flatness term alone and cannot be an
        # artefact of the restriction.
        head = N >= HEAD_N
        if head.any():
            hi = int(np.argmax(np.where(head, zH, -np.inf)))
            head_zh_ceiling = {"lang": langs[hi], "zH": float(zH[hi]),
                               "N": int(N[hi]), "n_head_languages": int(head.sum()),
                               "below_zh_magnet": bool(zH[hi] <= ZH_MAGNET)}
        else:
            head_zh_ceiling = None
        flat_idx = np.where(flat_mask)[0]
        rows = [{"lang": langs[i], "N": int(N[i]), "zH": round(float(zH[i]), 4),
                 "support_val": int(support_val[i]), "fp_val": int(fp_val[i]),
                 "magnet_ratio": round(float(magnet_ratio[i]), 4)}
                for i in flat_idx]
        pd.DataFrame(rows, columns=["lang", "N", "zH", "support_val", "fp_val",
                                    "magnet_ratio"]).to_csv(
            flat_set_csv(subset, out_root), index=False)

        # Diagnostic detail for the record: which languages the rule flagged at
        # all, and how close the rest came to firing. An is_magnet language with
        # N < HEAD_N is already in group A, so the flat rule contributes nothing
        # the low-resource group does not already cover -- recording the list
        # makes that visible instead of leaving it to be inferred from a count.
        magnet_langs = [{"lang": langs[i], "N": int(N[i]),
                         "zH": round(float(zH[i]), 4),
                         "magnet_ratio": round(float(magnet_ratio[i]), 4),
                         "ge_head_n": bool(N[i] >= HEAD_N)}
                        for i in np.where(is_magnet)[0].tolist()]
        order = np.argsort(-zH)
        near = [{"lang": langs[i], "N": int(N[i]),
                 "zH": round(float(zH[i]), 4),
                 "magnet_ratio": round(float(magnet_ratio[i]), 4),
                 "support_val": int(support_val[i]), "fp_val": int(fp_val[i])}
                for i in order[:5].tolist()]
        rec = {
            "n_languages": n_lang,
            "n_scripts": int(len(np.unique(scripts))),
            "n_scripts_with_3_or_more": n_scripts_scored,
            "n_languages_with_a_scored_zH": n_zh_scored,
            "n_is_magnet": int(is_magnet.sum()),
            "is_magnet_langs": magnet_langs,
            "n_is_magnet_ge_head_n": sum(m["ge_head_n"] for m in magnet_langs),
            "max_N_among_magnets": (max(m["N"] for m in magnet_langs)
                                    if magnet_langs else None),
            "n_flat_set": int(len(flat_idx)),
            "head_zh_ceiling": head_zh_ceiling,
            "flat_set": rows,
            "zH_max": float(zH.max()), "zH_min": float(zH.min()),
            "magnet_ratio_max": float(magnet_ratio.max()),
            "n_calibval_lines": int(len(gold_idx)),
            "n_calibval_empty": int((pred_idx == EMPTY).sum()),
            "top5_by_zH": near,
            "thresholds": {"ZH_MAGNET": ZH_MAGNET, "ZH_EXTREME": ZH_EXTREME,
                           "MAGNET_RATIO_MIN": MAGNET_RATIO_MIN,
                           "HEAD_N": HEAD_N},
            "weight_matrix_sha256": sha_w,
            "flat_set_csv": flat_set_csv(subset, out_root),
        }
        report[subset] = rec
        print(f"\nsubset-{subset} ({conf['bench']}): {n_lang} languages, "
              f"{rec['n_scripts']} scripts of which "
              f"{n_scripts_scored} carry >= 3 languages (zH is 0 by "
              f"construction for the other "
              f"{n_lang - n_zh_scored} languages); "
              f"is_magnet {rec['n_is_magnet']}, flat set {rec['n_flat_set']}",
              flush=True)
        if rows:
            for r in rows:
                print(f"    {r}")
    return report


# ---------------------------------------------------------------------------
# STAGE "tau"
# ---------------------------------------------------------------------------

def _build_clamped(subset: int, out_root: str):
    """The clamped matrix at the CARRIED constant, gated both ways."""
    conf = _subset_conf(subset)
    weights, langs, N, _tc = load_subset_model_data(subset)
    W = np.array(weights, dtype=np.float32)
    del weights
    sha_w = _sha256_bytes(W.tobytes())
    special = _special_columns_checked(conf["model"])
    target = CARRIED_CONSTANTS["unseen_token_constant"]
    Wc, n_mod = build_equalized_weights(W, target, special_idx=special)
    stats = verify_one_sided_clamp(W, target, special, n_mod,
                                   label=f"subset-{subset} ")

    _add_unilid_to_path()
    from unilid.calibration import apply_unseen_token_constant
    Wp, n_mod_pkg = apply_unseen_token_constant(W, target, special)
    if n_mod_pkg != n_mod or not np.array_equal(Wc, Wp):
        raise RuntimeError(
            f"subset-{subset}: analysis-side build_equalized_weights and "
            f"package-side apply_unseen_token_constant disagree at c={target} "
            f"({n_mod} vs {n_mod_pkg} rows modified, arrays equal="
            f"{np.array_equal(Wc, Wp)}); threshold estimation and inference "
            f"would run on different matrices")
    del Wp, W

    fp = {"model": conf["model"],
          "model_file_sha256": _sha256_file(conf["model"]),
          "sha256_base_W": sha_w,
          "sha256_clamped_W": _sha256_bytes(Wc.tobytes()),
          "langs_sha256": _langs_sha(langs),
          "floor_target": float(target),
          "special_columns": [int(c) for c in special],
          "clamp": stats,
          "package_clamp_agrees": True}
    os.makedirs(out_root, exist_ok=True)
    with open(clamp_fp(subset, out_root), "w") as f:
        json.dump(fp, f, indent=2)
    return Wc, langs, N, fp


def _calibrate_group(model, langs, N, idx, q_of, out_csv: str,
                     abort_on_shortfall: bool) -> list[dict]:
    """Verbatim port of analysis/mistralnemo_eval.py::_calibrate_group, which is
    itself the port of analysis/solo_gates.py's run("floor21") (group A) and
    analysis/gate_variants.py::_calibrate_flat4_tau5 (group B). One rng shared
    sequentially across languages, as the reference does."""
    rng = np.random.default_rng(CALIB_SEED)
    calib_rows = []
    for li in idx.tolist():
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
        if low_calib and abort_on_shortfall:
            raise RuntimeError(
                f"{lang} has {len(gaps)} finite winning calibration margins, "
                f"below MIN_CALIB_LINES={MIN_CALIB_LINES}; this language is "
                f"in the flat set with N={int(N[li]):,} >= HEAD_N={HEAD_N:,}, "
                "so a shortfall indicates a wiring error, not a genuine "
                "low-resource case, and aborts rather than excluding it")
        q_l = q_of(li)
        zero_strength = q_l <= 0.0
        excluded = low_calib or zero_strength
        cause = ("low_calibration" if low_calib
                 else "zero_strength" if zero_strength else "")
        tau = float("-inf") if excluded else float(np.percentile(gaps, q_l))
        calib_rows.append({"lang": lang, "n_scoreable": len(ctopk),
                           "n_self_won": len(wins), "tau": tau,
                           "excluded": excluded, "cause": cause,
                           "n_finite_margins": len(gaps),
                           "n_sampled": len(lines), "q_l": float(q_l),
                           "N": int(N[li])})
        print(f"tau calibration: {lang} tau={tau:.4f} nats "
              f"({len(gaps):,} finite winning margins of {len(wins):,} "
              f"self-won, {len(ctopk):,} scoreable of {len(lines):,} sampled, "
              f"q_L={q_l:.4f})", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
    pd.DataFrame(calib_rows, columns=["lang", "n_scoreable", "n_self_won",
                                      "tau", "excluded", "cause"]
                 ).to_csv(out_csv, index=False)
    return calib_rows


def run_tau(subsets: list[int], out_root: str) -> dict:
    report = {}
    for subset in subsets:
        conf = _subset_conf(subset)
        Wc, langs, N, fp = _build_clamped(subset, out_root)
        lang_to_idx = {l: i for i, l in enumerate(langs)}

        flat_path = _require(flat_set_csv(subset, out_root),
                             f"subset-{subset} flat set (run --stage flatrule)")
        flat_df = pd.read_csv(flat_path)
        flat_langs = set(flat_df.lang) if len(flat_df) else set()
        for lang in flat_langs:
            n_l = int(N[lang_to_idx[lang]])
            if n_l < HEAD_N:
                raise RuntimeError(
                    f"flat-set language {lang} has N={n_l:,} < HEAD_N "
                    f"({HEAD_N:,}); the two groups are disjoint by construction")
        flat_idx = np.array(sorted(lang_to_idx[l] for l in flat_langs),
                            dtype=np.int64)
        tail_idx = np.where(N < HEAD_N)[0]

        model = _load_unilid_model(conf["model"])
        if list(model.langs) != list(langs):
            raise RuntimeError(f"subset-{subset}: UnilidModel language order "
                               "differs from the header order")
        print(f"subset-{subset}: pushing the clamped matrix "
              f"(c={CARRIED_CONSTANTS['unseen_token_constant']}) to the Rust "
              f"cache for tau calibration...", flush=True)
        model.model.set_weight_sets_numpy(Wc)
        del Wc

        q_a = (lambda li: MARGIN_Q
               * (1.0 - min(float(N[li]), float(HEAD_N)) / HEAD_N))
        rows_a = _calibrate_group(model, langs, N, tail_idx, q_a,
                                  tau_group_a_csv(subset, out_root),
                                  abort_on_shortfall=False)
        if len(rows_a) != EXPECTED_GROUP_A[subset]:
            raise RuntimeError(f"subset-{subset}: group A produced "
                               f"{len(rows_a)} rows, expected "
                               f"{EXPECTED_GROUP_A[subset]}")
        if len(flat_idx):
            rows_b = _calibrate_group(model, langs, N, flat_idx,
                                      lambda li: float(MARGIN_Q),
                                      tau_group_b_csv(subset, out_root),
                                      abort_on_shortfall=True)
        else:
            rows_b = []
            pd.DataFrame([], columns=["lang", "n_scoreable", "n_self_won",
                                      "tau", "excluded", "cause"]).to_csv(
                tau_group_b_csv(subset, out_root), index=False)
            print(f"subset-{subset}: group B is EMPTY -- the flat-magnet rule "
                  f"flagged no language with N >= HEAD_N on this model's own "
                  f"weight matrix. The gate runs with group A only. This is a "
                  f"RESULT of the generalizability test, recorded as such.",
                  flush=True)
        del model

        report[subset] = {
            "clamp": fp,
            "group_a": {"n": len(rows_a),
                        "n_excluded": sum(r["excluded"] for r in rows_a),
                        "excluded_langs": [r["lang"] for r in rows_a
                                           if r["excluded"]],
                        "causes": {c: sum(1 for r in rows_a if r["cause"] == c)
                                   for c in sorted({r["cause"] for r in rows_a})},
                        "rows": rows_a,
                        "csv": tau_group_a_csv(subset, out_root)},
            "group_b": {"n": len(rows_b), "rows": rows_b,
                        "csv": tau_group_b_csv(subset, out_root)},
        }
    return report


# ---------------------------------------------------------------------------
# STAGE "bundle"
# ---------------------------------------------------------------------------

def _rows_from_csv(path: str) -> dict:
    """analysis/build_release_calibration.py::_rows_from_csv, without its
    fixed-row-count assertion (the counts here are 14/12/12, not 1,080/4)."""
    import math
    _add_unilid_to_path()
    from unilid.calibration import TauRow
    df = pd.read_csv(path)
    expected = ["lang", "n_scoreable", "n_self_won", "tau", "excluded", "cause"]
    if list(df.columns) != expected:
        raise RuntimeError(f"{path}: unexpected columns {list(df.columns)}")
    if len(df) and df.lang.duplicated().any():
        raise RuntimeError(f"{path}: duplicate language rows")
    rows = {}
    for rec in df.itertuples(index=False):
        tau = float(rec.tau)
        excluded = bool(rec.excluded)
        cause = "" if (isinstance(rec.cause, float) and math.isnan(rec.cause)) \
            else str(rec.cause)
        if excluded != (tau == float("-inf")):
            raise RuntimeError(f"{path}: row {rec.lang} has excluded="
                               f"{excluded} with tau={tau!r}")
        rows[rec.lang] = TauRow(tau=tau, excluded=excluded, cause=cause,
                                n_scoreable=int(rec.n_scoreable),
                                n_self_won=int(rec.n_self_won))
    return rows


def run_bundle(subsets: list[int], out_root: str, model_root: str) -> dict:
    _add_unilid_to_path()
    from unilid.calibration import Calibration
    from unilid.model_io import load_unilid_raw, read_calibration, write_unilid

    report = {}
    for subset in subsets:
        conf = _subset_conf(subset)
        _w, langs, _N, train_counts = load_subset_model_data(subset)
        del _w
        with open(_require(clamp_fp(subset, out_root),
                           f"subset-{subset} clamp fingerprint")) as f:
            fp = json.load(f)
        if fp["floor_target"] != CARRIED_CONSTANTS["unseen_token_constant"]:
            raise RuntimeError(
                f"subset-{subset}: clamp fingerprint floor_target "
                f"{fp['floor_target']} != carried unseen_token_constant "
                f"{CARRIED_CONSTANTS['unseen_token_constant']}")

        group_a = _rows_from_csv(_require(tau_group_a_csv(subset, out_root),
                                          f"subset-{subset} group A tau CSV"))
        group_b = _rows_from_csv(_require(tau_group_b_csv(subset, out_root),
                                          f"subset-{subset} group B tau CSV"))
        provenance = {
            "derived": (
                "transfer of the promoted configuration gate_flat4_prox21's "
                "calibration PROCEDURE to a subset-fitted model (author ruling "
                "2026-09-02). Every constant is carried from the full corrected "
                "model unswept; only group membership and the per-language "
                "thresholds are recomputed, because the procedure defines them "
                "as functions of this model's own weight matrix and training "
                "distribution."),
            "carried_from": ("outputs_corrected_round/release/"
                             "calibration_glotlidc_corrected.json"),
            "base_weight_matrix_sha256": fp["sha256_base_W"],
            "clamped_weight_matrix_sha256": fp["sha256_clamped_W"],
            "langs_sha256": fp["langs_sha256"],
            "special_columns": fp["special_columns"],
            "n_rows_clamped": fp["clamp"]["n_modified"],
            "tau_csv_sha256": {
                "group_a": _sha256_file(tau_group_a_csv(subset, out_root)),
                "group_b": _sha256_file(tau_group_b_csv(subset, out_root))},
            "model_path": conf["model"],
            "subset": subset, "benchmark": conf["bench"],
        }
        if not group_b:
            provenance["group_b_empty"] = (
                "the flat-magnet rule flagged no language with N >= head_n on "
                "this model's own weight matrix; the gate runs with group A "
                "alone. Recorded as a result of the generalizability test.")

        cal = Calibration(group_a=group_a, group_b=group_b,
                          train_counts=train_counts, provenance=provenance,
                          **CARRIED_CONSTANTS)
        cal.runtime_for(langs)                      # full consistency validation

        reparsed = Calibration.from_json_bytes(cal.to_json_bytes())
        for name, orig in (("group_a", group_a), ("group_b", group_b)):
            rt = getattr(reparsed, name)
            if set(rt) != set(orig):
                raise RuntimeError(f"round trip changed {name} language set")
            for lang, row in orig.items():
                if rt[lang] != row:
                    raise RuntimeError(f"round trip changed {name}[{lang}]")
        if reparsed.train_counts != cal.train_counts:
            raise RuntimeError("round trip changed train_counts")
        cal.to_json_file(calibration_json(subset, out_root))

        base_tok_bytes, weights, langs_raw = load_unilid_raw(conf["model"])
        if list(langs_raw) != list(langs):
            raise RuntimeError(f"subset-{subset}: load_unilid_raw language "
                               "order differs")
        W = np.array(weights, dtype=np.float32)
        del weights
        if _sha256_bytes(W.tobytes()) != fp["sha256_base_W"]:
            raise RuntimeError(
                f"subset-{subset}: the matrix being bundled is not the matrix "
                f"the thresholds were calibrated against")
        out_model = calibrated_model(subset, model_root)
        if os.path.abspath(out_model) == os.path.abspath(conf["model"]):
            raise RuntimeError("the version-2 container must not overwrite the "
                               "version-1 one")
        write_unilid(out_model, base_tok_bytes, langs_raw, W, cal)
        del W

        # The version-2 container must carry the identical weights: the clamp is
        # applied at load time, not baked in. Asserted, not assumed.
        _bt2, w2, langs2 = load_unilid_raw(out_model)
        sha2 = _sha256_bytes(np.array(w2, dtype=np.float32).tobytes())
        del w2
        if sha2 != fp["sha256_base_W"] or list(langs2) != list(langs):
            raise RuntimeError(
                f"subset-{subset}: the version-2 container's weight matrix "
                f"({sha2[:16]}...) differs from the version-1 one "
                f"({fp['sha256_base_W'][:16]}...)")
        back = read_calibration(out_model)
        if back is None or back.to_json_bytes() != cal.to_json_bytes():
            raise RuntimeError(f"subset-{subset}: the bundled calibration does "
                               "not read back byte-identically")
        print(f"subset-{subset}: wrote {out_model} (version 2, {len(langs)} "
              f"languages, group A {len(group_a)} / group B {len(group_b)})",
              flush=True)
        report[subset] = {
            "calibrated_model": out_model,
            "calibrated_model_sha256": _sha256_file(out_model),
            "calibration_json": calibration_json(subset, out_root),
            "n_group_a": len(group_a), "n_group_b": len(group_b),
            "weight_matrix_sha256": fp["sha256_base_W"],
            "constants": dict(CARRIED_CONSTANTS),
        }
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--stage", required=True,
                   choices=["calibval", "flatrule", "tau", "bundle"])
    p.add_argument("--subsets", default="83,80,77",
                   help="comma-separated subset ids (default: all three)")
    p.add_argument("--scratch-root", default=SCRATCH_ROOT)
    p.add_argument("--out-root", default=OUT_ROOT)
    p.add_argument("--model-root", default=SCRATCH_DIR,
                   help="where the version-2 containers are written")
    p.add_argument("--smoke-val-limit", type=int, default=None,
                   help="SMOKE TEST ONLY: cap the validation half at N lines. "
                        "Changes every downstream number, so it is refused "
                        "unless --out-root and --scratch-root are both "
                        "non-default.")
    p.add_argument("--report-out", default=None,
                   help="write this stage's structured report to a JSON file")
    a = p.parse_args(argv)

    subsets = [int(s) for s in a.subsets.split(",") if s.strip()]
    for s in subsets:
        _subset_conf(s)

    if a.smoke_val_limit is not None:
        if (os.path.abspath(a.out_root) == os.path.abspath(OUT_ROOT)
                or os.path.abspath(a.scratch_root) == os.path.abspath(SCRATCH_ROOT)):
            raise SystemExit(
                "FATAL: --smoke-val-limit changes every downstream number and "
                "must not write into the real output or scratch roots; pass "
                "--out-root and --scratch-root explicitly")

    if a.stage == "calibval":
        rep = run_calibval(subsets, a.scratch_root, a.smoke_val_limit)
    elif a.stage == "flatrule":
        rep = run_flatrule(subsets, a.scratch_root, a.out_root)
    elif a.stage == "tau":
        rep = run_tau(subsets, a.out_root)
    elif a.stage == "bundle":
        rep = run_bundle(subsets, a.out_root, a.model_root)
    else:                                        # pragma: no cover - argparse
        raise SystemExit(f"unknown stage {a.stage}")

    if a.report_out and rep is not None:
        os.makedirs(os.path.dirname(os.path.abspath(a.report_out)),
                    exist_ok=True)
        with open(a.report_out, "w") as f:
            json.dump({"stage": a.stage, "git_commit": _git_commit(),
                       "constants": dict(CARRIED_CONSTANTS),
                       "subsets": {str(k): v for k, v in rep.items()}},
                      f, indent=2, default=str)
        print(f"Wrote {a.report_out}")


if __name__ == "__main__":
    main()
