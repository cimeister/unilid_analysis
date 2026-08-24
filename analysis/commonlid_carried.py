"""CommonLID out-of-domain check for the top carried configurations (Exp 39).

Configurations: baseline (b=0), floor21 (the top-ranked carried configuration on
the balanced validation set), gt_margin_adaptive (the leader on the primary
quantity). The adaptive gate applies its training-data-calibrated tau values
unchanged; portability to new domains without refitting is the property under
test. Reuses commonlid_eval's readers and macro-aware scoring convention.

Gates: the recomputed baseline macro-aware accuracy must equal the recorded 0.8452
and the baseline tag-level macro-F1 the recorded 0.7228 (both Exp 12), at 4
decimals, else abort before the expensive passes.

THOSE TWO GATES BIND THE RELEASED MODEL ONLY. 0.8452 and 0.7228 are measurements
of the released weights, so holding another model to them would compare two
models and call the expected difference a wiring failure, and would make the
corrected round's own CommonLID predictions -- the whole deliverable of such a
run -- unreachable. For a non-default model the same two comparisons are computed
at the same 4-decimal resolution, printed, and written into the report labelled
INFORMATIONAL, NOT A GATE; nothing is withheld and nothing exits nonzero
(analysis/paper_breakdowns.py's `gates_binding` rule, mirrored here and in
analysis/external_bench_eval.py).

FLOOR TARGET. The clamp constant is the one recorded in the model's own
<scratch-dir>/fingerprint_floor21.json, not the module constant FLOOR_TARGET.
That constant (-21.0, analysis.full_test_floor21) is the RELEASED chain's
guard-selected value; the floor target is a measured per-model selection and the
corrected base model's own round-grid sweep selected -17.0. For the default model
the recorded value IS the module constant and the check and its message are
exactly what they were. The `n_mod == n_lang` assertion is likewise replaced by
analysis.floor_equalization.verify_one_sided_clamp: all 1,940 released rows moving
at c = -21 is an incidental fact of that model, not a property of the clamp.

CONFIGURATIONS. `--configs` names which of baseline / floor21 / gt_margin_adaptive
to run; the default is all three, which is what every released run did.
gt_margin_adaptive is refused for a non-default model: it needs three released-model
artifacts (outputs/diagnostic/tau_gt_margin_adaptive.csv, the thresholds fit on the
released weights; analysis.full_test_gt.GT_CSV plus fingerprint_gt.json, the Exp 27
Good-Turing record) and analysis.full_test_gt.build_gt_weights finds the special
columns by the SPECIAL_P = 0.2 probability the token defect produced, which a
corrected model does not have. Naming it with another model aborts and says so;
it is not skipped silently, and it feeds no paper cell.

Outputs: <out-dir>/tables/commonlid_carried.md and per-line predictions in
<out-dir>/diagnostic/commonlid_carried_preds.npz. The default out-dir is the
released tree, `outputs`; a non-default model must pass its own
(analysis.model_context.resolve_out_root), since those two files are the released
model's published record and the .npz is what analysis/commonlid_calibrated.py's
score stage wires itself against.
"""
from __future__ import annotations

import hashlib
import json
import os

import numpy as np
import pandas as pd

from analysis.commonlid_eval import (_read_commonlid, _load_macro_member_to_macro,
                                     _canonical_correct)
from analysis.model_context import DEFAULT_OUT_ROOT, resolve, resolve_out_root
from analysis.transfer_sweep import _load_model_data, _load_unilid_model
from analysis.floor_equalization import (build_equalized_weights,
                                         _special_columns,
                                         verify_one_sided_clamp)
from analysis.full_test_gt import build_gt_weights, GT_CSV
from analysis.full_test_margin import HEAD_N
from analysis.margin_diagnostic import _gap, TOPK_MARGIN
from analysis.metrics import compute_metrics

TAU_CSV = "outputs/diagnostic/tau_gt_margin_adaptive.csv"
TARGET_N = 100_000                  # the Exp 33/35 reassignment-target bar
EXPECTED_BASELINE_ACC = 0.8452      # recorded in commonlid_eval.md (Exp 12)
EXPECTED_BASELINE_TAG_F1 = 0.7228   # tag-level macro-F1 recorded in commonlid_eval.md (Exp 12)
FLOOR_TARGET = -21.0                # the RELEASED chain's clamp; see the docstring
OUT_MD = "outputs/tables/commonlid_carried.md"
SCORE_CHUNK = 20_000

# The floor-21 fingerprint written by the model's own floor-21 stage, by name; its
# `floor_target` field is the clamp this module builds at. Same file and field as
# analysis.external_bench_eval.FINGERPRINT_FLOOR21_NAME and
# analysis.mistralnemo_eval.BASE_FLOOR21_FP_NAME, spelled again here rather than
# imported because analysis.external_bench_eval imports SCORE_CHUNK from this
# module and importing it back would be circular.
FINGERPRINT_FLOOR21_NAME = "fingerprint_floor21.json"

# The three configurations, in report order. "baseline" is the reference arm every
# delta is taken against and is therefore always required.
CONFIGS_ALL = ("baseline", "floor21", "gt_margin_adaptive")

# UNILID writes exactly four special tokens into every model's vocabulary.
# analysis/floor_equalization.py's own run() and analysis.full_test_gt's
# build_gt_weights both check for exactly this many, but
# analysis.floor_equalization._special_columns does not: it drops a token that is
# absent from the vocabulary without comment. On a corrected matrix, whose
# specials sit BELOW every real token, a short list leaves a special column in the
# row minimum, so build_equalized_weights clamps the wrong entries and
# verify_one_sided_clamp, given the same short list, agrees. The count is
# therefore checked at the call site. analysis/commonlid_calibrated.py imports
# this rather than re-typing it.
N_SPECIAL_COLS = 4


def run(out_md: str = None, model_path: str = None,
        scratch_dir: str = None, out_dir: str = None,
        configs=None) -> str:
    m2M = _load_macro_member_to_macro()
    ctx = resolve(model_path, scratch_dir, purpose="CommonLID carried scoring")
    # The repo-side counterpart of that guard. Both outputs default to the released
    # model's published record: outputs/tables/commonlid_carried.md is the Exp 39
    # table and outputs/diagnostic/commonlid_carried_preds.npz is the array
    # analysis/commonlid_calibrated.py's score stage wires itself against. For the
    # default model resolve_out_root returns the root and never raises, so nothing
    # about a default run changes.
    out_root = resolve_out_root(ctx, out_dir, purpose="CommonLID carried scoring")
    if out_md is None:
        out_md = os.path.join(out_root, "tables", "commonlid_carried.md")
    preds_npz = os.path.join(out_root, "diagnostic", "commonlid_carried_preds.npz")

    configs = CONFIGS_ALL if configs is None else tuple(configs)
    unknown = [c for c in configs if c not in CONFIGS_ALL]
    if unknown:
        raise RuntimeError(f"unknown configuration(s) {unknown}; known: "
                           f"{list(CONFIGS_ALL)}")
    if "baseline" not in configs:
        raise RuntimeError("configuration 'baseline' is the reference arm every "
                           "delta in this table is taken against and cannot be "
                           "omitted")
    configs = tuple(c for c in CONFIGS_ALL if c in configs)
    # A partial run writes a table with fewer rows and an .npz with fewer keys.
    # resolve_out_root lets the RELEASED model keep the default root, so without
    # this a `--configs baseline` run against it would overwrite the published Exp
    # 39 record with a truncated artifact that carries no sign of the truncation,
    # and would break analysis/commonlid_calibrated.py's score stage, which
    # requires the "baseline" and "floor21" keys.
    if (configs != CONFIGS_ALL
            and os.path.realpath(out_root) == os.path.realpath(DEFAULT_OUT_ROOT)):
        raise RuntimeError(
            f"--configs {','.join(configs)} runs {len(configs)} of the "
            f"{len(CONFIGS_ALL)} configurations, and the resolved output root is "
            f"the released tree ({out_root}). {out_md} and the per-line .npz "
            f"beside it are the published Exp 39 record and the array "
            f"analysis/commonlid_calibrated.py's score stage wires itself "
            f"against; a partial run would overwrite them with a table that has "
            f"lost rows and an .npz that has lost keys. Pass --out-dir pointing "
            f"at a separate root, or run all {len(CONFIGS_ALL)} configurations.")
    # --out-md is the one path that does not come from out_root, so it would
    # otherwise escape the guard entirely: a corrected-model run with a correct
    # --out-dir could still be pointed at the released table. Same rule as
    # analysis/degeneracy_scan_mistralnemo.py's own out_md refusal.
    real_md, real_root = os.path.realpath(out_md), os.path.realpath(out_root)
    if real_md != real_root and not real_md.startswith(real_root + os.sep):
        raise RuntimeError(
            f"--out-md {out_md} (resolves to {real_md}) is outside the resolved "
            f"output root {out_root} ({real_root}). Every path this run writes "
            f"must sit under the root the model-context guard approved, or the "
            f"guard decides nothing.")
    if "gt_margin_adaptive" in configs and not ctx.is_default_model:
        raise RuntimeError(
            f"configuration gt_margin_adaptive cannot run against "
            f"{ctx.model_path}, which is not the released model. It needs three "
            f"released-model artifacts: {TAU_CSV} (thresholds fit on the released "
            f"weights, never refit for another model), {GT_CSV} and "
            f"{os.path.join(ctx.scratch_dir, 'fingerprint_gt.json')} (the Exp 27 "
            f"Good-Turing record). analysis.full_test_gt.build_gt_weights also "
            f"finds the special columns by the SPECIAL_P = 0.2 probability the "
            f"token defect produced, which a corrected model does not have. Pass "
            f"--configs baseline,floor21; that arm feeds no paper cell.")

    # Does the recorded-value comparison BIND this run? EXPECTED_BASELINE_ACC and
    # EXPECTED_BASELINE_TAG_F1 are measurements of the RELEASED weights, so only the
    # released model can be held to them; under any other model the difference is the
    # point of the run. Same rule and same test as analysis/paper_breakdowns.py's
    # `gates_binding` and analysis/external_bench_eval.py's.
    gates_binding = ctx.is_default_model

    print(f"CommonLID (carried) against {ctx.describe()}\n"
          f"  configs {', '.join(configs)}\n"
          f"  report  {out_md}", flush=True)
    weights, langs, _m = _load_model_data(ctx.model_path)
    W = np.array(weights, dtype=np.float32)
    del weights
    prf = pd.read_csv("outputs/diagnostic/full_test_per_lang_prf.csv")
    if prf.lang.tolist() != langs:
        raise RuntimeError("PRF language order differs from the model")
    N = prf.N.values
    gated = set(int(i) for i in np.where(N < HEAD_N)[0])

    tau = {}
    if "gt_margin_adaptive" in configs:
        taus = pd.read_csv(TAU_CSV)
        tau_by_lang = dict(zip(taus.lang, taus.tau))
        missing = [langs[i] for i in gated if langs[i] not in tau_by_lang]
        if missing:
            raise RuntimeError(f"tau missing for {len(missing)} gated languages")
        tau = {i: float(tau_by_lang[langs[i]]) for i in gated}

    texts, tags = _read_commonlid()
    tags = np.array(tags)
    print(f"{len(texts):,} CommonLID lines")
    model = _load_unilid_model(ctx.model_path)
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
    acc0_ok = round(acc0, 4) == EXPECTED_BASELINE_ACC
    if gates_binding and not acc0_ok:
        raise RuntimeError(f"baseline CommonLID accuracy {acc0:.4f} != recorded "
                           f"{EXPECTED_BASELINE_ACC}")
    tag_f1_0 = macro_f1_tag(base_idx)
    tag_f1_0_ok = round(tag_f1_0, 4) == EXPECTED_BASELINE_TAG_F1
    if gates_binding and not tag_f1_0_ok:
        raise RuntimeError(f"baseline CommonLID tag-level macro-F1 {tag_f1_0:.4f} != "
                           f"recorded {EXPECTED_BASELINE_TAG_F1}")
    gate_lines = []
    if not gates_binding:
        gate_lines = _cross_model_gate_lines(
            ctx.model_path,
            (("baseline macro-aware accuracy", acc0, EXPECTED_BASELINE_ACC, acc0_ok),
             ("baseline tag-level macro-F1", tag_f1_0, EXPECTED_BASELINE_TAG_F1,
              tag_f1_0_ok)))
        for line in gate_lines:
            print(line, flush=True)

    scr = ctx.scratch_dir
    fp_path = os.path.join(scr, FINGERPRINT_FLOOR21_NAME)
    with open(fp_path) as f:
        fp = json.load(f)
    # The clamp comes from the model's own fingerprint, so this pass cannot be built
    # at a different c than the full-pool predictions it is compared against. A
    # missing field aborts naming the artifact rather than falling back to the
    # released chain's FLOOR_TARGET.
    if "floor_target" not in fp:
        raise RuntimeError(
            f"{fp_path} records no floor_target field, so the clamp this pass must "
            f"build at cannot be read. The module constant FLOOR_TARGET "
            f"({FLOOR_TARGET}, analysis.full_test_floor21) is the RELEASED chain's "
            f"guard-selected value and must not stand in for it.")
    target = float(fp["floor_target"])
    if gates_binding and target != FLOOR_TARGET:
        raise RuntimeError(
            f"{fp_path} records floor_target {target}, expected FLOOR_TARGET "
            f"{FLOOR_TARGET} (analysis.full_test_floor21); the fingerprint was "
            f"built under a different floor than this run uses")

    print(f"floor21 pass (clamp c = {target})...")
    special = _special_columns(ctx.model_path)
    if len(special) != N_SPECIAL_COLS:
        raise RuntimeError(
            f"{ctx.model_path}: found {len(special)} special columns, expected "
            f"{N_SPECIAL_COLS}. analysis.floor_equalization._special_columns "
            f"drops a special token that is absent from the vocabulary without "
            f"comment, and a short list leaves a special column in the row "
            f"minimum, so the clamp would lower the wrong entries.")
    w21, n_mod = build_equalized_weights(W, target, special_idx=special)
    verify_one_sided_clamp(W, target, special, n_mod)
    if hashlib.sha256(w21.tobytes()).hexdigest() != fp["sha256_w21"]:
        raise RuntimeError("rebuilt floor-21 matrix does not match the recorded "
                           f"fingerprint ({fp_path})")
    if "floor21" in configs:
        model.model.set_weight_sets(w21.tolist())
        del w21
        results["floor21"] = predict_idx()
    else:
        del w21

    reassigned = kept_no_target = None
    if "gt_margin_adaptive" in configs:
        print("gt_margin_adaptive pass...")
        gt = pd.read_csv(GT_CSV)
        wgt = build_gt_weights(W, gt, langs)
        with open(os.path.join(scr, "fingerprint_gt.json")) as f:
            if hashlib.sha256(wgt.tobytes()).hexdigest() != json.load(f)["sha256_wgt"]:
                raise RuntimeError("rebuilt gt_min matrix does not match the "
                                   "recorded fingerprint")
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
                head_cands = [int(c[0]) for c in cands[1:]
                              if N[int(c[0])] >= TARGET_N]
                if not head_cands:
                    kept_no_target += 1
                    continue
                gt_idx[line] = head_cands[0]
                reassigned += 1
        results["gt_margin_adaptive"] = gt_idx
    else:
        del W

    os.makedirs(os.path.dirname(preds_npz), exist_ok=True)
    np.savez_compressed(preds_npz, tags=tags, **results)

    adaptive_sentence = (" The adaptive gate runs with its training-calibrated "
                         "tau values unchanged (no refitting on this domain)."
                         if "gt_margin_adaptive" in configs else "")
    L = ["# CommonLID (web-domain) check of the top carried configurations "
         "(Exp 39)\n",
         f"{len(texts):,} lines; macro-aware accuracy convention of Exp 12 "
         "(prediction correct if its iso code, or its macrolanguage, matches the "
         "gold tag)." + adaptive_sentence + "\n",
         "| config | macro-aware accuracy | delta | tag-level macro-F1 | delta |",
         "|---|---|---|---|---|"]
    for c, pred in results.items():
        acc = macro_aware_acc(pred)
        d = "" if c == "baseline" else f"{acc - acc0:+.4f}"
        tag_f1 = macro_f1_tag(pred)
        dt = "" if c == "baseline" else f"{tag_f1 - tag_f1_0:+.4f}"
        L.append(f"| {c} | {acc:.4f} | {d} | {tag_f1:.4f} | {dt} |")
    if gates_binding:
        L.append(f"\nReference points from Exp 12: frequency prior +0.0067, learned "
                 f"bias +0.0427.")
    if reassigned is not None:
        L.append(f"Adaptive-gate activity on this domain: {reassigned:,} reassignments; "
                 f"{kept_no_target:,} below-tau lines kept for lack of a top-resource "
                 "candidate in the top-5.")
    if configs != CONFIGS_ALL:
        L.append(f"\nConfigurations run: {len(configs)} of the "
                 f"{len(CONFIGS_ALL)} configurations "
                 f"({', '.join(CONFIGS_ALL)}): {', '.join(configs)}."
                 + ("" if "gt_margin_adaptive" in configs else
                    " gt_margin_adaptive was not requested: it needs "
                    "released-model thresholds and the Exp 27 Good-Turing record, "
                    "and feeds no paper cell."))
    if not gates_binding:
        L.append(f"\nModel: {ctx.model_path}")
        L.append(f"Clamp: c = {target} ({fp_path})")
        L.append("")
        L.extend(gate_lines)
    os.makedirs(os.path.dirname(out_md), exist_ok=True)
    with open(out_md, "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))
    return out_md


def _cross_model_gate_lines(model_path: str, comparisons) -> list:
    """The non-default-model form of the two recorded-value comparisons.

    EXPECTED_BASELINE_ACC and EXPECTED_BASELINE_TAG_F1 are measurements of the
    RELEASED weights. Comparing another model's recomputed baseline against them is
    a cross-model comparison: a difference is the expected outcome, not a
    reproduction failure. The comparison is still computed at the same 4-decimal
    resolution and reported in full; it withholds no number and fails no exit code.
    Wording and rule mirror analysis/external_bench_eval.py's
    `_cross_model_baseline_message` and analysis/paper_breakdowns.py's
    `_cross_model_message`."""
    L = [f"INFORMATIONAL, NOT A GATE: the two recorded CommonLID baseline values "
         f"below are measurements of the RELEASED model and this run scored "
         f"{model_path}, so a difference is an expected cross-model difference, "
         f"not a regression and not a reproduction failure. Every configuration "
         f"in the table above WAS computed and written, with this run's own "
         f"numbers."]
    for name, measured, recorded, ok in comparisons:
        verdict = "agrees with" if ok else "differs from"
        L.append(f"- {name}: {measured:.4f} {verdict} the recorded {recorded} "
                 f"(Exp 12/39, outputs/tables/commonlid_carried.md), difference "
                 f"{measured - recorded:+.4f}.")
    return L


def main(argv=None):
    import argparse
    from analysis.model_context import add_arguments
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-md", default=None,
                    help="report path (default: <out-dir>/tables/"
                         "commonlid_carried.md)")
    ap.add_argument("--out-dir", default=None,
                    help="repo-side output root; required when --model is not "
                         "the released model")
    ap.add_argument("--configs", default=None,
                    help="comma-separated subset of "
                         f"{','.join(CONFIGS_ALL)} (default: all three)")
    add_arguments(ap)
    a = ap.parse_args(argv)
    configs = (None if a.configs is None else
               [c.strip() for c in a.configs.split(",") if c.strip()])
    run(out_md=a.out_md, model_path=a.model_path, scratch_dir=a.scratch_dir,
        out_dir=a.out_dir, configs=configs)


if __name__ == "__main__":
    main()
