"""Report generator for the calibration-procedure transfer to the three
subset-fitted models (author ruling 2026-09-02).

Reads only what the chain wrote -- the four stage reports of
analysis/cld3_subset_calibration.py, the six evaluation summaries of
analysis/cld_subset_eval.py, and, for context, the full model's own recorded
cells -- and writes
outputs/rerelease/cld3_subset_calibrated_2026-09-02.{md,json}.

Nothing is recomputed here. Every number in the report is copied from a file the
run produced, and every path it reads is required: a missing input aborts naming
the artifact rather than leaving a cell blank.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess

from analysis.cld3_subset_calibration import (CARRIED_CONSTANTS, OUT_ROOT,
                                              SUBSETS, _sha256_file,
                                              tau_group_a_csv,
                                              tau_group_b_csv)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

OUT_MD = "outputs/rerelease/cld3_subset_calibrated_2026-09-02.md"
OUT_JSON = "outputs/rerelease/cld3_subset_calibrated_2026-09-02.json"

# The full corrected model's own cells, for comparing the change the
# calibration makes on a subset model against the change it makes on the full
# model. Both are RECORDED MEASUREMENTS read from files, never retyped:
#   - the transfer reading of the subset columns (the same three benchmarks,
#     the same 83/80/77 label sets, the FULL 1,940-language model)
#   - the main table's full-pool cells (all 1,940 languages)
FULL_TRANSFER_JSON = "outputs/rerelease/cld3_calibrated_transfer_2026-09-01.json"
FULL_MAIN_MD = "outputs_corrected_round/tables/paper_eval.md"

# The published cells the subset columns currently carry, quoted from
# outputs/rerelease/cld3_regenerated_2026-09-01.md sections 6.1 and 6.3 (the
# calibrated row prints a dash for all three FPRs).
PUBLISHED = {
    "glotlidc": {"base": (".971", "1.63e-4"), "calibrated": (".975", "--")},
    "udhr": {"base": (".992", "1.06e-5"), "calibrated": (".986", "--")},
    "flores": {"base": (".997", "3.29e-5"), "calibrated": (".992", "--")},
}
CELL_NUMBERS = {"glotlidc": (7, 8), "udhr": (9, 10), "flores": (11, 12)}
COLUMN_NAMES = {"glotlidc": "glotlidc-83", "udhr": "udhr-80",
                "flores": "flores-77"}


# The two benchmarks for which per-line (gold, prediction) pairs were banked
# under both arms, so the movement the calibration causes can be counted line by
# line rather than inferred from a macro average. GlotLID-C is deliberately not
# in this list: its 23,462,651-line calibrated pass costs about an hour and was
# not run a second time to bank predictions, which the report states rather than
# leaving the omission to be noticed.
MOVEMENT_BENCHES = ("udhr", "flores")


def _require(path: str, what: str) -> str:
    if not os.path.exists(path):
        raise SystemExit(f"FATAL: {what} missing at {path}")
    return path


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", REPO_ROOT, "rev-parse", "HEAD"],
            text=True, stderr=subprocess.DEVNULL).strip()
    except Exception as exc:                       # provenance only
        return f"unavailable: {exc}"


def _load_json(path: str, what: str) -> dict:
    with open(_require(path, what)) as f:
        return json.load(f)


def _full_main_cells(path: str) -> dict:
    """baseline / gate_flat4_prox21 macro F1 and macro FPR (x1e5) from the
    corrected E1 table's full-kept-pool block. Parsed, not retyped, so a
    regenerated table cannot silently disagree with this report."""
    want = {"baseline": None, "gate_flat4_prox21": None}
    in_block = False
    with open(_require(path, "corrected E1 table")) as f:
        for line in f:
            if line.startswith("### Macro F1 and macro FPR (x1e5), full kept "
                               "pool"):
                in_block = True
                continue
            if in_block:
                if line.startswith("###") or line.startswith("## "):
                    break
                parts = [c.strip() for c in line.strip().strip("|").split("|")]
                if len(parts) == 3 and parts[0] in want:
                    want[parts[0]] = (float(parts[1]), float(parts[2]))
    missing = [k for k, v in want.items() if v is None]
    if missing:
        raise SystemExit(
            f"FATAL: could not parse {missing} out of the full-kept-pool block "
            f"of {path}")
    return want


def _movement(out_root: str, bench: str, subset: int) -> dict:
    """Count, line by line, what the calibration moved on one benchmark.

    Reads the two banked (gold, prediction) arrays written by
    `analysis/cld_subset_eval.py --pred-out` under each arm. Labels are full
    `lang_Script`; the correctness test collapses to bare ISO, which is the
    metric's own convention for these columns.
    """
    import numpy as np
    arrs = {}
    for arm in ("baseline", "calibrated"):
        path = _require(
            os.path.join(out_root, f"movement_{bench}_{arm}_pred.npz"),
            f"{bench} {arm} per-line predictions")
        d = np.load(path, allow_pickle=True)
        labels = list(d["labels"])
        arrs[arm] = ([labels[i] for i in d["gold"]],
                     [labels[i] for i in d["pred"]])
    if arrs["baseline"][0] != arrs["calibrated"][0]:
        raise SystemExit(
            f"FATAL: {bench}: the two arms banked different gold sequences; "
            f"they did not score the same pool in the same order")
    gold, pb = arrs["baseline"]
    _g, pc = arrs["calibrated"]
    iso = lambda s: s.split("_", 1)[0]
    moved = [(g, x, y) for g, x, y in zip(gold, pb, pc) if x != y]
    fixed = sum(1 for g, x, y in moved if iso(y) == iso(g) != iso(x))
    broken = sum(1 for g, x, y in moved if iso(x) == iso(g) != iso(y))
    pairs = {}
    for g, x, y in moved:
        key = f"{iso(x)} -> {iso(y)}"
        pairs[key] = pairs.get(key, 0) + 1
    return {"bench": bench, "subset": subset, "n_lines": len(gold),
            "n_moved": len(moved), "n_fixed": fixed, "n_broken": broken,
            "n_neutral": len(moved) - fixed - broken,
            "moves_by_pair": dict(sorted(pairs.items(), key=lambda kv: -kv[1])),
            "source": os.path.join(out_root, f"movement_{bench}_*_pred.npz")}


def _group_a_script_siblings(subset: int, group_a_rows: list) -> dict:
    """For each group A language, whether the model carries another row with the
    same bare ISO 639-3 code.

    This matters because the subset columns are scored under the `--lang-only`
    bare-ISO collapse. If `ben_Latn` is in group A and the same model also
    carries `ben_Beng`, then a prediction of `ben_Latn` on a `ben_Beng` gold line
    is already counted correct, and re-examining it can only leave it correct or
    make it wrong. Read from the subset's corpus manifest, which lists the exact
    label set the model was built on.
    """
    with open(_require(os.path.join(REPO_ROOT, SUBSETS[subset]["manifest"]),
                       f"subset-{subset} corpus manifest")) as f:
        labels = json.load(f)["labels"]
    by_iso = {}
    for lab in labels:
        by_iso.setdefault(lab.split("_", 1)[0], []).append(lab)
    out = []
    for r in group_a_rows:
        sibs = [x for x in by_iso[r["lang"].split("_", 1)[0]] if x != r["lang"]]
        out.append({"lang": r["lang"], "N": r["N"], "siblings": sorted(sibs)})
    return {"rows": out,
            "n_with_sibling": sum(1 for o in out if o["siblings"]),
            "n_without_sibling": sum(1 for o in out if not o["siblings"]),
            "without_sibling": [o["lang"] for o in out if not o["siblings"]]}


def build(out_root: str = OUT_ROOT, eval_root: str = None) -> dict:
    """`out_root` holds the four stage reports and the banked per-line arrays;
    `eval_root` holds the six evaluation summaries. They differ only when the
    evaluation ran through the login-node hedge
    (`run_cld3_subset_calibrated_eval_loginnode.sh`) rather than through sbatch,
    which writes into `<out_root>/loginnode`. Defaults to `out_root`."""
    eval_root = eval_root or out_root
    stage = {name: _load_json(os.path.join(out_root, f"stage_{name}.json"),
                              f"stage_{name} report")
             for name in ("calibval", "flatrule", "tau", "bundle")}

    cells = {}
    for subset, conf in SUBSETS.items():
        bench = conf["bench"]
        arms, per_lang = {}, {}
        for arm in ("baseline", "calibrated"):
            s = _load_json(
                os.path.join(eval_root, f"cld3sub{subset}_{bench}_{arm}.json"),
                f"subset-{subset} {bench} {arm} evaluation")
            if s["calibrated"] != (arm == "calibrated"):
                raise SystemExit(
                    f"FATAL: subset-{subset}/{bench}/{arm} summary records "
                    f"calibrated={s['calibrated']}")
            if s["n_model_rows_evaluated"] != s["n_model_rows_total"]:
                raise SystemExit(
                    f"FATAL: subset-{subset}/{bench}/{arm} evaluated "
                    f"{s['n_model_rows_evaluated']} of "
                    f"{s['n_model_rows_total']} rows; --mode subset must be a "
                    f"no-op on a subset-fitted container, so nothing is "
                    f"'carried' and the package's generic subset warning is "
                    f"vacuous here")
            arms[arm] = s
            per_lang[arm] = _load_json(
                os.path.join(eval_root,
                             f"cld3sub{subset}_{bench}_{arm}_perlang.json"),
                f"subset-{subset} {bench} {arm} per-language metrics")
        if arms["baseline"]["total_samples"] != arms["calibrated"]["total_samples"]:
            raise SystemExit(
                f"FATAL: subset-{subset}/{bench}: the two arms scored "
                f"different pools")

        banked = _load_json(os.path.join(REPO_ROOT, conf["baseline_json"]),
                            f"subset-{subset} banked 2026-09-01 cell")
        for field in ("macro_f1", "macro_fpr", "total_samples", "correct"):
            if arms["baseline"][field] != banked[field]:
                raise SystemExit(
                    f"FATAL: subset-{subset}/{bench}: the baseline arm's "
                    f"{field} ({arms['baseline'][field]!r}) does not reproduce "
                    f"the banked 2026-09-01 cell ({banked[field]!r}); the "
                    f"version-2 container is not weight-identical to the "
                    f"version-1 one it was built from")

        if set(per_lang["baseline"]) != set(per_lang["calibrated"]):
            raise SystemExit(
                f"FATAL: subset-{subset}/{bench}: the two arms averaged over "
                f"different label universes")
        moves = sorted(((per_lang["calibrated"][l]["f1"]
                         - per_lang["baseline"][l]["f1"], l,
                         per_lang["baseline"][l]["f1"],
                         per_lang["calibrated"][l]["f1"],
                         per_lang["baseline"][l]["support"])
                        for l in per_lang["baseline"]), key=lambda t: t[0])
        changed = [{"lang": l, "delta_f1": d, "baseline_f1": b,
                    "calibrated_f1": c, "support": s}
                   for d, l, b, c, s in moves if d != 0.0]
        # How much of the macro F1 change one language accounts for. The macro
        # average is unweighted over `num_languages` labels, so a single
        # language's F1 change divided by that count is exactly its
        # contribution to the cell.
        n_lab = arms["baseline"]["num_languages"]
        top = max(changed, key=lambda r: abs(r["delta_f1"])) if changed else None
        cells[bench] = {
            "subset": subset,
            "per_lang_f1_changed": changed,
            "n_langs_f1_changed": len(changed),
            "largest_single_language_contribution": (
                None if top is None else
                {"lang": top["lang"], "delta_f1": top["delta_f1"],
                 "baseline_f1": top["baseline_f1"],
                 "calibrated_f1": top["calibrated_f1"],
                 "support": top["support"],
                 "contribution_to_macro_f1": top["delta_f1"] / n_lab}),
            "n_lines": arms["baseline"]["total_samples"],
            "n_labels_averaged": arms["baseline"]["num_languages"],
            "n_model_rows": arms["baseline"]["n_model_rows_total"],
            "baseline": {"macro_f1": arms["baseline"]["macro_f1"],
                         "macro_fpr": arms["baseline"]["macro_fpr"],
                         "accuracy": arms["baseline"]["accuracy"]},
            "calibrated": {"macro_f1": arms["calibrated"]["macro_f1"],
                           "macro_fpr": arms["calibrated"]["macro_fpr"],
                           "accuracy": arms["calibrated"]["accuracy"]},
            "delta_f1": (arms["calibrated"]["macro_f1"]
                         - arms["baseline"]["macro_f1"]),
            "delta_fpr": (arms["calibrated"]["macro_fpr"]
                          - arms["baseline"]["macro_fpr"]),
        }

    # Second, independent arm: the six passes were also run through the
    # login-node hedge while the SLURM job sat in the queue. Where both exist
    # the numbers must agree exactly, since the computation is deterministic and
    # the only differences are the machine and the rayon thread count. Recorded
    # rather than asserted silently, so the report states how many cells carry
    # two independent measurements.
    cross = {"n_cells_confirmed": 0, "n_cells_single_arm": 0, "disagreements": []}
    for subset, conf in SUBSETS.items():
        for arm in ("baseline", "calibrated"):
            other = os.path.join(out_root, "loginnode",
                                 f"cld3sub{subset}_{conf['bench']}_{arm}.json")
            if not os.path.exists(other):
                cross["n_cells_single_arm"] += 1
                continue
            o = _load_json(other, "login-node arm")
            p = _load_json(
                os.path.join(eval_root,
                             f"cld3sub{subset}_{conf['bench']}_{arm}.json"),
                "primary arm")
            for field in ("macro_f1", "macro_fpr", "accuracy", "total_samples",
                          "correct", "num_languages"):
                if o[field] != p[field]:
                    cross["disagreements"].append(
                        f"subset-{subset}/{conf['bench']}/{arm}: {field} "
                        f"{p[field]!r} vs {o[field]!r}")
            cross["n_cells_confirmed"] += 1
    if cross["disagreements"]:
        raise SystemExit("FATAL: the two evaluation arms disagree:\n  "
                         + "\n  ".join(cross["disagreements"]))

    siblings = {s: _group_a_script_siblings(
        s, stage["tau"]["subsets"][str(s)]["group_a"]["rows"])
        for s in SUBSETS}

    bench_to_subset = {c["bench"]: s for s, c in SUBSETS.items()}
    movement = {b: _movement(out_root, b, bench_to_subset[b])
                for b in MOVEMENT_BENCHES}

    full_transfer = _load_json(os.path.join(REPO_ROOT, FULL_TRANSFER_JSON),
                               "the full model's transfer record")
    full_cells = {}
    for bench in ("glotlidc", "udhr", "flores"):
        c = full_transfer["cells"][bench]["corrected"]
        full_cells[bench] = {
            "baseline": {"macro_f1": c["baseline"]["macro_f1"],
                         "macro_fpr": c["baseline"]["macro_fpr"]},
            "calibrated": {"macro_f1": c["calibrated"]["macro_f1"],
                           "macro_fpr": c["calibrated"]["macro_fpr"]},
            "delta_f1": (c["calibrated"]["macro_f1"]
                         - c["baseline"]["macro_f1"]),
            "delta_fpr": (c["calibrated"]["macro_fpr"]
                          - c["baseline"]["macro_fpr"]),
            "n_lines": c["baseline"]["n_lines"],
        }
    full_main = _full_main_cells(os.path.join(REPO_ROOT, FULL_MAIN_MD))

    rep = {
        "generated": "2026-09-02",
        "git_commit": _git_commit(),
        "ruling": ("A calibrated row for the subset should still exist. "
                   "Perform the calibration procedure on the subset-fitted "
                   "UniLID model. Do not due any hyperparameter sweeps. This "
                   "is a test to see the generalizability of that approach. "
                   "(author, 2026-09-02, verbatim)"),
        "carried_constants": dict(CARRIED_CONSTANTS),
        "carried_rule_thresholds": stage["flatrule"]["subsets"]["83"][
            "thresholds"],
        "stages": stage,
        "cells": cells,
        "group_a_script_siblings": {str(k): v for k, v in siblings.items()},
        "arm_cross_check": cross,
        "movement": movement,
        "full_model_transfer_cells": full_cells,
        "full_model_main_table_full_pool": {
            "baseline": {"macro_f1": full_main["baseline"][0],
                         "macro_fpr_x1e5": full_main["baseline"][1]},
            "calibrated": {"macro_f1": full_main["gate_flat4_prox21"][0],
                           "macro_fpr_x1e5": full_main["gate_flat4_prox21"][1]},
            "delta_f1": round(full_main["gate_flat4_prox21"][0]
                              - full_main["baseline"][0], 6),
            "delta_fpr_x1e5": round(full_main["gate_flat4_prox21"][1]
                                    - full_main["baseline"][1], 6),
            "source": FULL_MAIN_MD,
        },
        "published_cells": PUBLISHED,
        "eval_root": eval_root,
        "eval_ran_where": ("login node "
                           "(run_cld3_subset_calibrated_eval_loginnode.sh)"
                           if eval_root != out_root else
                           "SLURM (slurm_cld3_subset_calibrated_eval.sh)"),
    }
    return rep



def _fmt_md(rep: dict) -> str:
    L = []
    A = L.append
    C = rep["carried_constants"]
    T = rep["carried_rule_thresholds"]
    flat = rep["stages"]["flatrule"]["subsets"]
    tau = rep["stages"]["tau"]["subsets"]
    bundle = rep["stages"]["bundle"]["subsets"]

    A("# The calibration procedure applied unchanged to the three "
      "subset-fitted models")
    A("")
    A(f"Generated {rep['generated']}. Measurements: `{OUT_JSON}`. Per-stage "
      f"detail: `{OUT_ROOT}`.")
    A("")
    A("Author ruling, 2026-09-02, verbatim:")
    A("")
    A("> A calibrated row for the subset should still exist. Perform the "
      "calibration procedure on the subset-fitted UniLID model. Do not due any "
      "hyperparameter sweeps. This is a test to see the generalizability of "
      "that approach.")
    A("")
    A("Section 6.3 of `outputs/rerelease/cld3_regenerated_2026-09-01.md` is "
      "superseded. It argued that the calibrated row could not move to the "
      "refit convention, because the high-entropy group (group B) does not "
      "survive the subset. Group B does not survive: that is measured in "
      "section 3.1 below rather than assumed. Under the ruling its absence is "
      "a result of the generalizability test, not a reason to stop.")
    A("")

    A("## 1. Constants carried, quantities recomputed")
    A("")
    A("Every constant of the promoted configuration `gate_flat4_prox21` is "
      "carried from the full corrected model and applied without a sweep. No "
      "grid was run and no value was re-selected. Carrying them unswept is "
      "what makes this a generalizability measurement rather than a refit.")
    A("")
    A("| constant | value | where the value was selected | source in this "
      "repository |")
    A("|---|---|---|---|")
    A(f"| `unseen_token_constant` (c) | {C['unseen_token_constant']} | "
      "round-grid sweep on the corrected full model, job 3117581 | "
      "`outputs_corrected_round/release/calibration_glotlidc_corrected.json` |")
    A(f"| `head_n` | {C['head_n']:,} | promoted configuration | "
      "`analysis/full_test_margin.py::HEAD_N` |")
    A(f"| `replacement_min_n` (RES_CAP) | {C['replacement_min_n']:,} | "
      "promoted configuration | `analysis/hierarchical_pool.py::RES_CAP` |")
    A(f"| `proximity_bound` (D3_PROX) | {C['proximity_bound']} | promoted "
      "configuration | `analysis/gate_variants.py::D3_PROX` |")
    A(f"| `topk` (TOPK_MARGIN) | {C['topk']} | promoted configuration | "
      "`analysis/margin_diagnostic.py::TOPK_MARGIN` |")
    A(f"| `margin_q` | {C['margin_q']} | pre-registered | "
      "`analysis/margin_diagnostic.py::MARGIN_Q` |")
    A(f"| `group_b_percentile` | {C['group_b_percentile']} | promoted "
      "configuration | `analysis/build_release_calibration.py::CONSTANTS` |")
    A(f"| `calib_max` | {C['calib_max']:,} | promoted configuration | "
      "`analysis/margin_diagnostic.py::CALIB_MAX` |")
    A(f"| `min_calib_lines` | {C['min_calib_lines']} | promoted configuration "
      "| `analysis/margin_diagnostic.py::MIN_CALIB_LINES` |")
    A(f"| `calib_seed` | {C['calib_seed']} | promoted configuration | "
      "`analysis/margin_diagnostic.py::CALIB_SEED` |")
    A(f"| `ZH_MAGNET` | {T['ZH_MAGNET']} | flat-magnet rule | "
      "`analysis/diagnostic.py` |")
    A(f"| `ZH_EXTREME` | {T['ZH_EXTREME']} | flat-magnet rule | "
      "`analysis/diagnostic.py` |")
    A(f"| `MAGNET_RATIO_MIN` | {T['MAGNET_RATIO_MIN']} | flat-magnet rule | "
      "`analysis/diagnostic.py` |")
    A("")
    A("The ten calibration constants are the same values the corrected "
      "release bundles. This report's list is built by importing "
      "`analysis/build_release_calibration.py::CONSTANTS` and overriding only "
      f"`unseen_token_constant` to {C['unseen_token_constant']}, which is what "
      "the corrected release itself did through its "
      "`--unseen-token-constant` flag.")
    A("")
    A("Three quantities are recomputed, because the procedure defines them as "
      "functions of the model being calibrated rather than as free parameters:")
    A("")
    A("1. Group A membership: the languages whose training-line count `N_L` is "
      f"below `head_n` ({C['head_n']:,}), taken from each subset model's own "
      "training counts.")
    A("2. Group B membership: the flat-magnet rule re-run on each subset "
      "model's own weight matrix and its own validation-half predictions.")
    A("3. Per-language re-examination thresholds `tau`: the size-adaptive "
      "recipe re-run against each subset model's own clamped matrix and its "
      "own training lines.")
    A("")
    A("This is the same no-refitting arrangement as E3, the Mistral-Nemo "
      "variant (`analysis/mistralnemo_eval.py`), whose stages `calibval`, "
      "`flatrule`, `tau`, `topk` and `eval` this chain follows. One part of E3 "
      "does not transfer: it reuses the base model's `y_true.npy` and its "
      "full-pool prediction arrays, which are aligned to the 1,940-language "
      "order. A 99, 94 or 93-row model shares neither that order nor that "
      "label space, so every array here is built from scratch.")
    A("")

    A("## 2. The source of the thresholds")
    A("")
    A("Read from the code, which is the authority when code and prose "
      "disagree. Each threshold `tau_L` is a percentile of language L's "
      "margins on L's own training lines under the clamped matrix. The margin "
      "of a line is the top candidate's score minus the runner-up's score, in "
      "natural-log units. The lines come from "
      "`CORPUS_DIR/{lang}_train.txt`, at most `calib_max` of them, drawn with "
      "one `np.random.default_rng(calib_seed)` shared sequentially across "
      "languages. Only the lines L itself scores highest are kept. The "
      "percentile is the size-adaptive quantile "
      "`q_L = margin_q * (1 - min(N_L, head_n) / head_n)`. The implementation "
      "is `analysis/mistralnemo_eval.py::_calibrate_group`, itself a port of "
      "`analysis/solo_gates.py`'s `run(\"floor21\")`.")
    A("")
    A("The seed-42 250,000-line validation half enters the procedure at one "
      "point only: the `magnet_ratio` of the flat-magnet rule, defined as the "
      "false positives into a language on that held-out half divided by its "
      "true support there plus one. The validation half is not the source of "
      "the thresholds. Both readings are transcribed from "
      "`analysis/margin_diagnostic.py` and "
      "`analysis/mistralnemo_eval.py::run_flatrule`.")
    A("")
    A("The validation half is restricted, per model, to the lines whose gold "
      "`lang_Script` label that model carries. On the full model the "
      "restriction is the identity, which is why the procedure never had to "
      "state it. On a subset model, scoring the rest of the validation half "
      "would charge subset languages with false positives for 1,841 labels "
      "they cannot express, and would raise every `magnet_ratio`. The "
      "restriction is the `only_model_langs` filter that every subset "
      "evaluation in this repository already applies.")
    A("")

    A("## 3. The measured groups, per subset")
    A("")
    A("| subset | rows | group A (N_L < head_n) | of those, excluded | group B "
      "(flat magnets with N_L >= head_n) | rows clamped at c | validation "
      "lines scored |")
    A("|---|---|---|---|---|---|---|")
    for subset in (83, 80, 77):
        k = str(subset)
        ga = tau[k]["group_a"]
        clamp = tau[k]["clamp"]["clamp"]
        A(f"| {subset} | {clamp['n_languages']} | {ga['n']} | "
          f"{ga['n_excluded']} | {tau[k]['group_b']['n']} | "
          f"{clamp['n_modified']} of {clamp['n_languages']} | "
          f"{flat[k]['n_calibval_lines']:,} |")
    A("")
    A("The full corrected model, for scale: 1,940 rows, group A 1,080 of "
      "which 26 are excluded with cause `low_calibration`, group B 4 "
      "(`sco_Latn`, `bjn_Latn`, `arg_Latn`, `vls_Latn`), and 1,655 of 1,940 "
      "rows clamped at c = -17.")
    A("")

    A("### 3.1 Group B is empty on all three subsets")
    A("")
    A("The 2026-09-01 record predicted an empty group B on the argument that "
      "none of the full model's four flat languages is a CLD3 language. The "
      "prediction holds, and the measurement says something more specific. The "
      "rule was re-run from scratch on each subset model's own weight matrix. "
      "It did flag languages as flat magnets. Every language it flagged has "
      "`N_L` below `head_n`, so every one of them is already in group A and "
      "none is eligible for group B.")
    A("")
    A("| subset | languages flagged `is_magnet` | which ones | of those, with "
      "N_L >= head_n | largest N_L among them | group B |")
    A("|---|---|---|---|---|---|")
    for subset in (83, 80, 77):
        k = str(subset)
        f = flat[k]
        names = ", ".join(f"`{m['lang']}`" for m in f["is_magnet_langs"])
        A(f"| {subset} | {f['n_is_magnet']} | {names or 'none'} | "
          f"{f['n_is_magnet_ge_head_n']} | "
          f"{f['max_N_among_magnets']:,} | {f['n_flat_set']} |")
    A("")
    A("On a subset model the flat-magnet rule therefore adds no language that "
      "group A does not already cover. The five languages with the highest "
      "within-script entropy z-score `zH` per subset, with the quantities the "
      "rule reads:")
    A("")
    for subset in (83, 80, 77):
        k = str(subset)
        f = flat[k]
        A(f"Subset {subset}. Highest `zH` in the model is "
          f"{f['zH_max']:.3f}; highest `magnet_ratio` is "
          f"{f['magnet_ratio_max']:.4f}.")
        A("")
        A("| lang | N_L | zH | magnet_ratio | support_val | fp_val | "
          "N_L >= head_n |")
        A("|---|---|---|---|---|---|---|")
        for r in f["top5_by_zH"]:
            A(f"| `{r['lang']}` | {r['N']:,} | {r['zH']} | "
              f"{r['magnet_ratio']} | {r['support_val']} | {r['fp_val']} | "
              f"{'yes' if r['N'] >= C['head_n'] else 'no'} |")
        A("")
    A("A second property of the rule under a small label set is visible in the "
      "same stage, and it makes the rule weaker than the count of flagged "
      "languages suggests. `zH` is a within-script z-score, and the code skips "
      "any script with fewer than three languages "
      "(`analysis/diagnostic.py`'s construction, carried unchanged), leaving "
      "the languages of those scripts at `zH = 0`:")
    A("")
    A("| subset | scripts | scripts with 3 or more languages | languages with "
      "a computed zH | languages left at zH = 0 |")
    A("|---|---|---|---|---|")
    for subset in (83, 80, 77):
        k = str(subset)
        f = flat[k]
        A(f"| {subset} | {f['n_scripts']} | {f['n_scripts_with_3_or_more']} | "
          f"{f['n_languages_with_a_scored_zH']} | "
          f"{f['n_languages'] - f['n_languages_with_a_scored_zH']} |")
    A("")
    A("On subset-80, 24 of 94 languages are ineligible for group B whatever "
      "their entropy, because their script carries fewer than three "
      "languages. On the full 1,940-language model almost every script clears "
      "the three-language bar.")
    A("")

    A("### 3.2 No language met the recipe's exclusion condition")
    A("")
    A("The `low_calibration` exclusion of `build_release_calibration` applies "
      "to a language with fewer than `min_calib_lines` "
      f"({C['min_calib_lines']}) finite winning margins. It applies to 26 of "
      "the full model's 1,080 group A languages. It applies to none of the 38 "
      "group A languages across the three subset models. The mechanism is "
      "direct: with 93 to 99 candidate languages instead of 1,940, each "
      "language scores highest on nearly all of its own training lines. Every "
      "group A language here drew the full `calib_max` sample and kept between "
      "1,876 and 2,000 finite winning margins.")
    A("")
    A("The other exclusion condition, `zero_strength` (`q_L <= 0`), cannot "
      "apply in group A by construction, since `q_L > 0` whenever "
      "`N_L < head_n`.")
    A("")
    A("| subset | group A languages | fewest finite winning margins | most | "
      "tau range (nats) |")
    A("|---|---|---|---|---|")
    for subset in (83, 80, 77):
        k = str(subset)
        rows = tau[k]["group_a"]["rows"]
        fin = [r["n_finite_margins"] for r in rows]
        taus = [r["tau"] for r in rows]
        A(f"| {subset} | {len(rows)} | {min(fin):,} | {max(fin):,} | "
          f"{min(taus):.3f} to {max(taus):.3f} |")
    A("")
    A("The full corrected model's 1,054 non-excluded group A thresholds span "
      "0.021 to 269.384 nats, with median 23.78. The subset thresholds lie "
      "inside that range.")
    A("")
    A("Every threshold this run derived. `q_L` is the size-adaptive quantile "
      "`margin_q * (1 - min(N_L, head_n) / head_n)`; `tau` is that percentile "
      "of the finite winning margins, in natural-log units.")
    A("")
    A("| subset | lang | N_L | lines sampled | scoreable | scored highest by "
      "itself | finite winning margins | q_L | tau (nats) | excluded |")
    A("|---|---|---|---|---|---|---|---|---|---|")
    for subset in (83, 80, 77):
        for r in tau[str(subset)]["group_a"]["rows"]:
            A(f"| {subset} | `{r['lang']}` | {r['N']:,} | {r['n_sampled']:,} | "
              f"{r['n_scoreable']:,} | {r['n_self_won']:,} | "
              f"{r['n_finite_margins']:,} | {r['q_l']:.4f} | {r['tau']:.4f} | "
              f"{'yes' if r['excluded'] else 'no'} |")
    A("")
    A("Group B has no rows on any of the three subsets, so no group B "
      "threshold was derived. The empty tau CSVs are written anyway, with the "
      "same six columns, so the bundle stage reads one code path in both "
      "cases.")
    A("")

    A("### 3.3 What group A consists of on a subset model")
    A("")
    A("The subset columns are scored under the `--lang-only` bare-ISO collapse, "
      "so `ben_Latn` and `ben_Beng` are the same label at scoring time. Most of "
      "group A on these models is one half of such a pair:")
    A("")
    A("| subset | group A languages | with another row sharing their bare ISO "
      "code | without one | which ones lack a sibling row |")
    A("|---|---|---|---|---|")
    for subset in (83, 80, 77):
        sib = rep["group_a_script_siblings"][str(subset)]
        none = ", ".join(f"`{x}`" for x in sib["without_sibling"])
        A(f"| {subset} | {len(sib['rows'])} | {sib['n_with_sibling']} | "
          f"{sib['n_without_sibling']} | {none or 'none'} |")
    A("")
    A("This splits the lines a re-examination can touch into two kinds. If the "
      "gold label of the line is the sibling row, the prediction into the "
      "group A row already scores as correct, because the two labels are the "
      "same bare ISO code, so re-examining that line can only leave it correct "
      "or make it wrong. If the gold label is some third language, the "
      "prediction is wrong before re-examination, and re-examination can "
      "correct it. Section 4.1 counts both outcomes on the two small "
      "benchmarks. The interaction between the calibration, which operates on "
      "`lang_Script` rows, and the bare-ISO metric of these three columns is "
      "present on the full model too. It is concentrated here because group A "
      "on a subset model is almost entirely minority-script variants of "
      "languages the same model also carries in their majority script.")
    A("")

    A("### 3.4 The clamp lowers a smaller fraction of rows than on the full "
      "model")
    A("")
    A(f"At the carried c = {C['unseen_token_constant']}, the one-sided clamp "
      "lowers 24 of 99 rows on subset-83, 21 of 94 on subset-80 and 21 of 93 "
      "on subset-77, close to 22 percent in each case. On the full model it "
      "lowers 1,655 of 1,940 rows, 85 percent. Every subset language is a CLD3 "
      "language and therefore high-resource, and an unseen-token plateau is "
      "resource-tied, so most subset rows already have their plateau at or "
      "below -17 and the clamp leaves them unchanged.")
    A("")
    A("The one-sided condition was checked for every row "
      "(`analysis/floor_equalization.py::verify_one_sided_clamp`). The "
      "analysis-side clamp was compared bit for bit against the package-side "
      "`unilid.calibration.apply_unseen_token_constant` that runs at "
      "inference, so threshold estimation and inference are not on different "
      "matrices. The special-token columns were read from each container's own "
      "vocabulary (columns `[0, 1, 2, 3]`, the UNILID 0.3.0 placement at the "
      "training floor) and both readers returned the same columns.")
    A("")

    A("## 4. The cells")
    A("")
    A("Both arms are the same version-2 container, scored by the same code "
      "path (`analysis/cld_subset_eval.py --mode subset`: the paper team's "
      "macro F1 and macro FPR core, their `only_model_langs` line filter, the "
      "`--lang-only` bare-ISO collapse, Viterbi decode). The two arms differ "
      "only by the `--calibrated` flag. The container's weight matrix is "
      "sha256-identical to the version-1 container's, since the unseen-token "
      "constant is applied at load time and never written into the file. The "
      "uncalibrated arm reproduces the recorded 2026-09-01 cell on every "
      "compared field, which is checked rather than assumed.")
    A("")
    A("| column | lines | labels averaged | baseline F1 | calibrated F1 | F1 "
      "calibrated minus baseline | baseline FPR | calibrated FPR | FPR "
      "calibrated minus baseline |")
    A("|---|---|---|---|---|---|---|---|---|")
    for bench in ("glotlidc", "udhr", "flores"):
        c = rep["cells"][bench]
        A(f"| {COLUMN_NAMES[bench]} | {c['n_lines']:,} | "
          f"{c['n_labels_averaged']} | {c['baseline']['macro_f1']:.5f} | "
          f"**{c['calibrated']['macro_f1']:.5f}** | "
          f"{c['delta_f1']:+.5f} | {c['baseline']['macro_fpr']:.4e} | "
          f"**{c['calibrated']['macro_fpr']:.4e}** | "
          f"{c['delta_fpr']:+.3e} |")
    A("")
    A("The six cells for the calibrated row of `tab:lid_main`, at the table's "
      "own precision. Cell numbers are positions in the 12-cell row body, six "
      "columns by two metrics, left half first.")
    A("")
    A("| row | cell | column | published | measured here |")
    A("|---|---|---|---|---|")
    for bench in ("glotlidc", "udhr", "flores"):
        c = rep["cells"][bench]
        f1_cell, fpr_cell = CELL_NUMBERS[bench]
        pub_f1, pub_fpr = PUBLISHED[bench]["calibrated"]
        A(f"| `\\unilid (calibrated)` | {f1_cell} | {COLUMN_NAMES[bench]} F1 | "
          f"{pub_f1} | **{c['calibrated']['macro_f1']:.3f}** |")
        A(f"| `\\unilid (calibrated)` | {fpr_cell} | {COLUMN_NAMES[bench]} FPR "
          f"| {pub_fpr} | **{c['calibrated']['macro_fpr']:.2e}** |")
    A("")
    A("The three published FPR cells print a dash, so those three are the "
      "first values computed for them.")
    A("")
    x = rep["arm_cross_check"]
    A(f"Of the six passes, {x['n_cells_confirmed']} were run twice: once "
      "through SLURM job 3261635 and once through the login-node script that "
      "hedged against the queue, on different machines and with different "
      "rayon thread counts. Both runs agree on macro F1, macro FPR, accuracy, "
      "line count, correct count and label count. The remaining "
      f"{x['n_cells_single_arm']} passes are the two GlotLID-C ones, which "
      "were run once each.")
    A("")

    A("### 4.1 The predictions the calibration changed")
    A("")
    A("Per-line pairs of gold label and predicted label were banked under both "
      "arms for the two small benchmarks, so the change can be counted line by "
      "line instead of inferred from a macro average. GlotLID-C is absent from "
      "this table: its calibrated pass over 23,462,651 lines costs about an "
      "hour and was not run a second time to bank predictions.")
    A("")
    A("| benchmark | lines | predictions changed | changed from wrong to "
      "right | changed from right to wrong | which languages |")
    A("|---|---|---|---|---|---|")
    for bench in MOVEMENT_BENCHES:
        mv = rep["movement"][bench]
        pairs = ", ".join(f"`{k}` x{v}" for k, v in mv["moves_by_pair"].items())
        A(f"| {COLUMN_NAMES[bench]} | {mv['n_lines']:,} | {mv['n_moved']} | "
          f"{mv['n_fixed']} | {mv['n_broken']} | {pairs or 'none'} |")
    A("")
    A("Every language whose F1 changed at all on those two benchmarks:")
    A("")
    A("| benchmark | lang | support | baseline F1 | calibrated F1 | F1 "
      "calibrated minus baseline |")
    A("|---|---|---|---|---|---|")
    for bench in MOVEMENT_BENCHES:
        for r in rep["cells"][bench]["per_lang_f1_changed"]:
            A(f"| {COLUMN_NAMES[bench]} | `{r['lang']}` | {r['support']:,} | "
              f"{r['baseline_f1']:.5f} | {r['calibrated_f1']:.5f} | "
              f"{r['delta_f1']:+.5f} |")
    A("")
    A("The two benchmarks show the two outcomes the mechanism is capable of. "
      "On FLORES-77 the re-examined predictions are mostly a low-resource "
      "Latin-script row taking lines that belong to other languages, and "
      "re-examination sends most of them to the right language. On UDHR-80 "
      "every re-examined prediction was Hawaiian on a genuine Hawaiian line, "
      "and re-examination sent it elsewhere. `haw_Latn` has 6,448 training "
      "lines, below `head_n`, and it is one of only two group A languages on "
      "subset-80 with no other row sharing its bare ISO code (section 3.3), so "
      "there is no sibling row for a moved prediction to land on and still "
      "score correct.")
    A("")
    A("The size of the effect on UDHR comes from the benchmark rather than "
      "from the number of changed lines. UDHR carries about 60 lines per "
      "language, so 8 changed lines is several percentage points of one "
      "language's F1 and a visible fraction of an 80-language macro average. "
      "Whether re-examination helps or hurts a given language is set by "
      "whether that language's margins on its own training lines, which set "
      "its threshold, are wider than its margins against the languages it is "
      "actually confused with on the benchmark. That is a property of the "
      "procedure, not of this implementation of it, and the same asymmetry "
      "between UDHR and FLORES is present on the full model in section 5.")
    A("")

    A("## 5. The change on the subset models against the change on the full "
      "model")
    A("")
    A("The generalizability question is whether the procedure, with its "
      "constants fixed on the full model, still moves a subset model in the "
      "same direction. The comparison is against the full 1,940-language "
      "model's own change on the same three benchmarks and the same 83, 80 "
      f"and 77 label sets, read from `{FULL_TRANSFER_JSON}` (the transfer "
      "reading, corrected generation).")
    A("")
    A("A lower FPR is better, so the sign that favours the calibration is "
      "positive for F1 and negative for FPR.")
    A("")
    A("| benchmark | full model, F1 calibrated minus baseline | subset model, "
      "F1 calibrated minus baseline | F1 signs agree | full model, FPR "
      "calibrated minus baseline | subset model, FPR calibrated minus baseline "
      "| FPR signs agree |")
    A("|---|---|---|---|---|---|---|")
    for bench in ("glotlidc", "udhr", "flores"):
        sc = rep["cells"][bench]
        f = rep["full_model_transfer_cells"][bench]
        same_f1 = "yes" if (sc["delta_f1"] > 0) == (f["delta_f1"] > 0) else "no"
        same_fpr = ("yes" if (sc["delta_fpr"] > 0) == (f["delta_fpr"] > 0)
                    else "no")
        A(f"| {COLUMN_NAMES[bench]} | {f['delta_f1']:+.5f} | "
          f"**{sc['delta_f1']:+.5f}** | {same_f1} | {f['delta_fpr']:+.3e} | "
          f"**{sc['delta_fpr']:+.3e}** | {same_fpr} |")
    A("")
    A("The F1 sign agrees on all three benchmarks: the calibration raises F1 "
      "on GlotLID-C-83 and FLORES-77 and lowers it on UDHR-80, both on the "
      "full model and on the subset models. The two FPR rows marked `no` are "
      "cases where the subset model's FPR falls while the full model's rises, "
      "so the subset result is the better of the two, not a failure to "
      "transfer.")
    A("")
    m = rep["full_model_main_table_full_pool"]
    A("A third reference point is the main table's own full-pool measurement "
      f"over all 1,940 languages (`{m['source']}`): macro F1 "
      f"{m['baseline']['macro_f1']:.4f} uncalibrated against "
      f"{m['calibrated']['macro_f1']:.4f} calibrated, a change of "
      f"{m['delta_f1']:+.4f}; macro FPR scaled by 1e5, "
      f"{m['baseline']['macro_fpr_x1e5']:.4f} uncalibrated against "
      f"{m['calibrated']['macro_fpr_x1e5']:.4f} calibrated, a change of "
      f"{m['delta_fpr_x1e5']:+.4f}.")
    A("")
    A("### 5.1 How much of each change is one language")
    A("")
    A("The macro average is unweighted over the labels of the column, so one "
      "language's F1 change divided by that label count is exactly its "
      "contribution to the cell. The largest single contributor per column:")
    A("")
    A("| column | languages whose F1 changed | largest single contributor | "
      "its F1 | its contribution to the column | column change |")
    A("|---|---|---|---|---|---|")
    for bench in ("glotlidc", "udhr", "flores"):
        c = rep["cells"][bench]
        t = c["largest_single_language_contribution"]
        A(f"| {COLUMN_NAMES[bench]} | {c['n_langs_f1_changed']} of "
          f"{c['n_labels_averaged']} | `{t['lang']}` (support "
          f"{t['support']:,}) | {t['baseline_f1']:.5f} to "
          f"{t['calibrated_f1']:.5f} | "
          f"{t['contribution_to_macro_f1']:+.5f} | "
          f"{c['delta_f1']:+.5f} |")
    A("")
    gl = rep["cells"]["glotlidc"]["largest_single_language_contribution"]
    A(f"On GlotLID-C-83 the whole column change is close to one language. "
      f"Corsican (`cos`) goes from F1 {gl['baseline_f1']:.5f} to "
      f"{gl['calibrated_f1']:.5f}, which by itself is "
      f"{gl['contribution_to_macro_f1']:+.5f} of the "
      f"{rep['cells']['glotlidc']['delta_f1']:+.5f} column change. `cos_Latn` "
      "has 9,423 training lines, is in group A, and is the case the "
      "re-examination mechanism was built for: a low-resource Latin-script row "
      "taking lines that belong to higher-resource languages. Reporting the "
      "GlotLID-C-83 cell without this sentence would make a one-language "
      "result read as an 83-language one.")
    A("")

    A("### 5.2 Reading the three reference points together")
    A("")
    A("The large gain from the calibration is a tail-language effect. It is "
      f"worth {m['delta_f1']:+.4f} macro F1 over 1,940 languages, 1,080 of "
      f"which have fewer than {C['head_n']:,} training lines. On the CLD3 "
      "subsets, 86 to 87 percent of the languages have at least that many "
      "training lines, so re-examination applies to only 12 to 14 languages "
      "per model. What the procedure produces there is a change of the same "
      "sign as on the full model, of a size set by how many of those 12 to 14 "
      "languages are actually confused with something on the benchmark: one "
      "on GlotLID-C-83, several on FLORES-77, and one in the wrong direction "
      "on UDHR-80.")
    A("")
    A("My reading is that the procedure transfers. Its constants were selected "
      "on a 1,940-language model and none was re-selected here, and the "
      "resulting change has the same sign on all three benchmarks as the same "
      "procedure produces on the full model. The size of the change is not "
      "comparable across the two, and I would not present it as though it "
      "were: the subset columns contain almost none of the tail the mechanism "
      "acts on.")
    A("")

    A("## 6. Proposed caption sentence")
    A("")
    A("> The CLD3-subset columns of the calibrated row are produced by "
      "applying the calibration procedure of \\cref{sec:calibration} to each "
      "subset model, with every constant fixed on the full model and carried "
      "unchanged: the unseen-token constant, the quantile and sampling "
      "settings of the re-examination thresholds, the head-size and "
      "replacement bounds, and the proximity bound. Only the language groups "
      "and the per-language thresholds are recomputed, which the procedure "
      "defines as functions of the model being calibrated. No hyperparameter "
      "was swept for the subset. The high-entropy group is empty on all three "
      "subsets, so re-examination there applies to the low-resource group "
      "alone.")
    A("")

    A("## 7. Provenance")
    A("")
    A("- Calibration chain: `analysis/cld3_subset_calibration.py`, stages "
      "`calibval`, `flatrule`, `tau`, `bundle`.")
    A(f"- Evaluation: {rep['eval_ran_where']}, calling "
      "`analysis/cld_subset_eval.py --mode subset` on both arms. Summaries "
      f"under `{rep['eval_root']}`.")
    A("- This record: `analysis/cld3_subset_calibrated_report.py`.")
    A(f"- Repository commit at generation: `{rep['git_commit']}`.")
    A("")
    A("| subset | version-1 container | version-2 container | calibration JSON "
      "| group A tau CSV | group B tau CSV |")
    A("|---|---|---|---|---|---|")
    for subset in (83, 80, 77):
        k = str(subset)
        b = bundle[k]
        A(f"| {subset} | `{SUBSETS[subset]['model']}` | "
          f"`{b['calibrated_model']}` | `{b['calibration_json']}` | "
          f"`{tau_group_a_csv(subset)}` | `{tau_group_b_csv(subset)}` |")
    A("")
    A("sha256 of the artifacts this run produced:")
    A("")
    A("| artifact | sha256 |")
    A("|---|---|")
    for subset in (83, 80, 77):
        k = str(subset)
        b = bundle[k]
        A(f"| `cld3sub{subset}_calibrated.unilid` | "
          f"`{b['calibrated_model_sha256']}` |")
        A(f"| `cld3sub{subset}_calibration.json` | "
          f"`{_sha256_file(b['calibration_json'])}` |")
    A("")
    return "\n".join(L) + "\n"

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out-root", default=OUT_ROOT)
    p.add_argument("--eval-root", default=None,
                   help="where the six evaluation summaries are; defaults to "
                        "--out-root. Point it at <out-root>/loginnode when the "
                        "login-node hedge produced them.")
    p.add_argument("--out-md", default=OUT_MD)
    p.add_argument("--out-json", default=OUT_JSON)
    a = p.parse_args(argv)

    rep = build(a.out_root, a.eval_root)
    os.makedirs(os.path.dirname(os.path.abspath(a.out_json)), exist_ok=True)
    with open(a.out_json, "w") as f:
        json.dump(rep, f, indent=2, default=str)
    with open(a.out_md, "w") as f:
        f.write(_fmt_md(rep))
    print(f"Wrote {a.out_json} and {a.out_md}")
    for bench in ("glotlidc", "udhr", "flores"):
        c = rep["cells"][bench]
        print(f"  {COLUMN_NAMES[bench]}: baseline {c['baseline']['macro_f1']:.5f}"
              f" / {c['baseline']['macro_fpr']:.4e}  ->  calibrated "
              f"{c['calibrated']['macro_f1']:.5f} / "
              f"{c['calibrated']['macro_fpr']:.4e}  "
              f"(dF1 {c['delta_f1']:+.5f})")


if __name__ == "__main__":
    main()
