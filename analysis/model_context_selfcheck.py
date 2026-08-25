"""Self-check for analysis/model_context.py: trigger every refusal.

The guard exists to stop a second model's results being written over the
released model's artifacts, which are reachable through symlinks into the
durable store. A guard that is only read and never fired is not evidence, so
this fires each branch and reports what it caught. Run it after any change to
the resolution rules.

  python -m analysis.model_context_selfcheck
"""
from __future__ import annotations

import json
import os
import sys
import tempfile

from analysis.model_context import (DEFAULT_OUT_ROOT, UnsafeModelContext,
                                    default_model_path, default_scratch_dir,
                                    resolve, resolve_out_root,
                                    store_backed_entries)

CORRECTED = ("/capstor/scratch/cscs/cmeister747/unilid_analysis/corrected/"
             "glotlidc_corrected.unilid")
A_STORE_FILE = ("/capstor/store/cscs/swissai/a0229/cmeister/unilid_analysis/"
                "glotlidc.unilid")

# The Mistral-Nemo chain's own corrected pair. That chain has a third and a
# fourth axis the others do not: nine repo-side artifacts under --out-dir, and
# FLOOR_TARGET, a clamp constant whose module default belongs to the released
# chain and whose corrected value (-17.0) was selected by measurement.
NEMO_CORRECTED = ("/capstor/scratch/cscs/cmeister747/unilid_analysis/corrected/"
                  "glotlid_mistralnemo_fp64_corrected.unilid")
NEMO_SCRATCH_CORRECTED = ("/capstor/scratch/cscs/cmeister747/unilid_analysis/"
                          "full_test_eval_mistralnemo_corrected")
# The corrected BASE model's scratch root, whose fingerprint_floor21.json is the
# record of the round-grid sweep that selected the corrected clamp.
BASE_SCRATCH_CORRECTED = ("/capstor/scratch/cscs/cmeister747/unilid_analysis/"
                          "full_test_eval_corrected")
# A root that is neither the default output root nor inside it nor store-backed,
# used only to get PAST the out-root rule so the floor-target rule is what fires.
# Nothing is ever written there: both refusals happen inside configure(), before
# any stage function runs.
NEMO_OUT_PROBE = ("/capstor/scratch/cscs/cmeister747/unilid_analysis/"
                  "outputs_selfcheck_probe")


def _expect_refusal(label, **kwargs) -> bool:
    try:
        resolve(**kwargs)
    except UnsafeModelContext as e:
        print(f"  PASS  {label}\n          {str(e).splitlines()[0]}")
        return True
    print(f"  FAIL  {label}: the call was allowed")
    return False


def _expect_out_root_refusal(label, ctx, out_dir) -> bool:
    try:
        resolve_out_root(ctx, out_dir, purpose="self-check")
    except UnsafeModelContext as e:
        print(f"  PASS  {label}\n          {str(e).splitlines()[0]}")
        return True
    print(f"  FAIL  {label}: the call was allowed")
    return False


def _expect_out_root_allowed(label, ctx, out_dir, expected) -> bool:
    try:
        root = resolve_out_root(ctx, out_dir, purpose="self-check")
    except UnsafeModelContext as e:
        print(f"  FAIL  {label}: refused unexpectedly ({e})")
        return False
    if root != expected:
        print(f"  FAIL  {label}: returned {root!r}, expected {expected!r}")
        return False
    print(f"  PASS  {label} (root {root!r})")
    return True


def _expect(label, got, want) -> bool:
    if got != want:
        print(f"  FAIL  {label}: got {got!r}, expected {want!r}")
        return False
    print(f"  PASS  {label} ({got!r})")
    return True


def _check_published_cell_gates():
    """analysis/paper_breakdowns.py's two reproduction gates compare against the
    cells published in paper/submission.tex, which the RELEASED model produced.
    For the released model they bind: a MISMATCH withholds the affected .tex and
    exits 1. For any other model a difference is the expected result of the run,
    so the .tex must still be written and the exit code must not signal failure --
    otherwise a corrected-model run can never produce the corrected tables, which
    are its whole deliverable. Both decisions are pure functions precisely so this
    can be checked without running the script."""
    from analysis.paper_breakdowns import (_breakdowns_exit_code, _publish_tex)
    print("\npublished-cell reproduction gates bind the released model only "
          "(analysis.paper_breakdowns):")
    return [
        _expect("released model, gate passed -> .tex written",
                _publish_tex(True, True), True),
        _expect("released model, gate MISMATCH -> .tex withheld",
                _publish_tex(False, True), False),
        _expect("other model, comparison differs -> .tex still written",
                _publish_tex(False, False), True),
        _expect("released model, gate MISMATCH -> exit 1",
                _breakdowns_exit_code(False, True, True, True), 1),
        _expect("other model, both comparisons differ -> exit 0",
                _breakdowns_exit_code(False, False, True, False), 0),
        _expect("other model, label-basis diagnostic failed -> exit 1",
                _breakdowns_exit_code(True, True, False, False), 1),
    ]


def _check_mistralnemo_chain():
    """analysis/mistralnemo_eval.py's two extra axes.

    (1) Its nine repo-side artifacts (the flat-set CSV/MD, both tau CSVs, the tau
    build note, the eval .md/.tex and the two per-language F1 CSVs) were bare
    "outputs/..." literals, so the flatrule/tau/eval stages overwrote the
    RELEASED model's E3 record in place for any --model -- and outputs/tables/
    mistralnemo_eval.md is cited by paper/tables/calibrated_nemo.tex. They now go
    through resolve_out_root, and out_path()'s default must still spell every one
    of them exactly as the pre---out-dir literal did, or a default run would
    quietly write somewhere new.

    (2) FLOOR_TARGET was imported from analysis.full_test_floor21 (-21.0) and used
    unchanged by tau/topk/eval. That is a measured per-model selection, not a
    universal constant: the corrected chain's own sweep selected -17.0. A
    corrected run inheriting -21.0 would have produced floor-21 and gated numbers
    for the wrong clamp, indistinguishable in the tables from a correct run.
    _resolve_floor_target is a pure function of (context, value, base scratch)
    precisely so every branch can be fired here without running a stage."""
    from analysis import mistralnemo_eval as mne

    print("\noutput-root and floor-target rules of the Mistral-Nemo chain "
          "(analysis.mistralnemo_eval):")
    out = []

    # (1) the nine paths, plus the degeneracy scan it READS, unchanged by default.
    expected_default = {
        "flat_set_csv": "outputs/diagnostic/mistralnemo_flat_set.csv",
        "flat_set_md": "outputs/tables/mistralnemo_flat_set.md",
        "tau_floor21_csv": "outputs/diagnostic/tau_mistralnemo_floor21_gate.csv",
        "tau_flat_csv": "outputs/diagnostic/tau_mistralnemo_flat.csv",
        "tau_build_md": "outputs/tables/mistralnemo_tau_build.md",
        "eval_md": "outputs/tables/mistralnemo_eval.md",
        "eval_tex": "outputs/tables/mistralnemo_eval.tex",
        "per_lang_fullpool_csv":
            "outputs/diagnostic/mistralnemo_per_lang_f1_fullpool.csv",
        "per_lang_judge_csv":
            "outputs/diagnostic/mistralnemo_per_lang_f1_judge.csv",
        "degeneracy_md": "outputs/tables/degenerate_rows_mistralnemo.md",
    }
    if sorted(expected_default) != sorted(mne.OUT_REL):
        print(f"  FAIL  OUT_REL keys changed: {sorted(mne.OUT_REL)}")
        out.append(False)
    else:
        bad = [k for k, v in expected_default.items() if mne.out_path(k) != v]
        out.append(_expect("all 10 default artifact paths unchanged by --out-dir",
                           bad, []))

    if not os.path.exists(NEMO_CORRECTED):
        print(f"  SKIP  {NEMO_CORRECTED} is absent; the corrected Mistral-Nemo "
              f"weights are needed for the remaining cases")
        return out

    nemo_default_ctx = resolve(default_model=mne.DEFAULT_PACKED_MODEL_PATH,
                               default_scratch=mne.DEFAULT_SCRATCH_DIR_NEMO)
    nemo_corrected_ctx = resolve(NEMO_CORRECTED, NEMO_SCRATCH_CORRECTED,
                                 default_model=mne.DEFAULT_PACKED_MODEL_PATH,
                                 default_scratch=mne.DEFAULT_SCRATCH_DIR_NEMO)
    out += [
        _expect_out_root_refusal("non-default Nemo model with no --out-dir",
                                 nemo_corrected_ctx, None),
        _expect_out_root_refusal("non-default Nemo model into the default root",
                                 nemo_corrected_ctx, DEFAULT_OUT_ROOT),
        _expect_out_root_allowed("released Nemo model keeps the default root",
                                 nemo_default_ctx, None, DEFAULT_OUT_ROOT),
    ]

    def _floor_refusal(label, ctx, value, base):
        try:
            mne._resolve_floor_target(ctx, value, base)
        except (UnsafeModelContext, RuntimeError) as e:
            print(f"  PASS  {label}\n          {str(e).splitlines()[0]}")
            return True
        print(f"  FAIL  {label}: the call was allowed")
        return False

    def _floor_allowed(label, ctx, value, base, expected):
        try:
            got = mne._resolve_floor_target(ctx, value, base)
        except (UnsafeModelContext, RuntimeError) as e:
            print(f"  FAIL  {label}: refused unexpectedly ({e})")
            return False
        return _expect(label, got, expected)

    with tempfile.TemporaryDirectory() as no_fp:
        out += [
            _floor_allowed("released Nemo model keeps the module default clamp",
                           nemo_default_ctx, None, None,
                           mne.DEFAULT_FLOOR_TARGET),
            _floor_refusal("non-default Nemo model with no --floor-target",
                           nemo_corrected_ctx, None, BASE_SCRATCH_CORRECTED),
            _floor_refusal("non-default Nemo model, clamp given but no "
                           "--base-scratch to check it against",
                           nemo_corrected_ctx, -17.0, None),
            _floor_refusal("non-default Nemo model, clamp given but the base "
                           "root has no fingerprint_floor21.json",
                           nemo_corrected_ctx, -17.0, no_fp),
        ]

    fp = os.path.join(BASE_SCRATCH_CORRECTED, mne.BASE_FLOOR21_FP_NAME)
    if not os.path.exists(fp):
        print(f"  SKIP  {fp} is absent; the recorded corrected clamp is needed "
              f"for the cross-check cases")
        return out
    with open(fp) as f:
        recorded = float(json.load(f)["floor_target"])
    out += [
        _floor_allowed(f"non-default Nemo model, clamp matching the recorded "
                       f"sweep result", nemo_corrected_ctx, recorded,
                       BASE_SCRATCH_CORRECTED, recorded),
        _floor_refusal("non-default Nemo model, clamp with the wrong SIGN vs "
                       "the recorded sweep result", nemo_corrected_ctx,
                       -recorded, BASE_SCRATCH_CORRECTED),
        _floor_refusal("non-default Nemo model, released chain's clamp against "
                       "the corrected record", nemo_corrected_ctx,
                       mne.DEFAULT_FLOOR_TARGET, BASE_SCRATCH_CORRECTED),
    ]
    return out


def _expect_allowed(label, **kwargs) -> bool:
    try:
        ctx = resolve(**kwargs)
    except UnsafeModelContext as e:
        print(f"  FAIL  {label}: refused unexpectedly ({e})")
        return False
    kind = "default" if ctx.is_default_model else "non-default"
    print(f"  PASS  {label} ({kind} model)")
    return True


def main(argv=None) -> int:
    scratch = default_scratch_dir()
    backed = store_backed_entries(scratch)
    print(f"default model  : {default_model_path()}")
    print(f"default scratch: {scratch}")
    print(f"  {len(backed)} store-backed entries there, which is the hazard "
          f"this guard exists for\n")

    if not os.path.exists(CORRECTED):
        print(f"SKIP: {CORRECTED} is absent; the corrected weights are needed to "
              f"exercise the non-default branches")
        return 0

    results = []
    with tempfile.TemporaryDirectory() as clean:
        with tempfile.TemporaryDirectory() as linked:
            os.symlink(A_STORE_FILE, os.path.join(linked, "y_true.npy"))
            results = [
                _expect_refusal("non-default model into the default root",
                                model_path=CORRECTED),
                _expect_refusal("non-default model into the default root, named",
                                model_path=CORRECTED, scratch_dir=scratch),
                _expect_refusal("non-default model into a store-backed root",
                                model_path=CORRECTED, scratch_dir=linked),
                _expect_allowed("non-default model into a fresh root",
                                model_path=CORRECTED, scratch_dir=clean),
                _expect_allowed("released model into its own root"),
                _expect_refusal("model file that does not exist",
                                model_path="/nonexistent/model.unilid"),
            ]

            # The output root is the second half of the pair: the memmap scratch
            # root is what resolve() protects, and outputs/ is where the reporting
            # scripts (analysis/paper_eval.py, analysis/paper_breakdowns.py) write
            # the published tables and read their own CSVs back.
            print("\noutput-root rule (analysis.model_context.resolve_out_root):")
            corrected_ctx = resolve(model_path=CORRECTED, scratch_dir=clean)
            default_ctx = resolve()
            fresh_out = os.path.join(clean, "outputs_corrected")
            results += [
                _expect_out_root_refusal(
                    "non-default model with no --out-dir", corrected_ctx, None),
                _expect_out_root_refusal(
                    "non-default model into the default output root",
                    corrected_ctx, DEFAULT_OUT_ROOT),
                _expect_out_root_refusal(
                    "non-default model into a subdirectory of the default root",
                    corrected_ctx, os.path.join(DEFAULT_OUT_ROOT, "corrected")),
                _expect_out_root_refusal(
                    "non-default model into a store-backed output root",
                    corrected_ctx, linked),
                _expect_out_root_allowed(
                    "non-default model into a fresh output root", corrected_ctx,
                    fresh_out, fresh_out),
                _expect_out_root_allowed(
                    "released model keeps the default output root", default_ctx,
                    None, DEFAULT_OUT_ROOT),
            ]

    print(f"\n{sum(results)}/{len(results)} resolver cases behaved as specified")

    results += _check_published_cell_gates()
    results += _check_mistralnemo_chain()

    print("\nevery wired entry point refuses the corrected model on its default "
          "root:")
    results += _check_entry_points()
    print(f"\n{sum(results)}/{len(results)} total cases behaved as specified")
    return 0 if all(results) else 1


# Each entry point, with the argv that must be refused. The point of listing them
# here is that adding a script to the chain without a guard shows up as a failure
# rather than as an absence.
ENTRY_POINTS = [
    ("analysis.full_test_floor21", ["--model", CORRECTED]),
    ("analysis.solo_gates", ["floor21", "--model", CORRECTED]),
    ("analysis.gate_variants", ["topk", "--model", CORRECTED]),
    ("analysis.commonlid_carried", ["--model", CORRECTED]),
    ("analysis.commonlid_calibrated", ["--stage", "score", "--model", CORRECTED]),
    # Both CommonLID entry points have the second half of the pair as well: their
    # repo-side outputs default to the released record. outputs/tables/
    # commonlid_calibrated.md is where tables/commonlid.tex's three rows come
    # from, and outputs/diagnostic/commonlid_carried_preds.npz is the array the
    # calibrated score stage wires itself against. Past the scratch-root rule, the
    # out-root rule is what has to fire; both refusals happen before any model is
    # loaded and nothing is written under either root.
    ("analysis.commonlid_carried",
     ["--model", CORRECTED, "--scratch-dir", BASE_SCRATCH_CORRECTED]),
    ("analysis.commonlid_calibrated",
     ["--stage", "score", "--model", CORRECTED,
      "--scratch-dir", BASE_SCRATCH_CORRECTED]),
    # The eval stage used to reject --model outright; it now resolves it, so both
    # halves of the pair have to be proved for that stage too.
    ("analysis.commonlid_calibrated", ["--stage", "eval", "--model", CORRECTED]),
    ("analysis.commonlid_calibrated",
     ["--stage", "eval", "--model", CORRECTED,
      "--scratch-dir", BASE_SCRATCH_CORRECTED]),
    # The external-benchmark chain has both halves of the pair: the memmap root
    # (fingerprint_floor21.json, the scored npz) and a repo-side root whose
    # default IS the released E2 record -- outputs/tables/external_bench_*.md is
    # where lid_main.tex's UDHR/FLORES cells come from. The first argv must be
    # refused by resolve(), the second (past that rule) by resolve_out_root();
    # both refusals happen inside configure(), so no stage runs and nothing is
    # written under either root.
    ("analysis.external_bench_eval",
     ["--stage", "eval", "--bench", "udhr", "--model", CORRECTED]),
    ("analysis.external_bench_eval",
     ["--stage", "eval", "--bench", "udhr", "--model", CORRECTED,
      "--scratch-dir", BASE_SCRATCH_CORRECTED]),
    ("analysis.mistralnemo_eval", ["--stage", "eval", "--model", CORRECTED]),
    # The Mistral-Nemo chain's own model, past the scratch-root rule, so the
    # out-root rule is what has to fire...
    ("analysis.mistralnemo_eval",
     ["--stage", "tau", "--model", NEMO_CORRECTED,
      "--scratch-dir", NEMO_SCRATCH_CORRECTED,
      "--base-scratch", BASE_SCRATCH_CORRECTED]),
    # ...and past that too, so the missing --floor-target is what has to fire.
    # Both refusals happen inside configure(); no stage runs and NEMO_OUT_PROBE
    # is never created.
    ("analysis.mistralnemo_eval",
     ["--stage", "tau", "--model", NEMO_CORRECTED,
      "--scratch-dir", NEMO_SCRATCH_CORRECTED,
      "--base-scratch", BASE_SCRATCH_CORRECTED,
      "--out-dir", NEMO_OUT_PROBE]),
    ("analysis.release_gates", ["--mode", "base", "--model", CORRECTED]),
    ("analysis.paper_eval", ["--model", CORRECTED]),
    ("analysis.paper_breakdowns", ["--part", "residual", "--model", CORRECTED]),
    # length_bias has both halves of the pair. Its repo-side default root holds
    # outputs/tables/length_bias.{md,tex}, the artifacts behind the published
    # tab:lenbias-delta; the first argv must be refused by resolve() on the
    # scratch root, the second (past that rule) by resolve_out_root(). Both
    # refusals happen before the test file is opened.
    ("analysis.length_bias", ["--subset", "golden", "--model", CORRECTED]),
    ("analysis.length_bias",
     ["--subset", "golden", "--model", CORRECTED,
      "--scratch-dir", BASE_SCRATCH_CORRECTED]),
]


def _check_entry_points():
    import subprocess
    import sys as _sys
    out = []
    for module, argv in ENTRY_POINTS:
        proc = subprocess.run([_sys.executable, "-m", module] + argv,
                              capture_output=True, text=True, timeout=900)
        blob = proc.stdout + proc.stderr
        refused = proc.returncode != 0 and (
            "UnsafeModelContext" in blob or "refusing to gate" in blob)
        out.append(refused)
        print(f"  {'PASS' if refused else 'FAIL'}  {module}")
        if not refused:
            print(f"          exit {proc.returncode}: {blob.strip()[-300:]}")
    return out


if __name__ == "__main__":
    sys.exit(main())
