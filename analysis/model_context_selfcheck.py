"""Self-check for analysis/model_context.py: trigger every refusal.

The guard exists to stop a second model's results being written over the
released model's artifacts, which are reachable through symlinks into the
durable store. A guard that is only read and never fired is not evidence, so
this fires each branch and reports what it caught. Run it after any change to
the resolution rules.

  python -m analysis.model_context_selfcheck
"""
from __future__ import annotations

import os
import sys
import tempfile

from analysis.model_context import (UnsafeModelContext, default_model_path,
                                    default_scratch_dir, resolve,
                                    store_backed_entries)

CORRECTED = ("/capstor/scratch/cscs/cmeister747/unilid_analysis/corrected/"
             "glotlidc_corrected.unilid")
A_STORE_FILE = ("/capstor/store/cscs/swissai/a0229/cmeister/unilid_analysis/"
                "glotlidc.unilid")


def _expect_refusal(label, **kwargs) -> bool:
    try:
        resolve(**kwargs)
    except UnsafeModelContext as e:
        print(f"  PASS  {label}\n          {str(e).splitlines()[0]}")
        return True
    print(f"  FAIL  {label}: the call was allowed")
    return False


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

    print(f"\n{sum(results)}/{len(results)} resolver cases behaved as specified")

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
    ("analysis.mistralnemo_eval", ["--stage", "eval", "--model", CORRECTED]),
    ("analysis.release_gates", ["--mode", "base", "--model", CORRECTED]),
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
