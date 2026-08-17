"""Build the release calibration JSON for the base model from the artifacts of
record, using the package's own Calibration class (so every validation the
package enforces runs at build time).

Sources (all fixed, nothing recomputed):
- outputs/diagnostic/tau_floor21_gate.csv  (group A thresholds, 1,080 rows,
  26 excluded with cause low_calibration; produced by analysis/solo_gates.py)
- outputs/diagnostic/tau_flat4.csv         (group B thresholds, 4 rows)
- config.TRAIN_COUNTS_FILE                 (N_L for all 1,940 languages)
- full_test_eval store dir fingerprint_floor21.json (matrix sha256 provenance)
- the promoted configuration's constants (EXPERIMENTAL_SETUP.md
  gate_flat4_prox21; analysis/margin_diagnostic.py, gate_variants.py,
  hierarchical_pool.py)

Output: outputs/release/calibration_glotlidc.json

Run from the repo root with the UNILID checkout importable:
  PYTHONPATH=UNILID python -m analysis.build_release_calibration
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "UNILID"))

from unilid.calibration import Calibration, TauRow  # noqa: E402

from analysis.config import TRAIN_COUNTS_FILE  # noqa: E402

TAU_GROUP_A_CSV = "outputs/diagnostic/tau_floor21_gate.csv"
TAU_GROUP_B_CSV = "outputs/diagnostic/tau_flat4.csv"
FINGERPRINT_JSON = ("/capstor/store/cscs/swissai/a0229/cmeister/unilid_analysis/"
                    "full_test_eval/fingerprint_floor21.json")
MODEL_PATH = ("/capstor/store/cscs/swissai/a0229/cmeister/unilid_analysis/"
              "glotlidc.unilid")
OUT_JSON = "outputs/release/calibration_glotlidc.json"

# The promoted configuration gate_flat4_prox21 (EXPERIMENTAL_SETUP.md; constants
# from analysis/margin_diagnostic.py, gate_variants.py, hierarchical_pool.py,
# full_test_floor21.py).
CONSTANTS = {
    "unseen_token_constant": -21.0,
    "head_n": 18_000,
    "replacement_min_n": 100_000,
    "proximity_bound": 21.0,
    "topk": 5,
    "margin_q": 5.0,
    "group_b_percentile": 5.0,
    "calib_max": 2_000,
    "min_calib_lines": 200,
    "calib_seed": 0,
}

# Counts and membership recorded from the released model's calibration. These are
# assertions, not defaults: the script must refuse to bundle a calibration whose
# shape it does not recognize.
#
# For a re-release they have to be re-derived, not relaxed. Group A membership is
# fixed by N_L < head_n and so cannot change, but its excluded count can, because
# exclusion depends on how many calibration lines a language is top-scoring on and
# the special-token correction changes that. Group B is identified from
# predictions, 0.72% of which change, so its four members may not be the same four.
# Overriding them is a deliberate act with --expect-group-b / --expect-group-a-*,
# and whatever is passed must be recorded in EXPERIMENTAL_SETUP.md.
EXPECTED_GROUP_A_ROWS = 1_080
EXPECTED_GROUP_A_EXCLUDED = 26
EXPECTED_GROUP_B_ROWS = 4
EXPECTED_GROUP_B_LANGS = {"sco_Latn", "bjn_Latn", "arg_Latn", "vls_Latn"}


def _rows_from_csv(path: str, expected_rows: int) -> dict:
    df = pd.read_csv(path)
    if list(df.columns) != ["lang", "n_scoreable", "n_self_won", "tau",
                            "excluded", "cause"]:
        raise RuntimeError(f"{path}: unexpected columns {list(df.columns)}")
    if len(df) != expected_rows:
        raise RuntimeError(f"{path}: {len(df)} rows, expected {expected_rows}")
    if df.lang.duplicated().any():
        raise RuntimeError(f"{path}: duplicate language rows")
    if df.excluded.dtype != bool:
        raise RuntimeError(f"{path}: excluded column dtype {df.excluded.dtype}, "
                           f"expected bool")
    rows = {}
    for rec in df.itertuples(index=False):
        tau = float(rec.tau)
        excluded = bool(rec.excluded)
        cause = "" if (isinstance(rec.cause, float) and math.isnan(rec.cause)) \
            else str(rec.cause)
        if excluded != (tau == float("-inf")):
            raise RuntimeError(
                f"{path}: row {rec.lang} has excluded={excluded} with "
                f"tau={tau!r}")
        rows[rec.lang] = TauRow(tau=tau, excluded=excluded, cause=cause,
                                n_scoreable=int(rec.n_scoreable),
                                n_self_won=int(rec.n_self_won))
    return rows


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", dest="model_path", default=MODEL_PATH,
                    help="model whose language list the calibration is validated "
                         "against (default: the released model)")
    ap.add_argument("--fingerprint", default=FINGERPRINT_JSON,
                    help="fingerprint_floor21.json supplying the matrix shas")
    ap.add_argument("--tau-group-a", default=TAU_GROUP_A_CSV)
    ap.add_argument("--tau-group-b", default=TAU_GROUP_B_CSV)
    ap.add_argument("--out", dest="out_json", default=OUT_JSON)
    ap.add_argument("--unseen-token-constant", type=float, default=None,
                    help="override c; must match the fingerprint's floor_target")
    ap.add_argument("--expect-group-a-rows", type=int, default=EXPECTED_GROUP_A_ROWS)
    ap.add_argument("--expect-group-a-excluded", type=int,
                    default=EXPECTED_GROUP_A_EXCLUDED)
    ap.add_argument("--expect-group-b", default=None,
                    help="comma-separated group B languages, when the "
                         "re-identification returns a different set")
    a = ap.parse_args(argv)

    constants = dict(CONSTANTS)
    if a.unseen_token_constant is not None:
        constants["unseen_token_constant"] = a.unseen_token_constant
    expected_group_b = (set(x.strip() for x in a.expect_group_b.split(","))
                        if a.expect_group_b else EXPECTED_GROUP_B_LANGS)

    group_a = _rows_from_csv(a.tau_group_a, a.expect_group_a_rows)
    n_excl = sum(r.excluded for r in group_a.values())
    if n_excl != a.expect_group_a_excluded:
        raise RuntimeError(f"group A has {n_excl} excluded rows, expected "
                           f"{a.expect_group_a_excluded}")
    group_b = _rows_from_csv(a.tau_group_b, len(expected_group_b))
    if set(group_b) != expected_group_b:
        raise RuntimeError(f"group B languages {sorted(group_b)} != expected "
                           f"{sorted(expected_group_b)}")
    if any(r.excluded for r in group_b.values()):
        raise RuntimeError("group B has excluded rows; expected none")

    with open(TRAIN_COUNTS_FILE) as f:
        train_counts = json.load(f)
    if len(train_counts) != 1_940:
        raise RuntimeError(f"train_counts has {len(train_counts)} entries, "
                           f"expected 1,940")

    with open(a.fingerprint) as f:
        fp = json.load(f)
    if fp["floor_target"] != constants["unseen_token_constant"]:
        raise RuntimeError(
            f"fingerprint floor_target {fp['floor_target']} != "
            f"unseen_token_constant {constants['unseen_token_constant']}; the "
            f"clamped matrix was not built at the constant being bundled")

    def _sha(path):
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()

    provenance = {
        "derived": ("gate_flat4_prox21 (promoted 2026-08-06); thresholds from "
                    "analysis/solo_gates.py and the flat-four calibration; see "
                    "the paper's development-protocol appendix"),
        "base_weight_matrix_sha256": fp["sha256_base_W"],
        "clamped_weight_matrix_sha256": fp["sha256_w21"],
        "langs_sha256": fp["langs_sha256"],
        "tau_csv_sha256": {"group_a": _sha(a.tau_group_a),
                            "group_b": _sha(a.tau_group_b)},
    }
    # Record the inputs only when they are not the recorded defaults. With the
    # defaults, this script must still reproduce the shipped calibration JSON
    # byte-for-byte, which is how the published artifact is verified; adding
    # fields unconditionally would silently break that check. When a different
    # model or fingerprint is used the paths are load-bearing and go in.
    if os.path.abspath(a.model_path) != os.path.abspath(MODEL_PATH):
        provenance["model_path"] = a.model_path
    if os.path.abspath(a.fingerprint) != os.path.abspath(FINGERPRINT_JSON):
        provenance["fingerprint_json"] = a.fingerprint

    cal = Calibration(
        group_a=group_a, group_b=group_b,
        train_counts={k: int(v) for k, v in train_counts.items()},
        provenance=provenance, **constants)

    # Validate against the actual released model's language list.
    from unilid.model_io import load_unilid
    _base_tok, _w, langs = load_unilid(a.model_path)
    cal.runtime_for(langs)
    del _w

    # Round trip and value-level re-verification against the CSVs.
    reparsed = Calibration.from_json_bytes(cal.to_json_bytes())
    for name, orig in (("group_a", group_a), ("group_b", group_b)):
        rt = getattr(reparsed, name)
        if set(rt) != set(orig):
            raise RuntimeError(f"round trip changed {name} language set")
        for lang, row in orig.items():
            if rt[lang] != row:
                raise RuntimeError(
                    f"round trip changed {name}[{lang}]: {rt[lang]} != {row}")
    if reparsed.train_counts != cal.train_counts:
        raise RuntimeError("round trip changed train_counts")

    os.makedirs(os.path.dirname(a.out_json), exist_ok=True)
    with open(a.out_json, "wb") as f:
        f.write(cal.to_json_bytes())
    print(f"Wrote {a.out_json} ({os.path.getsize(a.out_json):,} bytes): "
          f"{len(group_a)} group A rows ({n_excl} excluded), "
          f"{len(group_b)} group B rows, {len(train_counts)} train counts")


if __name__ == "__main__":
    main()
