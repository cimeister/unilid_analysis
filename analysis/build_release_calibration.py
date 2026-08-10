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


def main():
    group_a = _rows_from_csv(TAU_GROUP_A_CSV, EXPECTED_GROUP_A_ROWS)
    n_excl = sum(r.excluded for r in group_a.values())
    if n_excl != EXPECTED_GROUP_A_EXCLUDED:
        raise RuntimeError(f"group A has {n_excl} excluded rows, expected "
                           f"{EXPECTED_GROUP_A_EXCLUDED}")
    group_b = _rows_from_csv(TAU_GROUP_B_CSV, EXPECTED_GROUP_B_ROWS)
    if set(group_b) != EXPECTED_GROUP_B_LANGS:
        raise RuntimeError(f"group B languages {sorted(group_b)} != expected "
                           f"{sorted(EXPECTED_GROUP_B_LANGS)}")
    if any(r.excluded for r in group_b.values()):
        raise RuntimeError("group B has excluded rows; expected none")

    with open(TRAIN_COUNTS_FILE) as f:
        train_counts = json.load(f)
    if len(train_counts) != 1_940:
        raise RuntimeError(f"train_counts has {len(train_counts)} entries, "
                           f"expected 1,940")

    with open(FINGERPRINT_JSON) as f:
        fp = json.load(f)
    if fp["floor_target"] != CONSTANTS["unseen_token_constant"]:
        raise RuntimeError(
            f"fingerprint floor_target {fp['floor_target']} != "
            f"unseen_token_constant {CONSTANTS['unseen_token_constant']}")

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
        "tau_csv_sha256": {"group_a": _sha(TAU_GROUP_A_CSV),
                            "group_b": _sha(TAU_GROUP_B_CSV)},
    }

    cal = Calibration(
        group_a=group_a, group_b=group_b,
        train_counts={k: int(v) for k, v in train_counts.items()},
        provenance=provenance, **CONSTANTS)

    # Validate against the actual released model's language list.
    from unilid.model_io import load_unilid
    _base_tok, _w, langs = load_unilid(MODEL_PATH)
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

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "wb") as f:
        f.write(cal.to_json_bytes())
    print(f"Wrote {OUT_JSON} ({os.path.getsize(OUT_JSON):,} bytes): "
          f"{len(group_a)} group A rows ({n_excl} excluded), "
          f"{len(group_b)} group B rows, {len(train_counts)} train counts")


if __name__ == "__main__":
    main()
