"""Align the released and corrected c sweeps position by position.

The corrected sweep uses the published grid shifted by log 5, so grid position k
asks the same question of both models: released F_k and corrected F_k + log 5
clamp the same rows to the same place relative to each row's own seen tokens.
Aligning by position is therefore the like-for-like comparison, and the number of
rows modified at each position is the check that the alignment is real.

Reads the two sweep tables rather than rescoring.

  python -m analysis.c_selection_comparison -o outputs/rerelease/c_selection_comparison.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
RELEASED_MD = REPO / "outputs/tables/floor_equalization.md"
CORRECTED_MD = REPO / "outputs_corrected/tables/floor_equalization.md"
LOG5 = float(np.log(5.0))


def parse(md_path: Path) -> list[dict]:
    rows = []
    for line in md_path.read_text().splitlines():
        if not line.startswith("|") or line.startswith("|:") or "n_modified" in line:
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) < 3:
            continue
        name = cells[0]
        m = re.match(r"^(baseline|floor(-?[\d.]+))$", name)
        if not m:
            continue
        rows.append({"config": name,
                     "F": None if name == "baseline" else float(m.group(2)),
                     "n_modified": int(cells[1]),
                     "val_overall": float(cells[2])})
        if len(rows) == 5:
            break
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--output", default=None)
    a = ap.parse_args(argv)

    rel = parse(RELEASED_MD)
    cor = parse(CORRECTED_MD)
    if len(rel) != len(cor):
        raise SystemExit(f"{len(rel)} released rows against {len(cor)} corrected")

    print(f"{'pos':<4} {'released F':>11} {'corrected F':>12} "
          f"{'n_mod rel':>10} {'n_mod cor':>10} "
          f"{'val rel':>8} {'val cor':>8} {'diff':>8}  shift")
    out = []
    for i, (r, c) in enumerate(zip(rel, cor)):
        shift = None if r["F"] is None else c["F"] - r["F"]
        rec = {"position": i, "released_F": r["F"], "corrected_F": c["F"],
               "shift": shift, "shift_matches_log5":
                   None if shift is None else abs(shift - LOG5) < 1e-3,
               "n_modified_released": r["n_modified"],
               "n_modified_corrected": c["n_modified"],
               "n_modified_equal": r["n_modified"] == c["n_modified"],
               "val_overall_released": r["val_overall"],
               "val_overall_corrected": c["val_overall"]}
        out.append(rec)
        f_r = "baseline" if r["F"] is None else f"{r['F']:.4f}"
        f_c = "baseline" if c["F"] is None else f"{c['F']:.4f}"
        sh = "" if shift is None else f"{shift:+.4f} vs log5 {LOG5:.4f}"
        print(f"{i:<4} {f_r:>11} {f_c:>12} {r['n_modified']:>10,} "
              f"{c['n_modified']:>10,} {r['val_overall']:>8.4f} "
              f"{c['val_overall']:>8.4f} "
              f"{c['val_overall'] - r['val_overall']:>+8.4f}  {sh}")

    clamped = [x for x in out if x["released_F"] is not None]
    best_rel = max(clamped, key=lambda x: x["val_overall_released"])
    best_cor = max(clamped, key=lambda x: x["val_overall_corrected"])
    runner_rel = sorted(clamped, key=lambda x: -x["val_overall_released"])[1]
    runner_cor = sorted(clamped, key=lambda x: -x["val_overall_corrected"])[1]

    print(f"\nreleased  selects position {best_rel['position']} "
          f"(F = {best_rel['released_F']}) over position {runner_rel['position']} "
          f"by {best_rel['val_overall_released'] - runner_rel['val_overall_released']:.4f}")
    print(f"corrected selects position {best_cor['position']} "
          f"(F = {best_cor['corrected_F']:.4f}) over position "
          f"{runner_cor['position']} by "
          f"{best_cor['val_overall_corrected'] - runner_cor['val_overall_corrected']:.4f}")

    all_equal = all(x["n_modified_equal"] for x in clamped)
    all_shift = all(x["shift_matches_log5"] for x in clamped)
    print(f"\nsame rows clamped at every position: {all_equal}")
    print(f"every grid step is exactly log 5:      {all_shift}")

    result = {"log5": LOG5, "positions": out,
              "released_selected_position": best_rel["position"],
              "corrected_selected_position": best_cor["position"],
              "released_margin_over_runner_up":
                  best_rel["val_overall_released"] - runner_rel["val_overall_released"],
              "corrected_margin_over_runner_up":
                  best_cor["val_overall_corrected"] - runner_cor["val_overall_corrected"],
              "n_modified_identical_at_every_position": all_equal,
              "grid_shift_is_log5_at_every_position": all_shift}
    if a.output:
        Path(a.output).write_text(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
