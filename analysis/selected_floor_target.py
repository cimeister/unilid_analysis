"""Print the unseen-token constant a floor_equalization sweep selected.

Lets a downstream job read the constant from the sweep's own output instead of
having it typed in by hand, so the floor-c pass can be submitted with a SLURM
dependency before the sweep has run. Typing it in by hand would mean either
waiting for the sweep or guessing, and guessing is exactly how a chain ends up
built at a constant the predictions were not scored under.

  python -m analysis.selected_floor_target outputs_corrected_round/tables/floor_equalization.md
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

SELECTED = re.compile(r"Best config selected on val:\s*\*\*floor(-?[\d.]+)\*\*")


def selected(md_path: str) -> float:
    text = Path(md_path).read_text()
    m = SELECTED.search(text)
    if not m:
        raise SystemExit(f"no selected config found in {md_path}; the sweep did "
                         f"not finish, or its output format changed")
    if "**baseline**" in text.split("Best config selected on val:")[1][:40]:
        raise SystemExit(f"{md_path} reports baseline selected, meaning nothing "
                         f"passed the guard; there is no constant to carry")
    return float(m.group(1))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("sweep_md")
    a = ap.parse_args(argv)
    print(f"{selected(a.sweep_md):.10f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
