"""Regenerate the N_test column of the paper's resource-tier table on the scored pool.

Single source: outputs/diagnostic/paper_eval_per_lang_f1_fullpool.csv (camera-ready E1
artifact; per-language training count N and scored-pool support). The published column was
computed on the full 45,627,279-line test file; the table's F1 cells reproduce on the
45,377,279-line scored pool (E4), so the counts are re-derived on the same pool.

Gates (abort on any mismatch, no substitution):
- support sums to exactly 45,377,279 (the scored pool);
- per-tier language counts equal the published 56/40/458/526/398/462.

Output: outputs/tables/resource_tier_ntest.md
"""
import csv
import sys

SRC = "outputs/diagnostic/paper_eval_per_lang_f1_fullpool.csv"
OUT = "outputs/tables/resource_tier_ntest.md"
POOL = 45_377_279
TIERS = [("<500", 0, 500), ("500-1k", 500, 1_000), ("1k-12k", 1_000, 12_000),
         ("12k-18k", 12_000, 18_000), ("18k-35k", 18_000, 35_000),
         ("35k+", 35_000, None)]
EXPECTED_NLANG = [56, 40, 458, 526, 398, 462]

rows = list(csv.DictReader(open(SRC)))
if len(rows) != 1940:
    sys.exit(f"ABORT: expected 1,940 languages in {SRC}, found {len(rows)}")
total = sum(int(r["support"]) for r in rows)
if total != POOL:
    sys.exit(f"ABORT: support sums to {total}, expected {POOL}")

lines = ["# resource-tier N_test on the scored pool (45,377,279 lines)", "",
         f"Source: {SRC}", "", "| tier | n_lang | N_test |", "|---|---|---|"]
for (name, lo, hi), expect in zip(TIERS, EXPECTED_NLANG):
    sel = [r for r in rows if int(r["N"]) >= lo and (hi is None or int(r["N"]) < hi)]
    if len(sel) != expect:
        sys.exit(f"ABORT: tier {name} has {len(sel)} languages, published table has {expect}")
    lines.append(f"| {name} | {len(sel)} | {sum(int(r['support']) for r in sel)} |")

with open(OUT, "w") as f:
    f.write("\n".join(lines) + "\n")
print("\n".join(lines))
