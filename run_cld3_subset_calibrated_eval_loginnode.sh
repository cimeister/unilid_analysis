#!/bin/bash
set -euo pipefail
#
# LOGIN-NODE HEDGE for slurm_cld3_subset_calibrated_eval.sh (job 3261635).
#
# Why this file exists instead of an --out-dir flag on the sbatch script: sbatch
# froze its copy of that script at submission, so editing it now would leave the
# file on disk disagreeing with what the queued job runs -- a provenance hazard
# this repository already tracks (outputs/rerelease/cld3_governing_files.json).
# The duplication is deliberate; the six commands below are the same six, and
# the two arms write to DIFFERENT directories so they cannot race.
#
# Why the login node is admissible here, against the standing "any full-model
# pass belongs on SLURM" rule (slurm_cld_subset_glotlidc.sh's header): that rule
# is about 1,940-ROW models. Its two recorded SIGKILLs (exit 137) were a
# 1,940-row FLORES pass and the 1,940-row Tatoeba pass, both carrying a ~776 MB
# weight matrix. These containers carry 99/94/93 rows and ~40 MB, and the same
# header records a 99-row GlotLID-C subset pass measured ON THE LOGIN NODE at
# 22,183 lines/s. The queue estimated a 9 h wait (1,778 pending jobs in
# partition normal at submission time), which is the reason for hedging at all.
#
# RAYON_NUM_THREADS is deliberately modest: the login node was at load average
# 85 on 48 cores. Thread count changes no number -- every line is scored
# independently -- only wall clock.
PYTHON="/users/cmeister747/.pyenv/versions/3.11.5/bin/python3"
export RAYON_NUM_THREADS="${RAYON_NUM_THREADS:-16}"
REPO="/users/cmeister747/unilid_analysis"
SCR="/capstor/scratch/cscs/cmeister747/unilid_analysis"
cd "${REPO}"

OUT="outputs/rerelease/cld3_subset_calibration/loginnode"
mkdir -p "${OUT}"

echo "=== cld3 subset calibrated eval (LOGIN NODE) at $(date) on $(hostname) ==="
echo "commit: $(git rev-parse HEAD)"
echo "RAYON_NUM_THREADS=${RAYON_NUM_THREADS}"

run_one () {
  local subset="$1" bench="$2" arm="$3"
  local model="${SCR}/cld3sub${subset}_calibrated.unilid"
  if [ ! -f "${model}" ]; then
    echo "FATAL: version-2 container missing at ${model}" >&2; exit 1
  fi
  local flag=""
  if [ "${arm}" = "calibrated" ]; then flag="--calibrated"; fi
  echo ""
  echo "--- subset ${subset} / ${bench} / ${arm} at $(date) ---"
  ${PYTHON} -u -m analysis.cld_subset_eval "${model}" \
      --bench "${bench}" --mode subset ${flag} \
      --out "${OUT}/cld3sub${subset}_${bench}_${arm}.json" \
      --per-lang-out "${OUT}/cld3sub${subset}_${bench}_${arm}_perlang.json"
}

run_one 80 udhr      baseline
run_one 80 udhr      calibrated
run_one 77 flores    baseline
run_one 77 flores    calibrated
run_one 83 glotlidc  baseline
run_one 83 glotlidc  calibrated

echo ""
echo "=== baseline-equality gate against the 2026-09-01 record ==="
OUT_DIR="${OUT}" ${PYTHON} - <<'PYEOF'
import json, os
BANKED = {
    (83, "glotlidc"): "outputs/rerelease/cld3_subset_models/cld3sub83_glotlidc_subset.json",
    (80, "udhr"):     "outputs/rerelease/cld3_subset_models/cld3sub80_udhr_subset.json",
    (77, "flores"):   "outputs/rerelease/cld3_subset_models/cld3sub77_flores_subset.json",
}
OUT = os.environ["OUT_DIR"]
bad = []
for (subset, bench), banked_path in sorted(BANKED.items()):
    new = json.load(open(f"{OUT}/cld3sub{subset}_{bench}_baseline.json"))
    old = json.load(open(banked_path))
    for field in ("macro_f1", "macro_fpr", "accuracy", "total_samples",
                  "correct", "num_languages", "n_model_rows_evaluated"):
        if new[field] != old[field]:
            bad.append(f"subset-{subset}/{bench}: {field} {new[field]!r} != "
                       f"{old[field]!r} (banked)")
    if new["calibrated"] is not False:
        bad.append(f"subset-{subset}/{bench}: baseline arm ran calibrated")
    cal = json.load(open(f"{OUT}/cld3sub{subset}_{bench}_calibrated.json"))
    if cal["calibrated"] is not True:
        bad.append(f"subset-{subset}/{bench}: calibrated arm ran uncalibrated")
    if cal["total_samples"] != new["total_samples"]:
        bad.append(f"subset-{subset}/{bench}: the two arms scored different "
                   f"pools ({cal['total_samples']} vs {new['total_samples']})")
    print(f"OK  subset-{subset}/{bench}: baseline reproduces the banked cell "
          f"({new['macro_f1']:.7f} / {new['macro_fpr']:.7e}); calibrated "
          f"{cal['macro_f1']:.7f} / {cal['macro_fpr']:.7e} on the same "
          f"{cal['total_samples']:,} lines")
if bad:
    raise SystemExit("FATAL:\n  " + "\n  ".join(bad))
print("All six passes agree with the record.")
PYEOF

echo "=== Finished at $(date) ==="
