#!/bin/bash
#SBATCH --job-name=cld3-subcal-eval
#SBATCH --account=infra01
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=200G
#SBATCH --time=08:00:00
# 8 h for all six passes. Only GlotLID-C costs anything: the uncalibrated pass
# over its 23,462,651 subset-gold lines is recorded at 761 s in
# outputs/rerelease/cld3_subset_models/cld3sub83_glotlidc_subset.json, and the
# calibrated path scores each line twice (one top-k pass plus one segmentation
# pass under the final language, UNILID/unilid/model_io.py
# ::_predict_batch_calibrated), so the pair is budgeted at well under 2 h.
# UDHR (5,509 lines) and FLORES (77,924) finish in seconds. Nothing is written
# incrementally, so a timeout loses a whole pass -- the reason the budget is
# pessimistic and is set HERE rather than overridden at submission, so this file
# and the job cannot disagree about what governed the run.
#SBATCH --output=/capstor/scratch/cscs/cmeister747/unilid_analysis/logs/cld3subcal_%x_%j.out
#SBATCH --error=/capstor/scratch/cscs/cmeister747/unilid_analysis/logs/cld3subcal_%x_%j.err
set -euo pipefail
#
# The calibrated and baseline CLD3-subset cells of the three subset-fitted
# models, under the author's 2026-09-02 ruling:
#
#   "A calibrated row for the subset should still exist. Perform the calibration
#    procedure on the subset-fitted UniLID model. Do not due any hyperparameter
#    sweeps. This is a test to see the generalizability of that approach."
#
# The calibration itself was built by analysis/cld3_subset_calibration.py
# (stages calibval / flatrule / tau / bundle), with every selected constant
# CARRIED from the full corrected model and nothing swept. This job only scores.
#
# BOTH arms run on the SAME version-2 container, so the calibration delta is a
# difference between two flags and not between two files. The container's weight
# matrix is sha256-identical to the version-1 one (asserted at bundle time: the
# unseen-token constant is applied at load time, never baked in), so the
# uncalibrated arm must reproduce the recorded 2026-09-01 cell exactly. That
# equality is checked below and FAILS the job if it does not hold.
PYTHON="/users/cmeister747/.pyenv/versions/3.11.5/bin/python3"
export RAYON_NUM_THREADS="${SLURM_CPUS_PER_TASK:-64}"
REPO="/users/cmeister747/unilid_analysis"
SCR="/capstor/scratch/cscs/cmeister747/unilid_analysis"
cd "${REPO}"

OUT="outputs/rerelease/cld3_subset_calibration"
mkdir -p "${OUT}"

echo "=== cld3 subset calibrated eval at $(date) on $(hostname) ==="
echo "commit: $(git rev-parse HEAD)"

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

# Cheap benchmarks first, so a walltime overrun cannot cost the four cells that
# take seconds; GlotLID-C last.
run_one 80 udhr      baseline
run_one 80 udhr      calibrated
run_one 77 flores    baseline
run_one 77 flores    calibrated
run_one 83 glotlidc  baseline
run_one 83 glotlidc  calibrated

echo ""
echo "=== baseline-equality gate against the 2026-09-01 record ==="
${PYTHON} - <<'PYEOF'
import json
BANKED = {
    (83, "glotlidc"): "outputs/rerelease/cld3_subset_models/cld3sub83_glotlidc_subset.json",
    (80, "udhr"):     "outputs/rerelease/cld3_subset_models/cld3sub80_udhr_subset.json",
    (77, "flores"):   "outputs/rerelease/cld3_subset_models/cld3sub77_flores_subset.json",
}
OUT = "outputs/rerelease/cld3_subset_calibration"
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
