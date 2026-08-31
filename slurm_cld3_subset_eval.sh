#!/bin/bash
#SBATCH --job-name=cld3-subset-eval
#SBATCH --account=infra01
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=200G
#SBATCH --time=05:00:00
# 5 h for all three, matching what slurm_cld_subset_glotlidc.sh gives the
# comparable gate jobs. It is set here and NOT overridden at submission, so the
# jobs and this file cannot disagree about what governed them. Only the
# GlotLID-C pass needs it (23.46M scored lines of a 45,627,279-line file, ~2.25 h
# at the 22,183 lines/s that script records for a 99-row model); UDHR (5,509
# lines) and FLORES (77,924) finish in seconds. Nothing is written
# incrementally, so a timeout loses the whole pass -- the reason the earlier 3 h
# and 1 h overrides were withdrawn.
#SBATCH --output=/capstor/scratch/cscs/cmeister747/unilid_analysis/logs/cld3sub_%x_%j.out
#SBATCH --error=/capstor/scratch/cscs/cmeister747/unilid_analysis/logs/cld3sub_%x_%j.err
set -euo pipefail
#
# Evaluate ONE subset-vocabulary model (slurm_cld3_subset_train.sh) on ITS OWN
# benchmark, under the convention analysis/cld_subset_eval.py implements:
# --lang-only bare-ISO collapse, the paper team's only_model_langs line filter,
# their macro F1 / macro FPR core, Viterbi decode, uncalibrated.
#
#   SUBSET=83 -> glotlidc  (the CLD3-subset GlotLID-C cell, 83 languages)
#   SUBSET=80 -> udhr      (the CLD3-subset UDHR cell, 80 languages)
#   SUBSET=77 -> flores    (the CLD3-subset FLORES-200 cell, 77 languages)
#
# --mode subset restricts the model to the labels whose bare ISO is in the
# subset file. For a subset-trained model that restriction is a no-op -- its
# label set IS the subset -- so what --mode subset contributes here is the line
# filter and the bare-ISO metric, which is exactly what running the paper team's
# own eval_*.py --lang-only against such a model would do. The no-op is asserted
# below rather than assumed: n_model_rows_evaluated must equal
# n_model_rows_total, otherwise the container is not the subset model it claims
# to be.
#
# SLURM, not the login node: the glotlidc pass streams the whole 7.1 GB /
# 45,627,279-line test file and scores the 23.46M lines that survive the subset
# filter, and the login node has SIGKILLed full-file passes twice
# (slurm_cld_subset_glotlidc.sh's header). Run all three here so one convention
# and one machine produce all three cells.
PYTHON="/users/cmeister747/.pyenv/versions/3.11.5/bin/python3"
export RAYON_NUM_THREADS="${SLURM_CPUS_PER_TASK:-64}"
REPO="/users/cmeister747/unilid_analysis"
SCR="/capstor/scratch/cscs/cmeister747/unilid_analysis"
cd "${REPO}"

: "${SUBSET:?FATAL: SUBSET must be set to 83, 80 or 77 via --export}"
case "${SUBSET}" in
  83) BENCH="glotlidc" ;;
  80) BENCH="udhr" ;;
  77) BENCH="flores" ;;
  *) echo "FATAL: SUBSET=${SUBSET} is not one of 83, 80, 77" >&2; exit 1 ;;
esac

MODEL="${SCR}/cld3sub${SUBSET}.unilid"
if [ ! -f "${MODEL}" ]; then
  echo "FATAL: subset model missing at ${MODEL}" >&2; exit 1
fi

OUT="outputs/rerelease/cld3_subset_models"
mkdir -p "${OUT}"

echo "=== cld3sub${SUBSET} on ${BENCH} at $(date) on $(hostname) ==="
echo "commit: $(git rev-parse HEAD)"
echo "model:  ${MODEL} -> $(readlink -f "${MODEL}")"

${PYTHON} -u -m analysis.cld_subset_eval "${MODEL}" \
    --bench "${BENCH}" --mode subset \
    --out "${OUT}/cld3sub${SUBSET}_${BENCH}_subset.json" \
    --per-lang-out "${OUT}/cld3sub${SUBSET}_${BENCH}_subset_perlang.json" \
    --pred-out "${OUT}/cld3sub${SUBSET}_${BENCH}_subset_pred.npz"

${PYTHON} - "${OUT}/cld3sub${SUBSET}_${BENCH}_subset.json" <<'PYEOF'
import json, sys
s = json.load(open(sys.argv[1]))
if s["n_model_rows_evaluated"] != s["n_model_rows_total"]:
    raise SystemExit(
        f"FATAL: --mode subset dropped rows from a subset-trained model: "
        f"{s['n_model_rows_evaluated']} of {s['n_model_rows_total']} were "
        f"evaluated. This container is not the subset model it claims to be.")
print(f"OK  all {s['n_model_rows_total']} rows are in the subset; "
      f"macro F1 {s['macro_f1']:.7f}  macro FPR {s['macro_fpr']:.7e}")
PYEOF

echo "=== Finished at $(date) ==="
