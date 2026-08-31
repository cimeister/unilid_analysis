#!/bin/bash
#SBATCH --job-name=unilid-cldsubset
#SBATCH --account=infra01
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=200G
#SBATCH --time=05:00:00
#SBATCH --output=/capstor/scratch/cscs/cmeister747/unilid_analysis/logs/cldsubset_%x_%j.out
#SBATCH --error=/capstor/scratch/cscs/cmeister747/unilid_analysis/logs/cldsubset_%x_%j.err
set -euo pipefail
#
# All five `tab:lid_main` cells that analysis/cld_subset_eval.py can compute for
# ONE model, in one allocation:
#
#   glotlidc subset  (83 languages)  -- the CLD3-subset GlotLID-C cell
#   udhr     full    (366 labels)    -- the full-set UDHR cell
#   udhr     subset  (80 languages)  -- the CLD3-subset UDHR cell
#   flores   full    (190 labels)    -- the full-set FLORES-200 cell
#   flores   subset  (77 languages)  -- the CLD3-subset FLORES-200 cell
#
# The sixth cell (GlotLID-C over all 1,940 labels) is NOT computed here: it is a
# ~12 h pass per model, and the recorded prediction files already reproduce it
# to 7 significant digits (outputs/rerelease/cld_subset_convention_sweep.json).
# YPRED, when set, gates the scorer on GlotLID-C instead, at 1/450 of that cost.
#
# Takes TAG, MODEL and (optionally) YPRED from the environment (--export).
#
# SLURM, not the login node, for two measured reasons:
#   * the glotlidc pass streams the whole 7.1 GB / 45,627,279-line test file
#     (the pool `tab:lid_main`'s caption states the carried rows used -- NOT the
#     45,377,279-line scored pool) and scores the 23.46M lines that survive the
#     subset filter;
#   * the login node SIGKILLed (exit 137) a 1,940-row FLORES pass over 192,280
#     lines on 2026-08-31, the same way it killed the Tatoeba pass on
#     2026-08-23 (SESSION_STATUS.md). Any full-model pass belongs here.
#
# Time: glotlidc subset measured at 22,183 lines/s for a 99-row model on the
# login node (~18 min of scoring; the Python-side parse of 45.6M lines
# dominates, ~1 h total). The y_pred gate is a second full pass over the same
# file with the 1,940-row model on 1/450 of the lines, ~20 min. flores full is
# ~4 min, udhr full ~1 min, both subsets under a minute. 5 h against a measured
# ~1.5-2 h.
#
# Memory: the restricted matrix is 99 x vocab float32 (40-60 MB); the full model
# for the y_pred gate is at most 1,940 x 151,670 float32 = 1.2 GB, doubled by
# the Rust cache push. The test file is streamed, never materialised. 200G is
# the partition's comfortable floor, not a requirement.
PYTHON="/users/cmeister747/.pyenv/versions/3.11.5/bin/python3"
export RAYON_NUM_THREADS=64
cd /users/cmeister747/unilid_analysis

: "${TAG:?FATAL: TAG not set (--export=ALL,TAG=...,MODEL=...)}"
: "${MODEL:?FATAL: MODEL not set (--export=ALL,TAG=...,MODEL=...)}"
if [ ! -f "${MODEL}" ]; then echo "FATAL: model missing at ${MODEL}" >&2; exit 1; fi

OUT="outputs/rerelease/cld_subset"
mkdir -p "${OUT}"

echo "=== ${TAG} at $(date) on $(hostname) ==="
echo "commit: $(git rev-parse HEAD)"
echo "model:  ${MODEL} -> $(readlink -f "${MODEL}")"
echo "ypred:  ${YPRED:-<none>}"

for spec in "glotlidc subset" "udhr full" "udhr subset" "flores full" "flores subset"; do
  set -- ${spec}
  echo "########## ${TAG} $1 $2 at $(date) ##########"
  ${PYTHON} -u -m analysis.cld_subset_eval "${MODEL}" \
      --bench "$1" --mode "$2" \
      --out "${OUT}/${TAG}_$1_$2.json" \
      --per-lang-out "${OUT}/${TAG}_$1_$2_perlang.json" \
      --pred-out "${OUT}/${TAG}_$1_$2_pred.npz"
done

if [ -n "${YPRED:-}" ]; then
  if [ ! -f "${YPRED}" ]; then echo "FATAL: y_pred missing at ${YPRED}" >&2; exit 1; fi
  echo "########## ${TAG} scoring-path gate vs ${YPRED} at $(date) ##########"
  ${PYTHON} -u -m analysis.ypred_scoring_gate "${MODEL}" \
      --ypred "${YPRED}" \
      --out "${OUT}/${TAG}_ypred_scoring_gate.json"
fi

echo "=== Finished at $(date) ==="
