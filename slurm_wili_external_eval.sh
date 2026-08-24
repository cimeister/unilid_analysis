#!/bin/bash
#SBATCH --job-name=unilid-wili-extbench
#SBATCH --account=infra01
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=400G
#SBATCH --time=04:00:00
#SBATCH --output=/capstor/scratch/cscs/cmeister747/unilid_analysis/logs/wili_%x_%j.out
#SBATCH --error=/capstor/scratch/cscs/cmeister747/unilid_analysis/logs/wili_%x_%j.err
set -euo pipefail
#
# One model, one external benchmark (Tatoeba or UDHR), through
# analysis/wili_external_eval.py.
#
# This exists because the login node KILLED the Tatoeba pass: the three-model
# --fp64 run died silently part-way through the second model (log stopped at
# 9,600,000 of 13,101,022 lines, no traceback, no output json). The first model
# had already completed and written its json, so the pass is not reproducible
# on the login node and every remaining Tatoeba model runs here instead. UDHR
# (27,757 rows, ~4 s) stays on the login node.
#
# Takes MODEL and BENCH from the environment (--export). No default for either:
# a wrong-model run that looks successful is the failure mode this repo cares
# about most.
PYTHON="/users/cmeister747/.pyenv/versions/3.11.5/bin/python3"
REPO="/users/cmeister747/unilid_analysis"

if [ -z "${MODEL:-}" ]; then echo "FATAL: MODEL not set" >&2; exit 1; fi
if [ -z "${BENCH:-}" ]; then echo "FATAL: BENCH not set" >&2; exit 1; fi
if [ ! -f "${MODEL}" ]; then echo "FATAL: missing model ${MODEL}" >&2; exit 1; fi

cd "${REPO}"
echo "=== ${BENCH} ${MODEL} at $(date) on $(hostname) ==="
echo "commit: $(git -C "${REPO}" rev-parse HEAD)"

${PYTHON} -u -m analysis.wili_external_eval \
  --bench "${BENCH}" --models "${MODEL}" ${GATE_FLAG:-}

echo "=== Finished at $(date) ==="
