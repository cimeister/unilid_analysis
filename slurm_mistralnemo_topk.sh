#!/bin/bash
#SBATCH --job-name=unilid-mistralnemo-topk
#SBATCH --account=infra01
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=100G
#SBATCH --time=08:00:00
#SBATCH --output=/capstor/scratch/cscs/cmeister747/unilid_analysis/logs/mistralnemo_topk_%j.out
#SBATCH --error=/capstor/scratch/cscs/cmeister747/unilid_analysis/logs/mistralnemo_topk_%j.err
set -euo pipefail
PYTHON="/users/cmeister747/.pyenv/versions/3.11.5/bin/python3"
export RAYON_NUM_THREADS=64
cd /users/cmeister747/unilid_analysis

# E3 pre-registration (EXPERIMENTS_PLAN.md, "Camera-ready evaluation program
# (2026-08-06)", E3). Runs STAGE topk: one full-pool pass under the
# variant's floor-21 matrix, writing pred_nemo_floor21.npy for every kept
# line (top-k pass's own rank-1) and banking the full top-5 candidate list
# only for lines whose rank-1 falls in the expanded label set (N < HEAD_N,
# or the variant's flat set). Requires STAGE flatrule and STAGE tau to have
# already completed (their outputs are read for the expanded-set
# definition and embedded, by sha256, in this stage's own fingerprint).
# Chunked and resumable (progress_topk.json); a rerun of this script after
# a timeout or crash picks up from the last completed chunk. Expected
# runtime ~2h (comparable to the full-pool baseline pass, since the
# top-k call scores every row regardless of TOPK_MARGIN); budget padded to
# 8h, matching analysis/full_test_floor21.py's own full-pool-pass precedent
# (slurm_full_test_floor21.sh: 64 CPU, 100G, 8h, measured well under budget).

echo "=== mistralnemo STAGE topk at $(date) on $(hostname) ==="
${PYTHON} -m analysis.mistralnemo_eval --stage topk

echo "=== Finished at $(date) ==="
