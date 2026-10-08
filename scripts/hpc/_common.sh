# shellcheck shell=bash
# Shared setup for every SLURM job script under scripts/hpc/. Each job script cds to
# $SLURM_SUBMIT_DIR (the repo root) and sources this file, which:
#
#   * copies stderr to logs/hpc/<job-name>_<jobid>.err and DELETES that file when the job exits
#     0. An .err file in logs/hpc/ therefore marks a job that failed, timed out or was
#     cancelled. Its stderr also stays in the .out, interleaved with stdout, so the .out reads
#     in order;
#   * loads module Anaconda3/2025.06-1 and activates conda env $CONDA_ENV (default celldform);
#   * exports PYTHON (the env's python) for run_py;
#   * prints the job, node, GPU and torch build it got;
#   * when the job exits 0, commits and pushes its outputs (scripts/hpc/_autocommit.sh; opt out
#     with sbatch --export=ALL,AUTOCOMMIT=0).
#
# The job scripts set no `#SBATCH --error`: SLURM creates that file at job start whether or not
# anything goes wrong, so the error file is managed here instead.

[ -f pyproject.toml ] || { echo "submit from the repo root (no pyproject.toml in $PWD)" >&2; exit 1; }
mkdir -p logs/hpc   # too late for SLURM's own --output (it must exist before sbatch), harmless here

HPC_ERR="logs/hpc/${SLURM_JOB_NAME:-celldform}_${SLURM_JOB_ID:-$$}.err"
exec 2> >(tee -a "$HPC_ERR" >&2)

_hpc_finish() {
    local rc=$?
    if [ "$rc" -eq 0 ]; then
        rm -f "$HPC_ERR"
    else
        echo "[celldform] job FAILED (exit $rc) at $(date) -- stderr kept in $HPC_ERR" >>"$HPC_ERR"
    fi
    echo "[celldform] job ${SLURM_JOB_ID:-local} finished (exit $rc) $(date)"
    # Success only: commit + push the run's outputs (scripts/hpc/_autocommit.sh). It runs after the
    # last line above, so the committed .out is complete; its outcome goes to logs/hpc/autocommit.log.
    if [ "$rc" -eq 0 ] && [ -n "${SLURM_JOB_ID:-}" ]; then
        ( set +eu; source scripts/hpc/_autocommit.sh && hpc_autocommit ) || true
    fi
}
trap _hpc_finish EXIT
# Time limit, scancel and preemption send SIGTERM. Exit non-zero so the .err is kept.
trap 'echo "[celldform] terminated by SIGTERM (time limit, scancel or preemption)" >>"$HPC_ERR"; exit 143' TERM

echo "[celldform] job ${SLURM_JOB_ID:-local} (${SLURM_JOB_NAME:-celldform}) on ${SLURMD_NODENAME:-$(hostname)} started $(date)"
echo "[celldform] args: ${*:-(none)}"
HPC_ARGS="${*:-}"   # recorded in the auto-commit message

CONDA_ENV="${CONDA_ENV:-celldform}"
set +u                           # Lmod's init and conda's activate reference unset variables
type module >/dev/null 2>&1 || source /etc/profile.d/z00_lmod.sh   # Lmod, if not inherited
module load Anaconda3/2025.06-1
source "$EBROOTANACONDA3/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
set -u
PYTHON="$(command -v python)"
export PYTHON

if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=index,name,memory.total,driver_version --format=csv   # this job's GPU(s) only
    "$PYTHON" -c "import torch; print('[celldform] torch', torch.__version__, '| cuda', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no GPU')"
fi

# run_py <script.py> [args...]: run a project script unbuffered with the env's python
run_py() { "$PYTHON" -u "$@"; }
