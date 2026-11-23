# shellcheck shell=bash
# Shared setup for every SLURM job script under scripts/hpc/. Each job script cds to
# $SLURM_SUBMIT_DIR (the repo root) and sources this file, which:
#
#   * keeps ONE log per job in logs/hpc/: <job-name>_<jobid>.out when the job exits 0, and
#     <job-name>_<jobid>.err -- the COMPLETE log -- when it fails, times out or is cancelled. On
#     failure the .out (stdout and stderr interleaved, in order) is renamed to .err, so the error
#     is in it whichever stream printed it (nvidia-smi, for one, prints its errors to stdout),
#     and a FAILED line with the exit code ends it. (2026-10-15, user request.)
#   * pushes that .err to GitHub when a job fails, so the failure is visible there (user request,
#     2026-10-15; scripts/hpc/_autocommit.sh hpc_autocommit_failure -- only the .err is committed);
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
# tee ignores SIGTERM: scancel / a time limit signal the whole job, and a dead tee would kill this
# shell on its next stderr write (bash's own "Terminated") before the EXIT trap below could run.
exec 2> >(trap '' TERM; exec tee -a "$HPC_ERR" >&2)

HPC_OUT="logs/hpc/${SLURM_JOB_NAME:-celldform}_${SLURM_JOB_ID:-$$}.out"   # = #SBATCH --output=logs/hpc/%x_%j.out
# Start-time marker: files git ignores (checkpoints/, data/) that are newer than this are the job's
# untracked outputs; the auto-commit lists them as "sync:" lines so the notifier email can give an
# rsync command for exactly those paths (2026-10-27, user request).
HPC_START_MARK="logs/hpc/.${SLURM_JOB_NAME:-celldform}_${SLURM_JOB_ID:-$$}.start"
touch "$HPC_START_MARK"
_hpc_finish() {
    local rc=$?
    echo "[celldform] job ${SLURM_JOB_ID:-local} finished (exit $rc) $(date)"
    if [ "$rc" -eq 0 ]; then
        rm -f "$HPC_ERR"
    else
        # The .out holds everything (stderr is copied into it too): it becomes the .err, the one log
        # of a failed job. If --output was overridden there is no such .out; the stderr copy stays.
        [ -f "$HPC_OUT" ] && mv -f "$HPC_OUT" "$HPC_ERR"
        echo "[celldform] job FAILED (exit $rc) at $(date) -- full job log (stdout + stderr): $HPC_ERR" >>"$HPC_ERR"
        # push the .err so the failure shows on GitHub (only the .err; never partial outputs)
        [ -n "${SLURM_JOB_ID:-}" ] && { ( set +eu; source scripts/hpc/_autocommit.sh && hpc_autocommit_failure "$rc" ) || true; }
    fi
    # Success only: commit + push the run's outputs (scripts/hpc/_autocommit.sh). It runs after the
    # last line above, so the committed .out is complete; its outcome goes to logs/hpc/autocommit.log.
    if [ "$rc" -eq 0 ] && [ -n "${SLURM_JOB_ID:-}" ]; then
        ( set +eu; source scripts/hpc/_autocommit.sh && hpc_autocommit ) || true
    fi
    rm -f "$HPC_START_MARK"
}
trap _hpc_finish EXIT
# Time limit, scancel and preemption send SIGTERM. Exit non-zero so the .err is kept.
trap 'echo "[celldform] terminated by SIGTERM (time limit, scancel or preemption)"; exit 143' TERM   # stdout: ends up in the .err

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
    # nvidia-smi is a diagnostic only: it needs NVML, which breaks when the driver's user-space
    # libraries are updated without a reboot ("Driver/library version mismatch", 2026-10-10) while
    # CUDA itself keeps working. So its failure is logged, not fatal -- the CUDA check below decides.
    nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=index,name,memory.total,driver_version --format=csv 2>&1 \
        || echo "[celldform] nvidia-smi unavailable (NVML) -- not fatal; checking CUDA through torch"
    "$PYTHON" -W ignore -c "import sys, torch
ok = torch.cuda.is_available()
print('[celldform] torch', torch.version.cuda and torch.__version__, '| cuda', ok, '|', torch.cuda.get_device_name(0) if ok else 'no GPU')
sys.exit(0 if ok else '[celldform] CUDA unavailable on an allocated GPU -- stopping before any work')"
fi

# run_py <script.py> [args...]: run a project script unbuffered with the env's python
run_py() { "$PYTHON" -u "$@"; }
