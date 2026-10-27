# shellcheck shell=bash
# Commit and push a finished job's log and outputs. Called by scripts/hpc/_common.sh when a job
# ends; never changes the job's exit code, and never writes to the job's log after committing it.
#
#   hpc_autocommit           the job exited 0 (2026-10-08, user decision: run outputs only, never code):
#                              * this job's log, logs/hpc/<job-name>_<jobid>.out
#                              * results/ and logs/ (other than logs/hpc/), as far as .gitignore admits
#                                them -- eval logs (results/**/*.jsonl), training logs (logs/**/*.log);
#                                figures, CSVs and checkpoints stay out
#   hpc_autocommit_failure   the job failed, timed out or was cancelled (2026-10-15, user request:
#                            "push .err files when error occurs so that I can know"):
#                              * ONLY this job's log, logs/hpc/<job-name>_<jobid>.err -- the complete log
#                                of the failed job -- with the exit code and the error lines in the
#                                commit message. A failed run's partial outputs are never committed.
# Only those paths are committed (git commit --only): anything else you have staged or edited,
# code included, is left exactly as it was.
#
# Where it pushes: the current branch, to github.com/oppongbaah/cancer_cell_deformation over SSH
# with the repo's deploy key (~/.ssh/celldform_deploy). Your own remote and credentials are not
# used or changed. On main/master it does nothing -- auto-commits go to a working branch.
#
# Switches (sbatch --export=ALL,...):
#   AUTOCOMMIT=0           skip it for this job (success and failure alike)
#   AUTOCOMMIT_PUSH=0      commit locally, don't push
#   AUTOCOMMIT_REMOTE=url  push elsewhere (testing)
# Outcome: one line per job in logs/hpc/autocommit.log (not tracked).

_ac_job()  { echo "${SLURM_JOB_NAME:-celldform}_${SLURM_JOB_ID:-$$}"; }
_ac_note() { echo "$(date '+%F %T') $(_ac_job): $*" >>logs/hpc/autocommit.log; }

# _ac_begin: checks + lock, in THIS shell (a $(...) subshell would drop the lock on return).
# Sets AC_BRANCH; returns non-zero when the job must not commit.
_ac_begin() {
    [ "${AUTOCOMMIT:-1}" = "0" ] && return 1
    git rev-parse --is-inside-work-tree >/dev/null 2>&1 || { _ac_note "not a git repo -- skipped"; return 1; }
    AC_BRANCH=$(git symbolic-ref --short -q HEAD) || { _ac_note "detached HEAD -- skipped"; return 1; }
    case "$AC_BRANCH" in main|master) _ac_note "on $AC_BRANCH -- skipped (auto-commits go to a working branch)"; return 1 ;; esac
    # One job at a time: jobs that finish together would otherwise collide on git's index lock.
    exec 9>"$(git rev-parse --git-dir)/celldform-autocommit.lock"
    flock -w 600 9 || { _ac_note "lock busy for 10 min -- skipped"; return 1; }
}

# _ac_commit_push <branch> <subject> <body> <paths...>: commit exactly <paths>, push, log, unlock.
_ac_commit_push() {
    local branch=$1 subject=$2 body=$3; shift 3
    local remote="${AUTOCOMMIT_REMOTE:-git@github.com:oppongbaah/cancer_cell_deformation.git}"
    local key="${AUTOCOMMIT_KEY:-$HOME/.ssh/celldform_deploy}"
    git add -- "$@" 2>>logs/hpc/autocommit.log
    if git diff --cached --quiet -- "$@"; then
        _ac_note "nothing new to commit"; flock -u 9; return 0
    fi
    git commit --only -q -m "$subject" -m "$body" \
        -m "Committed automatically by scripts/hpc/_autocommit.sh when the job ended." \
        -- "$@" 2>>logs/hpc/autocommit.log || { _ac_note "commit failed -- see above"; flock -u 9; return 0; }
    local sha; sha=$(git rev-parse --short HEAD)
    if [ "${AUTOCOMMIT_PUSH:-1}" = "0" ]; then
        _ac_note "committed $sha on $branch (push disabled)"
    elif GIT_SSH_COMMAND="ssh -i $key -o IdentitiesOnly=yes -o BatchMode=yes" \
            timeout 120 git push -q "$remote" "HEAD:refs/heads/$branch" 2>>logs/hpc/autocommit.log; then
        # keep the local view of origin/<branch> in step with what was just pushed
        git rev-parse -q --verify "refs/remotes/origin/$branch" >/dev/null && \
            git update-ref "refs/remotes/origin/$branch" HEAD
        _ac_note "committed $sha on $branch and pushed"
    else
        _ac_note "committed $sha on $branch; PUSH FAILED (kept locally -- push it yourself)"
    fi
    flock -u 9
    return 0
}

# _ac_sync_paths <start marker>: the git-ignored files under checkpoints/, data/ and results/ written
# since the marker (the job's untracked outputs), as at most 10 directories -- deeper ones are merged
# into their parent until 10 or fewer remain. One "sync: <path>" line each, then the total size.
# Jobs running at the same time can show up in each other's lists; that only means extra copying.
_ac_sync_paths() {
    local mark=$1 dirs d
    [ -f "$mark" ] || return 0
    dirs=$(find checkpoints data results -type f -newer "$mark" 2>/dev/null \
           | git check-ignore --stdin 2>/dev/null | xargs -r -d '\n' dirname | sort -u)
    [ -z "$dirs" ] && return 0
    while [ "$(wc -l <<<"$dirs")" -gt 10 ]; do
        dirs=$(sed 's#/[^/]*$##' <<<"$dirs" | sort -u)   # parent dir; never above checkpoints/ etc.
        # drop any directory already inside another listed one
        dirs=$(awk '{for (k in kept) if (index($0, k"/")==1) next; kept[$0]; print}' <<<"$dirs")
    done
    while IFS= read -r d; do echo "sync: $d"; done <<<"$dirs"
    echo "sync-size: $(xargs -r -d '\n' du -sch <<<"$dirs" | tail -1 | cut -f1)"
}

hpc_autocommit() {
    _ac_begin || return 0
    local branch=$AC_BRANCH
    local out="logs/hpc/$(_ac_job).out"
    # results/, every logs/ subfolder except hpc/ (other jobs' logs, possibly still being written),
    # and this job's own log. (A ":(exclude)logs/hpc" pathspec would also drop this job's log.)
    local paths=() d
    [ -d results ] && paths+=(results)
    for d in logs/*/; do [ "$d" != "logs/hpc/" ] && paths+=("${d%/}"); done
    [ -f "$out" ] && paths+=("$out")
    [ ${#paths[@]} -eq 0 ] && { _ac_note "no outputs to commit"; flock -u 9; return 0; }
    local summary
    summary=$(grep -hE "pooled +DSC|per-cell mean|=> (PASSED|FAILED)" "$out" 2>/dev/null | tail -3 | sed 's/^ *//')
    local sync; sync=$(_ac_sync_paths "${HPC_START_MARK:-}")
    _ac_commit_push "$branch" "run: ${SLURM_JOB_NAME:-job} finished (job ${SLURM_JOB_ID:-local})" \
        "args: ${HPC_ARGS:-(none)}"$'\n'"node: ${SLURMD_NODENAME:-$(hostname)}"${summary:+$'\n\n'"$summary"}${sync:+$'\n\n'"$sync"} \
        "${paths[@]}"
}

hpc_autocommit_failure() {   # <exit code>
    local rc=${1:-?}
    _ac_begin || return 0
    local branch=$AC_BRANCH
    local err="logs/hpc/$(_ac_job).err"
    [ -f "$err" ] || { _ac_note "failed (exit $rc) but no $err to commit"; flock -u 9; return 0; }
    # the error itself, for the commit message: the last lines that look like one, else the log's tail
    local errors
    errors=$(grep -E "Error|error|Traceback|FAILED|Failed|failed|Killed|CANCELLED|terminated|No such|denied|mismatch" "$err" \
             | grep -v "^\[celldform\] job FAILED" | tail -6)
    [ -z "$errors" ] && errors=$(grep -v "^\[celldform\]" "$err" | tail -6)
    _ac_commit_push "$branch" "run: ${SLURM_JOB_NAME:-job} FAILED (exit $rc, job ${SLURM_JOB_ID:-local})" \
        "args: ${HPC_ARGS:-(none)}"$'\n'"node: ${SLURMD_NODENAME:-$(hostname)}"$'\n'"log: $err"$'\n\n'"$errors" \
        "$err"
}
