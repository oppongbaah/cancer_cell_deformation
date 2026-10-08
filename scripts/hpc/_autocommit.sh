# shellcheck shell=bash
# Commit and push a finished job's outputs. Called by scripts/hpc/_common.sh when a job exits 0;
# never changes the job's exit code, and never writes to the job's .out after committing it.
#
# What it commits (2026-10-08, user decision: run outputs only, never code):
#   * this job's log, logs/hpc/<job-name>_<jobid>.out
#   * results/ and logs/ (other than logs/hpc/), as far as .gitignore admits them -- i.e. eval logs
#     (results/**/*.jsonl) and training logs (logs/**/*.log); figures, CSVs and checkpoints stay out
# Only those paths are committed (git commit --only): anything else you have staged or edited,
# code included, is left exactly as it was.
#
# Where it pushes: the current branch, to github.com/oppongbaah/cancer_cell_deformation over SSH
# with the repo's deploy key (~/.ssh/celldform_deploy). Your own remote and credentials are not
# used or changed. On main/master it does nothing -- auto-commits go to a working branch.
#
# Switches (sbatch --export=ALL,...):
#   AUTOCOMMIT=0           skip it for this job
#   AUTOCOMMIT_PUSH=0      commit locally, don't push
#   AUTOCOMMIT_REMOTE=url  push elsewhere (testing)
# Outcome: one line per job in logs/hpc/autocommit.log (not tracked).

hpc_autocommit() {
    [ "${AUTOCOMMIT:-1}" = "0" ] && return 0
    local job="${SLURM_JOB_NAME:-celldform}_${SLURM_JOB_ID:-$$}"
    local out="logs/hpc/${job}.out" log="logs/hpc/autocommit.log"
    local remote="${AUTOCOMMIT_REMOTE:-git@github.com:oppongbaah/cancer_cell_deformation.git}"
    local key="${AUTOCOMMIT_KEY:-$HOME/.ssh/celldform_deploy}"
    note() { echo "$(date '+%F %T') ${job}: $*" >>"$log"; }

    git rev-parse --is-inside-work-tree >/dev/null 2>&1 || { note "not a git repo -- skipped"; return 0; }
    local branch
    branch=$(git symbolic-ref --short -q HEAD) || { note "detached HEAD -- skipped"; return 0; }
    case "$branch" in main|master) note "on $branch -- skipped (auto-commits go to a working branch)"; return 0 ;; esac

    # One job at a time: jobs that finish together would otherwise collide on git's index lock.
    exec 9>"$(git rev-parse --git-dir)/celldform-autocommit.lock"
    flock -w 600 9 || { note "lock busy for 10 min -- skipped"; return 0; }

    # results/, every logs/ subfolder except hpc/ (other jobs' logs, possibly still being written),
    # and this job's own log. (A ":(exclude)logs/hpc" pathspec would also drop this job's log.)
    local paths=() d
    [ -d results ] && paths+=(results)
    for d in logs/*/; do [ "$d" != "logs/hpc/" ] && paths+=("${d%/}"); done
    [ -f "$out" ] && paths+=("$out")
    [ ${#paths[@]} -eq 0 ] && { note "no outputs to commit"; flock -u 9; return 0; }
    git add -- "${paths[@]}" 2>>"$log"
    if git diff --cached --quiet -- "${paths[@]}"; then
        note "nothing new to commit"; flock -u 9; return 0
    fi
    local summary
    summary=$(grep -hE "pooled +DSC|per-cell mean|=> (PASSED|FAILED)" "$out" 2>/dev/null | tail -3 | sed 's/^ *//')
    git commit --only -q -m "run: ${SLURM_JOB_NAME:-job} finished (job ${SLURM_JOB_ID:-local})" \
        -m "args: ${HPC_ARGS:-(none)}"$'\n'"node: ${SLURMD_NODENAME:-$(hostname)}"${summary:+$'\n\n'"$summary"} \
        -m "Committed automatically by scripts/hpc/_autocommit.sh when the job exited 0." \
        -- "${paths[@]}" 2>>"$log" || { note "commit failed -- see above"; flock -u 9; return 0; }
    local sha; sha=$(git rev-parse --short HEAD)

    if [ "${AUTOCOMMIT_PUSH:-1}" = "0" ]; then
        note "committed $sha on $branch (push disabled)"
    elif GIT_SSH_COMMAND="ssh -i $key -o IdentitiesOnly=yes -o BatchMode=yes" \
            timeout 120 git push -q "$remote" "HEAD:refs/heads/$branch" 2>>"$log"; then
        # keep the local view of origin/<branch> in step with what was just pushed
        git rev-parse -q --verify "refs/remotes/origin/$branch" >/dev/null && \
            git update-ref "refs/remotes/origin/$branch" HEAD
        note "committed $sha on $branch and pushed"
    else
        note "committed $sha on $branch; PUSH FAILED (kept locally -- push it yourself)"
    fi
    flock -u 9
    return 0
}
