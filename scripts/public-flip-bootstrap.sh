#!/usr/bin/env bash
# Idempotent pre-flip / post-flip automation for the public-flip runbook.
# See docs/public-flip-runbook.md for the operator-facing flow.
#
# Subcommands:
#   pre-flip   — runs on the still-PRIVATE repo. Sets description, topics,
#                branch protection.
#   post-flip  — runs after the repo is PUBLIC. Enables secret scanning,
#                push protection, Dependabot security updates, and private
#                vulnerability reporting.
#
# Idempotency contract: re-running either subcommand must complete with
# exit code 0 and produce only `already configured` lines if there is
# nothing to do. Every state-changing call detects current state first.
#
# Usage:
#   scripts/public-flip-bootstrap.sh pre-flip
#   scripts/public-flip-bootstrap.sh post-flip
#   scripts/public-flip-bootstrap.sh --help
set -euo pipefail

REPO="NguyenJus/custom-sam-peft"
DESCRIPTION="Parameter-efficient finetuning of SAM3.1 for instance segmentation on a single consumer GPU"
TOPICS=(
  sam
  sam3
  segmentation
  instance-segmentation
  peft
  lora
  qlora
  fine-tuning
  pytorch
  huggingface
  computer-vision
  colab
)
REQUIRED_STATUS_CHECKS=(
  "test"
  "lock-check"
  "lint-hygiene"
  "gpu-deselect-check"
  "pip-audit"
  "gitleaks"
)

usage() {
  cat <<EOF
Usage: $0 <pre-flip|post-flip>

  pre-flip   Set description, topics, branch protection on \$REPO.
  post-flip  Enable secret scanning, push protection, Dependabot security
             updates, and private vulnerability reporting on \$REPO.

Both subcommands are idempotent. Requires gh CLI authenticated as a repo
admin.
EOF
}

# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------

log()  { printf '[%s] %s\n' "$(date -u +%H:%M:%SZ)" "$*"; }
warn() { printf '[%s] WARN: %s\n' "$(date -u +%H:%M:%SZ)" "$*" >&2; }

require_gh() {
  command -v gh >/dev/null || { warn "gh CLI not on PATH"; exit 2; }
  gh auth status >/dev/null 2>&1 || { warn "gh not authenticated; run 'gh auth login'"; exit 2; }
}

step_status() {
  local kind="$1" desc="$2"
  case "$kind" in
    applied)    log "applied:           $desc" ;;
    configured) log "already configured: $desc" ;;
    *) warn "unknown status kind: $kind"; exit 3 ;;
  esac
}

# ----------------------------------------------------------------------------
# pre-flip subcommand steps
# ----------------------------------------------------------------------------

set_description() {
  local current
  current="$(gh repo view "$REPO" --json description --jq .description)"
  if [ "$current" = "$DESCRIPTION" ]; then
    step_status configured "repo description matches"
    return
  fi
  gh repo edit "$REPO" --description "$DESCRIPTION" >/dev/null
  step_status applied "set repo description"
}

set_topics() {
  local current_csv
  current_csv="$(gh repo view "$REPO" --json repositoryTopics \
    --jq '[(.repositoryTopics // [])[] | .name] | join(",")')"

  local to_add=()
  for t in "${TOPICS[@]}"; do
    case ",$current_csv," in
      *",$t,"*) : ;;  # already present
      *) to_add+=("$t") ;;
    esac
  done

  if [ ${#to_add[@]} -eq 0 ]; then
    step_status configured "all ${#TOPICS[@]} topics present"
    return
  fi
  gh repo edit "$REPO" --add-topic "$(IFS=,; echo "${to_add[*]}")" >/dev/null
  step_status applied "added topics: ${to_add[*]}"
}

set_branch_protection() {
  local checks_json
  checks_json="$(printf '%s\n' "${REQUIRED_STATUS_CHECKS[@]}" \
    | python3 -c "import json,sys; print(json.dumps([{'context': c.strip(), 'app_id': -1} for c in sys.stdin if c.strip()]))")"

  local desired
  desired="$(cat <<EOF
{
  "required_status_checks": {
    "strict": true,
    "checks": ${checks_json}
  },
  "enforce_admins": false,
  "required_pull_request_reviews": {
    "required_approving_review_count": 0,
    "dismiss_stale_reviews": false,
    "require_code_owner_reviews": false
  },
  "restrictions": null,
  "required_linear_history": true,
  "allow_force_pushes": false,
  "allow_deletions": false,
  "block_creations": false
}
EOF
)"

  local current put_out put_rc=0
  # Branch protection requires a public repo (or GitHub Pro on private repos).
  # If the GET 404s/403s, attempt PUT; if the PUT also 403s, warn and skip —
  # the operator must re-run pre-flip after the repo is public (step 4).
  current="$(gh api "repos/$REPO/branches/main/protection" 2>/dev/null)" || true

  if [ -z "$current" ]; then
    put_out="$(gh api "repos/$REPO/branches/main/protection" \
      -X PUT \
      --input - <<<"$desired" 2>&1)" && put_rc=0 || put_rc=$?
    if [ "$put_rc" -ne 0 ]; then
      if printf '%s' "$put_out" | grep -qE 'HTTP 403|Upgrade to GitHub Pro|must be a public repository'; then
        warn "branch protection skipped: repo is private and account lacks GitHub Pro."
        warn "Re-run pre-flip after flipping the repo to public (step 4)."
        return
      fi
      warn "branch protection PUT failed (rc=$put_rc): $put_out"
      exit "$put_rc"
    fi
    step_status applied "set branch protection on main (was: unset)"
    return
  fi

  local current_proj desired_proj
  current_proj="$(echo "$current" | python3 -c "
import json, sys
d = json.load(sys.stdin)
print(json.dumps({
    'strict': d.get('required_status_checks', {}).get('strict'),
    'checks': sorted(c.get('context','') for c in d.get('required_status_checks', {}).get('checks', [])),
    'linear': d.get('required_linear_history', {}).get('enabled'),
    'force': d.get('allow_force_pushes', {}).get('enabled'),
    'deletions': d.get('allow_deletions', {}).get('enabled'),
}, sort_keys=True))
")"
  desired_proj="$(echo "$desired" | python3 -c "
import json, sys
d = json.load(sys.stdin)
print(json.dumps({
    'strict': d['required_status_checks']['strict'],
    'checks': sorted(c['context'] for c in d['required_status_checks']['checks']),
    'linear': d['required_linear_history'],
    'force': d['allow_force_pushes'],
    'deletions': d['allow_deletions'],
}, sort_keys=True))
")"

  if [ "$current_proj" = "$desired_proj" ]; then
    step_status configured "branch protection on main matches"
    return
  fi
  warn "branch-protection drift detected; desired vs. current:"
  diff <(printf '%s\n' "$desired_proj") <(printf '%s\n' "$current_proj") || true
  local drift_out drift_rc=0
  drift_out="$(gh api "repos/$REPO/branches/main/protection" \
    -X PUT \
    --input - <<<"$desired" 2>&1)" && drift_rc=0 || drift_rc=$?
  if [ "$drift_rc" -ne 0 ]; then
    if printf '%s' "$drift_out" | grep -qE 'HTTP 403|Upgrade to GitHub Pro|must be a public repository'; then
      warn "branch protection skipped: repo is private and account lacks GitHub Pro."
      warn "Re-run pre-flip after flipping the repo to public (step 4)."
      return
    fi
    warn "branch protection PUT failed (rc=$drift_rc): $drift_out"
    exit "$drift_rc"
  fi
  step_status applied "updated branch protection on main"
}

# ----------------------------------------------------------------------------
# post-flip subcommand steps
# ----------------------------------------------------------------------------

api_idempotent_put() {
  local label="$1" path="$2"
  local out rc=0
  if out="$(gh api "$path" -X PUT 2>&1)"; then
    step_status applied "$label"
    return
  fi
  rc=$?
  if printf '%s' "$out" | grep -qE 'HTTP 409|HTTP 422|already enabled|already configured'; then
    step_status configured "$label"
    return
  fi
  warn "$label failed (rc=$rc): $out"
  exit "$rc"
}

set_secret_scanning() {
  local current
  current="$(gh api "repos/$REPO" --jq '.security_and_analysis')"
  local ss psp
  ss="$(echo "$current" | python3 -c "import json,sys; print(json.load(sys.stdin).get('secret_scanning',{}).get('status','disabled'))")"
  psp="$(echo "$current" | python3 -c "import json,sys; print(json.load(sys.stdin).get('secret_scanning_push_protection',{}).get('status','disabled'))")"

  if [ "$ss" = "enabled" ] && [ "$psp" = "enabled" ]; then
    step_status configured "secret scanning + push protection"
    return
  fi
  gh api "repos/$REPO" -X PATCH \
    -F 'security_and_analysis[secret_scanning][status]=enabled' \
    -F 'security_and_analysis[secret_scanning_push_protection][status]=enabled' \
    >/dev/null
  step_status applied "enabled secret scanning + push protection"
}

# ----------------------------------------------------------------------------
# subcommand dispatch
# ----------------------------------------------------------------------------

cmd_pre_flip() {
  require_gh
  log "pre-flip starting on $REPO"
  set_description
  set_topics
  set_branch_protection
  log "pre-flip done."
}

cmd_post_flip() {
  require_gh
  log "post-flip starting on $REPO"
  set_secret_scanning
  api_idempotent_put "Dependabot vulnerability alerts" "repos/$REPO/vulnerability-alerts"
  api_idempotent_put "Dependabot automated security fixes" "repos/$REPO/automated-security-fixes"
  api_idempotent_put "private vulnerability reporting"   "repos/$REPO/private-vulnerability-reporting"
  log "post-flip done."
}

case "${1:-}" in
  pre-flip)  shift; cmd_pre_flip "$@" ;;
  post-flip) shift; cmd_post_flip "$@" ;;
  -h|--help|help|"") usage; exit 0 ;;
  *) warn "unknown subcommand: $1"; usage; exit 2 ;;
esac
