#!/usr/bin/env bash
# =============================================================================
# update.sh — Update the MOSS Image Auto-Captioner in place.
#
# Checks the remote repository for new commits and, if present, fast-forwards
# the local checkout WITHOUT re-cloning the project. Your virtual environment,
# downloaded models (~/.cache/app_models or Unsloth folder) and app config are
# untouched; only Python dependencies are refreshed when requirements.txt
# actually changed between your current version and the incoming one.
#
# Usage:
#   ./update.sh            # check + apply updates (asks before touching files)
#   ./update.sh --check    # only report whether an update is available
#   ./update.sh --help     # this message
# =============================================================================
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

CHECK_ONLY=0
for arg in "$@"; do
  case "$arg" in
    --check) CHECK_ONLY=1 ;;
    -h|--help)
      sed -n '2,15p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    *)
      echo "update.sh: unknown option '$arg' (try --help)" >&2
      exit 2
      ;;
  esac
done

# ---------- helpers ----------------------------------------------------------
info()  { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
ok()    { printf '\033[1;32mOK:\033[0m %s\n' "$*"; }
warn()  { printf '\033[1;33mWARNING:\033[0m %s\n' "$*" >&2; }
fail()  { printf '\033[1;31mERROR:\033[0m %s\n' "$*" >&2; }

die_with_fix() {
  # $1 = short reason, rest = suggested fixes
  fail "$1"
  shift
  echo "  Suggested fixes:" >&2
  for line in "$@"; do echo "    - $line" >&2; done
  exit 1
}

# ---------- preflight --------------------------------------------------------
if [ ! -d "$DIR/.git" ]; then
  die_with_fix "This directory does not look like a git checkout ($DIR/.git missing)." \
    "Run this script from inside the cloned repository." \
    "If you installed by copying files instead of cloning, re-clone once:" \
    "git clone https://github.com/paoloinverse/MOSS-Image-Toriigate-dataset-autocaptioner.git"
fi

if ! command -v git >/dev/null 2>&1; then
  die_with_fix "git is not installed." "sudo apt update && sudo apt install -y git"
fi

BRANCH="$(git rev-parse --abbrev-ref HEAD)"
if [ "$BRANCH" = "HEAD" ]; then
  die_with_fix "The checkout is in 'detached HEAD' state (e.g. checked out a tag)." \
    "Return to a branch first:  git checkout main"
fi

REMOTE_URL="$(git remote get-url origin 2>/dev/null || true)"
if [ -z "$REMOTE_URL" ]; then
  die_with_fix "No 'origin' remote configured." \
    "Add it with:  git remote add origin https://github.com/paoloinverse/MOSS-Image-Toriigate-dataset-autocaptioner.git"
fi

# The venv is optional: if absent we still update the code and just remind.
VENV="$DIR/.venv"
if [ ! -x "$VENV/bin/python" ] && [ ! -x "$VENV/bin/pip" ]; then
  warn "No virtual environment found at $VENV — dependency refresh will be skipped."
  VENV=""
fi

# ---------- fetch ------------------------------------------------------------
info "Checking remote repository for updates (branch: $BRANCH)..."
if ! git fetch --quiet origin "$BRANCH"; then
  die_with_fix "Could not reach the remote ($REMOTE_URL)." \
    "Check your internet connection." \
    "If GitHub needs credentials, test manually:  git pull --dry-run" \
    "You can also retry later; nothing was changed."
fi

LOCAL_REV="$(git rev-parse HEAD)"
REMOTE_REV="$(git rev-parse FETCH_HEAD)"

if [ "$LOCAL_REV" = "$REMOTE_REV" ]; then
  ok "Already up to date (no new commits on origin/$BRANCH)."
  exit 0
fi

BEHIND="$(git rev-list --count HEAD..FETCH_HEAD)"
AHEAD="$(git rev-list --count FETCH_HEAD..HEAD)"
SHORT_LOCAL="$(git rev-parse --short HEAD)"
SHORT_REMOTE="$(git rev-parse --short FETCH_HEAD)"

info "You are $BEHIND commit(s) behind origin/$BRANCH ($SHORT_LOCAL -> $SHORT_REMOTE)."
echo "--- Incoming changes ---"
git --no-pager log --oneline --no-decorate "HEAD..FETCH_HEAD" | head -n 25
if [ "$BEHIND" -gt 25 ]; then echo "  ... and $((BEHIND - 25)) more."; fi
echo "--- Files that will change ---"
git --no-pager diff --stat "HEAD..FETCH_HEAD"
echo

if [ "$CHECK_ONLY" = "1" ]; then
  info "--check mode: nothing was modified. Run ./update.sh to apply."
  exit 0
fi

# ---------- safety checks before touching the working tree -------------------
if [ "$AHEAD" -gt 0 ]; then
  die_with_fix "Your local branch has $AHEAD commit(s) NOT present upstream (diverged history)." \
    "This script only supports clean fast-forward updates to avoid losing work." \
    "Inspect the difference yourself:  git log origin/$BRANCH..HEAD" \
    "Then either push/rebase your commits, or reset deliberately:  git reset --hard origin/$BRANCH (discards local commits!)"
fi

if ! git diff --quiet || ! git diff --cached --quiet; then
  die_with_fix "You have uncommitted modifications in tracked files." \
    "Stash them and update:   git stash && ./update.sh && git stash pop" \
    "Or commit them first, resolve the divergence as described above."
fi
if [ -n "$(git ls-files --others --exclude-standard)" ]; then
  warn "Untracked files exist (they will NOT be touched): $(git ls-files --others --exclude-standard | head -n 5 | tr '\n' ' ')..."
fi

# Did dependencies change between our version and the incoming one?
REQ_CHANGED=0
if ! git diff --quiet "HEAD..FETCH_HEAD" -- requirements.txt; then
  REQ_CHANGED=1
fi

# ---------- apply ------------------------------------------------------------
read -r -p "Apply the update now? [y/N] " REPLY
case "$REPLY" in
  [yY]|[yY][eE][sS]) ;;
  *) info "Update cancelled; nothing was changed."; exit 0 ;;
esac

info "Fast-forwarding to origin/$BRANCH..."
if ! git merge --ff-only --quiet FETCH_HEAD; then
  die_with_fix "Fast-forward failed (the remote history moved during the check)." \
    "Simply run ./update.sh again."
fi
ok "Code updated: $SHORT_LOCAL -> $(git rev-parse --short HEAD)"

# ---------- refresh dependencies only if needed ------------------------------
if [ "$REQ_CHANGED" = "1" ]; then
  if [ -n "$VENV" ]; then
    info "requirements.txt changed — refreshing Python dependencies in $VENV ..."
    "$VENV/bin/pip" install --upgrade pip --quiet
    "$VENV/bin/pip" install -r requirements.txt
    ok "Dependencies updated."
  else
    warn "requirements.txt changed but no virtual environment was found."
    warn "Create one and install deps manually:"
    echo "      python3 -m venv .venv && .venv/bin/pip install -r requirements.txt" >&2
  fi
else
  info "requirements.txt unchanged — skipping dependency reinstall (faster, keeps venv intact)."
fi

echo
ok "Update complete. Start the app with:  ./run.sh"
