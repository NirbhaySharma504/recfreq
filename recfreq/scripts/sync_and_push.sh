#!/bin/bash
# Copy finished results from the training server into this repo, then commit and push.
# Usage:  RECFREQ_SERVER=user@host bash recfreq/scripts/sync_and_push.sh "commit message"
# Copies: summaries of every finished run (no ckpt.pt), queue logs, figures, audit/scope outputs.
set -e
SERVER=${RECFREQ_SERVER:?set RECFREQ_SERVER=user@host of the training server}
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO/recfreq"
TMP=$(mktemp)
ssh -o ConnectTimeout=20 "$SERVER" 'cd ~/recfreq/results/runs && for d in */; do [ -f "$d/done.json" ] && echo "/${d%/}/***"; done' > "$TMP"
echo "$(wc -l < "$TMP") finished runs on the server"
rsync -az --exclude="ckpt.pt" --include-from="$TMP" --exclude="*" "$SERVER":~/recfreq/results/runs/ results/runs/
rsync -az --include="*.log" --include="*.txt" --exclude="*" "$SERVER":~/recfreq/results/ results/
for d in figs_scope verify_out; do rsync -az "$SERVER":~/recfreq/$d ./ ; done
rm -f "$TMP"
cd "$REPO"
git add -A
if git diff --cached --quiet; then echo "nothing new to commit"; exit 0; fi
git commit -q -m "${1:-Add run-6 (law scope) results}"
git push origin main
git log --oneline -1
