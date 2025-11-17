#!/usr/bin/env bash
set -euo pipefail

API="http://localhost:8000"
OUT_DIR="./sample_runs"
DRYPRS_DIR="./packages/tools/_dry_prs"

mkdir -p "$OUT_DIR"
mkdir -p "$DRYPRS_DIR"

declare -a ISSUES=(
  "Repo: https://github.com/langflow-ai/langflow |
  Issue: 'Support for Multi-Agent Orchestration Where a Main Agent Reads Shared Storage and Integrates Background Agent Outputs Into the Dialogue'|
  README : https://github.com/langflow-ai/langflow/blob/main/README.md "
  
  "Repo: https://github.com/codecrafters-io/build-your-own-x | 
  Issue: 'Broken Link for Python:Build a Reddit bot' |
  README : https://github.com/codecrafters-io/build-your-own-x/blob/master/README.md "

  "Repo: https://github.com/n8n-io/n8n | 
  Issue : 'rootStore.restUrl always returns /rest → WebSockets fail with invalid URL' |
  README: https://github.com/n8n-io/n8n/blob/master/README.md "
)

timestamp() { date +%Y%m%dT%H%M%S; }

for i in "${!ISSUES[@]}"; do
  RUN_ID="run-$((i+1))-$(timestamp)"
  RUN_DIR="${OUT_DIR}/${RUN_ID}"
  mkdir -p "$RUN_DIR"

  ISSUE_ENTRY="${ISSUES[$i]}"


  REPO_URL=$(echo "$ISSUE_ENTRY" | sed -n "s/.*Repo:\s*\([^|]*\).*/\1/p" | xargs)

  ISSUE_TEXT=$(echo "$ISSUE_ENTRY" | sed -n "s/.*Issue:\s*'\([^']*\)'.*/\1/p" | head -1)

  README_PATH=$(echo "$ISSUE_ENTRY" | sed -n "s/.*README\s*:\s*\([^|]*\).*/\1/p" | xargs)
  
  # Fallback if parsing fails
  if [ -z "$REPO_URL" ]; then
    REPO_URL="https://github.com/example/repo"
  fi
  if [ -z "$ISSUE_TEXT" ]; then
    ISSUE_TEXT="$ISSUE_ENTRY"
  fi

  echo "=== Starting sample run: $RUN_ID ==="
  echo "Repo URL: $REPO_URL"
  echo "Issue text: $ISSUE_TEXT"
  echo "README path: ${README_PATH:-(none)}"

  # POST to /start with repo_url and optional readme_file_path
  if [ -n "$README_PATH" ]; then
    RESPONSE=$(curl -sS -X POST "${API}/start" \
      -H "Content-Type: application/json" \
      -d "$(jq -n --arg repo_url "$REPO_URL" --arg issue "$ISSUE_TEXT" --arg readme "$README_PATH" '{repo_url:$repo_url, issue_text:$issue, readme_file_path:$readme}')" )
  else
    RESPONSE=$(curl -sS -X POST "${API}/start" \
      -H "Content-Type: application/json" \
      -d "$(jq -n --arg repo_url "$REPO_URL" --arg issue "$ISSUE_TEXT" '{repo_url:$repo_url, issue_text:$issue}')" )
  fi

  echo "$RESPONSE" | jq '.' > "${RUN_DIR}/response.json"
  echo "Saved response to ${RUN_DIR}/response.json"

  # Copy any created dry-run PR artifacts (if present)
  if [ -d "$DRYPRS_DIR" ]; then
    cp -r "${DRYPRS_DIR}" "${RUN_DIR}/_dry_prs" || true
  fi

  # Snapshot metrics
  if [ -f "metrics/metrics.json" ]; then
    cp metrics/metrics.json "${RUN_DIR}/metrics.json" || true
  fi

  echo "Completed run ${RUN_ID}"
  echo
done

echo "All sample runs completed. Check ${OUT_DIR}."
