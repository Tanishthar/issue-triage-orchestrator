#!/usr/bin/env bash
set -euo pipefail

API="http://localhost:8000"
OUT_DIR="./sample_runs"
DRYPRS_DIR="./packages/tools/_dry_prs"

mkdir -p "$OUT_DIR"
mkdir -p "$DRYPRS_DIR"

declare -a ISSUES=(
  "Repo: sample-repo-1 | Issue: 'App crashes on startup when config has empty value. Steps: open app, click start.'"
  "Repo: sample-repo-2 | Issue: 'Unit tests failing intermittently in CI after dependency upgrade to x.y.z. Steps: run pytest -k integration.'"
  "Repo: sample-repo-3 | Issue: 'Feature X returns incorrect values for edge-case input. Steps: call /api/value?x=9999.'"
)

timestamp() { date +%Y%m%dT%H%M%S; }

for i in "${!ISSUES[@]}"; do
  RUN_ID="run-$((i+1))-$(timestamp)"
  RUN_DIR="${OUT_DIR}/${RUN_ID}"
  mkdir -p "$RUN_DIR"

  ISSUE_TEXT="${ISSUES[$i]}"

  echo "=== Starting sample run: $RUN_ID ==="
  echo "Issue text: $ISSUE_TEXT"

  # POST to /start — adjust payload shape if your API uses different keys
  RESPONSE=$(curl -sS -X POST "${API}/start" \
    -H "Content-Type: application/json" \
    -d "$(jq -n --arg repo "sample-repo-${i}" --arg issue "$ISSUE_TEXT" '{repo:$repo, issue_text:$issue}')" )

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
