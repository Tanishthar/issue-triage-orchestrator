#!/usr/bin/env bash
set -euo pipefail

API="http://localhost:8000"
OUT_DIR="./sample_runs"
DRYPRS_DIR="./packages/tools/_dry_prs"

mkdir -p "$OUT_DIR"
mkdir -p "$DRYPRS_DIR"

declare -a ISSUES=(
  "Repo: https://github.com/pcqpcq/open-source-android-apps |
  Issue: 'bcdboot.exe failed with code -1073741510!'|
  README : https://github.com/pcqpcq/open-source-android-apps/blob/master/README.md "
  "Repo: https://github.com/tortuvshin/open-source-flutter-apps | 
  Issue: 'App crashes on startup!' |
  README : https://github.com/tortuvshin/open-source-flutter-apps/blob/main/README.md "
  "Repo: https://github.com/opencv/opencv | 
  Issue : 'repeated complex calculation in audio buffer duration computation in CvCapture_MSMF::grabFrame()' |
  README: https://github.com/opencv/opencv/blob/4.x/README.md "
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
