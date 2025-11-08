# Design Notes — Issue Triage Orchestrator

## Goal recap
Implement a 2–3 agent orchestrator that triages GitHub issues end-to-end:
- classify severity
- extract repro steps
- propose fix sketch
- generate failing unit test
- create dry-run PR artifact

## Orchestration style
- Custom LangGraph-like graph where each node is a typed Pydantic state transition.
- State is JSON with explicit types; tools accept typed input and return typed output.
- Benefits: deterministic schemas, easier unit tests, and tractable retries.

## Why this graph?
- Simplicity: linear triage pipeline fits a small DAG.
- Observability: per-step logs + typed state allow replay and auditing.
- Failure handling: retries and circuit-breaker at tool-call boundary prevents runaway loops.

## Tools
- Safe HTTP Fetcher: respects robots.txt and caches results (low-cost recon).
- Mock Vector Store: FAISS-like local mock for retrieval.
- Executor Sandbox: restricted static analysis + dry-run behaviors for generating failing tests.

## Reliability patterns
- Retries with exponential backoff for transient errors.
- Circuit-breaker trips after N consecutive tool failures — returns graceful fallback and records metrics.
- Step logger persists to `metrics/step_logs.json` and broadcasts to /ws.

## Evaluation harness
- Runs automated checks: schema validation, number of unique sources, PR artifact presence, failing test presence.
- Metrics recorded: `steps_count`, `tool_error_rate`, `eval_score`.

## Failure modes & mitigations
1. **LLM or external tool timeout** — Mitigate: timeouts + retries + fallback prompt that asks for minimal output.
2. **Malicious repo content** — Mitigate: static-only analysis in the sandbox, do not run code. Use syntax checks only.
3. **Inconsistent state** — Mitigate: atomic writes to metrics/ and PR artifacts; idempotent run IDs.
4. **High cost LLM usage** — Mitigate: cap tokens, use cached responses or smaller LLMs for sketching.

## Security considerations
- No real PRs unless GITHUB_TOKEN provided. Dry-run artifacts are stored locally.
- Sandbox is a static analysis mock; never `exec` code from untrusted repos.
- Secrets in `.env` — recommend Docker secrets or vault in production.

## Next steps (stretch)
- Implement LLM selection and budget-based switching.
- Introduce a small Qdrant/FAISS-backed cache for tool outputs.
- Add canary runs to detect drift in eval_score across sample runs.