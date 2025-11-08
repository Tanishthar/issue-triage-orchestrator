from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from packages.tools.mock_tool import mock_classify_issue
from packages.agents.reliability import log_step
from packages.tools import http_fetcher, vector_store, executor
from packages.eval import harness as eval_harness

app = FastAPI(title="Issue Triage Orchestrator", version="0.4.0")

class OrchestratorState(BaseModel):
    repo_url: str
    issue_text: str
    severity: Optional[str] = None
    repro_steps: Optional[str] = None
    proposed_fix: Optional[str] = None
    pr_url: Optional[str] = None
    status: str = "initialized"

@app.get("/")
async def root():
    return {"message": "Issue Triage Orchestrator running 🚀"}

@app.post("/start")
async def start_orchestration(state: OrchestratorState):
    try:
        log_step("start_orchestration", f"Received issue: {state.issue_text}")

        # 1) classify severity
        severity = mock_classify_issue(state.issue_text)
        state.severity = severity
        log_step("start_orchestration", f"Issue classified as {severity}")

        # 2) fetch repo README (best-effort)
        readme_result = None
        try:
            readme_url = state.repo_url.rstrip("/") + "/raw/main/README.md"
            fetch_res = http_fetcher.fetch(readme_url)
            readme_text = fetch_res["text"][:10000]  # trim for safety
            readme_result = {"fetched": True, "from_cache": fetch_res["from_cache"], "len": len(readme_text)}
            log_step("start_orchestration", f"Fetched README ({readme_url}) len={len(readme_text)}")
        except Exception as e:
            readme_result = {"fetched": False, "error": str(e)}
            log_step("start_orchestration", f"README fetch failed: {e}")

        # 3) index README into vector store (if fetched)
        vs = vector_store.MockVectorStore()
        if readme_result.get("fetched"):
            vs.add(f"{state.repo_url}::readme", readme_text)
            log_step("start_orchestration", "Indexed README into vector store")

        # 4) create a tiny failing test snippet (mock) and syntax-check
        failing_test_code = (
            "import pytest\n\n"
            "def test_sample_behavior():\n"
            "    assert 1 == 2  # intentionally failing test skeleton\n"
        )
        exec_check = executor.syntax_check(failing_test_code)
        log_step("start_orchestration", f"Executor check: {exec_check.get('message')}")
        state.status = "completed"

        metrics = eval_harness.compute_metrics()

        return {
            "final_state": state.model_dump(),
            "readme_result": readme_result,
            "executor_check": exec_check,
            "metrics": metrics,
            "message": "Run completed (dry-run)."
        }

    except Exception as e:
        log_step("start_orchestration", f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
async def get_metrics():
    """
    Returns the latest persisted metrics.json (if available),
    otherwise returns an empty dict.
    """
    return eval_harness.get_latest_metrics()