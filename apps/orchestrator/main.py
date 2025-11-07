from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from packages.tools.mock_tool import mock_classify_issue
from packages.agents.reliability import log_step

app = FastAPI(title="Issue Triage Orchestrator", version="0.2.0")

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
        severity = mock_classify_issue(state.issue_text)
        state.severity = severity
        state.status = "classified"
        log_step("start_orchestration", f"Issue classified as {severity}")
        return {"final_state": state.model_dump(), "message": "Classification done!"}
    except Exception as e:
        log_step("start_orchestration", f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
