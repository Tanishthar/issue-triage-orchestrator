from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

app = FastAPI(title="Issue Triage Orchestrator", version="0.1.0")

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
    # Temporary mock-run: echo + set status
    try:
        state.status = "started"
        # In later phases this endpoint will enqueue tasks / start the agent graph
        return {"received_state": state.model_dump(), "message": "Workflow started!"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
