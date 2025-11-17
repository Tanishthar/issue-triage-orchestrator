import os
import json
import sys
import asyncio

# Fix for Windows event loop policy
# This ensures httpx.AsyncClient works properly in FastAPI on Windows
if sys.platform == 'win32':
    # Use WindowsSelectorEventLoopPolicy instead of WindowsProactorEventLoopPolicy
    # ProactorEventLoop can cause issues with asyncio subprocess and some async operations
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel
from typing import Optional, List, Dict

from packages.agents.reliability import log_step
from packages.ws.manager import manager as ws_manager
from packages.tools import http_fetcher, vector_store, executor
from packages.eval import harness as eval_harness
from packages.tools.github_mock import create_dry_pr
from packages.agents.llm_agent import LLMAgent
from packages.agents.tool_wrappers import register_all_tools

# Try to import httpx for Ollama healthcheck
try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False

app = FastAPI(title="Issue Triage Orchestrator", version="0.5.0")

LOG_FILE = "metrics/step_logs.json"

# Global variable to store Ollama probe status
_ollama_probe_status: Optional[Dict] = None

# --- CORS Configuration ---

origins = [
    # Allow the origin where your frontend (e.g., Next.js) is running
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    # Add any other specific origins your client might use (e.g., development staging)
    # NOTE: You can also use "*" to allow ALL origins, but this is less secure
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins, # <--- 2. Specify allowed client domains/ports
    allow_credentials=True, # Allow cookies/authentication headers
    allow_methods=["*"], # Allow all standard HTTP methods (GET, POST, PUT, DELETE, etc.)
    allow_headers=["*"], # Allow all headers (Content-Type, Authorization, etc.)
)

class OrchestratorState(BaseModel):
    repo_url: str
    issue_text: str
    readme_file_path: Optional[str] = None  # Optional: URL or local file path to README
    model: Optional[str] = None  # LLM model to use. If not provided, uses DEFAULT_LLM_MODEL env var or "ollama:llama3.1" as fallback. Only Ollama models are supported.
    severity: Optional[str] = None
    repro_steps: Optional[List[str]] = None  # Changed from Optional[str] to Optional[List[str]]
    proposed_fix: Optional[str] = None
    pr_url: Optional[str] = None
    status: str = "initialized"

def _probe_ollama() -> Dict:
    """Probe Ollama connectivity and return status"""
    ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    probe_urls = [
        ollama_base_url,
        ollama_base_url.replace("127.0.0.1", "localhost") if "127.0.0.1" in ollama_base_url else ollama_base_url.replace("localhost", "127.0.0.1"),
        "http://host.docker.internal:11434"
    ]
    
    if not HTTPX_AVAILABLE:
        return {
            "status": "error",
            "message": "httpx not available",
            "ollama_base_url": ollama_base_url,
            "probe_urls": probe_urls
        }
    
    for url in probe_urls:
        try:
            with httpx.Client(timeout=5.0) as client:
                response = client.get(f"{url}/api/tags")
                response.raise_for_status()
                # Try to get model list
                models = []
                try:
                    data = response.json()
                    if "models" in data:
                        models = [m.get("name", "") for m in data["models"]]
                except:
                    pass
                return {
                    "status": "healthy",
                    "ollama_base_url": url,
                    "probe_urls": probe_urls,
                    "models": models,
                    "message": f"Ollama is reachable at {url}"
                }
        except Exception as e:
            continue
    
    return {
        "status": "unhealthy",
        "ollama_base_url": ollama_base_url,
        "probe_urls": probe_urls,
        "message": f"Ollama is not reachable at any of the probe URLs: {probe_urls}",
        "error": "Connection failed to all probe URLs"
    }

@app.on_event("startup")
async def startup_event():
    """Probe Ollama at startup"""
    global _ollama_probe_status
    _ollama_probe_status = _probe_ollama()
    log_step("startup", f"Ollama probe status: {_ollama_probe_status.get('status')} - {_ollama_probe_status.get('message')}")

@app.get("/")
async def root():
    return {"message": "Issue Triage Orchestrator running 🚀"}

@app.get("/health")
async def healthcheck():
    """Lightweight healthcheck endpoint that probes Ollama and returns probe status"""
    global _ollama_probe_status
    # Re-probe on each request for real-time status
    probe_status = _probe_ollama()
    _ollama_probe_status = probe_status
    return {
        "service": "issue-triage-orchestrator",
        "status": "running",
        "ollama": probe_status
    }

@app.post("/start")
async def start_orchestration(state: OrchestratorState):
    try:
        log_step("start_orchestration", f"Received issue: {state.issue_text}")
        
        # Track all tool calls (both direct and LLM-initiated)
        total_tool_calls = 0
        log_step("tool_tracking", f"Initialized tool call counter: {total_tool_calls}")
        
        # Get model from state, environment variable, or default
        model = state.model or os.getenv("DEFAULT_LLM_MODEL", "ollama:llama3.1")
        log_step("start_orchestration", f"Using LLM model: {model}")

        # Get Ollama base URL from environment or use default
        # Use 127.0.0.1 for better Windows compatibility
        ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")

        # Initialize LLM agent with selected model and Ollama base URL
        agent = LLMAgent(model=model, ollama_base_url=ollama_base_url)
        register_all_tools(agent)
        log_step("start_orchestration", f"Initialized LLM agent with model {model} (base_url: {ollama_base_url}) and tools")

        # 1) classify severity using LLM
        severity = await agent.classify_severity(state.issue_text, state.repo_url)
        state.severity = severity
        log_step("start_orchestration", f"Issue classified as {severity} by LLM")

        # 2) fetch repo README (best-effort)
        readme_result = None
        readme_text = ""
        try:
            if state.readme_file_path:
                # Use provided README path/URL
                if state.readme_file_path.startswith("http://") or state.readme_file_path.startswith("https://"):
                    # It's a URL
                    readme_url = state.readme_file_path
                    log_step("tool_usage", "Used tool: fetch_url")
                    total_tool_calls += 1
                    fetch_res = http_fetcher.fetch(readme_url)
                    readme_text = fetch_res["text"][:10000]  # trim for safety
                    readme_result = {"fetched": True, "from_cache": fetch_res["from_cache"], "len": len(readme_text), "source": "provided_url"}
                    log_step("start_orchestration", f"Fetched README from provided URL ({readme_url}) len={len(readme_text)}")
                else:
                    # It's a local file path
                    if os.path.exists(state.readme_file_path):
                        with open(state.readme_file_path, "r", encoding="utf-8") as f:
                            readme_text = f.read()[:10000]  # trim for safety
                        readme_result = {"fetched": True, "len": len(readme_text), "source": "local_file"}
                        log_step("start_orchestration", f"Loaded README from local file ({state.readme_file_path}) len={len(readme_text)}")
                    else:
                        readme_result = {"fetched": False, "error": f"File not found: {state.readme_file_path}"}
                        log_step("start_orchestration", f"README file not found: {state.readme_file_path}")
            else:
                # Auto-detect README URL from repo
                readme_url = state.repo_url.rstrip("/") + "/raw/main/README.md"
                try:
                    log_step("tool_usage", "Used tool: fetch_url")
                    total_tool_calls += 1
                    fetch_res = http_fetcher.fetch(readme_url)
                    readme_text = fetch_res["text"][:10000]  # trim for safety
                    readme_result = {"fetched": True, "from_cache": fetch_res["from_cache"], "len": len(readme_text), "source": "auto_detected"}
                    log_step("start_orchestration", f"Fetched README (auto-detected: {readme_url}) len={len(readme_text)}")
                except Exception as e:
                    readme_result = {"fetched": False, "error": str(e), "source": "auto_detected"}
                    log_step("start_orchestration", f"README auto-fetch failed: {e}")
        except Exception as e:
            readme_result = {"fetched": False, "error": str(e)}
            log_step("start_orchestration", f"README fetch failed: {e}")

        # 3) index README into vector store (if fetched)
        vs = vector_store.MockVectorStore()
        vector_search_results = None
        if readme_result.get("fetched"):
            vs.add(f"{state.repo_url}::readme", readme_text)
            log_step("start_orchestration", "Indexed README into vector store")
            
            # Optionally search vector store for relevant context
            try:
                log_step("tool_usage", "Used tool: search_documentation")
                total_tool_calls += 1
                search_results = vs.search(state.issue_text[:200], top_k=2)
                vector_search_results = [{"doc_id": doc_id, "score": score} for doc_id, score in search_results]
                log_step("start_orchestration", f"Found {len(vector_search_results)} relevant docs in vector store")
            except Exception as e:
                log_step("start_orchestration", f"Vector search failed: {str(e)[:200]}")

        # 4) extract repro steps and propose fix using LLM
        repro_steps = await agent.extract_repro_steps(state.issue_text)
        state.repro_steps = repro_steps  # Assign list to state
        log_step("start_orchestration", f"Extracted {len(repro_steps)} repro steps using LLM")
        
        # Build context for fix proposal
        context = {
            "readme_fetched": readme_result.get("fetched", False),
            "vector_search_results": vector_search_results
        }
        
        # Use LLM to propose fix with tool-driven reasoning if needed
        if readme_result.get("fetched") and len(readme_text) > 100:
            # Use tool-driven reasoning for complex issues
            reasoning_query = f"""Analyze this GitHub issue and propose a fix:

Repository: {state.repo_url}
Issue: {state.issue_text}
Severity: {severity}
Repro Steps: {chr(10).join(f"{i+1}. {step}" for i, step in enumerate(repro_steps))}

You can use tools to:
1. Search the documentation for relevant context
2. Fetch additional information if needed
3. Check code syntax

After gathering context, propose a fix sketch and generate a failing pytest test."""
            
            reasoning_result = await agent.reason_with_tools(reasoning_query, max_iterations=3)
            llm_tool_calls_count = len(reasoning_result.get('tool_calls', []))
            total_tool_calls += llm_tool_calls_count
            log_step("start_orchestration", f"LLM reasoning completed with {reasoning_result.get('iterations', 0)} iterations")
            log_step("start_orchestration", f"LLM tool calls made: {llm_tool_calls_count}")
            log_step("start_orchestration", f"Total tool calls made: {total_tool_calls}")

            # Extract fix from reasoning or fall back to direct proposal
            if reasoning_result.get("answer"):
                # Try to extract fix from reasoning answer
                fix_sketch, failing_test_code = await agent.propose_fix(
                    state.issue_text, severity, repro_steps,
                    repo_readme=readme_text, context=context
                )
            else:
                fix_sketch, failing_test_code = await agent.propose_fix(
                    state.issue_text, severity, repro_steps,
                    repo_readme=readme_text, context=context
                )
        else:
            # Direct proposal without tool reasoning
            fix_sketch, failing_test_code = await agent.propose_fix(
                state.issue_text, severity, repro_steps,
                repo_readme=readme_text, context=context
            )
        
        log_step("start_orchestration", "Generated fix sketch and failing test using LLM")

        # 5) syntax-check the generated failing test
        log_step("tool_usage", "Used tool: check_code_syntax")
        total_tool_calls += 1
        exec_check = executor.syntax_check(failing_test_code)
        log_step("start_orchestration", f"Executor check for generated test: {exec_check.get('message')}")

        # 6) create dry-run PR artifact locally (mock)
        pr_title = f"[Triage] proposed fix - {state.issue_text[:60]}"
        pr_body = f"Automated triage: proposed fix sketch\\n\\n{fix_sketch}\\n\\nRepro steps:\\n" + "\\n".join(f"- {s}" for s in repro_steps)
        branch = f"triage/proposed-fix-{state.severity or 'unknown'}-{state.repo_url.split('/')[-1][:12]}"
        pr_files = {
            # place tests in a conventional path
            f"tests/test_regression_{state.repo_url.split('/')[-1][:12]}.py": failing_test_code
        }
        pr_result = create_dry_pr(pr_title, pr_body, branch, pr_files)
        state.pr_url = pr_result["pr_url"]
        log_step("start_orchestration", f"Created dry-run PR: {state.pr_url}")

        state.status = "completed"
        
        # Log final tool call count
        log_step("start_orchestration", f"Final total tool calls: {total_tool_calls}")

        metrics = eval_harness.compute_metrics()
        
        # Broadcast metrics update to WebSocket clients
        try:
            await ws_manager.broadcast({"type": "metrics_update", "metrics": metrics})
        except Exception:
            # best-effort only; do not raise
            pass

        return {
            "final_state": state.model_dump(),
            "readme_result": readme_result,
            "executor_check": exec_check,
            "metrics": metrics,
            "pr_result": pr_result,
            "message": "Run completed with LLM agent.",
            "llm_mode": "real" if not agent.use_mock else "mock"
        }

    except Exception as e:
        # Safely convert exception to string (handles ndarray and other non-serializable types)
        error_msg = str(e)
        # If the error message is too long or contains complex objects, truncate it
        if len(error_msg) > 1000:
            error_msg = error_msg[:1000] + "... (truncated)"
        log_step("start_orchestration", f"Error: {error_msg}")
        raise HTTPException(status_code=500, detail=error_msg)

@app.get("/metrics")
async def get_metrics():
    """
    Returns the latest persisted metrics.json (if available),
    otherwise returns an empty dict.
    """
    return eval_harness.get_latest_metrics()

@app.get("/logs")
async def get_logs(tail: int = 200):
    """
    Return the last `tail` log entries from metrics/step_logs.json.
    Useful for the UI to show a snapshot without WS.
    """
    if not os.path.exists(LOG_FILE):
        return []
    try:
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            logs = json.load(f)
    except Exception:
        return []
    # Return last `tail` entries
    return logs[-tail:]

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint that:
      - accepts connection
      - sends an initial snapshot message: { type: "snapshot", metrics: {...}, logs: [...] }
      - then keeps connection open; new log lines are broadcast by WSManager
    """
    # Connect and register
    await ws_manager.connect(websocket)
    try:
        # Send initial snapshot
        metrics = eval_harness.get_latest_metrics()
        # send tail of logs
        logs = []
        if os.path.exists(LOG_FILE):
            try:
                with open(LOG_FILE, "r", encoding="utf-8") as f:
                    logs = json.load(f)
            except Exception:
                logs = []
        initial_payload = {"type": "snapshot", "metrics": metrics, "log_lines": logs[-500:]}
        await websocket.send_text(json.dumps(initial_payload))

        # keep the connection alive; optionally accept client messages
        while True:
            try:
                msg = await websocket.receive_text()
            except WebSocketDisconnect:
                break
            except Exception:
                # Any other receive error -> disconnect
                break
            # Optionally support subscribe messages in the future:
            # ignore incoming messages for now
    finally:
        await ws_manager.disconnect(websocket)