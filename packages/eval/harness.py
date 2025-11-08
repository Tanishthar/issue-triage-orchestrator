import json
import os
from datetime import datetime
from typing import Dict, Any

LOG_FILE = "metrics/step_logs.json"
METRICS_FILE = "metrics/metrics.json"

def _load_logs(path: str = LOG_FILE):
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except Exception:
            return []

def compute_metrics(log_path: str = LOG_FILE, out_path: str = METRICS_FILE) -> Dict[str, Any]:
    """
    Reads step logs and computes:
      - steps_count: total number of log entries
      - unique_steps: number of distinct step names
      - error_count: number of entries that look like errors
      - tool_error_rate: error_count / steps_count (0..1)
      - avg_errors_per_step: error_count / unique_steps
      - eval_score: heuristic score (0..100)
    Persists metrics to out_path and returns the metrics dict.
    """
    logs = _load_logs(log_path)
    steps_count = len(logs)
    unique_steps = len({entry.get("step") for entry in logs if entry.get("step")})
    # error heuristics: look for common words indicating failure
    error_indicators = ["failed", "error", "Exception", "FAIL", "Circuit open", "Circuit breaker OPEN", "transient"]
    error_count = 0
    for entry in logs:
        msg = entry.get("message", "")
        if any(ind in msg for ind in error_indicators):
            error_count += 1

    tool_error_rate = (error_count / steps_count) if steps_count > 0 else 0.0
    avg_errors_per_step = (error_count / unique_steps) if unique_steps > 0 else 0.0

    # Simple heuristic for eval_score:
    # - starts at 100
    # - penalize by error rate (heavy)
    # - small additional penalty if average errors per step is high (repeated failures)
    # - clamp between 0 and 100
    score = 100.0
    score *= max(0.0, 1.0 - tool_error_rate)           # penalize by error rate
    # penalize repeated failures per step up to 50% extra penalty
    repeat_penalty = min(0.5, avg_errors_per_step / 5.0)
    score *= max(0.0, 1.0 - repeat_penalty)

    metrics = {
        "generated_at": datetime.utcnow().isoformat(),
        "steps_count": steps_count,
        "unique_steps": unique_steps,
        "error_count": error_count,
        "tool_error_rate": round(tool_error_rate, 4),
        "avg_errors_per_step": round(avg_errors_per_step, 4),
        "eval_score": int(round(score)),
    }

    # ensure directory exists
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return metrics

def get_latest_metrics(path: str = METRICS_FILE):
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except Exception:
            return {}
