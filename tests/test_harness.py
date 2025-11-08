import json
import os
from packages.eval import harness

def test_compute_metrics_on_empty_logs(tmp_path, monkeypatch):
    # create an empty log file
    logfile = tmp_path / "step_logs.json"
    logfile.write_text("[]")
    out = tmp_path / "metrics.json"
    metrics = harness.compute_metrics(str(logfile), str(out))
    assert metrics["steps_count"] == 0
    assert metrics["error_count"] == 0
    assert metrics["tool_error_rate"] == 0.0
    assert "eval_score" in metrics
