import React, { useEffect, useRef, useState } from "react";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000";
// Construct WS URL from API_BASE if WS_BASE is not explicitly set
const WS_BASE = process.env.NEXT_PUBLIC_WS_BASE || (API_BASE.replace(/^http/, "ws") + "/ws");

type Metrics = {
  generated_at?: string;
  steps_count?: number;
  unique_steps?: number;
  error_count?: number;
  tool_error_rate?: number;
  avg_errors_per_step?: number;
  eval_score?: number;
};

export default function DashboardPage() {
  const [running, setRunning] = useState(false);
  const [metrics, setMetrics] = useState<Metrics>({});
  const [logs, setLogs] = useState<string[]>([]);
  const wsRef = useRef<WebSocket | null>(null);
  const [connected, setConnected] = useState(false);
  
  // Form state
  const [repoUrl, setRepoUrl] = useState("");
  const [issueText, setIssueText] = useState("");
  const [readmePath, setReadmePath] = useState("");
  const [selectedModel, setSelectedModel] = useState("ollama:llama3.1");

  // Start a run by hitting POST /start
  async function startRun(e?: React.FormEvent) {
    e?.preventDefault();
    
    // Validation
    if (!repoUrl.trim()) {
      alert("Please enter a GitHub repository URL");
      return;
    }
    if (!issueText.trim()) {
      alert("Please enter issue text");
      return;
    }
    
    setRunning(true);
    try {
      const payload: any = {
        repo_url: repoUrl.trim(),
        issue_text: issueText.trim(),
        model: selectedModel,  // Include selected model
      };
      
      // Add optional readme_file_path if provided
      if (readmePath.trim()) {
        payload.readme_file_path = readmePath.trim();
      }
      
      const res = await fetch(`${API_BASE}/start`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!res.ok) throw new Error(await res.text());
      const body = await res.json();
      if (body.metrics) setMetrics(body.metrics);
      // Optional: backend may also push WS messages
    } catch (err) {
      console.error("Start run failed", err);
      alert("Start failed: " + String(err));
    } finally {
      setRunning(false);
    }
  }

  // HTTP polling fallback when no WS
  async function fetchMetrics() {
    try {
      const res = await fetch(`${API_BASE}/metrics`);
      if (!res.ok) return;
      const data = await res.json();
      setMetrics(data);
    } catch (e) {
      // ignore
    }
  }

  // Fetch logs from HTTP endpoint
  async function fetchLogs() {
    try {
      const res = await fetch(`${API_BASE}/logs?tail=500`);
      if (!res.ok) return;
      const data = await res.json();
      if (Array.isArray(data)) {
        // Convert log entries to string format for display
        // Reverse so latest logs appear on top
        const logStrings = data
          .map((entry: any) => {
            const timestamp = entry.timestamp || "";
            const step = entry.step || "";
            const message = entry.message || "";
            return `[${timestamp}] ${step}: ${message}`;
          })
          .reverse(); // Reverse to show latest first
        setLogs(logStrings);
      }
    } catch (e) {
      // ignore
    }
  }

  // Initial load: fetch metrics and logs
  useEffect(() => {
    fetchMetrics();
    fetchLogs();
  }, []);

  // connect WebSocket if WS_BASE provided
  useEffect(() => {
    if (!WS_BASE) {
      // polling fallback
      const mi = setInterval(fetchMetrics, 3000);
      const li = setInterval(fetchLogs, 3000);
      fetchMetrics();
      fetchLogs();
      return () => {
        clearInterval(mi);
        clearInterval(li);
      };
    }

    let ws: WebSocket;
    try {
      ws = new WebSocket(WS_BASE);
      wsRef.current = ws;
    } catch (e) {
      console.warn("WebSocket init failed, falling back to polling", e);
      const mi = setInterval(fetchMetrics, 3000);
      const li = setInterval(fetchLogs, 3000);
      fetchMetrics();
      fetchLogs();
      return () => {
        clearInterval(mi);
        clearInterval(li);
      };
    }

    ws.onopen = () => {
      console.info("WS connected");
      setConnected(true);
      // Initial snapshot is sent automatically by the server, but we can request it explicitly
      // The server sends snapshot on connect, so we don't need to request it
    };

    ws.onmessage = (ev) => {
      try {
        const data = JSON.parse(ev.data);
        if (data.type === "snapshot") {
          // Initial snapshot from server
          if (data.metrics) setMetrics(data.metrics);
          if (data.log_lines && Array.isArray(data.log_lines)) {
            // Reverse so latest logs appear on top
            const logStrings = data.log_lines
              .map((entry: any) => {
                const timestamp = entry.timestamp || "";
                const step = entry.step || "";
                const message = entry.message || "";
                return `[${timestamp}] ${step}: ${message}`;
              })
              .reverse(); // Reverse to show latest first
            setLogs(logStrings);
          }
        } else if (data.type === "metrics_update") {
          // Metrics update from server
          if (data.metrics) setMetrics(data.metrics);
        } else if (data.type === "log_line" || data.log_line) {
          // Real-time log line update - prepend to show latest on top
          const entry = data.log_line;
          if (entry) {
            const timestamp = entry.timestamp || "";
            const step = entry.step || "";
            const message = entry.message || "";
            const logString = `[${timestamp}] ${step}: ${message}`;
            // Prepend new log and keep last 500 (oldest will be removed from end)
            setLogs((prev) => [logString, ...prev].slice(0, 500));
          }
        } else {
          // Legacy format or other updates
          if (data.metrics) setMetrics(data.metrics);
          if (data.log_lines && Array.isArray(data.log_lines)) {
            const logStrings = data.log_lines.map((entry: any) => {
              const timestamp = entry.timestamp || "";
              const step = entry.step || "";
              const message = entry.message || "";
              return `[${timestamp}] ${step}: ${message}`;
            });
            // Prepend and keep last 500
            setLogs((prev) => [...logStrings.reverse(), ...prev].slice(0, 500));
          }
        }
      } catch (err) {
        console.warn("WS message parse error", err);
      }
    };

    ws.onclose = () => {
      console.info("WS closed");
      setConnected(false);
      // fall back to polling
      const mi = setInterval(fetchMetrics, 3000);
      const li = setInterval(fetchLogs, 3000);
      fetchMetrics();
      fetchLogs();
      // attempt reconnect after a delay
      setTimeout(() => {
        // reload effect will try to reconnect if WS_BASE still set
        window.location.reload();
      }, 5000);
    };

    ws.onerror = (e) => {
      console.warn("WS error", e);
    };

    return () => {
      try {
        ws.close();
      } catch (e) {}
      wsRef.current = null;
      setConnected(false);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [WS_BASE]);

  // auto-scroll logs to top (since latest are on top)
  const logsRef = React.useRef<HTMLDivElement | null>(null);
  useEffect(() => {
    if (logsRef.current) {
      logsRef.current.scrollTop = 0; // Scroll to top since latest logs are first
    }
  }, [logs]);

  return (
    <main className="min-h-screen bg-gray-50 p-6">
      <div className="max-w-4xl mx-auto">
        <header className="flex items-center justify-between mb-6">
          <div>
            <h1 className="text-2xl font-semibold">Issue Triage Orchestrator</h1>
            <div className="text-sm text-gray-500">Backend: {API_BASE}</div>
          </div>

          <div className="flex items-center gap-3">
            <div className="flex items-center gap-2">
              <span className={`inline-block w-3 h-3 rounded-full ${connected ? "bg-green-500" : "bg-gray-400"}`}></span>
              <span className="text-sm text-gray-700">{connected ? "Live" : "Idle"}</span>
            </div>
          </div>
        </header>

        {/* Input Form Section */}
        <section className="mb-6 p-4 bg-white rounded shadow">
          <h2 className="text-xl font-semibold mb-4">Run Issue Triage</h2>
          <form onSubmit={startRun} className="space-y-4">
            <div>
              <label htmlFor="repo-url" className="block text-sm font-medium text-gray-700 mb-1">
                GitHub Repository URL <span className="text-red-500">*</span>
              </label>
              <input
                id="repo-url"
                type="text"
                value={repoUrl}
                onChange={(e) => setRepoUrl(e.target.value)}
                placeholder="https://github.com/owner/repo"
                disabled={running}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:bg-gray-100"
                required
              />
            </div>

            <div>
              <label htmlFor="issue-text" className="block text-sm font-medium text-gray-700 mb-1">
                Issue Text <span className="text-red-500">*</span>
              </label>
              <textarea
                id="issue-text"
                value={issueText}
                onChange={(e) => setIssueText(e.target.value)}
                placeholder="Describe the issue, error message, or problem..."
                disabled={running}
                rows={5}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:bg-gray-100 font-mono text-sm"
                required
              />
            </div>

            <div>
              <label htmlFor="readme-path" className="block text-sm font-medium text-gray-700 mb-1">
                README File Path/URL <span className="text-gray-500 text-xs">(Optional)</span>
              </label>
              <input
                id="readme-path"
                type="text"
                value={readmePath}
                onChange={(e) => setReadmePath(e.target.value)}
                placeholder="https://github.com/owner/repo/raw/main/README.md or /path/to/README.md"
                disabled={running}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:bg-gray-100"
              />
              <p className="mt-1 text-xs text-gray-500">
                Leave empty to auto-detect from repository, or provide a URL or local file path
              </p>
            </div>

            <div>
              <label htmlFor="model-select" className="block text-sm font-medium text-gray-700 mb-1">
                LLM Model <span className="text-gray-500 text-xs">(Select AI model)</span>
              </label>
              <select
                id="model-select"
                value={selectedModel}
                onChange={(e) => setSelectedModel(e.target.value)}
                disabled={running}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:bg-gray-100 bg-white"
              >
                <option value="ollama:llama3.1">Llama 3.1 (Ollama - Local) - Default</option>
                <option value="gemini-2.5-flash">Gemini 2.5 Flash (Google)</option>
              </select>
              <p className="mt-1 text-xs text-gray-500">
                Choose the AI model for issue triage. Requires GEMINI_API_KEY for Gemini models. For Ollama models, ensure Ollama is running locally at http://localhost:11434.
              </p>
            </div>

            <div className="flex justify-end">
              <button
                type="submit"
                disabled={running || !repoUrl.trim() || !issueText.trim()}
                className="px-6 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {running ? "Running..." : "Start Triage"}
              </button>
            </div>
          </form>
        </section>

        <section className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
          <div className="p-4 bg-white rounded shadow">
            <h3 className="font-medium">Eval Score</h3>
            <div className="text-3xl mt-2">{metrics.eval_score ?? "—"}</div>
            <div className="text-sm text-gray-500 mt-1">generated: {metrics.generated_at ?? "—"}</div>
          </div>

          <div className="p-4 bg-white rounded shadow">
            <h3 className="font-medium">Steps</h3>
            <div className="text-2xl mt-2">{metrics.steps_count ?? "—"}</div>
            <div className="text-sm text-gray-500 mt-1">unique: {metrics.unique_steps ?? "—"}</div>
          </div>

          <div className="p-4 bg-white rounded shadow">
            <h3 className="font-medium">Errors</h3>
            <div className="text-2xl mt-2">{metrics.error_count ?? "—"}</div>
            <div className="text-sm text-gray-500 mt-1">rate: {metrics.tool_error_rate ?? "—"}</div>
          </div>
        </section>

        <section className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="p-4 bg-white rounded shadow">
            <h3 className="font-medium mb-2">Logs</h3>
            <div ref={logsRef} className="h-64 overflow-auto bg-gray-100 p-2 rounded">
              {logs.length === 0 ? (
                <div className="text-sm text-gray-500">No logs yet. {WS_BASE ? "Waiting for live updates..." : "Metrics polling is active."}</div>
              ) : (
                logs.map((l, i) => (
                  <div key={i} className="text-xs font-mono text-gray-700">{l}</div>
                ))
              )}
            </div>
          </div>

          <div className="p-4 bg-white rounded shadow">
            <h3 className="font-medium mb-2">Raw Metrics</h3>
            <pre className="text-xs font-mono bg-gray-100 p-2 rounded">{JSON.stringify(metrics, null, 2)}</pre>
          </div>
        </section>

        <footer className="mt-6 text-sm text-gray-500">WS: {WS_BASE || "not configured (polling fallback)"}</footer>
      </div>
    </main>
  );
}