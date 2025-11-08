import React, { useEffect, useRef, useState } from "react";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000";
const WS_BASE = process.env.NEXT_PUBLIC_WS_BASE || ""; // e.g. ws://localhost:8000/ws

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

  // Start a run by hitting POST /start
  async function startRun() {
    setRunning(true);
    try {
      const payload = { repo_url: "https://github.com/test/repo", issue_text: "Sample crash on startup" };
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

  // connect WebSocket if WS_BASE provided
  useEffect(() => {
    if (!WS_BASE) {
      // polling fallback
      const mi = setInterval(fetchMetrics, 3000);
      fetchMetrics();
      return () => clearInterval(mi);
    }

    let ws: WebSocket;
    try {
      ws = new WebSocket(WS_BASE);
      wsRef.current = ws;
    } catch (e) {
      console.warn("WebSocket init failed, falling back to polling", e);
      const mi = setInterval(fetchMetrics, 3000);
      fetchMetrics();
      return () => clearInterval(mi);
    }

    ws.onopen = () => {
      console.info("WS connected");
      setConnected(true);
      // Optionally request an initial snapshot
      ws.send(JSON.stringify({ type: "subscribe", channel: "metrics_and_logs" }));
    };

    ws.onmessage = (ev) => {
      try {
        const data = JSON.parse(ev.data);
        if (data.metrics) setMetrics(data.metrics);
        if (data.log_line) {
          setLogs((prev) => [...prev, data.log_line].slice(-500)); // keep last 500
        }
        if (data.log_lines && Array.isArray(data.log_lines)) {
          setLogs((prev) => [...prev, ...data.log_lines].slice(-500));
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
      fetchMetrics();
      // attempt reconnect after a delay
      const to = setTimeout(() => {
        // reload effect will try to reconnect if WS_BASE still set
        window.location.reload();
      }, 5000);

      // cleanup on unmount
      return () => {
        clearInterval(mi);
        clearTimeout(to);
      };
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

  // auto-scroll logs
  const logsRef = React.useRef<HTMLDivElement | null>(null);
  useEffect(() => {
    if (logsRef.current) {
      logsRef.current.scrollTop = logsRef.current.scrollHeight;
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

            <button
              onClick={startRun}
              disabled={running}
              className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 disabled:opacity-50"
            >
              {running ? "Starting..." : "Start Run"}
            </button>
          </div>
        </header>

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