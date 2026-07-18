import { useEffect, useState } from "react";

const API_URL = import.meta.env.VITE_API_URL ?? "http://localhost:8000";

type HealthState = "loading" | "ok" | "degraded" | "unreachable";

export function App() {
  const [health, setHealth] = useState<HealthState>("loading");

  useEffect(() => {
    let cancelled = false;

    async function checkHealth() {
      try {
        const response = await fetch(`${API_URL}/health`);
        const body = (await response.json()) as { status?: string };
        if (!cancelled) {
          setHealth(body.status === "ok" ? "ok" : "degraded");
        }
      } catch {
        if (!cancelled) {
          setHealth("unreachable");
        }
      }
    }

    checkHealth();
    return () => {
      cancelled = true;
    };
  }, []);

  return (
    <main style={{ fontFamily: "system-ui, sans-serif", padding: "2rem", maxWidth: 640 }}>
      <h1>Lexme</h1>
      <p>Agentic RAG over Spanish legislation — skeleton stub.</p>
      <p>
        API health: <strong>{health}</strong>
      </p>
    </main>
  );
}
