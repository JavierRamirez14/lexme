/**
 * The one place the SPA talks to the API. Wraps the `/ask` and `/health`
 * endpoints, maps transport and HTTP errors to typed results, and reads the base
 * URL from the Vite env so the same build works in dev and in compose.
 */

import type { AskResponse } from "../types";

const API_URL = import.meta.env.VITE_API_URL ?? "http://localhost:8000";

export type HealthStatus = "checking" | "ok" | "degraded" | "unreachable";

export class AskError extends Error {}

/** Submit a Mode 1 question, returning the parsed answer or throwing `AskError`. */
export async function askQuestion(question: string, signal?: AbortSignal): Promise<AskResponse> {
  let response: Response;
  try {
    response = await fetch(`${API_URL}/ask`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
      signal,
    });
  } catch {
    throw new AskError("No se pudo contactar con el servicio.");
  }
  if (!response.ok) {
    throw new AskError(`El servicio respondió con un error (${response.status}).`);
  }
  return (await response.json()) as AskResponse;
}

/** Probe the API health endpoint, collapsing the result to a display status. */
export async function fetchHealth(signal?: AbortSignal): Promise<HealthStatus> {
  try {
    const response = await fetch(`${API_URL}/health`, { signal });
    const body = (await response.json()) as { status?: string };
    return body.status === "ok" ? "ok" : "degraded";
  } catch {
    return "unreachable";
  }
}
