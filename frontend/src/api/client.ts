/**
 * The one place the SPA talks to the API. Wraps `/ask/stream` (the live agentic
 * run, read as Server-Sent Events over `fetch`) and `/health`, maps transport and
 * HTTP errors to typed results, and reads the base URL from the Vite env so the
 * same build works in dev and in compose.
 */

import type { AgenticTrace, AskResponse } from "../types";

const API_URL = import.meta.env.VITE_API_URL ?? "http://localhost:8000";

export type HealthStatus = "checking" | "ok" | "degraded" | "unreachable";

export class AskError extends Error {}

/** Callbacks the streamed consultation drives: a step per node, then the result. */
export interface StreamHandlers {
  onStep: (step: string, agentic: AgenticTrace) => void;
  onResult: (response: AskResponse) => void;
}

/**
 * Submit a Mode 1 question and narrate the run as it happens: `onStep` fires after
 * each graph node with the live agentic trace, `onResult` with the final answer.
 * Throws `AskError` on a transport or HTTP failure; a caller abort is silent.
 */
export async function askQuestionStream(
  question: string,
  handlers: StreamHandlers,
  signal?: AbortSignal,
): Promise<void> {
  let response: Response;
  try {
    response = await fetch(`${API_URL}/ask/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
      signal,
    });
  } catch {
    if (signal?.aborted) return;
    throw new AskError("No se pudo contactar con el servicio.");
  }
  if (!response.ok || !response.body) {
    throw new AskError(`El servicio respondió con un error (${response.status}).`);
  }
  await consume(response.body, handlers);
}

/** Read the SSE body frame by frame, dispatching each `step`/`result` event. */
async function consume(body: ReadableStream<Uint8Array>, handlers: StreamHandlers): Promise<void> {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let boundary = buffer.indexOf("\n\n");
    while (boundary !== -1) {
      dispatch(buffer.slice(0, boundary), handlers);
      buffer = buffer.slice(boundary + 2);
      boundary = buffer.indexOf("\n\n");
    }
  }
}

/** Parse one SSE frame and route it to the matching handler. */
function dispatch(frame: string, handlers: StreamHandlers): void {
  let event = "message";
  let data = "";
  for (const line of frame.split("\n")) {
    if (line.startsWith("event:")) event = line.slice("event:".length).trim();
    else if (line.startsWith("data:")) data += line.slice("data:".length).trim();
  }
  if (!data) return;
  if (event === "result") {
    handlers.onResult(JSON.parse(data) as AskResponse);
  } else if (event === "step") {
    const payload = JSON.parse(data) as { step: string; agentic: AgenticTrace };
    handlers.onStep(payload.step, payload.agentic);
  }
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
