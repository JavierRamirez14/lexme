/**
 * The one place the SPA talks to the API. Wraps `/ask/stream` (the live agentic
 * run, read as Server-Sent Events over `fetch`), its `/ask/resume/stream`
 * counterpart for a run that paused to ask something, and `/health`. It maps
 * transport and HTTP errors to typed results, and reads the base URL from the
 * Vite env so the same build works in dev and in compose.
 */

import type { AgenticTrace, AskResponse } from "../types";
import type { ContractAnalysis } from "../mode2/types";

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
  await streamRun("/ask/stream", { question }, handlers, signal);
}

/**
 * Answer the disambiguating question a paused run asked and let it continue on the
 * same `threadId`, narrated exactly like the first leg. An empty `answer` is a
 * valid reply: the run proceeds on a stated assumption instead.
 */
export async function resumeQuestionStream(
  threadId: string,
  answer: string,
  handlers: StreamHandlers,
  signal?: AbortSignal,
): Promise<void> {
  await streamRun("/ask/resume/stream", { thread_id: threadId, answer }, handlers, signal);
}

/** POST `body` to an SSE endpoint and drive `handlers` with the events it sends. */
async function streamRun(
  path: string,
  body: object,
  handlers: StreamHandlers,
  signal?: AbortSignal,
): Promise<void> {
  let response: Response;
  try {
    response = await fetch(`${API_URL}${path}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
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

/**
 * Upload a lease to `/contract/analyze` and return its analysis. The document is
 * sent as multipart form data and never leaves this request; the API holds it in
 * memory and persists nothing. Throws `AskError` on a transport or HTTP failure.
 */
export async function analyzeContract(
  file: File,
  signal?: AbortSignal,
): Promise<ContractAnalysis> {
  const body = new FormData();
  body.append("file", file);
  let response: Response;
  try {
    response = await fetch(`${API_URL}/contract/analyze`, {
      method: "POST",
      body,
      signal,
    });
  } catch {
    if (signal?.aborted) throw new AskError("cancelado");
    throw new AskError("No se pudo contactar con el servicio.");
  }
  if (!response.ok) {
    throw new AskError(`El servicio respondió con un error (${response.status}).`);
  }
  return (await response.json()) as ContractAnalysis;
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
