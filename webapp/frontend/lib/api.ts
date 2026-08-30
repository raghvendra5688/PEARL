import type { PredictErrorEvent, PredictResult, ProfileResponse, ResolveResponse } from "./types";

// Empty string in production (BUILD_STATIC export, served same-origin by FastAPI --
// see ARCHITECTURE.md #2). In dev, NEXT_PUBLIC_API_BASE (set in .env.development.local) points
// straight at the backend, bypassing Next's rewrite proxy -- it can't reliably
// carry /api/predict's SSE stream through. See ARCHITECTURE.md #14.
const API_BASE = process.env.NEXT_PUBLIC_API_BASE ?? "";

async function parseJsonOrThrow<T>(res: Response): Promise<T> {
  const body = await res.json().catch(() => null);
  if (!res.ok) {
    const message =
      body && typeof body === "object" && "error" in body
        ? String((body as { error: unknown }).error)
        : "Request failed.";
    throw new Error(message);
  }
  return body as T;
}

export async function resolveMolecule(
  inputType: "smiles" | "formula",
  value: string
): Promise<ResolveResponse> {
  const res = await fetch(`${API_BASE}/api/resolve`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ input_type: inputType, value }),
  });
  return parseJsonOrThrow<ResolveResponse>(res);
}

export async function getProfile(smiles: string): Promise<ProfileResponse> {
  const res = await fetch(`${API_BASE}/api/profile?smiles=${encodeURIComponent(smiles)}`);
  return parseJsonOrThrow<ProfileResponse>(res);
}

export type PredictEvent =
  | { type: "result"; data: PredictResult }
  | { type: "error"; data: PredictErrorEvent }
  | { type: "done" };

/**
 * Consumes /api/predict's SSE stream. Uses a manual fetch + ReadableStream parser
 * rather than the browser's EventSource, since EventSource can't send a POST body
 * (we need to send the molecule + selected datasets) -- see ARCHITECTURE.md #3.
 */
export async function* streamPredictions(
  smiles: string,
  datasets: string[]
): AsyncGenerator<PredictEvent> {
  const res = await fetch(`${API_BASE}/api/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ smiles, datasets }),
  });
  if (!res.ok || !res.body) {
    throw new Error("Could not start the prediction stream.");
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    // sse-starlette sends CRLF ("\r\n") line endings, not bare "\n" -- normalize to
    // LF so the \n\n boundary/line-splitting logic below stays correct either way.
    buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, "\n").replace(/\r/g, "\n");

    let boundary = buffer.indexOf("\n\n");
    while (boundary !== -1) {
      const rawEvent = buffer.slice(0, boundary);
      buffer = buffer.slice(boundary + 2);
      boundary = buffer.indexOf("\n\n");

      let eventType = "message";
      const dataLines: string[] = [];
      for (const line of rawEvent.split("\n")) {
        if (line.startsWith("event:")) eventType = line.slice(6).trim();
        else if (line.startsWith("data:")) dataLines.push(line.slice(5).trim());
      }
      const dataStr = dataLines.join("\n");
      if (!dataStr) continue;

      if (eventType === "result") {
        yield { type: "result", data: JSON.parse(dataStr) as PredictResult };
      } else if (eventType === "error") {
        yield { type: "error", data: JSON.parse(dataStr) as PredictErrorEvent };
      } else if (eventType === "done") {
        yield { type: "done" };
      }
    }
  }
}
