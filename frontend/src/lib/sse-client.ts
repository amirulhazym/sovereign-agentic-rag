import {
  DEFAULT_API_BASE_URL,
  parseErrorResponse,
} from "@/lib/api-client";
import type {
  ChatStreamRequest,
  CitationsEventPayload,
  DoneEventPayload,
  ErrorEventPayload,
  NodeStatusEventPayload,
  SSEEvent,
  TokenEventPayload,
} from "@/types/api";

export interface StreamChatOptions {
  payload: ChatStreamRequest;
  onEvent: (event: SSEEvent) => void;
  signal?: AbortSignal;
  baseUrl?: string;
  fetchFn?: typeof fetch;
}

export function parseSseFrame(block: string): SSEEvent | null {
  let eventName = "";
  const dataLines: string[] = [];

  for (const rawLine of block.split("\n")) {
    if (!rawLine || rawLine.startsWith(":")) {
      continue;
    }
    if (rawLine.startsWith("event:")) {
      eventName = rawLine.slice("event:".length).trim();
    } else if (rawLine.startsWith("data:")) {
      dataLines.push(rawLine.slice("data:".length).trimStart());
    }
  }

  if (!dataLines.length) {
    return null;
  }

  const parsedJson = JSON.parse(dataLines.join("\n")) as unknown;
  switch (eventName) {
    case "node_status":
      return { type: "node_status", payload: parsedJson as NodeStatusEventPayload };
    case "token":
      return { type: "token", payload: parsedJson as TokenEventPayload };
    case "citations":
      return { type: "citations", payload: parsedJson as CitationsEventPayload };
    case "done":
      return { type: "done", payload: parsedJson as DoneEventPayload };
    case "error":
      return { type: "error", payload: parsedJson as ErrorEventPayload };
    default:
      return null;
  }
}

export async function streamChatEvents({
  payload,
  onEvent,
  signal,
  baseUrl = DEFAULT_API_BASE_URL,
  fetchFn = fetch,
}: StreamChatOptions): Promise<void> {
  if (signal?.aborted) {
    return;
  }

  const response = await fetchFn(`${baseUrl.replace(/\/$/, "")}/api/v1/chat/stream`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Accept: "text/event-stream",
    },
    body: JSON.stringify(payload),
    signal,
  });

  if (!response.ok) {
    throw await parseErrorResponse(response);
  }
  if (!response.body) {
    throw new Error("Streaming response did not provide a readable body.");
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder("utf-8");
  let buffer = "";

  const flushCompleteFrames = () => {
    while (true) {
      const separatorIdx = buffer.indexOf("\n\n");
      if (separatorIdx === -1) {
        break;
      }
      const frame = buffer.slice(0, separatorIdx);
      buffer = buffer.slice(separatorIdx + 2);
      const parsed = parseSseFrame(frame);
      if (parsed) {
        onEvent(parsed);
      }
      if (signal?.aborted) {
        return;
      }
    }
  };

  try {
    while (true) {
      if (signal?.aborted) {
        await reader.cancel();
        return;
      }
      const { value, done } = await reader.read();
      if (done) {
        buffer += decoder.decode().replace(/\r\n/g, "\n");
        flushCompleteFrames();
        if (!signal?.aborted && buffer.trim()) {
          const trailing = parseSseFrame(buffer);
          if (trailing) {
            onEvent(trailing);
          }
        }
        break;
      }
      buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, "\n");
      flushCompleteFrames();
      if (signal?.aborted) {
        await reader.cancel();
        return;
      }
    }
  } catch (err) {
    if (err instanceof DOMException && err.name === "AbortError") {
      return;
    }
    if (signal?.aborted) {
      return;
    }
    throw err;
  } finally {
    reader.releaseLock();
  }
}
