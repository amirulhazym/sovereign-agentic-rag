import { describe, expect, it, vi } from "vitest";
import { streamChatEvents } from "@/lib/sse-client";
import type { ChatStreamRequest, SSEEvent } from "@/types/api";

function makeChunkedStream(chunks: string[]): ReadableStream<Uint8Array> {
  const encoder = new TextEncoder();
  return new ReadableStream<Uint8Array>({
    start(controller) {
      for (const chunk of chunks) {
        controller.enqueue(encoder.encode(chunk));
      }
      controller.close();
    },
  });
}

describe("streamChatEvents (typed SSE ReadableStream client)", () => {
  it("parses chunked SSE frames split across arbitrary byte boundaries", async () => {
    const rawWire = [
      'event: node_status\ndata: {"node":"query_router","status":"completed","attempt":0,"latency_ms":11,"detail":"retrieve"}\n\n',
      'event: token\ndata: {"delta":"Klausa 4.2 "}\n\n',
      'event: token\ndata: {"delta":"membolehkan 2.0x OT [hr_employment_and_overtime_policy_my:p2]."}\n\n',
      'event: citations\ndata: {"sources":[{"citation_key":"[hr_employment_and_overtime_policy_my:p2]","doc_id":"hr_employment_and_overtime_policy_my","filename":"hr_employment_and_overtime_policy_my.md","page":2,"score":0.93,"breadcrumb":"HR > Klausa 4.2","snippet":"Bayaran 2.0x gaji sejam atau cuti gantian."}]}\n\n',
      'event: done\ndata: {"message_id":"msg_1","session_id":"sess_1","mode":"LOCAL_ONPREM","retries_used":0,"cache_hit_tokens":1024,"total_latency_ms":290,"refusal_triggered":false}\n\n',
    ].join("");

    // Intentionally slice mid-event and mid-JSON key boundaries
    const slices: string[] = [];
    for (let i = 0; i < rawWire.length; i += 17) {
      slices.push(rawWire.slice(i, i + 17));
    }

    const fetchMock = vi.fn().mockResolvedValue(
      new Response(makeChunkedStream(slices), {
        status: 200,
        headers: { "Content-Type": "text/event-stream; charset=utf-8" },
      }),
    );

    const events: SSEEvent[] = [];
    const payload: ChatStreamRequest = {
      collection_id: "malaysian_enterprise_policy",
      question: "Apa prosedur tuntutan OT kalau kerja public holiday?",
      inference_mode: "LOCAL_ONPREM",
    };

    await streamChatEvents({
      payload,
      onEvent: (evt) => events.push(evt),
      fetchFn: fetchMock,
      baseUrl: "http://localhost:8000",
    });

    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(events.map((e) => e.type)).toEqual([
      "node_status",
      "token",
      "token",
      "citations",
      "done",
    ]);

    const doneEvent = events[4];
    expect(doneEvent.type).toBe("done");
    if (doneEvent.type === "done") {
      expect(doneEvent.payload.cache_hit_tokens).toBe(1024);
      expect(doneEvent.payload.session_id).toBe("sess_1");
    }
  });

  it("emits SSE error frames cleanly without losing recoverable metadata", async () => {
    const wire =
      'event: error\ndata: {"code":"PROVIDER_UNAVAILABLE","message":"Ollama connection refused","recoverable":true}\n\n';
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(makeChunkedStream([wire]), {
        status: 200,
        headers: { "Content-Type": "text/event-stream" },
      }),
    );

    const events: SSEEvent[] = [];
    await streamChatEvents({
      payload: { collection_id: "it_infra_sops", question: "RPO tier 1?" },
      onEvent: (evt) => events.push(evt),
      fetchFn: fetchMock,
      baseUrl: "http://localhost:8000",
    });

    expect(events).toHaveLength(1);
    expect(events[0]).toEqual({
      type: "error",
      payload: {
        code: "PROVIDER_UNAVAILABLE",
        message: "Ollama connection refused",
        recoverable: true,
      },
    });
  });

  it("propagates AbortController cancellation mid-stream cleanly", async () => {
    const controller = new AbortController();
    const encoder = new TextEncoder();

    const cancellableStream = new ReadableStream<Uint8Array>({
      start(streamCtrl) {
        streamCtrl.enqueue(
          encoder.encode('event: token\ndata: {"delta":"Partial "}\n\n'),
        );
      },
    });

    const fetchMock = vi.fn().mockResolvedValue(
      new Response(cancellableStream, {
        status: 200,
        headers: { "Content-Type": "text/event-stream" },
      }),
    );

    const events: SSEEvent[] = [];
    const runPromise = streamChatEvents({
      payload: { collection_id: "it_infra_sops", question: "Explain failover" },
      onEvent: (evt) => {
        events.push(evt);
        controller.abort();
      },
      signal: controller.signal,
      fetchFn: fetchMock,
      baseUrl: "http://localhost:8000",
    });

    await expect(runPromise).resolves.toBeUndefined();
    expect(events).toHaveLength(1);
    expect(events[0].type).toBe("token");
  });

  it("throws typed ApiClientError on non-200 HTTP error envelopes", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(
        JSON.stringify({
          error: {
            code: "COLLECTION_NOT_FOUND",
            message: "Collection 'missing' does not exist.",
            detail: null,
            request_id: "req_123",
          },
        }),
        { status: 404, headers: { "Content-Type": "application/json" } },
      ),
    );

    await expect(
      streamChatEvents({
        payload: { collection_id: "missing", question: "Hello" },
        onEvent: vi.fn(),
        fetchFn: fetchMock,
        baseUrl: "http://localhost:8000",
      }),
    ).rejects.toThrow("COLLECTION_NOT_FOUND");
  });
});
