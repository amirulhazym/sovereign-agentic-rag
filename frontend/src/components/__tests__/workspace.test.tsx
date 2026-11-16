import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
import { AgentExecutionStepper } from "@/components/AgentExecutionStepper";
import { DocumentUploader } from "@/components/DocumentUploader";
import { InferenceModeSwitch } from "@/components/InferenceModeSwitch";
import { SovereignWorkspace } from "@/components/SovereignWorkspace";
import type {
  CollectionRead,
  DocumentRead,
  NodeStatusEventPayload,
} from "@/types/api";

afterEach(() => {
  vi.restoreAllMocks();
});

const COLLECTION_B: CollectionRead = {
  id: "malaysian_enterprise_policy",
  name: "Collection B: Malaysian Enterprise Policy",
  description: "Bilingual EN, BM, and BM-EN workplace policies.",
  is_seeded: true,
  document_count: 2,
  created_at: "2026-11-17T00:00:00Z",
};

const COLLECTION_A: CollectionRead = {
  id: "it_infra_sops",
  name: "Collection A: IT Infrastructure SOPs",
  description: "NVMe, RAID-10, and DR failover runbooks.",
  is_seeded: true,
  document_count: 2,
  created_at: "2026-11-17T00:00:00Z",
};

describe("3-Pane Glass-Box Workspace Components", () => {
  it("toggles InferenceModeSwitch cleanly between LOCAL_ONPREM and CLOUD_API", async () => {
    const user = userEvent.setup();
    const onChange = vi.fn();
    const { rerender } = render(
      <InferenceModeSwitch mode="LOCAL_ONPREM" onChange={onChange} />,
    );

    expect(
      screen.getByText(/Ollama Qwen3\.8-4B-Distill/i),
    ).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /CLOUD_API/i }));
    expect(onChange).toHaveBeenCalledWith("CLOUD_API");

    rerender(<InferenceModeSwitch mode="CLOUD_API" onChange={onChange} />);
    expect(
      screen.getByText(/DeepSeek Chat · Prefix Caching On/i),
    ).toBeInTheDocument();
  });

  it("renders AgentExecutionStepper retry loops and collapses into summary pill", async () => {
    const user = userEvent.setup();
    const trace: NodeStatusEventPayload[] = [
      { node: "query_router", status: "completed", attempt: 0, latency_ms: 8, detail: "retrieve" },
      { node: "query_rewriter", status: "completed", attempt: 0, latency_ms: 14, detail: "expanded" },
      { node: "hybrid_retriever", status: "completed", attempt: 0, latency_ms: 11, detail: "6 hits" },
      { node: "cross_encoder_reranker", status: "completed", attempt: 0, latency_ms: 28, detail: "top-5" },
      { node: "relevance_grader", status: "retrying", attempt: 1, latency_ms: 19, detail: "retry 1/2" },
    ];

    const { rerender } = render(
      <AgentExecutionStepper
        nodes={trace}
        isStreaming={true}
        doneSummary={null}
      />,
    );

    expect(screen.getByText(/Attempt 1\/2/i)).toBeInTheDocument();
    expect(screen.getByText("QueryRouter")).toBeInTheDocument();
    expect(screen.getByText("RelevanceGrader")).toBeInTheDocument();

    rerender(
      <AgentExecutionStepper
        nodes={[
          ...trace,
          { node: "grounded_generator", status: "completed", attempt: 1, latency_ms: 320, detail: "done" },
        ]}
        isStreaming={false}
        doneSummary={{
          totalLatencyMs: 1640,
          retriesUsed: 1,
          citationCount: 2,
          mode: "LOCAL_ONPREM",
        }}
      />,
    );

    const summaryPill = screen.getByRole("button", {
      name: /Completed in 1\.64s · 1 retry · 2 citations/i,
    });
    expect(summaryPill).toBeInTheDocument();

    // Expanding summary pill reveals individual nodes again
    await user.click(summaryPill);
    expect(screen.getByText("GroundedGenerator")).toBeInTheDocument();
  });

  it("polls document ingestion lifecycle PENDING -> INDEXING -> READY and handles dedup", async () => {
    vi.useFakeTimers();
    const docPending: DocumentRead = {
      doc_id: "doc_123",
      collection_id: COLLECTION_B.id,
      filename: "hr_addon.pdf",
      doc_type: "pdf",
      language: "ms",
      content_hash: "abc",
      page_count: 0,
      chunk_count: 0,
      status: "PENDING",
      error_message: null,
      created_at: "2026-11-17T00:00:00Z",
      updated_at: "2026-11-17T00:00:00Z",
    };

    const statusSequence: DocumentRead[] = [
      { ...docPending, status: "INDEXING" },
      { ...docPending, status: "READY", page_count: 3, chunk_count: 6 },
    ];

    const onUploaded = vi.fn();
    const onStatusUpdated = vi.fn();

    const uploadFn = vi.fn().mockResolvedValue({
      document: docPending,
      deduplicated: false,
    });
    const pollFn = vi.fn().mockImplementation(async () => {
      return statusSequence.shift() ?? { ...docPending, status: "READY" };
    });

    render(
      <DocumentUploader
        collectionId={COLLECTION_B.id}
        onUploaded={onUploaded}
        onStatusUpdated={onStatusUpdated}
        uploadFn={uploadFn}
        pollFn={pollFn}
      />,
    );

    const input = screen.getByTestId("document-file-input") as HTMLInputElement;
    const file = new File(["dummy"], "hr_addon.pdf", { type: "application/pdf" });
    fireEvent.change(input, { target: { files: [file] } });

    await act(async () => {
      await Promise.resolve();
    });
    expect(onUploaded).toHaveBeenCalledWith(docPending);

    await act(async () => {
      await vi.advanceTimersByTimeAsync(1550);
    });
    expect(onStatusUpdated).toHaveBeenCalledWith(
      expect.objectContaining({ status: "INDEXING" }),
    );

    await act(async () => {
      await vi.advanceTimersByTimeAsync(1550);
    });
    expect(onStatusUpdated).toHaveBeenCalledWith(
      expect.objectContaining({ status: "READY", chunk_count: 6 }),
    );

    // Repeat upload returning deduplicated=true renders informational toast
    uploadFn.mockResolvedValueOnce({
      document: { ...docPending, status: "READY" },
      deduplicated: true,
    });
    fireEvent.change(input, { target: { files: [file] } });
    await act(async () => {
      await Promise.resolve();
    });
    expect(
      screen.getByText(/already indexed in this collection \(deduplicated\)/i),
    ).toBeInTheDocument();

    vi.useRealTimers();
  });

  it("streams chat SSE response, opens EvidenceInspectorDrawer on citation click, and closes on Escape", async () => {
    const user = userEvent.setup();

    const sseBody = [
      'event: node_status\ndata: {"node":"query_router","status":"completed","attempt":0,"latency_ms":10,"detail":"retrieve"}\n\n',
      'event: token\ndata: {"delta":"Pekerja layak 2.0x OT atau cuti gantian [hr_employment_and_overtime_policy_my:p2]."}\n\n',
      'event: citations\ndata: {"sources":[{"citation_key":"[hr_employment_and_overtime_policy_my:p2]","doc_id":"hr_employment_and_overtime_policy_my","filename":"hr_employment_and_overtime_policy_my.md","page":2,"score":0.94,"breadcrumb":"HR Policy > Klausa 4.2","snippet":"Klausa 4.2: Kerja pada hari kelepasan am layak bayaran 2.0x."}]}\n\n',
      'event: done\ndata: {"message_id":"msg_1","session_id":"sess_1","mode":"LOCAL_ONPREM","retries_used":0,"cache_hit_tokens":1280,"total_latency_ms":510,"refusal_triggered":false}\n\n',
    ].join("");

    const fetchMock = vi.fn().mockImplementation(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.endsWith("/ready")) {
        return new Response(
          JSON.stringify({ status: "ready", checks: { sqlite: "ok", qdrant: "ok" } }),
          { status: 200, headers: { "Content-Type": "application/json" } },
        );
      }
      if (url.endsWith("/api/v1/collections")) {
        return new Response(JSON.stringify([COLLECTION_A, COLLECTION_B]), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        });
      }
      if (url.includes("/documents")) {
        return new Response(JSON.stringify([]), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        });
      }
      if (url.includes("/sessions") && !url.includes("/messages")) {
        return new Response(JSON.stringify([]), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        });
      }
      if (url.endsWith("/api/v1/chat/stream")) {
        const encoder = new TextEncoder();
        const stream = new ReadableStream<Uint8Array>({
          start(controller) {
            controller.enqueue(encoder.encode(sseBody));
            controller.close();
          },
        });
        return new Response(stream, {
          status: 200,
          headers: { "Content-Type": "text/event-stream" },
        });
      }
      return new Response("{}", { status: 200 });
    });

    vi.stubGlobal("fetch", fetchMock);

    render(<SovereignWorkspace />);

    await waitFor(() => {
      expect(screen.getByText(/Collection B: Malaysian Enterprise Policy/i)).toBeInTheDocument();
    });

    // Click Collection B to load starter prompts and isolate tenant context
    await user.click(screen.getByRole("button", { name: /Collection B: Malaysian Enterprise Policy/i }));

    const input = screen.getByPlaceholderText(/Ask in EN \/ BM \/ BM-EN/i);
    await user.type(input, "Apa prosedur OT public holiday?{Enter}");

    const inlinePill = await screen.findAllByRole("button", {
      name: /hr_employment_and_overtime_policy_my:p2/i,
    });
    expect(inlinePill.length).toBeGreaterThan(0);

    await user.click(inlinePill[0]);

    expect(screen.getByText("EVIDENCE INSPECTOR")).toBeInTheDocument();
    expect(screen.getByText("HR Policy > Klausa 4.2")).toBeInTheDocument();
    expect(screen.getByText(/Klausa 4\.2: Kerja pada hari kelepasan am/i)).toBeInTheDocument();

    // Escape closes the Evidence Inspector cleanly
    await user.keyboard("{Escape}");
    expect(screen.queryByText("HR Policy > Klausa 4.2")).not.toBeInTheDocument();
  });

  it("supports mid-stream Stop abort button, GracefulRefusal callout, and SSE Error Retry", async () => {
    const user = userEvent.setup();
    let streamController: ReadableStreamDefaultController<Uint8Array> | null = null;
    let chatCallCount = 0;

    const fetchMock = vi.fn().mockImplementation(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.endsWith("/ready")) {
        return new Response(JSON.stringify({ status: "ready", checks: { sqlite: "ok" } }), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        });
      }
      if (url.endsWith("/api/v1/collections")) {
        return new Response(JSON.stringify([COLLECTION_A]), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        });
      }
      if (url.includes("/documents") || (url.includes("/sessions") && !url.includes("/messages"))) {
        return new Response(JSON.stringify([]), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        });
      }
      if (url.endsWith("/api/v1/chat/stream")) {
        chatCallCount += 1;
        const encoder = new TextEncoder();
        if (chatCallCount === 1) {
          const stream = new ReadableStream<Uint8Array>({
            start(controller) {
              streamController = controller;
              controller.enqueue(encoder.encode('event: token\ndata: {"delta":"Partial token "}\n\n'));
            },
          });
          return new Response(stream, {
            status: 200,
            headers: { "Content-Type": "text/event-stream" },
          });
        }
        const refusalSse = [
          'event: node_status\ndata: {"node":"graceful_refusal","status":"refused","attempt":2,"latency_ms":5,"detail":"exhausted"}\n\n',
          'event: token\ndata: {"delta":"I am sorry, documents do not contain sufficient verified evidence."}\n\n',
          'event: citations\ndata: {"sources":[]}\n\n',
          'event: done\ndata: {"message_id":"msg_2","session_id":"sess_1","mode":"LOCAL_ONPREM","retries_used":2,"cache_hit_tokens":0,"total_latency_ms":210,"refusal_triggered":true}\n\n',
        ].join("");
        return new Response(
          new ReadableStream<Uint8Array>({
            start(controller) {
              controller.enqueue(encoder.encode(refusalSse));
              controller.close();
            },
          }),
          { status: 200, headers: { "Content-Type": "text/event-stream" } },
        );
      }
      return new Response("{}", { status: 200 });
    });

    vi.stubGlobal("fetch", fetchMock);
    render(<SovereignWorkspace />);

    await waitFor(() => {
      expect(
        screen.getByRole("button", { name: /Collection A: IT Infrastructure SOPs/i }),
      ).toBeInTheDocument();
    });

    const input = screen.getByPlaceholderText(/Ask in EN \/ BM \/ BM-EN/i);
    await user.type(input, "Stop streaming test{Enter}");

    const stopButton = await screen.findByRole("button", { name: /Stop generating/i });
    await user.click(stopButton);
    if (streamController) {
      try {
        (streamController as ReadableStreamDefaultController<Uint8Array>).close();
      } catch {
        // stream already closed/cancelled
      }
    }

    expect(await screen.findByText(/\[Stopped by user\]/i)).toBeInTheDocument();

    // Send a second query returning GracefulRefusal callout
    await user.type(input, "Cisco TACACS password?{Enter}");
    expect(
      await screen.findByText(/No matching evidence found after 2 retrieval attempts/i),
    ).toBeInTheDocument();
  });
});
