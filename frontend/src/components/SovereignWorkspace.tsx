"use client";

import { Activity, Menu, Shield, X } from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";
import { ChatWorkspace, type UIMessage } from "@/components/ChatWorkspace";
import { CollectionSidebar } from "@/components/CollectionSidebar";
import { EvidenceInspectorDrawer } from "@/components/EvidenceInspectorDrawer";
import { InferenceModeSwitch } from "@/components/InferenceModeSwitch";
import {
  createCollection,
  fetchCollectionDocuments,
  fetchCollections,
  fetchCollectionSessions,
  fetchReadyProbe,
  fetchSessionMessages,
  seedCollections,
} from "@/lib/api-client";
import { streamChatEvents } from "@/lib/sse-client";
import type {
  ChatSessionRead,
  CitationItem,
  CollectionCreate,
  CollectionRead,
  DocumentRead,
  InferenceMode,
  NodeStatusEventPayload,
  ReadinessResponse,
} from "@/types/api";

export function SovereignWorkspace() {
  const [collections, setCollections] = useState<CollectionRead[]>([]);
  const [activeCollectionId, setActiveCollectionId] = useState<string>("");
  const [documents, setDocuments] = useState<DocumentRead[]>([]);
  const [isLoadingDocs, setIsLoadingDocs] = useState(false);
  const [isSeeding, setIsSeeding] = useState(false);

  const [inferenceMode, setInferenceMode] = useState<InferenceMode>("LOCAL_ONPREM");
  const [readiness, setReadiness] = useState<ReadinessResponse | null>(null);
  const [totalCacheHits, setTotalCacheHits] = useState(0);

  const [sessions, setSessions] = useState<ChatSessionRead[]>([]);
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null);
  const [messages, setMessages] = useState<UIMessage[]>([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const [streamingNodes, setStreamingNodes] = useState<NodeStatusEventPayload[]>([]);
  const [selectedCitation, setSelectedCitation] = useState<CitationItem | null>(null);
  const [mobileSidebarOpen, setMobileSidebarOpen] = useState(false);

  const abortControllerRef = useRef<AbortController | null>(null);

  const loadWorkspaceBootstrap = useCallback(async () => {
    const [colsResult, readyResult] = await Promise.allSettled([
      fetchCollections(),
      fetchReadyProbe(),
    ]);
    if (readyResult.status === "fulfilled") {
      setReadiness(readyResult.value);
    }
    if (colsResult.status === "fulfilled") {
      const list = colsResult.value;
      setCollections(list);
      if (list.length > 0) {
        setActiveCollectionId((prev) =>
          prev && list.some((c) => c.id === prev) ? prev : list[0].id,
        );
      }
    }
  }, []);

  useEffect(() => {
    void loadWorkspaceBootstrap();
  }, [loadWorkspaceBootstrap]);

  const loadCollectionScopedData = useCallback(async (collectionId: string) => {
    if (!collectionId) {
      setDocuments([]);
      setSessions([]);
      return;
    }
    setIsLoadingDocs(true);
    try {
      const [docs, colSessions] = await Promise.all([
        fetchCollectionDocuments(collectionId),
        fetchCollectionSessions(collectionId),
      ]);
      setDocuments(docs);
      setSessions(colSessions);
    } catch {
      setDocuments([]);
      setSessions([]);
    } finally {
      setIsLoadingDocs(false);
    }
  }, []);

  useEffect(() => {
    void loadCollectionScopedData(activeCollectionId);
  }, [activeCollectionId, loadCollectionScopedData]);

  const handleSelectCollection = (collectionId: string) => {
    if (collectionId === activeCollectionId) {
      setMobileSidebarOpen(false);
      return;
    }
    abortControllerRef.current?.abort();
    setIsStreaming(false);
    setStreamingNodes([]);
    setActiveCollectionId(collectionId);
    setActiveSessionId(null);
    setMessages([]);
    setSelectedCitation(null);
    setMobileSidebarOpen(false);
  };

  const handleCreateCollection = async (payload: CollectionCreate) => {
    const created = await createCollection(payload);
    setCollections((prev) => [...prev, created]);
    handleSelectCollection(created.id);
  };

  const handleSeedCollections = async () => {
    setIsSeeding(true);
    try {
      await seedCollections();
      await loadWorkspaceBootstrap();
      if (activeCollectionId) {
        await loadCollectionScopedData(activeCollectionId);
      }
    } finally {
      setIsSeeding(false);
    }
  };

  const handleDocumentUploaded = (doc: DocumentRead) => {
    setDocuments((prev) => [doc, ...prev.filter((d) => d.doc_id !== doc.doc_id)]);
    setCollections((prev) =>
      prev.map((col) =>
        col.id === doc.collection_id
          ? { ...col, document_count: col.document_count + 1 }
          : col,
      ),
    );
  };

  const handleDocumentStatusUpdated = (updated: DocumentRead) => {
    setDocuments((prev) => {
      const exists = prev.some((d) => d.doc_id === updated.doc_id);
      if (!exists) {
        return [updated, ...prev];
      }
      return prev.map((d) => (d.doc_id === updated.doc_id ? updated : d));
    });
  };

  const handleSelectSession = async (sessionId: string) => {
    abortControllerRef.current?.abort();
    setIsStreaming(false);
    setStreamingNodes([]);
    setActiveSessionId(sessionId);
    setSelectedCitation(null);
    const history = await fetchSessionMessages(sessionId);
    setMessages(
      history.map((m) => ({
        id: m.message_id,
        role: m.role,
        content: m.content,
        citations: m.citations,
        agentTrace: m.agent_trace,
        doneSummary:
          m.role === "assistant"
            ? {
                totalLatencyMs: m.latency_ms,
                retriesUsed: m.agent_trace.filter((n) => n.status === "retrying").length,
                citationCount: m.citations.length,
                mode: m.inference_mode,
                cacheHitTokens: m.cache_hit_tokens,
              }
            : null,
      })),
    );
  };

  const handleNewThread = () => {
    abortControllerRef.current?.abort();
    setIsStreaming(false);
    setStreamingNodes([]);
    setActiveSessionId(null);
    setMessages([]);
    setSelectedCitation(null);
  };

  const handleStopStreaming = () => {
    abortControllerRef.current?.abort();
    setIsStreaming(false);
    setMessages((prev) => {
      const next = [...prev];
      const last = next[next.length - 1];
      if (last && last.role === "assistant") {
        next[next.length - 1] = { ...last, stoppedByUser: true };
      }
      return next;
    });
  };

  const handleSendQuestion = async (question: string) => {
    if (!activeCollectionId || isStreaming) {
      return;
    }

    const userMsg: UIMessage = {
      id: `user_${Date.now()}`,
      role: "user",
      content: question,
      citations: [],
      agentTrace: [],
      doneSummary: null,
    };
    const assistantMsgId = `asst_${Date.now()}`;
    const assistantPlaceholder: UIMessage = {
      id: assistantMsgId,
      role: "assistant",
      content: "",
      citations: [],
      agentTrace: [],
      doneSummary: null,
    };

    setMessages((prev) => [...prev, userMsg, assistantPlaceholder]);
    setIsStreaming(true);
    setStreamingNodes([]);

    const controller = new AbortController();
    abortControllerRef.current = controller;

    let liveNodes: NodeStatusEventPayload[] = [];
    let liveCitations: CitationItem[] = [];

    try {
      await streamChatEvents({
        payload: {
          collection_id: activeCollectionId,
          question,
          session_id: activeSessionId,
          inference_mode: inferenceMode,
        },
        signal: controller.signal,
        onEvent: (event) => {
          if (event.type === "node_status") {
            liveNodes = [...liveNodes, event.payload];
            setStreamingNodes(liveNodes);
          } else if (event.type === "token") {
            setMessages((prev) =>
              prev.map((m) =>
                m.id === assistantMsgId
                  ? { ...m, content: m.content + event.payload.delta }
                  : m,
              ),
            );
          } else if (event.type === "citations") {
            liveCitations = event.payload.sources;
            setMessages((prev) =>
              prev.map((m) =>
                m.id === assistantMsgId ? { ...m, citations: liveCitations } : m,
              ),
            );
          } else if (event.type === "done") {
            setActiveSessionId(event.payload.session_id);
            if (event.payload.cache_hit_tokens > 0) {
              setTotalCacheHits((c) => c + event.payload.cache_hit_tokens);
            }
            setMessages((prev) =>
              prev.map((m) =>
                m.id === assistantMsgId
                  ? {
                      ...m,
                      agentTrace: liveNodes,
                      refusalTriggered: event.payload.refusal_triggered,
                      doneSummary: {
                        totalLatencyMs: event.payload.total_latency_ms,
                        retriesUsed: event.payload.retries_used,
                        citationCount: liveCitations.length,
                        mode: event.payload.mode,
                        cacheHitTokens: event.payload.cache_hit_tokens,
                      },
                    }
                  : m,
              ),
            );
            void fetchCollectionSessions(activeCollectionId).then(setSessions).catch(() => {});
          } else if (event.type === "error") {
            setMessages((prev) =>
              prev.map((m) =>
                m.id === assistantMsgId
                  ? { ...m, agentTrace: liveNodes, errorBanner: event.payload }
                  : m,
              ),
            );
          }
        },
      });
    } catch (err) {
      if (!(err instanceof DOMException && err.name === "AbortError")) {
        setMessages((prev) =>
          prev.map((m) =>
            m.id === assistantMsgId
              ? {
                  ...m,
                  agentTrace: liveNodes,
                  errorBanner: {
                    code: "STREAM_REQUEST_FAILED",
                    message: err instanceof Error ? err.message : "Failed to reach backend SSE stream.",
                    recoverable: true,
                  },
                }
              : m,
          ),
        );
      }
    } finally {
      setIsStreaming(false);
    }
  };

  const activeCollection =
    collections.find((c) => c.id === activeCollectionId) ?? null;

  return (
    <div className="flex h-screen flex-col overflow-hidden bg-canvas text-textPrimary">
      {/* Top 56px Sovereign Header Bar */}
      <header className="flex h-14 shrink-0 items-center justify-between gap-3 border-b border-borderSubtle bg-surface px-4">
        <div className="flex items-center gap-2.5">
          <button
            type="button"
            aria-label="Toggle collections drawer"
            onClick={() => setMobileSidebarOpen((v) => !v)}
            className="rounded p-1.5 text-textSecondary hover:bg-elevated hover:text-textPrimary lg:hidden"
          >
            {mobileSidebarOpen ? (
              <X className="h-4 w-4" aria-hidden="true" />
            ) : (
              <Menu className="h-4 w-4" aria-hidden="true" />
            )}
          </button>
          <Shield className="h-5 w-5 text-sovereign" aria-hidden="true" />
          <span className="font-mono text-sm font-semibold tracking-tight text-textPrimary">
            sovereign-agentic-rag
          </span>
        </div>

        <div className="flex items-center gap-3">
          <div className="hidden items-center gap-2 sm:flex">
            <span
              className={`inline-flex items-center gap-1 rounded-full border px-2 py-0.5 font-mono text-[10px] ${
                readiness?.status === "ready"
                  ? "border-sovereign/40 bg-sovereign/10 text-sovereign"
                  : "border-warning/40 bg-warning/10 text-warning"
              }`}
            >
              <Activity className="h-3 w-3" aria-hidden="true" />
              {readiness?.status === "ready" ? "Ready" : "Checking"}
            </span>
            {totalCacheHits > 0 && (
              <span className="rounded-full border border-cloud/40 bg-cloud/10 px-2 py-0.5 font-mono text-[10px] text-indigo-300">
                Cache: {totalCacheHits} tok
              </span>
            )}
          </div>

          <InferenceModeSwitch
            mode={inferenceMode}
            onChange={setInferenceMode}
            disabled={isStreaming}
          />
        </div>
      </header>

      {/* Main 3-Pane Body */}
      <div className="relative flex min-h-0 flex-1 overflow-hidden">
        {/* Desktop Left Sidebar */}
        <div className="hidden lg:flex">
          <CollectionSidebar
            collections={collections}
            activeCollectionId={activeCollectionId}
            documents={documents}
            isLoadingDocs={isLoadingDocs}
            isSeeding={isSeeding}
            onSelectCollection={handleSelectCollection}
            onCreateCollection={handleCreateCollection}
            onSeedCollections={handleSeedCollections}
            onDocumentUploaded={handleDocumentUploaded}
            onDocumentStatusUpdated={handleDocumentStatusUpdated}
          />
        </div>

        {/* Mobile Slide-Over Sidebar */}
        {mobileSidebarOpen && (
          <div className="fixed inset-y-14 left-0 z-30 flex w-72 shadow-2xl lg:hidden">
            <CollectionSidebar
              collections={collections}
              activeCollectionId={activeCollectionId}
              documents={documents}
              isLoadingDocs={isLoadingDocs}
              isSeeding={isSeeding}
              onSelectCollection={handleSelectCollection}
              onCreateCollection={handleCreateCollection}
              onSeedCollections={handleSeedCollections}
              onDocumentUploaded={handleDocumentUploaded}
              onDocumentStatusUpdated={handleDocumentStatusUpdated}
            />
          </div>
        )}

        {/* Center Chat & Stepper Pane */}
        <ChatWorkspace
          activeCollection={activeCollection}
          inferenceMode={inferenceMode}
          sessions={sessions}
          activeSessionId={activeSessionId}
          messages={messages}
          isStreaming={isStreaming}
          streamingNodes={streamingNodes}
          onSendQuestion={(q) => void handleSendQuestion(q)}
          onStopStreaming={handleStopStreaming}
          onSelectCitation={setSelectedCitation}
          onNewThread={handleNewThread}
          onSelectSession={(id) => void handleSelectSession(id)}
        />

        {/* Right Evidence Inspector */}
        <EvidenceInspectorDrawer
          citation={selectedCitation}
          onClose={() => setSelectedCitation(null)}
        />
      </div>
    </div>
  );
}
