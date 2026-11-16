"use client";

import {
  AlertCircle,
  AlertTriangle,
  History,
  MessageSquarePlus,
  RotateCcw,
  Send,
  Square,
} from "lucide-react";
import { useRef, useState } from "react";
import {
  AgentExecutionStepper,
  type DoneSummaryInfo,
} from "@/components/AgentExecutionStepper";
import type {
  ChatSessionRead,
  CitationItem,
  CollectionRead,
  ErrorEventPayload,
  InferenceMode,
  NodeStatusEventPayload,
} from "@/types/api";

export interface UIMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  citations: CitationItem[];
  agentTrace: NodeStatusEventPayload[];
  doneSummary: DoneSummaryInfo | null;
  refusalTriggered?: boolean;
  stoppedByUser?: boolean;
  errorBanner?: ErrorEventPayload | null;
}

interface ChatWorkspaceProps {
  activeCollection: CollectionRead | null;
  inferenceMode: InferenceMode;
  sessions: ChatSessionRead[];
  activeSessionId: string | null;
  messages: UIMessage[];
  isStreaming: boolean;
  streamingNodes: NodeStatusEventPayload[];
  onSendQuestion: (question: string) => void;
  onStopStreaming: () => void;
  onSelectCitation: (citation: CitationItem) => void;
  onNewThread: () => void;
  onSelectSession: (sessionId: string) => void;
}

const STARTER_PROMPTS_BY_COLLECTION: Record<string, string[]> = {
  it_infra_sops: [
    "What are the RPO and RTO targets for Tier-1 Critical workloads?",
    "What does error code ERR_NVME_0x04 indicate and how should engineers resolve it?",
    "Explain STONITH quorum fencing and post-failover /ready verification steps.",
  ],
  malaysian_enterprise_policy: [
    "Apa prosedur tuntutan OT kalau kerja public holiday?",
    "Which inference mode is permitted for CONFIDENTIAL NRIC/salary records under PDPA?",
    "Berapakah elaun makan, hotel, dan perbatuan mileage bagi perjalanan luar kawasan?",
  ],
};

const DEFAULT_STARTER_PROMPTS = [
  "Summarize the primary rules and key thresholds stored in this collection.",
  "Compare two sections across the uploaded documents with [doc_id:pX] citations.",
  "Test GracefulRefusal: What is the root BGP TACACS+ password?",
];

function renderAnswerWithCitations(
  text: string,
  citations: CitationItem[],
  onSelectCitation: (citation: CitationItem) => void,
) {
  if (!text) {
    return null;
  }
  const byKey = new Map<string, CitationItem>();
  for (const item of citations) {
    byKey.set(item.citation_key, item);
  }

  const parts = text.split(/(\[[A-Za-z0-9_.\-]+:p\d+\])/g);
  return parts.map((part, idx) => {
    const matched = byKey.get(part);
    if (matched) {
      return (
        <button
          key={`${part}-${idx}`}
          type="button"
          onClick={() => onSelectCitation(matched)}
          className="citation-pill mx-0.5 inline-flex items-center rounded px-1.5 py-0.5 font-mono text-[11px] font-medium transition hover:brightness-125 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-teal-400"
        >
          {part}
        </button>
      );
    }
    return <span key={idx}>{part}</span>;
  });
}

export function ChatWorkspace({
  activeCollection,
  inferenceMode,
  sessions,
  activeSessionId,
  messages,
  isStreaming,
  streamingNodes,
  onSendQuestion,
  onStopStreaming,
  onSelectCitation,
  onNewThread,
  onSelectSession,
}: ChatWorkspaceProps) {
  const [draft, setDraft] = useState("");
  const lastSubmittedRef = useRef<string>("");

  const starters =
    (activeCollection && STARTER_PROMPTS_BY_COLLECTION[activeCollection.id]) ||
    DEFAULT_STARTER_PROMPTS;

  const submitPrompt = (text: string) => {
    const trimmed = text.trim();
    if (!trimmed || !activeCollection || isStreaming) {
      return;
    }
    lastSubmittedRef.current = trimmed;
    setDraft("");
    onSendQuestion(trimmed);
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      submitPrompt(draft);
    }
  };

  return (
    <section className="flex min-w-0 flex-1 flex-col bg-canvas">
      {/* Workspace Thread Sub-bar */}
      <div className="flex flex-wrap items-center justify-between gap-2 border-b border-borderSubtle bg-surface/50 px-4 py-2">
        <div className="flex min-w-0 items-center gap-2">
          <span className="truncate text-xs font-semibold text-textPrimary">
            {activeCollection ? activeCollection.name : "Select or Seed a Collection"}
          </span>
          {activeCollection && (
            <span className="hidden rounded bg-elevated px-2 py-0.5 font-mono text-[10px] text-textSecondary sm:inline">
              scope: {activeCollection.id}
            </span>
          )}
        </div>

        <div className="flex items-center gap-2">
          {sessions.length > 0 && (
            <div className="flex items-center gap-1">
              <History className="h-3.5 w-3.5 text-textSecondary" aria-hidden="true" />
              <select
                aria-label="Conversation history"
                value={activeSessionId ?? ""}
                onChange={(e) => {
                  if (e.target.value) {
                    onSelectSession(e.target.value);
                  }
                }}
                className="max-w-[180px] rounded border border-borderSubtle bg-surface px-2 py-1 text-[11px] text-textPrimary focus:border-sovereign focus:outline-none"
              >
                <option value="" disabled>
                  Recent threads ({sessions.length})
                </option>
                {sessions.map((s) => (
                  <option key={s.session_id} value={s.session_id}>
                    {s.title}
                  </option>
                ))}
              </select>
            </div>
          )}

          <button
            type="button"
            onClick={onNewThread}
            className="inline-flex items-center gap-1 rounded-md border border-borderSubtle bg-surface px-2.5 py-1 text-[11px] font-medium text-textSecondary hover:border-sovereign/50 hover:text-textPrimary focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sovereign"
          >
            <MessageSquarePlus className="h-3.5 w-3.5 text-sovereign" aria-hidden="true" />
            <span>New Thread</span>
          </button>
        </div>
      </div>

      {/* Chat Canvas */}
      <div className="flex-1 overflow-y-auto px-4 py-5">
        {messages.length === 0 ? (
          <div className="mx-auto flex max-w-2xl flex-col items-center justify-center py-10 text-center">
            <span className="rounded-full border border-sovereign/30 bg-sovereign/10 px-3 py-1 font-mono text-[11px] text-sovereign">
              Bounded LangGraph (max_retries=2) · Qdrant Hybrid RRF
            </span>
            <h1 className="mt-4 text-lg font-semibold text-textPrimary">
              Ask verified questions scoped to{" "}
              <span className="text-sovereign">
                {activeCollection ? activeCollection.name : "your workspace"}
              </span>
            </h1>
            <p className="mt-1.5 text-xs text-textSecondary">
              Every factual statement is grounded in hydrated ParentChunks with clickable{" "}
              <span className="font-mono text-teal-300">[doc_id:pX]</span> evidence or a{" "}
              <span className="font-mono text-warning">GracefulRefusal</span>.
            </p>

            <div className="mt-6 grid w-full gap-2.5 text-left">
              {starters.map((prompt) => (
                <button
                  key={prompt}
                  type="button"
                  disabled={!activeCollection || isStreaming}
                  onClick={() => submitPrompt(prompt)}
                  className="rounded-lg border border-borderSubtle bg-surface/80 px-3.5 py-2.5 text-xs text-textPrimary transition hover:border-sovereign/60 hover:bg-elevated focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sovereign disabled:opacity-50"
                >
                  {prompt}
                </button>
              ))}
            </div>
          </div>
        ) : (
          <div className="mx-auto max-w-3xl space-y-5">
            {messages.map((msg, idx) => {
              const isLastAssistant =
                msg.role === "assistant" && idx === messages.length - 1;
              const nodesToShow =
                isLastAssistant && isStreaming ? streamingNodes : msg.agentTrace;

              if (msg.role === "user") {
                return (
                  <div key={msg.id} className="flex justify-end">
                    <div className="max-w-[82%] rounded-xl border border-borderSubtle bg-elevated px-4 py-2.5 text-sm text-textPrimary">
                      {msg.content}
                    </div>
                  </div>
                );
              }

              return (
                <div key={msg.id} className="space-y-2">
                  <AgentExecutionStepper
                    nodes={nodesToShow}
                    isStreaming={isLastAssistant && isStreaming}
                    doneSummary={msg.doneSummary}
                  />

                  <div className="rounded-xl border border-borderSubtle bg-surface px-4 py-3.5">
                    <div className="text-sm leading-relaxed text-textPrimary whitespace-pre-wrap">
                      {msg.content ? (
                        renderAnswerWithCitations(
                          msg.content,
                          msg.citations,
                          onSelectCitation,
                        )
                      ) : isLastAssistant && isStreaming ? (
                        <span className="font-mono text-xs text-textSecondary">
                          Waiting for grounded tokens...
                        </span>
                      ) : null}
                    </div>

                    {msg.stoppedByUser && (
                      <p className="mt-2 font-mono text-[11px] text-warning">
                        [Stopped by user]
                      </p>
                    )}

                    {/* Graceful Refusal Info Callout */}
                    {msg.refusalTriggered && msg.citations.length === 0 && (
                      <div className="mt-3 flex items-start gap-2 rounded-lg border border-warning/40 bg-warning/10 px-3 py-2 text-xs text-amber-200">
                        <AlertTriangle
                          className="mt-0.5 h-4 w-4 shrink-0 text-warning"
                          aria-hidden="true"
                        />
                        <span>
                          No matching evidence found after 2 retrieval attempts — answer
                          intentionally withheld to prevent hallucination.
                        </span>
                      </div>
                    )}

                    {/* Mid-stream SSE Error Banner */}
                    {msg.errorBanner && (
                      <div
                        role="alert"
                        className="mt-3 flex flex-wrap items-center justify-between gap-2 rounded-lg border border-danger/50 bg-danger/10 px-3 py-2 text-xs text-rose-200"
                      >
                        <div className="flex items-center gap-2">
                          <AlertCircle
                            className="h-4 w-4 shrink-0 text-danger"
                            aria-hidden="true"
                          />
                          <span>
                            <strong className="font-mono">{msg.errorBanner.code}:</strong>{" "}
                            {msg.errorBanner.message}
                          </span>
                        </div>
                        {lastSubmittedRef.current && (
                          <button
                            type="button"
                            onClick={() => submitPrompt(lastSubmittedRef.current)}
                            className="inline-flex items-center gap-1 rounded border border-danger/40 bg-surface px-2 py-1 font-mono text-[11px] text-textPrimary hover:bg-elevated"
                          >
                            <RotateCcw className="h-3 w-3" aria-hidden="true" />
                            Retry
                          </button>
                        )}
                      </div>
                    )}

                    {/* Citation Summary Cards Strip */}
                    {msg.citations.length > 0 && (
                      <div className="mt-3 border-t border-borderSubtle pt-2.5">
                        <p className="mb-1.5 font-mono text-[10px] uppercase tracking-wider text-textSecondary">
                          Verified Sources ({msg.citations.length})
                        </p>
                        <div className="flex flex-wrap gap-1.5">
                          {msg.citations.map((cit) => (
                            <button
                              key={cit.citation_key}
                              type="button"
                              onClick={() => onSelectCitation(cit)}
                              className="citation-pill inline-flex items-center gap-1.5 rounded-md px-2 py-1 font-mono text-[11px] transition hover:brightness-125 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-teal-400"
                            >
                              <span>{cit.citation_key}</span>
                              <span className="text-[10px] text-textSecondary">
                                {cit.score.toFixed(2)}
                              </span>
                            </button>
                          ))}
                        </div>
                      </div>
                    )}

                    {/* Done Telemetry */}
                    {msg.doneSummary && (
                      <div className="mt-2.5 flex flex-wrap items-center gap-3 font-mono text-[10px] text-textSecondary">
                        <span
                          className={
                            msg.doneSummary.mode === "LOCAL_ONPREM"
                              ? "text-sovereign"
                              : "text-indigo-300"
                          }
                        >
                          mode: {msg.doneSummary.mode}
                        </span>
                        <span>latency: {msg.doneSummary.totalLatencyMs}ms</span>
                        {typeof msg.doneSummary.cacheHitTokens === "number" &&
                          msg.doneSummary.cacheHitTokens > 0 && (
                            <span className="text-indigo-300">
                              cache_hit_tokens: {msg.doneSummary.cacheHitTokens}
                            </span>
                          )}
                      </div>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>

      {/* Prompt Input Composer */}
      <div className="border-t border-borderSubtle bg-surface/70 px-4 py-3">
        <div className="mx-auto flex max-w-3xl items-end gap-2">
          <textarea
            rows={2}
            value={draft}
            disabled={!activeCollection}
            onChange={(e) => setDraft(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={
              activeCollection
                ? `Ask in EN / BM / BM-EN (${inferenceMode})...`
                : "Select or seed a collection to begin..."
            }
            className="min-h-[46px] flex-1 resize-none rounded-lg border border-borderSubtle bg-canvas px-3 py-2 text-sm text-textPrimary placeholder:text-textSecondary focus:border-sovereign focus:outline-none focus-visible:ring-2 focus-visible:ring-sovereign/40 disabled:opacity-50"
          />

          {isStreaming ? (
            <button
              type="button"
              onClick={onStopStreaming}
              aria-label="Stop generating"
              className="inline-flex h-[46px] items-center gap-1.5 rounded-lg bg-danger px-4 text-xs font-semibold text-white transition hover:bg-rose-500 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-rose-400"
            >
              <Square className="h-3.5 w-3.5 fill-current" aria-hidden="true" />
              <span>Stop</span>
            </button>
          ) : (
            <button
              type="button"
              disabled={!activeCollection || !draft.trim()}
              onClick={() => submitPrompt(draft)}
              className="inline-flex h-[46px] items-center gap-1.5 rounded-lg bg-sovereign px-4 text-xs font-semibold text-slate-950 transition hover:bg-emerald-400 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sovereign disabled:cursor-not-allowed disabled:opacity-50"
            >
              <Send className="h-3.5 w-3.5" aria-hidden="true" />
              <span>Send</span>
            </button>
          )}
        </div>
      </div>
    </section>
  );
}
