"use client";

import {
  AlertTriangle,
  CheckCircle2,
  ChevronDown,
  ChevronRight,
  Circle,
  Loader2,
  RefreshCw,
} from "lucide-react";
import { useState } from "react";
import type { InferenceMode, NodeStatusEventPayload } from "@/types/api";

export interface DoneSummaryInfo {
  totalLatencyMs: number;
  retriesUsed: number;
  citationCount: number;
  mode: InferenceMode;
  cacheHitTokens?: number;
}

interface AgentExecutionStepperProps {
  nodes: NodeStatusEventPayload[];
  isStreaming: boolean;
  doneSummary: DoneSummaryInfo | null;
}

const PIPELINE_NODES = [
  { id: "query_router", label: "QueryRouter" },
  { id: "query_rewriter", label: "QueryRewriter" },
  { id: "hybrid_retriever", label: "HybridRetriever" },
  { id: "cross_encoder_reranker", label: "CrossEncoderReranker" },
  { id: "relevance_grader", label: "RelevanceGrader" },
  { id: "grounded_generator", label: "GroundedGenerator" },
] as const;

export function AgentExecutionStepper({
  nodes,
  isStreaming,
  doneSummary,
}: AgentExecutionStepperProps) {
  const [expanded, setExpanded] = useState(false);

  if (nodes.length === 0 && !isStreaming && !doneSummary) {
    return null;
  }

  const latestByNode = new Map<string, NodeStatusEventPayload>();
  for (const evt of nodes) {
    latestByNode.set(evt.node, evt);
  }
  const refusalEvent = latestByNode.get("graceful_refusal");

  if (doneSummary && !isStreaming && !expanded) {
    const seconds = (doneSummary.totalLatencyMs / 1000).toFixed(2);
    const retryLabel = doneSummary.retriesUsed === 1 ? "retry" : "retries";
    const citationLabel = doneSummary.citationCount === 1 ? "citation" : "citations";
    const summaryText = `Completed in ${seconds}s · ${doneSummary.retriesUsed} ${retryLabel} · ${doneSummary.citationCount} ${citationLabel}`;

    return (
      <div className="mb-2 flex items-center gap-2">
        <button
          type="button"
          onClick={() => setExpanded(true)}
          className="inline-flex items-center gap-1.5 rounded-full border border-borderSubtle bg-surface px-3 py-1 font-mono text-[11px] text-textSecondary transition hover:border-sovereign/50 hover:text-textPrimary focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sovereign"
        >
          <ChevronRight className="h-3 w-3 text-sovereign" aria-hidden="true" />
          <span>{summaryText}</span>
        </button>
      </div>
    );
  }

  return (
    <div className="mb-3 rounded-lg border border-borderSubtle bg-surface/90 p-2.5">
      <div className="mb-2 flex items-center justify-between">
        <span className="font-mono text-[10px] font-semibold uppercase tracking-wider text-textSecondary">
          Agent Execution Stepper (Glass Box)
        </span>
        {doneSummary && !isStreaming && (
          <button
            type="button"
            onClick={() => setExpanded(false)}
            className="inline-flex items-center gap-1 font-mono text-[10px] text-textSecondary hover:text-textPrimary focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sovereign"
          >
            <ChevronDown className="h-3 w-3" aria-hidden="true" />
            Collapse
          </button>
        )}
      </div>

      <div className="flex flex-wrap items-center gap-1.5">
        {PIPELINE_NODES.map((step) => {
          const event =
            step.id === "grounded_generator" && refusalEvent
              ? refusalEvent
              : latestByNode.get(step.id);
          const status = event?.status ?? "idle";
          const displayLabel =
            step.id === "grounded_generator" && refusalEvent
              ? "GracefulRefusal"
              : step.label;

          let badgeStyle =
            "border-borderSubtle bg-canvas/70 text-textSecondary";
          if (status === "started") {
            badgeStyle = "border-sovereign/60 bg-sovereign/10 text-textPrimary";
          } else if (status === "completed") {
            badgeStyle = "border-sovereign/40 bg-sovereign/15 text-emerald-200";
          } else if (status === "retrying") {
            badgeStyle = "border-warning/60 bg-warning/15 text-amber-200";
          } else if (status === "refused" || refusalEvent) {
            badgeStyle = "border-danger/50 bg-danger/15 text-rose-200";
          }

          return (
            <div
              key={step.id}
              title={event?.detail || step.label}
              className={`inline-flex items-center gap-1.5 rounded-md border px-2 py-1 text-[11px] ${badgeStyle}`}
            >
              {status === "started" && (
                <Loader2 className="h-3 w-3 animate-spin text-sovereign" aria-hidden="true" />
              )}
              {status === "completed" && (
                <CheckCircle2 className="h-3 w-3 text-sovereign" aria-hidden="true" />
              )}
              {status === "retrying" && (
                <RefreshCw className="h-3 w-3 text-warning" aria-hidden="true" />
              )}
              {status === "refused" && (
                <AlertTriangle className="h-3 w-3 text-danger" aria-hidden="true" />
              )}
              {status === "idle" && (
                <Circle className="h-2.5 w-2.5 text-slate-600" aria-hidden="true" />
              )}

              <span className="font-medium">{displayLabel}</span>

              {event && status === "retrying" && (
                <span className="rounded bg-warning/20 px-1 font-mono text-[10px] text-warning">
                  Attempt {event.attempt}/2
                </span>
              )}

              {event && event.latency_ms > 0 && status !== "retrying" && (
                <span className="font-mono text-[10px] text-textSecondary">
                  {event.latency_ms}ms
                </span>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
