"use client";

import { BookOpen, FileText, Hash, Layers, X } from "lucide-react";
import { useEffect } from "react";
import type { CitationItem } from "@/types/api";

interface EvidenceInspectorDrawerProps {
  citation: CitationItem | null;
  onClose: () => void;
}

export function EvidenceInspectorDrawer({
  citation,
  onClose,
}: EvidenceInspectorDrawerProps) {
  useEffect(() => {
    if (!citation) {
      return;
    }
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        onClose();
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [citation, onClose]);

  if (!citation) {
    return (
      <aside className="hidden h-full w-[340px] shrink-0 flex-col border-l border-borderSubtle bg-surface p-4 xl:flex">
        <div className="mb-3 flex items-center gap-2 border-b border-borderSubtle pb-2.5">
          <BookOpen className="h-4 w-4 text-sovereign" aria-hidden="true" />
          <span className="font-mono text-[11px] font-semibold uppercase tracking-wider text-textSecondary">
            EVIDENCE INSPECTOR
          </span>
        </div>
        <div className="flex flex-1 flex-col items-center justify-center rounded-lg border border-dashed border-borderSubtle bg-canvas/50 p-6 text-center">
          <p className="text-xs font-medium text-textPrimary">No Citation Selected</p>
          <p className="mt-1 text-[11px] text-textSecondary">
            Click any inline <span className="font-mono text-teal-300">[doc_id:pX]</span> badge
            or source card to inspect its hydrated ParentChunk and Breadcrumb.
          </p>
        </div>
      </aside>
    );
  }

  return (
    <aside
      aria-label="Evidence Inspector"
      className="fixed inset-x-0 bottom-0 z-40 flex max-h-[75vh] flex-col rounded-t-2xl border-t border-borderSubtle bg-surface p-4 shadow-2xl lg:static lg:inset-auto lg:max-h-none lg:w-[360px] lg:shrink-0 lg:rounded-none lg:border-l lg:border-t-0"
    >
      <div className="mb-3 flex items-center justify-between border-b border-borderSubtle pb-2.5">
        <div className="flex items-center gap-2">
          <BookOpen className="h-4 w-4 text-sovereign" aria-hidden="true" />
          <span className="font-mono text-[11px] font-semibold uppercase tracking-wider text-textSecondary">
            EVIDENCE INSPECTOR
          </span>
        </div>
        <button
          type="button"
          onClick={onClose}
          aria-label="Close Evidence Inspector"
          className="rounded p-1 text-textSecondary hover:bg-elevated hover:text-textPrimary focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sovereign"
        >
          <X className="h-4 w-4" aria-hidden="true" />
        </button>
      </div>

      <div className="space-y-3 overflow-y-auto pr-1 text-xs">
        <div className="flex flex-wrap items-center justify-between gap-2 rounded-lg border border-borderSubtle bg-canvas/70 px-3 py-2">
          <span className="citation-pill rounded px-2 py-0.5 font-mono text-[11px] font-semibold">
            {citation.citation_key}
          </span>
          <span className="font-mono text-[11px] text-sovereign">
            Score: {citation.score.toFixed(2)}
          </span>
        </div>

        <div className="space-y-2 rounded-lg border border-borderSubtle bg-canvas/50 p-3">
          <div className="flex items-center gap-2 text-textPrimary">
            <FileText className="h-3.5 w-3.5 shrink-0 text-textSecondary" aria-hidden="true" />
            <span className="truncate font-medium" title={citation.filename}>
              {citation.filename}
            </span>
          </div>
          <div className="flex items-center gap-2 font-mono text-[11px] text-textSecondary">
            <Hash className="h-3.5 w-3.5 shrink-0" aria-hidden="true" />
            <span>
              doc_id: {citation.doc_id} · Page {citation.page}
            </span>
          </div>
          <div className="flex items-start gap-2 text-[11px] text-textSecondary">
            <Layers className="mt-0.5 h-3.5 w-3.5 shrink-0 text-sovereign" aria-hidden="true" />
            <span>{citation.breadcrumb}</span>
          </div>
        </div>

        <div className="space-y-1.5">
          <p className="font-mono text-[10px] font-semibold uppercase tracking-wider text-textSecondary">
            Parent Chunk Evidence
          </p>
          <div className="rounded-lg border border-borderSubtle bg-elevated/80 p-3 text-xs leading-relaxed text-textPrimary whitespace-pre-wrap">
            {citation.snippet}
          </div>
        </div>
      </div>
    </aside>
  );
}
