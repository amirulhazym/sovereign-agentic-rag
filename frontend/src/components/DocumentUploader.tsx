"use client";

import { UploadCloud } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { pollDocumentStatus, uploadDocument } from "@/lib/api-client";
import type { DocumentRead, UploadDispatchResult } from "@/types/api";

interface DocumentUploaderProps {
  collectionId: string;
  onUploaded: (document: DocumentRead) => void;
  onStatusUpdated: (document: DocumentRead) => void;
  uploadFn?: (collectionId: string, file: File) => Promise<UploadDispatchResult>;
  pollFn?: (docId: string) => Promise<DocumentRead>;
  pollIntervalMs?: number;
}

export function DocumentUploader({
  collectionId,
  onUploaded,
  onStatusUpdated,
  uploadFn = uploadDocument,
  pollFn = pollDocumentStatus,
  pollIntervalMs = 1500,
}: DocumentUploaderProps) {
  const inputRef = useRef<HTMLInputElement | null>(null);
  const activeTimersRef = useRef<Set<number>>(new Set());
  const cancelledRef = useRef<boolean>(false);

  const [isDragging, setIsDragging] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [toast, setToast] = useState<{ kind: "info" | "error"; text: string } | null>(null);

  useEffect(() => {
    cancelledRef.current = false;
    const timers = activeTimersRef.current;
    return () => {
      cancelledRef.current = true;
      timers.forEach((timerId) => window.clearTimeout(timerId));
      timers.clear();
    };
  }, [collectionId]);

  const schedulePoll = (docId: string) => {
    const timerId = window.setTimeout(async () => {
      activeTimersRef.current.delete(timerId);
      if (cancelledRef.current) {
        return;
      }
      try {
        const latest = await pollFn(docId);
        if (cancelledRef.current) {
          return;
        }
        onStatusUpdated(latest);
        if (latest.status === "PENDING" || latest.status === "INDEXING") {
          schedulePoll(docId);
        }
      } catch (err) {
        if (!cancelledRef.current) {
          setToast({
            kind: "error",
            text: err instanceof Error ? err.message : "Failed to poll document status.",
          });
        }
      }
    }, pollIntervalMs);
    activeTimersRef.current.add(timerId);
  };

  const handleFiles = async (files: FileList | null) => {
    if (!files || files.length === 0 || !collectionId) {
      return;
    }
    setIsUploading(true);
    setToast(null);
    try {
      for (const file of Array.from(files)) {
        const result = await uploadFn(collectionId, file);
        if (result.deduplicated) {
          setToast({
            kind: "info",
            text: "Document already indexed in this collection (deduplicated).",
          });
          onStatusUpdated(result.document);
          continue;
        }
        onUploaded(result.document);
        if (result.document.status === "PENDING" || result.document.status === "INDEXING") {
          schedulePoll(result.document.doc_id);
        }
      }
    } catch (err) {
      setToast({
        kind: "error",
        text: err instanceof Error ? err.message : "Document upload failed.",
      });
    } finally {
      setIsUploading(false);
      if (inputRef.current) {
        inputRef.current.value = "";
      }
    }
  };

  return (
    <div className="space-y-2">
      <div
        onDragOver={(e) => {
          e.preventDefault();
          setIsDragging(true);
        }}
        onDragLeave={() => setIsDragging(false)}
        onDrop={(e) => {
          e.preventDefault();
          setIsDragging(false);
          void handleFiles(e.dataTransfer.files);
        }}
        className={`rounded-lg border border-dashed p-3 text-center transition ${
          isDragging
            ? "border-sovereign bg-sovereign/10"
            : "border-borderSubtle bg-canvas/70 hover:border-slate-500"
        }`}
      >
        <input
          ref={inputRef}
          data-testid="document-file-input"
          type="file"
          accept=".pdf,.md,.txt"
          multiple
          className="sr-only"
          onChange={(e) => void handleFiles(e.target.files)}
        />
        <button
          type="button"
          disabled={isUploading}
          onClick={() => inputRef.current?.click()}
          className="flex w-full flex-col items-center gap-1 text-xs text-textSecondary hover:text-textPrimary focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sovereign"
        >
          <UploadCloud className="h-4 w-4 text-sovereign" aria-hidden="true" />
          <span className="font-medium text-textPrimary">
            {isUploading ? "Uploading..." : "Upload .pdf / .md / .txt"}
          </span>
          <span className="text-[11px] text-textSecondary">
            Drag &amp; drop or click (async SHA-256)
          </span>
        </button>
      </div>

      {toast && (
        <p
          role="status"
          className={`rounded-md border px-2.5 py-1.5 text-[11px] leading-snug ${
            toast.kind === "error"
              ? "border-danger/40 bg-danger/10 text-rose-200"
              : "border-sovereign/30 bg-sovereign/10 text-emerald-200"
          }`}
        >
          {toast.text}
        </p>
      )}
    </div>
  );
}
