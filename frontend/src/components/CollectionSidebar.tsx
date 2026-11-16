"use client";

import {
  CheckCircle2,
  Database,
  FileText,
  FolderPlus,
  Loader2,
  Sparkles,
  XCircle,
} from "lucide-react";
import { useState } from "react";
import { DocumentUploader } from "@/components/DocumentUploader";
import type { CollectionCreate, CollectionRead, DocumentRead } from "@/types/api";

interface CollectionSidebarProps {
  collections: CollectionRead[];
  activeCollectionId: string;
  documents: DocumentRead[];
  isLoadingDocs: boolean;
  isSeeding: boolean;
  onSelectCollection: (collectionId: string) => void;
  onCreateCollection: (payload: CollectionCreate) => Promise<void>;
  onSeedCollections: () => Promise<void>;
  onDocumentUploaded: (doc: DocumentRead) => void;
  onDocumentStatusUpdated: (doc: DocumentRead) => void;
}

function StatusBadge({ doc }: { doc: DocumentRead }) {
  switch (doc.status) {
    case "READY":
      return (
        <span
          title={`${doc.page_count} pages · ${doc.chunk_count} chunks`}
          className="inline-flex items-center gap-1 rounded bg-sovereign/15 px-1.5 py-0.5 font-mono text-[10px] font-medium text-sovereign"
        >
          <CheckCircle2 className="h-3 w-3" aria-hidden="true" />
          READY
        </span>
      );
    case "FAILED":
      return (
        <span
          title={doc.error_message || "Ingestion failed"}
          className="inline-flex items-center gap-1 rounded bg-danger/15 px-1.5 py-0.5 font-mono text-[10px] font-medium text-danger"
        >
          <XCircle className="h-3 w-3" aria-hidden="true" />
          FAILED
        </span>
      );
    case "INDEXING":
    case "PENDING":
    default:
      return (
        <span className="inline-flex items-center gap-1 rounded bg-warning/15 px-1.5 py-0.5 font-mono text-[10px] font-medium text-warning">
          <Loader2 className="h-3 w-3 animate-spin" aria-hidden="true" />
          {doc.status}
        </span>
      );
  }
}

export function CollectionSidebar({
  collections,
  activeCollectionId,
  documents,
  isLoadingDocs,
  isSeeding,
  onSelectCollection,
  onCreateCollection,
  onSeedCollections,
  onDocumentUploaded,
  onDocumentStatusUpdated,
}: CollectionSidebarProps) {
  const [showCreateForm, setShowCreateForm] = useState(false);
  const [newName, setNewName] = useState("");
  const [newDesc, setNewDesc] = useState("");
  const [createError, setCreateError] = useState<string | null>(null);
  const [isCreating, setIsCreating] = useState(false);

  const handleCreateSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    const trimmed = newName.trim();
    if (trimmed.length < 2) {
      setCreateError("Collection name must be at least 2 characters.");
      return;
    }
    const slug = trimmed
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, "_")
      .replace(/^_+|_+$/g, "")
      .slice(0, 60);
    if (slug.length < 2) {
      setCreateError("Name must contain at least 2 alphanumeric characters.");
      return;
    }

    setIsCreating(true);
    setCreateError(null);
    try {
      await onCreateCollection({
        id: slug,
        name: trimmed,
        description: newDesc.trim(),
      });
      setNewName("");
      setNewDesc("");
      setShowCreateForm(false);
    } catch (err) {
      setCreateError(err instanceof Error ? err.message : "Failed to create collection.");
    } finally {
      setIsCreating(false);
    }
  };

  return (
    <aside className="flex h-full w-full flex-col border-r border-borderSubtle bg-surface lg:w-[280px] lg:shrink-0">
      {/* Collections Header */}
      <div className="flex items-center justify-between border-b border-borderSubtle px-3 py-2.5">
        <span className="flex items-center gap-1.5 font-mono text-[11px] font-semibold uppercase tracking-wider text-textSecondary">
          <Database className="h-3.5 w-3.5 text-sovereign" aria-hidden="true" />
          Collections
        </span>
        <button
          type="button"
          onClick={() => setShowCreateForm((prev) => !prev)}
          className="inline-flex items-center gap-1 rounded px-2 py-1 text-[11px] font-medium text-textSecondary hover:bg-elevated hover:text-textPrimary focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sovereign"
        >
          <FolderPlus className="h-3.5 w-3.5" aria-hidden="true" />
          <span>New</span>
        </button>
      </div>

      {/* Inline Create Collection Form */}
      {showCreateForm && (
        <form
          onSubmit={(e) => void handleCreateSubmit(e)}
          className="space-y-2 border-b border-borderSubtle bg-canvas/60 p-3"
        >
          <input
            type="text"
            value={newName}
            onChange={(e) => setNewName(e.target.value)}
            placeholder="Collection Name (e.g. Legal Audits)"
            className="w-full rounded border border-borderSubtle bg-surface px-2.5 py-1.5 text-xs text-textPrimary placeholder:text-textSecondary focus:border-sovereign focus:outline-none"
          />
          <input
            type="text"
            value={newDesc}
            onChange={(e) => setNewDesc(e.target.value)}
            placeholder="Short scope description"
            className="w-full rounded border border-borderSubtle bg-surface px-2.5 py-1.5 text-xs text-textPrimary placeholder:text-textSecondary focus:border-sovereign focus:outline-none"
          />
          {createError && <p className="text-[11px] text-danger">{createError}</p>}
          <div className="flex justify-end gap-2">
            <button
              type="button"
              onClick={() => setShowCreateForm(false)}
              className="rounded px-2 py-1 text-[11px] text-textSecondary hover:text-textPrimary"
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={isCreating}
              className="rounded bg-sovereign px-2.5 py-1 text-[11px] font-semibold text-slate-950 hover:bg-emerald-400 disabled:opacity-50"
            >
              {isCreating ? "Creating..." : "Create"}
            </button>
          </div>
        </form>
      )}

      {/* Collection Selector List */}
      <div className="max-h-52 space-y-1 overflow-y-auto border-b border-borderSubtle p-2">
        {collections.length === 0 ? (
          <div className="rounded-lg border border-borderSubtle bg-canvas/50 p-3 text-center">
            <p className="text-xs text-textSecondary">No collections yet.</p>
          </div>
        ) : (
          collections.map((col) => {
            const active = col.id === activeCollectionId;
            return (
              <button
                key={col.id}
                type="button"
                onClick={() => onSelectCollection(col.id)}
                className={`flex w-full items-start justify-between gap-2 rounded-lg px-2.5 py-2 text-left transition focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sovereign ${
                  active
                    ? "bg-elevated text-textPrimary ring-1 ring-sovereign/50"
                    : "text-textSecondary hover:bg-elevated/60 hover:text-textPrimary"
                }`}
              >
                <div className="min-w-0">
                  <p className="truncate text-xs font-medium text-textPrimary">{col.name}</p>
                  <p className="truncate font-mono text-[10px] text-textSecondary">{col.id}</p>
                </div>
                <span className="shrink-0 rounded bg-canvas px-1.5 py-0.5 font-mono text-[10px] text-textSecondary">
                  {col.document_count}
                </span>
              </button>
            );
          })
        )}

        <button
          type="button"
          disabled={isSeeding}
          onClick={() => void onSeedCollections()}
          className="mt-1 flex w-full items-center justify-center gap-1.5 rounded-md border border-borderSubtle bg-canvas px-2.5 py-1.5 text-[11px] font-medium text-textSecondary transition hover:border-sovereign/60 hover:text-textPrimary focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sovereign disabled:opacity-50"
        >
          <Sparkles className="h-3.5 w-3.5 text-sovereign" aria-hidden="true" />
          <span>{isSeeding ? "Seeding Demos..." : "Seed Demo Collections"}</span>
        </button>
      </div>

      {/* Documents & Uploader */}
      <div className="flex min-h-0 flex-1 flex-col p-3">
        <div className="mb-2 flex items-center justify-between">
          <span className="font-mono text-[11px] font-semibold uppercase tracking-wider text-textSecondary">
            Documents ({documents.filter((d) => d.status === "READY").length} Ready)
          </span>
        </div>

        {activeCollectionId && (
          <div className="mb-3">
            <DocumentUploader
              collectionId={activeCollectionId}
              onUploaded={onDocumentUploaded}
              onStatusUpdated={onDocumentStatusUpdated}
            />
          </div>
        )}

        <div className="min-h-0 flex-1 space-y-1.5 overflow-y-auto pr-0.5">
          {isLoadingDocs ? (
            <p className="py-4 text-center font-mono text-xs text-textSecondary">
              Loading documents...
            </p>
          ) : documents.length === 0 ? (
            <p className="py-4 text-center text-xs text-textSecondary">
              No documents indexed in this workspace yet.
            </p>
          ) : (
            documents.map((doc) => (
              <div
                key={doc.doc_id}
                className="flex items-center justify-between gap-2 rounded-md border border-borderSubtle bg-canvas/70 px-2.5 py-2"
              >
                <div className="flex min-w-0 items-center gap-2">
                  <FileText className="h-3.5 w-3.5 shrink-0 text-textSecondary" aria-hidden="true" />
                  <div className="min-w-0">
                    <p className="truncate text-xs text-textPrimary" title={doc.filename}>
                      {doc.filename}
                    </p>
                    <p className="font-mono text-[10px] text-textSecondary">
                      {doc.language.toUpperCase()} · {doc.page_count}p
                    </p>
                  </div>
                </div>
                <StatusBadge doc={doc} />
              </div>
            ))
          )}
        </div>
      </div>
    </aside>
  );
}
