/**
 * Strict TypeScript domain contracts mirroring FastAPI Pydantic v2 responses
 * and typed SSE wire events from `POST /api/v1/chat/stream`.
 */

export type InferenceMode = "LOCAL_ONPREM" | "CLOUD_API";

export type DocumentStatus = "PENDING" | "INDEXING" | "READY" | "FAILED";

export type LanguageCode = "en" | "ms" | "mixed";

export type DocType = "pdf" | "md" | "txt";

export interface CollectionRead {
  id: string;
  name: string;
  description: string;
  is_seeded: boolean;
  document_count: number;
  created_at: string;
}

export interface CollectionCreate {
  id: string;
  name: string;
  description?: string;
  is_seeded?: boolean;
}

export interface DocumentRead {
  doc_id: string;
  collection_id: string;
  filename: string;
  doc_type: DocType;
  language: LanguageCode;
  content_hash: string;
  page_count: number;
  chunk_count: number;
  status: DocumentStatus;
  error_message: string | null;
  created_at: string;
  updated_at: string;
}

export interface UploadDispatchResult {
  document: DocumentRead;
  deduplicated: boolean;
}

export interface SeedReport {
  collections_created: number;
  documents_ingested: number;
  documents_skipped_duplicate: number;
  collection_ids: string[];
}

export interface CitationItem {
  citation_key: string;
  doc_id: string;
  filename: string;
  page: number;
  score: number;
  breadcrumb: string;
  snippet: string;
}

export interface ChatStreamRequest {
  collection_id: string;
  question: string;
  session_id?: string | null;
  inference_mode?: InferenceMode | null;
  language_filter?: LanguageCode | null;
  doc_type_filter?: DocType | null;
}

export interface NodeStatusEventPayload {
  node: string;
  status: string;
  attempt: number;
  latency_ms: number;
  detail: string;
}

export interface TokenEventPayload {
  delta: string;
}

export interface CitationsEventPayload {
  sources: CitationItem[];
}

export interface DoneEventPayload {
  message_id: string;
  session_id: string;
  mode: InferenceMode;
  retries_used: number;
  cache_hit_tokens: number;
  total_latency_ms: number;
  refusal_triggered: boolean;
}

export interface ErrorEventPayload {
  code: string;
  message: string;
  recoverable: boolean;
}

export type SSEEvent =
  | { type: "node_status"; payload: NodeStatusEventPayload }
  | { type: "token"; payload: TokenEventPayload }
  | { type: "citations"; payload: CitationsEventPayload }
  | { type: "done"; payload: DoneEventPayload }
  | { type: "error"; payload: ErrorEventPayload };

export interface ChatSessionRead {
  session_id: string;
  collection_id: string;
  title: string;
  created_at: string;
}

export interface ChatMessageRead {
  message_id: string;
  session_id: string;
  role: "user" | "assistant";
  content: string;
  inference_mode: InferenceMode;
  citations: CitationItem[];
  agent_trace: NodeStatusEventPayload[];
  cache_hit_tokens: number;
  latency_ms: number;
  created_at: string;
}

export interface ReadinessResponse {
  status: "ready" | "degraded";
  checks: Record<string, string>;
}

export interface ApiErrorDetail {
  code: string;
  message: string;
  detail: unknown;
  request_id: string;
}

export interface ApiErrorEnvelope {
  error: ApiErrorDetail;
}
