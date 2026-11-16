import type {
  ApiErrorEnvelope,
  ChatMessageRead,
  ChatSessionRead,
  CollectionCreate,
  CollectionRead,
  DocumentRead,
  ReadinessResponse,
  SeedReport,
  UploadDispatchResult,
} from "@/types/api";

export const DEFAULT_API_BASE_URL =
  process.env.NEXT_PUBLIC_API_URL?.replace(/\/$/, "") || "http://localhost:8000";

export class ApiClientError extends Error {
  readonly status: number;
  readonly code: string;
  readonly detail: unknown;

  constructor(status: number, code: string, message: string, detail: unknown = null) {
    super(`[${code}] ${message}`);
    this.name = "ApiClientError";
    this.status = status;
    this.code = code;
    this.detail = detail;
  }
}

export async function parseErrorResponse(response: Response): Promise<ApiClientError> {
  let rawText = "";
  try {
    rawText = await response.text();
    const parsed = JSON.parse(rawText) as Partial<ApiErrorEnvelope>;
    if (parsed.error && typeof parsed.error.code === "string") {
      return new ApiClientError(
        response.status,
        parsed.error.code,
        parsed.error.message,
        parsed.error.detail ?? null,
      );
    }
  } catch {
    // Fall back to raw response status/body below
  }
  return new ApiClientError(
    response.status,
    `HTTP_${response.status}`,
    rawText || response.statusText || "Request failed",
  );
}

async function requestJson<T>(
  path: string,
  init: RequestInit = {},
  baseUrl: string = DEFAULT_API_BASE_URL,
  fetchFn: typeof fetch = fetch,
): Promise<T> {
  const response = await fetchFn(`${baseUrl.replace(/\/$/, "")}${path}`, init);
  if (!response.ok) {
    throw await parseErrorResponse(response);
  }
  return (await response.json()) as T;
}

export function fetchCollections(
  baseUrl?: string,
  fetchFn?: typeof fetch,
): Promise<CollectionRead[]> {
  return requestJson<CollectionRead[]>("/api/v1/collections", {}, baseUrl, fetchFn);
}

export function createCollection(
  payload: CollectionCreate,
  baseUrl?: string,
  fetchFn?: typeof fetch,
): Promise<CollectionRead> {
  return requestJson<CollectionRead>(
    "/api/v1/collections",
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    },
    baseUrl,
    fetchFn,
  );
}

export function seedCollections(
  baseUrl?: string,
  fetchFn?: typeof fetch,
): Promise<SeedReport> {
  return requestJson<SeedReport>("/api/v1/collections/seed", { method: "POST" }, baseUrl, fetchFn);
}

export function fetchCollectionDocuments(
  collectionId: string,
  baseUrl?: string,
  fetchFn?: typeof fetch,
): Promise<DocumentRead[]> {
  return requestJson<DocumentRead[]>(
    `/api/v1/collections/${encodeURIComponent(collectionId)}/documents`,
    {},
    baseUrl,
    fetchFn,
  );
}

export function uploadDocument(
  collectionId: string,
  file: File,
  baseUrl?: string,
  fetchFn?: typeof fetch,
): Promise<UploadDispatchResult> {
  const formData = new FormData();
  formData.append("file", file);
  return requestJson<UploadDispatchResult>(
    `/api/v1/collections/${encodeURIComponent(collectionId)}/documents`,
    {
      method: "POST",
      body: formData,
    },
    baseUrl,
    fetchFn,
  );
}

export function pollDocumentStatus(
  docId: string,
  baseUrl?: string,
  fetchFn?: typeof fetch,
): Promise<DocumentRead> {
  return requestJson<DocumentRead>(
    `/api/v1/documents/${encodeURIComponent(docId)}/status`,
    {},
    baseUrl,
    fetchFn,
  );
}

export function fetchCollectionSessions(
  collectionId: string,
  baseUrl?: string,
  fetchFn?: typeof fetch,
): Promise<ChatSessionRead[]> {
  return requestJson<ChatSessionRead[]>(
    `/api/v1/collections/${encodeURIComponent(collectionId)}/sessions`,
    {},
    baseUrl,
    fetchFn,
  );
}

export function fetchSessionMessages(
  sessionId: string,
  baseUrl?: string,
  fetchFn?: typeof fetch,
): Promise<ChatMessageRead[]> {
  return requestJson<ChatMessageRead[]>(
    `/api/v1/sessions/${encodeURIComponent(sessionId)}/messages`,
    {},
    baseUrl,
    fetchFn,
  );
}

export async function fetchReadyProbe(
  baseUrl: string = DEFAULT_API_BASE_URL,
  fetchFn: typeof fetch = fetch,
): Promise<ReadinessResponse> {
  const response = await fetchFn(`${baseUrl.replace(/\/$/, "")}/ready`);
  try {
    return (await response.json()) as ReadinessResponse;
  } catch {
    return {
      status: "degraded",
      checks: { api: `http_${response.status}` },
    };
  }
}
