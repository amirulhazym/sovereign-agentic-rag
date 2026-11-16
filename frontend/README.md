# Frontend Workspace (`frontend/`)

Glass-Box **Next.js App Router + TypeScript + Tailwind CSS** console for `sovereign-agentic-rag`, built from [docs/Design-System.md](../docs/Design-System.md).

---

## 1. Local Setup & Verification

```bash
cd frontend
npm install

# Typed SSE client + component unit tests (Vitest + React Testing Library)
npm test

# Strict TypeScript check and ESLint
npm run typecheck
npm run lint

# Standalone production bundle (requires Linux/Docker container or NTFS drive;
# Windows exFAT volumes trigger Node.js fs.readlink EISDIR during trace collection)
npm run build
```

Set `NEXT_PUBLIC_API_URL` (defaults to `http://localhost:8000`) when pointing the UI to a remote or containerized FastAPI backend.

---

## 2. Implemented Workspace Architecture

- **[`src/types/api.ts`](src/types/api.ts):** Strict TypeScript interfaces and discriminated `SSEEvent` union (`node_status`, `token`, `citations`, `done`, `error`) mirroring FastAPI Pydantic v2 payloads.
- **[`src/lib/sse-client.ts`](src/lib/sse-client.ts) & [`src/lib/api-client.ts`](src/lib/api-client.ts):** `fetch()` + `ReadableStream` SSE consumer supporting chunked frame boundaries, `AbortController` cancellation, and REST helpers (`collections`, `seed`, async upload + polling, `/ready`, session history).
- **[`src/components/InferenceModeSwitch.tsx`](src/components/InferenceModeSwitch.tsx):** Top header toggle switching between `LOCAL_ONPREM` (`Ollama Qwen3.8-4B-Distill + FastEmbed bge-m3`) and `CLOUD_API` (`DeepSeek Chat · Prefix Caching On`) without clearing conversation state.
- **[`src/components/CollectionSidebar.tsx`](src/components/CollectionSidebar.tsx) & [`src/components/DocumentUploader.tsx`](src/components/DocumentUploader.tsx):** Multi-tenant workspace switcher, `+ New Collection` form, `Seed Demo Collections` trigger, drag-and-drop `.pdf`/`.md`/`.txt` upload, SHA-256 deduplication notice, and `1500ms` ingestion status polling (`PENDING -> INDEXING -> READY | FAILED`).
- **[`src/components/AgentExecutionStepper.tsx`](src/components/AgentExecutionStepper.tsx):** Live Glass-Box stepper rendering the 6 LangGraph nodes, latency badges, `Attempt N/2` retry loops, `GracefulRefusal`, and collapsible summary pill.
- **[`src/components/ChatWorkspace.tsx`](src/components/ChatWorkspace.tsx) & [`src/components/EvidenceInspectorDrawer.tsx`](src/components/EvidenceInspectorDrawer.tsx):** Collection-tailored starter prompts, inline clickable `[doc_id:pX]` pills, Stop button (`AbortController`), Graceful Refusal callout, SSE error alert with Retry, and collapsible Evidence Inspector drawer (`Escape` closes).
