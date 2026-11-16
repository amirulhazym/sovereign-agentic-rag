# Sovereign Agentic RAG Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `sovereign-agentic-rag` (`v3` production architecture) — a zero-infrastructure-cost, dual-mode (`LOCAL_ONPREM` / `CLOUD_API`) Agentic RAG platform with Qdrant Hybrid RRF, bounded LangGraph orchestration, Next.js SSE workspace UI, and a decoupled `LLM-as-a-Judge` vs. `JEV-as-a-Judge` evaluation suite.

**Architecture:** Layered FastAPI backend (`routers -> services -> agents -> repositories/providers`) communicating with async SQLite + self-hosted Qdrant, exposed via typed SSE (`POST /api/v1/chat/stream`) to a Next.js App Router TypeScript UI, and benchmarked externally via a standalone `eval/` CLI harness.

**Tech Stack:** Python 3.11+, FastAPI, Pydantic v2, LangGraph, Qdrant (`qdrant-client`), FastEmbed (ONNX), PyMuPDF4LLM, `aiosqlite`/SQLAlchemy, Ollama (`hf.co/empero-ai/Qwen3.8-4B-Distill-GGUF:Q4_K_M` / `Qwen3.8-9B-Distill-GGUF:Q4_K_M`), DeepSeek API, Next.js (App Router, TypeScript, Tailwind CSS), Docker Compose.

## Global Constraints
- Strictly `$0` infrastructure cost (self-hosted Qdrant, SQLite, Ollama/FastEmbed; optional DeepSeek API).
- Zero import-time side effects: instantiate models/DB clients in FastAPI `lifespan` and wire via `Depends()`.
- Protect 16GB RAM budget: use FastEmbed (ONNX) instead of PyTorch `sentence-transformers` in `backend/`, default `LOCAL_LLM_MODEL` to `hf.co/empero-ai/Qwen3.8-4B-Distill-GGUF:Q4_K_M` (`OLLAMA_KEEP_ALIVE=5m`, `OLLAMA_NUM_PARALLEL=1`), and isolate `JEVJudge` weights inside `eval/`.
- Strip `<think>...</think>` spans on `empero-ai/Qwen3.8-*-Distill-GGUF` outputs and enforce `temperature=0.0` + `/no_think` / JSON mode on control nodes (`QueryRouter`, `QueryRewriter`, `RelevanceGrader`).
- Order `GroundedGenerator` prompt static-to-dynamic ($\ge 1,024$ token static prefix -> collection header -> history -> sorted `ParentChunk`s -> current question) for DeepSeek Prefix Caching.
- Cap LangGraph loops at `max_retries=2`; route exhausted retries to deterministic bilingual `GracefulRefusal` (`citations: []`).
- Normalize and strip phantom citations via `CitationValidator` against retrieved `ParentChunk` metadata.
- Deduplicate uploads via `(collection_id, sha256)` in SQLite; run ingestion via `BackgroundTasks` + `asyncio.to_thread`.
- Set SSE headers (`X-Accel-Buffering: no`, `Cache-Control: no-cache, no-transform`) and wire `request.is_disconnected()` to Next.js `AbortController`.
- Enforce strict TDD: Red -> Verify Red -> Green -> Verify Green -> Refactor -> Commit.

---

## Phase 2: Backend Core, Storage & Dual-Mode Inference Seam

### Task 1: Backend Package Scaffold, Settings & Structured Error Envelopes
**Files:**
- Create: `backend/pyproject.toml`, `backend/app/main.py`, `backend/app/core/config.py`, `backend/app/core/exceptions.py`, `backend/app/schemas/errors.py`
- Test: `backend/tests/test_health_and_errors.py`

**Interfaces:**
- Produces: `Settings` (Pydantic `BaseSettings`), `AppError` hierarchy, `create_app() -> FastAPI`.

- [x] **Step 1:** Write failing tests verifying `/health` returns `200`, unknown routes/domain errors return uniform `{"error": {"code", "message", "detail", "request_id"}}` JSON envelopes, and importing `app.main` performs zero model downloads.
- [x] **Step 2:** Run `pytest tests/test_health_and_errors.py -v` -> confirm RED.
- [x] **Step 3:** Implement `config.py`, `exceptions.py`, `errors.py`, and `create_app()` factory with FastAPI `lifespan`.
- [x] **Step 4:** Re-run `pytest tests/test_health_and_errors.py -v` -> confirm GREEN.
- [x] **Step 5:** Commit `feat(backend): scaffold FastAPI app factory, config, and error envelopes`.

### Task 2: Async SQLite Repository (`collections`, `documents`, `parent_chunks`, `chat_history`)
**Files:**
- Create: `backend/app/repositories/sqlite_repo.py`, `backend/app/schemas/workspace.py`
- Test: `backend/tests/test_sqlite_repo.py`

**Interfaces:**
- Consumes: `Settings.sqlite_url`
- Produces: `SQLiteRepository` async CRUD for collections, SHA-256 deduplicated documents (`PENDING|INDEXING|READY|FAILED`), `ParentChunk` batch storage/hydration, and chat history.

- [x] **Step 1:** Write failing tests against an in-memory/temp SQLite DB testing collection creation, `(collection_id, content_hash)` duplicate check, status transitions, and `ParentChunk` hydration preserving order.
- [x] **Step 2:** Run `pytest tests/test_sqlite_repo.py -v` -> confirm RED.
- [x] **Step 3:** Implement `SQLiteRepository` with `aiosqlite` / SQLAlchemy 2.0 async engine + schema initialization.
- [x] **Step 4:** Re-run `pytest tests/test_sqlite_repo.py -v` -> confirm GREEN.
- [x] **Step 5:** Commit `feat(backend): add async SQLite repository for collections, documents, parents, and chat`.

### Task 3: Qdrant Hybrid Search Repository (`Dense + Sparse BM25 + RRF`) & InMemory Fake
**Files:**
- Create: `backend/app/repositories/qdrant_repo.py`
- Test: `backend/tests/test_qdrant_repo.py`

**Interfaces:**
- Produces: `VectorRepository` Protocol, `QdrantHybridRepository`, `InMemoryVectorRepository` (used by downstream unit tests).

- [x] **Step 1:** Write failing tests verifying `collection_id` payload isolation, combined Dense + Sparse RRF scoring behavior, and document deletion by `doc_id` using `InMemoryVectorRepository` + unit mocks validating `QdrantClient` `Prefetch` + `Fusion.RRF` payload filters.
- [x] **Step 2:** Run `pytest tests/test_qdrant_repo.py -v` -> confirm RED.
- [x] **Step 3:** Implement `VectorRepository`, `InMemoryVectorRepository`, and `QdrantHybridRepository`.
- [x] **Step 4:** Re-run `pytest tests/test_qdrant_repo.py -v` -> confirm GREEN.
- [x] **Step 5:** Commit `feat(backend): add Qdrant hybrid RRF repository seam and in-memory fake`.

### Task 4: Dual-Mode `LLMProvider` Seam (`LOCAL_ONPREM` Ollama & `CLOUD_API` DeepSeek)
**Files:**
- Create: `backend/app/providers/base.py`, `backend/app/providers/local_onprem.py`, `backend/app/providers/cloud_deepseek.py`, `backend/app/providers/prompts.py`
- Test: `backend/tests/test_llm_providers.py`

**Interfaces:**
- Produces: `LLMProvider` Protocol (`generate_structured`, `stream_answer`), `LocalOnPremLLMAdapter`, `CloudDeepSeekLLMAdapter`, `build_grounded_messages()` (static-to-dynamic prefix layout $\ge 1,024$ tokens).

- [x] **Step 1:** Write failing tests asserting: (a) `build_grounded_messages()` puts static system prompt ($\ge 1,024$ tokens) before sorted `ParentChunk` blocks and query; (b) control-node requests attach `/no_think` for Ollama and `temperature=0.0` JSON mode for DeepSeek; (c) streaming adapter yields token deltas and extracts `cache_hit_tokens`.
- [x] **Step 2:** Run `pytest tests/test_llm_providers.py -v` -> confirm RED.
- [x] **Step 3:** Implement provider interfaces, prompt prefix builder, and mocked-HTTP-tested Ollama/DeepSeek adapters.
- [x] **Step 4:** Re-run `pytest tests/test_llm_providers.py -v` -> confirm GREEN.
- [x] **Step 5:** Commit `feat(backend): implement dual-mode LLMProvider seam with DeepSeek prefix caching`.

### Task 5: Lightweight `EmbeddingProvider` & `RerankerProvider` (FastEmbed ONNX)
**Files:**
- Create: `backend/app/providers/embeddings.py`
- Test: `backend/tests/test_embeddings_and_reranker.py`

**Interfaces:**
- Produces: `EmbeddingProvider` Protocol, `FastEmbedProvider` (ONNX `BAAI/bge-m3` + `Qdrant/bm25`), `ONNXRerankerProvider`, and lightweight test fakes.

- [x] **Step 1:** Write failing tests for dense/sparse vector shapes, rerank ordering, and lazy loading behavior (never loading models at module import).
- [x] **Step 2:** Run `pytest tests/test_embeddings_and_reranker.py -v` -> confirm RED.
- [x] **Step 3:** Implement `FastEmbedProvider` and `ONNXRerankerProvider` wrapped safely for 16GB RAM execution.
- [x] **Step 4:** Re-run `pytest tests/test_embeddings_and_reranker.py -v` -> confirm GREEN.
- [x] **Step 5:** Commit `feat(backend): add FastEmbed ONNX embeddings and reranker providers`.

---

## Phase 3: Document Ingestion (`PyMuPDF4LLM` + Parent-Child + `BackgroundTasks`) & Seed Collections

### Task 6: Structure-Aware Extraction & Hierarchical Parent-Child Chunker
**Files:**
- Create: `backend/app/services/chunking.py`
- Test: `backend/tests/test_chunking.py`

**Interfaces:**
- Produces: `extract_pages(file_bytes, doc_type) -> list[ExtractedPage]`, `build_parent_child_chunks(pages, doc_id, filename) -> tuple[list[ParentChunk], list[ChildChunk]]`.

- [x] **Step 1:** Write failing tests verifying Markdown heading/table preservation, `.pdf`/`.md`/`.txt` parsing, `ParentChunk` (~1,200–1,500 tokens) vs. `ChildChunk` (~300–400 tokens, 50 overlap), and `Breadcrumb` formatting `[filename | Page X > Section]`.
- [x] **Step 2:** Run `pytest tests/test_chunking.py -v` -> confirm RED.
- [x] **Step 3:** Implement `PyMuPDF4LLM` extractor and `HierarchicalChunker`.
- [x] **Step 4:** Re-run `pytest tests/test_chunking.py -v` -> confirm GREEN.
- [x] **Step 5:** Commit `feat(ingestion): add PyMuPDF4LLM extraction and parent-child breadcrumb chunker`.

### Task 7: `DocumentIngestionService` (SHA-256 Deduplication & Async Lifecycle) + REST Routers
**Files:**
- Create: `backend/app/services/ingestion.py`, `backend/app/api/v1/routers/collections.py`, `backend/app/api/v1/routers/documents.py`, `backend/app/core/dependencies.py`
- Test: `backend/tests/test_ingestion_api.py`

**Interfaces:**
- Consumes: `SQLiteRepository`, `VectorRepository`, `EmbeddingProvider`, `chunking.py`
- Produces: `POST /api/v1/collections`, `GET /api/v1/collections`, `POST /api/v1/collections/{id}/documents` (`202` or `200` duplicate), `GET /api/v1/documents/{doc_id}/status`.

- [x] **Step 1:** Write failing API contract tests for collection CRUD, `202 Accepted` async upload, `PENDING -> INDEXING -> READY` transition via `asyncio.to_thread`, `FAILED` error capture, and `200 OK` SHA-256 deduplication on repeat upload.
- [x] **Step 2:** Run `pytest tests/test_ingestion_api.py -v` -> confirm RED.
- [x] **Step 3:** Implement `DocumentIngestionService`, dependency wiring, and routers.
- [x] **Step 4:** Re-run `pytest tests/test_ingestion_api.py -v` -> confirm GREEN.
- [x] **Step 5:** Commit `feat(ingestion): add async background ingestion service and collection/document APIs`.

### Task 8: Pre-Curated Seed Collections (`it_infra_sops` & `malaysian_enterprise_policy`) + Seeder
**Files:**
- Create: `data/seed_collections/it_infra_sops/*.md`, `data/seed_collections/malaysian_enterprise_policy/*.md`, `backend/app/services/seeder.py`
- Test: `backend/tests/test_seeder.py`

**Interfaces:**
- Produces: `seed_demo_collections()` and `POST /api/v1/collections/seed`.

- [x] **Step 1:** Author rich, realistic bilingual seed documents (Collection A: server hardware spec tables, NVMe/RAID SOPs, deployment runbooks; Collection B: English, Bahasa Melayu, and BM-EN code-switching Malaysian HR/PDPA compliance policies) and write failing idempotent seeding test.
- [x] **Step 2:** Run `pytest tests/test_seeder.py -v` -> confirm RED.
- [x] **Step 3:** Implement `SeederService` and `/api/v1/collections/seed` endpoint.
- [x] **Step 4:** Re-run `pytest tests/test_seeder.py -v` -> confirm GREEN.
- [x] **Step 5:** Commit `feat(data): add pre-seeded bilingual demo collections and idempotent seeder`.

---

## Phase 4: Bounded LangGraph State Machine & FastAPI SSE Streaming

### Task 9: `CitationValidator` & Deterministic `GracefulRefusal`
**Files:**
- Create: `backend/app/agents/validators.py`
- Test: `backend/tests/test_citation_validator.py`

**Interfaces:**
- Produces: `validate_and_normalize_citations(text, allowed_parents) -> tuple[str, list[CitationItem]]`, `build_graceful_refusal(language) -> str`.

- [x] **Step 1:** Write failing tests verifying bracket variant normalization (`[hr_pol, page 4]` -> `[hr_pol:p4]`), phantom citation stripping, and `ms` vs. `en` refusal messages.
- [x] **Step 2:** Run `pytest tests/test_citation_validator.py -v` -> confirm RED.
- [x] **Step 3:** Implement regex `CitationValidator` and bilingual refusal templates.
- [x] **Step 4:** Re-run `pytest tests/test_citation_validator.py -v` -> confirm GREEN.
- [x] **Step 5:** Commit `feat(agents): add CitationValidator and bilingual GracefulRefusal`.

### Task 10: LangGraph State, 6 Nodes & Bounded Retry Graph (`max_retries=2`)
**Files:**
- Create: `backend/app/agents/state.py`, `backend/app/agents/nodes.py`, `backend/app/agents/graph.py`
- Test: `backend/tests/test_langgraph_workflow.py`

**Interfaces:**
- Consumes: `LLMProvider`, `EmbeddingProvider`, `RerankerProvider`, `VectorRepository`, `SQLiteRepository`
- Produces: `AgenticRAGWorkflow` compiling the 6-node `StateGraph` (`QueryRouter -> QueryRewriter -> HybridRetriever -> CrossEncoderReranker -> RelevanceGrader -> GroundedGenerator | GracefulRefusal`).

- [x] **Step 1:** Write failing tests covering: (a) direct route bypass; (b) retry loop when `RelevanceGrader` rejects once then approves on attempt 2; (c) strict loop cap at `max_retries=2` triggering `GracefulRefusal`; (d) `/no_think` on control nodes.
- [x] **Step 2:** Run `pytest tests/test_langgraph_workflow.py -v` -> confirm RED.
- [x] **Step 3:** Implement `RAGGraphState`, node classes, conditional edges, and `compile_rag_graph()`.
- [x] **Step 4:** Re-run `pytest tests/test_langgraph_workflow.py -v` -> confirm GREEN.
- [x] **Step 5:** Commit `feat(agents): implement 6-node bounded LangGraph state machine`.

### Task 11: FastAPI SSE Chat Endpoint (`POST /api/v1/chat/stream`) & Readiness Probe
**Files:**
- Create: `backend/app/schemas/chat.py`, `backend/app/services/chat_service.py`, `backend/app/api/v1/routers/chat.py`, `backend/app/api/v1/routers/health.py`
- Test: `backend/tests/test_chat_sse_contract.py`

**Interfaces:**
- Produces: `POST /api/v1/chat/stream` emitting `node_status`, `token`, `citations`, `done`, `error` with anti-buffering headers and `request.is_disconnected()` abort handling, plus `GET /ready`.

- [x] **Step 1:** Write failing SSE contract tests checking event ordering (`node_status` -> `token` -> `citations` -> `done`), mid-stream `error` event on provider exception, anti-buffering headers, client disconnect termination, and SQLite chat history persistence.
- [x] **Step 2:** Run `pytest tests/test_chat_sse_contract.py -v` -> confirm RED.
- [x] **Step 3:** Implement `ChatService`, SSE serializer, disconnect check, and `/ready` probe.
- [x] **Step 4:** Re-run `pytest tests/test_chat_sse_contract.py -v` -> confirm GREEN.
- [x] **Step 5:** Commit `feat(api): add typed SSE streaming endpoint and readiness probes`.

---

## Phase 5: Standalone Evaluation Sub-Project (`eval/`)

### Task 12: `eval/` Package, Schemas, `RAGTargetAdapter` & `golden_dataset.jsonl`
**Files:**
- Create: `eval/pyproject.toml`, `eval/eval_harness/schemas.py`, `eval/eval_harness/target_adapter.py`, `eval/datasets/golden_dataset.jsonl`
- Test: `eval/tests/test_target_adapter_and_dataset.py`

**Interfaces:**
- Produces: `EvalSample`, `RAGOutput`, `JudgeVerdict`, `RAGTargetAdapter`, `HTTPTargetAdapter` (parses SSE stream), `RecordedTraceTargetAdapter`.

- [x] **Step 1:** Author 36 curated bilingual samples in `golden_dataset.jsonl` (including `unanswerable_refusal` traps) and write failing tests validating JSONL schema parsing and SSE response parsing in `HTTPTargetAdapter`.
- [x] **Step 2:** Run `pytest eval/tests/test_target_adapter_and_dataset.py -v` -> confirm RED.
- [x] **Step 3:** Implement schemas, `HTTPTargetAdapter`, and `RecordedTraceTargetAdapter` using `httpx.AsyncClient`.
- [x] **Step 4:** Re-run `pytest` -> confirm GREEN.
- [x] **Step 5:** Commit `feat(eval): add standalone eval schemas, HTTP SSE adapter, and 36-sample golden dataset`.

### Task 13: `BaseJudgeEvaluator`, `LLMJudgeEvaluator` & `JEVJudgeEvaluator`
**Files:**
- Create: `eval/eval_harness/judges/base.py`, `eval/eval_harness/judges/llm_judge.py`, `eval/eval_harness/judges/jev_judge.py`
- Test: `eval/tests/test_judges.py`

**Interfaces:**
- Produces: `BaseJudgeEvaluator.evaluate(sample, output) -> JudgeVerdict` implemented by `LLMJudgeEvaluator` and `JEVJudgeEvaluator`.

- [x] **Step 1:** Write failing tests verifying: (a) both judges return identical `JudgeVerdict` contract bounds `[0.0, 1.0]`; (b) `LLMJudgeEvaluator` calculates token cost and handles trap refusals; (c) `JEVJudgeEvaluator` runs deterministically (`StdDev == 0.0` across repeated calls, `$0.00` cost) using sentence entailment scoring.
- [x] **Step 2:** Run `pytest eval/tests/test_judges.py -v` -> confirm RED.
- [x] **Step 3:** Implement `BaseJudgeEvaluator`, `LLMJudgeEvaluator`, and `JEVJudgeEvaluator`.
- [x] **Step 4:** Re-run `pytest eval/tests/test_judges.py -v` -> confirm GREEN.
- [x] **Step 5:** Commit `feat(eval): implement LLMJudgeEvaluator and deterministic JEVJudgeEvaluator`.

### Task 14: Benchmark Statistics (`Variance`, `Cohen's Kappa`) & CLI Report Runner
**Files:**
- Create: `eval/eval_harness/metrics.py`, `eval/eval_harness/runner.py`, `eval/reports/.gitkeep`, `eval/README.md`
- Test: `eval/tests/test_benchmark_metrics.py`

**Interfaces:**
- Produces: `compute_benchmark_summary()`, CLI `python -m eval_harness.runner --runs 3` outputting `benchmark_report.md` and `.json`.

- [x] **Step 1:** Write failing unit tests for latency percentiles (`p50`, `p95`), multi-run score variance (`StdDev`), raw agreement %, and Cohen's Kappa ($\kappa$) edge cases (including full agreement and chance-level agreement).
- [x] **Step 2:** Run `pytest eval/tests/test_benchmark_metrics.py -v` -> confirm RED.
- [x] **Step 3:** Implement `metrics.py` and async CLI `runner.py`.
- [x] **Step 4:** Re-run `pytest eval/tests/test_benchmark_metrics.py -v` -> confirm GREEN.
- [x] **Step 5:** Commit `feat(eval): add multi-run variance, Cohen's Kappa, and comparison report runner`.

---

## Phase 6: Next.js Workspace Frontend (App Router + TypeScript)

### Task 15: Next.js Scaffold, Design Tokens & Typed SSE Client (`fetch` + `ReadableStream`)
**Files:**
- Create: `frontend/package.json`, `frontend/tsconfig.json`, `frontend/src/types/api.ts`, `frontend/src/lib/sse-client.ts`, `frontend/src/lib/api-client.ts`, `frontend/src/app/layout.tsx`, `frontend/src/app/globals.css`
- Test: `frontend/src/lib/__tests__/sse-client.test.ts`

**Interfaces:**
- Produces: `streamChatEvents(payload, onEvent, signal)` supporting chunked SSE boundaries across arbitrary byte splits and `AbortController`.

- [x] **Step 1:** Write failing Vitest unit tests passing split/chunked SSE frames (`event: node_status`, `event: token`, `event: citations`, `event: done`, `event: error`) across `ReadableStream` boundaries and testing abort cancellation.
- [x] **Step 2:** Run `npm test` -> confirm RED.
- [x] **Step 3:** Scaffold Next.js App Router + Tailwind tokens from `docs/Design-System.md` and implement `sse-client.ts`.
- [x] **Step 4:** Run `npm test && npm run typecheck` -> confirm GREEN.
- [x] **Step 5:** Commit `feat(frontend): scaffold Next.js workspace and typed SSE ReadableStream client`.

### Task 16: 3-Pane Workspace UI (`CollectionSidebar`, `ModeSwitch`, `AgentStepper`, `CitationInspector`)
**Files:**
- Create: `frontend/src/components/CollectionSidebar.tsx`, `frontend/src/components/DocumentUploader.tsx`, `frontend/src/components/InferenceModeSwitch.tsx`, `frontend/src/components/AgentExecutionStepper.tsx`, `frontend/src/components/ChatWorkspace.tsx`, `frontend/src/components/EvidenceInspectorDrawer.tsx`, `frontend/src/components/SovereignWorkspace.tsx`, `frontend/src/app/page.tsx`, `frontend/README.md`
- Test: `frontend/src/components/__tests__/workspace.test.tsx`

**Interfaces:**
- Consumes: `sse-client.ts`, backend `/api/v1/*` endpoints.

- [x] **Step 1:** Write failing React Testing Library tests covering `InferenceModeSwitch` toggling `LOCAL_ONPREM` <-> `CLOUD_API`, `AgentExecutionStepper` rendering retry attempts, clickable `[doc_id:pX]` citation pills opening `EvidenceInspectorDrawer`, and document upload status polling (`PENDING -> READY`).
- [x] **Step 2:** Run `npm test` -> confirm RED.
- [x] **Step 3:** Implement all workspace components per `docs/Design-System.md`.
- [x] **Step 4:** Run `npm test && npm run typecheck && npm run lint` -> confirm GREEN (`npm run build` deferred to Phase 7 Linux Docker build due to Windows `exFAT` drive `readlink EISDIR` limitation on `F:\`).
- [x] **Step 5:** Commit `feat(frontend): build 3-pane workspace with live Agent Stepper and citation inspector`.

---

## Phase 7: Docker Compose Orchestration, Portfolio `README.md` & Full Verification

### Task 17: Multi-Stage Dockerfiles, `docker-compose.yml` & `.env.example`
**Files:**
- Create: `backend/Dockerfile`, `frontend/Dockerfile`, `docker-compose.yml`, `.env.example`, `.dockerignore`, `frontend/.dockerignore`, `frontend/public/.gitkeep`

- [x] **Step 1:** Author multi-stage Dockerfiles (`python:3.11-slim` backend, `node:20-alpine` standalone Next.js), `docker-compose.yml` orchestrating `qdrant`, `backend`, `frontend`, and optional `ollama` profile with healthchecks (`condition: service_healthy`) and named volumes.
- [x] **Step 2:** Validate `docker-compose.yml` schema (`yaml.safe_load` verified; live `docker compose config && docker compose build` to be executed on a host with Docker Desktop installed).
- [x] **Step 3:** Commit `build(docker): add multi-stage Dockerfiles and Compose orchestration`.

### Task 18: Portfolio `README.md` & End-to-End Verification Gate
**Files:**
- Update: `README.md`

- [x] **Step 1:** Author `README.md` including the required **"Engineering Progression: From `v1`/`v2` (`malay-qa-bot-rag`) to `v3` Production Architecture"** matrix, quickstart (`LOCAL_ONPREM` & `CLOUD_API`), architecture diagram, and verified suite commands.
- [x] **Step 2:** Run fresh verification across all local suites (`backend pytest + ruff`, `eval pytest + ruff`, `frontend vitest + typecheck + lint`) and confirm zero failures or skipped assertions.
- [x] **Step 3:** Commit `docs: finalize portfolio README and Phase 7 completion`.
