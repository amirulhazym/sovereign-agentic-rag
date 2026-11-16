# Product Requirements Document (PRD): `sovereign-agentic-rag`

> **Status:** Locked Specification (Fasa 1)  
> **Owner:** Amirulhazym  
> **Lineage:** Evolution of [`malay-qa-bot-rag`](https://github.com/amirulhazym/malay-qa-bot-rag) (`v1_malay_selfhosted` & `v2_multilingual_api`) into Production-Grade Sovereign Agentic RAG (`v3`)

---

## 1. Executive Summary & Dual Mission

`sovereign-agentic-rag` is a containerized, collection-scoped **Agentic Retrieval-Augmented Generation (Agentic RAG)** platform engineered for zero infrastructure cost (`$0`) and strict data sovereignty. Designed for both air-gapped on-premises deployments (inspired by enterprise sovereign AI workloads such as Maistorage Technology Malaysia) and cost-optimized cloud inference, it cleanly isolates knowledge into multi-tenant `Collection` workspaces and streams observable agent reasoning to a Next.js workspace UI.

### Dual Mission
1. **Career & Portfolio Impact:** Demonstrate enterprise AI engineering readiness across FastAPI, Pydantic v2, Qdrant Hybrid Search, LangGraph bounded state machines, Next.js App Router + TypeScript, Docker Compose orchestration, and scientific RAG evaluation (`LLM-as-a-Judge` vs. `JEV-as-a-Judge`).
2. **Deep Hands-On Mastery (Anti-Black-Box):** Every `Module` keeps a small, clear `Interface` and clean `Seam` without bloated framework abstractions so each layer can be traced, tested, and reverse-engineered directly.

---

## 2. Problem Statement

Auditing the 2025 `malay-qa-bot-rag` (`v1` and `v2`) codebase and early `v3` checkpoint surfaced **five critical production flaws**:

1. **Linear One-Shot Control Flow (No Self-Correction or Sovereignty Guardrails):** `v2` executes `encode -> pinecone.query -> cross_encoder -> gemini` unconditionally. If a user asks a Bahasa Melayu–English (`BM-EN`) code-switched query or uses domain acronyms, bad retrieval passes straight to generation. Worse, the early `v3` prototype falls back to external web search (Tavily), leaking sensitive enterprise queries outside the tenant perimeter.
2. **Pure Dense Retrieval & Naive Fixed-Width Chunking:** `v1` uses local FAISS (`allow_dangerous_deserialization=True`) and `v2` uses Pinecone cosine dense search with `RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)`. Fixed-width splitting slices across hardware specification tables and Malaysian statutory clauses, stripping page numbers and losing exact codes/part numbers (`Context Precision` plateaued at `0.336` in `v2`). Pickled BM25 indices in early `v3` fail concurrent multi-tenant updates and pre-scoring payload filtering.
3. **Monolithic Blocking Backend & Duplicated Logic:** `v2` `main.py` instantiates heavy ML models at import time, blocks synchronous HTTP `POST` calls for 3–15s, couples backend + Streamlit in one container via `sleep 20`, and forces `v2_adapter.py` to copy-paste endpoint code for evaluation (zero `Locality`).
4. **Cloud Lock-In vs. Non-Functional Tiny Local Models:** `v1` relies on `google/flan-t5-small` (77M parameters, scoring `0.037` Answer Relevancy on Malay synthesis), while `v2` hardcodes Google Gemini and Pinecone Serverless—impossible to run on-premises without rewriting the app.
5. **Brittle Notebook Evaluation:** `03-Evaluation-Framework.ipynb` executes `eval()` on CSV strings and hits repeated RAGAs `IndexError` / `NaN` failures on bilingual outputs, providing zero latency/cost/variance observability and no standalone CLI execution.

---

## 3. Solution Overview

`sovereign-agentic-rag` resolves every bottleneck above through seven locked architectural pillars and three Q&A specifications:

- **Isolated Multi-Tenant `Collection` Workspaces:** Upload `.pdf`, `.md`, and `.txt` documents scoped by `collection_id`, `language`, and `doc_type`, pre-seeded with:
  - **Collection A (`it_infra_sops`):** Enterprise IT & Server Infrastructure SOPs (hardware specs, NVMe/RAID runbooks, deployment guides).
  - **Collection B (`malaysian_enterprise_policy`):** Malaysian Enterprise Policy & Compliance (English, Bahasa Melayu, and BM-EN code-switching).
- **Zero-Cost Dual-Mode Inference (`InferenceMode` Seam):**
  - `LOCAL_ONPREM`: Ollama (`qwen3:4b` default for 16GB RAM edge safety, configurable to `qwen3:8b`) + FastEmbed (`BAAI/bge-m3` / `Qdrant/bm25`) + ONNX Cross-Encoder reranker.
  - `CLOUD_API`: DeepSeek API (`deepseek-chat` / `deepseek-reasoner`) via OpenAI-compatible client ordered specifically for **Prefix Context Caching**.
- **Native Qdrant Hybrid Search + SQLite Async Persistence:** Sub-10ms Qdrant **Dense + Sparse BM25** vectors fused server-side via **Reciprocal Rank Fusion (RRF)** with indexed payload filtering, paired with async SQLite (`aiosqlite` / SQLAlchemy) for `DocumentRecord` lifecycles, `ParentChunk` storage, chat history, and evaluation runs.
- **Structure-Aware Ingestion (`PyMuPDF4LLM` + `ParentChunk`/`ChildChunk` + `BackgroundTasks`):** Preserves Markdown headings, tables, and page numbers; embeds compact `ChildChunk` slices prefixed with `Breadcrumb` paths; hydrates deduplicated `ParentChunk` texts; executes asynchronously via FastAPI `BackgroundTasks` (`PENDING -> INDEXING -> READY | FAILED`).
- **Bounded LangGraph State Machine (`max_retries=2`):** Explicit 6-node graph (`QueryRouter` -> `QueryRewriter` -> `HybridRetriever` -> `CrossEncoderReranker` -> `RelevanceGrader` -> `GroundedGenerator`) that emits `[doc_id:pX]` inline citations validated by `CitationValidator` or terminates in a deterministic `GracefulRefusal`.
- **Typed SSE Streaming & Glass-Box Next.js Workspace:** `POST /api/v1/chat/stream` emits 5 discriminated SSE events (`node_status`, `token`, `citations`, `done`, `error`) driving a real-time **Agent Execution Stepper** and clickable evidence cards.
- **Standalone Evaluation Harness (`eval/`):** Decoupled evaluation toolkit comparing generative **`LLMJudge`** against discriminative **`JEVJudge`** across `golden_dataset.jsonl`, measuring Latency (ms), API Token Cost ($), Score Variance, and `InterJudgeAgreement` (Cohen's Kappa).

---

## 4. User Stories

### Knowledge Worker & End-User
1. As an **Enterprise Knowledge Worker**, I want to select an isolated `Collection` before chatting, so that IT SOP queries never mix with HR compliance policies.
2. As an **IT Infrastructure Engineer**, I want hybrid keyword + semantic search over SOPs, so that exact hardware codes (e.g., `ERR_NVME_0x04`, `RAID-10`) and conceptual troubleshooting steps both retrieve accurately.
3. As a **Malaysian Compliance Officer**, I want to ask questions in English, Bahasa Melayu, or mixed BM-EN code-switching (e.g., *"Apa prosedur tuntutan OT kalau kerja public holiday?"*), so that I receive grounded answers in my matching language.
4. As a **Knowledge Worker**, I want to see live **Agent Execution Stepper** progress while waiting for an answer, so that I understand whether the agent is routing, rewriting, retrieving, reranking, grading, or retrying.
5. As a **Knowledge Worker**, I want tokens streamed in real time as they generate, so that responses feel immediate instead of blocking on a spinner.
6. As a **Compliance Reviewer**, I want clickable `[doc_id:pX]` inline citations beside every factual claim, so that I can inspect the exact source `ParentChunk`, breadcrumb, and page number in one click.
7. As a **Knowledge Worker**, I want the system to refuse politely (`GracefulRefusal`) when my question has no evidence in the selected `Collection`, so that I am never misled by fabricated answers.
8. As a **User**, I want a **Stop Generating** button that immediately aborts the active SSE stream on both client and backend, so that I do not wait for unwanted generations.

### Document Ingestion & Workspace Management
9. As a **Workspace Owner**, I want to upload `.pdf`, `.md`, and `.txt` files via drag-and-drop and immediately receive a `202 Accepted` tracking ID, so that large PDFs do not freeze the browser or cause HTTP timeouts.
10. As a **Workspace Owner**, I want to track live document ingestion status (`PENDING` -> `INDEXING` -> `READY` or `FAILED`), so that I know precisely when a document is searchable.
11. As a **Workspace Owner**, I want duplicate uploads of the exact same file within a `Collection` to be detected via SHA-256 hash, so that my vector index is never polluted with redundant chunks.
12. As a **Portfolio Reviewer**, I want Collection A and Collection B pre-seeded (or seedable in one click), so that I can test bilingual and technical Agentic RAG immediately after `docker compose up`.

### Administrator & Sovereign AI Architect
13. As a **Data-Sovereign Admin**, I want to switch `InferenceMode` between `LOCAL_ONPREM` and `CLOUD_API` in the UI and `.env`, so that sensitive workloads remain 100% air-gapped on Ollama while non-sensitive workloads leverage DeepSeek caching.
14. As an **Edge Workstation Operator (16GB RAM)**, I want `LOCAL_ONPREM` inference to use lightweight FastEmbed (ONNX) and `qwen3:4b` by default, so that the stack runs smoothly without out-of-memory crashes.
15. As a **FinOps / Cloud Admin**, I want `CLOUD_API` prompts structured with a static prefix before dynamic context, so that DeepSeek Context Caching maximizes cache-hit token discounts and reports `cache_hit_tokens` in telemetry.
16. As a **Backend Engineer**, I want interactive OpenAPI/Swagger documentation at `/docs`, `/health`, and `/ready` probes, and structured JSON error envelopes, so that the API is observable and production-ready.

### AI Evaluation Engineer (`eval/`)
17. As an **AI Evaluation Engineer**, I want `eval/` decoupled from `backend/` internals via `RAGTargetAdapter`, so that I can benchmark `sovereign-agentic-rag` or any external HTTP RAG system without modifying evaluator code.
18. As an **AI Evaluation Engineer**, I want a uniform `BaseJudgeEvaluator -> JudgeVerdict` interface comparing `LLMJudge` and `JEVJudge` on the same `golden_dataset.jsonl`, so that comparison metrics are apples-to-apples.
19. As an **AI Evaluation Engineer**, I want `golden_dataset.jsonl` to include `unanswerable_refusal` trap queries alongside bilingual factual and multi-doc comparison queries, so that hallucination resistance is quantitatively tested.
20. As an **AI Evaluation Engineer**, I want multi-run stability testing (`--runs 3`), latency (ms), API cost ($), and `InterJudgeAgreement` (Cohen's Kappa) exported to Markdown and JSON reports, so that I can prove whether fast `$0` discriminative verifiers can replace expensive generative LLM judges.

### Learner / Reverse-Engineer
21. As a **Full-Stack AI Engineer (Amirul)**, I want clean deep modules with concise "why" docstrings and an **"Under the Hood: Reverse-Engineering Guide"** in documentation, so that I master FastAPI DI, SSE, Qdrant RRF, LangGraph reducers, and Docker orchestration without black-box magic.
22. As a **Hiring Manager / Technical Reviewer**, I want a clear engineering progression matrix in `README.md` (`v1 -> v2 -> v3`), so that I can trace how empirical evaluation drove every architectural upgrade.

---

## 5. Locked Implementation Decisions & Edge-Case Mitigations

### 5.1 Core Stack & Q1–Q3 Locked Choices
- **Q1 (`Pilihan A`) — Ingestion:** `PyMuPDF4LLM` layout-aware extraction (`.pdf`, `.md`, `.txt`) -> Two-tier `ParentChunk` (`1200–1500` tokens in SQLite) / `ChildChunk` (`300–400` tokens, `50`-token overlap, prepended `Breadcrumb` in Qdrant) -> Async execution via FastAPI `BackgroundTasks` + `asyncio.to_thread` + polling `GET /api/v1/documents/{doc_id}/status`.
- **Q2 (`Pilihan A`) — SSE Contract:** Typed discriminated SSE envelopes using standard `event:` + JSON `data:` lines for five event types: `node_status`, `token`, `citations`, `done`, and `error`.
- **Q3 (`Pilihan A`) — Evaluation Harness:** Standalone `eval/` package querying `backend/` over HTTP via `RAGTargetAdapter`, grading via `BaseJudgeEvaluator -> JudgeVerdict` (`faithfulness`, `answer_relevancy`, `context_recall`, `PASS/FAIL`, latency, token cost), running against `eval/datasets/golden_dataset.jsonl`.

### 5.2 Mandatory Production Edge-Case Mitigations (Bahagian 1)
1. **16GB RAM OOM Prevention (`LOCAL_ONPREM`):** Run embeddings via FastEmbed (Rust/ONNX, `<500MB` RAM) instead of PyTorch `sentence-transformers` in `backend/`; default Ollama LLM to `qwen3:4b` (Q4_K_M, configurable to `qwen3:8b`) with `OLLAMA_KEEP_ALIVE=5m` and `OLLAMA_NUM_PARALLEL=1`; keep `JEVJudge` verifier weights isolated inside the standalone `eval/` CLI process.
2. **Control-Node Latency Optimization:** Suppress `<think>` chain-of-thought reasoning tokens (`/no_think` for Qwen3; `deepseek-chat` at `temperature=0.0` with JSON mode) on `QueryRouter`, `QueryRewriter`, and `RelevanceGrader`. Stream full tokens only on `GroundedGenerator`.
3. **DeepSeek Prefix Context Caching (`CLOUD_API`):** Construct chat messages strictly static-to-dynamic: (1) Static System Prompt + Citation Rules + Few-Shot Examples ($\ge 1,024$ tokens), (2) Collection Metadata Header, (3) Append-only Conversation History, (4) Deterministically sorted `ParentChunk` blocks (`sorted by doc_id, page_number`) + Current Question.
4. **Citation Format Drift & Phantom Citations (`CitationValidator`):** Label prompt sources as `<<<SOURCE [doc_id:pX]>>> ... <<<END_SOURCE>>>`, normalize bracket variants via regex, and strip any `[doc_id:pX]` absent from the retrieved `ParentChunk` set.
5. **Idempotent Upload Deduplication:** Compute `sha256(file_bytes)` on upload; if `(collection_id, content_hash)` exists with `status="READY"`, return the existing `DocumentRecord` without re-indexing.
6. **Bounded Retry Exhaustion -> `GracefulRefusal`:** After `retry_count == 2` with zero relevant parents, bypass speculative LLM generation and emit a deterministic bilingual refusal matching the query language with `citations: []`.
7. **Zero-Friction Cold-Start Seeding:** Ship pre-curated documents for Collection A and Collection B in `data/seed_collections/` plus `seed_demo_collections.py` and `POST /api/v1/collections/seed`.
8. **SSE Anti-Buffering & Disconnect Propagation:** Return `X-Accel-Buffering: no`, `Cache-Control: no-cache, no-transform`, and `Connection: keep-alive`; check `await request.is_disconnected()` inside the FastAPI generator wired to Next.js `AbortController`.

---

## 6. Testing & Verification Decisions

- **TDD Discipline:** Every production module is built test-first (`Red -> Verify Red -> Green -> Verify Green -> Refactor -> Commit`).
- **Interface-Level Testing:** Tests exercise each module through its public `Interface` using real logic and lightweight in-memory `Adapter` fakes (e.g., `InMemoryVectorRepository`, `FakeLLMProvider`) for fast deterministic CI without requiring live Ollama or DeepSeek keys.
- **Integration Verification:** Dedicated contract tests validate SSE event sequencing (`node_status -> token -> citations -> done`), SHA-256 upload idempotency, Qdrant RRF payload isolation (when Docker Qdrant is running), and `eval/` report generation.

---

## 7. Out of Scope

- Paid external vector databases (Pinecone, Weaviate Cloud) or proprietary closed APIs beyond the optional `CLOUD_API` DeepSeek adapter.
- Public internet web search fallbacks (e.g., Tavily API) that compromise air-gapped data sovereignty.
- Heavy GPU-bound vision OCR pipelines (e.g., `Docling` deep layout models) that violate the 16GB RAM local workstation budget.
- Multi-user OAuth/RBAC identity servers (workspace isolation is enforced cleanly at the `collection_id` metadata layer).
