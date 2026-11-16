# System Architecture & Reverse-Engineering Guide: `sovereign-agentic-rag`

> **Audience:** Engineers implementing, reviewing, or reverse-engineering every layer of `sovereign-agentic-rag`.  
> **Vocabulary:** Canonical terms follow [GLOSSARY.md](../GLOSSARY.md).

---

## 1. End-to-End Agentic RAG Pipeline & System Architecture

`sovereign-agentic-rag` combines a **bounded 6-node LangGraph agent loop** with four decoupled infrastructure tiers orchestrated via Docker Compose at **$0 infrastructure cost**.

### 1.1 End-to-End Agentic Pipeline: User Query to Streamed Response

Unlike `v1`/`v2`'s linear one-shot pipe, every query traverses an autonomous decision, retrieval, reranking, self-reflective grading, and bounded self-correction loop:

```mermaid
flowchart TD
    User(["1. User Query (EN / BM / BM-EN)<br/>+ collection_id + InferenceMode"]) -->|"POST /api/v1/chat/stream"| Router["2. QueryRouter (/no_think JSON)<br/>Detect Language & Classify Intent"]

    Router -->|"route == direct<br/>(Greeting / Meta)"| Gen
    Router -->|"single_collection / multi_doc_compare"| Rewriter["3. QueryRewriter (/no_think JSON)<br/>Bilingual BM<->EN & Acronym Expansion"]

    Rewriter --> Retriever["4. HybridRetriever<br/>FastEmbed Dense (bge-m3) + Sparse (bm25)"]
    Retriever --> Qdrant[("Qdrant Server-Side RRF<br/>Pre-Filter: collection_id")]
    Qdrant -->|"Matched ChildChunk parent_ids"| SQLiteHydrate[("SQLite ParentChunk Hydration<br/>Full Tables, Clauses & Page Numbers")]
    SQLiteHydrate --> Reranker["5. CrossEncoderReranker<br/>ONNX bge-reranker-v2-m3 (Top 5)"]
    Reranker --> Grader{"6. RelevanceGrader (/no_think)<br/>Self-Reflective Evidence Gate"}

    Grader -->|"Insufficient & retry_count < 2<br/>(Agentic Self-Correction Loop)"| Rewriter
    Grader -->|"Insufficient & retry_count == 2"| Refusal["Deterministic GracefulRefusal<br/>(Bilingual Refusal, citations: [])"]
    Grader -->|"Grounded Evidence Approved"| Gen["7. GroundedGenerator<br/>Static-to-Dynamic Prompt (>=1,024 tok prefix)"]

    Gen -->|"LOCAL_ONPREM"| LocalLLM["Ollama / HF GGUF<br/>(Qwen3.8 Quantized Series)"]
    Gen -->|"CLOUD_API"| CloudLLM["DeepSeek API<br/>(Prefix Context Caching)"]
    LocalLLM --> Validator["8. CitationValidator<br/>Normalize [doc_id:pX] & Strip Phantoms"]
    CloudLLM --> Validator

    Validator --> SSE["9. Typed SSE Stream -> Next.js UI<br/>node_status | token | citations | done | error"]
    Refusal --> SSE
    SSE --> UI(["Next.js Glass-Box Workspace<br/>Live Agent Stepper + Markdown + Clickable Citations"])
```

### ASCII End-to-End Pipeline Trace (User Query $\rightarrow$ Agentic Loop $\rightarrow$ User Response)

```text
[USER QUERY IN NEXT.JS]  ("Apa prosedur tuntutan OT kalau kerja public holiday?")
   │  POST /api/v1/chat/stream  (collection_id + InferenceMode + AbortController)
   ▼
[FASTAPI SSE ROUTER] ──► Opens HTTP 200 text/event-stream (X-Accel-Buffering: no)
   │
   ▼
┌──────────────────────────────────── LANGGRAPH BOUNDED AGENT LOOP (max_retries=2) ────────────────────────────────────┐
│                                                                                                                       │
│  [1. QueryRouter] (/no_think JSON) ──► emits SSE `node_status`                                                        │
│       ├─ route == "direct" ─────────────────────────────────────────────────────────────┐                             │
│       └─ route == "single_collection" | "multi_doc_compare"                             │                             │
│            ▼                                                                            │                             │
│  [2. QueryRewriter] (/no_think JSON) ◄────────────────────────┐                         │                             │
│       │  Expands BM <-> EN & acronyms (OT -> overtime /       │                         │                             │
│       │  kerja lebih masa); shifts query terms on retry       │                         │                             │
│       ▼                                                       │                         │                             │
│  [3. HybridRetriever] ──► FastEmbed (bge-m3 Dense + bm25)     │                         │                             │
│       │                   ──► Qdrant RRF (collection_id)      │  AGENTIC SELF-CORRECTION│                             │
│       │                   ──► SQLite ParentChunk Hydration    │  (retry_count < 2)      │                             │
│       ▼                                                       │                         │                             │
│  [4. CrossEncoderReranker] ──► ONNX bge-reranker-v2-m3 Top-5  │                         │                             │
│       ▼                                                       │                         │                             │
│  [5. RelevanceGrader] (/no_think JSON) ──► Sufficient? ───────┘ (NO & retries < 2)      │                             │
│       ├─ NO & retry_count == 2 ──► [Deterministic GracefulRefusal] (citations: []) ─────┼──► SSE `token` + `done`     │
│       └─ YES (Evidence Approved) ───────────────────────────────────────────────────────┘                             │
│            ▼                                                                                                          │
│  [6. GroundedGenerator] ──► Static Prefix (>=1,024 tok) + History + Sorted Parents + Question                         │
│       ├─ LOCAL_ONPREM: Ollama / llama.cpp (Qwen3.8 quantized GGUF / free source)                                      │
│       └─ CLOUD_API:    DeepSeek API (deepseek-chat, Prefix Context Caching)                                           │
│            ▼  streams `event: token` + runs regex `CitationValidator` ([doc_id:pX])                                   │
└───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
   │
   ▼  emits `event: citations` + `event: done` (cache_hit_tokens, latency_ms) & persists to SQLite
[NEXT.JS WORKSPACE UI]  Renders Live Agent Stepper + Streaming Answer + Clickable [doc_id:pX] Evidence Drawer
```

### 1.2 Container & Persistence Topology (4 Decoupled Tiers)

```mermaid
flowchart TB
    subgraph Client["Frontend Tier (Next.js App Router + TypeScript)"]
        UI["Workspace UI (3-Pane Glass-Box)<br/>Collection Selector | Mode Switcher | Agent Stepper | Citation Cards"]
    end

    subgraph Backend["Backend Tier (Layered FastAPI + Pydantic v2)"]
        Router["API v1 Routers (/chat/stream, /collections, /documents, /health, /ready)"]
        Ingest["DocumentIngestionService (PyMuPDF4LLM + Parent-Child + SHA-256)"]
        Graph["AgenticRAGWorkflow (LangGraph State Machine, max_retries=2)"]
        Adapters["Inference Seam (LLMProvider & EmbeddingProvider)"]
        Router --> Ingest
        Router --> Graph
        Graph --> Adapters
    end

    subgraph Data["Persistence Tier (Docker Volumes)"]
        Qdrant[("Qdrant Vector DB<br/>Named Dense + Sparse BM25<br/>Server-Side RRF + Payload Filters")]
        SQLite[("SQLite (aiosqlite)<br/>Collections, DocumentRecords,<br/>ParentChunks, ChatHistory")]
    end

    subgraph Inference["Dual-Mode Inference Providers"]
        Local["LOCAL_ONPREM (Air-Gapped)<br/>Ollama / HF GGUF (Qwen3.8 Quantized Series)<br/>FastEmbed (bge-m3 / bm25) + ONNX Reranker"]
        Cloud["CLOUD_API (Cache-Optimized)<br/>DeepSeek API (deepseek-chat)<br/>Static-to-Dynamic Prefix Caching"]
    end

    subgraph Eval["Decoupled Evaluation Sub-Project (eval/)"]
        Harness["CLI Runner + RAGTargetAdapter<br/>LLMJudge vs JEVJudge -> Report"]
    end

    UI <-->|"REST + Typed SSE (POST /api/v1/chat/stream)"| Router
    Ingest --> Qdrant
    Ingest --> SQLite
    Graph --> Qdrant
    Graph --> SQLite
    Adapters -->|"LOCAL_ONPREM"| Local
    Adapters -->|"CLOUD_API"| Cloud
    Harness -->|"HTTP Contract"| Router
```

---

## 2. Repository Directory Structure

```text
sovereign-agentic-rag/
├── AGENTS.md                           # Strict rules, CLI commands, and TDD workflow
├── GLOSSARY.md                         # Canonical domain & deep-module vocabulary
├── README.md                           # Portfolio overview & v1/v2 -> v3 progression
├── docker-compose.yml                  # Orchestrates frontend, backend, qdrant, ollama profile
├── .env.example                        # Template environment variables (zero secrets)
├── .gitignore
├── data/
│   └── seed_collections/
│       ├── it_infra_sops/              # Collection A: IT & Server Infrastructure SOPs
│       └── malaysian_enterprise_policy/# Collection B: MY Policy & Compliance (EN/BM/Mixed)
├── docs/
│   ├── PRD.md                          # Product requirements, user stories, locked specs
│   ├── Architecture.md                 # This document + Under the Hood guide
│   ├── Design-System.md                # Next.js UI tokens, 3-pane layout, stepper specs
│   ├── Eval-Subproject.md              # LLM-as-a-Judge vs JEV-as-a-Judge design
│   └── superpowers/
│       └── plans/
│           └── 2026-11-16-sovereign-agentic-rag.md
├── backend/
│   ├── Dockerfile                      # Multi-stage slim Python 3.11+ build
│   ├── pyproject.toml
│   ├── app/
│   │   ├── main.py                     # FastAPI factory, lifespan, CORS, Swagger /docs
│   │   ├── core/
│   │   │   ├── config.py               # Pydantic BaseSettings (.env & runtime overrides)
│   │   │   ├── exceptions.py           # Structured JSON error envelope & handlers
│   │   │   └── dependencies.py         # FastAPI Depends() wiring (no import side effects)
│   │   ├── schemas/                    # Pydantic v2 request/response & SSE schemas
│   │   ├── repositories/
│   │   │   ├── sqlite_repo.py          # Async SQLAlchemy/aiosqlite persistence
│   │   │   └── qdrant_repo.py          # Qdrant Hybrid (Dense+Sparse RRF) + InMemoryFake
│   │   ├── providers/
│   │   │   ├── base.py                 # LLMProvider, EmbeddingProvider, RerankerProvider
│   │   │   ├── local_onprem.py         # Ollama + FastEmbed (ONNX) + ONNX Cross-Encoder
│   │   │   └── cloud_deepseek.py       # DeepSeek API with Prefix Context Caching layout
│   │   ├── services/
│   │   │   ├── ingestion.py            # PyMuPDF4LLM, Parent-Child + Breadcrumb, SHA-256
│   │   │   ├── chat_service.py         # Orchestrates session history + LangGraph SSE stream
│   │   │   └── seeder.py               # Idempotent Collection A & B seeding
│   │   ├── agents/
│   │   │   ├── state.py                # RAGGraphState TypedDict/Pydantic definition
│   │   │   ├── nodes.py                # 6 LangGraph nodes + CitationValidator + Refusal
│   │   │   └── graph.py                # Compiled bounded StateGraph (max_retries=2)
│   │   └── api/
│   │       └── v1/routers/             # health.py, collections.py, documents.py, chat.py
│   └── tests/                          # Unit + SSE contract + ingestion integration tests
├── frontend/
│   ├── Dockerfile                      # Multi-stage Node 20 standalone Next.js build
│   ├── package.json
│   └── src/
│       ├── app/                        # App Router pages & layout
│       ├── components/                 # Sidebar, ModeSwitch, AgentStepper, ChatPanel, Citations
│       ├── lib/                        # Typed SSE fetch+ReadableStream client & API client
│       └── types/                      # TypeScript mirrors of FastAPI Pydantic schemas
└── eval/
    ├── pyproject.toml                  # Standalone dependencies (isolated from backend RAM)
    ├── datasets/
    │   └── golden_dataset.jsonl        # Bilingual EN/BM/Mixed + Refusal test cases
    ├── eval_harness/
    │   ├── schemas.py                  # EvalSample, RAGOutput, JudgeVerdict, BenchmarkReport
    │   ├── target_adapter.py           # RAGTargetAdapter (HTTP SSE/REST client)
    │   ├── judges/
    │   │   ├── base.py                 # BaseJudgeEvaluator seam
    │   │   ├── llm_judge.py            # Generative LLMJudgeEvaluator (DeepSeek / Qwen3.8)
    │   │   └── jev_judge.py            # Discriminative JEVJudgeEvaluator (Local NLI/Verifier)
    │   ├── metrics.py                  # Variance, latency percentiles, cost, Cohen's Kappa
    │   └── runner.py                   # CLI entrypoint comparing LLM vs JEV side-by-side
    ├── reports/                        # Generated .json and .md comparison reports
    └── tests/
```

---

## 3. Deep-Module Boundaries & Seams (`codebase-design`)

Each core `Seam` hides complex implementation behind a small `Interface` and is backed by at least two real `Adapter` implementations (or production + deterministic test fake):

| `Seam` (`Module`) | Minimal `Interface` | `Adapter` Implementations | `Leverage` & `Locality` Payoff |
| :--- | :--- | :--- | :--- |
| **`LLMProvider`** | `generate_structured(messages, schema)` & `stream_answer(messages)` | 1. `LocalOnPremLLMAdapter` (Ollama)<br>2. `CloudDeepSeekLLMAdapter` (DeepSeek)<br>3. `FakeLLMProvider` (tests) | Swapping `LOCAL_ONPREM` $\leftrightarrow$ `CLOUD_API` requires zero changes in LangGraph nodes or routers. |
| **`EmbeddingProvider`** | `embed_dense(texts)` & `embed_sparse(texts)` | 1. `FastEmbedProvider` (ONNX `bge-m3` / `Qdrant/bm25`)<br>2. `FakeEmbeddingProvider` (tests) | Prevents PyTorch OOM on 16GB RAM and decouples vector math from ingestion and retrieval. |
| **`VectorRepository`** | `upsert_chunks(chunks)`, `hybrid_search(collection_id, dense, sparse, filters, top_k)`, `delete_document(doc_id)` | 1. `QdrantHybridRepository`<br>2. `InMemoryVectorRepository` (tests) | Isolates Qdrant RRF prefetch syntax and multi-tenant filtering inside one testable module. |
| **`DocumentIngestionService`** | `enqueue_upload(collection_id, file)` & `process_document(doc_id)` | Single deep service orchestrating `PyMuPDF4LLM` + `HierarchicalChunker` + SHA-256 deduplicator | Guarantees identical `ParentChunk`/`ChildChunk` processing for UI uploads and demo seeding. |
| **`BaseJudgeEvaluator`** | `evaluate(sample: EvalSample, output: RAGOutput) -> JudgeVerdict` | 1. `LLMJudgeEvaluator` (DeepSeek/Qwen3.8)<br>2. `JEVJudgeEvaluator` (local NLI/Cross-Encoder) | Allows apples-to-apples latency, cost, variance, and Cohen's Kappa benchmarking. |

---

## 4. Data Schemas & Storage Design

### 4.1 SQLite Schema (`aiosqlite` + SQLAlchemy 2.0 Async)

1. **`collections`**:
   - `id` (`TEXT PRIMARY KEY`, e.g., `it_infra_sops`), `name`, `description`, `is_seeded` (`BOOLEAN`), `created_at` (`TIMESTAMP`).
2. **`documents`**:
   - `doc_id` (`TEXT PRIMARY KEY`), `collection_id` (`FK -> collections.id`), `filename`, `doc_type` (`pdf` | `md` | `txt`), `language` (`en` | `ms` | `mixed`), `content_hash` (`TEXT` SHA-256), `page_count` (`INTEGER`), `chunk_count` (`INTEGER`), `status` (`PENDING` | `INDEXING` | `READY` | `FAILED`), `error_message` (`TEXT NULL`), `created_at`, `updated_at`.
   - **Unique Index:** `(collection_id, content_hash)` enforces idempotent upload deduplication.
3. **`parent_chunks`**:
   - `parent_id` (`TEXT PRIMARY KEY`, e.g., `{doc_id}_p{page}_s{idx}`), `doc_id` (`FK -> documents.doc_id ON DELETE CASCADE`), `collection_id` (`TEXT INDEXED`), `page_number` (`INTEGER`), `section_header` (`TEXT`), `breadcrumb` (`TEXT`), `content` (`TEXT`), `token_count` (`INTEGER`).
4. **`chat_sessions` & `chat_messages`**:
   - `chat_sessions`: `session_id` (`PK`), `collection_id` (`FK`), `title`, `created_at`.
   - `chat_messages`: `message_id` (`PK`), `session_id` (`FK`), `role` (`user` | `assistant`), `content` (`TEXT`), `citations_json` (`TEXT`), `inference_mode` (`TEXT`), `agent_trace_json` (`TEXT`), `cache_hit_tokens` (`INTEGER`), `latency_ms` (`INTEGER`), `created_at`.

### 4.2 Qdrant Hybrid Collection Schema (`sovereign_rag_chunks`)

- **Named Dense Vector (`"dense"`):** Size matched to `BAAI/bge-m3` (`1024`-dim, Cosine distance) or configured FastEmbed dense model.
- **Named Sparse Vector (`"sparse"`):** `SparseVectorParams(modifier=Modifier.IDF)` populated by `Qdrant/bm25` or `bge-m3` lexical weights.
- **Indexed Payload Fields (for sub-10ms pre-filtering):**
  - `collection_id` (`keyword` — mandatory tenant filter on every query)
  - `doc_id` (`keyword`)
  - `parent_id` (`keyword` — used to hydrate full `ParentChunk` from SQLite)
  - `language` (`keyword` — `en` | `ms` | `mixed`)
  - `doc_type` (`keyword` — `pdf` | `md` | `txt`)
  - `page_number` (`integer`)
- **Hybrid Query Execution:** Uses Qdrant's native `query_points` with two `Prefetch` branches (`using="dense"`, `limit=25` and `using="sparse"`, `limit=25`, both constrained by `Filter(must=[FieldCondition(key="collection_id", ...)])`), fused via `FusionQuery(fusion=Fusion.RRF)`.

---

## 5. Bounded LangGraph State Machine (`max_retries=2`)

```mermaid
stateDiagram-v2
    [*] --> QueryRouter
    QueryRouter --> GroundedGenerator: route == "direct" (greeting / meta)
    QueryRouter --> QueryRewriter: route == "single_collection" | "multi_doc_compare"
    QueryRewriter --> HybridRetriever
    HybridRetriever --> CrossEncoderReranker
    CrossEncoderReranker --> RelevanceGrader
    RelevanceGrader --> GroundedGenerator: relevant_parents_found == true
    RelevanceGrader --> QueryRewriter: relevant_parents_found == false AND retry_count < 2
    RelevanceGrader --> GracefulRefusal: relevant_parents_found == false AND retry_count >= 2
    GroundedGenerator --> [*]
    GracefulRefusal --> [*]
```

### `RAGGraphState` Fields
- `question: str` — Original user query.
- `collection_id: str` — Mandatory tenant filter.
- `chat_history: list[dict[str, str]]` — Prior conversation turns.
- `inference_mode: Literal["LOCAL_ONPREM", "CLOUD_API"]`
- `detected_language: Literal["en", "ms", "mixed"]`
- `route: Literal["direct", "single_collection", "multi_doc_compare"]`
- `rewritten_queries: list[str]` — Bilingual/acronym expansions (e.g., `OT` -> `overtime / kerja lebih masa`).
- `candidate_parents: list[ParentEvidence]` — Hydrated deduplicated `ParentChunk` records from Qdrant `ChildChunk` hits.
- `reranked_parents: list[ParentEvidence]` — Top-5 parents scored by local ONNX Cross-Encoder.
- `relevant_parents: list[ParentEvidence]` — Parents approved by `RelevanceGrader`.
- `retry_count: int` — Strictly bounded `[0, 2]`.
- `answer: str` & `validated_citations: list[CitationItem]`
- `refusal_triggered: bool`

### Node Responsibilities
1. **`QueryRouter`:** Classifies intent (`direct`, `single_collection`, `multi_doc_compare`) and detects language (`en`/`ms`/`mixed`). Uses `/no_think` + JSON mode for `<300ms` latency.
2. **`QueryRewriter`:** Expands acronyms and generates bilingual keyword variants (BM $\leftrightarrow$ EN). On `retry_count > 0`, shifts focus away from failed terms. Uses `/no_think`.
3. **`HybridRetriever`:** Computes dense + sparse query vectors, executes Qdrant RRF search scoped by `collection_id`, fetches unique `parent_id`s, and hydrates full `ParentChunk` content from SQLite.
4. **`CrossEncoderReranker`:** Scores `(question, parent.content)` pairs using local ONNX Cross-Encoder (`BAAI/bge-reranker-v2-m3`) and retains top `N=5` candidates.
5. **`RelevanceGrader`:** Self-reflective check verifying whether the top candidates actually contain evidence to answer `question` (`/no_think` structured JSON). If insufficient and `retry_count < 2`, increments `retry_count` and loops back to `QueryRewriter`. If `retry_count >= 2`, routes to deterministic `GracefulRefusal`.
6. **`GroundedGenerator` + `CitationValidator`:** Streams the answer citing sources exclusively as `[doc_id:pX]`. `CitationValidator` normalizes format drift and strips phantom citations not present in `relevant_parents`.

---

## 6. FastAPI REST & SSE API Contract

### 6.1 Endpoints
| Method & Path | Purpose | Status Codes |
| :--- | :--- | :--- |
| `GET /health` | Liveness probe (process alive) | `200 OK` |
| `GET /ready` | Readiness probe (verifies SQLite, Qdrant & active LLM provider reachability) | `200 OK`, `503 Service Unavailable` |
| `GET /api/v1/collections` | List all collections with document counts | `200 OK` |
| `POST /api/v1/collections` | Create a new isolated collection workspace | `201 Created`, `409 Conflict` |
| `POST /api/v1/collections/seed` | Idempotently ingest pre-curated Collection A & B docs | `202 Accepted` |
| `GET /api/v1/collections/{id}/documents` | List documents and statuses in a collection | `200 OK`, `404 Not Found` |
| `POST /api/v1/collections/{id}/documents` | Upload `.pdf`/`.md`/`.txt`, deduplicate via SHA-256, trigger `BackgroundTasks` | `202 Accepted`, `200 OK` (duplicate), `415 Unsupported Media Type` |
| `GET /api/v1/documents/{doc_id}/status` | Poll ingestion lifecycle (`PENDING -> INDEXING -> READY | FAILED`) | `200 OK`, `404 Not Found` |
| `POST /api/v1/chat/stream` | Execute LangGraph workflow and stream typed SSE events | `200 OK` (`text/event-stream`) |
| `GET /docs` | Interactive Swagger/OpenAPI documentation | `200 OK` |

### 6.2 Structured HTTP Error Envelope
All non-streaming errors return a uniform Pydantic envelope:
```json
{
  "error": {
    "code": "COLLECTION_NOT_FOUND",
    "message": "Collection 'unknown_ws' does not exist.",
    "detail": null,
    "request_id": "req_8f2a9c"
  }
}
```

### 6.3 Typed Discriminated SSE Event Contract (`POST /api/v1/chat/stream`)
Headers: `Content-Type: text/event-stream`, `Cache-Control: no-cache, no-transform`, `X-Accel-Buffering: no`.

```text
event: node_status
data: {"node": "QueryRouter", "status": "completed", "attempt": 0, "latency_ms": 145, "detail": "route=single_collection lang=mixed"}

event: node_status
data: {"node": "HybridRetriever", "status": "completed", "attempt": 1, "latency_ms": 9, "detail": "14 child hits -> 5 unique parents"}

event: node_status
data: {"node": "RelevanceGrader", "status": "completed", "attempt": 1, "latency_ms": 210, "detail": "2 relevant parents approved"}

event: token
data: {"delta": "Menurut polisi HR "}

event: citations
data: {"sources": [{"citation_key": "[hr_policy_my:p12]", "doc_id": "hr_policy_my", "filename": "Malaysian_HR_Policy.pdf", "page": 12, "score": 0.91, "breadcrumb": "Malaysian_HR_Policy.pdf | Page 12 > Overtime Claims", "snippet": "Klausa 4.2: Kerja pada hari kelepasan am..."}]}

event: done
data: {"message_id": "msg_901", "session_id": "sess_12", "mode": "CLOUD_API", "retries_used": 0, "cache_hit_tokens": 1280, "total_latency_ms": 1640}

event: error
data: {"code": "OLLAMA_UNAVAILABLE", "message": "Local Ollama daemon unreachable on port 11434.", "recoverable": true}
```

---

## 7. The 8 Production Edge-Case Mitigations

1. **16GB RAM Budget (`LOCAL_ONPREM` OOM Prevention):**
   - Budget breakdown on 16GB edge workstation: OS/Docker/Qdrant/Next.js (~5.5 GB) + FastEmbed ONNX (`BAAI/bge-m3` + `Qdrant/bm25`, <0.5 GB) + ONNX Cross-Encoder (<0.6 GB) = ~6.6 GB baseline, leaving **~9.4 GB headroom** for the **`Qwen3.8` series** quantized model configured via `LOCAL_LLM_MODEL` (e.g., Ollama pulling low-bit Hugging Face / Unsloth Dynamic v3.1 GGUFs like `hf.co/unsloth/Qwen3.8-27B-GGUF` small quants, `Qwen3.8-Flash-Next-GGUF`, compact quantized `qwen3.8` releases, or a free OpenAI-compatible `Qwen3.8` endpoint when running on a 16GB CPU-only laptop; larger quants like `Q4_K_M` ~14–17GB are seamlessly supported on 24GB+ workstations via `.env`). Configured with `OLLAMA_KEEP_ALIVE=5m` and `OLLAMA_NUM_PARALLEL=1`. `eval/` runs in a separate CLI process so `JEVJudge` weights never occupy `backend/` RAM.
2. **Disabling `<think>` Overhead on Control Nodes:**
   - `QueryRouter`, `QueryRewriter`, and `RelevanceGrader` disable reasoning/thinking overhead (`/no_think` or `reasoning_effort="none"` / `think=false` for `Qwen3.8` via Ollama, and `deepseek-chat` with `temperature=0.0` + `response_format={"type": "json_object"}` for DeepSeek), eliminating multi-second chain-of-thought stalls before retrieval even begins.
3. **DeepSeek Prefix Context Caching (`CLOUD_API`):**
   - Messages are ordered strictly **static-to-dynamic**:
     1. Static System Instruction + Citation Syntax Rules + Bilingual Few-Shot Examples ($\ge 1,024$ tokens, byte-identical across requests).
     2. Static Collection Descriptor.
     3. Prior Chat History (append-only prefix across turns).
     4. Deterministically sorted `ParentChunk` entries (`sorted(parents, key=lambda p: (p.doc_id, p.page_number, p.parent_id))`) + Current User Question.
4. **`CitationValidator` Guardrail:**
   - Prompt wraps chunks in `<<<SOURCE [doc_id:pX]>>> ... <<<END_SOURCE>>>`. A regex post-processor normalizes variations (`[doc_id, p. X]`, `[doc_id:X]`) into `[doc_id:pX]` and strips any citation key not present in the session's `relevant_parents` map.
5. **Idempotent SHA-256 Upload Deduplication:**
   - `sha256(file_bytes)` is checked against `(collection_id, content_hash)` in SQLite. Duplicate uploads return the existing `DocumentRecord` (`status="READY"`) without creating duplicate Qdrant vectors.
6. **Bounded Retry Exhaustion -> `GracefulRefusal`:**
   - When `retry_count == 2` ends with empty `relevant_parents`, the workflow skips generative speculation and emits a fixed bilingual refusal matching `detected_language` with `citations: []`.
7. **Zero-Friction Cold-Start Seeding:**
   - `data/seed_collections/` + `POST /api/v1/collections/seed` populates Collection A and Collection B automatically on startup or button click.
8. **SSE Anti-Buffering & Client Disconnect Propagation:**
   - Explicit anti-buffering response headers (`X-Accel-Buffering: no`, `Cache-Control: no-cache, no-transform`) paired with `await request.is_disconnected()` polling inside the FastAPI generator wired to Next.js `AbortController`.

---

## 8. Under the Hood: Reverse-Engineering Guide (Panduan Belajar Bahasa Melayu + EN)

Bahagian ini direka khas supaya anda boleh membedah (*reverse-engineer*) mekanik sebenar di sebalik setiap teknologi tanpa sebarang *black-box*.

### 8.1 FastAPI Dependency Injection (`Depends`) & `BackgroundTasks`
- **Dalam `v2`:** Anda menulis `retriever_model = SentenceTransformer(...)` terus di baris atas `main.py`. Kesannya, setiap kali fail di-import (termasuk sewaktu `pytest`), Python terus memulakan model ke dalam RAM.
- **Dalam `v3` (`lifespan` + `Depends`):**
  1. FastAPI mencipta semua client berat (SQLite engine, Qdrant client, FastEmbed session) **sekali sahaja** di dalam `app.state` semasa fungsi `@asynccontextmanager async def lifespan(app)` apabila server bermula.
  2. Setiap *route* menerima objek tersebut melalui parameter `service: ChatService = Depends(get_chat_service)`. Semasa unit test, `app.dependency_overrides[get_chat_service]` menukar servis sebenar kepada versi ujian dalam memori dengan satu baris kod!
  3. Untuk muat naik PDF, `background_tasks.add_task(ingestion_service.process_document, doc_id)` memulangkan HTTP `202 Accepted` dalam `<50ms`, manakala tugas membaca PDF & menjana vektor berjalan di dalam `asyncio.to_thread(...)` supaya *event loop* utama kekal bebas melayani chat pengguna lain.

### 8.2 Matematik Di Sebalik Qdrant Hybrid Search (Dense + Sparse + RRF)
Mengapa gabungan **Dense + Sparse BM25** jauh lebih baik daripada carian vector biasa?
- **Dense Vector (`BAAI/bge-m3`):** Menukar ayat menjadi senarai 1,024 nombor apungan. Hebat untuk makna semantik (contoh: *"kerosakan cakera keras"* masih dapat mencari *"NVMe drive failure"*), tetapi lemah pada kod spesifik seperti `ERR_NVME_0x04` atau `Seksyen 60D`.
- **Sparse Vector (`Qdrant/bm25`):** Menyimpan berat kata kunci tepat (berdasarkan kekerapan perkataan dan keunikannya merentasi dokumen / IDF). Sangat tajam untuk nombor model, kod ralat, dan istilah undang-undang.
- **Reciprocal Rank Fusion (RRF):** Bagaimana Qdrant menggabungkan kedua-dua senarai tanpa perlu menormalisasi skala skor yang berbeza? Formula RRF hanya melihat **kedudukan (*rank*)** dokumen dalam setiap senarai:
  $$RRF\_Score(d) = \frac{1}{k + rank_{dense}(d)} + \frac{1}{k + rank_{sparse}(d)} \quad (\text{nilai lalai } k = 60)$$
  *Contoh:* Jika Dokumen X berada di tempat ke-2 dalam Dense dan tempat ke-1 dalam Sparse, skornya ialah $\frac{1}{62} + \frac{1}{61} = 0.0325$. Dokumen yang konsisten tinggi di kedua-dua senarai secara automatik naik ke puncak!

### 8.3 LangGraph State Machine & Event Streaming
- Berbeza dengan *chain* linear `v2`, LangGraph menyimpan satu objek keadaan berpusat (`RAGGraphState`) yang mengalir dari satu nod ke nod lain.
- Setiap nod hanyalah satu fungsi Python yang menerima `state` semasa dan memulangkan kamus (*dict*) medan yang ingin dikemas kini sahaja (contoh: `{"retry_count": state["retry_count"] + 1}`).
- Apabila FastAPI memanggil `graph.astream_events(initial_state, version="v2")`, LangGraph memancarkan acara dalaman (`on_chain_start`, `on_chain_end`, `on_chat_model_stream`) yang terus diterjemahkan oleh FastAPI kepada event SSE `node_status` dan `token`.

### 8.4 Bagaimana DeepSeek Prefix Context Caching Menjimatkan Kos Hingga ~90%
- Enjin inferens DeepSeek menyimpan *KV Cache* (ingatan matematik *attention* bagi token yang telah diproses) untuk bahagian awal (*prefix*) prompt di dalam memori GPU pelayan mereka.
- Jika panggilan API kedua mempunyai **awalan bait yang 100% sama** ($\ge 64$ token/blok, sasaran kita $\ge 1,024$ token statik di atas), pelayan DeepSeek tidak perlu mengira semula token tersebut dan mengenakan caj *cache hit* jauh lebih murah, di samping memulangkan statistik `prompt_cache_hit_tokens` dalam objek `usage` untuk dipaparkan di UI!

### 8.5 Rangkaian Docker Compose (`docker-compose.yml`)
- Di dalam Docker Compose, setiap servis (`frontend`, `backend`, `qdrant`, `ollama`) mendapat nama DNS dalaman mengikut nama servisnya.
- Sebab itulah `backend` menghubungi Qdrant di `http://qdrant:6333` (bukan `localhost:6333`, kerana `localhost` di dalam kontena merujuk kepada kontena backend itu sendiri!).
- Kualiti permulaan dikawal oleh `depends_on: qdrant: condition: service_healthy` supaya backend tidak *crash* sebelum Qdrant siap dibuka — menggantikan teknik `sleep 20` yang rapuh dalam `v2` (`setup.sh`).
