# Sovereign Agentic RAG (`sovereign-agentic-rag`)

**A zero-infrastructure-cost (`$0`), containerized, bilingual (EN / BM / BM-EN Code-Switching) Agentic RAG platform built for air-gapped on-premises sovereignty and cache-optimized cloud inference.**

> **Project Status:** **All Phases Complete (`Phases 1–7`)** — FastAPI layered backend + Qdrant Hybrid RRF + `PyMuPDF4LLM` ingestion + Seed Collections + Bounded 6-node LangGraph SSE engine (`backend/`), Standalone `LLMJudge` vs. `JEVJudge` harness (`eval/`), Glass-Box Next.js 3-pane Workspace UI (`frontend/`), and multi-stage Docker Compose orchestration (`docker-compose.yml`, `.env.example`) are complete.

---

## Key Capabilities (Designed Architecture)

- **Collection-Isolated Multi-Tenant Workspaces:** Supports `.pdf`, `.md`, and `.txt` uploads with strict `collection_id`, `language`, and `doc_type` payload isolation, pre-seeded with:
  - **Collection A (`it_infra_sops`):** Enterprise IT & Server Infrastructure SOPs (hardware specs, NVMe/RAID runbooks, deployment SOPs).
  - **Collection B (`malaysian_enterprise_policy`):** Malaysian Enterprise Policy & Compliance (English, Bahasa Melayu, and BM-EN workplace code-switching).
- **Dual-Mode Inference Adapter (`LOCAL_ONPREM` $\leftrightarrow$ `CLOUD_API`):**
  - `LOCAL_ONPREM`: 100% air-gapped execution on 16GB RAM workstations using **Ollama / llama.cpp (`hf.co/empero-ai/Qwen3.8-4B-Distill-GGUF:Q4_K_M` [2.78 GB] default, or `hf.co/empero-ai/Qwen3.8-9B-Distill-GGUF:Q4_K_M` [5.78 GB] high-precision option via `LOCAL_LLM_MODEL`)**, **FastEmbed ONNX (`BAAI/bge-m3` + `Qdrant/bm25`)**, and an ONNX Cross-Encoder reranker.
  - `CLOUD_API`: **DeepSeek API** ordered strictly static-to-dynamic ($\ge 1,024$-token static prefix) to maximize **Prefix Context Caching** hit discounts.
- **Sub-10ms Qdrant Hybrid Search + Structure-Aware Ingestion:** Native Dense + Sparse BM25 vectors fused server-side via **Reciprocal Rank Fusion (RRF)**, fed by **`PyMuPDF4LLM`** and two-tier **`ParentChunk` / `ChildChunk` + `Breadcrumb`** chunking via non-blocking FastAPI `BackgroundTasks`.
- **Bounded LangGraph State Machine (`max_retries=2`):** 6 observable nodes (`QueryRouter` $\rightarrow$ `QueryRewriter` $\rightarrow$ `HybridRetriever` $\rightarrow$ `CrossEncoderReranker` $\rightarrow$ `RelevanceGrader` $\rightarrow$ `GroundedGenerator`) with `CitationValidator` (`[doc_id:pX]`) and deterministic bilingual `GracefulRefusal` (no external web search leakage).
- **Glass-Box Next.js Workspace & Typed SSE:** `POST /api/v1/chat/stream` emits 5 discriminated SSE events (`node_status`, `token`, `citations`, `done`, `error`) powering a live **Agent Execution Stepper** and clickable evidence drawers.
- **Standalone Evaluation Sub-Project (`eval/`):** Decoupled HTTP evaluation harness comparing generative **`LLMJudgeEvaluator`** vs. deterministic local **`JEVJudgeEvaluator`** (currently implemented with an air-gapped ONNX Multilingual NLI Cross-Encoder `mDeBERTa-v3` at `$0`, sub-300ms median latency, and `0.000` variance — with TypeSafe **JEV System-1 (`arXiv:2609.26550`)** confidence-cascading targeted for `v3.1`) across Latency, Token Cost, Score Variance, and Cohen's Kappa (`InterJudgeAgreement`).

---

## Engineering Progression: From `v1`/`v2` (`malay-qa-bot-rag`) to `v3` Production Architecture

This repository evolves the 2025 prototype [`amirulhazym/malay-qa-bot-rag`](https://github.com/amirulhazym/malay-qa-bot-rag) (`v1_malay_selfhosted` and `v2_multilingual_api`) into a production-ready enterprise architecture:

| Engineering Dimension | `v1_malay_selfhosted` (2025) | `v2_multilingual_api` (2025) | `sovereign-agentic-rag` (`v3` Implemented) |
| :--- | :--- | :--- | :--- |
| **Control Flow** | Linear `RetrievalQA` chain | Linear 1-shot `encode -> retrieve -> rerank -> generate` | **Bounded LangGraph State Machine** (`max_retries=2`) with self-reflective grading, query rewriting & `GracefulRefusal` |
| **Retrieval & Chunking** | Local FAISS + `MiniLM-L12-v2` + 300-char recursive split | Cloud Pinecone (pure dense cosine) + 1,000-char split (`Context Precision` plateaued at `0.336`) | **Qdrant Hybrid Search (Dense + Sparse BM25 via RRF)** + **`PyMuPDF4LLM` `ParentChunk`/`ChildChunk` + `Breadcrumb`** + `collection_id` isolation |
| **Backend & Transport** | Monolithic Streamlit script | Single-file FastAPI `main.py` (import-time model load, blocking `POST`, duplicated `v2_adapter.py`) | **Layered FastAPI** (`routers -> services -> agents -> repositories/providers`), `lifespan` + `Depends()`, async `BackgroundTasks`, **typed SSE** |
| **Frontend UI** | Streamlit | Streamlit polling `requests.post(..., timeout=30)` | **Next.js (App Router, TypeScript, Tailwind)** with live **Agent Execution Stepper**, mode switch, & clickable `[doc_id:pX]` citation cards |
| **Sovereignty & Cost** | Broken local `flan-t5-small` (`0.037` relevancy) | Cloud lock-in (Pinecone + Gemini) | **`$0` Infra Dual-Mode Adapter:** `LOCAL_ONPREM` (`empero-ai/Qwen3.8-4B/9B-Distill-GGUF` + FastEmbed ONNX) $\leftrightarrow$ `CLOUD_API` (DeepSeek Prefix Caching) |
| **Containerization** | Hugging Face Spaces SDK | Single container launching 2 processes via `sleep 20` | **Multi-stage Docker Compose** (`frontend`, `backend`, `qdrant`, `ollama` profile) with `service_healthy` probes & named volumes (plus `F:` Zero-Docker launcher) |
| **Evaluation** | Manual / Qualitative | Jupyter Notebook (`03-Evaluation-Framework.ipynb`) with RAGAs (`IndexError` / `NaN` failures) | **Standalone `eval/` CLI Harness** comparing **`LLM-as-a-Judge`** vs. deterministic **`JEVJudge` (Local ONNX NLI proxy; TypeSafe JEV cascade on v3.1 roadmap)** (Latency, Cost, Variance, Cohen's Kappa) |

---

## Unified End-to-End Agentic RAG System Architecture

```mermaid
flowchart TD
    subgraph UI_Tier["1. Next.js Workspace UI (App Router + TypeScript)"]
        UploadUI["Document Uploader (.pdf / .md / .txt)<br/>+ Collection Selector + Mode Switcher"]
        ChatUI["User Query Input (EN / BM / BM-EN)<br/>+ Stop Button (AbortController)"]
        ResponseUI["Glass-Box Response View<br/>Live Agent Stepper | Streaming Answer | Clickable [doc_id:pX] Inspector"]
    end

    subgraph Ingest_Track["2. Async Structure-Aware Ingestion Pipeline (FastAPI BackgroundTasks)"]
        Hash["SHA-256 Deduplicator<br/>(HTTP 202 Accepted -> asyncio.to_thread)"]
        Parser["PyMuPDF4LLM Layout Parser<br/>(Preserve Headings, Tables & Page Numbers)"]
        Chunker["Two-Tier Hierarchical Chunker<br/>Parent (1,200-1,500 tok) + Child (300-400 tok + Breadcrumb)"]
        FastEmbedIngest["FastEmbed ONNX<br/>Dense (bge-m3) + Sparse (bm25)"]
        Hash --> Parser --> Chunker --> FastEmbedIngest
    end

    subgraph Storage_Tier["3. Shared Sovereign Persistence Layer (Docker Volumes)"]
        SQLite[("SQLite Async (aiosqlite)<br/>collections | documents | parent_chunks | chat_history")]
        Qdrant[("Qdrant Vector DB<br/>Named Dense + Sparse BM25 Vectors<br/>Server-Side RRF + collection_id Payload Index")]
    end

    subgraph Agentic_Loop["4. Bounded LangGraph Agentic RAG State Machine (max_retries = 2)"]
        N1["Node 1: QueryRouter (/no_think JSON)<br/>Detect Language (EN/BM/Mixed) & Classify Intent"]
        N2["Node 2: QueryRewriter (/no_think JSON)<br/>Expand Acronyms & Bilingual BM<->EN Terms"]
        N3["Node 3: HybridRetriever<br/>Dense+Sparse RRF Search -> SQLite ParentChunk Hydration"]
        N4["Node 4: CrossEncoderReranker<br/>Local ONNX bge-reranker-v2-m3 (Top-5 Parents)"]
        N5{"Node 5: RelevanceGrader (/no_think)<br/>Self-Reflective Evidence Sufficiency Check"}
        Refusal["Deterministic GracefulRefusal<br/>(Bilingual Polite Refusal, citations: [])"]
        N6["Node 6: GroundedGenerator + CitationValidator<br/>Static-to-Dynamic Prompt (>=1,024 tok) + Regex [doc_id:pX] Guard"]

        N1 -->|"route: retrieve / compare"| N2
        N1 -->|"route: direct"| N6
        N2 --> N3 --> N4 --> N5
        N5 -->|"Insufficient & retry_count < 2<br/>(Autonomous Self-Correction Loop)"| N2
        N5 -->|"Insufficient & retry_count == 2"| Refusal
        N5 -->|"Evidence Approved"| N6
    end

    subgraph Inference_Seam["5. Dual-Mode Inference Provider Seam ($0 Infra)"]
        LocalLLM["LOCAL_ONPREM (Air-Gapped 16GB RAM)<br/>Ollama GGUF: empero-ai/Qwen3.8-4B-Distill (2.78GB)<br/>or empero-ai/Qwen3.8-9B-Distill (5.78GB)"]
        CloudLLM["CLOUD_API (Cache-Optimized)<br/>DeepSeek API (deepseek-chat)<br/>Prefix Context Cache Hit"]
    end

    subgraph Eval_Tier["6. Decoupled Standalone Evaluation Harness (eval/ CLI)"]
        EvalRunner["golden_dataset.jsonl (36 EN/BM/Mixed + Refusal Traps)<br/>-> RAGTargetAdapter -> LLMJudge vs JEVJudge (Local NLI)<br/>-> Latency, Cost ($), Variance & Cohen's Kappa Report"]
    end

    UploadUI -->|"POST /api/v1/collections/{id}/documents"| Hash
    Chunker -->|"Store Full ParentChunks"| SQLite
    FastEmbedIngest -->|"Upsert ChildChunks + Metadata"| Qdrant

    ChatUI -->|"POST /api/v1/chat/stream"| N1
    N3 <-->|"1. Child RRF Match"| Qdrant
    N3 <-->|"2. Hydrate Unique Parents"| SQLite

    N1 -.->|"Control JSON"| Inference_Seam
    N2 -.->|"Control JSON"| Inference_Seam
    N5 -.->|"Control JSON"| Inference_Seam
    N6 <-->|"Stream Tokens (LOCAL_ONPREM)"| LocalLLM
    N6 <-->|"Stream Tokens (CLOUD_API)"| CloudLLM

    N6 -->|"Typed SSE Stream:<br/>node_status | token | citations | done | error"| ResponseUI
    Refusal -->|"Typed SSE Stream:<br/>node_status | token | done"| ResponseUI
    EvalRunner -->|"HTTP SSE Contract"| N1
```

---

---

## Quickstart (`LOCAL_ONPREM` & `CLOUD_API`)

### Option A: Containerized Stack (`Docker Compose`)

```bash
# 1. Configure runtime environment
cp .env.example .env

# 2a. Launch with host Ollama (recommended for 16GB RAM edge workstations)
ollama pull hf.co/empero-ai/Qwen3.8-4B-Distill-GGUF:Q4_K_M
docker compose up --build -d

# 2b. Or launch fully containerized including bundled Ollama service
docker compose --profile local-ollama up --build -d

# 3. Seed Collection A (IT Infra SOPs) & Collection B (Malaysian Policy EN/BM/BM-EN)
curl -X POST http://localhost:8000/api/v1/collections/seed

# Open Next.js Glass-Box Workspace: http://localhost:3000
# Open FastAPI Swagger Docs:         http://localhost:8000/docs
```

### Option B: Native Workstation Launch (`100% Zero-Docker`, Drive-Pinned Caching)

If Docker Desktop is unavailable or you are running on a constrained OS partition:

```powershell
# 1. Pull local quantized Qwen3.8 model via Ollama (e.g. with OLLAMA_MODELS pinned to secondary drive)
ollama pull hf.co/empero-ai/Qwen3.8-4B-Distill-GGUF:Q4_K_M

# 2. Launch Ollama + FastAPI (embedded persistent Qdrant + SQLite) + Next.js in one command
.\start-local-f-drive.ps1

# 3. Seed Collection A & Collection B demos (or click "Seed Demo Collections" in the UI)
Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/api/v1/collections/seed"
```

To enable **`CLOUD_API`** (DeepSeek Prefix Context Caching), set `DEEPSEEK_API_KEY` in `.env` and switch modes directly in the UI header bar.

---

## Documentation Index

- **[Product Requirements Document (`docs/PRD.md`)](docs/PRD.md)** — Problem statement, 22 user stories, locked specs, and 8 production edge-case mitigations.
- **[System Architecture & Reverse-Engineering Guide (`docs/Architecture.md`)](docs/Architecture.md)** — Component seams, SQLite & Qdrant schemas, LangGraph 6-node workflow, SSE contract, and *"Under the Hood"* deep dives.
- **[Design System (`docs/Design-System.md`)](docs/Design-System.md)** — UI tokens, 3-pane Glass-Box workspace layout, Agent Execution Stepper, and Evidence Inspector specs.
- **[Evaluation Sub-Project (`docs/Eval-Subproject.md`)](docs/Eval-Subproject.md)** — `BaseJudgeEvaluator` seam, `LLMJudge` vs. `JEVJudge` scoring algorithms, `golden_dataset.jsonl`, and Cohen's Kappa reporting.
- **[Standalone Evaluation Harness Guide (`eval/README.md`)](eval/README.md)** — Setup and CLI benchmark commands (`--runs 3`, `HTTPTargetAdapter`, ONNX NLI / DeepSeek / Ollama).
- **[Next.js Frontend Workspace Guide (`frontend/README.md`)](frontend/README.md)** — 3-pane Glass-Box workspace components, typed `ReadableStream` SSE client, and Vitest/build commands.
- **[Domain & Architecture Glossary (`GLOSSARY.md`)](GLOSSARY.md)** — Canonical project vocabulary.
- **[TDD Implementation Plan (`docs/superpowers/plans/2026-11-16-sovereign-agentic-rag.md`)](docs/superpowers/plans/2026-11-16-sovereign-agentic-rag.md)** — Task-by-task build roadmap (Phases 2–7).
- **[Agent Engineering Rules (`AGENTS.md`)](AGENTS.md)** — Architectural invariants, TDD discipline, and verification commands.

---

## Observed Runtime Characteristics & `v3.1` Optimization Roadmap

Live end-to-end profiling on a CPU-only 16GB RAM Windows workstation (`Ollama Qwen3.8-4B-Distill-GGUF:Q4_K_M` at ~5.6 tok/s + FastEmbed `Xenova/bge-m3` / `bge-reranker-v2-m3-int8`) surfaced clear, concrete engineering findings for `v3.1`:

- **Why CPU `LOCAL_ONPREM` Latency Reached `150–310s`:** Qdrant Hybrid RRF + ONNX Cross-Encoder retrieval completes in ~3–6s, whereas `LocalOnPremLLMAdapter` currently calls `/v1/chat/completions` with `/no_think` text across 4 serial LLM nodes (`QueryRouter`, `QueryRewriter`, `RelevanceGrader`, `GroundedGenerator`). Unlike Ollama's native `/api/chat` with `"think": false` (proven in `eval/` to cut single-call latency `14x` from `181s` to `12.9s`), `/v1/chat/completions` allows hidden `<think>` generation on CPU, compounded by prefilling the `>=1,024`-token static DeepSeek prefix prompt on single-slot Ollama. Switching to **`CLOUD_API` (`deepseek-chat`)** bypasses local CPU bottlenecks (~`2.5–6s` expected with Prefix Context Caching).
- **`v3.1` Planned Priorities:**
  1. **Sub-35s CPU Local Path:** Move `LocalOnPremLLMAdapter` to native `/api/chat` (`"think": false` + `num_predict` budget), add a compact local prompt profile, skip `QueryRewriter` on attempt `0` (`retry_count > 0` only), confidence-gate `RelevanceGrader` via reranker scores, and add short numeric `[1]` citation aliases + retrieved-source fallback (fixing `0 citations` when 4B models paraphrase section names or leak `[hr_policy_my:p12]` from few-shot examples).
  2. **True TypeSafe JEV (`arXiv:2609.26550`) Cascade:** Rename the air-gapped ONNX NLI scorer (`mDeBERTa-v3`) clearly as `LocalNLIVerifier`, add `TypeSafeJEVEvaluator` (`Noul`/`Choice`/`Score` Decisions API), and implement confidence-gated escalation (`Accept When Confident, Escalate When Unsure`) alongside Langfuse/LangSmith tracing.
  3. **Dual-Surface RBAC UX (`/chat` + `/studio`):** Split non-technical `/chat` (Markdown rendering, persistent session history sidebar, 👍/👎 feedback) from `/studio` developer observability (full multi-attempt trace waterfall, reranker scores, KB chunk inspector, `DELETE` endpoints, eval hub).

---

## Verified Local Suites

```bash
# Backend unit, ingestion, LangGraph & SSE contract tests (41 tests)
cd backend && .venv/Scripts/pytest.exe tests -v && .venv/Scripts/ruff.exe check app tests

# Standalone eval harness tests, lint & side-by-side benchmark CLI (15 tests)
cd eval && .venv/Scripts/pytest.exe tests -v && .venv/Scripts/ruff.exe check eval_harness tests
.venv/Scripts/python.exe -m eval_harness.runner --dataset datasets/golden_dataset.jsonl --runs 3

# Next.js workspace tests, strict TypeScript & ESLint (9 tests; run `npm run build` on Linux/Docker/NTFS)
cd frontend && npm test && npm run typecheck && npm run lint
```

---

## Author

**Amirulhazym** — [GitHub](https://github.com/amirulhazym) · [Previous Prototype (`malay-qa-bot-rag`)](https://github.com/amirulhazym/malay-qa-bot-rag)
