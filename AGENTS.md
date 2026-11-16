# Agent Instructions (`sovereign-agentic-rag`)

Authoritative engineering guide for AI coding agents working in this repository.

## 1. Context Pointers (Read Before Acting)

- **Vocabulary:** Read [GLOSSARY.md](GLOSSARY.md) whenever naming or modifying domain entities, seams, or adapters; enforce exact canonical terms (`Collection`, `ParentChunk`, `ChildChunk`, `Breadcrumb`, `CitationKey`, `InferenceMode`, `GracefulRefusal`, `JudgeVerdict`).
- **Product Scope & User Stories:** Read [docs/PRD.md](docs/PRD.md) before adding endpoints, features, or UI workflows.
- **Backend, Qdrant, LangGraph & SSE Contracts:** Read [docs/Architecture.md](docs/Architecture.md) before modifying `backend/` modules, SQLite tables, Qdrant schemas, prompt prefix ordering, or SSE events.
- **Frontend UI & Tokens:** Read [docs/Design-System.md](docs/Design-System.md) before building or editing `frontend/` components.
- **Evaluation Harness (`eval/`):** Read [docs/Eval-Subproject.md](docs/Eval-Subproject.md) before modifying `eval/` judges, dataset schemas, or benchmark reports.
- **Step-by-Step Build Plan:** Follow [docs/superpowers/plans/2026-11-16-sovereign-agentic-rag.md](docs/superpowers/plans/2026-11-16-sovereign-agentic-rag.md) task-by-task during implementation.

---

## 2. Architectural & Boundary Invariants

1. **Zero Import-Time Side Effects:** Never instantiate ML models, Ollama/DeepSeek clients, or Qdrant/SQLite connections at Python module import time. Wire resources inside FastAPI `lifespan` and inject via `Depends()`.
2. **Strict Package Decoupling (`eval/` vs. `backend/`):** `eval/` communicates with `backend/` exclusively via `RAGTargetAdapter` over HTTP (or recorded JSONL outputs). Keep verifier weights inside `eval/` so they never enter `backend/` RAM.
3. **16GB RAM Protection (`LOCAL_ONPREM`):** Use FastEmbed (`ONNX`) rather than PyTorch `sentence-transformers` inside `backend/`. Configure quantized `Qwen3.8` series GGUF models / free OpenAI-compatible local sources via `LOCAL_LLM_MODEL` (supporting low-bit Dynamic v3.1 quants on 16GB workstations and larger quants on 24GB+ GPUs).
4. **Fast Control Nodes & Cache-Friendly Generation:**
   - Disable reasoning overhead (`/no_think` / `think=false` for `Qwen3.8`) and enforce `temperature=0.0` + JSON mode on `QueryRouter`, `QueryRewriter`, and `RelevanceGrader`.
   - Order `GroundedGenerator` messages strictly static-to-dynamic (Static System Prompt + Rules $\ge 1,024$ tokens $\rightarrow$ Collection Descriptor $\rightarrow$ Prior History $\rightarrow$ Sorted `ParentChunk` blocks $\rightarrow$ Current Question) for DeepSeek Prefix Context Caching.
5. **Bounded Agent Loop & Sovereignty:** Cap LangGraph retries strictly at `max_retries=2`. When retries exhaust with no relevant evidence, emit deterministic `GracefulRefusal` (`citations: []`). Keep all retrieval strictly inside the active `collection_id`; never call external web search.
6. **Non-Blocking Ingestion & Streaming:** Run CPU-bound PDF parsing (`PyMuPDF4LLM`) and embedding inside `asyncio.to_thread()` via FastAPI `BackgroundTasks`. Check `await request.is_disconnected()` inside SSE generators.

---

## 3. Coding Standards & Deep-Module Style

- **Python (`backend/`, `eval/`):** Python 3.11+, explicit type annotations on every function signature, Pydantic v2 (`ConfigDict`), async I/O (`aiosqlite`, `httpx.AsyncClient`, `AsyncQdrantClient`). Write concise docstrings explaining **why** a seam or invariant exists.
- **TypeScript (`frontend/`):** Next.js App Router, strict TypeScript (`strict: true`), typed discriminated unions mirroring FastAPI SSE envelopes (`node_status`, `token`, `citations`, `done`, `error`).
- **Deep Modules (`codebase-design`):** Keep interfaces minimal and hide orchestration details inside cohesive modules rather than sprawling shallow wrappers.

---

## 4. Mandatory TDD & Verification Discipline

Every code change follows **Red $\rightarrow$ Verify Red $\rightarrow$ Green $\rightarrow$ Verify Green $\rightarrow$ Refactor $\rightarrow$ Commit**:

1. Write a failing test targeting the module's public `Interface` first (using in-memory fakes for external LLM/Qdrant seams).
2. Run the targeted test command and confirm it fails for the expected missing behavior.
3. Write the minimal implementation to make the test pass.
4. Re-run the full relevant suite (`pytest`, `npm test` / `tsc`, `ruff`) and read actual output before claiming completion or committing.

---

## 5. Verification Commands

```bash
# Backend unit & contract tests (run inside backend/)
pytest -v

# Backend linting & type checking
ruff check app tests

# Standalone evaluation harness (run inside eval/)
pytest -v
python -m eval_harness.runner --dataset datasets/golden_dataset.jsonl --runs 3

# Frontend typecheck, lint, and tests (run inside frontend/)
npm run typecheck
npm run lint
npm test

# Full containerized stack
docker compose up --build -d
```
