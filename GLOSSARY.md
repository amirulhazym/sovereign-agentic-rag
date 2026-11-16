# Domain & Architecture Glossary (`sovereign-agentic-rag`)

Canonical vocabulary used across `docs/`, `AGENTS.md`, `backend/`, `frontend/`, and `eval/`. Every module and specification must use these terms consistently without inventing synonyms.

---

## 1. Domain Terms

### `Collection`
An isolated tenant knowledge workspace containing uploaded or pre-seeded enterprise documents (`collection_id`). Every vector query and chat turn is strictly scoped to a single `Collection` via indexed payload filtering so data never leaks across workspaces.
- Pre-seeded demos:
  - Collection A (`it_infra_sops`): Enterprise IT & Server Infrastructure SOPs.
  - Collection B (`malaysian_enterprise_policy`): Malaysian Enterprise Policy & Compliance (English, Bahasa Melayu, and BM-EN code-switching).

### `DocumentRecord`
The persistent lifecycle metadata record for an uploaded or seeded file (`.pdf`, `.md`, `.txt`) stored in SQLite. Tracks `doc_id`, `collection_id`, `filename`, `content_hash` (SHA-256), `language`, `doc_type`, `status` (`PENDING` | `INDEXING` | `READY` | `FAILED`), and optional `error_message`.

### `ParentChunk`
A large, structure-preserving knowledge unit (~1,200–1,500 tokens) split along Markdown heading or page boundaries and persisted in SQLite (`parent_chunks`). Preserves full hardware tables, numbered statutory clauses, and exact `page_number` provenance so the LLM never reasons over truncated fragments.

### `ChildChunk`
A compact, search-optimized text unit (~300–400 tokens, 50-token overlap) derived from a `ParentChunk`, prefixed with a `Breadcrumb`, and indexed in Qdrant with both Dense and Sparse vectors plus `parent_id` metadata. Retrieval matches `ChildChunk` vectors and hydrates their deduplicated `ParentChunk` records.

### `Breadcrumb`
A deterministic structural prefix prepended to every `ChildChunk` prior to embedding and indexing (e.g., `[SOP_Server_2026.pdf | Page 4 > Storage Maintenance > NVMe Hot-Swap]`). Anchors section hierarchy and document context directly inside the vector representation.

### `CitationKey`
The canonical inline citation token `[doc_id:pX]` (e.g., `[hr_policy_my:p4]`) embedded in generated answers and matched against retrieved `ParentChunk` metadata via `CitationValidator` to render clickable evidence cards in the UI.

### `InferenceMode`
The runtime sovereignty toggle (`LOCAL_ONPREM` vs. `CLOUD_API`) selecting which `LLMProvider` and `EmbeddingProvider` adapters handle a request:
- `LOCAL_ONPREM`: 100% air-gapped self-hosted execution via Ollama (`qwen3:4b` default / `qwen3:8b`) + FastEmbed (`BAAI/bge-m3` / `Qdrant/bm25`) + ONNX Cross-Encoder.
- `CLOUD_API`: Cost-optimized DeepSeek API (`deepseek-chat` / `deepseek-reasoner`) arranged for **Prefix Context Caching**.

### `GracefulRefusal`
A deterministic, polite refusal response emitted in the user's detected language (`ms` or `en`) with `citations: []` whenever the LangGraph `RelevanceGrader` exhausts `max_retries=2` without finding sufficient grounded evidence. Prevents hallucination and replaces `v3` prototype's external web search fallback (Tavily).

### `EvalSample`
A single bilingual benchmark test case in `eval/datasets/golden_dataset.jsonl` containing `id`, `collection`, `language` (`ms` | `en` | `mixed`), `query_type` (`factual` | `table_lookup` | `multi_doc_compare` | `acronym_clause` | `unanswerable_refusal`), `question`, `ground_truth_answer`, `expected_citations`, and `ground_truth_contexts`.

### `JudgeVerdict`
The uniform output contract returned by every `BaseJudgeEvaluator` implementation for an `EvalSample`, containing normalized `[0.0, 1.0]` scores (`faithfulness`, `answer_relevancy`, `context_recall`), a binary threshold decision (`PASS` | `FAIL`), and benchmark telemetry (`latency_ms`, `input_tokens`, `output_tokens`, `estimated_cost_usd`).

### `LLMJudge` (`LLM-as-a-Judge`)
A generative evaluator implementation (`LLMJudgeEvaluator`) using DeepSeek or Qwen3 with structured JSON output to grade `EvalSample` outputs across Faithfulness, Answer Relevancy, and Context Recall.

### `JEVJudge` (`JEV-as-a-Judge`)
A non-generative discriminative decision/verifier evaluator (`JEVJudgeEvaluator`) using a local multilingual Cross-Encoder / Natural Language Inference (NLI) model to classify groundedness and relevance deterministically (`0.0` stochastic variance, `$0.00` API cost, sub-100ms latency).

### `InterJudgeAgreement`
The statistical alignment between `LLMJudge` and `JEVJudge` binary `PASS`/`FAIL` decisions across the golden dataset, reported as raw percentage agreement and Cohen's Kappa ($\kappa$).

---

## 2. Architecture Terms (per `codebase-design`)

### `Module`
Any cohesive unit exposing a small `Interface` over a rich implementation (a class, package, or vertical slice). Avoid substituting vague words like "component" or "unit".

### `Interface`
Everything a caller or test must know to use a `Module` correctly: type signatures, invariants, ordering constraints, error modes, configuration requirements, and performance bounds.

### `Depth`
Leverage at the `Interface`: a **deep module** hides substantial complexity (e.g., hybrid vector search + RRF + parent hydration) behind a concise `Interface`, whereas a **shallow module** merely forwards calls without absorbing complexity.

### `Seam`
A deliberate location where behavior can be swapped via an `Adapter` without modifying callers (e.g., `LLMProvider`, `VectorRepository`, `BaseJudgeEvaluator`, `RAGTargetAdapter`). Every `Seam` must earn its keep by having at least two real `Adapter` implementations (production + alternative or test fake).

### `Adapter`
A concrete implementation satisfying a `Seam`'s `Interface` (e.g., `LocalOnPremAdapter` and `CloudDeepSeekAdapter` satisfying `LLMProvider`).

### `Leverage`
The amplification gained when one deep `Module` serves multiple callers (e.g., both FastAPI SSE endpoints and `eval/` invoking the same RAG workflow `Interface` instead of duplicating logic as `v2` did in `v2_adapter.py`).

### `Locality`
Keeping related rules, state transitions, and bug fixes concentrated inside one `Module` so changes do not scatter across callers.
