# Design System & UI Architecture: `sovereign-agentic-rag`

> **Aesthetic Direction:** **Glass-Box Enterprise Precision** — inspired by sovereign on-premises AI consoles and modern developer workbenches. High contrast, zero clutter, explicit system observability, and bilingual (EN/BM) clarity.

---

## 1. Visual Identity & Design Tokens

### 1.1 Color Palette (Dark-Slate Sovereign Default)
| Token | Hex | Usage |
| :--- | :--- | :--- |
| `--bg-canvas` | `#090D16` | Primary application shell background |
| `--bg-surface` | `#111827` | Sidebar, header bar, evidence inspector drawer |
| `--bg-elevated` | `#1E293B` | Cards, active collection pill, chat assistant container |
| `--border-subtle` | `#273244` | 1px crisp structural dividers between panes |
| `--text-primary` | `#F8FAFC` | Primary headings, chat prose, active labels |
| `--text-secondary` | `#94A3B8` | Subtitles, metadata breadcrumbs, timestamps |
| `--accent-sovereign` | `#10B981` | `LOCAL_ONPREM` mode badge, `READY` doc status, verified node checkmarks |
| `--accent-cloud` | `#6366F1` | `CLOUD_API` (DeepSeek) mode badge, cache-hit telemetry pill |
| `--accent-warning` | `#F59E0B` | `INDEXING` progress, LangGraph loop retry badge (`Attempt 2/2`) |
| `--accent-danger` | `#F43F5E` | `FAILED` ingestion status, SSE `error` banner, `GracefulRefusal` indicator |
| `--citation-pill-bg` | `#0F292E` | Inline `[doc_id:pX]` clickable pill background (`border: #14B8A6`) |

### 1.2 Typography
- **UI & Prose (`--font-sans`):** `Inter` / `Geist Sans` — crisp legibility for multi-paragraph bilingual Malay & English policy answers.
- **Telemetry, Citations & Code (`--font-mono`):** `JetBrains Mono` — used strictly for `[doc_id:pX]` citation badges, latency pills (`9ms RRF`), `cache_hit_tokens`, SHA-256 prefixes, and LangGraph node identifiers.

---

## 2. 3-Pane Responsive Workspace Layout

```text
+---------------------------------------------------------------------------------------------------+
| TOP HEADER BAR (56px)                                                                             |
| [sovereign-agentic-rag]   Collection: [B: MY Policy v]   Mode: [(O) LOCAL_ONPREM | CLOUD_API]     |
|                                                          Health: [Ready]  Telemetry: [Cache: 1.2k]|
+-------------------------+-----------------------------------------------+-------------------------+
| LEFT SIDEBAR (280px)    | CENTER CHAT CANVAS (Flex-1)                   | RIGHT INSPECTOR (360px) |
|                         |                                               | (Collapsible Drawer)    |
| COLLECTIONS             | [Agent Execution Stepper - Live Glass Box]    |                         |
| * A: IT & Server SOPs   |  (✓) Route -> (✓) Rewrite -> (✓) Hybrid RRF   | EVIDENCE INSPECTOR      |
| > B: MY Policy (Active) |  -> (✓) Rerank -> (⟳) Grade [Attempt 1/2]   |                         |
| + New Collection        |                                               | Selected: [hr_pol:p12]  |
|                         | ASSISTANT MESSAGE                             | File: MY_HR_Policy.pdf  |
| DOCUMENTS (4 Ready)     | Pekerja layak menuntut bayaran 2x atau cuti   | Page: 12 | Score: 0.91  |
| [Upload .pdf/.md/.txt]  | gantian [hr_pol:p12]...                       |                         |
|  - MY_HR_Policy.pdf [✓] |                                               | Breadcrumb:             |
|  - PDPA_Guide.pdf   [⟳] | [Citation Cards Strip]                        | MY_HR_Policy > OT Rules |
|    (INDEXING 64%)       | [hr_pol:p12] [pdpa_my:p4]                     |                         |
|                         |                                               | Parent Chunk Text:      |
| [Seed Demo Collections] | [Ask in EN / BM / BM-EN...]  [Send / Stop ■]  | "Klausa 4.2: Kerja pada |
|                         |                                               |  hari kelepasan am..."  |
+-------------------------+-----------------------------------------------+-------------------------+
```

- **Mobile / Narrow Viewport (<1024px):** Left Sidebar collapses into a slide-over drawer; Right Evidence Inspector opens as a bottom/modal sheet when any citation pill is clicked.

---

## 3. Core Component Specifications

### 3.1 `InferenceModeSwitch` (Sovereignty Toggle)
- Two-segment pill control in the Top Header Bar:
  - **`LOCAL_ONPREM`** (Emerald shield icon): Displays active local stack subtitle (`Ollama qwen3:4b + FastEmbed bge-m3`).
  - **`CLOUD_API`** (Indigo cloud-bolt icon): Displays DeepSeek cache status (`DeepSeek Chat · Prefix Caching On`).
- Switching modes sets the active request parameter sent to `POST /api/v1/chat/stream` without losing the current conversation or active `Collection`.

### 3.2 `CollectionManager` & `DocumentUploader` (Left Sidebar)
- **Collection Selector:** Lists collections with document count badges; switching collections resets the active chat thread cleanly so contexts never bleed across tenants.
- **Drag-and-Drop Uploader:** Accepts `.pdf`, `.md`, `.txt`. Immediately on drop:
  1. Shows optimistic row with `PENDING` status after receiving HTTP `202 Accepted`.
  2. Polls `GET /api/v1/documents/{doc_id}/status` every `1500ms` while status is `PENDING` or `INDEXING`, updating badge color (`Amber` pulse -> `Emerald` `READY` or `Rose` `FAILED` with tooltip error message).
  3. If backend returns `200 OK` (SHA-256 duplicate detected), shows non-intrusive toast: *"Document already indexed in this collection (deduplicated)."*
- **Seed Demo Button:** Visible whenever `collections` is empty (or via utility menu) to invoke `POST /api/v1/collections/seed` for instant Collection A & B availability.

### 3.3 `AgentExecutionStepper` (Center Canvas — Live LangGraph Observability)
Driven directly by SSE `event: node_status`:
- Displays the 6 pipeline stages horizontally (or compact vertical timeline on mobile):
  `QueryRouter` $\rightarrow$ `QueryRewriter` $\rightarrow$ `HybridRetriever` $\rightarrow$ `CrossEncoderReranker` $\rightarrow$ `RelevanceGrader` $\rightarrow$ `GroundedGenerator`
- **Node States:**
  - `idle`: muted slate outline
  - `started`: animated emerald/indigo ring
  - `completed`: solid checkmark + mono latency badge (`9ms`, `145ms`) + expandable detail tooltip (`14 child hits -> 5 parents`)
  - `retrying`: amber loop badge (`Retry 2/2`) displayed on `RelevanceGrader` $\rightarrow$ `QueryRewriter` when initial retrieval fails relevance check.
  - `refused`: slate/rose badge when `retry_count == 2` terminates in `GracefulRefusal`.
- Collapses cleanly to a single summary pill (*"Completed in 1.64s · 0 retries · 2 citations"*) once `event: done` arrives, expandable anytime for inspection.

### 3.4 `StreamingMessageBubble` & `CitationPill`
- Renders incremental Markdown (`event: token`) safely without layout jitter.
- Regex matches canonical `[doc_id:pX]` tokens and converts them into interactive `<button>` pills styled with `--citation-pill-bg` matched against `event: citations`.
- **Stop Generating (`AbortController`):** While streaming is active, the Send button transforms into a crimson **Stop (`■`)** button that aborts the `fetch()` stream immediately and marks the partial message `[Stopped by user]`.

### 3.5 `EvidenceInspectorDrawer` (Right Pane)
- Opens when clicking any `[doc_id:pX]` inline pill or bottom Citation Summary Card.
- Displays:
  - Document filename & `doc_id`
  - Exact `page_number` & `Breadcrumb` hierarchy
  - Reranker relevance score (`0.00–1.00`)
  - Full hydrated `ParentChunk` text (preserving Markdown tables and statutory clauses).

---

## 4. State Handling & Accessibility Rules

- **Empty State:** When a new session opens on Collection A or B, show 3 clickable **Starter Prompts** tailored to that collection (including a BM-EN code-switched prompt for Collection B, e.g., *"Apa prosedur tuntutan OT kalau kerja public holiday?"*).
- **Graceful Refusal Card:** When `citations` is empty and refusal is triggered, display a subtle amber info callout below the answer: *"No matching evidence found after 2 retrieval attempts — answer intentionally withheld to prevent hallucination."*
- **SSE Stream Error Banner:** On `event: error`, render an inline Rose alert showing `code` and `message` (e.g., `OLLAMA_UNAVAILABLE`) plus a **Retry** button without wiping the user's prompt.
- **Keyboard Accessibility:** `Enter` submits query (`Shift+Enter` for newline), `Escape` closes the Right Evidence Inspector, and all interactive controls maintain visible focus rings (`focus-visible:ring-2`).
