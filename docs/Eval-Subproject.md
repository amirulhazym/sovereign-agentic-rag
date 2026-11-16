# Evaluation Sub-Project Specification (`eval/`): `LLM-as-a-Judge` vs. `JEV-as-a-Judge`

> **Design Goal:** A standalone-capable, reproducible evaluation harness that replaces `malay-qa-bot-rag`'s brittle Jupyter RAGAs notebook (`03-Evaluation-Framework.ipynb`) and empirically tests whether a fast, `$0`, deterministic discriminative verifier (**`JEVJudge`**) can rival or augment a generative **`LLMJudge`**.  
> **Vocabulary:** Canonical terms follow [GLOSSARY.md](../GLOSSARY.md).

---

## 1. Standalone Decoupled Architecture

The `eval/` package maintains its own `eval/pyproject.toml` and executes as an independent CLI process. Keeping `eval/` decoupled from `backend/` delivers two architectural guarantees:
1. **16GB RAM Protection:** Local `JEVJudge` verifier weights load only during offline evaluation runs and never compete with Ollama or FastEmbed inside the live `backend/` API container.
2. **Universal `RAGTargetAdapter` Seam:** Evaluators interact with the RAG system solely through `RAGTargetAdapter` (`HTTPTargetAdapter` calling `POST /api/v1/chat/stream` or a replayable JSONL trace adapter), making `eval/` portable to `v1`/`v2` baselines or any external RAG service.

```text
eval/datasets/golden_dataset.jsonl
        │
        ▼
┌───────────────────┐     HTTP SSE / Trace     ┌────────────────────────────┐
│  RAGTargetAdapter │ ◄──────────────────────► │  sovereign-agentic-rag API │
└─────────┬─────────┘                          └────────────────────────────┘
          │ produces RAGOutput (answer, retrieved_contexts, citations, latency)
          ▼
┌──────────────────────────────────────────┐
│     BaseJudgeEvaluator (Unified Seam)    │
│  evaluate(sample, output) -> JudgeVerdict│
└──────────────┬──────────────┬────────────┘
               │              │
      ┌────────▼──────┐  ┌────▼─────────────┐
      │   LLMJudge    │  │     JEVJudge     │
      │ (DeepSeek /   │  │ (Local NLI /     │
      │ Qwen3.8 JSON) │  │  Cross-Encoder)  │
      └────────┬──────┘  └────┬─────────────┘
               └──────┬───────┘
                      ▼
       eval/reports/benchmark_report.{md,json}
   (Latency, Cost $, Variance StdDev, Cohen's Kappa)
```

---

## 2. Core Data Contracts & `BaseJudgeEvaluator` Seam

### 2.1 `EvalSample` (`golden_dataset.jsonl` Row Schema)
```python
class ExpectedCitation(BaseModel):
    doc_id: str
    page: int

class EvalSample(BaseModel):
    id: str
    collection: Literal["it_infra_sops", "malaysian_enterprise_policy"]
    language: Literal["en", "ms", "mixed"]
    query_type: Literal[
        "factual",
        "table_lookup",
        "multi_doc_compare",
        "acronym_clause",
        "unanswerable_refusal",
    ]
    question: str
    ground_truth_answer: str
    expected_citations: list[ExpectedCitation]
    ground_truth_contexts: list[str]
```

### 2.2 `RAGOutput` (Collected by `RAGTargetAdapter`)
```python
class RAGOutput(BaseModel):
    sample_id: str
    answer: str
    retrieved_contexts: list[str]
    citations: list[str]
    refusal_triggered: bool
    target_latency_ms: float
    cache_hit_tokens: int = 0
```

### 2.3 `JudgeVerdict` & `BaseJudgeEvaluator`
```python
class JudgeVerdict(BaseModel):
    sample_id: str
    judge_name: str
    faithfulness: float = Field(ge=0.0, le=1.0)
    answer_relevancy: float = Field(ge=0.0, le=1.0)
    context_recall: float = Field(ge=0.0, le=1.0)
    composite_score: float = Field(ge=0.0, le=1.0)
    decision: Literal["PASS", "FAIL"]
    latency_ms: float
    input_tokens: int = 0
    output_tokens: int = 0
    estimated_cost_usd: float = 0.0
    rationale: str | None = None

class BaseJudgeEvaluator(ABC):
    name: str

    @abstractmethod
    async def evaluate(
        self,
        sample: EvalSample,
        output: RAGOutput,
        pass_threshold: float = 0.70,
    ) -> JudgeVerdict:
        """Grade a single RAG output across the 3 standardized metrics."""
```

- **Composite Score & Binary Decision:**
  `composite_score = (faithfulness + answer_relevancy + context_recall) / 3.0`  
  `decision = "PASS" if composite_score >= pass_threshold else "FAIL"`
- **`unanswerable_refusal` Trap Handling:** When `sample.query_type == "unanswerable_refusal"`, faithful behavior requires `output.refusal_triggered == True` (or explicit refusal semantics with zero fabricated claims) rather than guessing; hallucinations on trap queries score `faithfulness = 0.0`.

---

## 3. Judge Implementations

### 3.1 Judge 1: `LLMJudgeEvaluator` (`LLM-as-a-Judge`)
- **Backends:** OpenAI-compatible client targeting DeepSeek (`deepseek-chat`, `temperature=0.0`, `response_format={"type": "json_object"}`) or local Ollama / HF GGUF (`Qwen3.8` quantized via `LOCAL_LLM_MODEL` with `/no_think`).
- **Prompt Structure:** Static system grading rubric ($\ge 1,024$ tokens for DeepSeek prefix caching) followed by dynamic `(question, ground_truth_answer, retrieved_contexts, answer)` payload.
- **Metrics Scored:**
  1. **Faithfulness:** Fraction of factual claims in `output.answer` strictly supported by `output.retrieved_contexts`.
  2. **Answer Relevancy:** Degree to which `output.answer` directly and completely addresses `sample.question` in the expected language.
  3. **Context Recall:** Fraction of statements in `sample.ground_truth_answer` / `ground_truth_contexts` present within `output.retrieved_contexts`.
- **Telemetry:** Extracts exact `prompt_tokens`, `completion_tokens`, and `prompt_cache_hit_tokens` from API usage metadata to compute `estimated_cost_usd`.

### 3.2 Judge 2: `JEVJudgeEvaluator` (`JEV-as-a-Judge`)
- **Model Slot:** Local non-generative discriminative verifier (default configurable multilingual NLI Cross-Encoder such as `MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7` or multilingual Cross-Encoder `BAAI/bge-reranker-v2-m3` running via ONNX/CPU). Supports English, Bahasa Melayu, and BM-EN code-switching natively without generative JSON fragility.
- **Deterministic Metric Algorithm (Zero Token Generation):**
  1. **Faithfulness (Sentence-Level Entailment):** Split `output.answer` (stripping `[doc_id:pX]` citation tags) into atomic sentences $S = [s_1, \dots, s_m]$. For each sentence $s_i$, compute NLI probability $P(\text{entailment} \mid \text{premise}=\text{contexts}, \text{hypothesis}=s_i)$ normalized over $\{\text{entailment}, \text{contradiction}\}$ (or max across individual context chunks). Average across all claims.
  2. **Answer Relevancy (Discriminative Query-Answer Scoring):** Score $(sample.question, output.answer)$ through the multilingual relevance cross-encoder / NLI pair and normalize via logistic sigmoid into $[0.0, 1.0]$.
  3. **Context Recall (Reference Coverage):** Split `sample.ground_truth_answer` into reference clauses $R = [r_1, \dots, r_k]$; for each $r_j$, compute $\max_{c \in \text{retrieved\_contexts}} P(\text{entailment} \mid c, r_j)$ and average across $R$.
- **Properties:** `$0.00` API cost, deterministic (`StdDev == 0.000` across repeated runs), and ~20–100ms latency per sample.

---

## 4. Golden Dataset Specification (`eval/datasets/golden_dataset.jsonl`)

The curated golden dataset contains **36 bilingual samples** covering:
- **Collection A (`it_infra_sops`, 15 samples):** Hardware specs (`table_lookup`), NVMe/RAID error codes (`acronym_clause`), multi-step maintenance SOPs (`factual`), cross-runbook comparisons (`multi_doc_compare`), and out-of-scope traps (`unanswerable_refusal`).
- **Collection B (`malaysian_enterprise_policy`, 21 samples):** Balanced across pure English (`en`), Bahasa Melayu (`ms`), and Malaysian workplace code-switching (`mixed`, e.g., *"Apa prosedur tuntutan OT kalau kerja public holiday?"*), plus `unanswerable_refusal` traps.

### Sample JSONL Line
```json
{"id":"pol_mix_001","collection":"malaysian_enterprise_policy","language":"mixed","query_type":"acronym_clause","question":"Apa prosedur tuntutan OT kalau kerja public holiday?","ground_truth_answer":"Pekerja layak menuntut bayaran 2x atau cuti gantian tertakluk kepada kelulusan pengurus dalam tempoh 7 hari bekerja.","expected_citations":[{"doc_id":"hr_policy_my","page":12}],"ground_truth_contexts":["Klausa 4.2: Kerja pada hari kelepasan am layak menerima bayaran 2x ganda atau cuti gantian dengan kelulusan pengurus dalam masa 7 hari."]}
```

---

## 5. Side-by-Side Benchmark Report & Formulas

Running `python -m eval_harness.runner --dataset datasets/golden_dataset.jsonl --runs 3` executes both judges across `N=3` iterations per sample and produces `eval/reports/benchmark_report.md` and `.json`:

| Benchmark Dimension | Metric Definition |
| :--- | :--- |
| **Judge Latency (ms)** | `mean`, `p50`, and `p95` evaluation duration per sample in milliseconds |
| **API Token Cost ($)** | Sum of `(prompt_tokens * input_rate + completion_tokens * output_rate)` minus cache-hit savings (`$0.00` for `JEVJudge`) |
| **Score Variance / Determinism** | Mean sample standard deviation $\sigma$ of `composite_score` across `--runs 3` iterations (`0.000` expected for `JEVJudge`) |
| **Raw Agreement Rate (%)** | Fraction of samples where `LLMJudge.decision == JEVJudge.decision` (`PASS`/`FAIL`) |
| **Cohen's Kappa ($\kappa$)** | Chance-corrected agreement: $\kappa = \frac{p_o - p_e}{1 - p_e}$, where $p_o$ is observed agreement and $p_e$ is expected chance agreement |

### Under the Hood: Mengapa Perbandingan Ini Penting?
Dalam projek `v2`, `03-Evaluation-Framework.ipynb` hanya mengira purata markah RAGAs tanpa menyedari dua kelemahan besar *Generative Judge*: (1) lambat & berisiko pulangkan `NaN`/`IndexError`, dan (2) markah boleh berubah bila diulang. Dengan membandingkan **`LLMJudge`** menentang **`JEVJudge`** di bawah satu *interface* `BaseJudgeEvaluator`, anda membuktikan secara saintifik bila model pengesah pantas berharga `$0` boleh menggantikan LLM dalam *CI/CD regression gate*.
