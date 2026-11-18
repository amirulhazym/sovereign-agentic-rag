"""Generative LLM-as-a-Judge evaluator supporting DeepSeek (Prefix Caching) and local Qwen3.8."""

import json
import re
import time
from typing import Any, Literal

import httpx

from eval_harness.judges.base import (
    BaseJudgeEvaluator,
    build_verdict,
    is_refusal_response,
)
from eval_harness.schemas import EvalSample, JudgeVerdict, RAGOutput

_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", flags=re.DOTALL | re.IGNORECASE)
_JSON_OBJECT_RE = re.compile(r"\{.*\}", flags=re.DOTALL)

# Deliberately >= 1,024 tokens static grading rubric placed in message[0] so DeepSeek Prefix
# Context Caching hits across every sample and iteration (--runs 3).
STATIC_JUDGE_RUBRIC = """\
You are an impartial, strict enterprise RAG Evaluation Auditor grading answers
produced by a sovereign, multi-tenant Agentic RAG platform operating across English
(EN), Bahasa Melayu (BM), and Malaysian workplace code-switching (BM-EN).

Your sole task is to output a single valid JSON object evaluating a RAG system
response across three normalized [0.0, 1.0] metrics:
- "faithfulness": float between 0.0 and 1.0
- "answer_relevancy": float between 0.0 and 1.0
- "context_recall": float between 0.0 and 1.0
- "rationale": concise one-sentence justification explaining exact deductions or status.

======================================================================
SECTION 1: STRICT SCORING RUBRIC & DEFINITIONS
======================================================================

1. FAITHFULNESS (Groundedness in Retrieved Contexts) — Range [0.0, 1.0]:
   Measure the exact fraction of atomic factual claims in `candidate_answer` that
   are directly supported by facts stated in `retrieved_contexts`.
   - 1.00: Every factual statement, metric, SLA duration, hardware specification,
     firmware version, statutory clause, ringgit rate, or threshold in
     `candidate_answer` is explicitly supported by `retrieved_contexts`. Valid citation
     tags like `[doc_id:pX]` do not penalize faithfulness.
   - 0.75: Almost all core claims are grounded, with minor harmless stylistic
     phrasing that does not introduce unverified numbers, policies, timelines, or
     technical assertions.
   - 0.50: Roughly half of the factual assertions are grounded while other claims
     rely on external assumptions, missing table rows, or extrapolated advice not
     present in `retrieved_contexts`.
   - 0.25: Most claims cannot be traced to `retrieved_contexts`, or numbers/SLAs are
     mismatched between the answer and the provided snippets.
   - 0.00: Complete hallucination, fabricated numbers/commands/policies, empty
     fabrication, OR answering an `unanswerable_refusal` trap with guessed facts
     instead of politely refusing.
   - IMPORTANT: Never reward answers that inject outside general knowledge (e.g.,
     standard Malaysian EPF/SOCSO statutory rates, external Cisco IOS defaults, or
     GPU specs) when those numbers do not appear in `retrieved_contexts`.

2. ANSWER RELEVANCY (Question Alignment & Completeness) — Range [0.0, 1.0]:
   Measure how directly, accurately, and completely `candidate_answer` resolves
   `question` in a natural register matching English, Bahasa Melayu, or BM-EN
   code-switching.
   - 1.00: Directly answers every sub-part of the user's question without off-topic
     padding, evasive hedging, or ignoring constraints (such as time limits, approval
     rules, or hardware comparisons).
   - 0.75: Answers the main question accurately, though a minor secondary qualifier or
     formatting aspect could be clearer.
   - 0.50: Partially addresses the question (for example, answers one of two compared
     items in a `multi_doc_compare` query or omits a mandatory threshold).
   - 0.25: Tangential response discussing related keywords without actually resolving
     the user's core information need.
   - 0.00: Completely irrelevant answer, refusal emitted on a clearly answerable
     question when valid context exists, or unintelligible output.

3. CONTEXT RECALL (Reference Coverage in Retrieved Evidence) — Range [0.0, 1.0]:
   Compare `ground_truth_answer` (and `ground_truth_contexts`) against
   `retrieved_contexts` to measure what fraction of required reference facts were
   successfully retrieved by the search pipeline.
   - 1.00: All essential facts, table cells, clauses, and numeric targets needed to
     form `ground_truth_answer` are present inside `retrieved_contexts`.
   - 0.75: Most required reference statements appear in `retrieved_contexts`, missing
     only a minor supporting nuance.
   - 0.50: Approximately half of the required facts were retrieved (for example,
     retrieved only Page 1 but missed Page 2 in a two-document comparison query).
   - 0.25: Minimal overlap; retrieved text mentions the general topic without the
     specific target clause or specification number.
   - 0.00: `retrieved_contexts` contains none of the ground-truth evidence (unless
     `query_type` is `unanswerable_refusal` AND the system triggered a refusal).

======================================================================
SECTION 2: SPECIAL RULES FOR BILINGUAL & ADVERSARIAL EDGE CASES
======================================================================

Rule A — Bilingual & BM-EN Code-Switching Equivalence:
Enterprise queries frequently mix English technical nouns with Bahasa Melayu grammar
(e.g., "Apa prosedur tuntutan OT kalau kerja public holiday?", "Bandingkan Max Case
Temp untuk Enterprise-NVMe-U2-3840").
If `retrieved_contexts` uses Bahasa Melayu ("hari kelepasan am", "7 hari bekerja",
"resit cukai rasmi") and `candidate_answer` accurately conveys the identical fact in
English, Bahasa Melayu, or mixed BM-EN without distorting numbers or conditions,
treat the translation as 100% faithful. Similarly, if an IT SOP context is written in
English and the user asked in Bahasa Melayu, translating technical conditions
faithfully into Bahasa Melayu earns full marks.

Rule B — Exact Error Codes, Acronyms & Hardware Tables:
Pay extreme precision attention to exact codes (`ERR_NVME_0x04`, `ERR_NVME_0x12`,
`ERR_FAN_0x09`, `FW-4.8.2-PROD`, `FW-2.1.9-BOOT`, `Seksyen 12B`, `Klausa 4.2`), unit
comparisons (`<= 5 minutes` vs `<= 15 minutes`, `150 MB/s`, `70C` vs `75C`,
`RM0.85/km`, `RM120.00`, `RM280.00`, `72 jam`, `14 hari`), and boolean tier mandates
(`LOCAL_ONPREM ONLY` vs `CLOUD_API` zero-retention). Any fabricated unit shift or
altered threshold immediately lowers faithfulness by at least 0.50.

Rule C — Unanswerable Refusal Traps (`query_type == "unanswerable_refusal"`):
When `query_type` equals `"unanswerable_refusal"`, the knowledge base intentionally
lacks the requested fact (Zero-Web-Fallback Policy).
- Case 1 (Correct Behavior): `refusal_triggered` is `true` OR `candidate_answer`
  explicitly states that internal documents do not contain sufficient verified
  evidence while making zero fabricated factual claims. Score:
  `{"faithfulness": 1.0, "answer_relevancy": 1.0, "context_recall": 1.0}`.
- Case 2 (Failure / Hallucination): `candidate_answer` guesses a number, rate,
  password, or policy from pre-training memory (for example inventing an EPF
  percentage or TACACS+ password). Score:
  `{"faithfulness": 0.0, "answer_relevancy": 0.0, "context_recall": 0.0}`.

Rule D — False Refusal on Answerable Queries (`query_type != "unanswerable_refusal"`):
If the sample IS answerable from the knowledge base (`ground_truth_contexts` non-empty)
yet the RAG pipeline emitted `refusal_triggered: true`, score `faithfulness: 1.0`
(no hallucination occurred) but set `answer_relevancy: 0.0` and `context_recall: 0.0`
so the composite score fails below threshold.

======================================================================
SECTION 3: CALIBRATION EXAMPLES (FEW-SHOT ANCHORS)
======================================================================

Calibration Example 1 (High-Precision BM-EN Response):
Question: "Apa prosedur tuntutan OT kalau kerja public holiday?"
Ground Truth: "Pekerja bukan eksekutif layak menuntut bayaran 2.0x gaji sejam atau
cuti gantian dengan kelulusan bertulis pengurus dalam tempoh 7 hari bekerja."
Retrieved Contexts: ["Kerja Pada Hari Kelepasan Am (Public Holiday): Pekerja kategori
bukan eksekutif ... bayaran pada kadar 2.0x gaji sejam atau memilih cuti gantian,
tertakluk kepada kelulusan bertulis pengurus bahagian dalam tempoh 7 hari bekerja
melalui portal HRIS."]
Candidate Answer: "Boleh tuntut bayaran 2.0x gaji sejam atau cuti gantian, dengan
kelulusan bertulis pengurus dalam masa 7 hari bekerja melalui HRIS
[hr_employment_and_overtime_policy_my:p2]."
Expected JSON: {"faithfulness": 1.0, "answer_relevancy": 1.0, "context_recall": 1.0,
"rationale": "All rates, approval authority, and 7-day window match retrieved policy."}

Calibration Example 2 (Subtle Number Fabrication):
Question: "How much is the outstation hotel cap?"
Ground Truth: "Maximum RM280.00 per night with tax receipt within 14 days."
Retrieved Contexts: ["Had penginapan hotel rasmi: Maksimum RM280.00 semalam (resit
cukai rasmi wajib dilampirkan dalam tempoh 14 hari selepas pulang)."]
Candidate Answer: "The official hotel cap is RM350.00 per night, and receipts must be
submitted within 14 days."
Expected JSON: {"faithfulness": 0.5, "answer_relevancy": 0.7, "context_recall": 1.0,
"rationale": "Retrieved context supports 14-day window, but RM350.00 contradicts the
documented RM280.00 limit."}

Calibration Example 3 (Multi-Doc Partial Context Recall):
Question: "Which DR tier protects Qdrant Production and what post-failover integrity
check connects Qdrant and SQLite?"
Ground Truth: "Tier-1 Critical protects Qdrant (RPO <= 5 min); engineers run
curl /ready and check Qdrant point counts match SQLite chunk_count."
Retrieved Contexts: ["| Tier-1 Critical | Core Auth, Primary SQLite/Postgres Ledger,
Qdrant Production | Synchronous NVMe replication | <= 5 minutes | <= 15 minutes |"]
Candidate Answer: "Tier-1 Critical protects Qdrant Production with RPO <= 5 minutes
[disaster_recovery_failover_runbook:p1]."
Expected JSON: {"faithfulness": 1.0, "answer_relevancy": 0.55, "context_recall": 0.5,
"rationale": "First sub-question grounded accurately, but page 2 integrity check was
not retrieved."}

Calibration Example 4 (Proper Unanswerable Trap Refusal):
Query Type: "unanswerable_refusal"
Question: "What is the Cisco Nexus TACACS+ password?"
Candidate Answer: "I'm sorry, the uploaded documents do not contain sufficient
verified evidence to answer this question." (`refusal_triggered`: true)
Expected JSON: {"faithfulness": 1.0, "answer_relevancy": 1.0, "context_recall": 1.0,
"rationale": "Correctly triggered GracefulRefusal with zero fabricated claims."}

Calibration Example 5 (Hallucinated Answer on Trap):
Query Type: "unanswerable_refusal"
Question: "What is the employer EPF percentage above RM5,000?"
Candidate Answer: "Employer EPF contribution is 12% for salaries above RM5,000."
(`refusal_triggered`: false)
Expected JSON: {"faithfulness": 0.0, "answer_relevancy": 0.0, "context_recall": 0.0,
"rationale": "Fabricated ungrounded statutory knowledge on out-of-scope trap."}

Respond ONLY with the single JSON object matching the exact keys above. Do not emit
markdown fences or extraneous commentary."""


def strip_think_tags(text: str) -> str:
    """Strip Qwen3.8 `<think>...</think>` reasoning spans prior to JSON parsing."""
    stripped = _THINK_BLOCK_RE.sub("", text)
    if "<think>" in stripped.lower():
        idx = stripped.lower().find("<think>")
        stripped = stripped[:idx]
    return stripped.strip()


def _parse_judge_json(raw_text: str) -> dict[str, Any]:
    cleaned = strip_think_tags(raw_text)
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.DOTALL).strip()
    try:
        data = json.loads(cleaned)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        match = _JSON_OBJECT_RE.search(cleaned)
        if match:
            try:
                data = json.loads(match.group(0))
                if isinstance(data, dict):
                    return data
            except json.JSONDecodeError:
                pass
        # Recover core numeric metrics if generative rationale was truncated mid-stream
        recovered: dict[str, Any] = {}
        for key in ("faithfulness", "answer_relevancy", "context_recall"):
            num_match = re.search(rf'"{key}"\s*:\s*([0-9]+(?:\.[0-9]+)?)', cleaned)
            if num_match:
                recovered[key] = float(num_match.group(1))
        if len(recovered) == 3:
            rat_match = re.search(r'"rationale"\s*:\s*"([^"]*)', cleaned)
            recovered["rationale"] = rat_match.group(1).strip() if rat_match else ""
            return recovered
    raise ValueError(f"LLMJudge failed to produce valid JSON object: {raw_text[:200]!r}")


class LLMJudgeEvaluator(BaseJudgeEvaluator):
    """Generative LLM-as-a-Judge supporting DeepSeek API (cached) and Ollama Qwen3.8."""

    def __init__(
        self,
        *,
        backend: Literal["deepseek", "ollama"] = "deepseek",
        model: str = "deepseek-chat",
        base_url: str | None = None,
        api_key: str = "",
        timeout_seconds: float = 300.0,
        input_miss_cost_per_1m: float = 0.27,
        input_hit_cost_per_1m: float = 0.07,
        output_cost_per_1m: float = 1.10,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._backend = backend
        self._model = model
        default_url = (
            "https://api.deepseek.com" if backend == "deepseek" else "http://localhost:11434"
        )
        self._base_url = (base_url or default_url).rstrip("/")
        self._api_key = api_key
        self._timeout = timeout_seconds
        self._miss_rate = input_miss_cost_per_1m if backend == "deepseek" else 0.0
        self._hit_rate = input_hit_cost_per_1m if backend == "deepseek" else 0.0
        self._out_rate = output_cost_per_1m if backend == "deepseek" else 0.0
        self._injected_client = http_client
        self.name = f"LLMJudge({backend}:{model})"

    def _build_dynamic_user_prompt(self, sample: EvalSample, output: RAGOutput) -> str:
        contexts_block = "\n".join(
            f"[{idx}] {ctx}" for idx, ctx in enumerate(output.retrieved_contexts, start=1)
        )
        gt_contexts_block = "\n".join(
            f"[{idx}] {ctx}" for idx, ctx in enumerate(sample.ground_truth_contexts, start=1)
        )
        prefix = "/no_think\n" if self._backend == "ollama" else ""
        return (
            f"{prefix}"
            f"sample_id: {sample.id}\n"
            f"query_type: {sample.query_type}\n"
            f"language: {sample.language}\n"
            f"refusal_triggered: {str(output.refusal_triggered).lower()}\n\n"
            f"question:\n{sample.question}\n\n"
            f"ground_truth_answer:\n{sample.ground_truth_answer}\n\n"
            f"ground_truth_contexts:\n{gt_contexts_block or '(none)'}\n\n"
            f"retrieved_contexts:\n{contexts_block or '(none)'}\n\n"
            f"candidate_answer:\n{output.answer}\n"
        )

    def _compute_cost_usd(
        self, prompt_tokens: int, completion_tokens: int, cache_hit_tokens: int
    ) -> float:
        if self._backend != "deepseek":
            return 0.0
        cache_hits = min(max(0, cache_hit_tokens), max(0, prompt_tokens))
        cache_misses = max(0, prompt_tokens - cache_hits)
        cost = (
            (cache_misses / 1_000_000.0) * self._miss_rate
            + (cache_hits / 1_000_000.0) * self._hit_rate
            + (max(0, completion_tokens) / 1_000_000.0) * self._out_rate
        )
        return round(cost, 8)

    async def _invoke_backend(
        self, client: httpx.AsyncClient, user_prompt: str
    ) -> tuple[str, int, int, int]:
        messages = [
            {"role": "system", "content": STATIC_JUDGE_RUBRIC},
            {"role": "user", "content": user_prompt},
        ]
        if self._backend == "deepseek":
            headers = {"Content-Type": "application/json"}
            if self._api_key:
                headers["Authorization"] = f"Bearer {self._api_key}"
            response = await client.post(
                f"{self._base_url}/v1/chat/completions",
                headers=headers,
                json={
                    "model": self._model,
                    "messages": messages,
                    "temperature": 0.0,
                    "response_format": {"type": "json_object"},
                },
            )
            response.raise_for_status()
            data = response.json()
            raw_content = str(data["choices"][0]["message"]["content"])
            usage = data.get("usage", {})
            prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
            completion_tokens = int(usage.get("completion_tokens", 0) or 0)
            cache_hit_tokens = int(
                usage.get("prompt_cache_hit_tokens")
                or usage.get("prompt_tokens_details", {}).get("cached_tokens", 0)
                or 0
            )
            return raw_content, prompt_tokens, completion_tokens, cache_hit_tokens

        response = await client.post(
            f"{self._base_url}/api/chat",
            json={
                "model": self._model,
                "messages": messages,
                "stream": False,
                "think": False,
                "format": "json",
                "options": {"temperature": 0.0, "num_ctx": 8192},
            },
        )
        response.raise_for_status()
        data = response.json()
        raw_content = str(data["message"]["content"])
        prompt_tokens = int(data.get("prompt_eval_count", 0) or 0)
        completion_tokens = int(data.get("eval_count", 0) or 0)
        return raw_content, prompt_tokens, completion_tokens, 0

    async def evaluate(
        self,
        sample: EvalSample,
        output: RAGOutput,
        pass_threshold: float = 0.70,
    ) -> JudgeVerdict:
        """Invoke generative judge LLM with static rubric prefix and enforce trap invariants."""
        started_at = time.perf_counter()
        user_prompt = self._build_dynamic_user_prompt(sample, output)

        if self._injected_client is not None:
            raw, in_tok, out_tok, hit_tok = await self._invoke_backend(
                self._injected_client, user_prompt
            )
        else:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                raw, in_tok, out_tok, hit_tok = await self._invoke_backend(client, user_prompt)

        latency_ms = (time.perf_counter() - started_at) * 1000.0
        parsed = _parse_judge_json(raw)
        faithfulness = float(parsed.get("faithfulness", 0.0))
        relevancy = float(parsed.get("answer_relevancy", 0.0))
        recall = float(parsed.get("context_recall", 0.0))
        rationale = str(parsed.get("rationale", "")) or None

        # Enforce strict unanswerable_refusal invariant deterministically on top of LLM scores
        refused = is_refusal_response(output)
        if sample.query_type == "unanswerable_refusal":
            if refused:
                faithfulness, relevancy, recall = 1.0, 1.0, 1.0
            else:
                faithfulness = 0.0
                recall = 0.0
                relevancy = min(relevancy, 0.25)
        elif output.refusal_triggered:
            faithfulness = 1.0
            relevancy = 0.0
            recall = 0.0

        return build_verdict(
            sample_id=sample.id,
            judge_name=self.name,
            faithfulness=faithfulness,
            answer_relevancy=relevancy,
            context_recall=recall,
            pass_threshold=pass_threshold,
            latency_ms=latency_ms,
            input_tokens=in_tok,
            output_tokens=out_tok,
            estimated_cost_usd=self._compute_cost_usd(in_tok, out_tok, hit_tok),
            rationale=rationale,
        )
