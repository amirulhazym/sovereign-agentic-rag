"""Contract tests for BaseJudgeEvaluator, LLMJudgeEvaluator, and JEVJudgeEvaluator."""

import json
import statistics
from typing import Any

import httpx
import numpy as np
import pytest

from eval_harness.judges.base import BaseJudgeEvaluator
from eval_harness.judges.jev_judge import JEVJudgeEvaluator, ONNXMultilingualNLIBackend
from eval_harness.judges.llm_judge import (
    STATIC_JUDGE_RUBRIC,
    LLMJudgeEvaluator,
)
from eval_harness.schemas import EvalSample, JudgeVerdict, RAGOutput


def _sample_policy() -> EvalSample:
    return EvalSample(
        id="pol_mix_001",
        collection="malaysian_enterprise_policy",
        language="mixed",
        query_type="acronym_clause",
        question="Apa prosedur tuntutan OT kalau kerja public holiday?",
        ground_truth_answer=(
            "Pekerja bukan eksekutif layak menuntut bayaran 2.0x gaji sejam atau cuti "
            "gantian dengan kelulusan pengurus dalam tempoh 7 hari bekerja."
        ),
        expected_citations=[],
        ground_truth_contexts=[
            "Kerja pada hari kelepasan am layak menuntut bayaran 2.0x gaji sejam atau cuti "
            "gantian dengan kelulusan pengurus dalam tempoh 7 hari bekerja."
        ],
    )


def _sample_trap() -> EvalSample:
    return EvalSample(
        id="it_ref_001",
        collection="it_infra_sops",
        language="en",
        query_type="unanswerable_refusal",
        question="What is the TACACS+ root password for our Cisco Nexus switches?",
        ground_truth_answer="Refusal: credentials are not present in internal documents.",
        expected_citations=[],
        ground_truth_contexts=[],
    )


async def test_llm_judge_deepseek_prefix_cache_cost_and_think_stripping() -> None:
    captured_payloads: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        captured_payloads.append(body)
        return httpx.Response(
            status_code=200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": (
                                "<think>Evaluating sentence claims against context...</think>"
                                '{"faithfulness": 0.96, "answer_relevancy": 0.92, '
                                '"context_recall": 0.90, "rationale": "Grounded in Clause 4.2."}'
                            )
                        }
                    }
                ],
                "usage": {
                    "prompt_tokens": 1500,
                    "completion_tokens": 120,
                    "prompt_cache_hit_tokens": 1100,
                },
            },
        )

    assert len(STATIC_JUDGE_RUBRIC.split()) >= 850  # >= ~1,024 tokens for DeepSeek caching

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(
        transport=transport, base_url="https://api.deepseek.com"
    ) as client:
        judge: BaseJudgeEvaluator = LLMJudgeEvaluator(
            backend="deepseek",
            model="deepseek-chat",
            api_key="test-key",
            http_client=client,
        )
        verdict = await judge.evaluate(
            _sample_policy(),
            RAGOutput(
                sample_id="pol_mix_001",
                answer="Layak bayaran 2.0x atau cuti gantian dalam 7 hari [hr:p2].",
                retrieved_contexts=_sample_policy().ground_truth_contexts,
                citations=["[hr:p2]"],
                refusal_triggered=False,
                target_latency_ms=210.0,
            ),
        )

    assert isinstance(verdict, JudgeVerdict)
    assert verdict.judge_name.startswith("LLMJudge")
    assert verdict.decision == "PASS"
    assert verdict.composite_score == pytest.approx((0.96 + 0.92 + 0.90) / 3.0, abs=1e-3)
    assert verdict.input_tokens == 1500
    assert verdict.output_tokens == 120
    assert verdict.estimated_cost_usd > 0.0

    # Verify static rubric precedes dynamic user payload & temperature=0.0 + json_object enforced
    sent = captured_payloads[0]
    assert sent["temperature"] == 0.0
    assert sent["response_format"] == {"type": "json_object"}
    assert sent["messages"][0]["role"] == "system"
    assert sent["messages"][0]["content"] == STATIC_JUDGE_RUBRIC
    assert sent["messages"][1]["role"] == "user"


async def test_llm_judge_ollama_local_is_zero_cost_and_penalizes_hallucinated_trap() -> None:
    captured_payloads: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        captured_payloads.append(body)
        return httpx.Response(
            status_code=200,
            json={
                "message": {
                    "content": (
                        "<think>hallucinated model check</think>\n"
                        '{"faithfulness": 0.85, "answer_relevancy": 0.90, '
                        '"context_recall": 0.80, "rationale": "Answered without refusal."}'
                    )
                },
                "prompt_eval_count": 1300,
                "eval_count": 64,
            },
        )

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport, base_url="http://localhost:11434") as client:
        judge = LLMJudgeEvaluator(
            backend="ollama",
            model="hf.co/empero-ai/Qwen3.8-4B-Distill-GGUF:Q4_K_M",
            base_url="http://localhost:11434",
            http_client=client,
        )
        verdict = await judge.evaluate(
            _sample_trap(),
            RAGOutput(
                sample_id="it_ref_001",
                answer="The TACACS+ password is Admin@1234.",
                retrieved_contexts=[],
                citations=[],
                refusal_triggered=False,
                target_latency_ms=420.0,
            ),
        )

    assert verdict.estimated_cost_usd == 0.0
    assert "/no_think" in captured_payloads[0]["messages"][1]["content"]
    # Hallucinating on an unanswerable_refusal trap must force faithfulness=0.0 and FAIL
    assert verdict.faithfulness == 0.0
    assert verdict.decision == "FAIL"


class _FakeDeterministicNLIBackend:
    """In-memory discriminative scorer fake verifying deterministic behavior & averaging."""

    def score_entailment(self, pairs: list[tuple[str, str]]) -> list[float]:
        scores: list[float] = []
        for premise, hypothesis in pairs:
            p_tokens = set(premise.lower().split())
            h_tokens = {tok.strip(".,:;()[]") for tok in hypothesis.lower().split() if tok}
            if not h_tokens:
                scores.append(0.0)
                continue
            overlap = len(p_tokens & h_tokens) / len(h_tokens)
            scores.append(min(1.0, max(0.05, overlap * 1.15)))
        return scores

    def score_relevance(self, pairs: list[tuple[str, str]]) -> list[float]:
        return [0.91 for _ in pairs]


async def test_jev_judge_is_deterministic_zero_cost_and_handles_traps() -> None:
    judge: BaseJudgeEvaluator = JEVJudgeEvaluator(scorer=_FakeDeterministicNLIBackend())
    sample = _sample_policy()
    output = RAGOutput(
        sample_id=sample.id,
        answer=(
            "Pekerja layak menuntut bayaran 2.0x gaji sejam atau cuti gantian "
            "[hr_employment_and_overtime_policy_my:p2]. "
            "Kelulusan pengurus dalam tempoh 7 hari bekerja diperlukan."
        ),
        retrieved_contexts=sample.ground_truth_contexts,
        citations=["[hr_employment_and_overtime_policy_my:p2]"],
        refusal_triggered=False,
        target_latency_ms=145.0,
    )

    verdicts = [await judge.evaluate(sample, output) for _ in range(3)]
    scores = [v.composite_score for v in verdicts]
    assert statistics.pstdev(scores) == 0.0
    assert all(v.estimated_cost_usd == 0.0 for v in verdicts)
    assert all(v.input_tokens == 0 and v.output_tokens == 0 for v in verdicts)
    assert verdicts[0].decision == "PASS"

    # Unanswerable trap: faithful refusal must score 1.0 PASS; hallucination must score 0.0
    trap = _sample_trap()
    refused = await judge.evaluate(
        trap,
        RAGOutput(
            sample_id=trap.id,
            answer="Maaf, dokumen koleksi tidak mengandungi bukti yang mencukupi.",
            retrieved_contexts=[],
            citations=[],
            refusal_triggered=True,
            target_latency_ms=88.0,
        ),
    )
    assert refused.faithfulness == 1.0
    assert refused.decision == "PASS"

    hallucinated = await judge.evaluate(
        trap,
        RAGOutput(
            sample_id=trap.id,
            answer="The root password is Password123.",
            retrieved_contexts=[],
            citations=[],
            refusal_triggered=False,
            target_latency_ms=92.0,
        ),
    )
    assert hallucinated.faithfulness == 0.0
    assert hallucinated.decision == "FAIL"


def test_onnx_nli_backend_lazy_loads_and_computes_entailment_softmax() -> None:
    load_count = 0

    class FakeSession:
        def run(self, _outputs: Any, inputs: dict[str, np.ndarray]) -> list[np.ndarray]:
            batch = inputs["input_ids"].shape[0]
            # NLI logits: [entailment, neutral, contradiction]
            row = np.array([[4.0, 0.2, -2.0]], dtype=np.float32)
            return [np.repeat(row, batch, axis=0)]

    class FakeEncoded:
        def __init__(self) -> None:
            self.ids = [1, 2, 3, 4]
            self.attention_mask = [1, 1, 1, 1]
            self.type_ids = [0, 0, 1, 1]

    class FakeTokenizer:
        def enable_truncation(self, max_length: int) -> None:
            assert max_length == 512

        def enable_padding(self, pad_id: int = 0) -> None:
            pass

        def encode_batch(self, pairs: list[tuple[str, str]]) -> list[FakeEncoded]:
            return [FakeEncoded() for _ in pairs]

    def fake_loader(repo_id: str) -> tuple[Any, Any, dict[str, int]]:
        nonlocal load_count
        load_count += 1
        return FakeSession(), FakeTokenizer(), {"entailment": 0, "neutral": 1, "contradiction": 2}

    backend = ONNXMultilingualNLIBackend(loader=fake_loader)
    assert load_count == 0  # Zero import/construction side effects

    scores = backend.score_entailment([("Premise", "Hypothesis"), ("Premise 2", "Hypothesis 2")])
    assert load_count == 1
    assert len(scores) == 2
    # e^4 / (e^4 + e^-2) = 1 / (1 + e^-6) ~= 0.9975
    assert all(0.99 < s <= 1.0 for s in scores)
