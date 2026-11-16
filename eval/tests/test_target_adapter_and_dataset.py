"""Contract tests for golden dataset schema loading and RAGTargetAdapter seams."""

from pathlib import Path

import httpx
import pytest

from eval_harness.schemas import EvalSample, JudgeVerdict, RAGOutput, load_golden_dataset
from eval_harness.target_adapter import (
    HTTPTargetAdapter,
    RAGTargetAdapter,
    RecordedTraceTargetAdapter,
)

DATASET_PATH = Path(__file__).resolve().parents[1] / "datasets" / "golden_dataset.jsonl"


def test_golden_dataset_contains_36_valid_bilingual_samples() -> None:
    samples = load_golden_dataset(DATASET_PATH)

    assert len(samples) == 36
    assert all(isinstance(sample, EvalSample) for sample in samples)
    assert len({sample.id for sample in samples}) == 36

    it_samples = [s for s in samples if s.collection == "it_infra_sops"]
    pol_samples = [s for s in samples if s.collection == "malaysian_enterprise_policy"]
    assert len(it_samples) == 15
    assert len(pol_samples) == 21

    # Collection B must balance en, ms, and BM-EN mixed queries (7 each)
    assert len([s for s in pol_samples if s.language == "en"]) == 7
    assert len([s for s in pol_samples if s.language == "ms"]) == 7
    assert len([s for s in pol_samples if s.language == "mixed"]) == 7

    # All 5 required query types must be covered including unanswerable traps
    query_types = {s.query_type for s in samples}
    assert query_types == {
        "factual",
        "table_lookup",
        "multi_doc_compare",
        "acronym_clause",
        "unanswerable_refusal",
    }
    traps = [s for s in samples if s.query_type == "unanswerable_refusal"]
    assert len(traps) >= 6
    assert all(len(t.expected_citations) == 0 for t in traps)


def test_judge_verdict_enforces_unit_bounds() -> None:
    verdict = JudgeVerdict(
        sample_id="pol_mix_001",
        judge_name="JEVJudge",
        faithfulness=0.95,
        answer_relevancy=0.9,
        context_recall=0.85,
        composite_score=0.9,
        decision="PASS",
        latency_ms=42.5,
    )
    assert verdict.estimated_cost_usd == 0.0

    with pytest.raises(ValueError):
        JudgeVerdict(
            sample_id="bad",
            judge_name="JEVJudge",
            faithfulness=1.2,
            answer_relevancy=0.5,
            context_recall=0.5,
            composite_score=0.73,
            decision="PASS",
            latency_ms=10.0,
        )


async def test_http_target_adapter_parses_chunked_sse_frames() -> None:
    sse_wire = (
        'event: node_status\n'
        'data: {"node":"query_router","status":"completed",'
        '"attempt":0,"latency_ms":12,"detail":"retrieve"}\n\n'
        'event: token\n'
        'data: {"delta":"Bayaran 2.0x "}\n\n'
        'event: token\n'
        'data: {"delta":"atau cuti gantian '
        '[hr_employment_and_overtime_policy_my:p2]."}\n\n'
        'event: citations\n'
        'data: {"sources":[{"citation_key":"[hr_employment_and_overtime_policy_my:p2]",'
        '"doc_id":"hr_employment_and_overtime_policy_my",'
        '"filename":"hr_employment_and_overtime_policy_my.md",'
        '"page":2,"score":0.91,"breadcrumb":"HR > Klausa 4.2",'
        '"snippet":"Kerja pada hari kelepasan am layak menuntut '
        'bayaran 2.0x atau cuti gantian."}]}\n\n'
        'event: done\n'
        'data: {"message_id":"msg_01","session_id":"sess_01","mode":"LOCAL_ONPREM",'
        '"retries_used":0,"cache_hit_tokens":1280,'
        '"total_latency_ms":340,"refusal_triggered":false}\n\n'
    )

    class ChunkedSSEStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            raw = sse_wire.encode("utf-8")
            # Intentionally split mid-line / mid-UTF-8-boundary-safe byte slices
            for idx in range(0, len(raw), 19):
                yield raw[idx : idx + 19]

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/v1/chat/stream"
        return httpx.Response(
            status_code=200,
            headers={"content-type": "text/event-stream; charset=utf-8"},
            stream=ChunkedSSEStream(),
        )

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport, base_url="http://mock-backend") as client:
        adapter: RAGTargetAdapter = HTTPTargetAdapter(
            base_url="http://mock-backend",
            inference_mode="LOCAL_ONPREM",
            http_client=client,
        )
        sample = load_golden_dataset(DATASET_PATH)[0]
        output = await adapter.evaluate_sample(sample)

    expected_answer = (
        "Bayaran 2.0x atau cuti gantian "
        "[hr_employment_and_overtime_policy_my:p2]."
    )
    assert isinstance(output, RAGOutput)
    assert output.sample_id == sample.id
    assert output.answer == expected_answer
    assert output.citations == ["[hr_employment_and_overtime_policy_my:p2]"]
    assert output.retrieved_contexts == [
        "Kerja pada hari kelepasan am layak menuntut bayaran 2.0x atau cuti gantian."
    ]
    assert output.refusal_triggered is False
    assert output.cache_hit_tokens == 1280
    assert output.target_latency_ms == 340.0


async def test_http_target_adapter_raises_on_sse_error_frame() -> None:
    sse_err = (
        'event: error\n'
        'data: {"code":"PROVIDER_TIMEOUT","message":"Ollama timed out","recoverable":true}\n\n'
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code=200,
            headers={"content-type": "text/event-stream"},
            content=sse_err.encode("utf-8"),
        )

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport, base_url="http://mock-backend") as client:
        adapter = HTTPTargetAdapter(base_url="http://mock-backend", http_client=client)
        sample = load_golden_dataset(DATASET_PATH)[0]
        with pytest.raises(RuntimeError, match="PROVIDER_TIMEOUT"):
            await adapter.evaluate_sample(sample)


async def test_recorded_trace_target_adapter_replays_saved_outputs(tmp_path: Path) -> None:
    sample = load_golden_dataset(DATASET_PATH)[0]
    trace_file = tmp_path / "trace.jsonl"
    recorded = RAGOutput(
        sample_id=sample.id,
        answer=sample.ground_truth_answer,
        retrieved_contexts=sample.ground_truth_contexts,
        citations=[f"[{c.doc_id}:p{c.page}]" for c in sample.expected_citations],
        refusal_triggered=False,
        target_latency_ms=95.0,
        cache_hit_tokens=512,
    )
    trace_file.write_text(recorded.model_dump_json() + "\n", encoding="utf-8")

    adapter: RAGTargetAdapter = RecordedTraceTargetAdapter.from_jsonl(trace_file)
    replayed = await adapter.evaluate_sample(sample)
    assert replayed == recorded
