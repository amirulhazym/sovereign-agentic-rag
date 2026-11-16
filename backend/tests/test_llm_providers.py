"""Tests for Dual-Mode LLMProvider seam, Qwen3.8 think-stripping, and DeepSeek prefix caching."""

import json

import httpx
from pydantic import BaseModel

from app.providers.base import ChatMessage, StreamChunk
from app.providers.cloud_deepseek import CloudDeepSeekLLMAdapter
from app.providers.local_onprem import (
    LocalOnPremLLMAdapter,
    StreamingThinkFilter,
    strip_thinking_blocks,
)
from app.providers.prompts import (
    STATIC_GROUNDED_SYSTEM_PROMPT,
    build_grounded_messages,
)
from app.schemas.workspace import ParentChunkRecord


class _RouterDecision(BaseModel):
    route: str
    language: str


def _sample_parent(doc_id: str, page: int, content: str) -> ParentChunkRecord:
    return ParentChunkRecord(
        parent_id=f"{doc_id}_p{page}_s0",
        doc_id=doc_id,
        collection_id="malaysian_enterprise_policy",
        filename=f"{doc_id}.pdf",
        page_number=page,
        section_header="Policy",
        breadcrumb=f"{doc_id}.pdf | Page {page} > Policy",
        content=content,
        token_count=20,
    )


def test_build_grounded_messages_enforces_static_prefix_and_sorted_sources() -> None:
    """Verify static >=1024-token prefix precedes history and sorted SOURCE blocks."""
    # Rough word/subword estimate ensuring DeepSeek 1,024-token prefix caching threshold is met
    assert len(STATIC_GROUNDED_SYSTEM_PROMPT.split()) >= 850

    p_late = _sample_parent("hr_my", 12, "Klausa 4.2 OT 2x")
    p_early = _sample_parent("hr_my", 3, "Waktu kerja 9am-6pm")

    messages = build_grounded_messages(
        collection_id="malaysian_enterprise_policy",
        collection_description="Malaysian HR & PDPA compliance documents",
        history=[
            ChatMessage(role="user", content="Hi"),
            ChatMessage(role="assistant", content="Hello, how can I help?"),
        ],
        parents=[p_late, p_early],
        question="Apa prosedur tuntutan OT?",
    )

    assert messages[0].role == "system"
    assert messages[0].content == STATIC_GROUNDED_SYSTEM_PROMPT
    assert messages[1].role == "system"
    assert "malaysian_enterprise_policy" in messages[1].content
    assert [m.role for m in messages[2:4]] == ["user", "assistant"]

    user_prompt = messages[-1].content
    assert "<<<SOURCE [hr_my:p3]>>>" in user_prompt
    assert "<<<SOURCE [hr_my:p12]>>>" in user_prompt
    # Page 3 must precede page 12 deterministically regardless of input order
    assert user_prompt.index("[hr_my:p3]") < user_prompt.index("[hr_my:p12]")
    assert user_prompt.endswith("Apa prosedur tuntutan OT?")


def test_qwen3_8_think_block_stripping_handles_chunk_boundaries() -> None:
    """Verify <think>...</think> reasoning spans are stripped even across split SSE chunks."""
    raw = "<think>\nLet me analyze step by step.\n</think>\nMenurut polisi [hr_my:p12], bayaran 2x."
    assert strip_thinking_blocks(raw) == "Menurut polisi [hr_my:p12], bayaran 2x."

    filt = StreamingThinkFilter()
    split_tokens = [
        "<thi",
        "nk>internal ",
        "reasoning</th",
        "ink>\n\n",
        "Bayaran ",
        "2x [hr_my:p12].",
    ]
    emitted = "".join(filt.feed(tok) for tok in split_tokens) + filt.flush()
    assert emitted == "Bayaran 2x [hr_my:p12]."


async def test_local_onprem_adapter_control_node_and_streaming() -> None:
    """Verify LocalOnPremLLMAdapter uses /no_think + temp=0.0 for JSON and streams cleanly."""
    captured_payloads: list[dict] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        captured_payloads.append(body)
        if not body.get("stream"):
            msg_json = '{"route": "single_collection", "language": "ms"}'
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": f"<think>quick</think>{msg_json}"}}]},
            )
        sse_body = (
            'data: {"choices":[{"delta":{"content":"<think>plan</think>Menurut "}}]}\n\n'
            'data: {"choices":[{"delta":{"content":"[hr_my:p12]."}}]}\n\n'
            "data: [DONE]\n\n"
        )
        return httpx.Response(200, content=sse_body.encode("utf-8"))

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport, base_url="http://mock-ollama/v1") as client:
        adapter = LocalOnPremLLMAdapter(
            client=client,
            model="hf.co/empero-ai/Qwen3.8-4B-Distill-GGUF:Q4_K_M",
        )
        decision = await adapter.generate_structured(
            [ChatMessage(role="user", content="Apa prosedur OT?")],
            _RouterDecision,
        )
        assert decision.route == "single_collection"
        assert decision.language == "ms"
        assert captured_payloads[0]["temperature"] == 0.0
        assert "/no_think" in captured_payloads[0]["messages"][-1]["content"]

        stream_chunks: list[StreamChunk] = []
        async for chunk in adapter.stream_answer(
            [ChatMessage(role="user", content="Apa prosedur OT?")]
        ):
            stream_chunks.append(chunk)

        text = "".join(c.delta for c in stream_chunks if c.delta)
        assert text == "Menurut [hr_my:p12]."
        assert captured_payloads[1]["temperature"] == 0.6
        assert captured_payloads[1]["top_p"] == 0.95


async def test_cloud_deepseek_adapter_extracts_prefix_cache_hits() -> None:
    """Verify CloudDeepSeekLLMAdapter enforces JSON mode and reads prompt_cache_hit_tokens."""
    captured_payloads: list[dict] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        captured_payloads.append(body)
        if not body.get("stream"):
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {"message": {"content": '{"route": "direct", "language": "en"}'}}
                    ]
                },
            )
        usage_json = (
            '{"choices":[],"usage":{"prompt_tokens":1400,'
            '"completion_tokens":12,"prompt_cache_hit_tokens":1280}}'
        )
        sse_body = (
            'data: {"choices":[{"delta":{"content":"Hello "}}]}\n\n'
            'data: {"choices":[{"delta":{"content":"world"}}]}\n\n'
            f"data: {usage_json}\n\n"
            "data: [DONE]\n\n"
        )
        return httpx.Response(200, content=sse_body.encode("utf-8"))

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(
        transport=transport, base_url="https://mock-deepseek.local"
    ) as client:
        adapter = CloudDeepSeekLLMAdapter(
            client=client,
            api_key="sk-test",
            model="deepseek-chat",
        )
        decision = await adapter.generate_structured(
            [ChatMessage(role="user", content="Hello")],
            _RouterDecision,
        )
        assert decision.route == "direct"
        assert captured_payloads[0]["response_format"] == {"type": "json_object"}
        assert captured_payloads[0]["temperature"] == 0.0

        chunks: list[StreamChunk] = []
        async for chunk in adapter.stream_answer([ChatMessage(role="user", content="Hello")]):
            chunks.append(chunk)

        assert "".join(c.delta for c in chunks if c.delta) == "Hello world"
        assert any(c.cache_hit_tokens == 1280 for c in chunks)
