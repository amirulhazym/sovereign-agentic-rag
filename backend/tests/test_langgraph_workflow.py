"""Tests for the bounded 6-node LangGraph Agentic RAG state machine (max_retries=2)."""

from pathlib import Path

from app.agents.graph import AgenticRAGWorkflow, WorkflowEvent
from app.agents.state import GraderOutput, RewriterOutput, RouterOutput
from app.providers.base import FakeLLMProvider
from app.providers.embeddings import FakeEmbeddingProvider, FakeRerankerProvider
from app.repositories.qdrant_repo import (
    InMemoryVectorRepository,
    VectorChunkUpsert,
)
from app.repositories.sqlite_repo import SQLiteRepository
from app.schemas.workspace import (
    ChildChunkRecord,
    CollectionCreate,
    DocumentCreate,
    ParentChunkRecord,
)


async def _seed_policy_doc(
    sqlite_repo: SQLiteRepository,
    vector_repo: InMemoryVectorRepository,
    embedder: FakeEmbeddingProvider,
    *,
    collection_id: str = "malaysian_enterprise_policy",
) -> None:
    await sqlite_repo.create_collection(
        CollectionCreate(
            id=collection_id,
            name="Malaysian Enterprise Policy",
            description="HR overtime and compliance policies",
        )
    )
    await sqlite_repo.create_document(
        DocumentCreate(
            doc_id="hr_policy_my",
            collection_id=collection_id,
            filename="hr_policy_my.md",
            doc_type="md",
            language="mixed",
            content_hash="hash_hr_12",
        )
    )
    parent = ParentChunkRecord(
        parent_id="hr_policy_my_p12_s0",
        doc_id="hr_policy_my",
        collection_id=collection_id,
        filename="hr_policy_my.md",
        page_number=12,
        section_header="Klausa 4.2 Overtime",
        breadcrumb="hr_policy_my.md | Page 12 > Klausa 4.2 Overtime",
        content=(
            "Klausa 4.2: Kerja pada hari kelepasan am (public holiday) layak menerima "
            "bayaran 2.0x gaji sejam atau cuti gantian dengan kelulusan pengurus dalam "
            "tempoh 7 hari bekerja."
        ),
        token_count=26,
    )
    await sqlite_repo.upsert_parent_chunks([parent])

    child_text = (
        "[hr_policy_my.md | Page 12 > Klausa 4.2 Overtime]\n"
        "Klausa 4.2 kerja lebih masa overtime public holiday hari kelepasan am bayaran 2.0x."
    )
    child = ChildChunkRecord(
        child_id="hr_policy_my_p12_s0_c0",
        parent_id=parent.parent_id,
        doc_id="hr_policy_my",
        collection_id=collection_id,
        filename="hr_policy_my.md",
        page_number=12,
        language="mixed",
        doc_type="md",
        breadcrumb=parent.breadcrumb,
        content=child_text,
    )
    dense = (await embedder.embed_dense([child_text]))[0]
    sparse = (await embedder.embed_sparse([child_text]))[0]
    await vector_repo.upsert_chunks(
        [VectorChunkUpsert(chunk=child, dense_vector=dense, sparse_vector=sparse)]
    )


async def _collect_events(
    workflow: AgenticRAGWorkflow,
    *,
    question: str,
    collection_id: str,
) -> list[WorkflowEvent]:
    events: list[WorkflowEvent] = []
    async for event in workflow.astream_run(
        question=question,
        collection_id=collection_id,
        collection_description="Malaysian Enterprise Policy",
        chat_history=[],
        inference_mode="CLOUD_API",
    ):
        events.append(event)
    return events


async def test_workflow_direct_route_bypasses_retrieval(tmp_path: Path) -> None:
    """Verify greetings routed as 'direct' skip retrieval and stream tokens directly."""
    sqlite_repo = SQLiteRepository(f"sqlite+aiosqlite:///{(tmp_path / 'wf1.db').as_posix()}")
    await sqlite_repo.init_schema()
    try:
        llm = FakeLLMProvider(
            structured_responses=[RouterOutput(route="direct", detected_language="ms")],
            stream_tokens=["Salam! ", "Ada apa yang boleh saya bantu?"],
        )
        workflow = AgenticRAGWorkflow(
            sqlite_repo=sqlite_repo,
            vector_repo=InMemoryVectorRepository(),
            embedding_provider=FakeEmbeddingProvider(dense_dim=16),
            reranker_provider=FakeRerankerProvider(),
            llm_provider=llm,
            max_retries=2,
        )

        events = await _collect_events(
            workflow, question="Hai, apa khabar?", collection_id="malaysian_enterprise_policy"
        )
        node_names = [e.node for e in events if e.event_type == "node_status"]
        assert "QueryRouter" in node_names
        assert "GroundedGenerator" in node_names
        assert "HybridRetriever" not in node_names

        done = next(e for e in events if e.event_type == "done")
        assert done.retries_used == 0
        assert done.refusal_triggered is False
        assert done.answer == "Salam! Ada apa yang boleh saya bantu?"
    finally:
        await sqlite_repo.close()


async def test_workflow_retries_once_then_succeeds_with_validated_citations(
    tmp_path: Path,
) -> None:
    """Verify RelevanceGrader rejection triggers retry loop and CitationValidator cleans output."""
    sqlite_repo = SQLiteRepository(f"sqlite+aiosqlite:///{(tmp_path / 'wf2.db').as_posix()}")
    await sqlite_repo.init_schema()
    vector_repo = InMemoryVectorRepository()
    embedder = FakeEmbeddingProvider(dense_dim=16)
    await _seed_policy_doc(sqlite_repo, vector_repo, embedder)

    try:
        llm = FakeLLMProvider(
            structured_responses=[
                RouterOutput(route="single_collection", detected_language="mixed"),
                # Attempt 0 rewrite -> Grader rejects
                RewriterOutput(queries=["OT rules"], reasoning="initial"),
                GraderOutput(
                    is_sufficient=False,
                    approved_parent_ids=[],
                    reason="Need BM holiday terminology",
                ),
                # Attempt 1 rewrite -> Grader approves
                RewriterOutput(
                    queries=["kerja lebih masa overtime public holiday hari kelepasan am"],
                    reasoning="expanded BM terms",
                ),
                GraderOutput(
                    is_sufficient=True,
                    approved_parent_ids=["hr_policy_my_p12_s0"],
                    reason="Matched Klausa 4.2",
                ),
            ],
            stream_tokens=[
                "Pekerja layak menuntut bayaran 2.0x ",
                "[hr_policy_my, page 12] ",
                "[phantom_doc:p99].",
            ],
            cache_hit_tokens=1024,
        )
        workflow = AgenticRAGWorkflow(
            sqlite_repo=sqlite_repo,
            vector_repo=vector_repo,
            embedding_provider=embedder,
            reranker_provider=FakeRerankerProvider(),
            llm_provider=llm,
            max_retries=2,
        )

        events = await _collect_events(
            workflow,
            question="Apa prosedur tuntutan OT kalau kerja public holiday?",
            collection_id="malaysian_enterprise_policy",
        )

        rewriter_events = [
            e
            for e in events
            if (
                e.event_type == "node_status"
                and e.node == "QueryRewriter"
                and e.status == "completed"
            )
        ]
        assert len(rewriter_events) == 2

        citations_event = next(e for e in events if e.event_type == "citations")
        assert [c.citation_key for c in citations_event.citations] == ["[hr_policy_my:p12]"]

        done = next(e for e in events if e.event_type == "done")
        assert done.retries_used == 1
        assert done.cache_hit_tokens == 1024
        assert done.refusal_triggered is False
        assert "[hr_policy_my:p12]" in done.answer
        assert "[phantom_doc:p99]" not in done.answer
    finally:
        await sqlite_repo.close()


async def test_workflow_caps_at_max_retries_2_and_emits_graceful_refusal(
    tmp_path: Path,
) -> None:
    """Verify bounded max_retries=2 halts looping and emits deterministic GracefulRefusal."""
    sqlite_repo = SQLiteRepository(f"sqlite+aiosqlite:///{(tmp_path / 'wf3.db').as_posix()}")
    await sqlite_repo.init_schema()
    vector_repo = InMemoryVectorRepository()
    embedder = FakeEmbeddingProvider(dense_dim=16)
    await _seed_policy_doc(sqlite_repo, vector_repo, embedder)

    try:
        llm = FakeLLMProvider(
            structured_responses=[
                RouterOutput(route="single_collection", detected_language="ms"),
                # Initial (attempt 0) + Retry 1 + Retry 2 = 3 retrieval cycles total
                RewriterOutput(queries=["kata laluan wifi"], reasoning="attempt 0"),
                GraderOutput(is_sufficient=False, approved_parent_ids=[], reason="irrelevant"),
                RewriterOutput(queries=["password guest wifi"], reasoning="retry 1"),
                GraderOutput(is_sufficient=False, approved_parent_ids=[], reason="irrelevant"),
                RewriterOutput(queries=["akses rangkaian tetamu"], reasoning="retry 2"),
                GraderOutput(is_sufficient=False, approved_parent_ids=[], reason="irrelevant"),
            ],
            stream_tokens=["MUST NOT BE CALLED"],
        )
        workflow = AgenticRAGWorkflow(
            sqlite_repo=sqlite_repo,
            vector_repo=vector_repo,
            embedding_provider=embedder,
            reranker_provider=FakeRerankerProvider(),
            llm_provider=llm,
            max_retries=2,
        )

        events = await _collect_events(
            workflow,
            question="Apakah kata laluan Wi-Fi tetamu?",
            collection_id="malaysian_enterprise_policy",
        )

        rewriter_completed = [
            e
            for e in events
            if (
                e.event_type == "node_status"
                and e.node == "QueryRewriter"
                and e.status == "completed"
            )
        ]
        # Exactly 3 retrieval cycles (attempt 0, retry 1, retry 2)
        assert len(rewriter_completed) == 3

        citations_event = next(e for e in events if e.event_type == "citations")
        assert citations_event.citations == []

        done = next(e for e in events if e.event_type == "done")
        assert done.retries_used == 2
        assert done.refusal_triggered is True
        assert "maaf" in done.answer.lower()
        assert "MUST NOT BE CALLED" not in done.answer
    finally:
        await sqlite_repo.close()


async def test_workflow_enforces_strict_collection_isolation(tmp_path: Path) -> None:
    """Verify documents in Collection A never leak into queries targeting Collection B."""
    sqlite_repo = SQLiteRepository(f"sqlite+aiosqlite:///{(tmp_path / 'wf4.db').as_posix()}")
    await sqlite_repo.init_schema()
    vector_repo = InMemoryVectorRepository()
    embedder = FakeEmbeddingProvider(dense_dim=16)
    await _seed_policy_doc(sqlite_repo, vector_repo, embedder, collection_id="col_a")
    await sqlite_repo.create_collection(
        CollectionCreate(id="col_b", name="Empty Tenant", description="Empty")
    )

    try:
        llm = FakeLLMProvider(
            structured_responses=[
                RouterOutput(route="single_collection", detected_language="en"),
                RewriterOutput(queries=["overtime policy"], reasoning="attempt 0"),
                RewriterOutput(queries=["public holiday claim"], reasoning="retry 1"),
                RewriterOutput(queries=["Klausa 4.2"], reasoning="retry 2"),
            ]
        )
        workflow = AgenticRAGWorkflow(
            sqlite_repo=sqlite_repo,
            vector_repo=vector_repo,
            embedding_provider=embedder,
            reranker_provider=FakeRerankerProvider(),
            llm_provider=llm,
            max_retries=2,
        )

        events = await _collect_events(
            workflow,
            question="What is the overtime policy?",
            collection_id="col_b",
        )

        done = next(e for e in events if e.event_type == "done")
        assert done.refusal_triggered is True
        assert done.retries_used == 2
        assert "2.0x" not in done.answer
    finally:
        await sqlite_repo.close()
