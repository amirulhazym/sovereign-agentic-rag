"""Tests for CitationValidator normalization/phantom-stripping and bilingual GracefulRefusal."""

from app.agents.validators import (
    build_graceful_refusal,
    validate_and_normalize_citations,
)
from app.providers.embeddings import ScoredParent
from app.schemas.workspace import ParentChunkRecord


def _scored_parent(
    doc_id: str,
    page: int,
    content: str,
    score: float = 0.91,
) -> ScoredParent:
    return ScoredParent(
        parent=ParentChunkRecord(
            parent_id=f"{doc_id}_p{page}_s0",
            doc_id=doc_id,
            collection_id="malaysian_enterprise_policy",
            filename=f"{doc_id}.md",
            page_number=page,
            section_header="Section",
            breadcrumb=f"{doc_id}.md | Page {page} > Section",
            content=content,
            token_count=len(content.split()),
        ),
        score=score,
    )


def test_citation_validator_normalizes_variants_and_strips_phantoms() -> None:
    """Verify bracket format drift normalizes to [doc_id:pX] and hallucinated refs strip."""
    parents = [
        _scored_parent("hr_policy_my", 12, "Klausa 4.2: bayaran 2.0x hari kelepasan am.", 0.94),
        _scored_parent("pdpa_compliance", 2, "Seksyen 12B notifikasi dalam 72 jam.", 0.88),
    ]
    raw_answer = (
        "Untuk kerja public holiday, bayaran ialah 2.0x [hr_policy_my, page 12]. "
        "Kebocoran data perlu dilaporkan dalam 72 jam [pdpa_compliance:2]. "
        "Maklumat rekaan tidak sah [phantom_doc:p99] serta halaman palsu [hr_policy_my:p99]."
    )

    cleaned, items = validate_and_normalize_citations(raw_answer, parents)

    assert "[hr_policy_my:p12]" in cleaned
    assert "[pdpa_compliance:p2]" in cleaned
    assert "[phantom_doc:p99]" not in cleaned
    assert "[hr_policy_my:p99]" not in cleaned
    assert len(items) == 2
    assert items[0].citation_key == "[hr_policy_my:p12]"
    assert items[0].doc_id == "hr_policy_my"
    assert items[0].page == 12
    assert items[0].score == 0.94
    assert items[1].citation_key == "[pdpa_compliance:p2]"


def test_citation_validator_deduplicates_repeated_citations() -> None:
    """Verify repeated references to the same source emit one CitationItem."""
    parents = [_scored_parent("nvme_sop", 2, "ERR_NVME_0x04 PCIe link degradation.", 0.90)]
    raw_answer = (
        "Error ERR_NVME_0x04 means link degradation [nvme_sop:p2]. "
        "Reseat the drive before RMA [nvme_sop, p. 2]."
    )

    cleaned, items = validate_and_normalize_citations(raw_answer, parents)

    assert cleaned.count("[nvme_sop:p2]") == 2
    assert len(items) == 1
    assert items[0].filename == "nvme_sop.md"
    assert "ERR_NVME_0x04" in items[0].snippet


def test_build_graceful_refusal_matches_detected_language() -> None:
    """Verify deterministic refusal templates respect English vs. Malay/mixed queries."""
    en_refusal = build_graceful_refusal("en")
    ms_refusal = build_graceful_refusal("ms")
    mixed_refusal = build_graceful_refusal("mixed")

    assert "do not contain" in en_refusal.lower() or "insufficient" in en_refusal.lower()
    assert "maaf" in ms_refusal.lower()
    assert "maaf" in mixed_refusal.lower()
