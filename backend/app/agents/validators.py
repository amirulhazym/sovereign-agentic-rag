"""Post-generation CitationValidator and deterministic bilingual GracefulRefusal."""

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict

from app.providers.embeddings import ScoredParent
from app.schemas.workspace import ParentChunkRecord

_CITATION_PATTERN = re.compile(
    r"\[(?P<doc_id>[A-Za-z0-9_.\-]+)\s*(?:[:,]\s*(?:page|p\.?|)\s*|page\s+)(?P<page>\d+)\]"
)


class CitationItem(BaseModel):
    """Hydrated source citation attached to SSE `citations` event and UI cards."""

    model_config = ConfigDict(frozen=True)

    citation_key: str
    doc_id: str
    filename: str
    page: int
    score: float
    breadcrumb: str
    snippet: str


def _build_snippet(content: str, max_chars: int = 240) -> str:
    collapsed = re.sub(r"\s+", " ", content).strip()
    if len(collapsed) <= max_chars:
        return collapsed
    return collapsed[: max_chars - 3].rstrip() + "..."


def validate_and_normalize_citations(
    text: str,
    allowed_parents: list[ScoredParent] | list[ParentChunkRecord],
) -> tuple[str, list[CitationItem]]:
    """Normalize `[doc_id:pX]` citation drift and strip hallucinated phantom citations."""
    allowed_map: dict[tuple[str, int], tuple[ParentChunkRecord, float]] = {}
    for entry in allowed_parents:
        if isinstance(entry, ScoredParent):
            parent, score = entry.parent, entry.score
        else:
            parent, score = entry, 1.0
        key = (parent.doc_id, parent.page_number)
        if key not in allowed_map or score > allowed_map[key][1]:
            allowed_map[key] = (parent, score)

    emitted_items: dict[tuple[str, int], CitationItem] = {}

    def _replace_match(match: re.Match[str]) -> str:
        doc_id = match.group("doc_id")
        page = int(match.group("page"))
        key = (doc_id, page)
        if key not in allowed_map:
            return ""
        parent, score = allowed_map[key]
        canonical = f"[{doc_id}:p{page}]"
        if key not in emitted_items:
            emitted_items[key] = CitationItem(
                citation_key=canonical,
                doc_id=doc_id,
                filename=parent.filename,
                page=page,
                score=round(score, 4),
                breadcrumb=parent.breadcrumb,
                snippet=_build_snippet(parent.content),
            )
        return canonical

    normalized = _CITATION_PATTERN.sub(_replace_match, text)
    # Clean spacing artifacts left behind when phantom citations are stripped
    normalized = re.sub(r"[ \t]+([.,;:!?])", r"\1", normalized)
    normalized = re.sub(r"[ \t]{2,}", " ", normalized).strip()
    return normalized, list(emitted_items.values())


def build_graceful_refusal(language: Literal["en", "ms", "mixed"]) -> str:
    """Return a deterministic sovereign refusal when bounded retries find no grounded evidence."""
    if language == "en":
        return (
            "I'm sorry, the active collection documents do not contain sufficient verified "
            "evidence to answer this question accurately. Please upload the relevant "
            "documentation or refine your query."
        )
    return (
        "Maaf, dokumen dalam koleksi aktif ini tidak mengandungi bukti yang mencukupi "
        "untuk menjawab soalan anda dengan tepat. Sila muat naik dokumen berkaitan atau "
        "perincikan semula soalan anda."
    )
