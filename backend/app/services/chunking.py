"""Structure-aware document extraction (PyMuPDF4LLM) and Parent-Child Breadcrumb chunking.

Preserves Markdown headings, hardware tables, and exact page numbers so Qdrant ChildChunk
vectors retain high precision while SQLite ParentChunks preserve complete surrounding context.
"""

import re
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict

from app.core.exceptions import UnsupportedDocumentTypeError
from app.schemas.workspace import ChildChunkRecord, ParentChunkRecord

_SUPPORTED_EXTENSIONS: dict[str, Literal["pdf", "md", "txt"]] = {
    ".pdf": "pdf",
    ".md": "md",
    ".markdown": "md",
    ".txt": "txt",
}

_BM_WORDS = frozenset(
    {
        "yang",
        "dan",
        "atau",
        "untuk",
        "kepada",
        "dengan",
        "pada",
        "adalah",
        "ialah",
        "tidak",
        "boleh",
        "pekerja",
        "bayaran",
        "cuti",
        "tuntutan",
        "tuntut",
        "kerja",
        "hari",
        "tempoh",
        "dasar",
        "polisi",
        "syarikat",
        "jika",
        "sekiranya",
        "dalam",
        "mengikut",
        "layak",
        "kerana",
        "akan",
        "telah",
        "tiada",
        "maaf",
        "melalui",
        "gantian",
        "kelepasan",
        "pengurus",
        "menerima",
        "balik",
        "peraturan",
    }
)

_EN_WORDS = frozenset(
    {
        "the",
        "and",
        "for",
        "with",
        "from",
        "this",
        "that",
        "must",
        "shall",
        "should",
        "verify",
        "before",
        "after",
        "below",
        "above",
        "during",
        "ensure",
        "between",
        "when",
        "if",
        "not",
        "is",
        "are",
        "will",
        "can",
        "policy",
        "system",
        "employee",
        "temperature",
        "error",
        "public",
        "holiday",
        "overtime",
        "portal",
        "server",
        "storage",
        "drive",
        "replacement",
        "leave",
        "approval",
    }
)


class ExtractedPage(BaseModel):
    """Single document page normalized to Markdown text."""

    model_config = ConfigDict(frozen=True)

    page_number: int
    markdown: str


def detect_doc_type(filename: str) -> Literal["pdf", "md", "txt"]:
    """Validate and resolve canonical doc_type from filename extension."""
    lowered = filename.lower().strip()
    for ext, doc_type in _SUPPORTED_EXTENSIONS.items():
        if lowered.endswith(ext):
            return doc_type
    raise UnsupportedDocumentTypeError(filename)


def detect_language(text: str) -> Literal["en", "ms", "mixed"]:
    """Classify text as English, Bahasa Melayu, or BM-EN workplace code-switching."""
    tokens = re.findall(r"[a-z]+", text.lower())
    if not tokens:
        return "mixed"

    ms_hits = sum(1 for tok in tokens if tok in _BM_WORDS)
    en_hits = sum(1 for tok in tokens if tok in _EN_WORDS)
    total_hits = ms_hits + en_hits

    if total_hits == 0:
        return "mixed"
    if ms_hits >= 2 and en_hits >= 2 and min(ms_hits, en_hits) / total_hits >= 0.2:
        return "mixed"
    if ms_hits > en_hits:
        return "ms"
    if en_hits > ms_hits:
        return "en"
    return "mixed"


def _extract_pdf_pages(file_bytes: bytes) -> list[ExtractedPage]:
    """Extract PDF pages to Markdown via lazy PyMuPDF4LLM import."""
    import pymupdf
    import pymupdf4llm

    pages: list[ExtractedPage] = []
    with pymupdf.open(stream=file_bytes, filetype="pdf") as pdf_doc:
        page_chunks = pymupdf4llm.to_markdown(pdf_doc, page_chunks=True)
        for idx, chunk in enumerate(page_chunks, start=1):
            raw_text = str(chunk.get("text", "")).strip()
            if not raw_text:
                continue
            meta = chunk.get("metadata") or {}
            page_num = int(meta.get("page", idx)) if isinstance(meta, dict) else idx
            pages.append(ExtractedPage(page_number=page_num, markdown=raw_text))
    return pages


def _extract_text_pages(file_bytes: bytes) -> list[ExtractedPage]:
    """Split .md / .txt bytes by explicit `<!-- page: N -->` markers or form-feed."""
    decoded = file_bytes.decode("utf-8", errors="replace").strip()
    if not decoded:
        return []

    marker_pattern = re.compile(r"<!--\s*page:\s*(\d+)\s*-->", flags=re.IGNORECASE)
    matches = list(marker_pattern.finditer(decoded))
    if matches:
        pages: list[ExtractedPage] = []
        first_chunk = decoded[: matches[0].start()].strip()
        if first_chunk:
            pages.append(ExtractedPage(page_number=1, markdown=first_chunk))
        for i, match in enumerate(matches):
            page_num = int(match.group(1))
            start = match.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(decoded)
            body = decoded[start:end].strip()
            if body:
                pages.append(ExtractedPage(page_number=page_num, markdown=body))
        return pages

    if "\f" in decoded:
        return [
            ExtractedPage(page_number=idx, markdown=part.strip())
            for idx, part in enumerate(decoded.split("\f"), start=1)
            if part.strip()
        ]

    return [ExtractedPage(page_number=1, markdown=decoded)]


def extract_pages(file_bytes: bytes, filename: str) -> list[ExtractedPage]:
    """Parse uploaded .pdf, .md, or .txt bytes into structured Markdown pages."""
    doc_type = detect_doc_type(filename)
    if doc_type == "pdf":
        return _extract_pdf_pages(file_bytes)
    return _extract_text_pages(file_bytes)


@dataclass(frozen=True)
class _SectionBlock:
    section_header: str
    content: str


def _split_markdown_sections(markdown_text: str) -> list[_SectionBlock]:
    """Split page Markdown into heading-scoped sections without creating empty heading blocks."""
    heading_re = re.compile(r"^(#{1,4})\s+(.+?)\s*$")
    sections: list[_SectionBlock] = []
    current_header = "General"
    current_lines: list[str] = []

    for line in markdown_text.splitlines():
        match = heading_re.match(line)
        if match:
            body = "\n".join(current_lines).strip()
            if body:
                sections.append(_SectionBlock(section_header=current_header, content=body))
            current_header = match.group(2).strip()
            current_lines = []
        else:
            current_lines.append(line)

    trailing_body = "\n".join(current_lines).strip()
    if trailing_body:
        sections.append(_SectionBlock(section_header=current_header, content=trailing_body))
    return sections


def _split_oversized_section(
    text: str, max_tokens: int
) -> list[str]:
    """Split a section that exceeds parent_max_tokens along paragraph boundaries when possible."""
    words = text.split()
    if len(words) <= max_tokens:
        return [text]

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    buckets: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for para in paragraphs:
        para_tokens = len(para.split())
        if para_tokens > max_tokens:
            if current:
                buckets.append("\n\n".join(current))
                current = []
                current_tokens = 0
            para_words = para.split()
            for start in range(0, len(para_words), max_tokens):
                buckets.append(" ".join(para_words[start : start + max_tokens]))
            continue

        if current_tokens + para_tokens > max_tokens and current:
            buckets.append("\n\n".join(current))
            current = [para]
            current_tokens = para_tokens
        else:
            current.append(para)
            current_tokens += para_tokens

    if current:
        buckets.append("\n\n".join(current))
    return buckets


class HierarchicalChunker:
    """Builds ParentChunks (SQLite) and overlapping Breadcrumb ChildChunks (Qdrant)."""

    def __init__(
        self,
        *,
        parent_max_tokens: int = 1350,
        child_max_tokens: int = 350,
        child_overlap_tokens: int = 50,
    ) -> None:
        if child_overlap_tokens >= child_max_tokens:
            raise ValueError("child_overlap_tokens must be smaller than child_max_tokens.")
        self._parent_max_tokens = parent_max_tokens
        self._child_max_tokens = child_max_tokens
        self._child_overlap_tokens = child_overlap_tokens

    def chunk_document(
        self,
        *,
        pages: list[ExtractedPage],
        doc_id: str,
        collection_id: str,
        filename: str,
        doc_type: Literal["pdf", "md", "txt"],
        language: Literal["en", "ms", "mixed"],
    ) -> tuple[list[ParentChunkRecord], list[ChildChunkRecord]]:
        """Produce linked ParentChunkRecords and Breadcrumb-prefixed ChildChunkRecords."""
        parents: list[ParentChunkRecord] = []
        children: list[ChildChunkRecord] = []

        for page in pages:
            sections = _split_markdown_sections(page.markdown)
            parent_idx = 0
            for section in sections:
                slices = _split_oversized_section(section.content, self._parent_max_tokens)
                for slice_text in slices:
                    parent_id = f"{doc_id}_p{page.page_number}_s{parent_idx}"
                    parent_idx += 1
                    breadcrumb = f"{filename} | Page {page.page_number} > {section.section_header}"
                    token_count = len(slice_text.split())
                    parent_record = ParentChunkRecord(
                        parent_id=parent_id,
                        doc_id=doc_id,
                        collection_id=collection_id,
                        filename=filename,
                        page_number=page.page_number,
                        section_header=section.section_header,
                        breadcrumb=breadcrumb,
                        content=slice_text,
                        token_count=token_count,
                    )
                    parents.append(parent_record)
                    children.extend(
                        self._slice_children(
                            parent=parent_record,
                            doc_type=doc_type,
                            language=language,
                        )
                    )
        return parents, children

    def _slice_children(
        self,
        *,
        parent: ParentChunkRecord,
        doc_type: Literal["pdf", "md", "txt"],
        language: Literal["en", "ms", "mixed"],
    ) -> list[ChildChunkRecord]:
        words = parent.content.split()
        if not words:
            return []

        step = self._child_max_tokens - self._child_overlap_tokens
        child_records: list[ChildChunkRecord] = []
        child_idx = 0
        start = 0

        while start < len(words):
            end = min(start + self._child_max_tokens, len(words))
            body = " ".join(words[start:end])
            prefixed_content = f"[{parent.breadcrumb}]\n{body}"
            child_records.append(
                ChildChunkRecord(
                    child_id=f"{parent.parent_id}_c{child_idx}",
                    parent_id=parent.parent_id,
                    doc_id=parent.doc_id,
                    collection_id=parent.collection_id,
                    filename=parent.filename,
                    page_number=parent.page_number,
                    language=language,
                    doc_type=doc_type,
                    breadcrumb=parent.breadcrumb,
                    content=prefixed_content,
                )
            )
            if end == len(words):
                break
            start += step
            child_idx += 1

        return child_records
