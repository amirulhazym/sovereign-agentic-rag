"""Tests for structure-aware PyMuPDF4LLM extraction and hierarchical Parent-Child chunking."""

import pymupdf
import pytest

from app.core.exceptions import UnsupportedDocumentTypeError
from app.services.chunking import (
    ExtractedPage,
    HierarchicalChunker,
    detect_language,
    extract_pages,
)


def _build_sample_pdf_bytes() -> bytes:
    """Create a real 2-page PDF in memory to exercise PyMuPDF4LLM extraction."""
    doc = pymupdf.open()
    page1 = doc.new_page()
    page1.insert_text(
        (72, 72),
        "NVMe Maintenance SOP\n\n"
        "Section 1.1 Pre-Check\n"
        "Verify bay LED is solid amber and check nvme list before drive pull.\n"
        "Error ERR_NVME_0x04 indicates PCIe link degradation.",
        fontsize=11,
    )
    page2 = doc.new_page()
    page2.insert_text(
        (72, 72),
        "Section 2.1 Thermal Limits\n"
        "Ambient inlet temperature must remain at or below 35C during fan swap.",
        fontsize=11,
    )
    return doc.tobytes()


def test_extract_pages_parses_pdf_markdown_and_rejects_unsupported() -> None:
    """Verify real PDF page extraction, Markdown page markers, and media type guards."""
    pdf_pages = extract_pages(_build_sample_pdf_bytes(), filename="nvme_sop.pdf")
    assert len(pdf_pages) == 2
    assert pdf_pages[0].page_number == 1
    assert "ERR_NVME_0x04" in pdf_pages[0].markdown
    assert pdf_pages[1].page_number == 2
    assert "35C" in pdf_pages[1].markdown

    md_content = (
        b"# HR Policy\nIntro text\n<!-- page: 2 -->\n## Overtime\nKlausa 4.2 bayaran 2x."
    )
    md_pages = extract_pages(md_content, filename="hr_policy.md")
    assert [p.page_number for p in md_pages] == [1, 2]
    assert "Klausa 4.2" in md_pages[1].markdown

    with pytest.raises(UnsupportedDocumentTypeError):
        extract_pages(b"binary", filename="malware.exe")


def test_detect_language_distinguishes_en_ms_and_mixed() -> None:
    """Verify heuristic language detector identifies EN, BM, and BM-EN code-switching."""
    assert (
        detect_language(
            "Verify the server chassis airflow and ensure the temperature stays below threshold."
        )
        == "en"
    )
    assert (
        detect_language(
            "Pekerja layak menerima bayaran balik atau cuti gantian mengikut peraturan syarikat."
        )
        == "ms"
    )
    assert (
        detect_language(
            "Untuk kerja pada public holiday, pekerja boleh tuntut overtime melalui portal HRIS."
        )
        == "mixed"
    )


def test_hierarchical_chunker_builds_parent_and_breadcrumb_child_chunks() -> None:
    """Verify ParentChunks split by heading and ChildChunks prepend Breadcrumbs with overlap."""
    long_section = " ".join(f"token{i}" for i in range(650))
    pages = [
        ExtractedPage(
            page_number=4,
            markdown=(
                "## Storage Maintenance\n"
                "### NVMe Hot-Swap\n"
                f"{long_section}\n\n"
                "### Thermal Table\n"
                "| Component | Max Temp |\n"
                "| --- | --- |\n"
                "| NVMe U.2 | 70C |"
            ),
        )
    ]

    chunker = HierarchicalChunker(
        parent_max_tokens=1200,
        child_max_tokens=300,
        child_overlap_tokens=50,
    )
    parents, children = chunker.chunk_document(
        pages=pages,
        doc_id="nvme_sop",
        collection_id="it_infra_sops",
        filename="nvme_sop.pdf",
        doc_type="pdf",
        language="en",
    )

    assert len(parents) == 2
    assert parents[0].section_header == "NVMe Hot-Swap"
    assert parents[0].breadcrumb == "nvme_sop.pdf | Page 4 > NVMe Hot-Swap"
    assert parents[1].section_header == "Thermal Table"
    assert "| NVMe U.2 | 70C |" in parents[1].content

    # Long 650-token section splits into multiple overlapping child chunks (~300 tok, 50 overlap)
    hot_swap_children = [c for c in children if c.parent_id == parents[0].parent_id]
    assert len(hot_swap_children) >= 3
    for child in hot_swap_children:
        assert child.content.startswith("[nvme_sop.pdf | Page 4 > NVMe Hot-Swap]\n")
        assert child.collection_id == "it_infra_sops"
        assert child.doc_id == "nvme_sop"
        assert child.page_number == 4
        assert child.language == "en"
        assert child.doc_type == "pdf"

    # Verify 50-token overlap continuity between consecutive children
    first_words = hot_swap_children[0].content.split("\n", 1)[1].split()
    second_words = hot_swap_children[1].content.split("\n", 1)[1].split()
    assert first_words[-50:] == second_words[:50]
