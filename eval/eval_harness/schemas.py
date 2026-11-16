"""Pydantic v2 contracts and JSONL loader for the decoupled evaluation harness."""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ExpectedCitation(BaseModel):
    """Ground-truth source citation expected for an answerable EvalSample."""

    model_config = ConfigDict(frozen=True)

    doc_id: str = Field(min_length=1)
    page: int = Field(ge=1)


class EvalSample(BaseModel):
    """Single curated bilingual benchmark case in golden_dataset.jsonl."""

    model_config = ConfigDict(frozen=True)

    id: str = Field(min_length=1)
    collection: Literal["it_infra_sops", "malaysian_enterprise_policy"]
    language: Literal["en", "ms", "mixed"]
    query_type: Literal[
        "factual",
        "table_lookup",
        "multi_doc_compare",
        "acronym_clause",
        "unanswerable_refusal",
    ]
    question: str = Field(min_length=1)
    ground_truth_answer: str = Field(min_length=1)
    expected_citations: list[ExpectedCitation]
    ground_truth_contexts: list[str]


class RAGOutput(BaseModel):
    """Normalized response collected from any RAGTargetAdapter."""

    model_config = ConfigDict(frozen=True)

    sample_id: str
    answer: str
    retrieved_contexts: list[str]
    citations: list[str]
    refusal_triggered: bool
    target_latency_ms: float = Field(ge=0.0)
    cache_hit_tokens: int = Field(default=0, ge=0)


class JudgeVerdict(BaseModel):
    """Standardized verdict produced by every BaseJudgeEvaluator."""

    model_config = ConfigDict(frozen=True)

    sample_id: str
    judge_name: str
    faithfulness: float = Field(ge=0.0, le=1.0)
    answer_relevancy: float = Field(ge=0.0, le=1.0)
    context_recall: float = Field(ge=0.0, le=1.0)
    composite_score: float = Field(ge=0.0, le=1.0)
    decision: Literal["PASS", "FAIL"]
    latency_ms: float = Field(ge=0.0)
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    estimated_cost_usd: float = Field(default=0.0, ge=0.0)
    rationale: str | None = None


def load_golden_dataset(path: Path) -> list[EvalSample]:
    """Read and validate JSONL evaluation samples, rejecting duplicate IDs."""
    if not path.is_file():
        raise FileNotFoundError(f"Golden dataset not found: {path}")

    samples: list[EvalSample] = []
    seen_ids: set[str] = set()
    lines = path.read_text(encoding="utf-8").splitlines()
    for line_no, raw_line in enumerate(lines, start=1):
        stripped = raw_line.strip()
        if not stripped:
            continue
        try:
            sample = EvalSample.model_validate_json(stripped)
        except Exception as exc:
            raise ValueError(f"Invalid EvalSample on line {line_no} of {path}: {exc}") from exc
        if sample.id in seen_ids:
            raise ValueError(f"Duplicate sample id '{sample.id}' on line {line_no}")
        seen_ids.add(sample.id)
        samples.append(sample)
    return samples
