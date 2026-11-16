"""BaseJudgeEvaluator seam and canonical verdict/text normalization helpers."""

import re
from abc import ABC, abstractmethod
from typing import Literal

from eval_harness.schemas import EvalSample, JudgeVerdict, RAGOutput

_CITATION_TAG_RE = re.compile(
    r"\[[A-Za-z0-9_.\-]+\s*(?:[:,]\s*(?:page|p\.?|)\s*|page\s+)\d+\]"
)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?;\n])\s+|\n+")
_REFUSAL_MARKERS = (
    "do not contain sufficient verified evidence",
    "do not contain sufficient evidence",
    "cannot verify",
    "insufficient evidence",
    "not contain information",
    "tidak mengandungi bukti",
    "tiada bukti yang mencukupi",
    "tiada maklumat yang mencukupi",
    "tidak dapat disahkan",
    "maaf, dokumen",
)


def clamp_unit(value: float) -> float:
    """Clamp numeric scores strictly inside [0.0, 1.0]."""
    return round(min(1.0, max(0.0, float(value))), 4)


def strip_citation_tags(text: str) -> str:
    """Remove inline `[doc_id:pX]` citation tags prior to semantic/entailment grading."""
    return re.sub(r"\s+", " ", _CITATION_TAG_RE.sub("", text)).strip()


def split_claims(text: str) -> list[str]:
    """Split text into non-trivial atomic sentences/clauses across EN and BM."""
    cleaned = strip_citation_tags(text)
    if not cleaned:
        return []
    parts = [part.strip() for part in _SENTENCE_SPLIT_RE.split(cleaned)]
    return [part for part in parts if len(part) >= 4]


def is_refusal_response(output: RAGOutput) -> bool:
    """Return True when the target emitted GracefulRefusal or explicit refusal wording."""
    if output.refusal_triggered:
        return True
    lowered = output.answer.strip().lower()
    if not lowered:
        return False
    return any(marker in lowered for marker in _REFUSAL_MARKERS) and not output.citations


def build_verdict(
    *,
    sample_id: str,
    judge_name: str,
    faithfulness: float,
    answer_relevancy: float,
    context_recall: float,
    pass_threshold: float,
    latency_ms: float,
    input_tokens: int = 0,
    output_tokens: int = 0,
    estimated_cost_usd: float = 0.0,
    rationale: str | None = None,
) -> JudgeVerdict:
    """Compute canonical composite_score, binary PASS/FAIL decision, and construct JudgeVerdict."""
    f = clamp_unit(faithfulness)
    a = clamp_unit(answer_relevancy)
    c = clamp_unit(context_recall)
    composite = clamp_unit((f + a + c) / 3.0)
    decision: Literal["PASS", "FAIL"] = "PASS" if composite >= pass_threshold else "FAIL"
    return JudgeVerdict(
        sample_id=sample_id,
        judge_name=judge_name,
        faithfulness=f,
        answer_relevancy=a,
        context_recall=c,
        composite_score=composite,
        decision=decision,
        latency_ms=round(max(0.0, latency_ms), 3),
        input_tokens=max(0, input_tokens),
        output_tokens=max(0, output_tokens),
        estimated_cost_usd=round(max(0.0, estimated_cost_usd), 8),
        rationale=rationale,
    )


class BaseJudgeEvaluator(ABC):
    """Unified evaluator seam shared by generative LLMJudge and discriminative JEVJudge."""

    name: str

    @abstractmethod
    async def evaluate(
        self,
        sample: EvalSample,
        output: RAGOutput,
        pass_threshold: float = 0.70,
    ) -> JudgeVerdict:
        """Grade a single RAG output across Faithfulness, Answer Relevancy, and Context Recall."""
