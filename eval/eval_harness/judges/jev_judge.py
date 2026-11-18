"""Deterministic, zero-cost JEV-as-a-Judge using local NLI Cross-Encoder verification."""

import json
import math
import os
import re
import time
from collections.abc import Callable
from typing import Any, Protocol

import numpy as np

from eval_harness.judges.base import (
    BaseJudgeEvaluator,
    build_verdict,
    is_refusal_response,
    split_claims,
    strip_citation_tags,
)
from eval_harness.schemas import EvalSample, JudgeVerdict, RAGOutput

DEFAULT_NLI_MODEL_REPO = "Xenova/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
_TOKEN_RE = re.compile(r"[a-z0-9]+")


class NLIScorerBackend(Protocol):
    """Discriminative scoring seam supporting ONNX NLI models and deterministic test fakes."""

    def score_entailment(self, pairs: list[tuple[str, str]]) -> list[float]:
        """Return P(entailment | premise, hypothesis) in [0.0, 1.0] for each pair."""

    def score_relevance(self, pairs: list[tuple[str, str]]) -> list[float]:
        """Return discriminative relevance probability in [0.0, 1.0] for (question, answer)."""


def _default_onnx_loader(
    repo_id: str,
) -> tuple[Any, Any, dict[str, int]]:
    """Lazy-load quantized ONNX Runtime NLI session + fast Rust tokenizer from HF Hub."""
    import onnxruntime as ort
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer

    cache_dir = os.getenv("HF_HOME") or None
    tokenizer_path = hf_hub_download(
        repo_id=repo_id, filename="tokenizer.json", cache_dir=cache_dir
    )
    tokenizer = Tokenizer.from_file(tokenizer_path)

    label_map: dict[str, int] = {"entailment": 0, "neutral": 1, "contradiction": 2}
    try:
        config_path = hf_hub_download(
            repo_id=repo_id, filename="config.json", cache_dir=cache_dir
        )
        config_data = json.loads(open(config_path, encoding="utf-8").read())
        raw_id2label = config_data.get("id2label", {})
        if isinstance(raw_id2label, dict) and raw_id2label:
            parsed_map = {str(lbl).lower(): int(idx) for idx, lbl in raw_id2label.items()}
            if "entailment" in parsed_map:
                label_map = parsed_map
    except Exception:
        pass

    model_path: str | None = None
    for candidate in ("onnx/model_quantized.onnx", "onnx/model.onnx", "model.onnx"):
        try:
            model_path = hf_hub_download(
                repo_id=repo_id, filename=candidate, cache_dir=cache_dir
            )
            break
        except Exception:
            continue
    if model_path is None:
        raise FileNotFoundError(f"Could not locate ONNX model artifact inside '{repo_id}'.")

    session_opts = ort.SessionOptions()
    session_opts.intra_op_num_threads = 2
    session = ort.InferenceSession(
        model_path,
        sess_options=session_opts,
        providers=["CPUExecutionProvider"],
    )
    return session, tokenizer, label_map


class ONNXMultilingualNLIBackend:
    """Lazy-loaded ONNX NLI & cross-encoder scorer (`$0.00`, deterministic on CPU)."""

    def __init__(
        self,
        *,
        repo_id: str = DEFAULT_NLI_MODEL_REPO,
        max_length: int = 512,
        loader: Callable[[str], tuple[Any, Any, dict[str, int]]] | None = None,
    ) -> None:
        self._repo_id = repo_id
        self._max_length = max_length
        self._loader = loader or _default_onnx_loader
        self._session: Any = None
        self._tokenizer: Any = None
        self._label_map: dict[str, int] = {}

    def _ensure_loaded(self) -> tuple[Any, Any, dict[str, int]]:
        if self._session is None or self._tokenizer is None:
            session, tokenizer, label_map = self._loader(self._repo_id)
            tokenizer.enable_truncation(max_length=self._max_length)
            tokenizer.enable_padding(pad_id=0)
            self._session = session
            self._tokenizer = tokenizer
            self._label_map = label_map
        return self._session, self._tokenizer, self._label_map

    def _infer_logits(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        session, tokenizer, _ = self._ensure_loaded()
        encoded = tokenizer.encode_batch(pairs)
        input_ids = np.array([enc.ids for enc in encoded], dtype=np.int64)
        attention_mask = np.array([enc.attention_mask for enc in encoded], dtype=np.int64)
        feed: dict[str, np.ndarray] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        get_inputs = getattr(session, "get_inputs", None)
        if callable(get_inputs):
            expected_names = {inp.name for inp in get_inputs()}
            if "token_type_ids" in expected_names:
                feed["token_type_ids"] = np.array(
                    [enc.type_ids for enc in encoded], dtype=np.int64
                )
        logits = session.run(None, feed)[0]
        return np.asarray(logits, dtype=np.float64)

    def score_entailment(self, pairs: list[tuple[str, str]]) -> list[float]:
        """Score entailment probability normalized across {entailment, contradiction}."""
        if not pairs:
            return []
        logits = self._infer_logits(pairs)
        if logits.ndim == 2 and logits.shape[1] == 1:
            return [round(1.0 / (1.0 + math.exp(-float(val))), 4) for val in logits[:, 0]]

        ent_idx = self._label_map.get("entailment", 0)
        contra_idx = self._label_map.get("contradiction", 2)
        scores: list[float] = []
        for row in logits:
            pair_logits = np.array([row[ent_idx], row[contra_idx]], dtype=np.float64)
            pair_logits -= np.max(pair_logits)
            exp_vals = np.exp(pair_logits)
            prob_entail = float(exp_vals[0] / (exp_vals[0] + exp_vals[1]))
            scores.append(round(prob_entail, 4))
        return scores

    def score_relevance(self, pairs: list[tuple[str, str]]) -> list[float]:
        """Compute P(not contradiction) from NLI logits or sigmoid for 1-logit cross-encoders."""
        if not pairs:
            return []
        logits = self._infer_logits(pairs)
        if logits.ndim == 2 and logits.shape[1] == 1:
            return [round(1.0 / (1.0 + math.exp(-float(val))), 4) for val in logits[:, 0]]

        contra_idx = self._label_map.get("contradiction", 2)
        scores: list[float] = []
        for row in logits:
            shifted = row - np.max(row)
            probs = np.exp(shifted) / np.sum(np.exp(shifted))
            # A relevant answer neither contradicts nor stays orthogonal to the question
            score = max(0.0, min(1.0, float(1.0 - probs[contra_idx])))
            scores.append(round(score, 4))
        return scores


class LexicalDiscriminativeBackend:
    """Zero-download deterministic discriminative scorer for air-gapped CI smoke checks."""

    @staticmethod
    def _tokens(text: str) -> set[str]:
        return set(_TOKEN_RE.findall(text.lower()))

    def score_entailment(self, pairs: list[tuple[str, str]]) -> list[float]:
        scores: list[float] = []
        for premise, hypothesis in pairs:
            h_toks = self._tokens(hypothesis)
            if not h_toks:
                scores.append(0.0)
                continue
            p_toks = self._tokens(premise)
            recall = len(p_toks & h_toks) / len(h_toks)
            # Soft calibration toward high precision matches without LLM stochasticity
            scores.append(round(min(1.0, max(0.0, recall)), 4))
        return scores

    def score_relevance(self, pairs: list[tuple[str, str]]) -> list[float]:
        scores: list[float] = []
        for question, answer in pairs:
            q_toks = self._tokens(question)
            a_toks = self._tokens(answer)
            if not q_toks or not a_toks:
                scores.append(0.0)
                continue
            overlap = len(q_toks & a_toks) / len(q_toks)
            length_bonus = 0.35 if len(a_toks) >= 6 else 0.1
            scores.append(round(min(1.0, overlap * 0.7 + length_bonus), 4))
        return scores


class JEVJudgeEvaluator(BaseJudgeEvaluator):
    """Non-generative discriminative evaluator (`$0.00` cost, `0.0` stochastic variance)."""

    def __init__(
        self,
        *,
        scorer: NLIScorerBackend | None = None,
        name: str = "JEVJudge(mDeBERTa-v3-ONNX)",
    ) -> None:
        self._scorer: NLIScorerBackend = scorer or ONNXMultilingualNLIBackend()
        self.name = name

    def _max_entailment_over_contexts(
        self, claims: list[str], contexts: list[str]
    ) -> float:
        if not claims or not contexts:
            return 0.0
        pairs = [(ctx, claim) for claim in claims for ctx in contexts]
        raw_scores = self._scorer.score_entailment(pairs)
        stride = len(contexts)
        claim_scores = [
            max(raw_scores[idx * stride : (idx + 1) * stride]) for idx in range(len(claims))
        ]
        return sum(claim_scores) / len(claim_scores)

    async def evaluate(
        self,
        sample: EvalSample,
        output: RAGOutput,
        pass_threshold: float = 0.70,
    ) -> JudgeVerdict:
        """Deterministically evaluate faithfulness, relevancy, and recall without LLM generation."""
        started_at = time.perf_counter()
        refused = is_refusal_response(output)

        if sample.query_type == "unanswerable_refusal":
            latency_ms = (time.perf_counter() - started_at) * 1000.0
            if refused:
                return build_verdict(
                    sample_id=sample.id,
                    judge_name=self.name,
                    faithfulness=1.0,
                    answer_relevancy=1.0,
                    context_recall=1.0,
                    pass_threshold=pass_threshold,
                    latency_ms=latency_ms,
                    rationale="Verified GracefulRefusal on unanswerable trap query.",
                )
            return build_verdict(
                sample_id=sample.id,
                judge_name=self.name,
                faithfulness=0.0,
                answer_relevancy=0.0,
                context_recall=0.0,
                pass_threshold=pass_threshold,
                latency_ms=latency_ms,
                rationale="Failed unanswerable trap: answered without GracefulRefusal.",
            )

        if output.refusal_triggered:
            latency_ms = (time.perf_counter() - started_at) * 1000.0
            return build_verdict(
                sample_id=sample.id,
                judge_name=self.name,
                faithfulness=1.0,
                answer_relevancy=0.0,
                context_recall=0.0,
                pass_threshold=pass_threshold,
                latency_ms=latency_ms,
                rationale="False refusal triggered on an answerable sample.",
            )

        answer_claims = split_claims(output.answer)
        reference_claims = split_claims(sample.ground_truth_answer)
        clean_answer = strip_citation_tags(output.answer)

        faithfulness = self._max_entailment_over_contexts(answer_claims, output.retrieved_contexts)
        context_recall = self._max_entailment_over_contexts(
            reference_claims, output.retrieved_contexts
        )
        relevancy = (
            self._scorer.score_relevance([(sample.question, clean_answer)])[0]
            if clean_answer
            else 0.0
        )

        latency_ms = (time.perf_counter() - started_at) * 1000.0
        return build_verdict(
            sample_id=sample.id,
            judge_name=self.name,
            faithfulness=faithfulness,
            answer_relevancy=relevancy,
            context_recall=context_recall,
            pass_threshold=pass_threshold,
            latency_ms=latency_ms,
            input_tokens=0,
            output_tokens=0,
            estimated_cost_usd=0.0,
            rationale=(
                f"NLI claims={len(answer_claims)}, refs={len(reference_claims)}, "
                f"contexts={len(output.retrieved_contexts)}."
            ),
        )
