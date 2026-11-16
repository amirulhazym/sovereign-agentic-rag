"""Unit and CLI tests for benchmark statistics (Cohen's Kappa, Variance) and runner."""

import json
from pathlib import Path

import pytest

from eval_harness.judges.base import BaseJudgeEvaluator, build_verdict
from eval_harness.metrics import (
    BenchmarkReport,
    compute_benchmark_summary,
    compute_cohens_kappa,
    percentile,
    render_markdown_report,
)
from eval_harness.runner import main, run_benchmark
from eval_harness.schemas import EvalSample, JudgeVerdict, RAGOutput, load_golden_dataset
from eval_harness.target_adapter import RecordedTraceTargetAdapter


def test_percentile_handles_empty_single_and_interpolated_series() -> None:
    assert percentile([], 50.0) == 0.0
    assert percentile([42.0], 95.0) == 42.0
    assert percentile([10.0, 20.0, 30.0, 40.0, 50.0], 50.0) == 30.0
    assert percentile([10.0, 20.0, 30.0, 40.0, 50.0], 95.0) == pytest.approx(48.0)


def test_cohens_kappa_edge_cases() -> None:
    # Perfect unanimous PASS agreement must handle 1 - p_e == 0 gracefully -> 1.0
    assert compute_cohens_kappa(["PASS", "PASS", "PASS"], ["PASS", "PASS", "PASS"]) == 1.0

    # Mixed perfect agreement
    assert compute_cohens_kappa(
        ["PASS", "FAIL", "PASS", "FAIL"],
        ["PASS", "FAIL", "PASS", "FAIL"],
    ) == pytest.approx(1.0)

    # Chance-level independent agreement -> 0.0
    assert compute_cohens_kappa(
        ["PASS", "PASS", "FAIL", "FAIL"],
        ["PASS", "FAIL", "PASS", "FAIL"],
    ) == pytest.approx(0.0)

    # Complete opposite disagreement
    assert compute_cohens_kappa(
        ["PASS", "PASS", "FAIL", "FAIL"],
        ["FAIL", "FAIL", "PASS", "PASS"],
    ) == pytest.approx(-1.0)

    with pytest.raises(ValueError):
        compute_cohens_kappa(["PASS"], ["PASS", "FAIL"])


def test_compute_benchmark_summary_calculates_variance_cost_and_kappa() -> None:
    sample_ids = ["s1", "s2"]
    verdicts_by_judge: dict[str, list[list[JudgeVerdict]]] = {
        "LLMJudge": [
            [
                build_verdict(
                    sample_id="s1",
                    judge_name="LLMJudge",
                    faithfulness=0.90,
                    answer_relevancy=0.90,
                    context_recall=0.90,
                    pass_threshold=0.70,
                    latency_ms=800.0,
                    input_tokens=1400,
                    output_tokens=100,
                    estimated_cost_usd=0.0003,
                ),
                build_verdict(
                    sample_id="s2",
                    judge_name="LLMJudge",
                    faithfulness=0.40,
                    answer_relevancy=0.40,
                    context_recall=0.40,
                    pass_threshold=0.70,
                    latency_ms=900.0,
                    input_tokens=1400,
                    output_tokens=100,
                    estimated_cost_usd=0.0003,
                ),
            ],
            [
                build_verdict(
                    sample_id="s1",
                    judge_name="LLMJudge",
                    faithfulness=0.81,
                    answer_relevancy=0.81,
                    context_recall=0.81,
                    pass_threshold=0.70,
                    latency_ms=820.0,
                    input_tokens=1400,
                    output_tokens=100,
                    estimated_cost_usd=0.0003,
                ),
                build_verdict(
                    sample_id="s2",
                    judge_name="LLMJudge",
                    faithfulness=0.46,
                    answer_relevancy=0.46,
                    context_recall=0.46,
                    pass_threshold=0.70,
                    latency_ms=910.0,
                    input_tokens=1400,
                    output_tokens=100,
                    estimated_cost_usd=0.0003,
                ),
            ],
        ],
        "JEVJudge": [
            [
                build_verdict(
                    sample_id="s1",
                    judge_name="JEVJudge",
                    faithfulness=0.92,
                    answer_relevancy=0.92,
                    context_recall=0.92,
                    pass_threshold=0.70,
                    latency_ms=40.0,
                ),
                build_verdict(
                    sample_id="s2",
                    judge_name="JEVJudge",
                    faithfulness=0.35,
                    answer_relevancy=0.35,
                    context_recall=0.35,
                    pass_threshold=0.70,
                    latency_ms=45.0,
                ),
            ],
            [
                build_verdict(
                    sample_id="s1",
                    judge_name="JEVJudge",
                    faithfulness=0.92,
                    answer_relevancy=0.92,
                    context_recall=0.92,
                    pass_threshold=0.70,
                    latency_ms=38.0,
                ),
                build_verdict(
                    sample_id="s2",
                    judge_name="JEVJudge",
                    faithfulness=0.35,
                    answer_relevancy=0.35,
                    context_recall=0.35,
                    pass_threshold=0.70,
                    latency_ms=42.0,
                ),
            ],
        ],
    }

    report = compute_benchmark_summary(
        dataset_path="datasets/golden_dataset.jsonl",
        sample_count=len(sample_ids),
        runs_per_sample=2,
        pass_threshold=0.70,
        verdicts_by_judge=verdicts_by_judge,
    )
    assert isinstance(report, BenchmarkReport)
    assert report.raw_agreement_rate == pytest.approx(1.0)
    assert report.cohens_kappa == pytest.approx(1.0)

    llm_summary, jev_summary = report.judge_summaries
    assert llm_summary.total_cost_usd == pytest.approx(0.0012)
    assert llm_summary.mean_score_stddev > 0.04
    assert jev_summary.total_cost_usd == 0.0
    assert jev_summary.mean_score_stddev == 0.0

    md = render_markdown_report(report)
    assert "Cohen's Kappa" in md
    assert "JEVJudge" in md
    assert "LLMJudge" in md


def _stage_subset_and_trace(tmp_path: Path) -> tuple[Path, Path]:
    dataset_path = Path(__file__).resolve().parents[1] / "datasets" / "golden_dataset.jsonl"
    samples = load_golden_dataset(dataset_path)[:4]

    sub_dataset = tmp_path / "mini_golden.jsonl"
    sub_dataset.write_text(
        "\n".join(sample.model_dump_json() for sample in samples) + "\n",
        encoding="utf-8",
    )
    trace_path = tmp_path / "trace.jsonl"
    trace_path.write_text(
        "\n".join(
            RAGOutput(
                sample_id=s.id,
                answer=s.ground_truth_answer,
                retrieved_contexts=s.ground_truth_contexts,
                citations=[f"[{c.doc_id}:p{c.page}]" for c in s.expected_citations],
                refusal_triggered=(s.query_type == "unanswerable_refusal"),
                target_latency_ms=110.0,
                cache_hit_tokens=256,
            ).model_dump_json()
            for s in samples
        )
        + "\n",
        encoding="utf-8",
    )
    return sub_dataset, trace_path


async def test_run_benchmark_aggregates_multi_run_verdicts(tmp_path: Path) -> None:
    sub_dataset, trace_path = _stage_subset_and_trace(tmp_path)

    class _StubJudge(BaseJudgeEvaluator):
        def __init__(self, name: str, score: float) -> None:
            self.name = name
            self._score = score

        async def evaluate(
            self,
            sample: EvalSample,
            output: RAGOutput,
            pass_threshold: float = 0.70,
        ) -> JudgeVerdict:
            return build_verdict(
                sample_id=sample.id,
                judge_name=self.name,
                faithfulness=self._score,
                answer_relevancy=self._score,
                context_recall=self._score,
                pass_threshold=pass_threshold,
                latency_ms=12.0,
            )

    report = await run_benchmark(
        dataset_path=sub_dataset,
        target_adapter=RecordedTraceTargetAdapter.from_jsonl(trace_path),
        judges=[_StubJudge("StubLLM", 0.90), _StubJudge("StubJEV", 0.88)],
        runs=3,
        pass_threshold=0.70,
    )
    assert report.sample_count == 4
    assert report.runs_per_sample == 3
    assert report.cohens_kappa == 1.0


def test_cli_main_writes_md_and_json_reports(tmp_path: Path) -> None:
    sub_dataset, trace_path = _stage_subset_and_trace(tmp_path)

    out_dir = tmp_path / "reports"
    exit_code = main(
        [
            "--dataset",
            str(sub_dataset),
            "--trace-file",
            str(trace_path),
            "--runs",
            "3",
            "--jev-backend",
            "lexical",
            "--skip-llm",
            "--output-dir",
            str(out_dir),
        ]
    )
    assert exit_code == 0
    md_path = out_dir / "benchmark_report.md"
    json_path = out_dir / "benchmark_report.json"
    assert md_path.is_file()
    assert json_path.is_file()
    parsed = json.loads(json_path.read_text(encoding="utf-8"))
    assert parsed["sample_count"] == 4
    assert parsed["runs_per_sample"] == 3
    assert parsed["judge_summaries"][0]["mean_score_stddev"] == 0.0
