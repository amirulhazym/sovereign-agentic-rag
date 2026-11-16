"""Benchmark statistics: latency percentiles, multi-run variance, and Cohen's Kappa."""

import math
import statistics
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from eval_harness.schemas import JudgeVerdict


def percentile(values: list[float], q: float) -> float:
    """Compute linear-interpolated q-th percentile in [0, 100]."""
    if not values:
        return 0.0
    ordered = sorted(float(v) for v in values)
    if len(ordered) == 1:
        return round(ordered[0], 3)
    clamped_q = min(100.0, max(0.0, q))
    rank = (clamped_q / 100.0) * (len(ordered) - 1)
    lo = math.floor(rank)
    hi = math.ceil(rank)
    if lo == hi:
        return round(ordered[lo], 3)
    frac = rank - lo
    return round(ordered[lo] + frac * (ordered[hi] - ordered[lo]), 3)


def compute_cohens_kappa(
    decisions_a: list[Literal["PASS", "FAIL"]],
    decisions_b: list[Literal["PASS", "FAIL"]],
) -> float:
    """Compute chance-corrected inter-judge agreement Cohen's Kappa for binary decisions."""
    if len(decisions_a) != len(decisions_b):
        raise ValueError(
            f"Decision sequences must have identical length ({len(decisions_a)} vs "
            f"{len(decisions_b)})."
        )
    total = len(decisions_a)
    if total == 0:
        return 0.0

    matches = sum(1 for a, b in zip(decisions_a, decisions_b, strict=True) if a == b)
    p_o = matches / total

    pass_a = sum(1 for a in decisions_a if a == "PASS") / total
    fail_a = 1.0 - pass_a
    pass_b = sum(1 for b in decisions_b if b == "PASS") / total
    fail_b = 1.0 - pass_b
    p_e = (pass_a * pass_b) + (fail_a * fail_b)

    denom = 1.0 - p_e
    if abs(denom) < 1e-12:
        return 1.0 if abs(p_o - 1.0) < 1e-12 else 0.0
    return round((p_o - p_e) / denom, 4)


class JudgeSummary(BaseModel):
    """Aggregated performance, cost, and variance statistics for one judge."""

    model_config = ConfigDict(frozen=True)

    judge_name: str
    sample_count: int
    runs_per_sample: int
    mean_faithfulness: float
    mean_answer_relevancy: float
    mean_context_recall: float
    mean_composite_score: float
    pass_rate: float
    mean_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    total_cost_usd: float
    total_input_tokens: int
    total_output_tokens: int
    mean_score_stddev: float


class BenchmarkReport(BaseModel):
    """Complete side-by-side LLM-as-a-Judge vs. JEV-as-a-Judge comparison payload."""

    model_config = ConfigDict(frozen=True)

    generated_at: str
    dataset_path: str
    sample_count: int
    runs_per_sample: int
    pass_threshold: float
    judge_summaries: list[JudgeSummary]
    raw_agreement_rate: float | None = None
    cohens_kappa: float | None = None
    disagreements: list[dict[str, Any]]
    verdicts_by_judge: dict[str, list[list[JudgeVerdict]]]


def _summarize_judge(
    judge_name: str,
    runs: list[list[JudgeVerdict]],
    sample_count: int,
) -> JudgeSummary:
    flat = [verdict for run in runs for verdict in run]
    if not flat:
        return JudgeSummary(
            judge_name=judge_name,
            sample_count=sample_count,
            runs_per_sample=len(runs),
            mean_faithfulness=0.0,
            mean_answer_relevancy=0.0,
            mean_context_recall=0.0,
            mean_composite_score=0.0,
            pass_rate=0.0,
            mean_latency_ms=0.0,
            p50_latency_ms=0.0,
            p95_latency_ms=0.0,
            total_cost_usd=0.0,
            total_input_tokens=0,
            total_output_tokens=0,
            mean_score_stddev=0.0,
        )

    by_sample: dict[str, list[float]] = {}
    for verdict in flat:
        by_sample.setdefault(verdict.sample_id, []).append(verdict.composite_score)

    sample_stddevs = [
        statistics.stdev(scores) if len(scores) >= 2 else 0.0 for scores in by_sample.values()
    ]
    latencies = [v.latency_ms for v in flat]
    pass_count = sum(1 for v in flat if v.decision == "PASS")

    return JudgeSummary(
        judge_name=judge_name,
        sample_count=sample_count,
        runs_per_sample=len(runs),
        mean_faithfulness=round(statistics.mean(v.faithfulness for v in flat), 4),
        mean_answer_relevancy=round(statistics.mean(v.answer_relevancy for v in flat), 4),
        mean_context_recall=round(statistics.mean(v.context_recall for v in flat), 4),
        mean_composite_score=round(statistics.mean(v.composite_score for v in flat), 4),
        pass_rate=round(pass_count / len(flat), 4),
        mean_latency_ms=round(statistics.mean(latencies), 3),
        p50_latency_ms=percentile(latencies, 50.0),
        p95_latency_ms=percentile(latencies, 95.0),
        total_cost_usd=round(sum(v.estimated_cost_usd for v in flat), 8),
        total_input_tokens=sum(v.input_tokens for v in flat),
        total_output_tokens=sum(v.output_tokens for v in flat),
        mean_score_stddev=round(statistics.mean(sample_stddevs), 6) if sample_stddevs else 0.0,
    )


def compute_benchmark_summary(
    *,
    dataset_path: str,
    sample_count: int,
    runs_per_sample: int,
    pass_threshold: float,
    verdicts_by_judge: dict[str, list[list[JudgeVerdict]]],
) -> BenchmarkReport:
    """Aggregate multi-run verdicts into per-judge summaries and inter-judge Kappa."""
    summaries = [
        _summarize_judge(name, runs, sample_count) for name, runs in verdicts_by_judge.items()
    ]

    raw_agreement: float | None = None
    kappa: float | None = None
    disagreements: list[dict[str, Any]] = []

    judge_names = list(verdicts_by_judge.keys())
    if len(judge_names) >= 2:
        name_a, name_b = judge_names[0], judge_names[1]
        runs_a = verdicts_by_judge[name_a]
        runs_b = verdicts_by_judge[name_b]
        decisions_a: list[Literal["PASS", "FAIL"]] = []
        decisions_b: list[Literal["PASS", "FAIL"]] = []

        for run_idx, (verdicts_run_a, verdicts_run_b) in enumerate(
            zip(runs_a, runs_b, strict=True), start=1
        ):
            map_b = {v.sample_id: v for v in verdicts_run_b}
            for va in verdicts_run_a:
                vb = map_b.get(va.sample_id)
                if vb is None:
                    continue
                decisions_a.append(va.decision)
                decisions_b.append(vb.decision)
                if va.decision != vb.decision:
                    disagreements.append(
                        {
                            "run": run_idx,
                            "sample_id": va.sample_id,
                            name_a: {
                                "decision": va.decision,
                                "composite_score": va.composite_score,
                            },
                            name_b: {
                                "decision": vb.decision,
                                "composite_score": vb.composite_score,
                            },
                        }
                    )

        if decisions_a:
            matches = sum(
                1 for da, db in zip(decisions_a, decisions_b, strict=True) if da == db
            )
            raw_agreement = round(matches / len(decisions_a), 4)
            kappa = compute_cohens_kappa(decisions_a, decisions_b)

    return BenchmarkReport(
        generated_at=datetime.now(UTC).isoformat(),
        dataset_path=dataset_path,
        sample_count=sample_count,
        runs_per_sample=runs_per_sample,
        pass_threshold=pass_threshold,
        judge_summaries=summaries,
        raw_agreement_rate=raw_agreement,
        cohens_kappa=kappa,
        disagreements=disagreements,
        verdicts_by_judge=verdicts_by_judge,
    )


def render_markdown_report(report: BenchmarkReport) -> str:
    """Format a publication-ready Markdown benchmark report comparing judges side-by-side."""
    lines = [
        "# Evaluation Benchmark Report: `LLM-as-a-Judge` vs. `JEV-as-a-Judge`",
        "",
        f"- **Generated At (UTC):** `{report.generated_at}`",
        f"- **Dataset:** `{report.dataset_path}` (`{report.sample_count}` samples)",
        f"- **Iterations per Sample (`--runs`):** `{report.runs_per_sample}`",
        f"- **Pass Threshold:** `{report.pass_threshold:.2f}`",
        "",
        "## 1. Side-by-Side Judge Performance & Reliability",
        "",
        "| Judge | Composite | Faithfulness | Relevancy | Context Recall | Pass Rate | "
        "Mean Latency (ms) | P50 / P95 (ms) | Total Cost ($) | Score StdDev (sigma) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for s in report.judge_summaries:
        lines.append(
            f"| `{s.judge_name}` | {s.mean_composite_score:.4f} | {s.mean_faithfulness:.4f} | "
            f"{s.mean_answer_relevancy:.4f} | {s.mean_context_recall:.4f} | "
            f"{s.pass_rate * 100:.1f}% | {s.mean_latency_ms:.2f} | "
            f"{s.p50_latency_ms:.1f} / {s.p95_latency_ms:.1f} | "
            f"${s.total_cost_usd:.6f} | `{s.mean_score_stddev:.6f}` |"
        )

    lines.extend(["", "## 2. Inter-Judge Agreement", ""])
    if report.raw_agreement_rate is not None and report.cohens_kappa is not None:
        lines.append(
            f"- **Raw Agreement Rate:** `{report.raw_agreement_rate * 100:.2f}%`"
        )
        lines.append(f"- **Cohen's Kappa ($\\kappa$):** `{report.cohens_kappa:.4f}`")
    else:
        lines.append("- Single-judge run; pairwise Cohen's Kappa requires at least 2 judges.")

    lines.extend(["", "## 3. Binary Decision Disagreements", ""])
    if not report.disagreements:
        lines.append("No `PASS` / `FAIL` disagreements observed across evaluated runs.")
    else:
        lines.append(f"Detected `{len(report.disagreements)}` disagreement(s):")
        lines.append("")
        for item in report.disagreements[:20]:
            keys = [k for k in item if k not in {"run", "sample_id"}]
            summary_parts = [
                f"{k}={item[k]['decision']} ({item[k]['composite_score']:.3f})" for k in keys
            ]
            lines.append(
                f"- Run {item['run']} · `{item['sample_id']}`: " + ", ".join(summary_parts)
            )
    lines.append("")
    return "\n".join(lines)
