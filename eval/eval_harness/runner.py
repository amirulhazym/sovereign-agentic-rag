"""CLI benchmark runner orchestrating RAGTargetAdapter and dual Judge evaluations."""

import argparse
import asyncio
import os
from pathlib import Path
from typing import Literal

from eval_harness.judges.base import BaseJudgeEvaluator
from eval_harness.judges.jev_judge import (
    JEVJudgeEvaluator,
    LexicalDiscriminativeBackend,
    NLIScorerBackend,
    ONNXMultilingualNLIBackend,
)
from eval_harness.judges.llm_judge import LLMJudgeEvaluator
from eval_harness.metrics import (
    BenchmarkReport,
    compute_benchmark_summary,
    render_markdown_report,
)
from eval_harness.schemas import JudgeVerdict, RAGOutput, load_golden_dataset
from eval_harness.target_adapter import (
    HTTPTargetAdapter,
    RAGTargetAdapter,
    RecordedTraceTargetAdapter,
)


async def run_benchmark(
    *,
    dataset_path: Path,
    target_adapter: RAGTargetAdapter,
    judges: list[BaseJudgeEvaluator],
    runs: int = 3,
    pass_threshold: float = 0.70,
) -> BenchmarkReport:
    """Fetch RAGOutput once per sample, then evaluate each judge across `runs` iterations."""
    if runs < 1:
        raise ValueError(f"--runs must be >= 1, got {runs}")
    if not judges:
        raise ValueError("At least one judge evaluator must be provided.")

    samples = load_golden_dataset(dataset_path)
    outputs: list[RAGOutput] = []
    for sample in samples:
        outputs.append(await target_adapter.evaluate_sample(sample))

    verdicts_by_judge: dict[str, list[list[JudgeVerdict]]] = {}
    for judge in judges:
        judge_runs: list[list[JudgeVerdict]] = []
        for _run_idx in range(runs):
            run_verdicts: list[JudgeVerdict] = []
            for sample, output in zip(samples, outputs, strict=True):
                verdict = await judge.evaluate(
                    sample=sample,
                    output=output,
                    pass_threshold=pass_threshold,
                )
                run_verdicts.append(verdict)
            judge_runs.append(run_verdicts)
        verdicts_by_judge[judge.name] = judge_runs

    return compute_benchmark_summary(
        dataset_path=str(dataset_path),
        sample_count=len(samples),
        runs_per_sample=runs,
        pass_threshold=pass_threshold,
        verdicts_by_judge=verdicts_by_judge,
    )


def write_report_artifacts(report: BenchmarkReport, output_dir: Path) -> tuple[Path, Path]:
    """Persist benchmark_report.md and benchmark_report.json to output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)
    md_path = output_dir / "benchmark_report.md"
    json_path = output_dir / "benchmark_report.json"
    md_path.write_text(render_markdown_report(report), encoding="utf-8")
    json_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    return md_path, json_path


def build_cli_parser() -> argparse.ArgumentParser:
    """Create CLI argument parser for `python -m eval_harness.runner`."""
    parser = argparse.ArgumentParser(
        prog="eval_harness.runner",
        description="Benchmark LLM-as-a-Judge vs. JEV-as-a-Judge on sovereign-agentic-rag.",
    )
    parser.add_argument(
        "--dataset",
        default="datasets/golden_dataset.jsonl",
        help="Path to JSONL golden dataset (default: datasets/golden_dataset.jsonl).",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Number of repeated judge evaluations per sample to measure StdDev variance.",
    )
    parser.add_argument(
        "--pass-threshold",
        type=float,
        default=0.70,
        help="Minimum composite score in [0, 1] required for PASS verdict.",
    )
    parser.add_argument(
        "--target-url",
        default="http://localhost:8000",
        help="FastAPI backend base URL for HTTPTargetAdapter.",
    )
    parser.add_argument(
        "--trace-file",
        default="",
        help="Optional pre-recorded RAGOutput JSONL trace file (bypasses live HTTP target).",
    )
    parser.add_argument(
        "--inference-mode",
        choices=["LOCAL_ONPREM", "CLOUD_API"],
        default="LOCAL_ONPREM",
        help="Inference mode sent to backend /api/v1/chat/stream when querying live.",
    )
    parser.add_argument(
        "--llm-backend",
        choices=["ollama", "deepseek"],
        default="ollama",
        help="Generative LLMJudge backend.",
    )
    parser.add_argument(
        "--llm-model",
        default="",
        help="Override LLMJudge model name (defaults cleanly per --llm-backend).",
    )
    parser.add_argument(
        "--llm-base-url",
        default="",
        help="Override base URL for LLMJudge backend.",
    )
    parser.add_argument(
        "--skip-llm",
        action="store_true",
        help="Skip generative LLMJudge (useful for quick local JEVJudge-only runs).",
    )
    parser.add_argument(
        "--jev-backend",
        choices=["onnx", "lexical"],
        default="onnx",
        help="Discriminative JEVJudge verifier backend.",
    )
    parser.add_argument(
        "--jev-model",
        default="Xenova/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7",
        help="Hugging Face ONNX repo ID for JEVJudge when --jev-backend=onnx.",
    )
    parser.add_argument(
        "--output-dir",
        default="reports",
        help="Directory where benchmark_report.md and benchmark_report.json are saved.",
    )
    return parser


def _build_judges_from_args(args: argparse.Namespace) -> list[BaseJudgeEvaluator]:
    judges: list[BaseJudgeEvaluator] = []
    if not args.skip_llm:
        backend: Literal["ollama", "deepseek"] = args.llm_backend
        default_model = (
            os.getenv("LOCAL_LLM_MODEL", "hf.co/empero-ai/Qwen3.8-4B-Distill-GGUF:Q4_K_M")
            if backend == "ollama"
            else "deepseek-chat"
        )
        judges.append(
            LLMJudgeEvaluator(
                backend=backend,
                model=args.llm_model or default_model,
                base_url=args.llm_base_url or None,
                api_key=os.getenv("DEEPSEEK_API_KEY", ""),
            )
        )

    scorer: NLIScorerBackend
    if args.jev_backend == "lexical":
        scorer = LexicalDiscriminativeBackend()
        jev_name = "JEVJudge(lexical-deterministic)"
    else:
        scorer = ONNXMultilingualNLIBackend(repo_id=args.jev_model)
        jev_name = f"JEVJudge({args.jev_model})"
    judges.append(JEVJudgeEvaluator(scorer=scorer, name=jev_name))
    return judges


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint running benchmark and writing comparison artifacts."""
    parser = build_cli_parser()
    args = parser.parse_args(argv)

    dataset_path = Path(args.dataset)
    target_adapter: RAGTargetAdapter
    if args.trace_file:
        target_adapter = RecordedTraceTargetAdapter.from_jsonl(Path(args.trace_file))
    else:
        target_adapter = HTTPTargetAdapter(
            base_url=args.target_url,
            inference_mode=args.inference_mode,
        )

    judges = _build_judges_from_args(args)
    report = asyncio.run(
        run_benchmark(
            dataset_path=dataset_path,
            target_adapter=target_adapter,
            judges=judges,
            runs=args.runs,
            pass_threshold=args.pass_threshold,
        )
    )
    md_path, json_path = write_report_artifacts(report, Path(args.output_dir))
    print(f"Benchmark complete ({report.sample_count} samples x {report.runs_per_sample} runs).")
    print(f"Markdown report: {md_path}")
    print(f"JSON report:     {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
