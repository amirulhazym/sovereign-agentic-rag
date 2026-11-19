# Standalone Evaluation Harness (`eval/`)

Decoupled evaluation sub-project comparing **generative `LLM-as-a-Judge` (`LLMJudgeEvaluator`)** against **deterministic local discriminative `JEV-as-a-Judge` (`JEVJudgeEvaluator`)** across latency, API token cost (`$`), multi-run variance ($\sigma$), raw agreement rate, and **Cohen's Kappa ($\kappa$)**.

---

## 1. Standalone Setup (Zero Backend Coupling)

```bash
cd eval
uv venv .venv --python 3.11
uv pip install --python .venv/Scripts/python.exe -r requirements.txt
```

> Why separate? Keeping verifier weights (`onnxruntime` + `tokenizers` + `mDeBERTa-v3` ONNX) inside `eval/.venv` guarantees they never consume RAM inside the live `backend/` runtime container on 16GB workstations.

---

## 2. Package Architecture

- **[`eval_harness/schemas.py`](eval_harness/schemas.py):** Pydantic v2 contracts (`ExpectedCitation`, `EvalSample`, `RAGOutput`, `JudgeVerdict`) and `load_golden_dataset()`.
- **[`eval_harness/target_adapter.py`](eval_harness/target_adapter.py):** `RAGTargetAdapter` seam with two concrete adapters:
  - `HTTPTargetAdapter`: streams `POST /api/v1/chat/stream` over SSE (`token`, `citations`, `done`, `error`).
  - `RecordedTraceTargetAdapter`: replays saved JSONL `RAGOutput` traces for deterministic offline CI comparisons.
- **[`eval_harness/judges/base.py`](eval_harness/judges/base.py):** `BaseJudgeEvaluator` seam, `[doc_id:pX]` stripping, multilingual claim splitting, and bilingual refusal detection.
- **[`eval_harness/judges/llm_judge.py`](eval_harness/judges/llm_judge.py):** `LLMJudgeEvaluator` supporting DeepSeek API (static rubric $\ge 1,024$ tokens ordered first for Prefix Context Caching) and local Ollama (`empero-ai/Qwen3.8-*-Distill-GGUF` with `/no_think` and `<think>` tag stripping).
- **[`eval_harness/judges/jev_judge.py`](eval_harness/judges/jev_judge.py):** `JEVJudgeEvaluator` currently implemented as an air-gapped `$0` discriminative NLI verifier (`ONNXMultilingualNLIBackend` using `Xenova/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7`, plus `LexicalDiscriminativeBackend` for offline smoke checks). *Note:* Distinct from **TypeSafe Jev** ("System One" decision model, `arXiv:2609.26550`), whose `Noul`/`Choice`/`Score` confidence-cascade API is planned under this same `BaseJudgeEvaluator` seam in `v3.1`.
- **[`eval_harness/metrics.py`](eval_harness/metrics.py) & [`eval_harness/runner.py`](eval_harness/runner.py):** Computes percentiles (`p50`/`p95`), per-sample StdDev across `--runs N`, chance-corrected Cohen's Kappa ($\kappa$), and renders `reports/benchmark_report.{md,json}`.
- **[`datasets/golden_dataset.jsonl`](datasets/golden_dataset.jsonl):** 36 curated bilingual samples aligned with `data/seed_collections/` (`15` in `it_infra_sops`, `21` balanced `7 en / 7 ms / 7 mixed` in `malaysian_enterprise_policy`, including `6` `unanswerable_refusal` traps).

---

## 3. Running Tests & Benchmarks

```bash
# Unit & contract suite (15 tests, zero external downloads)
.venv/Scripts/pytest.exe -v
.venv/Scripts/ruff.exe check eval_harness tests

# Live side-by-side benchmark against running backend + local Ollama Qwen3.8 + ONNX NLI
.venv/Scripts/python.exe -m eval_harness.runner `
  --dataset datasets/golden_dataset.jsonl `
  --runs 3 `
  --target-url http://localhost:8000 `
  --inference-mode LOCAL_ONPREM `
  --llm-backend ollama `
  --jev-backend onnx

# Cloud DeepSeek judge comparison (uses DEEPSEEK_API_KEY env var)
.venv/Scripts/python.exe -m eval_harness.runner `
  --dataset datasets/golden_dataset.jsonl `
  --runs 3 `
  --inference-mode CLOUD_API `
  --llm-backend deepseek `
  --jev-backend onnx
```
