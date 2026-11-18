<#
.SYNOPSIS
Launch Sovereign Agentic RAG 100% natively on Drive F:\ without touching Drive C:\.

.DESCRIPTION
Pins Ollama models, FastEmbed ONNX caches, HuggingFace caches, Qdrant embedded disk storage,
and SQLite persistence strictly to Drive F:\, starts headless `ollama serve`, boots the FastAPI
backend on http://127.0.0.1:8000, and boots the Next.js workspace UI on http://127.0.0.1:3000.
#>

$ErrorActionPreference = "Stop"

$RepoRoot = $PSScriptRoot
$BackendDir = Join-Path $RepoRoot "backend"
$FrontendDir = Join-Path $RepoRoot "frontend"
$QdrantDir = Join-Path $RepoRoot "qdrant_storage"
$SqliteFile = (Join-Path $BackendDir "sovereign_rag.db") -replace "\\", "/"

# --- Lock EVERY runtime cache, model & DB path to F:\ (Zero-C:\ footprint) ---
$env:OLLAMA_MODELS = "F:\Ollama\models"
$env:OLLAMA_HOST = "127.0.0.1:11434"
$env:OLLAMA_KEEP_ALIVE = "5m"
$env:OLLAMA_NUM_PARALLEL = "1"
$env:HF_HOME = "F:\.cache\huggingface"
$env:FASTEMBED_CACHE_PATH = "F:\.cache\fastembed"
$env:FASTEMBED_CACHE_DIR = "F:/.cache/fastembed"
$env:PIP_CACHE_DIR = "F:\pip\cache"
$env:UV_CACHE_DIR = "F:\.cache\uv"
$env:npm_config_cache = "F:\npm-cache"
$env:TEMP = "F:\Temp\opencode"
$env:TMP = "F:\Temp\opencode"

$env:DEFAULT_INFERENCE_MODE = "LOCAL_ONPREM"
$env:LOCAL_LLM_BASE_URL = "http://127.0.0.1:11434/v1"
$env:LOCAL_LLM_MODEL = "hf.co/empero-ai/Qwen3.8-4B-Distill-GGUF:Q4_K_M"
$env:QDRANT_LOCAL_PATH = $QdrantDir -replace "\\", "/"
$env:SQLITE_URL = "sqlite+aiosqlite:///$SqliteFile"
$env:NEXT_PUBLIC_API_URL = "http://localhost:8000"

foreach ($dir in @($env:OLLAMA_MODELS, $env:HF_HOME, $env:FASTEMBED_CACHE_PATH, $QdrantDir, $env:TEMP)) {
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
}

Write-Host "[1/3] Checking Ollama on 127.0.0.1:11434 (models at $env:OLLAMA_MODELS)..." -ForegroundColor Cyan
$ollamaUp = $false
try {
    $null = Invoke-RestMethod -Uri "http://127.0.0.1:11434/api/tags" -TimeoutSec 2
    $ollamaUp = $true
} catch {
    $ollamaUp = $false
}

if (-not $ollamaUp) {
    Start-Process -FilePath "ollama" -ArgumentList "serve" -WindowStyle Hidden
    Start-Sleep -Seconds 3
}

Write-Host "[2/3] Starting FastAPI Backend on http://127.0.0.1:8000 (SQLite + Qdrant on F:\)..." -ForegroundColor Cyan
$UvicornExe = Join-Path $BackendDir ".venv\Scripts\uvicorn.exe"
$BackendProc = Start-Process -FilePath $UvicornExe `
    -ArgumentList "app.main:app", "--host", "127.0.0.1", "--port", "8000" `
    -WorkingDirectory $BackendDir `
    -PassThru

Write-Host "[3/3] Starting Next.js Workspace UI on http://localhost:3000..." -ForegroundColor Cyan
$FrontendProc = Start-Process -FilePath "npm.cmd" `
    -ArgumentList "run", "dev" `
    -WorkingDirectory $FrontendDir `
    -PassThru

Write-Host ""
Write-Host "Sovereign Agentic RAG is booting on F:\ drive!" -ForegroundColor Green
Write-Host "  Workspace UI : http://localhost:3000"
Write-Host "  Backend API  : http://127.0.0.1:8000/docs"
Write-Host "  Readiness    : http://127.0.0.1:8000/ready"
Write-Host "  Qdrant Disk  : $env:QDRANT_LOCAL_PATH"
Write-Host "  Ollama Models: $env:OLLAMA_MODELS"
Write-Host ""
Write-Host "Press Ctrl+C to stop Backend (PID $($BackendProc.Id)) & Frontend (PID $($FrontendProc.Id))." -ForegroundColor Yellow

try {
    Wait-Process -Id $BackendProc.Id, $FrontendProc.Id
} finally {
    Stop-Process -Id $BackendProc.Id -Force -ErrorAction SilentlyContinue
    Stop-Process -Id $FrontendProc.Id -Force -ErrorAction SilentlyContinue
}
