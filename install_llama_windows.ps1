# Install llama-cpp-python on Windows using binary wheels first (no compiler).
# Run from repo root:  powershell -ExecutionPolicy Bypass -File .\install_llama_windows.ps1

$ErrorActionPreference = "Stop"
Write-Host "Upgrading pip..."
python -m pip install --upgrade pip
Write-Host "Installing llama-cpp-python (prefer-binary + abetlen CPU wheel index)..."
python -m pip install llama-cpp-python --prefer-binary --extra-index-url "https://abetlen.github.io/llama-cpp-python/whl/cpu"
Write-Host "Done. Test: python -c ""from llama_cpp import Llama; print('ok')"""
