#!/usr/bin/env bash
set -euo pipefail

echo "========== No-Docker GPU Environment Setup =========="

echo
echo "Checking GPU..."
nvidia-smi

echo
echo "Checking Python..."
python3 --version

echo
echo "Creating virtual environment..."
python3 -m venv .venv-gpu
source .venv-gpu/bin/activate

echo
echo "Upgrading pip and installing uv..."
pip install --upgrade pip
pip install uv

echo
echo "Installing local benchmark dependencies..."
pip install -r requirements-dev.txt

echo
echo "Installing vLLM..."
# vLLM official docs recommend uv and automatic torch backend selection.
uv pip install vllm --torch-backend=auto

echo
echo "Installing SGLang..."
uv pip install sglang

echo
echo "Sanity check..."
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PY

python - <<'PY'
import vllm
print("vllm import ok")
PY

python - <<'PY'
import sglang
print("sglang import ok")
PY

echo
echo "Setup completed."
echo "Activate with:"
echo "source .venv-gpu/bin/activate"
