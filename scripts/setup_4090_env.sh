#!/usr/bin/env bash
set -euo pipefail

echo "========== 4090 Environment Setup =========="

echo
echo "Checking Python..."
python3 --version

echo
echo "Checking NVIDIA GPU..."
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
else
  echo "ERROR: nvidia-smi not found. Please choose a CUDA/NVIDIA GPU image."
  exit 1
fi

echo
echo "Checking Docker..."
if command -v docker >/dev/null 2>&1; then
  docker --version
else
  echo "ERROR: docker not found. Please choose a Docker-enabled GPU image or install Docker manually."
  exit 1
fi

echo
echo "Checking NVIDIA Docker runtime..."
if docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi; then
  echo "NVIDIA Docker runtime works."
else
  echo "ERROR: Docker cannot access GPU. NVIDIA Container Toolkit may be missing."
  exit 1
fi

echo
echo "Creating Python virtual environment..."
python3 -m venv .venv
source .venv/bin/activate

echo
echo "Installing benchmark dependencies..."
pip install --upgrade pip
pip install -r requirements-dev.txt

echo
echo "Environment setup completed."
echo "Activate with:"
echo "source .venv/bin/activate"
