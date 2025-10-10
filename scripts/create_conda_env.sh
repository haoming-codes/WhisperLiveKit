#!/usr/bin/env bash
set -euo pipefail

# Default values
ENV_NAME="${ENV_NAME:-whisperlivekit}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
USE_CUDA="${USE_CUDA:-0}"
CUDA_VERSION="${CUDA_VERSION:-12.1}"
INSTALL_SENTENCE_EXTRAS="${INSTALL_SENTENCE_EXTRAS:-0}"

usage() {
  cat <<USAGE
Usage: ENV_NAME=<name> PYTHON_VERSION=<version> USE_CUDA=<0|1> CUDA_VERSION=<major.minor> INSTALL_SENTENCE_EXTRAS=<0|1> $0

Environment variables:
  ENV_NAME                Name of the conda environment to create (default: whisperlivekit)
  PYTHON_VERSION          Python version to install in the environment (default: 3.10)
  USE_CUDA                Install GPU-enabled PyTorch stack (default: 0 / CPU-only)
  CUDA_VERSION            CUDA version to install when USE_CUDA=1 (default: 12.1)
  INSTALL_SENTENCE_EXTRAS Install optional sentence tokenization extras (default: 0 / disabled)

Examples:
  $0
  ENV_NAME=whisperlivekit-dev PYTHON_VERSION=3.11 $0
  USE_CUDA=1 CUDA_VERSION=12.1 $0
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "Error: conda is not on PATH. Please install Miniconda or Anaconda first." >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Initialize conda for the current shell session
if [[ -n "${BASH_VERSION:-}" ]]; then
  eval "$(conda shell.bash hook)"
else
  echo "Error: This script must be run from a bash-compatible shell." >&2
  exit 1
fi

if conda env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
  echo "Conda environment '$ENV_NAME' already exists. Skipping creation."
else
  echo "Creating conda environment '$ENV_NAME' with Python ${PYTHON_VERSION}..."
  conda create -y -n "$ENV_NAME" "python=${PYTHON_VERSION}"
fi

conda activate "$ENV_NAME"

if [[ "$USE_CUDA" == "1" ]]; then
  echo "Installing PyTorch with CUDA support (CUDA ${CUDA_VERSION})..."
  conda install -y pytorch pytorch-cuda="${CUDA_VERSION}" torchaudio -c pytorch -c nvidia
else
  echo "Installing PyTorch CPU build..."
  conda install -y pytorch torchaudio cpuonly -c pytorch
fi

echo "Installing WhisperLiveKit in editable mode..."
python -m pip install --upgrade pip
if [[ "$INSTALL_SENTENCE_EXTRAS" == "1" ]]; then
  python -m pip install -e "${REPO_ROOT}[sentence]"
else
  python -m pip install -e "${REPO_ROOT}"
fi

echo "Installing additional runtime dependencies..."
conda install -y ffmpeg -c conda-forge

echo "Environment '$ENV_NAME' is ready. Activate it with:"
echo "  conda activate $ENV_NAME"

echo "To run the server:"
echo "  whisperlivekit-server"
