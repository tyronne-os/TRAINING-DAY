#!/usr/bin/env bash
# setup_ace_renderer.sh — Install ACE renderer dependencies.
#
# No model downloads needed — Audio2Face-3D is a cloud gRPC API.
# This script:
#   1. Installs Python packages from requirements.txt
#   2. Downloads the NVIDIA ACE protobuf Python wheel from the official
#      Audio2Face-3D-Samples repository and installs it.
#      The wheel provides the generated gRPC stubs:
#        nvidia_ace.controller.v1_pb2
#        nvidia_ace.services.a2f_controller.v1_pb2_grpc
#        ... (all ACE proto packages)
#   3. Verifies the import resolves correctly.
#
# Usage:
#   cd /path/to/berylize_clique/ace_renderer
#   bash setup_ace_renderer.sh
#
# After setup, run the renderer:
#   NVIDIA_NGC_API_KEY=<your-key> python ace_renderer_app.py

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== ACE Renderer Setup ==="
echo "Working directory: $SCRIPT_DIR"

# ── 1. Install Python dependencies ──────────────────────────────────────────
echo ""
echo "[1/3] Installing Python packages from requirements.txt..."
pip install --upgrade pip
pip install -r requirements.txt

# ── 2. Install NVIDIA ACE proto wheel ────────────────────────────────────────
# The wheel ships pre-generated gRPC Python stubs for all ACE services.
# Source: https://github.com/NVIDIA/Audio2Face-3D-Samples/tree/main/proto/sample_wheel
WHEEL_URL="https://github.com/NVIDIA/Audio2Face-3D-Samples/raw/main/proto/sample_wheel/nvidia_ace-1.2.0-py3-none-any.whl"
WHEEL_FILE="nvidia_ace-1.2.0-py3-none-any.whl"

echo ""
echo "[2/3] Downloading NVIDIA ACE proto wheel..."
if command -v curl &>/dev/null; then
    curl -fsSL -o "$WHEEL_FILE" "$WHEEL_URL"
elif command -v wget &>/dev/null; then
    wget -q -O "$WHEEL_FILE" "$WHEEL_URL"
else
    echo "ERROR: neither curl nor wget found. Install one and retry." >&2
    exit 1
fi

echo "[2/3] Installing NVIDIA ACE proto wheel..."
pip install "$WHEEL_FILE"
rm -f "$WHEEL_FILE"

# ── 3. Verify imports ────────────────────────────────────────────────────────
echo ""
echo "[3/3] Verifying gRPC stub imports..."
python3 - <<'PYCHECK'
from nvidia_ace.controller.v1_pb2 import AudioStream, AudioStreamHeader, EndOfAudio
from nvidia_ace.a2f.v1_pb2 import AudioWithEmotion, FaceParameters
from nvidia_ace.audio.v1_pb2 import AudioHeader
from nvidia_ace.services.a2f_controller.v1_pb2_grpc import A2FControllerServiceStub
from nvidia_ace.animation_data.v1_pb2 import AnimationData
print("  nvidia_ace stubs OK")

import grpc.aio
print("  grpc.aio OK")

import cv2
import numpy as np
print("  opencv OK")

import safetensors
print("  safetensors OK")

import fastapi
print("  fastapi OK")
PYCHECK

echo ""
echo "=== Setup complete ==="
echo ""
echo "To start the ACE renderer:"
echo "  export NVIDIA_NGC_API_KEY=<your-ngc-api-key>"
echo "  export PORTRAITS_DIR=~/berylize_clique/berylize/avatars   # optional"
echo "  python ace_renderer_app.py"
echo ""
echo "The server listens on port 8001 (MuseTalk stays on 8000)."
echo ""
echo "NVIDIA function IDs (set NVIDIA_A2F_FUNCTION_ID to override):"
echo "  Claire (default): 462f7853-60e8-474a-9728-7b598e58472c"
echo "  Mark:             945ed566-a023-4677-9a49-61ede107fd5a"
echo "  James:            a2cc5cac-147d-4e46-b79d-4cea616e21b9"
