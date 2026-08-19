#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Open Renderer — drop-in replacement for AVTR-1
# MuseTalk (Apache 2.0) + FasterLivePortrait (MIT)
# Run once on berylize-node after setup.sh
# ─────────────────────────────────────────────────────────────────────────────
set -e

INSTALL_DIR="$HOME/open_renderer"
mkdir -p "$INSTALL_DIR"
cd "$INSTALL_DIR"

echo "════════════════════════════════════════════════"
echo "  Open Renderer — install"
echo "════════════════════════════════════════════════"

# ── 1. Python venv ────────────────────────────────────────────────────────────
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip setuptools wheel -q   # setuptools needed for pkg_resources

# ── 2. MuseTalk ───────────────────────────────────────────────────────────────
echo "[1/4] Cloning MuseTalk…"
if [ ! -d "MuseTalk" ]; then
  git clone https://github.com/TMElyralab/MuseTalk.git
fi
cd MuseTalk
pip install -r requirements.txt -q
pip install mmengine -q
# mmcv/mmdet are only needed for MuseTalk training — skip for inference
echo "[setup] Skipping mmcv/mmdet (training-only deps, not needed for inference)"
pip install "diffusers==0.27.2" -q
cd "$INSTALL_DIR"

# ── 3. FasterLivePortrait ────────────────────────────────────────────────────
echo "[2/4] Cloning FasterLivePortrait…"
if [ ! -d "FasterLivePortrait" ]; then
  git clone https://github.com/warmshao/FasterLivePortrait.git
fi
cd FasterLivePortrait
pip install -r requirements.txt -q
cd "$INSTALL_DIR"

# ── 4. Open Renderer deps ────────────────────────────────────────────────────
echo "[3/4] Installing Open Renderer deps…"
pip install "huggingface_hub[hf_transfer]" -q   # Rust-based downloader — up to 500MB/s+
export HF_HUB_ENABLE_HF_TRANSFER=1              # activate for all hf downloads below
pip install fastapi uvicorn safetensors httpx numpy opencv-python-headless \
  torch torchvision torchaudio --index-url https://download.openmmlab.com/mmcv/dist/cu121/torch2.1.0/index.html 2>/dev/null || \
  pip install fastapi uvicorn safetensors httpx numpy opencv-python-headless -q

# ── 5. Download weights ───────────────────────────────────────────────────────
echo "[4/4] Downloading model weights from HuggingFace…"

# MuseTalk weights
HF_HUB_ENABLE_HF_TRANSFER=1 python3 - << 'EOF'
from huggingface_hub import snapshot_download
import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

print("  → MuseTalk weights…")
snapshot_download(
    repo_id="TMElyralab/MuseTalk",
    local_dir=os.path.expanduser("~/open_renderer/weights/MuseTalk"),
    ignore_patterns=["*.msgpack", "*.h5"],
)

print("  → LivePortrait weights…")
snapshot_download(
    repo_id="KwaiVGI/LivePortrait",
    local_dir=os.path.expanduser("~/open_renderer/weights/LivePortrait"),
    ignore_patterns=["*.msgpack"],
)

print("  Weights downloaded.")
EOF

echo ""
echo "════════════════════════════════════════════════"
echo "  Done. Start with:"
echo "  source ~/open_renderer/venv/bin/activate"
echo "  python ~/open_renderer/app.py"
echo "════════════════════════════════════════════════"
