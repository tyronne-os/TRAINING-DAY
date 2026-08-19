#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Berylize — GCP Compute Engine setup script
# Run this once on a fresh Deep Learning VM (Ubuntu 22.04 + CUDA 12.x)
# GCP instance: g2-standard-8 (L4 GPU, 24 GB VRAM) or a2-highgpu-1g (A100)
# ─────────────────────────────────────────────────────────────────────────────
set -e

echo "═══════════════════════════════════════════════════"
echo "  Berylize — GCP setup"
echo "═══════════════════════════════════════════════════"

# ── 1. System deps ────────────────────────────────────────────────────────────
sudo apt-get update -q
sudo apt-get install -y -q git ffmpeg libsndfile1 python3-pip python3-venv curl

# ── 2. Install pixi (AVTR-1 package manager) ─────────────────────────────────
if ! command -v pixi &> /dev/null; then
  echo "[setup] Installing pixi (verified)…"
  PIXI_VERSION="0.77.0"
  PIXI_SHA256="bff2f77ef23178f0c73c7ddbc90ca57c68f8b75a5bd85ce8e7404f33b32852d5"
  PIXI_URL="https://github.com/prefix-dev/pixi/releases/download/v${PIXI_VERSION}/pixi-x86_64-unknown-linux-musl.tar.gz"
  curl -fsSL "$PIXI_URL" -o /tmp/pixi.tar.gz
  echo "${PIXI_SHA256}  /tmp/pixi.tar.gz" | sha256sum -c - || { echo "ABORT: pixi checksum mismatch — possible supply chain attack"; exit 1; }
  tar xzf /tmp/pixi.tar.gz -C /tmp
  mkdir -p "$HOME/.pixi/bin"
  mv /tmp/pixi "$HOME/.pixi/bin/pixi"
  chmod +x "$HOME/.pixi/bin/pixi"
  rm /tmp/pixi.tar.gz
  export PATH="$HOME/.pixi/bin:$PATH"
fi

# ── 3. Clone AVTR-1 ──────────────────────────────────────────────────────────
if [ ! -d "avtr-1" ]; then
  echo "[setup] Cloning AVTR-1…"
  git clone https://github.com/avaturn-live/avtr-1.git
fi

cd avtr-1

# ── 4. Install AVTR-1 deps + download weights ────────────────────────────────
echo "[setup] Installing AVTR-1 dependencies…"
pixi install

echo "[setup] Downloading AVTR-1 weights from HuggingFace…"
# Requires HF login: huggingface-cli login
pixi run python scripts/download_artifacts.py

echo "[setup] Building TensorRT engines (takes 10-20 min on first run)…"
pixi run -e renderer python scripts/build_engines.py

cd ..

# ── 5. Set up berylize ────────────────────────────────────────────────────────
echo "[setup] Setting up Berylize virtualenv…"
python3 -m venv venv
source venv/bin/activate

pip install --upgrade pip -q
pip install -r berylize/requirements.txt -q

# ── 6. Avatars directory ─────────────────────────────────────────────────────
mkdir -p berylize/avatars
echo "[setup] Drop your portrait images in berylize/avatars/"
echo "  Required: evedefault.jpg (or .png)"
echo "  Optional: jeff.jpg, nu.jpg, india.jpg, amanda.jpg"

# ── 7. Copy evedefault to AVTR-1 reference_frames ────────────────────────────
AVTR1_FRAMES="avtr-1/avatars_artifacts/reference_frames"
mkdir -p "$AVTR1_FRAMES"
if [ -f "berylize/avatars/evedefault.jpg" ]; then
  cp berylize/avatars/evedefault.jpg "$AVTR1_FRAMES/evedefault.jpg"
  echo "[setup] evedefault portrait copied to AVTR-1 frames dir"
else
  echo "[setup] ⚠ Drop evedefault.jpg in berylize/avatars/ then run:"
  echo "  cp berylize/avatars/evedefault.jpg $AVTR1_FRAMES/evedefault.jpg"
fi

# ── 8. Env file ───────────────────────────────────────────────────────────────
if [ ! -f "berylize/.env" ]; then
  cp berylize/.env.example berylize/.env
  echo "[setup] Created berylize/.env — add your OPENAI_API_KEY"
fi

# ── 9. Firewall reminder ──────────────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════"
echo "  Setup complete. Next steps:"
echo ""
echo "  1. Add OPENAI_API_KEY to berylize/.env"
echo ""
echo "  2. Open GCP firewall for port 8080:"
echo "     gcloud compute firewall-rules create berylize-http \\"
echo "       --allow tcp:8080 --target-tags berylize"
echo ""
echo "  3. Start AVTR-1 renderer (terminal 1):"
echo "     cd avtr-1 && pixi run -e streamer python scripts/run_renderer.py"
echo ""
echo "  4. Start Berylize server (terminal 2):"
echo "     source venv/bin/activate"
echo "     cd berylize && python server.py"
echo ""
echo "  5. Open in browser:"
echo "     http://EXTERNAL_IP:8080"
echo "═══════════════════════════════════════════════════"
