"""
ACE Renderer — drop-in replacement for AVTR-1's /process-audio-v3 endpoint.

Uses NVIDIA ACE Audio2Face-3D (cloud gRPC) for lip-sync blendshapes, then
renders them onto the portrait image to produce YUV I420 frames.

Identical API contract to open_renderer/app.py and the original AVTR-1:
  POST /process-audio-v3
    multipart fields:
      current_chunk        — int16 PCM, 3200 samples (200ms @ 16kHz)
      future_chunk         — int16 PCM, future lookahead
      current_chunk_listen — int16 PCM, listening track current
      future_chunk_listen  — int16 PCM, listening track future
      state                — safetensors blob (optional, None on first call)
    query params:
      avatar_id, bg_id, pixel_format (yuv_i420)
    response body:
      state_blob (safetensors) ++ 5 raw YUV I420 frames concatenated
    response headers:
      X-State-Length-Bytes, X-Frame-Length-Bytes, X-Num-Frames,
      X-Frame-Height, X-Frame-Width

  GET /health   → {"status": "ok", "avatars": [...]}
  GET /avatars  → {"avatars": [...], "backgrounds": [...]}

Listens on port 8001 (MuseTalk holds 8000).

Environment:
  NVIDIA_NGC_API_KEY    — required; NGC bearer token
  NVIDIA_A2F_FUNCTION_ID — optional; NVCF function UUID (default: Claire)
  PORTRAITS_DIR         — optional; path to portrait images (default below)
  OUT_W / OUT_H         — optional; output frame pixels (default 512)
  RENDERER_PORT         — optional (default 8001)
"""
from __future__ import annotations

import logging
import os
import struct
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, Query, UploadFile
from fastapi.responses import Response

from ace_client import ACEAudio2FaceClient
from blendshape_renderer import BlendshapeRenderer
from session_state import SessionState
from frame_packer import pack_frames, bytes_per_yuv_i420_frame

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
LOG = logging.getLogger("ace_renderer")

PORTRAITS_DIR = Path(
    os.environ.get("PORTRAITS_DIR", Path.home() / "berylize_clique/berylize/avatars")
)
OUT_W = int(os.environ.get("OUT_W", 512))
OUT_H = int(os.environ.get("OUT_H", 512))
PORT  = int(os.environ.get("RENDERER_PORT", 8001))

# Avatar registry: id → (portrait_bgr, BlendshapeRenderer)
_avatar_portraits: dict[str, np.ndarray] = {}
_avatar_renderers: dict[str, BlendshapeRenderer] = {}
_ace_client: ACEAudio2FaceClient | None = None

# How many YUV I420 frames to return per call — matches AVTR-1 expectation.
_FRAMES_PER_CALL = 5


def _load_avatars() -> None:
    if not PORTRAITS_DIR.exists():
        LOG.warning("Portraits dir not found: %s — no avatars loaded", PORTRAITS_DIR)
        return
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp"):
        for img_path in sorted(PORTRAITS_DIR.glob(ext)):
            avatar_id = img_path.stem
            portrait = cv2.imread(str(img_path))
            if portrait is None:
                LOG.warning("Could not decode portrait: %s", img_path)
                continue
            _avatar_portraits[avatar_id] = portrait
            _avatar_renderers[avatar_id] = BlendshapeRenderer(portrait)
            LOG.info("[ace_renderer] Registered avatar: %s", avatar_id)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _ace_client
    api_key = os.environ.get("NVIDIA_NGC_API_KEY", "")
    if not api_key:
        LOG.warning(
            "NVIDIA_NGC_API_KEY not set — ACE calls will fail at inference time"
        )
    _ace_client = ACEAudio2FaceClient(api_key=api_key)
    await _ace_client.__aenter__()
    LOG.info("[ace_renderer] ACE gRPC channel opened")

    _load_avatars()
    LOG.info("[ace_renderer] Ready on port %d. Avatars: %s", PORT, list(_avatar_renderers))
    yield

    await _ace_client.__aexit__(None, None, None)
    LOG.info("[ace_renderer] Shut down.")


app = FastAPI(lifespan=lifespan)


@app.get("/health")
async def health():
    return {"status": "ok", "avatars": list(_avatar_renderers)}


@app.get("/avatars")
async def list_avatars():
    return {
        "avatars": list(_avatar_renderers),
        "backgrounds": ["plain_white"],
    }


@app.post("/process-audio-v3")
async def process_audio_v3(
    current_chunk: UploadFile,
    future_chunk: UploadFile,
    current_chunk_listen: UploadFile,
    future_chunk_listen: UploadFile,
    state: UploadFile | None = None,
    avatar_id: str = Query("evedefault"),
    bg_id: str = Query("plain_white"),
    pixel_format: str = Query("yuv_i420"),
    cfg_self_audio: float = Query(2.0),
    cfg_other_audio: float = Query(2.0),
    cfg_kp: float = Query(4.0),
    noise_alpha: float = Query(2.0),
    noise_trunc_z: float = Query(1.2),
) -> Response:
    # ── Avatar lookup ─────────────────────────────────────────────────────────
    renderer = _avatar_renderers.get(avatar_id)
    portrait = _avatar_portraits.get(avatar_id)
    if renderer is None or portrait is None:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown avatar_id {avatar_id!r}; available: {sorted(_avatar_renderers)}",
        )

    # ── Deserialise session state ──────────────────────────────────────────────
    state_bytes = await state.read() if state else None
    session = SessionState.from_bytes(state_bytes)

    # ── Read audio ────────────────────────────────────────────────────────────
    cur_bytes = await current_chunk.read()
    fut_bytes = await future_chunk.read()
    # Listening tracks are available but ACE does not take a separate stream;
    # we accept them to satisfy the multipart schema.
    await current_chunk_listen.read()
    await future_chunk_listen.read()

    # Concatenate current + future for richer context (200ms + 200ms = 400ms).
    combined_pcm = cur_bytes + fut_bytes

    # ── Call ACE Audio2Face-3D ─────────────────────────────────────────────────
    blendshape_frames: list[dict[str, float]] = []
    try:
        if _ace_client is not None and os.environ.get("NVIDIA_NGC_API_KEY", ""):
            blendshape_frames = await _ace_client.audio_to_blendshapes(
                combined_pcm, sample_rate=16000
            )
    except Exception as exc:
        LOG.error("[ACE] audio_to_blendshapes failed: %s", exc)
        # Fall through to silence/neutral fallback below.

    # ── Select frames for this response ───────────────────────────────────────
    # ACE returns ~30fps × audio_duration frames; we need exactly _FRAMES_PER_CALL.
    # Temporal smoothing: blend with prior_blendshapes on the first frame.
    selected = _select_frames(
        blendshape_frames,
        n=_FRAMES_PER_CALL,
        prior=session.prior_blendshapes,
    )

    # ── Render BGR frames from blendshapes ────────────────────────────────────
    frames_bgr: list[np.ndarray] = []
    for weights in selected:
        try:
            frame = renderer.render_frame(weights)
        except Exception as exc:
            LOG.warning("[BlendshapeRenderer] render_frame error: %s — using still", exc)
            frame = cv2.resize(portrait, (256, 256))
        frames_bgr.append(frame)

    # ── Pack frames → YUV I420 ────────────────────────────────────────────────
    frame_bytes = pack_frames(frames_bgr, width=OUT_W, height=OUT_H)

    # ── Update + serialise session state ──────────────────────────────────────
    session.frame_count += len(frames_bgr)
    if selected:
        last_weights = selected[-1]
        names = list(last_weights.keys())
        vals = [last_weights[n] for n in names]
        session.prior_blendshapes = _weights_to_tensor(vals)
    new_state_bytes = session.to_bytes()

    # ── Build response (state blob ++ frame bytes) ─────────────────────────────
    frame_len = bytes_per_yuv_i420_frame(OUT_W, OUT_H)
    body = new_state_bytes + frame_bytes
    return Response(
        content=body,
        media_type="application/octet-stream",
        headers={
            "X-State-Length-Bytes": str(len(new_state_bytes)),
            "X-Frame-Length-Bytes": str(frame_len),
            "X-Num-Frames":         str(len(frames_bgr)),
            "X-Frame-Height":       str(OUT_H),
            "X-Frame-Width":        str(OUT_W),
        },
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

import torch


def _neutral_weights() -> dict[str, float]:
    """Return all-zero blendshape weights (neutral face)."""
    from ace_client import BLENDSHAPE_NAMES
    return {name: 0.0 for name in BLENDSHAPE_NAMES}


def _select_frames(
    frames: list[dict[str, float]],
    n: int,
    prior: "torch.Tensor | None",
) -> list[dict[str, float]]:
    """
    Return exactly n blendshape dicts, subsampled or padded from `frames`.

    If ACE returned nothing, returns n copies of the neutral pose.
    Applies a one-frame exponential smoothing blend with prior if provided.
    """
    if not frames:
        neutral = _neutral_weights()
        if prior is not None:
            from ace_client import BLENDSHAPE_NAMES
            prior_np = prior.cpu().numpy()
            # Decay toward neutral over 5 frames.
            decay = 0.7
            blended = {
                name: float(prior_np[i] * decay)
                for i, name in enumerate(BLENDSHAPE_NAMES)
                if i < len(prior_np)
            }
            return [blended] * n
        return [neutral] * n

    if len(frames) >= n:
        # Evenly subsample.
        step = len(frames) / n
        selected = [frames[int(i * step)] for i in range(n)]
    else:
        # Repeat last frame to pad.
        selected = list(frames)
        while len(selected) < n:
            selected.append(frames[-1])

    # Smooth first frame with prior blendshapes.
    if prior is not None:
        from ace_client import BLENDSHAPE_NAMES
        prior_np = prior.cpu().numpy()
        first = selected[0]
        alpha = 0.3  # weight toward prior
        smoothed = {
            name: alpha * float(prior_np[i]) + (1.0 - alpha) * first.get(name, 0.0)
            for i, name in enumerate(BLENDSHAPE_NAMES)
            if i < len(prior_np)
        }
        # Fill any keys present in first but not in BLENDSHAPE_NAMES.
        for k, v in first.items():
            if k not in smoothed:
                smoothed[k] = v
        selected[0] = smoothed

    return selected


def _weights_to_tensor(vals: list[float]) -> "torch.Tensor":
    import torch
    return torch.tensor(vals, dtype=torch.float32)


if __name__ == "__main__":
    uvicorn.run("ace_renderer_app:app", host="0.0.0.0", port=PORT, log_level="info")
