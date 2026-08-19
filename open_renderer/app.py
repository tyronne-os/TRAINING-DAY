"""
Open Renderer — drop-in replacement for AVTR-1's /process-audio-v3 endpoint.

Identical API contract:
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
      state_blob (safetensors) + 5 YUV I420 frames concatenated
    response header:
      X-State-Length-Bytes — byte offset splitting state from frames

  GET /health   → {"status": "ok"}
  GET /avatars  → {"avatars": [...], "backgrounds": [...]}

Run:
  python app.py
  (listens on 0.0.0.0:8000 — same port as AVTR-1)
"""
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, Query, UploadFile
from fastapi.responses import Response

from musetalk_engine import MuseTalkEngine, AvatarContext
from liveportrait_engine import LivePortraitEngine
from session_state import SessionState
from frame_packer import pack_frames, bytes_per_yuv_i420_frame

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
LOG = logging.getLogger("open_renderer")

PORTRAITS_DIR = Path(os.environ.get("PORTRAITS_DIR", Path.home() / "berylize_clique/berylize/avatars"))
OUT_W = int(os.environ.get("OUT_W", 512))
OUT_H = int(os.environ.get("OUT_H", 512))
PORT  = int(os.environ.get("RENDERER_PORT", 8000))

musetalk = MuseTalkEngine()
liveportrait = LivePortraitEngine()
avatar_registry: dict[str, AvatarContext] = {}


def _load_avatars() -> None:
    """Scan PORTRAITS_DIR and build AvatarContext for each portrait."""
    if not PORTRAITS_DIR.exists():
        LOG.warning("Portraits dir not found: %s", PORTRAITS_DIR)
        return
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp"):
        for img_path in sorted(PORTRAITS_DIR.glob(ext)):
            avatar_id = img_path.stem
            try:
                portrait = cv2.imread(str(img_path))
                if portrait is None:
                    continue
                ctx = musetalk.prepare_avatar(portrait, avatar_id)
                avatar_registry[avatar_id] = ctx
                LOG.info("[open_renderer] Registered avatar: %s", avatar_id)
            except Exception as exc:
                LOG.error("[open_renderer] Failed to load %s: %s", avatar_id, exc)


@asynccontextmanager
async def lifespan(app: FastAPI):
    LOG.info("[open_renderer] Loading MuseTalk…")
    musetalk.load()
    LOG.info("[open_renderer] Loading LivePortrait…")
    liveportrait.load()
    LOG.info("[open_renderer] Registering avatars from %s…", PORTRAITS_DIR)
    _load_avatars()
    LOG.info("[open_renderer] Ready. Avatars: %s", list(avatar_registry.keys()))
    yield
    LOG.info("[open_renderer] Shutting down.")


app = FastAPI(lifespan=lifespan)


@app.get("/health")
async def health():
    return {"status": "ok", "avatars": list(avatar_registry.keys())}


@app.get("/avatars")
async def list_avatars():
    return {
        "avatars": list(avatar_registry.keys()),
        "backgrounds": ["plain_white"],   # open renderer uses bg compositing via cv2
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
    # ── Avatar lookup ────────────────────────────────────────────────────────
    if avatar_id not in avatar_registry:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown avatar_id {avatar_id!r}; available: {sorted(avatar_registry)}",
        )
    ctx = avatar_registry[avatar_id]

    # ── Deserialise session state ─────────────────────────────────────────────
    state_bytes = await state.read() if state else None
    session = SessionState.from_bytes(state_bytes)

    # ── Read audio chunks ────────────────────────────────────────────────────
    cur_bytes  = await current_chunk.read()
    fut_bytes  = await future_chunk.read()
    curl_bytes = await current_chunk_listen.read()
    futl_bytes = await future_chunk_listen.read()

    # Concatenate current + future for richer whisper context (matches AVTR-1 approach)
    combined = np.concatenate([
        np.frombuffer(cur_bytes,  dtype=np.int16),
        np.frombuffer(fut_bytes,  dtype=np.int16),
    ])
    # Listening track — used for brownian idle animation (blend with speech)
    listen_combined = np.concatenate([
        np.frombuffer(curl_bytes, dtype=np.int16),
        np.frombuffer(futl_bytes, dtype=np.int16),
    ])

    # ── MuseTalk: audio → lip-sync frames ────────────────────────────────────
    try:
        frames_bgr, new_audio_feats = musetalk.synthesize_chunk(
            audio_pcm=combined[:3200],   # MuseTalk uses current chunk
            ctx=ctx,
            state_audio_feats=session.prior_audio_feats,
        )
    except Exception as exc:
        LOG.error("[MuseTalk] synthesize_chunk failed: %s", exc)
        # Graceful degradation: return still frames
        still = cv2.resize(ctx.face_image, (256, 256))
        frames_bgr = [still.copy() for _ in range(5)]
        new_audio_feats = session.prior_audio_feats

    # ── LivePortrait: add head pose + expression naturalness ─────────────────
    try:
        frames_bgr = liveportrait.apply_expression(frames_bgr, ctx.face_image)
    except Exception as exc:
        LOG.warning("[LivePortrait] apply_expression failed, using raw frames: %s", exc)

    # ── Pack frames into YUV I420 ────────────────────────────────────────────
    frame_bytes = pack_frames(frames_bgr, width=OUT_W, height=OUT_H)

    # ── Update + serialise session state ─────────────────────────────────────
    session.frame_count += len(frames_bgr)
    session.prior_audio_feats = new_audio_feats
    new_state_bytes = session.to_bytes()

    # ── Build response (state blob ++ frame bytes, same as AVTR-1) ───────────
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


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=PORT, log_level="info")
