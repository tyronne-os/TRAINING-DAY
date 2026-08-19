"""
pipeline/renderer.py
AVTR-1 REST client.
Calls the avtr1_renderer FastAPI service (POST /process-audio-v3)
per 200ms audio chunk (3200 samples @ 16kHz).
Returns 5 BGR frames per call, queued for WebRTC.

The idle loop feeds brownian noise as the listening track, keeping
the avatar alive and micro-animated at all times — even during silence.
"""

import asyncio
import logging
import os
import time

import cv2
import numpy as np

logger = logging.getLogger(__name__)

AVTR1_URL     = os.environ.get("AVTR1_RENDERER_URL", "http://localhost:8000")
CHUNK_SAMPLES = 3200          # 200ms @ 16kHz
FRAME_H       = 512
FRAME_W       = 512
FRAME_BUFFER_MAX = 45         # ~1.8s of frames
IDLE_INTERVAL = 0.18          # slightly under 200ms to stay ahead of playback


def _brownian_listen(n: int = CHUNK_SAMPLES, amplitude: float = 0.008) -> np.ndarray:
    """Brownian noise at ~-40dB — drives subtle listening-state micro-motion."""
    walk = np.cumsum(np.random.randn(n).astype(np.float32))
    walk /= (np.abs(walk).max() + 1e-8)
    return (walk * amplitude * 32767).clip(-32768, 32767).astype(np.int16)


SILENCE_PCM = np.zeros(CHUNK_SAMPLES, dtype=np.int16)


class AVTR1Renderer:
    """
    Thin async client around the AVTR-1 FastAPI renderer.
    Uses httpx.AsyncClient for non-blocking multipart POSTs.
    """

    def __init__(self):
        self._client      = None
        self._state: bytes | None = None
        self._avatar_id   = "evedefault"
        self._bg_id       = os.environ.get("DEFAULT_BG_ID", "plain_white")
        self.frame_buffer = asyncio.Queue(maxsize=FRAME_BUFFER_MAX)
        self._last_frame: np.ndarray | None = None
        self._speaking    = False
        self._idle_task: asyncio.Task | None = None
        self._healthy     = False
        self._init_client()

    def _init_client(self):
        try:
            import httpx
            self._client = httpx.AsyncClient(
                base_url=AVTR1_URL,
                timeout=2.5,
            )
            logger.info(f"AVTR-1 client targeting {AVTR1_URL}")
        except ImportError:
            logger.warning("httpx not installed — renderer in STUB mode.")

    # ── Health ───────────────────────────────────────────────────────────

    async def check_health(self) -> bool:
        if self._client is None:
            return False
        try:
            r = await self._client.get("/health")
            self._healthy = r.status_code == 200
        except Exception:
            self._healthy = False
        return self._healthy

    # ── Avatar switching ─────────────────────────────────────────────────

    async def set_avatar(self, name: str):
        self._avatar_id = name
        self._state = None   # reset motion state on identity switch
        logger.info(f"[RENDERER] avatar → {name}")

    # ── Core inference call ───────────────────────────────────────────────

    async def _call(
        self,
        speech: np.ndarray,
        listen: np.ndarray,
    ) -> list[np.ndarray]:
        """
        POST /process-audio-v3 → list of 5 BGR uint8 frames.
        Falls back to last-known frame on any error.
        """
        if self._client is None:
            return self._stub_frames()

        def pcm_bytes(arr: np.ndarray) -> bytes:
            return arr.astype(np.int16).tobytes()

        files: dict = {
            "current_chunk":        ("c.pcm", pcm_bytes(speech), "application/octet-stream"),
            "future_chunk":         ("f.pcm", pcm_bytes(speech), "application/octet-stream"),
            "current_chunk_listen": ("cl.pcm", pcm_bytes(listen), "application/octet-stream"),
            "future_chunk_listen":  ("fl.pcm", pcm_bytes(listen), "application/octet-stream"),
        }
        if self._state:
            files["state"] = ("state.safetensors", self._state, "application/octet-stream")

        params = {
            "avatar_id":    self._avatar_id,
            "bg_id":        self._bg_id,
            "pixel_format": "yuv_i420",
        }

        t0 = time.time()
        try:
            resp = await self._client.post(
                "/process-audio-v3",
                files=files,
                params=params,
            )
            resp.raise_for_status()

            state_len  = int(resp.headers["X-State-Length-Bytes"])
            frame_len  = int(resp.headers["X-Frame-Length-Bytes"])
            num_frames = int(resp.headers["X-Num-Frames"])
            h = int(resp.headers.get("X-Frame-Height", FRAME_H))
            w = int(resp.headers.get("X-Frame-Width",  FRAME_W))

            body       = resp.content
            self._state = body[:state_len]
            raw_frames  = body[state_len:]

            self._healthy = True
            latency_ms = (time.time() - t0) * 1000
            logger.debug(f"[RENDERER] {num_frames} frames in {latency_ms:.0f}ms")

            frames = []
            for i in range(num_frames):
                yuv_bytes = raw_frames[i * frame_len:(i + 1) * frame_len]
                yuv = np.frombuffer(yuv_bytes, dtype=np.uint8).reshape(
                    h * 3 // 2, w
                )
                bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)
                frames.append(bgr)
            return frames

        except Exception as e:
            self._healthy = False
            logger.warning(f"[RENDERER] AVTR-1 call failed: {e}")
            return self._stub_frames()

    def _stub_frames(self) -> list[np.ndarray]:
        frame = self._last_frame if self._last_frame is not None \
            else np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
        return [frame.copy() for _ in range(5)]

    # ── Frame queue ───────────────────────────────────────────────────────

    def _enqueue_frames(self, frames: list[np.ndarray]):
        for frame in frames:
            self._last_frame = frame
            try:
                self.frame_buffer.put_nowait(frame)
            except asyncio.QueueFull:
                try:
                    self.frame_buffer.get_nowait()
                    self.frame_buffer.put_nowait(frame)
                except Exception:
                    pass

    # ── TTS-driven audio path ─────────────────────────────────────────────

    async def feed_audio(self, speech_chunk: np.ndarray):
        """
        Called by TTS pipeline with each 35ms speech chunk.
        Accumulates to 200ms before calling AVTR-1.
        """
        frames = await self._call(speech_chunk, _brownian_listen())
        self._enqueue_frames(frames)

    # ── Idle presence loop ────────────────────────────────────────────────

    async def start_idle(self):
        """Starts the always-alive idle loop. Call once on server startup."""
        logger.info("[RENDERER] Starting idle presence loop")
        self._idle_task = asyncio.create_task(self._idle_loop())

    async def _idle_loop(self):
        """
        Drives the avatar with silence + brownian listening noise
        whenever TTS is not speaking. This keeps the avatar alive
        (micro-expressions, gaze drift, blinks) at all times.
        """
        while True:
            if not self._speaking:
                frames = await self._call(SILENCE_PCM, _brownian_listen())
                self._enqueue_frames(frames)
            await asyncio.sleep(IDLE_INTERVAL)

    def set_speaking(self, speaking: bool):
        """Toggle: True while TTS is streaming, False when done."""
        self._speaking = speaking
        logger.debug(f"[RENDERER] speaking={speaking}")

    async def stop_idle(self):
        if self._idle_task:
            self._idle_task.cancel()
            self._idle_task = None

    # ── WebRTC frame pull ─────────────────────────────────────────────────

    async def get_frame(self) -> np.ndarray | None:
        try:
            frame = self.frame_buffer.get_nowait()
            self._last_frame = frame
            return frame
        except asyncio.QueueEmpty:
            if self._last_frame is not None:
                return self._last_frame
            return None

    def blank_frame(self) -> np.ndarray:
        if self._last_frame is not None:
            return self._last_frame.copy()
        return np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
