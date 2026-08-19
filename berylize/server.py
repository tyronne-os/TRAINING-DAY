"""
server.py
Berylize — aiohttp + aiortc WebRTC server.
Pipeline: mic → FasterWhisperASR → CliqueBrain → ClueTTS → AVTR1Renderer → WebRTC

Routes:
  GET  /                → index.html
  GET  /static/...      → static files
  GET  /avatars/{name}  → portrait image
  GET  /status          → health JSON
  POST /offer           → WebRTC SDP offer
  POST /set_avatar      → switch active avatar (live, on existing session)
  POST /text_input      → text bypass (no mic)
"""

import asyncio
import json
import logging
import os
import time
from fractions import Fraction
from pathlib import Path

import aiofiles
import av
import cv2
import numpy as np
from aiohttp import web
from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription
from dotenv import load_dotenv

from pipeline import (
    ASREvent, ASRResult,
    AvatarRegistry, AVATAR_NAMES, DEFAULT_AGENT,
    AVTR1Renderer,
    CliqueBrain,
    ClueTTS,
    FasterWhisperASR,
)

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("berylize.server")

ROOT        = Path(__file__).parent
STATIC_DIR  = ROOT / "static"
AVATARS_DIR = Path(os.environ.get("AVATARS_DIR", "./avatars"))
AVTR1_FRAMES_DIR = os.environ.get(
    "AVTR1_FRAMES_DIR",
    "./checkpoints/avtr1/avatars_artifacts/reference_frames",
)

# ── Global singletons ─────────────────────────────────────────────────────────

registry: AvatarRegistry | None = None
renderer: AVTR1Renderer | None  = None
asr:      FasterWhisperASR | None = None

peer_connections: set[RTCPeerConnection] = set()
sessions: dict[str, "CliqueSession"] = {}   # pc_id → session


# ── WebRTC Tracks ─────────────────────────────────────────────────────────────

class AvatarVideoTrack(MediaStreamTrack):
    kind = "video"

    def __init__(self, rend: AVTR1Renderer):
        super().__init__()
        self.renderer = rend
        self._pts     = 0
        self._time_base = Fraction(1, 25)

    async def recv(self):
        frame_bgr = await self.renderer.get_frame()
        if frame_bgr is None:
            frame_bgr = self.renderer.blank_frame()

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        av_frame  = av.VideoFrame.from_ndarray(frame_rgb, format="rgb24")
        av_frame  = av_frame.reformat(format="yuv420p")
        av_frame.pts       = self._pts
        av_frame.time_base = self._time_base
        self._pts += 1

        await asyncio.sleep(1 / 25)
        return av_frame


class AvatarAudioTrack(MediaStreamTrack):
    kind = "audio"

    def __init__(self):
        super().__init__()
        self.queue: asyncio.Queue[np.ndarray] = asyncio.Queue(maxsize=200)
        self._pts     = 0
        self._silence = np.zeros(3200, dtype=np.float32)
        self._last    = self._silence.copy()

    async def recv(self):
        try:
            chunk = self.queue.get_nowait()
            self._last = chunk
        except asyncio.QueueEmpty:
            chunk = self._last

        pcm = (chunk * 32767).clip(-32768, 32767).astype(np.int16)

        frame = av.AudioFrame(format="s16", layout="mono")
        frame.samples     = len(pcm)
        frame.sample_rate = 16_000
        frame.pts         = self._pts
        frame.time_base   = Fraction(1, 16_000)
        frame.planes[0].update(pcm.tobytes())
        self._pts += len(pcm)

        await asyncio.sleep(3200 / 16_000)   # 200ms
        return frame


# ── Session ───────────────────────────────────────────────────────────────────

class CliqueSession:
    """One per WebRTC peer connection."""

    def __init__(self, avatar_name: str = DEFAULT_AGENT):
        self.active_avatar  = avatar_name
        self.brain          = CliqueBrain(active_agent=avatar_name)
        self.tts            = ClueTTS()
        self.asr_queue: asyncio.Queue[np.ndarray] = asyncio.Queue(maxsize=500)
        self.video_track    = AvatarVideoTrack(renderer)
        self.audio_track    = AvatarAudioTrack()
        self._tasks: list[asyncio.Task] = []
        self._responding    = False

    async def start(self):
        await renderer.set_avatar(self.active_avatar)
        self._tasks.append(asyncio.create_task(self._asr_loop()))

    async def set_avatar(self, name: str):
        self.active_avatar = name
        self.brain.set_agent(name)
        await renderer.set_avatar(name)

    # ── Barge-in ──────────────────────────────────────────────────────────

    def _barge_in(self):
        """Cancel brain + TTS streams immediately on user speech onset."""
        if self._responding:
            logger.info("[SESSION] barge-in — cancelling response")
            self.brain.cancel()
            self.tts.cancel()
            renderer.set_speaking(False)
            asr.set_avatar_speaking(False)
            # flush audio queue
            while not self.audio_track.queue.empty():
                try:
                    self.audio_track.queue.get_nowait()
                except asyncio.QueueEmpty:
                    break

    # ── ASR loop ─────────────────────────────────────────────────────────

    async def _asr_loop(self):
        async for result in asr.transcribe_stream(self.asr_queue):
            if result.event == ASREvent.BARGE_IN:
                self._barge_in()
            elif result.event == ASREvent.TRANSCRIPT and result.text.strip():
                asyncio.create_task(self._respond(result.text))

    # ── Response pipeline ─────────────────────────────────────────────────

    async def _respond(self, text: str):
        if self._responding:
            return   # already responding; barge-in should have cancelled
        self._responding = True
        t0 = time.time()

        renderer.set_speaking(True)
        asr.set_avatar_speaking(True)

        try:
            text_gen  = self.brain.respond(text)
            audio_gen = self.tts.synthesize_stream(text_gen, self.active_avatar)
            first     = True

            async for audio_chunk in audio_gen:
                await renderer.feed_audio(audio_chunk)
                try:
                    self.audio_track.queue.put_nowait(audio_chunk)
                except asyncio.QueueFull:
                    pass
                if first:
                    logger.info(f"[E2E] {(time.time()-t0)*1000:.0f}ms to first frame")
                    first = False

        finally:
            renderer.set_speaking(False)
            asr.set_avatar_speaking(False)
            self._responding = False

    async def handle_text(self, text: str):
        await self._respond(text)

    def feed_mic(self, chunk_i16: np.ndarray):
        try:
            self.asr_queue.put_nowait(chunk_i16)
        except asyncio.QueueFull:
            pass

    async def stop(self):
        for t in self._tasks:
            t.cancel()


# ── Routes ────────────────────────────────────────────────────────────────────

async def index(request: web.Request) -> web.Response:
    async with aiofiles.open(STATIC_DIR / "index.html", "r") as f:
        html = await f.read()
    return web.Response(content_type="text/html", text=html)


async def avatar_image(request: web.Request) -> web.Response:
    name = Path(request.match_info["name"]).name  # strip any path components
    for ext in [".jpg", ".jpeg", ".png", ".webp"]:
        p = AVATARS_DIR / f"{name}{ext}"
        if p.exists():
            async with aiofiles.open(p, "rb") as f:
                data = await f.read()
            ct = "image/jpeg" if ext in (".jpg", ".jpeg") else f"image/{ext[1:]}"
            return web.Response(body=data, content_type=ct)
    return web.Response(status=404)


async def status(request: web.Request) -> web.Response:
    healthy = await renderer.check_health()
    return web.json_response({
        "avtr1":    healthy,
        "avatars":  registry.list_avatars() if registry else [],
        "sessions": len(sessions),
    })


async def offer(request: web.Request) -> web.Response:
    params      = await request.json()
    avatar_name = params.get("avatar", DEFAULT_AGENT)

    session = CliqueSession(avatar_name=avatar_name)
    await session.start()

    pc = RTCPeerConnection()
    pc_id = str(id(pc))
    peer_connections.add(pc)
    sessions[pc_id] = session

    pc.addTrack(session.video_track)
    pc.addTrack(session.audio_track)

    @pc.on("track")
    async def on_track(track):
        if track.kind == "audio":
            asyncio.create_task(_consume_mic(track, session))

    @pc.on("connectionstatechange")
    async def on_state():
        logger.info(f"[WS] state={pc.connectionState}")
        if pc.connectionState in ("failed", "closed"):
            await session.stop()
            peer_connections.discard(pc)
            sessions.pop(pc_id, None)

    await pc.setRemoteDescription(RTCSessionDescription(
        sdp=params["sdp"], type=params.get("type", "offer")
    ))
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return web.json_response({
        "sdp":  pc.localDescription.sdp,
        "type": pc.localDescription.type,
        "session_id": pc_id,
    })


async def _consume_mic(track, session: CliqueSession):
    while True:
        try:
            frame: av.AudioFrame = await track.recv()
            pcm = np.frombuffer(frame.planes[0], dtype=np.int16)
            session.feed_mic(pcm)
        except Exception:
            break


async def set_avatar(request: web.Request) -> web.Response:
    data = await request.json()
    name = data.get("name", DEFAULT_AGENT)
    session_id = data.get("session_id")

    if name not in AVATAR_NAMES:
        return web.json_response({"error": f"Unknown avatar: {name}"}, status=400)

    if session_id and session_id in sessions:
        await sessions[session_id].set_avatar(name)
    else:
        # Switch on all active sessions
        for s in sessions.values():
            await s.set_avatar(name)

    return web.json_response({"status": "ok", "avatar": name})


async def text_input(request: web.Request) -> web.Response:
    data   = await request.json()
    text   = data.get("text", "").strip()
    avatar = data.get("avatar", DEFAULT_AGENT)
    session_id = data.get("session_id")

    if not text:
        return web.json_response({"error": "No text"}, status=400)

    if session_id and session_id in sessions:
        session = sessions[session_id]
    elif sessions:
        session = next(iter(sessions.values()))
    else:
        # No active WebRTC session — create ephemeral one
        session = CliqueSession(avatar_name=avatar)
        await session.start()

    asyncio.create_task(session.handle_text(text))
    return web.json_response({"status": "processing"})


# ── MJPEG Stream (HTTP fallback for WebRTC) ──────────────────────────────────

async def mjpeg_stream(request: web.Request) -> web.StreamResponse:
    resp = web.StreamResponse(
        status=200,
        headers={
            "Content-Type": "multipart/x-mixed-replace; boundary=frame",
            "Cache-Control": "no-cache, no-store",
            "Connection": "close",
        },
    )
    await resp.prepare(request)

    while True:
        frame_bgr = await renderer.get_frame()
        if frame_bgr is None:
            frame_bgr = renderer.blank_frame()

        _, jpeg = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
        data = jpeg.tobytes()

        try:
            await resp.write(
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n"
                b"Content-Length: " + str(len(data)).encode() + b"\r\n\r\n"
                + data + b"\r\n"
            )
        except (ConnectionResetError, ConnectionError):
            break

        await asyncio.sleep(1 / 25)

    return resp


async def latest_frame(request: web.Request) -> web.Response:
    frame_bgr = await renderer.get_frame()
    if frame_bgr is None:
        frame_bgr = renderer.blank_frame()
    _, jpeg = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return web.Response(body=jpeg.tobytes(), content_type="image/jpeg")


async def text_respond(request: web.Request) -> web.Response:
    data = await request.json()
    text = data.get("text", "").strip()
    avatar = data.get("avatar", DEFAULT_AGENT)

    if not text:
        return web.json_response({"error": "No text"}, status=400)

    brain = CliqueBrain(active_agent=avatar)
    full = []
    async for tok in brain.respond(text):
        full.append(tok)
    reply = "".join(full).strip()

    tts = ClueTTS()
    chunks = []

    async def text_gen():
        yield reply

    async for chunk in tts.synthesize_stream(text_gen(), avatar):
        await renderer.feed_audio(chunk)
        chunks.append(len(chunk))

    return web.json_response({"reply": reply, "audio_chunks": len(chunks)})


# ── Startup / Shutdown ────────────────────────────────────────────────────────

async def on_startup(app: web.Application):
    global registry, renderer, asr

    logger.info("=== Berylize startup ===")

    registry = AvatarRegistry(
        avatars_dir=str(AVATARS_DIR),
        avtr1_frames_dir=AVTR1_FRAMES_DIR,
    )

    renderer = AVTR1Renderer()
    await renderer.start_idle()

    asr = FasterWhisperASR()

    # Health check AVTR-1
    healthy = await renderer.check_health()
    if healthy:
        logger.info("AVTR-1 renderer: ONLINE")
    else:
        logger.warning("AVTR-1 renderer: OFFLINE — running in stub mode (start avtr1_renderer separately)")

    logger.info("=== Startup complete ===")


async def on_shutdown(app: web.Application):
    await renderer.stop_idle()
    coros = [pc.close() for pc in peer_connections]
    await asyncio.gather(*coros, return_exceptions=True)
    peer_connections.clear()
    sessions.clear()


def build_app() -> web.Application:
    app = web.Application()
    app.on_startup.append(on_startup)
    app.on_shutdown.append(on_shutdown)

    app.router.add_get("/",               index)
    app.router.add_get("/status",         status)
    app.router.add_get("/avatars/{name}", avatar_image)
    app.router.add_static("/static/", path=str(STATIC_DIR), name="static")
    app.router.add_post("/offer",        offer)
    app.router.add_post("/set_avatar",   set_avatar)
    app.router.add_post("/text_input",   text_input)
    app.router.add_get("/stream",        mjpeg_stream)
    app.router.add_get("/frame",         latest_frame)
    app.router.add_post("/text_respond", text_respond)

    return app


if __name__ == "__main__":
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", 8080))
    logger.info(f"Starting on {host}:{port}")
    web.run_app(build_app(), host=host, port=port)
