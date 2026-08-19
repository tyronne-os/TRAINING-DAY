"""
pipeline/tts.py
OpenAI tts-1-hd streaming TTS.
Buffers LLM text tokens to sentence boundaries, synthesises,
decodes to 16kHz mono float32, yields 200ms chunks (3200 samples)
aligned with AVTR-1's expected input.
"""

import asyncio
import io
import logging
import os
import time
from typing import AsyncGenerator

import numpy as np

logger = logging.getLogger(__name__)

SAMPLE_RATE   = 16_000
CHUNK_SAMPLES = 3200          # 200ms @ 16kHz — matches AVTR-1 chunk size
SENTENCE_ENDS = ".?!,"
MIN_FLUSH_CHARS = 50

VOICES = {
    "evedefault": "nova",
    "jeff":       "onyx",
    "nu":         "nova",
    "india":      "shimmer",
    "amanda":     "alloy",
}


class ClueTTS:
    def __init__(self):
        self._client   = None
        self._cancelled = False
        self._init_client()

    def _init_client(self):
        try:
            from openai import AsyncOpenAI
            self._client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
            logger.info("OpenAI TTS client initialised.")
        except ImportError:
            logger.warning("openai not installed — TTS in STUB mode.")

    def cancel(self):
        """Stop ongoing synthesis (barge-in)."""
        self._cancelled = True

    def reset(self):
        self._cancelled = False

    # ── Decode audio ─────────────────────────────────────────────────────

    def _decode(self, audio_bytes: bytes) -> np.ndarray:
        try:
            import soundfile as sf
            buf = io.BytesIO(audio_bytes)
            data, sr = sf.read(buf, dtype="float32", always_2d=False)
            if data.ndim == 2:
                data = data.mean(axis=1)
            if sr != SAMPLE_RATE:
                from scipy.signal import resample
                data = resample(data, int(len(data) * SAMPLE_RATE / sr)).astype(np.float32)
            return data
        except Exception as e:
            logger.error(f"TTS decode error: {e}")
            return np.zeros(CHUNK_SAMPLES, dtype=np.float32)

    def _chunk(self, audio: np.ndarray) -> list[np.ndarray]:
        """Split to 200ms chunks, zero-pad the last one."""
        out = []
        for i in range(0, len(audio), CHUNK_SAMPLES):
            chunk = audio[i:i + CHUNK_SAMPLES]
            if len(chunk) < CHUNK_SAMPLES:
                chunk = np.pad(chunk, (0, CHUNK_SAMPLES - len(chunk)))
            out.append(chunk)
        return out

    # ── Synthesise one sentence ───────────────────────────────────────────

    async def _synth(self, text: str, agent: str) -> list[np.ndarray]:
        if not text.strip():
            return []

        voice = VOICES.get(agent, "nova")

        if self._client is None:
            # stub: 0.8s of silence
            return [np.zeros(CHUNK_SAMPLES, dtype=np.float32)] * 4

        try:
            t0  = time.time()
            resp = await self._client.audio.speech.create(
                model="tts-1-hd",
                voice=voice,
                input=text.strip(),
                response_format="opus",
            )
            latency_ms = (time.time() - t0) * 1000
            logger.info(f"[TTS] synthesis: {latency_ms:.0f}ms for '{text[:40]}…'")
            audio = self._decode(resp.content)
            return self._chunk(audio)
        except Exception as e:
            logger.error(f"TTS synthesis error: {e}")
            return [np.zeros(CHUNK_SAMPLES, dtype=np.float32)] * 3

    # ── Public streaming API ──────────────────────────────────────────────

    async def synthesize_stream(
        self,
        text_gen: AsyncGenerator[str, None],
        agent: str,
    ) -> AsyncGenerator[np.ndarray, None]:
        """
        Buffer LLM tokens → sentence boundaries → synthesise → yield 200ms float32 chunks.
        Stops immediately if cancel() is called.
        """
        self.reset()
        buf = ""
        first_chunk = True
        t_first_token: float | None = None

        async for token in text_gen:
            if self._cancelled:
                logger.info("[TTS] cancelled by barge-in")
                return

            if t_first_token is None:
                t_first_token = time.time()
            buf += token

            should_flush = (
                any(buf.rstrip().endswith(e) for e in SENTENCE_ENDS)
                and len(buf) >= MIN_FLUSH_CHARS
            ) or len(buf) >= 200

            if should_flush:
                sentence, buf = buf.strip(), ""
                chunks = await self._synth(sentence, agent)
                for chunk in chunks:
                    if self._cancelled:
                        return
                    if first_chunk and t_first_token:
                        logger.info(
                            f"[TTS] first audio: {(time.time()-t_first_token)*1000:.0f}ms after first token"
                        )
                        first_chunk = False
                    yield chunk

        # flush remainder
        if buf.strip() and not self._cancelled:
            for chunk in await self._synth(buf.strip(), agent):
                if self._cancelled:
                    return
                yield chunk
