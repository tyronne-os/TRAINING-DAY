"""
pipeline/asr.py
faster-whisper streaming ASR with silero-VAD.
Emits text segments and BARGE_IN events.
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import AsyncGenerator

import numpy as np

logger = logging.getLogger(__name__)

SAMPLE_RATE      = 16_000
CHUNK_MS         = 30          # VAD chunk size in ms
CHUNK_SAMPLES    = int(SAMPLE_RATE * CHUNK_MS / 1000)   # 480 samples
SPEECH_PAD_MS    = 200         # pad before/after speech
SILENCE_DURATION = 0.5         # seconds of silence to trigger flush


class ASREvent(Enum):
    TRANSCRIPT = "transcript"
    BARGE_IN   = "barge_in"
    VAD_START  = "vad_start"
    VAD_END    = "vad_end"


@dataclass
class ASRResult:
    event:      ASREvent
    text:       str = ""
    confidence: float = 1.0
    timestamp:  float = 0.0


class FasterWhisperASR:
    """
    Streaming ASR using faster-whisper (CTranslate2 INT8).
    VAD via silero-vad v5 — fires BARGE_IN when speech detected
    while avatar is speaking.
    """

    def __init__(self, model_size: str = "distil-large-v3"):
        self.model_size = model_size
        self._model     = None
        self._vad       = None
        self._vad_utils = None
        self._speaking  = False   # avatar is currently speaking TTS
        self._load()

    # ── Load ────────────────────────────────────────────────────────────

    def _load(self):
        self._load_vad()
        self._load_whisper()

    def _load_vad(self):
        try:
            import torch
            model, utils = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                force_reload=False,
                trust_repo=True,
            )
            self._vad = model
            self._vad_utils = utils
            logger.info("silero-VAD loaded.")
        except Exception as e:
            logger.warning(f"silero-VAD load failed ({e}) — VAD in stub mode.")

    def _load_whisper(self):
        try:
            from faster_whisper import WhisperModel
            self._model = WhisperModel(
                self.model_size,
                device="cuda",
                compute_type="int8_float16",
            )
            logger.info(f"faster-whisper {self.model_size} loaded on CUDA.")
        except Exception as e:
            logger.warning(f"faster-whisper load failed ({e}) — ASR in stub mode.")

    # ── VAD ─────────────────────────────────────────────────────────────

    def _vad_prob(self, chunk_f32: np.ndarray) -> float:
        """Return speech probability for a 30ms chunk."""
        if self._vad is None:
            rms = float(np.sqrt(np.mean(chunk_f32 ** 2)))
            return 1.0 if rms > 0.01 else 0.0
        try:
            import torch
            tensor = torch.from_numpy(chunk_f32).unsqueeze(0)
            return float(self._vad(tensor, SAMPLE_RATE).item())
        except Exception:
            return 0.0

    # ── Transcribe ───────────────────────────────────────────────────────

    def _transcribe(self, audio_f32: np.ndarray) -> str:
        if self._model is None:
            return f"[stub: {len(audio_f32)} samples]"
        try:
            segments, _ = self._model.transcribe(
                audio_f32,
                language="en",
                beam_size=1,
                vad_filter=False,
                condition_on_previous_text=False,
            )
            return " ".join(s.text.strip() for s in segments).strip()
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return ""

    # ── Public ───────────────────────────────────────────────────────────

    def set_avatar_speaking(self, speaking: bool):
        """Call when avatar TTS starts/stops so barge-in can fire."""
        self._speaking = speaking

    async def transcribe_stream(
        self,
        audio_queue: asyncio.Queue,
    ) -> AsyncGenerator[ASRResult, None]:
        """
        Consumes int16 chunks from audio_queue.
        Yields ASRResult events:
          - BARGE_IN   → user spoke while avatar was speaking
          - VAD_START  → speech onset
          - TRANSCRIPT → completed utterance
        """
        speech_buffer: list[np.ndarray] = []
        residual       = np.zeros(0, dtype=np.float32)
        in_speech      = False
        silence_frames = 0
        SILENCE_NEEDED = int(SILENCE_DURATION * 1000 / CHUNK_MS)
        t_speech_start: float | None = None

        while True:
            try:
                chunk_i16: np.ndarray = await asyncio.wait_for(
                    audio_queue.get(), timeout=0.05
                )
            except asyncio.TimeoutError:
                # flush on prolonged silence
                if in_speech:
                    silence_frames += 1
                    if silence_frames > SILENCE_NEEDED * 2:
                        in_speech = False
                        if speech_buffer:
                            audio = np.concatenate(speech_buffer)
                            t0 = time.time()
                            loop = asyncio.get_event_loop()
                            text = await loop.run_in_executor(
                                None, self._transcribe, audio
                            )
                            latency_ms = (time.time() - t0) * 1000
                            logger.info(f"[ASR] {latency_ms:.0f}ms → '{text}'")
                            if text:
                                yield ASRResult(
                                    event=ASREvent.TRANSCRIPT,
                                    text=text,
                                    timestamp=t_speech_start or time.time(),
                                )
                            speech_buffer = []
                            silence_frames = 0
                            t_speech_start = None
                continue

            # int16 → float32
            chunk_f32 = chunk_i16.astype(np.float32) / 32768.0

            # accumulate to VAD chunk size
            residual = np.concatenate([residual, chunk_f32])
            while len(residual) >= CHUNK_SAMPLES:
                vad_chunk  = residual[:CHUNK_SAMPLES]
                residual   = residual[CHUNK_SAMPLES:]
                prob       = self._vad_prob(vad_chunk)
                is_speech  = prob > 0.5

                if is_speech:
                    silence_frames = 0
                    if not in_speech:
                        in_speech = True
                        t_speech_start = time.time()
                        yield ASRResult(event=ASREvent.VAD_START, timestamp=t_speech_start)
                        # BARGE_IN: user started speaking while avatar was talking
                        if self._speaking:
                            logger.info("[ASR] BARGE_IN detected")
                            yield ASRResult(event=ASREvent.BARGE_IN, timestamp=t_speech_start)
                    speech_buffer.append(vad_chunk)

                elif in_speech:
                    silence_frames += 1
                    speech_buffer.append(vad_chunk)   # include trailing silence

                    if silence_frames >= SILENCE_NEEDED:
                        in_speech = False
                        yield ASRResult(event=ASREvent.VAD_END)

                        if speech_buffer:
                            audio = np.concatenate(speech_buffer)
                            t0 = time.time()
                            loop = asyncio.get_event_loop()
                            text = await loop.run_in_executor(
                                None, self._transcribe, audio
                            )
                            latency_ms = (time.time() - t0) * 1000
                            logger.info(f"[ASR] {latency_ms:.0f}ms → '{text}'")
                            if text:
                                yield ASRResult(
                                    event=ASREvent.TRANSCRIPT,
                                    text=text,
                                    timestamp=t_speech_start or time.time(),
                                )
                        speech_buffer  = []
                        silence_frames = 0
                        t_speech_start = None
