"""
MuseTalk engine — audio → lip-sync frames.

Accepts 200ms int16 PCM chunks (3200 samples @ 16 kHz).
Outputs 5 BGR frames at 256×256, matching AVTR-1's 25 FPS contract.
"""
from __future__ import annotations

import sys
import os
import logging
from pathlib import Path
from dataclasses import dataclass

import cv2
import numpy as np
import torch

LOG = logging.getLogger(__name__)

WEIGHTS_DIR = Path(os.environ.get("MUSETALK_WEIGHTS", Path.home() / "open_renderer/weights/MuseTalk"))
MUSETALK_REPO = Path(os.environ.get("MUSETALK_REPO", Path.home() / "open_renderer/MuseTalk"))
SAMPLE_RATE = 16_000
CHUNK_SAMPLES = 3200   # 200ms
FRAMES_PER_CHUNK = 5   # 25 FPS × 0.2s
FACE_SIZE = 256


@dataclass
class AvatarContext:
    """Per-avatar state: reference face crops + embeddings."""
    avatar_id: str
    face_image: np.ndarray          # BGR 256×256
    face_embedding: torch.Tensor    # latent from appearance encoder
    prior_frames: list[np.ndarray]  # last N face frames for temporal consistency


class MuseTalkEngine:
    """Wraps MuseTalk for chunk-based streaming inference."""

    def __init__(self) -> None:
        self._loaded = False
        self._vae = None
        self._unet = None
        self._whisper = None
        self._face_det = None
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

    def load(self) -> None:
        if self._loaded:
            return
        if str(MUSETALK_REPO) not in sys.path:
            sys.path.insert(0, str(MUSETALK_REPO))

        LOG.info("[MuseTalk] Loading models from %s", WEIGHTS_DIR)
        try:
            from musetalk.whisper.audio2feature import Audio2Feature
            from musetalk.models.vae import VAE
            from musetalk.models.unet import UNet, PositionNet, AudioProjection

            self._whisper = Audio2Feature(
                model_path=str(WEIGHTS_DIR / "whisper" / "tiny.pt")
            )
            self._vae = VAE(
                model_path=str(WEIGHTS_DIR / "sd-vae-ft-mse")
            ).to(self._device)
            self._unet = UNet(
                unet_config=str(WEIGHTS_DIR / "musetalk" / "musetalk.json"),
                model_path=str(WEIGHTS_DIR / "musetalk" / "pytorch_model.bin"),
            ).to(self._device)
            self._loaded = True
            LOG.info("[MuseTalk] Models loaded on %s", self._device)
        except Exception as exc:
            LOG.error("[MuseTalk] Load failed: %s", exc)
            raise

    def prepare_avatar(self, portrait_bgr: np.ndarray, avatar_id: str) -> AvatarContext:
        """Precompute reference face embedding from a portrait image."""
        self.load()
        face = cv2.resize(portrait_bgr, (FACE_SIZE, FACE_SIZE))
        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face_t = torch.from_numpy(face_rgb).permute(2, 0, 1).float() / 255.0
        face_t = face_t.unsqueeze(0).to(self._device)
        with torch.no_grad():
            latent = self._vae.encode(face_t * 2 - 1).latent_dist.sample()
            latent = latent * 0.18215
        return AvatarContext(
            avatar_id=avatar_id,
            face_image=face,
            face_embedding=latent,
            prior_frames=[],
        )

    def synthesize_chunk(
        self,
        audio_pcm: np.ndarray,           # int16, 3200 samples
        ctx: AvatarContext,
        state_audio_feats: torch.Tensor | None = None,
    ) -> tuple[list[np.ndarray], torch.Tensor]:
        """
        Run one 200ms chunk → 5 BGR frames + updated audio feature state.

        Returns:
            frames: list of 5 BGR np.ndarray (256×256)
            new_state: audio feature tensor for next call (temporal continuity)
        """
        self.load()

        # Audio → whisper features
        audio_f32 = audio_pcm.astype(np.float32) / 32768.0
        whisper_chunks = self._whisper.get_audio_feature(audio_f32, fps=25)
        # whisper_chunks shape: (N, feature_dim) — take FRAMES_PER_CHUNK frames
        if len(whisper_chunks) < FRAMES_PER_CHUNK:
            pad = np.zeros((FRAMES_PER_CHUNK - len(whisper_chunks), whisper_chunks.shape[1]), dtype=np.float32)
            whisper_chunks = np.concatenate([whisper_chunks, pad], axis=0)
        whisper_chunks = whisper_chunks[:FRAMES_PER_CHUNK]

        frames: list[np.ndarray] = []
        with torch.no_grad():
            for i in range(FRAMES_PER_CHUNK):
                feat = torch.from_numpy(whisper_chunks[i:i+1]).to(self._device)
                # Inpaint lower face using MuseTalk UNet
                pred_latent = self._unet(ctx.face_embedding, feat)
                # Decode latent → RGB
                pred_img = self._vae.decode(pred_latent / 0.18215)
                pred_img = (pred_img.clamp(-1, 1) + 1) / 2.0
                pred_np = pred_img[0].permute(1, 2, 0).cpu().numpy()
                pred_np = (pred_np * 255).astype(np.uint8)
                pred_bgr = cv2.cvtColor(pred_np, cv2.COLOR_RGB2BGR)
                pred_bgr = cv2.resize(pred_bgr, (FACE_SIZE, FACE_SIZE))
                frames.append(pred_bgr)

        new_state = torch.from_numpy(whisper_chunks).to(self._device)
        return frames, new_state
