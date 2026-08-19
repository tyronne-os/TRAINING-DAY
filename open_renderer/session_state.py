"""
Session state serialisation — replaces AVTR-1's safetensors state blob.

AVTR-1 carried a past_context tensor (70 × 512) between calls.
We carry: prior whisper audio features + prior frame embeddings + frame counter.
Same wire format: safetensors blob, split off with X-State-Length-Bytes header.
"""
from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np
import torch
from safetensors.torch import save as st_save, load as st_load


@dataclass
class SessionState:
    frame_count: int = 0
    prior_audio_feats: torch.Tensor | None = None   # (N, feat_dim)
    prior_face_latent: torch.Tensor | None = None   # (1, 4, h, w)

    def to_bytes(self) -> bytes:
        tensors: dict[str, torch.Tensor] = {
            "frame_count": torch.tensor([self.frame_count], dtype=torch.int64),
        }
        if self.prior_audio_feats is not None:
            tensors["prior_audio_feats"] = self.prior_audio_feats.cpu().float()
        if self.prior_face_latent is not None:
            tensors["prior_face_latent"] = self.prior_face_latent.cpu().float()
        buf = io.BytesIO()
        st_save(tensors, buf)
        return buf.getvalue()

    @classmethod
    def from_bytes(cls, data: bytes | None) -> "SessionState":
        if not data:
            return cls()
        try:
            tensors = st_load(data)
            return cls(
                frame_count=int(tensors["frame_count"][0].item()),
                prior_audio_feats=tensors.get("prior_audio_feats"),
                prior_face_latent=tensors.get("prior_face_latent"),
            )
        except Exception:
            return cls()
