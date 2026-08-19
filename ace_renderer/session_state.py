"""
Session state serialisation for the ACE renderer.

Carries between /process-audio-v3 calls:
  - frame_count: total frames produced in this session
  - prior_blendshapes: last frame's blendshape weights (for smoothing)

Wire format: safetensors blob, read via X-State-Length-Bytes response header.
Compatible with the same header convention used by the open_renderer and AVTR-1.
"""
from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass

import numpy as np
import torch
from safetensors.torch import save_file as st_save_file, load_file as st_load_file


@dataclass
class SessionState:
    frame_count: int = 0
    # Last frame's blendshape weights as a float32 tensor shaped (52,).
    # Stored for temporal smoothing between successive API calls.
    prior_blendshapes: torch.Tensor | None = None

    def to_bytes(self) -> bytes:
        tensors: dict[str, torch.Tensor] = {
            "frame_count": torch.tensor([self.frame_count], dtype=torch.int64),
        }
        if self.prior_blendshapes is not None:
            tensors["prior_blendshapes"] = self.prior_blendshapes.cpu().float()
        fd, tmp = tempfile.mkstemp(suffix=".safetensors")
        os.close(fd)
        try:
            st_save_file(tensors, tmp)
            with open(tmp, "rb") as f:
                return f.read()
        finally:
            os.unlink(tmp)

    @classmethod
    def from_bytes(cls, data: bytes | None) -> "SessionState":
        if not data:
            return cls()
        try:
            fd, tmp = tempfile.mkstemp(suffix=".safetensors")
            os.close(fd)
            try:
                with open(tmp, "wb") as f:
                    f.write(data)
                tensors = st_load_file(tmp)
            finally:
                os.unlink(tmp)
            return cls(
                frame_count=int(tensors["frame_count"][0].item()),
                prior_blendshapes=tensors.get("prior_blendshapes"),
            )
        except Exception:
            return cls()
