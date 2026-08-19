"""
Frame packer — converts BGR np.ndarray frames to YUV I420 bytes.

AVTR-1 returned raw YUV I420 frames concatenated after the state blob.
Same wire format so berylize/pipeline/renderer.py needs zero changes.
"""
from __future__ import annotations

import numpy as np
import cv2


def bgr_to_yuv_i420(frame_bgr: np.ndarray, width: int, height: int) -> bytes:
    """Convert one BGR frame to YUV I420 bytes at the requested resolution."""
    resized = cv2.resize(frame_bgr, (width, height), interpolation=cv2.INTER_LINEAR)
    yuv = cv2.cvtColor(resized, cv2.COLOR_BGR2YUV_I420)
    return yuv.tobytes()


def pack_frames(frames_bgr: list[np.ndarray], width: int = 512, height: int = 512) -> bytes:
    """Concatenate N YUV I420 frames into raw bytes (same format as AVTR-1 output)."""
    return b"".join(bgr_to_yuv_i420(f, width, height) for f in frames_bgr)


def bytes_per_yuv_i420_frame(width: int, height: int) -> int:
    """Byte count for a single planar YUV I420 frame."""
    return width * height * 3 // 2
