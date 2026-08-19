"""
FasterLivePortrait engine — head pose + expression naturalness layer.

Takes MuseTalk's mouth-animated frames and applies natural head movement,
eye blinks, and subtle expression drift using LivePortrait's retargeting.
This upgrades MuseTalk from "moving mouth only" to full-face lifelike animation.
"""
from __future__ import annotations

import sys
import os
import logging
from pathlib import Path

import cv2
import numpy as np
import torch

LOG = logging.getLogger(__name__)

FLP_REPO = Path(os.environ.get("FLP_REPO", Path.home() / "open_renderer/FasterLivePortrait"))
LP_WEIGHTS = Path(os.environ.get("LP_WEIGHTS", Path.home() / "open_renderer/weights/LivePortrait"))

# Idle motion params — subtle, not robotic
BLINK_INTERVAL_FRAMES = 75    # ~3 seconds at 25 FPS
GAZE_DRIFT_SCALE = 0.008      # small natural gaze wander
HEAD_DRIFT_SCALE = 0.004      # subtle head micro-movement


class LivePortraitEngine:
    """Wraps FasterLivePortrait for per-frame expression retargeting."""

    def __init__(self) -> None:
        self._loaded = False
        self._pipeline = None
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._frame_count = 0

    def load(self) -> None:
        if self._loaded:
            return
        if str(FLP_REPO) not in sys.path:
            sys.path.insert(0, str(FLP_REPO))
        try:
            from src.config.argument_config import ArgumentConfig
            from src.config.inference_config import InferenceConfig
            from src.config.crop_config import CropConfig
            from src.live_portrait_pipeline import LivePortraitPipeline

            args = ArgumentConfig(
                source_image="",
                driving_video="",
                output_dir="",
            )
            inference_cfg = InferenceConfig(
                models_path=str(LP_WEIGHTS),
                flag_use_half_precision=True,
            )
            crop_cfg = CropConfig()
            self._pipeline = LivePortraitPipeline(
                inference_cfg=inference_cfg,
                crop_cfg=crop_cfg,
            )
            self._loaded = True
            LOG.info("[LivePortrait] Loaded on %s", self._device)
        except Exception as exc:
            LOG.warning("[LivePortrait] Could not load — running without head animation: %s", exc)
            self._pipeline = None
            self._loaded = True   # don't retry; degrade gracefully

    def _idle_motion(self, frame_idx: int) -> dict:
        """Generate naturalistic idle head/gaze motion coefficients."""
        t = frame_idx / 25.0  # seconds

        # Slow sinusoidal gaze drift
        gaze_x = np.sin(t * 0.3) * GAZE_DRIFT_SCALE
        gaze_y = np.cos(t * 0.17) * GAZE_DRIFT_SCALE

        # Micro head movement
        head_pitch = np.sin(t * 0.13) * HEAD_DRIFT_SCALE
        head_yaw   = np.cos(t * 0.09) * HEAD_DRIFT_SCALE

        # Blink — sharp on/off every ~3 seconds
        blink = 0.0
        phase = frame_idx % BLINK_INTERVAL_FRAMES
        if phase < 3:
            blink = float(phase) / 3.0
        elif phase < 6:
            blink = 1.0 - float(phase - 3) / 3.0

        return {
            "gaze": [gaze_x, gaze_y],
            "head_pitch": head_pitch,
            "head_yaw": head_yaw,
            "blink": blink,
        }

    def apply_expression(
        self,
        frames: list[np.ndarray],
        source_portrait: np.ndarray,
    ) -> list[np.ndarray]:
        """
        Apply head pose + expression retargeting to a batch of frames.

        If LivePortrait failed to load, returns input frames unchanged
        (graceful degradation — MuseTalk lip sync still works).
        """
        self.load()
        if self._pipeline is None:
            self._frame_count += len(frames)
            return frames

        result: list[np.ndarray] = []
        try:
            for frame in frames:
                motion = self._idle_motion(self._frame_count)
                self._frame_count += 1
                # Resize to LP's expected input
                src = cv2.resize(source_portrait, (256, 256))
                drv = cv2.resize(frame, (256, 256))
                # Use LivePortrait's retargeting to warp expression onto source
                out = self._pipeline.execute_frame(
                    source_rgb=cv2.cvtColor(src, cv2.COLOR_BGR2RGB),
                    driving_rgb=cv2.cvtColor(drv, cv2.COLOR_BGR2RGB),
                    motion_override=motion,
                )
                if out is not None:
                    out_bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
                    result.append(cv2.resize(out_bgr, (256, 256)))
                else:
                    result.append(frame)
        except Exception as exc:
            LOG.warning("[LivePortrait] Frame error, passing through: %s", exc)
            result = frames

        return result
