"""
Blendshape renderer — maps 52 ARKit coefficients to pixel-level face deformations.

NVIDIA Audio2Face-3D returns blendshape weights, not video frames.  This module
applies those weights to a static portrait image to synthesise an animated BGR
frame.  It is a best-effort 2D approximation that proves the end-to-end pipeline.

Implemented deformations (all others are no-ops in this approximation):
  JawOpen           — lower-face region warp downward + lip part
  EyeBlinkLeft/R    — eye-region darkening / closing blend
  EyeWideLeft/R     — upper-eyelid lift
  MouthSmileLeft/R  — mouth-corner upward pull
  MouthFrownLeft/R  — mouth-corner downward pull
  MouthFunnel       — lip-pucker / rounding
  BrowDownLeft/R    — brow lowering (inward shift)
  BrowInnerUp       — inner brow raise
  CheekPuff         — cheek region lateral expand

All transforms use cv2.remap (displacement-field warp) or alpha-blend compositing
on a working resolution of 256x256.  The portrait is cached after the first resize.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import NamedTuple

import cv2
import numpy as np

LOG = logging.getLogger("blendshape_renderer")

# Working resolution — upscale happens in frame_packer.
_W = 256
_H = 256

# ── Approximate facial region anchors (for a centred 256×256 face) ──────────
# These are tuned for a mid-closeup portrait; coarse but visually plausible.
_BROW_Y = 68           # brow baseline
_EYE_L_X, _EYE_L_Y = 80, 88      # left-eye centre
_EYE_R_X, _EYE_R_Y = 176, 88     # right-eye centre
_EYE_W, _EYE_H = 36, 14          # half-widths of eye ellipse masks
_MOUTH_Y = 172         # mouth centre y
_MOUTH_X = 128         # mouth centre x
_JAW_Y = 210           # jaw hinge y (below mouth)
_LOWER_FACE_TOP = 140  # y above which jaw warp has no effect


class BlendshapeRenderer:
    """
    Thread-compatible (but not thread-safe) per-avatar renderer.

    Instantiate once per avatar; call render_frame() per animation frame.
    """

    def __init__(self, portrait_bgr: np.ndarray) -> None:
        self._base: np.ndarray = cv2.resize(
            portrait_bgr, (_W, _H), interpolation=cv2.INTER_AREA
        ).astype(np.float32)
        # Pre-build identity remap grids (float32 required by cv2.remap)
        yy, xx = np.mgrid[0:_H, 0:_W].astype(np.float32)
        self._xx_id = xx
        self._yy_id = yy

    # ── Public API ───────────────────────────────────────────────────────────

    def render_frame(self, weights: dict[str, float]) -> np.ndarray:
        """
        Apply blendshape weights to the portrait and return a BGR uint8 frame.

        weights: mapping of ARKit blendshape name → coefficient in [0, 1].
        Missing names default to 0.
        """
        g = weights.get  # shorthand

        # Build displacement maps (dx, dy) initialised to identity.
        dx = self._xx_id.copy()
        dy = self._yy_id.copy()

        # ── Jaw open ─────────────────────────────────────────────────────────
        jaw_open = g("JawOpen", 0.0)
        if jaw_open > 0.01:
            max_shift_px = 22.0
            shift = jaw_open * max_shift_px
            # Smooth ramp starting at _LOWER_FACE_TOP, full shift at _JAW_Y.
            ramp = np.clip(
                (self._yy_id - _LOWER_FACE_TOP) / (_JAW_Y - _LOWER_FACE_TOP), 0.0, 1.0
            )
            dy -= ramp * shift  # pull source pixels upward → appears to push face down

        # ── Mouth smile ───────────────────────────────────────────────────────
        smile_l = g("MouthSmileLeft", 0.0)
        smile_r = g("MouthSmileRight", 0.0)
        if smile_l > 0.01 or smile_r > 0.01:
            mouth_mask = _gaussian_mask(self._yy_id, self._xx_id, _MOUTH_Y, _MOUTH_X, sy=24, sx=44)
            left_mask = mouth_mask * np.clip((_MOUTH_X - self._xx_id) / _MOUTH_X, 0.0, 1.0)
            right_mask = mouth_mask * np.clip((self._xx_id - _MOUTH_X) / (_W - _MOUTH_X), 0.0, 1.0)
            corner_lift = 10.0
            dy -= left_mask  * smile_l * corner_lift
            dy -= right_mask * smile_r * corner_lift
            dx -= left_mask  * smile_l * 6.0
            dx += right_mask * smile_r * 6.0

        # ── Mouth frown ───────────────────────────────────────────────────────
        frown_l = g("MouthFrownLeft", 0.0)
        frown_r = g("MouthFrownRight", 0.0)
        if frown_l > 0.01 or frown_r > 0.01:
            mouth_mask = _gaussian_mask(self._yy_id, self._xx_id, _MOUTH_Y, _MOUTH_X, sy=22, sx=40)
            left_mask = mouth_mask * np.clip((_MOUTH_X - self._xx_id) / _MOUTH_X, 0.0, 1.0)
            right_mask = mouth_mask * np.clip((self._xx_id - _MOUTH_X) / (_W - _MOUTH_X), 0.0, 1.0)
            dy += left_mask  * frown_l * 8.0
            dy += right_mask * frown_r * 8.0

        # ── Brow down ─────────────────────────────────────────────────────────
        brow_dl = g("BrowDownLeft", 0.0)
        brow_dr = g("BrowDownRight", 0.0)
        if brow_dl > 0.01 or brow_dr > 0.01:
            left_brow  = _gaussian_mask(self._yy_id, self._xx_id, _BROW_Y, _EYE_L_X, sy=12, sx=28)
            right_brow = _gaussian_mask(self._yy_id, self._xx_id, _BROW_Y, _EYE_R_X, sy=12, sx=28)
            dy -= left_brow  * brow_dl * 8.0
            dy -= right_brow * brow_dr * 8.0
            dx += left_brow  * brow_dl * 6.0
            dx -= right_brow * brow_dr * 6.0

        # ── Brow inner up ─────────────────────────────────────────────────────
        brow_up = g("BrowInnerUp", 0.0)
        if brow_up > 0.01:
            inner_mask = _gaussian_mask(self._yy_id, self._xx_id, _BROW_Y, _MOUTH_X, sy=12, sx=32)
            dy += inner_mask * brow_up * 6.0

        # ── Apply displacement remap ───────────────────────────────────────────
        warped = cv2.remap(
            self._base, dx, dy,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )

        # ── Eye blink (alpha-blend over warped result) ────────────────────────
        blink_l = g("EyeBlinkLeft", 0.0)
        blink_r = g("EyeBlinkRight", 0.0)
        if blink_l > 0.01 or blink_r > 0.01:
            warped = _apply_blink(warped, blink_l, blink_r)

        # ── Eye wide (slight brow/lid lift — done via overlay brightening) ────
        wide_l = g("EyeWideLeft", 0.0)
        wide_r = g("EyeWideRight", 0.0)
        if wide_l > 0.02 or wide_r > 0.02:
            warped = _apply_eye_wide(warped, wide_l, wide_r)

        # ── Cheek puff (lateral stretch of cheek regions) ─────────────────────
        cheek = g("CheekPuff", 0.0)
        if cheek > 0.02:
            warped = _apply_cheek_puff(warped, cheek)

        # ── Mouth funnel / pucker (lip vertical squeeze + slight protrude) ────
        funnel = g("MouthFunnel", 0.0)
        pucker = g("MouthPucker", 0.0)
        funnel_strength = max(funnel, pucker)
        if funnel_strength > 0.02:
            warped = _apply_mouth_pucker(warped, funnel_strength)

        return np.clip(warped, 0, 255).astype(np.uint8)


# ── Helper functions ─────────────────────────────────────────────────────────


def _gaussian_mask(
    yy: np.ndarray,
    xx: np.ndarray,
    cy: float,
    cx: float,
    sy: float,
    sx: float,
) -> np.ndarray:
    """Return a 2-D float32 Gaussian mask centred at (cy, cx) with std sy, sx."""
    return np.exp(-0.5 * (((yy - cy) / sy) ** 2 + ((xx - cx) / sx) ** 2)).astype(
        np.float32
    )


def _apply_blink(frame: np.ndarray, blink_l: float, blink_r: float) -> np.ndarray:
    """Darken / close eye regions proportional to blink coefficients."""
    yy, xx = np.mgrid[0:_H, 0:_W].astype(np.float32)
    mask_l = _gaussian_mask(yy, xx, _EYE_L_Y, _EYE_L_X, sy=_EYE_H, sx=_EYE_W)
    mask_r = _gaussian_mask(yy, xx, _EYE_R_Y, _EYE_R_X, sy=_EYE_H, sx=_EYE_W)
    # Create a darkened "closed eyelid" layer (brownish-skin tone average)
    skin = frame.mean(axis=(0, 1), keepdims=True) * 0.85
    closed = np.ones_like(frame) * skin

    alpha_l = (blink_l * mask_l)[..., None]
    alpha_r = (blink_r * mask_r)[..., None]
    result = frame * (1.0 - alpha_l) + closed * alpha_l
    result = result * (1.0 - alpha_r) + closed * alpha_r
    return result.astype(np.float32)


def _apply_eye_wide(frame: np.ndarray, wide_l: float, wide_r: float) -> np.ndarray:
    """Slightly brighten upper-eyelid region to simulate widened eyes."""
    yy, xx = np.mgrid[0:_H, 0:_W].astype(np.float32)
    lid_l = _gaussian_mask(yy, xx, _EYE_L_Y - 6, _EYE_L_X, sy=7, sx=_EYE_W)
    lid_r = _gaussian_mask(yy, xx, _EYE_R_Y - 6, _EYE_R_X, sy=7, sx=_EYE_W)
    bright = np.ones_like(frame) * 1.15
    alpha_l = (wide_l * 0.4 * lid_l)[..., None]
    alpha_r = (wide_r * 0.4 * lid_r)[..., None]
    result = frame * (1.0 - alpha_l) + frame * bright * alpha_l
    result = result * (1.0 - alpha_r) + result * bright * alpha_r
    return result.astype(np.float32)


def _apply_cheek_puff(frame: np.ndarray, amount: float) -> np.ndarray:
    """Laterally stretch the cheek regions outward."""
    yy, xx = np.mgrid[0:_H, 0:_W].astype(np.float32)
    cheek_y = _MOUTH_Y - 10
    cheek_l_mask = _gaussian_mask(yy, xx, cheek_y, 55, sy=28, sx=30)
    cheek_r_mask = _gaussian_mask(yy, xx, cheek_y, 201, sy=28, sx=30)
    dx = xx.copy()
    dx += cheek_l_mask * amount * 10.0
    dx -= cheek_r_mask * amount * 10.0
    return cv2.remap(
        frame, dx, yy, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
    ).astype(np.float32)


def _apply_mouth_pucker(frame: np.ndarray, amount: float) -> np.ndarray:
    """Vertically compress and slightly protrude the lip region."""
    yy, xx = np.mgrid[0:_H, 0:_W].astype(np.float32)
    lip_mask = _gaussian_mask(yy, xx, _MOUTH_Y, _MOUTH_X, sy=18, sx=32)
    # Vertical squeeze: pull upper lip down, lower lip up.
    upper = (yy < _MOUTH_Y).astype(np.float32)
    lower = (yy >= _MOUTH_Y).astype(np.float32)
    dy = yy.copy()
    dy += lip_mask * upper * amount * 6.0
    dy -= lip_mask * lower * amount * 6.0
    return cv2.remap(
        frame, xx, dy, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
    ).astype(np.float32)
