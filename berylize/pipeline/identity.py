"""
pipeline/identity.py
Avatar registry for AVTR-1.
Each avatar is a portrait image uploaded to the AVTR-1 renderer's
reference_frames directory. This module manages the mapping and
provides a simple registration API.
"""

import logging
import os
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)

AVATAR_NAMES = ["evedefault", "jeff", "nu", "india", "amanda"]

DEFAULT_AGENT = "evedefault"


class AvatarRegistry:
    """
    Maps avatar names to portrait images.
    On startup, copies any missing portraits into the AVTR-1
    reference_frames directory so avatar_id lookups resolve correctly.
    """

    def __init__(self, avatars_dir: str, avtr1_frames_dir: str | None = None):
        self.avatars_dir    = Path(avatars_dir)
        self.avtr1_dir      = Path(avtr1_frames_dir) if avtr1_frames_dir else None
        self._registry: dict[str, Path] = {}
        self._sync()

    def _sync(self):
        """
        Walk avatars_dir for known names, register each one,
        and optionally copy into the AVTR-1 reference_frames dir.
        """
        for name in AVATAR_NAMES:
            for ext in [".jpg", ".jpeg", ".png", ".webp"]:
                candidate = self.avatars_dir / f"{name}{ext}"
                if candidate.exists():
                    self._registry[name] = candidate
                    logger.info(f"[REGISTRY] {name} → {candidate}")
                    if self.avtr1_dir:
                        self._copy_to_avtr1(name, candidate)
                    break
            else:
                logger.warning(f"[REGISTRY] No portrait found for '{name}' in {self.avatars_dir}")

    def _copy_to_avtr1(self, name: str, src: Path):
        """Place portrait in AVTR-1 reference_frames so avatar_id resolves."""
        dest_dir = self.avtr1_dir
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / f"{name}{src.suffix}"
        if not dest.exists():
            shutil.copy2(src, dest)
            logger.info(f"[REGISTRY] copied {src.name} → {dest}")

    def get_portrait(self, name: str) -> Path | None:
        return self._registry.get(name)

    def register(self, name: str, image_path: str | Path):
        """Manually register a new avatar portrait at runtime."""
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Portrait not found: {path}")
        self._registry[name] = path
        if name not in AVATAR_NAMES:
            AVATAR_NAMES.append(name)
        if self.avtr1_dir:
            self._copy_to_avtr1(name, path)
        logger.info(f"[REGISTRY] registered {name}")

    def list_avatars(self) -> list[str]:
        return list(self._registry.keys())

    def portrait_url(self, name: str) -> str | None:
        p = self._registry.get(name)
        return f"/avatars/{name}" if p else None
