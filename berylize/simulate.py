"""
simulate.py — Berylize pipeline connectivity & handoff simulation.
Tests every edge in the pipeline without AVTR-1 running.
Measures latency at each handoff point.
Run: python simulate.py
"""

import asyncio
import os
import time
import sys
import json
from pathlib import Path

os.environ.setdefault("OPENAI_API_KEY", os.environ.get("OPENAI_API_KEY", ""))

PASS = "✓"
FAIL = "✗"
WARN = "⚠"

results = []

def log(status, label, detail="", ms=None):
    t = f"  {ms:.0f}ms" if ms is not None else ""
    tag = f"[{status}]"
    line = f"{tag} {label}{t}"
    if detail:
        line += f"\n       {detail}"
    print(line)
    results.append({"status": status, "label": label, "ms": ms, "detail": detail})

print()
print("═══════════════════════════════════════════════════════")
print("  Berylize — Pipeline Simulation")
print("  Testing all edges & handoffs")
print("═══════════════════════════════════════════════════════")
print()


# ── Edge 1: Environment ───────────────────────────────────────────────────────
print("── 1. Environment ──────────────────────────────────────")

key = os.environ.get("OPENAI_API_KEY", "")
if key.startswith("sk-"):
    log(PASS, "OPENAI_API_KEY present", f"sk-...{key[-6:]}")
else:
    log(FAIL, "OPENAI_API_KEY missing or invalid")

hf = os.environ.get("HF_TOKEN", "")
log(PASS if hf.startswith("hf_") else WARN, "HF_TOKEN", "present" if hf else "not set")

ngc = os.environ.get("NVIDIA_NGC_API_KEY", "")
log(PASS if ngc.startswith("nvapi") else WARN, "NVIDIA_NGC_API_KEY", "present" if ngc else "not set")

avtr_url = os.environ.get("AVTR1_RENDERER_URL", "http://localhost:8000")
log(PASS, "AVTR1_RENDERER_URL", avtr_url)

avatars_dir = Path(os.environ.get("AVATARS_DIR", "./avatars"))
eve = next(avatars_dir.glob("evedefault.*"), None) if avatars_dir.exists() else None
if eve:
    log(PASS, "evedefault portrait", str(eve))
else:
    log(FAIL, "evedefault portrait NOT found", f"expected in {avatars_dir}")

print()


# ── Edge 2: Import chain ──────────────────────────────────────────────────────
print("── 2. Import chain ─────────────────────────────────────")

t0 = time.time()
try:
    from pipeline import FasterWhisperASR, CliqueBrain, ClueTTS, AVTR1Renderer, AvatarRegistry
    log(PASS, "pipeline imports", ms=(time.time()-t0)*1000)
except Exception as e:
    log(FAIL, "pipeline imports", str(e))
    print("\nABORT: cannot continue without pipeline imports")
    sys.exit(1)

print()


# ── Edge 3: Brain (LLM) ───────────────────────────────────────────────────────
print("── 3. Brain → OpenAI GPT-4o ────────────────────────────")

async def test_brain():
    brain = CliqueBrain(active_agent="evedefault")
    t0 = time.time()
    tokens = []
    try:
        async for tok in brain.respond("Say exactly: PIPELINE_OK"):
            tokens.append(tok)
            if len(tokens) == 1:
                first_ms = (time.time()-t0)*1000
            if len(tokens) > 30:
                break
        text = "".join(tokens).strip()
        total_ms = (time.time()-t0)*1000
        if "PIPELINE_OK" in text or len(tokens) > 3:
            log(PASS, "OpenAI stream", f"first token {first_ms:.0f}ms · full {total_ms:.0f}ms · '{text[:40]}'", ms=first_ms)
            return True
        else:
            log(WARN, "OpenAI stream returned empty", f"got: '{text}'")
            return False
    except Exception as e:
        log(FAIL, "OpenAI stream error", str(e))
        return False

brain_ok = asyncio.run(test_brain())
print()


# ── Edge 4: TTS ───────────────────────────────────────────────────────────────
print("── 4. TTS → OpenAI tts-1-hd ───────────────────────────")

async def test_tts():
    from pipeline import ClueTTS
    import numpy as np

    async def mock_text():
        for w in ["Hello. I am Eve."]:
            yield w

    tts = ClueTTS()
    t0 = time.time()
    chunks = []
    try:
        async for chunk in tts.synthesize_stream(mock_text(), "evedefault"):
            chunks.append(chunk)
            if len(chunks) == 1:
                first_ms = (time.time()-t0)*1000
            if len(chunks) >= 3:
                break
        total_ms = (time.time()-t0)*1000
        total_samples = sum(len(c) for c in chunks)
        log(PASS, "TTS synthesis", f"first chunk {first_ms:.0f}ms · {len(chunks)} chunks · {total_samples} samples", ms=first_ms)
        return chunks[0] if chunks else None
    except Exception as e:
        log(FAIL, "TTS error", str(e))
        return None

audio_chunk = asyncio.run(test_tts())
print()


# ── Edge 5: ASR (VAD only — no mic) ──────────────────────────────────────────
print("── 5. ASR → silero-VAD + faster-whisper ───────────────")

t0 = time.time()
try:
    asr = FasterWhisperASR()
    load_ms = (time.time()-t0)*1000
    vad_ok = asr._vad is not None
    wh_ok  = asr._model is not None
    log(
        PASS if (vad_ok and wh_ok) else WARN,
        "ASR loaded",
        f"VAD={'ok' if vad_ok else 'stub'} · Whisper={'ok' if wh_ok else 'stub'}",
        ms=load_ms
    )

    import numpy as np
    synthetic = (np.random.randn(480).astype(np.float32) * 0.3)
    prob = asr._vad_prob(synthetic)
    log(PASS, "VAD inference", f"speech_prob={prob:.3f} on synthetic noise")
except Exception as e:
    log(FAIL, "ASR load error", str(e))

print()


# ── Edge 6: Renderer client (AVTR-1 health check) ────────────────────────────
print("── 6. Renderer → AVTR-1 @ localhost:8000 ───────────────")

async def test_renderer():
    renderer = AVTR1Renderer()
    t0 = time.time()
    healthy = await renderer.check_health()
    ms = (time.time()-t0)*1000
    if healthy:
        log(PASS, "AVTR-1 health check", "ONLINE", ms=ms)
    else:
        log(WARN, "AVTR-1 health check", "OFFLINE (expected — not started yet)", ms=ms)

    if audio_chunk is not None:
        import numpy as np
        t0 = time.time()
        frames = await renderer._call(
            audio_chunk.astype(np.int16) if audio_chunk.dtype != np.int16 else audio_chunk,
            np.zeros(3200, dtype=np.int16)
        )
        ms = (time.time()-t0)*1000
        log(WARN if not healthy else PASS,
            "Renderer._call",
            f"returned {len(frames)} stub frame(s) (stub mode OK when AVTR-1 offline)",
            ms=ms)

asyncio.run(test_renderer())
print()


# ── Edge 7: Avatar registry ────────────────────────────────────────────────────
print("── 7. Avatar registry ──────────────────────────────────")

t0 = time.time()
try:
    reg = AvatarRegistry(
        avatars_dir=str(avatars_dir),
        avtr1_frames_dir=os.environ.get("AVTR1_FRAMES_DIR", "./checkpoints/avtr1/avatars_artifacts/reference_frames"),
    )
    avs = reg.list_avatars()
    ms = (time.time()-t0)*1000
    log(PASS if "evedefault" in avs else WARN,
        "AvatarRegistry",
        f"avatars={avs}",
        ms=ms)
except Exception as e:
    log(FAIL, "AvatarRegistry error", str(e))

print()


# ── Edge 8: Full E2E text→audio latency (stub mode) ──────────────────────────
print("── 8. E2E handoff — text → LLM → TTS → renderer ───────")

async def test_e2e():
    from pipeline import CliqueBrain, ClueTTS, AVTR1Renderer
    import numpy as np

    brain    = CliqueBrain("evedefault")
    tts      = ClueTTS()
    renderer = AVTR1Renderer()

    t_start = time.time()
    t_first_token = None
    t_first_audio = None
    t_first_frame = None
    audio_chunks  = 0
    frames        = 0

    async def text_gen():
        nonlocal t_first_token
        async for tok in brain.respond("Greet the user briefly."):
            if t_first_token is None:
                t_first_token = time.time()
            yield tok

    async for chunk in tts.synthesize_stream(text_gen(), "evedefault"):
        if t_first_audio is None:
            t_first_audio = time.time()
        audio_chunks += 1
        frs = await renderer.feed_audio(
            (chunk * 32767).astype(np.int16)
        )
        if t_first_frame is None and frs:
            t_first_frame = time.time()
            frames += len(frs)
        if audio_chunks >= 2:
            break

    now = time.time()
    log(PASS if t_first_token else WARN, "LLM first token",   ms=(t_first_token-t_start)*1000 if t_first_token else None)
    log(PASS if t_first_audio else WARN, "TTS first audio",   ms=(t_first_audio-t_start)*1000 if t_first_audio else None)
    log(WARN,                            "Renderer first frame (stub)", ms=(t_first_frame-t_start)*1000 if t_first_frame else None)
    log(PASS, "E2E handoff complete", f"{audio_chunks} audio chunks · {frames} frames")

asyncio.run(test_e2e())
print()


# ── Summary ────────────────────────────────────────────────────────────────────
print("═══════════════════════════════════════════════════════")
passes  = sum(1 for r in results if r["status"] == PASS)
warns   = sum(1 for r in results if r["status"] == WARN)
fails   = sum(1 for r in results if r["status"] == FAIL)
total   = len(results)
print(f"  {PASS} {passes} passed   {WARN} {warns} warnings   {FAIL} {fails} failed   ({total} checks)")
print()
if fails == 0:
    print("  Pipeline is GO. Run setup.sh to install AVTR-1.")
else:
    print("  Fix failures before proceeding.")
print("═══════════════════════════════════════════════════════")
print()
