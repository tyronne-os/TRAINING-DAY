"""
NVIDIA ACE Audio2Face-3D gRPC client.

Connects to grpc.nvcf.nvidia.com:443 using SSL + NGC API key auth.
Streams int16 PCM audio via A2FControllerService.ProcessAudioStream and
returns per-frame blendshape weight dicts (52 ARKit coefficients, ~30 fps).

Auth references:
  https://github.com/NVIDIA/Audio2Face-3D-Samples
  NVCF endpoint: grpc.nvcf.nvidia.com:443
  Metadata headers: ("function-id", <uuid>), ("authorization", "Bearer <key>")

Function IDs (set via NVIDIA_A2F_FUNCTION_ID env var or pass directly):
  Claire (no-tongue): 462f7853-60e8-474a-9728-7b598e58472c
  Mark:               945ed566-a023-4677-9a49-61ede107fd5a
  James:              a2cc5cac-147d-4e46-b79d-4cea616e21b9
"""
from __future__ import annotations

import asyncio
import logging
import os
import struct
from typing import AsyncIterator

import grpc
import grpc.aio

from nvidia_ace.controller.v1_pb2 import (
    AudioStream,
    AudioStreamHeader,
    Event,
    EventType,
)
from nvidia_ace.a2f.v1_pb2 import (
    AudioWithEmotion,
    FaceParameters,
    BlendShapeParameters,
    EmotionPostProcessingParameters,
    EmotionParameters,
)
from nvidia_ace.audio.v1_pb2 import AudioHeader
from nvidia_ace.services.a2f_controller.v1_pb2_grpc import A2FControllerServiceStub

LOG = logging.getLogger("ace_client")

NVCF_ENDPOINT = "grpc.nvcf.nvidia.com:443"

# Default Claire no-tongue model; override with NVIDIA_A2F_FUNCTION_ID env var.
_DEFAULT_FUNCTION_ID = "462f7853-60e8-474a-9728-7b598e58472c"

# 52 standard ARKit blendshape names in the order ACE returns them.
BLENDSHAPE_NAMES: list[str] = [
    "EyeBlinkLeft", "EyeLookDownLeft", "EyeLookInLeft", "EyeLookOutLeft",
    "EyeLookUpLeft", "EyeSquintLeft", "EyeWideLeft",
    "EyeBlinkRight", "EyeLookDownRight", "EyeLookInRight", "EyeLookOutRight",
    "EyeLookUpRight", "EyeSquintRight", "EyeWideRight",
    "JawForward", "JawLeft", "JawRight", "JawOpen",
    "MouthClose", "MouthFunnel", "MouthPucker", "MouthLeft", "MouthRight",
    "MouthSmileLeft", "MouthSmileRight", "MouthFrownLeft", "MouthFrownRight",
    "MouthDimpleLeft", "MouthDimpleRight", "MouthStretchLeft", "MouthStretchRight",
    "MouthRollLower", "MouthRollUpper", "MouthShrugLower", "MouthShrugUpper",
    "MouthPressLeft", "MouthPressRight", "MouthLowerDownLeft", "MouthLowerDownRight",
    "MouthUpperUpLeft", "MouthUpperUpRight",
    "BrowDownLeft", "BrowDownRight", "BrowInnerUp", "BrowOuterUpLeft", "BrowOuterUpRight",
    "CheekPuff", "CheekSquintLeft", "CheekSquintRight",
    "NoseSneerLeft", "NoseSneerRight",
    "TongueOut",
]


def _make_channel(api_key: str, function_id: str) -> grpc.aio.Channel:
    """Create an authenticated SSL gRPC channel to NVCF."""
    credentials = grpc.ssl_channel_credentials()
    call_credentials = grpc.access_token_call_credentials(api_key)
    composite = grpc.composite_channel_credentials(credentials, call_credentials)
    # function-id goes as per-call metadata; put it on the channel default
    # metadata so every call carries it automatically.
    options = [
        ("grpc.max_send_message_length", 16 * 1024 * 1024),
        ("grpc.max_receive_message_length", 16 * 1024 * 1024),
    ]
    return grpc.aio.secure_channel(
        NVCF_ENDPOINT,
        composite,
        options=options,
        interceptors=None,
    ), function_id


class ACEAudio2FaceClient:
    """
    Async client for NVIDIA ACE Audio2Face-3D.

    Usage:
        client = ACEAudio2FaceClient(api_key=..., function_id=...)
        async with client:
            frames = await client.audio_to_blendshapes(pcm_int16, sample_rate=16000)
    """

    def __init__(
        self,
        api_key: str | None = None,
        function_id: str | None = None,
    ) -> None:
        self._api_key: str = api_key or os.environ.get("NVIDIA_NGC_API_KEY", "")
        self._function_id: str = (
            function_id
            or os.environ.get("NVIDIA_A2F_FUNCTION_ID", _DEFAULT_FUNCTION_ID)
        )
        if not self._api_key:
            raise ValueError(
                "NGC API key required: pass api_key= or set NVIDIA_NGC_API_KEY env var"
            )
        self._channel: grpc.aio.Channel | None = None

    async def __aenter__(self) -> "ACEAudio2FaceClient":
        credentials = grpc.ssl_channel_credentials()
        options = [
            ("grpc.max_send_message_length", 16 * 1024 * 1024),
            ("grpc.max_receive_message_length", 16 * 1024 * 1024),
        ]
        self._channel = grpc.aio.secure_channel(
            NVCF_ENDPOINT, credentials, options=options
        )
        return self

    async def __aexit__(self, *_) -> None:
        if self._channel is not None:
            await self._channel.close()
            self._channel = None

    def _rpc_metadata(self) -> list[tuple[str, str]]:
        return [
            ("function-id", self._function_id),
            ("authorization", f"Bearer {self._api_key}"),
        ]

    @staticmethod
    def _build_stream_header(sample_rate: int) -> AudioStream:
        audio_header = AudioHeader(
            audio_format=AudioHeader.AUDIO_FORMAT_PCM,
            channel_count=1,
            samples_per_second=sample_rate,
            bits_per_sample=16,
        )
        face_params = FaceParameters(
            lowerFace_smoothing=0.6,
            upperFace_smoothing=0.4,
            lowerFace_strength=1.0,
            upperFace_strength=1.0,
            face_mask_level=0.6,
            face_mask_softness=0.008,
            skin_strength=1.0,
        )
        blendshape_params = BlendShapeParameters(
            enable_clamping_bs_weight=True,
        )
        emotion_pp = EmotionPostProcessingParameters(
            emotion_contrast=1.0,
            live_blend_coef=0.7,
            enable_preferred_emotion=False,
            max_emotions=3,
        )
        header = AudioStreamHeader(
            audio_header=audio_header,
            face_params=face_params,
            blendshape_params=blendshape_params,
            emotion_post_processing_params=emotion_pp,
        )
        return AudioStream(audio_stream_header=header)

    @staticmethod
    def _build_audio_message(pcm_int16_bytes: bytes) -> AudioStream:
        return AudioStream(
            audio_with_emotion=AudioWithEmotion(audio_buffer=pcm_int16_bytes)
        )

    @staticmethod
    def _build_end_of_audio() -> AudioStream:
        return AudioStream(
            event=Event(event_type=EventType.END_OF_A2F_AUDIO_PROCESSING)
        )

    async def audio_to_blendshapes(
        self,
        pcm_int16: bytes,
        sample_rate: int = 16000,
        chunk_bytes: int = 6400,  # 3200 int16 samples = 200ms
    ) -> list[dict[str, float]]:
        """
        Send PCM int16 audio to ACE Audio2Face-3D and return blendshape dicts.

        Each returned dict maps the 52 ARKit blendshape names to float weights
        [0.0, 1.0].  The list length equals the number of animation frames
        returned by the service (nominally 30 fps × audio_duration seconds).

        Raises grpc.RpcError on service errors.  Returns an empty list if the
        service returns no animation data (e.g., silence).
        """
        if self._channel is None:
            raise RuntimeError("Client not started; use 'async with' context manager")

        stub = A2FControllerServiceStub(self._channel)
        frames: list[dict[str, float]] = []
        blendshape_names: list[str] = []

        stream = stub.ProcessAudioStream(metadata=self._rpc_metadata())

        async def _write() -> None:
            await stream.write(self._build_stream_header(sample_rate))
            for offset in range(0, len(pcm_int16), chunk_bytes):
                chunk = pcm_int16[offset : offset + chunk_bytes]
                await stream.write(self._build_audio_message(chunk))
            await stream.write(self._build_end_of_audio())
            await stream.done_writing()

        async def _read() -> None:
            async for response in stream:
                which = response.WhichOneof("stream_part")
                if which == "animation_data_stream_header":
                    hdr = response.animation_data_stream_header
                    if hdr.skel_animation_header.blend_shape_names:
                        blendshape_names.clear()
                        blendshape_names.extend(
                            hdr.skel_animation_header.blend_shape_names
                        )
                elif which == "animation_data":
                    anim = response.animation_data
                    for bs_frame in anim.skel_animation.blend_shape_weights:
                        names = blendshape_names if blendshape_names else BLENDSHAPE_NAMES
                        weight_dict: dict[str, float] = {
                            name: float(val)
                            for name, val in zip(names, bs_frame.values)
                        }
                        frames.append(weight_dict)
                elif which == "status":
                    status = response.status
                    if status.code != 0:
                        LOG.warning(
                            "ACE A2F status code %d: %s", status.code, status.message
                        )

        try:
            await asyncio.gather(_write(), _read())
        except grpc.aio.AioRpcError as exc:
            LOG.error("ACE gRPC error %s: %s", exc.code(), exc.details())
            raise

        return frames

    async def health_check(self) -> bool:
        """Return True if the NVCF endpoint is reachable and the key appears valid."""
        try:
            async with self:
                await self.audio_to_blendshapes(
                    _silence_bytes(samples=1600, sample_rate=16000),
                    sample_rate=16000,
                )
            return True
        except grpc.aio.AioRpcError:
            return False
        except Exception:
            return False


def _silence_bytes(samples: int = 3200, sample_rate: int = 16000) -> bytes:
    """Return `samples` int16 zero-samples as bytes (100ms of silence at 16kHz)."""
    return struct.pack(f"<{samples}h", *([0] * samples))
