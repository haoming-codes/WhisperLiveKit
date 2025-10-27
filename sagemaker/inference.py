"""SageMaker inference entry point for WhisperLiveKit."""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
from typing import Any, AsyncIterable, Dict

from whisperlivekit import AudioProcessor, TranscriptionEngine

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)


_ENGINE: TranscriptionEngine | None = None


def _strtobool(value: str) -> bool:
    return value.lower() in {"1", "true", "t", "yes", "y", "on"}


def _build_engine_kwargs() -> Dict[str, Any]:
    env_to_kw = {
        "WHISPER_MODEL_SIZE": "model_size",
        "WHISPER_BACKEND": "backend",
        "WHISPER_LANGUAGE": "lan",
        "WHISPER_TARGET_LANGUAGE": "target_language",
        "WHISPER_MIN_CHUNK_SIZE": "min_chunk_size",
        "WHISPER_DIAARIZATION": "diarization",
        "WHISPER_PUNCTUATION_SPLIT": "punctuation_split",
        "WHISPER_VAC": "vac",
        "WHISPER_VAD": "vad",
    }

    kwargs: Dict[str, Any] = {}
    for env_key, kw_key in env_to_kw.items():
        if env_key not in os.environ:
            continue
        raw = os.environ[env_key]
        if kw_key in {"diarization", "punctuation_split", "vac", "vad"}:
            kwargs[kw_key] = _strtobool(raw)
        elif kw_key == "min_chunk_size":
            kwargs[kw_key] = float(raw)
        else:
            kwargs[kw_key] = raw

    return kwargs


def model_fn(_model_dir: str) -> TranscriptionEngine:
    """Initialise (or reuse) the WhisperLiveKit engine."""
    global _ENGINE
    if _ENGINE is None:
        kwargs = _build_engine_kwargs()
        LOGGER.info("Loading TranscriptionEngine with options: %s", kwargs)
        _ENGINE = TranscriptionEngine(**kwargs)
    return _ENGINE


def _decode_json_audio(payload: Dict[str, Any]) -> bytes:
    if "audio" not in payload:
        raise ValueError("JSON payload must contain an 'audio' key")
    audio_field = payload["audio"]
    if isinstance(audio_field, str):
        return base64.b64decode(audio_field)
    raise ValueError("JSON 'audio' value must be a base64 string")


def input_fn(request_body: bytes, request_content_type: str | None) -> bytes:
    """Convert the incoming request into raw audio bytes."""
    content_type = (request_content_type or "").lower()
    if content_type in {"audio/webm", "application/octet-stream"}:
        if isinstance(request_body, (bytes, bytearray)):
            return bytes(request_body)
        return request_body.encode("utf-8")

    if content_type == "application/json" or not content_type:
        payload = json.loads(request_body)
        return _decode_json_audio(payload)

    raise ValueError(f"Unsupported content type: {request_content_type}")


async def _stream_to_processor(processor: AudioProcessor, audio_bytes: bytes, chunk_size: int = 131072) -> None:
    for index in range(0, len(audio_bytes), chunk_size):
        await processor.process_audio(audio_bytes[index : index + chunk_size])
    await processor.process_audio(b"")


async def _collect_transcription(results_generator: AsyncIterable) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "status": "no_audio_detected",
        "lines": [],
        "buffer_transcription": "",
        "buffer_diarization": "",
        "remaining_time_transcription": 0.0,
        "remaining_time_diarization": 0.0,
    }

    async for front_data in results_generator:
        if front_data.error:
            raise RuntimeError(front_data.error)
        result = {
            "status": front_data.status,
            "lines": [line.to_dict() for line in front_data.lines],
            "buffer_transcription": front_data.buffer_transcription,
            "buffer_diarization": front_data.buffer_diarization,
            "remaining_time_transcription": front_data.remaining_time_transcription,
            "remaining_time_diarization": front_data.remaining_time_diarization,
        }

    return result


async def _transcribe(engine: TranscriptionEngine, audio_bytes: bytes) -> Dict[str, Any]:
    processor = AudioProcessor(transcription_engine=engine)
    try:
        results_generator = await processor.create_tasks()
        stream_task = asyncio.create_task(_stream_to_processor(processor, audio_bytes))
        result = await _collect_transcription(results_generator)
        await stream_task
        return result
    finally:
        await processor.cleanup()


def predict_fn(input_data: bytes, model: TranscriptionEngine) -> Dict[str, Any]:
    """Run transcription inside a fresh asyncio event loop."""
    return asyncio.run(_transcribe(model, input_data))


def output_fn(prediction: Dict[str, Any], accept: str | None) -> bytes:
    accept = (accept or "application/json").lower()
    if accept != "application/json":
        raise ValueError(f"Unsupported accept type: {accept}")
    return json.dumps(prediction).encode("utf-8")
