#!/usr/bin/env python3
"""Latency benchmark utility for WhisperLiveKit modules.

This script measures latency for the core building blocks used in the
simultaneous transcription pipeline as well as the end-to-end pipeline itself.
It avoids spinning up the FastAPI server and instead simulates a streaming
client that feeds PCM audio chunks from local files.

The benchmark focuses on:

* Voice Activity Controller (Silero VAD via :class:`FixedVADIterator`).
* Streaming ASR backend (``SimulStreamingOnlineProcessor`` by default).
* Complete ``AudioProcessor`` pipeline (queues, background workers, formatter).

Example usage::

    python scripts/latency_benchmark.py examples/sample.wav \
        --model-size tiny --chunk-seconds 0.5

The script prints per-audio and aggregate statistics including total latency,
average latency per chunk for the individual modules, and the latency until the
first transcription output is observed.
"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass, field
from pathlib import Path
import sys
from time import perf_counter
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import librosa
import numpy as np
import soundfile as sf

CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from whisperlivekit.audio_processor import AudioProcessor
from whisperlivekit.core import TranscriptionEngine, online_factory
from whisperlivekit.silero_vad_iterator import FixedVADIterator


TARGET_SAMPLE_RATE = 16000


@dataclass
class Timer:
    """Simple helper to accumulate latency measurements."""

    name: str
    durations: List[float] = field(default_factory=list)

    def add(self, value: float) -> None:
        self.durations.append(value)

    @property
    def total(self) -> float:
        return float(sum(self.durations))

    @property
    def count(self) -> int:
        return len(self.durations)

    @property
    def average(self) -> float:
        if not self.durations:
            return 0.0
        return self.total / self.count

    @property
    def maximum(self) -> float:
        if not self.durations:
            return 0.0
        return max(self.durations)

    def summary(self) -> Dict[str, float]:
        return {
            "count": self.count,
            "total_sec": self.total,
            "avg_sec": self.average,
            "max_sec": self.maximum,
        }


@dataclass
class ModuleLatency:
    """Container for module timers."""

    vac: Timer = field(default_factory=lambda: Timer("Voice Activity Controller"))
    asr: Timer = field(default_factory=lambda: Timer("Streaming ASR step"))


@dataclass
class PipelineResult:
    total_latency: float
    first_token_latency: Optional[float]
    emitted_updates: int


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure latency of core WhisperLiveKit modules without starting the "
            "web server."
        )
    )
    parser.add_argument(
        "audio_files",
        nargs="+",
        type=Path,
        help="Path(s) to local audio files used for benchmarking.",
    )
    parser.add_argument(
        "--chunk-seconds",
        type=float,
        default=0.5,
        help="Duration of each simulated streaming chunk in seconds.",
    )
    parser.add_argument(
        "--model-size",
        type=str,
        default=None,
        help="Optional Whisper model size (overrides engine default).",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default=None,
        help="Optional transcription backend (default: simulstreaming).",
    )
    parser.add_argument(
        "--no-vac",
        action="store_true",
        help="Disable voice activity controller measurements.",
    )
    parser.add_argument(
        "--no-vad",
        action="store_true",
        help="Disable VAD inside the transcription engine.",
    )
    parser.add_argument(
        "--language",
        type=str,
        default=None,
        help="Override source language passed to the engine (default: auto).",
    )
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        choices=["transcribe", "translate"],
        help="Override transcription task (default: transcribe).",
    )
    parser.add_argument(
        "--real-time",
        action="store_true",
        help="Sleep between chunks to mimic real-time audio streaming.",
    )
    return parser.parse_args(argv)


def load_audio(path: Path, target_sr: int = TARGET_SAMPLE_RATE) -> np.ndarray:
    """Load an audio file and resample/convert it to mono float32."""

    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    audio, sr = sf.read(path, always_2d=False, dtype="float32")
    if audio.ndim > 1:
        audio = np.mean(audio, axis=-1)
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    return audio.astype(np.float32)


def float_to_pcm16(audio: np.ndarray) -> bytes:
    """Convert float32 audio (-1.0, 1.0) to PCM16 little-endian bytes."""

    clipped = np.clip(audio, -1.0, 1.0)
    int_audio = (clipped * np.iinfo(np.int16).max).astype("<i2")
    return int_audio.tobytes()


def iter_audio_chunks(audio: np.ndarray, chunk_size: int) -> Iterable[np.ndarray]:
    """Yield successive chunks from the audio array."""

    total_samples = len(audio)
    for start in range(0, total_samples, chunk_size):
        end = min(start + chunk_size, total_samples)
        yield audio[start:end]


def measure_module_latencies(
    audio: np.ndarray,
    chunk_seconds: float,
    engine: TranscriptionEngine,
    timers: Optional[ModuleLatency] = None,
) -> ModuleLatency:
    """Measure latencies for VAD and streaming ASR modules."""

    timers = timers or ModuleLatency()

    chunk_size = max(1, int(chunk_seconds * TARGET_SAMPLE_RATE))
    vac_iterator: Optional[FixedVADIterator] = None
    if engine.vac_model is not None:
        vac_iterator = FixedVADIterator(engine.vac_model)
    online_processor = online_factory(engine.args, engine.asr) if engine.asr else None

    stream_time = 0.0
    for chunk in iter_audio_chunks(audio, chunk_size):
        float_chunk = chunk.astype(np.float32)
        stream_time += len(float_chunk) / TARGET_SAMPLE_RATE

        if vac_iterator is not None:
            start = perf_counter()
            _ = vac_iterator(float_chunk)
            timers.vac.add(perf_counter() - start)

        if online_processor is not None:
            start = perf_counter()
            if hasattr(online_processor, "insert_audio"):
                online_processor.insert_audio(float_chunk, stream_time)
            elif hasattr(online_processor, "insert_audio_chunk"):
                online_processor.insert_audio_chunk(float_chunk, stream_time)
            else:
                raise AttributeError("Unsupported online processor API")
            try:
                online_processor.process_iter(is_last=False)
            except TypeError:
                online_processor.process_iter()
            timers.asr.add(perf_counter() - start)

    if online_processor is not None:
        start = perf_counter()
        try:
            online_processor.process_iter(is_last=True)
        except TypeError:
            online_processor.process_iter()
        timers.asr.add(perf_counter() - start)

    return timers


async def run_pipeline(
    audio: np.ndarray,
    chunk_seconds: float,
    engine: TranscriptionEngine,
    real_time: bool,
) -> PipelineResult:
    """Run the full AudioProcessor pipeline and measure latency."""

    chunk_size = max(1, int(chunk_seconds * TARGET_SAMPLE_RATE))
    pcm_chunks = [float_to_pcm16(chunk) for chunk in iter_audio_chunks(audio, chunk_size)]

    processor = AudioProcessor(transcription_engine=engine)
    formatter_stream = await processor.create_tasks()

    outputs: List[Tuple[float, object]] = []

    async def consume_formatter():
        async for message in formatter_stream:
            outputs.append((perf_counter(), message))

    consumer_task = asyncio.create_task(consume_formatter())

    start = perf_counter()
    for chunk_bytes in pcm_chunks:
        await processor.process_audio(chunk_bytes)
        if real_time:
            await asyncio.sleep(chunk_seconds)
    await processor.process_audio(b"")

    await consumer_task
    await processor.cleanup()

    total_latency = perf_counter() - start
    first_token_latency: Optional[float] = None
    for timestamp, message in outputs:
        lines = getattr(message, "lines", None)
        if lines:
            first_token_latency = timestamp - start
            break
    return PipelineResult(
        total_latency=total_latency,
        first_token_latency=first_token_latency,
        emitted_updates=len(outputs),
    )


def format_timer_summary(timer: Timer) -> str:
    if timer.count == 0:
        return "(no samples)"
    return (
        f"count={timer.count} total={timer.total:.3f}s "
        f"avg={timer.average:.4f}s max={timer.maximum:.4f}s"
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)

    engine_kwargs = {"pcm_input": True}
    if args.model_size:
        engine_kwargs["model_size"] = args.model_size
    if args.backend:
        engine_kwargs["backend"] = args.backend
    if args.no_vac:
        engine_kwargs["no_vac"] = True
    if args.no_vad:
        engine_kwargs["no_vad"] = True
    if args.language:
        engine_kwargs["lan"] = args.language
    if args.task:
        engine_kwargs["task"] = args.task

    engine = TranscriptionEngine(**engine_kwargs)
    aggregate_timers = ModuleLatency()
    aggregate_pipeline_latency = []
    aggregate_first_token = []

    for audio_path in args.audio_files:
        audio = load_audio(audio_path)
        duration = len(audio) / TARGET_SAMPLE_RATE
        print(f"\n=== Benchmarking {audio_path} ({duration:.2f}s) ===")

        file_timers = measure_module_latencies(audio, args.chunk_seconds, engine)
        pipeline_result = asyncio.run(
            run_pipeline(audio, args.chunk_seconds, engine, args.real_time)
        )

        aggregate_pipeline_latency.append(pipeline_result.total_latency)
        if pipeline_result.first_token_latency is not None:
            aggregate_first_token.append(pipeline_result.first_token_latency)
        aggregate_timers.vac.durations.extend(file_timers.vac.durations)
        aggregate_timers.asr.durations.extend(file_timers.asr.durations)

        print("Module latency summary:")
        print(
            f"  Voice Activity Controller: {format_timer_summary(file_timers.vac)}"
            if engine.vac_model is not None
            else "  Voice Activity Controller: disabled"
        )
        if engine.asr is not None:
            print(f"  Streaming ASR: {format_timer_summary(file_timers.asr)}")
        else:
            print("  Streaming ASR: disabled")

        print("Pipeline summary:")
        print(f"  Total pipeline latency: {pipeline_result.total_latency:.3f}s")
        if pipeline_result.first_token_latency is not None:
            print(
                "  Latency to first transcription: "
                f"{pipeline_result.first_token_latency:.3f}s"
            )
        else:
            print("  Latency to first transcription: unavailable")
        print(f"  Formatter updates emitted: {pipeline_result.emitted_updates}")

    if aggregate_pipeline_latency:
        avg_total = sum(aggregate_pipeline_latency) / len(aggregate_pipeline_latency)
        print("\n=== Aggregate results ===")
        print(f"Average total pipeline latency: {avg_total:.3f}s")
        if aggregate_first_token:
            avg_first = sum(aggregate_first_token) / len(aggregate_first_token)
            print(f"Average latency to first transcription: {avg_first:.3f}s")
        print("Module totals across all files:")
        print(
            f"  Voice Activity Controller: {format_timer_summary(aggregate_timers.vac)}"
            if engine.vac_model is not None
            else "  Voice Activity Controller: disabled"
        )
        if engine.asr is not None:
            print(f"  Streaming ASR: {format_timer_summary(aggregate_timers.asr)}")
        else:
            print("  Streaming ASR: disabled")


if __name__ == "__main__":
    main()
