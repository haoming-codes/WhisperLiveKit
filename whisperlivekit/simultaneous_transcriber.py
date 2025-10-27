"""Command-line script for streaming transcription without WebSockets.

This module streams a local audio file to the WhisperLiveKit processing
pipeline in small chunks, mimicking a real-time client. It records every
update produced by the ASR model, saves the final transcript, and computes
latency metrics for the run.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from time import monotonic
from typing import Any, Dict, AsyncIterable, List

import numpy as np
import torchaudio

from .audio_processor import AudioProcessor
from .parse_args import build_parser


@dataclass
class StreamingResults:
    """Container capturing information gathered during transcription."""

    transcript_lines: List[Dict[str, Any]]
    final_buffer: str
    event_latencies: List[float]
    chunk_send_times: List[float]
    start_time: float
    time_to_first_token: float | None
    processing_end_time: float
    audio_duration: float


def _prepare_waveform(path: Path, target_sample_rate: int = 16000) -> np.ndarray:
    """Load the audio file and return a mono waveform at the target sample rate."""

    waveform, sample_rate = torchaudio.load(path)
    if waveform.dim() > 1 and waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if sample_rate != target_sample_rate:
        resampler = torchaudio.transforms.Resample(sample_rate, target_sample_rate)
        waveform = resampler(waveform)

    waveform = waveform.squeeze(0).clamp(-1.0, 1.0)
    return waveform.numpy()


async def _stream_audio_chunks(
    processor: AudioProcessor,
    pcm_audio: np.ndarray,
    chunk_duration: float,
    chunk_interval: float,
    chunk_send_times: List[float],
) -> None:
    """Feed PCM audio chunks to the processor at a fixed cadence."""

    assert chunk_duration > 0, "Chunk duration must be positive."
    assert chunk_interval > 0, "Chunk interval must be positive."

    sample_rate = 16000
    samples_per_chunk = max(1, int(round(chunk_duration * sample_rate)))
    total_samples = pcm_audio.shape[0]

    start_time = monotonic()
    chunk_index = 0
    cursor = 0

    while cursor < total_samples:
        next_cursor = min(cursor + samples_per_chunk, total_samples)
        chunk = pcm_audio[cursor:next_cursor]
        chunk_bytes = chunk.tobytes()

        send_time = monotonic()
        chunk_send_times.append(send_time)
        await processor.process_audio(chunk_bytes)

        cursor = next_cursor
        chunk_index += 1
        target_next_time = start_time + chunk_index * chunk_interval
        delay = target_next_time - monotonic()
        if delay > 0:
            await asyncio.sleep(delay)

    await processor.process_audio(b"")


async def _consume_results(
    results_iter: AsyncIterable,
    log_path: Path,
    chunk_send_times: List[float],
    start_time: float,
) -> StreamingResults:
    """Consume transcription updates and persist them to disk."""

    transcript_lines: List[Dict[str, Any]] = []
    final_buffer = ""
    latencies: List[float] = []
    time_to_first_token: float | None = None
    last_latency_chunk_idx = -1

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        async for front_data in results_iter:
            event_time = monotonic()
            relative_time = event_time - start_time
            payload = front_data.to_dict()

            record = {
                "relative_timestamp_s": relative_time,
                "absolute_timestamp_s": event_time,
                "payload": payload,
            }
            json.dump(record, log_file, ensure_ascii=False)
            log_file.write("\n")
            log_file.flush()

            if payload.get("lines"):
                transcript_lines = payload["lines"]
            final_buffer = payload.get("buffer_transcription", final_buffer)

            if time_to_first_token is None and (
                payload.get("lines") or payload.get("buffer_transcription")
            ):
                time_to_first_token = relative_time

            if chunk_send_times:
                target_index = min(len(chunk_send_times) - 1, last_latency_chunk_idx + 1)
                latencies.append(event_time - chunk_send_times[target_index])
                last_latency_chunk_idx = target_index

    return StreamingResults(
        transcript_lines=transcript_lines,
        final_buffer=final_buffer,
        event_latencies=latencies,
        chunk_send_times=chunk_send_times,
        start_time=start_time,
        time_to_first_token=time_to_first_token,
        processing_end_time=monotonic(),
        audio_duration=0.0,
    )


def _write_transcript(path: Path, lines: List[Dict[str, Any]], buffer_text: str) -> None:
    """Persist the final transcript to disk in a readable format."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        if lines:
            for entry in lines:
                speaker = entry.get("speaker", "?")
                text = entry.get("text", "")
                start = entry.get("start", "00:00:00")
                end = entry.get("end", "00:00:00")
                detected_language = entry.get("detected_language")
                translation = entry.get("translation")

                line = f"[{start} - {end}] Speaker {speaker}: {text}"
                if detected_language:
                    line += f" (language: {detected_language})"
                handle.write(line + "\n")
                if translation:
                    handle.write(f"    Translation: {translation}\n")

        if buffer_text:
            handle.write("\n[Unfinalized buffer]\n")
            handle.write(buffer_text.strip() + "\n")


def _write_metrics(path: Path, metrics: Dict[str, Any]) -> None:
    """Write human-readable metrics and explanations to disk."""

    path.parent.mkdir(parents=True, exist_ok=True)
    explanations = {
        "total_processing_time_s": "Wall-clock time from the start of streaming until the last transcription event was observed.",
        "time_to_first_token_s": "Elapsed time between the first audio chunk being sent and the first transcription update being emitted.",
        "average_event_latency_s": "Average delay between sending a chunk and receiving the next transcription update.",
        "median_event_latency_s": "Median of the per-update delays, showing a typical latency unaffected by outliers.",
        "max_event_latency_s": "Maximum observed delay between chunk submission and transcription update, representing the worst case during this run.",
        "chunk_count": "Total number of audio chunks that were streamed to the engine.",
        "audio_duration_s": "Duration of the provided audio after resampling to 16 kHz mono.",
    }

    with path.open("w", encoding="utf-8") as handle:
        for key, value in metrics.items():
            if value is None:
                value_text = "N/A"
            elif isinstance(value, float):
                value_text = f"{value:.3f}"
            else:
                value_text = str(value)

            explanation = explanations.get(key, "")
            handle.write(f"{key}: {value_text}\n")
            if explanation:
                handle.write(f"    {explanation}\n")


def _prepare_argument_parser() -> argparse.ArgumentParser:
    """Combine the core WhisperLiveKit arguments with script-specific ones."""

    base_parser = build_parser(add_help=False)
    parser = argparse.ArgumentParser(
        parents=[base_parser],
        add_help=True,
        description="Stream an audio file through WhisperLiveKit without WebSockets.",
    )

    parser.add_argument(
        "input_file",
        type=Path,
        help="Path to the local audio file to transcribe.",
    )
    parser.add_argument(
        "--transcript-output",
        type=Path,
        default=Path("transcript.txt"),
        help="Destination file for the final transcript.",
    )
    parser.add_argument(
        "--json-log-output",
        type=Path,
        default=Path("transcription_events.jsonl"),
        help="File to receive JSONL logs of every transcription event.",
    )
    parser.add_argument(
        "--metrics-output",
        type=Path,
        default=Path("transcription_metrics.txt"),
        help="File to receive summary latency metrics.",
    )
    parser.add_argument(
        "--chunk-duration",
        type=float,
        default=0.5,
        help="Length of each streamed chunk in seconds.",
    )
    parser.add_argument(
        "--chunk-interval",
        type=float,
        default=0.5,
        help="Target interval between streamed chunks in seconds.",
    )

    return parser


async def _run_transcription(args: argparse.Namespace) -> StreamingResults:
    """Execute the asynchronous transcription pipeline."""

    if not args.input_file.exists():
        raise FileNotFoundError(f"Input file not found: {args.input_file}")

    waveform = _prepare_waveform(args.input_file)
    pcm_audio = (waveform * 32767.0).astype(np.int16)
    audio_duration = waveform.shape[0] / 16000.0

    chunk_send_times: List[float] = []
    start_time = monotonic()

    config = {
        key: value
        for key, value in vars(args).items()
        if key
        not in {
            "input_file",
            "transcript_output",
            "json_log_output",
            "metrics_output",
            "chunk_duration",
            "chunk_interval",
        }
    }

    # Always stream PCM data in this script unless the user explicitly opted in.
    if not config.get("pcm_input", False):
        config["pcm_input"] = True

    audio_processor = AudioProcessor(**config)
    results_generator = await audio_processor.create_tasks()

    stream_task = asyncio.create_task(
        _stream_audio_chunks(
            audio_processor,
            pcm_audio,
            args.chunk_duration,
            args.chunk_interval,
            chunk_send_times,
        )
    )

    consume_task = asyncio.create_task(
        _consume_results(
            results_generator,
            args.json_log_output,
            chunk_send_times,
            start_time,
        )
    )

    try:
        _, results = await asyncio.gather(stream_task, consume_task)
    except Exception:
        if not consume_task.done():
            consume_task.cancel()
            with suppress(asyncio.CancelledError):
                await consume_task
        raise
    finally:
        await audio_processor.cleanup()

    # Attach run-level metadata that is computed outside of the consumer loop.
    results.audio_duration = audio_duration
    return results


def main(argv: List[str] | None = None) -> None:
    parser = _prepare_argument_parser()
    args = parser.parse_args(argv)

    # Align option names with the rest of the codebase.
    if hasattr(args, "no_transcription"):
        args.transcription = not args.no_transcription
        delattr(args, "no_transcription")
    if hasattr(args, "no_vad"):
        args.vad = not args.no_vad
        delattr(args, "no_vad")

    results = asyncio.run(_run_transcription(args))

    _write_transcript(args.transcript_output, results.transcript_lines, results.final_buffer)

    latencies = results.event_latencies
    metrics = {
        "total_processing_time_s": results.processing_end_time - results.start_time,
        "time_to_first_token_s": results.time_to_first_token,
        "average_event_latency_s": statistics.fmean(latencies) if latencies else None,
        "median_event_latency_s": statistics.median(latencies) if latencies else None,
        "max_event_latency_s": max(latencies) if latencies else None,
        "chunk_count": len(results.chunk_send_times),
        "audio_duration_s": results.audio_duration,
    }

    _write_metrics(args.metrics_output, metrics)


if __name__ == "__main__":
    main()
