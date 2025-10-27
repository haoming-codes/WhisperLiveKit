#!/usr/bin/env python3
"""Simulate real-time Simul-Whisper transcription over a local file."""

from __future__ import annotations

import argparse
import math
import statistics
import time
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np

from whisperlivekit.parse_args import (
    create_argument_parser,
    finalize_parsed_args,
)
from whisperlivekit.simul_whisper import (
    SimulStreamingASR,
    SimulStreamingOnlineProcessor,
)
from whisperlivekit.simul_whisper.whisper.audio import SAMPLE_RATE, load_audio
from whisperlivekit.timed_objects import ASRToken, Transcript


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _iter_chunks(
    audio: np.ndarray, chunk_samples: int, step_samples: int
) -> Iterable[Tuple[int, int, np.ndarray]]:
    total = len(audio)
    start = 0
    if total == 0:
        return
    while start < total:
        end = min(start + chunk_samples, total)
        chunk = audio[start:end]
        yield start, end, chunk
        start += step_samples


def _percentile(sorted_values: Sequence[float], q: float) -> float:
    if not sorted_values:
        return math.nan
    if len(sorted_values) == 1:
        return sorted_values[0]
    k = (len(sorted_values) - 1) * (q / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_values[int(k)]
    lower = sorted_values[f]
    upper = sorted_values[c]
    return lower + (upper - lower) * (k - f)


def _summarize(values: List[float]) -> dict:
    if not values:
        return {
            "count": 0,
            "min": math.nan,
            "max": math.nan,
            "mean": math.nan,
            "median": math.nan,
            "p90": math.nan,
            "stdev": math.nan,
        }
    sorted_values = sorted(values)
    stdev = statistics.pstdev(values) if len(values) > 1 else 0.0
    return {
        "count": len(values),
        "min": sorted_values[0],
        "max": sorted_values[-1],
        "mean": statistics.fmean(values),
        "median": statistics.median(sorted_values),
        "p90": _percentile(sorted_values, 90.0),
        "stdev": stdev,
    }


def _format_float(value: float | None) -> str:
    if value is None or math.isnan(value):
        return "n/a"
    return f"{value:.3f}"


def build_parser() -> argparse.ArgumentParser:
    base_parser = create_argument_parser(add_help=False)
    parser = argparse.ArgumentParser(
        description=(
            "Run Simul-Whisper offline while simulating real-time streaming "
            "by sending fixed-size chunks to the model."
        ),
        parents=[base_parser],
        add_help=True,
    )
    parser.add_argument(
        "audio_path",
        type=Path,
        help="Path to the local audio file that should be transcribed.",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=Path("simul_transcript.txt"),
        help="Destination text file that will receive the detailed transcript log.",
    )
    parser.add_argument(
        "--metrics-file",
        type=Path,
        default=Path("simul_latency_metrics.txt"),
        help="Destination text file that will contain latency metrics and their explanations.",
    )
    parser.add_argument(
        "--chunk-duration",
        type=float,
        default=None,
        help=(
            "Chunk length in seconds to send per iteration. Defaults to the value of "
            "--min-chunk-size."
        ),
    )
    parser.add_argument(
        "--chunk-step",
        type=float,
        default=None,
        help=(
            "Spacing in seconds between chunk start times. Defaults to the same value as --chunk-duration."
        ),
    )
    parser.add_argument(
        "--disable-realtime-sleep",
        action="store_true",
        help="If set, process chunks back-to-back without waiting to emulate real-time playback.",
    )
    return parser


def _prepare_simul_kwargs(args) -> dict:
    return {
        "warmup_file": args.warmup_file,
        "min_chunk_size": args.min_chunk_size,
        "model_size": args.model_size,
        "model_cache_dir": args.model_cache_dir,
        "model_dir": args.model_dir,
        "lan": args.lan,
        "task": args.task,
        "disable_fast_encoder": args.disable_fast_encoder,
        "custom_alignment_heads": args.custom_alignment_heads,
        "frame_threshold": args.frame_threshold,
        "beams": args.beams,
        "decoder_type": args.decoder_type,
        "audio_max_len": args.audio_max_len,
        "audio_min_len": args.audio_min_len,
        "cif_ckpt_path": args.cif_ckpt_path,
        "never_fire": args.never_fire,
        "init_prompt": args.init_prompt,
        "static_init_prompt": args.static_init_prompt,
        "max_context_tokens": args.max_context_tokens,
        "model_path": args.model_path,
        "preload_model_count": args.preload_model_count,
    }


def main() -> None:
    parser = build_parser()
    args = finalize_parsed_args(parser.parse_args())

    if not args.audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {args.audio_path}")

    chunk_duration = args.chunk_duration or args.min_chunk_size or 0.5
    chunk_step = args.chunk_step or chunk_duration

    if chunk_duration <= 0:
        raise ValueError("--chunk-duration must be greater than zero")
    if chunk_step <= 0:
        raise ValueError("--chunk-step must be greater than zero")

    audio = load_audio(str(args.audio_path))
    audio = audio.astype(np.float32, copy=False)
    total_samples = len(audio)
    if total_samples == 0:
        raise ValueError("The provided audio file does not contain any samples")

    chunk_samples = max(1, int(round(chunk_duration * SAMPLE_RATE)))
    step_samples = max(1, int(round(chunk_step * SAMPLE_RATE)))
    total_audio_seconds = total_samples / SAMPLE_RATE

    simul_kwargs = _prepare_simul_kwargs(args)
    asr = SimulStreamingASR(**simul_kwargs)
    processor = SimulStreamingOnlineProcessor(asr)

    _ensure_parent(args.output_file)
    _ensure_parent(args.metrics_file)

    chunk_latencies: List[float] = []
    chunk_processing_times: List[float] = []
    token_latencies: List[float] = []
    all_tokens: List[ASRToken] = []

    wall_clock_start = time.perf_counter()

    with args.output_file.open("w", encoding="utf-8") as transcript_file:
        transcript_file.write("Simul-Whisper streaming transcript\n")
        transcript_file.write("==================================\n\n")
        transcript_file.write(f"Audio file: {args.audio_path}\n")
        transcript_file.write(f"Model size: {args.model_size}\n")
        if args.model_path:
            transcript_file.write(f"Model path override: {args.model_path}\n")
        transcript_file.write(f"Language: {args.lan}\n")
        transcript_file.write(f"Task: {args.task}\n")
        transcript_file.write(f"Chunk duration: {chunk_duration:.3f}s\n")
        transcript_file.write(f"Chunk step: {chunk_step:.3f}s\n")
        transcript_file.write(f"Total audio duration: {total_audio_seconds:.3f}s\n")
        transcript_file.write("\n")

        for index, (start_sample, end_sample, chunk) in enumerate(
            _iter_chunks(audio, chunk_samples, step_samples),
            start=1,
        ):
            chunk = chunk.astype(np.float32, copy=False)
            audio_start_time = start_sample / SAMPLE_RATE
            audio_end_time = end_sample / SAMPLE_RATE
            is_last = end_sample >= total_samples

            if not args.disable_realtime_sleep:
                target_time = wall_clock_start + audio_start_time
                now = time.perf_counter()
                if target_time > now:
                    time.sleep(target_time - now)

            processing_start = time.perf_counter()
            processor.insert_audio_chunk(chunk, audio_end_time)
            tokens, processed_until = processor.process_iter(is_last=is_last)
            processing_end = time.perf_counter()

            relative_start = processing_start - wall_clock_start
            relative_end = processing_end - wall_clock_start
            processing_time = processing_end - processing_start
            latency = processing_end - (wall_clock_start + audio_end_time)

            chunk_processing_times.append(processing_time)
            chunk_latencies.append(latency)

            transcript_file.write(
                f"Chunk {index}\n"
                f"  Audio window: {_format_float(audio_start_time)}s -> {_format_float(audio_end_time)}s\n"
                f"  Wall clock: {_format_float(relative_start)}s -> {_format_float(relative_end)}s "
                f"(processing {_format_float(processing_time)}s)\n"
                f"  Processed audio until: {_format_float(processed_until)}s\n"
                f"  Chunk latency: {_format_float(latency)}s\n"
            )

            if tokens:
                transcript_file.write("  Tokens emitted:\n")
            else:
                transcript_file.write("  Tokens emitted: none\n")

            for token_idx, token in enumerate(tokens, start=1):
                all_tokens.append(token)
                emission_time = relative_end
                token_latency = emission_time - token.end if token.end is not None else math.nan
                if token.end is not None:
                    token_latencies.append(token_latency)
                probability = (
                    f"{token.probability:.3f}" if token.probability is not None else "n/a"
                )
                transcript_file.write(
                    "    "
                    f"{token_idx}. text={token.text!r} "
                    f"start={_format_float(token.start)}s "
                    f"end={_format_float(token.end)}s "
                    f"speaker={token.speaker} "
                    f"prob={probability} "
                    f"lang={token.detected_language or 'n/a'} "
                    f"emission={_format_float(emission_time)}s "
                    f"latency={_format_float(token_latency)}s\n"
                )

            transcript_file.write("\n")
            transcript_file.flush()

        full_transcript = Transcript.from_tokens(all_tokens, sep=" ") if all_tokens else None
        transcript_file.write("Final transcript:\n")
        transcript_file.write((full_transcript.text if full_transcript else "") + "\n")

    wall_clock_end = time.perf_counter()
    wall_clock_duration = wall_clock_end - wall_clock_start

    chunk_summary = _summarize(chunk_latencies)
    processing_summary = _summarize(chunk_processing_times)
    token_summary = _summarize(token_latencies)
    real_time_factor = wall_clock_duration / total_audio_seconds

    with args.metrics_file.open("w", encoding="utf-8") as metrics_file:
        metrics_file.write("Simul-Whisper latency report\n")
        metrics_file.write("==============================\n\n")
        metrics_file.write(f"Audio duration: {total_audio_seconds:.3f} s\n")
        metrics_file.write(f"Wall-clock runtime: {wall_clock_duration:.3f} s\n")
        metrics_file.write(f"Real-time factor (RTF): {real_time_factor:.3f}\n")
        metrics_file.write(f"Chunks processed: {chunk_summary['count']}\n")
        metrics_file.write(f"Tokens emitted: {token_summary['count']}\n\n")

        metrics_file.write("Chunk latency statistics (wall clock end - audio chunk end):\n")
        for key in ("min", "max", "mean", "median", "p90", "stdev"):
            metrics_file.write(f"  {key}: {_format_float(chunk_summary[key])} s\n")
        metrics_file.write("\n")

        metrics_file.write("Chunk processing time statistics (wall clock duration per chunk):\n")
        for key in ("min", "max", "mean", "median", "p90", "stdev"):
            metrics_file.write(f"  {key}: {_format_float(processing_summary[key])} s\n")
        metrics_file.write("\n")

        metrics_file.write(
            "Token latency statistics (emission time - token end timestamp):\n"
        )
        for key in ("min", "max", "mean", "median", "p90", "stdev"):
            metrics_file.write(f"  {key}: {_format_float(token_summary[key])} s\n")
        metrics_file.write("\n")

        metrics_file.write("Metric explanations:\n")
        metrics_file.write(
            "- Chunk latency measures how long after the audio for a chunk finished "
            "playing the model produced its results. Negative values indicate the "
            "model stayed ahead of real time.\n"
        )
        metrics_file.write(
            "- Chunk processing time is the wall-clock duration spent processing each "
            "chunk. It highlights computational load per iteration.\n"
        )
        metrics_file.write(
            "- Token latency compares when a token was emitted against its timestamp in "
            "the audio. This indicates how quickly individual words become available.\n"
        )
        metrics_file.write(
            "- Real-time factor (RTF) compares total runtime with audio duration. Values "
            "below 1.0 mean the transcription kept up with or exceeded real time.\n"
        )


if __name__ == "__main__":
    main()
