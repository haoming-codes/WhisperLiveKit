#!/usr/bin/env python
"""Simulated streaming transcription runner.

This script feeds local audio to the SimulStreaming backend in 500 ms chunks
(and optionally any chunk duration provided) to emulate a real-time
transcription session without relying on WebSockets.
"""
from __future__ import annotations

import json
import logging
import statistics
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List

# Ensure the repository root is importable when running as a script
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import librosa

from whisperlivekit.parse_args import build_parser
from whisperlivekit.simul_whisper.backend import (
    SimulStreamingASR,
    SimulStreamingOnlineProcessor,
)
from whisperlivekit.timed_objects import ASRToken, Transcript


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _token_to_dict(token: ASRToken) -> Dict[str, Any]:
    data = asdict(token)
    # dataclasses.asdict converts enums to ints already; ensure floats are native
    if data.get("start") is not None:
        data["start"] = float(data["start"])
    if data.get("end") is not None:
        data["end"] = float(data["end"])
    if data.get("probability") is not None:
        data["probability"] = float(data["probability"])
    if data.get("speaker") is not None:
        data["speaker"] = int(data["speaker"])
    return data


def _transcript_from_tokens(tokens: Iterable[ASRToken]) -> str:
    if not tokens:
        return ""
    transcript = Transcript.from_tokens(list(tokens), sep="")
    return transcript.text or ""


def _log_event(log_file, event: str, **payload: Any) -> None:
    record = {"timestamp": time.time(), "event": event, **payload}
    log_file.write(json.dumps(record, ensure_ascii=False) + "\n")
    log_file.flush()


def main(argv: List[str] | None = None) -> int:
    parser = build_parser()
    parser.description = (
        "Simulated offline driver for the SimulStreaming backend. "
        "Streams a local file as small chunks and records every output."
    )
    parser.add_argument(
        "--input-file",
        required=True,
        help="Path to the local audio file to stream.",
    )
    parser.add_argument(
        "--transcript-output",
        type=str,
        default="simul_transcript.txt",
        help="Destination file for the concatenated transcript.",
    )
    parser.add_argument(
        "--log-output",
        type=str,
        default="simul_transcription_log.jsonl",
        help="Path to the JSONL log capturing incremental model outputs.",
    )
    parser.add_argument(
        "--latency-output",
        type=str,
        default="simul_latency_summary.json",
        help="Path to a JSON file containing latency metrics.",
    )
    parser.add_argument(
        "--chunk-duration",
        type=float,
        default=0.5,
        help="Duration of audio (in seconds) to send per chunk.",
    )
    parser.add_argument(
        "--chunk-interval",
        type=float,
        default=0.5,
        help="Wall-clock interval between chunk dispatches (seconds).",
    )
    parser.add_argument(
        "--no-sleep",
        action="store_true",
        help="Disable real-time sleeping between chunks (useful for tests).",
    )

    args = parser.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.log_level, logging.INFO))

    input_path = Path(args.input_file)
    transcript_path = Path(args.transcript_output)
    log_path = Path(args.log_output)
    latency_path = Path(args.latency_output)
    chunk_duration = float(args.chunk_duration)
    chunk_interval = float(args.chunk_interval)
    do_sleep = not args.no_sleep

    if not input_path.exists():
        raise FileNotFoundError(f"Input file {input_path} does not exist")

    _ensure_parent(transcript_path)
    _ensure_parent(log_path)
    _ensure_parent(latency_path)

    audio, sr = librosa.load(str(input_path), sr=SimulStreamingOnlineProcessor.SAMPLING_RATE)
    audio = audio.astype(np.float32)

    chunk_samples = max(int(chunk_duration * sr), 1)
    total_samples = audio.shape[-1]
    total_audio_duration = total_samples / sr

    reserved_keys = {
        "input_file",
        "transcript_output",
        "log_output",
        "latency_output",
        "chunk_duration",
        "chunk_interval",
        "no_sleep",
    }
    asr_kwargs = {k: v for k, v in vars(args).items() if k not in reserved_keys}

    if asr_kwargs.get("backend") and asr_kwargs["backend"] != "simulstreaming":
        logging.warning(
            "Overriding backend %s with 'simulstreaming' for this runner.",
            asr_kwargs["backend"],
        )
    asr_kwargs["backend"] = "simulstreaming"

    asr = SimulStreamingASR(**asr_kwargs)
    processor = SimulStreamingOnlineProcessor(asr=asr)

    latencies: List[Dict[str, Any]] = []
    committed_tokens: List[ASRToken] = []

    with log_path.open("w", encoding="utf-8") as log_file:
        _log_event(
            log_file,
            "configuration",
            configuration={k: v for k, v in asr_kwargs.items()},
            input_file=str(input_path),
            sample_rate=sr,
            chunk_duration_s=chunk_duration,
            chunk_interval_s=chunk_interval,
        )

        for chunk_index, start in enumerate(range(0, total_samples, chunk_samples)):
            end = min(start + chunk_samples, total_samples)
            audio_chunk = audio[start:end]
            if audio_chunk.size == 0:
                continue

            chunk_start_time = start / sr
            chunk_end_time = end / sr
            is_last = end >= total_samples

            send_time = time.time()
            processor.insert_audio_chunk(audio_chunk, audio_stream_end_time=chunk_end_time)
            inference_start = time.time()
            tokens, processed_up_to = processor.process_iter(is_last=is_last)
            inference_end = time.time()

            chunk_latency = inference_end - send_time
            inference_latency = inference_end - inference_start

            committed_tokens.extend(tokens)

            buffer_transcript = processor.get_buffer()
            emitted_tokens = [_token_to_dict(token) for token in tokens]

            latencies.append(
                {
                    "chunk_index": chunk_index,
                    "chunk_start": chunk_start_time,
                    "chunk_end": chunk_end_time,
                    "chunk_latency": chunk_latency,
                    "inference_latency": inference_latency,
                    "processed_audio_time": processed_up_to,
                    "emitted_token_count": len(tokens),
                }
            )

            _log_event(
                log_file,
                "chunk_processed",
                chunk_index=chunk_index,
                chunk_start=chunk_start_time,
                chunk_end=chunk_end_time,
                is_last=is_last,
                processed_audio_time=processed_up_to,
                chunk_latency_s=chunk_latency,
                inference_latency_s=inference_latency,
                emitted_tokens=emitted_tokens,
                buffer_transcript=asdict(buffer_transcript)
                if buffer_transcript
                else None,
                total_committed=len(committed_tokens),
            )

            if do_sleep and not is_last:
                elapsed = time.time() - send_time
                sleep_time = max(chunk_interval - elapsed, 0.0)
                time.sleep(sleep_time)

        # Ensure any remaining tokens are flushed
        flush_start = time.time()
        tokens, processed_up_to = processor.process_iter(is_last=True)
        flush_end = time.time()
        if tokens:
            committed_tokens.extend(tokens)
            latencies.append(
                {
                    "chunk_index": len(latencies),
                    "chunk_start": None,
                    "chunk_end": None,
                    "chunk_latency": flush_end - flush_start,
                    "inference_latency": flush_end - flush_start,
                    "processed_audio_time": processed_up_to,
                    "emitted_token_count": len(tokens),
                    "event": "flush",
                }
            )
            _log_event(
                log_file,
                "final_flush",
                processed_audio_time=processed_up_to,
                emitted_tokens=[_token_to_dict(token) for token in tokens],
                total_committed=len(committed_tokens),
            )

        final_transcript = _transcript_from_tokens(committed_tokens)
        _log_event(
            log_file,
            "transcription_completed",
            total_tokens=len(committed_tokens),
            transcript=final_transcript,
        )

    transcript_path.write_text(final_transcript, encoding="utf-8")

    chunk_latencies = [entry["chunk_latency"] for entry in latencies if entry.get("chunk_latency") is not None]
    inference_latencies = [
        entry["inference_latency"] for entry in latencies if entry.get("inference_latency") is not None
    ]

    summary: Dict[str, Any] = {
        "metrics": {
            "num_chunks": len(latencies),
            "total_audio_duration_s": total_audio_duration,
            "mean_chunk_latency_s": statistics.mean(chunk_latencies) if chunk_latencies else 0.0,
            "median_chunk_latency_s": statistics.median(chunk_latencies) if chunk_latencies else 0.0,
            "max_chunk_latency_s": max(chunk_latencies) if chunk_latencies else 0.0,
            "mean_inference_latency_s": statistics.mean(inference_latencies) if inference_latencies else 0.0,
            "real_time_factor": (sum(inference_latencies) / total_audio_duration)
            if total_audio_duration > 0 and inference_latencies
            else 0.0,
        },
        "descriptions": {
            "num_chunks": "Total number of processing steps, including any final flush.",
            "total_audio_duration_s": "Duration of the streamed audio in seconds.",
            "mean_chunk_latency_s": "Average wall-clock latency from chunk dispatch to model response.",
            "median_chunk_latency_s": "Median of the per-chunk latency distribution.",
            "max_chunk_latency_s": "Slowest observed chunk latency.",
            "mean_inference_latency_s": "Average time spent within the inference call itself.",
            "real_time_factor": "Sum of inference latencies divided by audio duration; <1 means faster than real time.",
        },
        "per_chunk": latencies,
    }

    latency_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
