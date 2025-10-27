"""Simulate a real-time transcription session without WebSockets."""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import librosa
import numpy as np

from whisperlivekit.core import TranscriptionEngine, online_factory
from whisperlivekit.parse_args import build_argument_parser
from whisperlivekit.timed_objects import ASRToken


LOGGER = logging.getLogger(__name__)


def _token_to_dict(token: ASRToken) -> Dict[str, Optional[float]]:
    """Convert a token dataclass to a JSON-serialisable dictionary."""

    return {
        "start": token.start,
        "end": token.end,
        "text": token.text,
        "speaker": token.speaker,
        "probability": token.probability,
        "detected_language": token.detected_language,
    }


def _ensure_parent(path: Path) -> None:
    if path.parent and not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)


def _normalise_config(namespace: argparse.Namespace) -> Dict[str, object]:
    config: Dict[str, object] = {}
    for key, value in vars(namespace).items():
        if isinstance(value, Path):
            config[key] = str(value)
        else:
            config[key] = value
    return config


def _write_json_line(handle, payload: Dict[str, object], indent: Optional[int]) -> None:
    json.dump(payload, handle, indent=indent)
    handle.write("\n")
    handle.flush()


def _compute_percentile(data: List[float], percentile: float) -> Optional[float]:
    if not data:
        return None
    array = np.array(data, dtype=np.float64)
    return float(np.percentile(array, percentile))


def build_simulation_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("input_file", type=Path, help="Path to the local audio file to transcribe.")
    parser.add_argument(
        "--transcript-output",
        type=Path,
        default=Path("simulated_transcript.txt"),
        help="Where to save the final transcript text.",
    )
    parser.add_argument(
        "--json-log-output",
        type=Path,
        default=Path("simulated_transcript_events.jsonl"),
        help="Where to store JSON-formatted streaming events.",
    )
    parser.add_argument(
        "--metrics-output",
        type=Path,
        default=Path("simulated_transcript_metrics.txt"),
        help="Where to write the latency metrics summary.",
    )
    parser.add_argument(
        "--chunk-duration",
        type=float,
        default=0.5,
        help="Length of the audio chunk (in seconds) sent to the model at each step.",
    )
    parser.add_argument(
        "--chunk-interval",
        type=float,
        default=0.5,
        help="How often (in seconds) to send a new chunk. Defaults to real-time playback.",
    )
    parser.add_argument(
        "--json-indent",
        type=int,
        default=None,
        help="Indentation level for JSON logs. Leave empty for compact JSON lines.",
    )
    parser.add_argument(
        "--no-sleep",
        action="store_true",
        help="Disable real-time sleeping. When set, chunks are processed as fast as possible.",
    )
    return parser


def load_audio(audio_path: Path, target_sr: int) -> np.ndarray:
    LOGGER.info("Loading audio from %s", audio_path)
    audio, sr = librosa.load(audio_path, sr=target_sr, mono=True)
    if sr != target_sr:
        LOGGER.warning("Audio resampled to %d Hz from %d Hz", target_sr, sr)
    return audio.astype(np.float32)


def simulate_transcription(
    input_file: Path,
    transcript_output: Path,
    json_log_output: Path,
    metrics_output: Path,
    chunk_duration: float,
    chunk_interval: float,
    json_indent: Optional[int],
    sleep_between_chunks: bool,
    engine_args: argparse.Namespace,
) -> None:
    sample_rate = 16000
    audio = load_audio(input_file, sample_rate)

    if audio.size == 0:
        raise ValueError("Input audio is empty; nothing to transcribe.")

    chunk_samples = max(1, int(round(chunk_duration * sample_rate)))
    total_samples = audio.shape[0]
    total_chunks = math.ceil(total_samples / chunk_samples)
    total_audio_duration = total_samples / sample_rate

    engine_args.min_chunk_size = chunk_duration
    engine_kwargs = _normalise_config(engine_args)

    LOGGER.info("Initialising transcription engine with backend %s", engine_kwargs.get("backend"))
    engine = TranscriptionEngine(**engine_kwargs)
    processor = online_factory(engine.args, engine.asr)

    _ensure_parent(json_log_output)
    _ensure_parent(transcript_output)
    _ensure_parent(metrics_output)

    stream_start = time.perf_counter()
    per_chunk_durations: List[float] = []
    per_token_latencies: List[float] = []
    committed_tokens: List[ASRToken] = []

    transcript_text = ""

    with json_log_output.open("w", encoding="utf-8") as log_handle:
        session_entry = {
            "event": "session_start",
            "timestamp": 0.0,
            "input_file": str(input_file.resolve()),
            "chunk_duration": chunk_duration,
            "chunk_interval": chunk_interval,
            "engine_config": engine_kwargs,
        }
        _write_json_line(log_handle, session_entry, json_indent)

        for chunk_index in range(total_chunks):
            chunk_start_sample = chunk_index * chunk_samples
            chunk_end_sample = min(total_samples, chunk_start_sample + chunk_samples)
            chunk_audio = audio[chunk_start_sample:chunk_end_sample]

            if chunk_audio.size == 0:
                continue

            target_time = stream_start + chunk_index * chunk_interval
            if sleep_between_chunks:
                wait_duration = target_time - time.perf_counter()
                if wait_duration > 0:
                    time.sleep(wait_duration)

            chunk_audio = np.ascontiguousarray(chunk_audio, dtype=np.float32)
            stream_time_end = chunk_end_sample / sample_rate

            processor.insert_audio_chunk(chunk_audio, audio_stream_end_time=stream_time_end)

            process_start = time.perf_counter()
            tokens, processed_upto = processor.process_iter()
            process_end = time.perf_counter()
            processing_duration = process_end - process_start
            per_chunk_durations.append(processing_duration)

            buffer_transcript = processor.get_buffer()
            buffer_text = buffer_transcript.text if buffer_transcript else ""

            event_payload = {
                "event": "chunk_processed",
                "chunk_index": chunk_index,
                "audio_range": {
                    "start": chunk_start_sample / sample_rate,
                    "end": stream_time_end,
                },
                "wall_clock": {
                    "processing_start": process_start - stream_start,
                    "processing_end": process_end - stream_start,
                    "processing_duration": processing_duration,
                },
                "processed_audio_upto": processed_upto,
                "committed_tokens": [_token_to_dict(token) for token in tokens],
                "buffer_text": buffer_text,
            }
            _write_json_line(log_handle, event_payload, json_indent)

            if tokens:
                emission_time = process_end - stream_start
                for token in tokens:
                    committed_tokens.append(token)
                    if token.end is not None:
                        per_token_latencies.append(max(0.0, emission_time - token.end))

        # Final flush for remaining hypotheses
        residual_tokens, residual_processed = processor.finish()
        if residual_tokens:
            flush_time = time.perf_counter() - stream_start
            event_payload = {
                "event": "final_flush",
                "processed_audio_upto": residual_processed,
                "residual_tokens": [_token_to_dict(token) for token in residual_tokens],
            }
            _write_json_line(log_handle, event_payload, json_indent)

            committed_tokens.extend(residual_tokens)
            for token in residual_tokens:
                if token.end is not None:
                    per_token_latencies.append(max(0.0, flush_time - token.end))

        transcript = processor.concatenate_tokens(committed_tokens)
        transcript_text = transcript.text if transcript else ""

        summary_event = {
            "event": "session_end",
            "total_chunks": total_chunks,
            "total_audio_duration": total_audio_duration,
            "total_tokens": len(committed_tokens),
            "final_transcript": transcript_text,
        }
        _write_json_line(log_handle, summary_event, json_indent)

    with transcript_output.open("w", encoding="utf-8") as transcript_handle:
        transcript_handle.write("WhisperLiveKit simulated streaming transcript\n")
        transcript_handle.write(f"Input file: {input_file}\n")
        transcript_handle.write(f"Backend: {engine_kwargs.get('backend')}\n")
        transcript_handle.write(f"Model: {engine_kwargs.get('model_size')}\n")
        transcript_handle.write(f"Chunk duration: {chunk_duration:.3f}s\n")
        transcript_handle.write("\n")
        transcript_handle.write(transcript_text)
        transcript_handle.write("\n")

    total_processing_time = sum(per_chunk_durations)
    real_time_factor = total_processing_time / total_audio_duration if total_audio_duration else 0.0

    latency_mean = float(np.mean(per_token_latencies)) if per_token_latencies else None
    latency_median = _compute_percentile(per_token_latencies, 50)
    latency_p90 = _compute_percentile(per_token_latencies, 90)
    latency_max = max(per_token_latencies) if per_token_latencies else None

    chunk_mean = float(np.mean(per_chunk_durations)) if per_chunk_durations else None
    chunk_median = _compute_percentile(per_chunk_durations, 50)
    chunk_p90 = _compute_percentile(per_chunk_durations, 90)
    chunk_max = max(per_chunk_durations) if per_chunk_durations else None

    with metrics_output.open("w", encoding="utf-8") as metrics_handle:
        metrics_handle.write("Latency metrics summary\n")
        metrics_handle.write("=======================\n\n")
        metrics_handle.write(f"Input file: {input_file}\n")
        metrics_handle.write(f"Audio duration: {total_audio_duration:.3f} s\n")
        metrics_handle.write(f"Chunk duration: {chunk_duration:.3f} s\n")
        metrics_handle.write(f"Chunk interval: {chunk_interval:.3f} s\n")
        metrics_handle.write(f"Chunks processed: {total_chunks}\n")
        metrics_handle.write(f"Total committed tokens: {len(committed_tokens)}\n")
        metrics_handle.write(f"Total processing time: {total_processing_time:.3f} s\n")
        metrics_handle.write(f"Real-time factor (processing / audio): {real_time_factor:.3f}\n")

        def _fmt(value: Optional[float]) -> str:
            return f"{value:.3f} s" if value is not None else "n/a"

        metrics_handle.write("\nChunk processing time statistics\n")
        metrics_handle.write(f"  Mean: {_fmt(chunk_mean)}\n")
        metrics_handle.write(f"  Median: {_fmt(chunk_median)}\n")
        metrics_handle.write(f"  P90: {_fmt(chunk_p90)}\n")
        metrics_handle.write(f"  Max: {_fmt(chunk_max)}\n")

        metrics_handle.write("\nToken emission latency (wall-clock emission minus token end time)\n")
        metrics_handle.write(f"  Mean: {_fmt(latency_mean)}\n")
        metrics_handle.write(f"  Median: {_fmt(latency_median)}\n")
        metrics_handle.write(f"  P90: {_fmt(latency_p90)}\n")
        metrics_handle.write(f"  Max: {_fmt(latency_max)}\n")

        metrics_handle.write("\nMetric explanations:\n")
        metrics_handle.write("- Real-time factor compares the cumulative processing time against the length of the audio. Values below 1.0 indicate faster-than-real-time decoding.\n")
        metrics_handle.write("- Token emission latency measures how long after a token's end timestamp the transcript was emitted during the simulation. Lower values indicate fresher partial results.\n")
        metrics_handle.write("- Chunk processing time statistics describe how long each streamed audio block took to be transcribed, reflecting instantaneous responsiveness.\n")


def main(argv: Optional[Iterable[str]] = None) -> None:
    simulation_parser = build_simulation_parser()
    simulation_args, remaining = simulation_parser.parse_known_args(argv)

    engine_parser = build_argument_parser()
    engine_args = engine_parser.parse_args(remaining)

    log_level = getattr(logging, engine_args.log_level.upper(), logging.INFO)
    logging.basicConfig(level=log_level, format="%(asctime)s - %(levelname)s - %(message)s")

    simulate_transcription(
        input_file=simulation_args.input_file,
        transcript_output=simulation_args.transcript_output,
        json_log_output=simulation_args.json_log_output,
        metrics_output=simulation_args.metrics_output,
        chunk_duration=simulation_args.chunk_duration,
        chunk_interval=simulation_args.chunk_interval,
        json_indent=simulation_args.json_indent,
        sleep_between_chunks=not simulation_args.no_sleep,
        engine_args=engine_args,
    )


if __name__ == "__main__":
    main()

