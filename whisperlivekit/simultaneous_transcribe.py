"""Utility script to run SimulStreaming transcription on a local file."""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import librosa
import numpy as np
import soundfile as sf

from whisperlivekit.core import TranscriptionEngine, online_factory
from whisperlivekit.parse_args import build_argument_parser, postprocess_parsed_args
from whisperlivekit.timed_objects import ASRToken


SAMPLING_RATE = 16000


def create_parser() -> argparse.ArgumentParser:
    parent_parser = build_argument_parser(add_help=False)
    parser = argparse.ArgumentParser(
        description="Run SimulStreaming transcription on a local audio file.",
        parents=[parent_parser],
        add_help=True,
    )
    parser.add_argument(
        "--input-file",
        required=True,
        type=str,
        help="Path to the local audio file to transcribe.",
    )
    parser.add_argument(
        "--transcript-output",
        type=str,
        default="simul_transcript.txt",
        help="Path of the text file where the final transcript will be written.",
    )
    parser.add_argument(
        "--json-log-output",
        type=str,
        default="simul_transcription_log.jsonl",
        help="Path of the JSONL file capturing every model output.",
    )
    parser.add_argument(
        "--latency-output",
        type=str,
        default="simul_latency_summary.txt",
        help="Path of the text file where latency metrics will be summarised.",
    )
    parser.add_argument(
        "--chunk-duration",
        type=float,
        default=0.5,
        help="Audio chunk duration (in seconds) used to simulate real-time streaming.",
    )
    parser.add_argument(
        "--no-sleep",
        action="store_true",
        help="Send chunks back-to-back without enforcing real-time pacing.",
    )
    parser.add_argument(
        "--mock-transcription",
        action="store_true",
        help="Simulate transcription without loading Whisper (useful for offline testing).",
    )
    return parser


def _ensure_directory(path: Path) -> None:
    if path.parent and not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)


def _load_audio(path: Path) -> Tuple[np.ndarray, int]:
    audio, sr = sf.read(path)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    if sr != SAMPLING_RATE:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLING_RATE)
        sr = SAMPLING_RATE
    audio = audio.astype(np.float32)
    return audio, sr


def _serialize_dataclass(obj: Any) -> Any:
    if obj is None:
        return None
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, list):
        return [_serialize_dataclass(item) for item in obj]
    return obj


def _tokens_to_dict(tokens: Iterable[ASRToken]) -> List[Dict[str, Any]]:
    return [asdict(token) for token in tokens]


def _write_json_line(file_handle, payload: Dict[str, Any]) -> None:
    json.dump(payload, file_handle)
    file_handle.write("\n")
    file_handle.flush()


class MockOnlineProcessor:
    """Minimal stand-in for SimulStreaming when the real model cannot be loaded."""

    def __init__(self):
        self.committed: List[ASRToken] = []
        self._pending_start: float = 0.0
        self._pending_end: float = 0.0
        self._pending_amplitude: float = 0.0

    def prepare_chunk(self, start: float, end: float) -> None:
        self._pending_start = start
        self._pending_end = end

    def insert_audio_chunk(self, audio: np.ndarray, audio_stream_end_time: float) -> None:
        if audio.size:
            self._pending_amplitude = float(np.mean(np.abs(audio)))
        else:
            self._pending_amplitude = 0.0

    def process_iter(self, is_last: bool = False):
        if self._pending_end <= self._pending_start:
            return [], self._pending_end
        if self._pending_amplitude < 1e-4:
            text = " (silence)"
        else:
            text = f" chunk{len(self.committed) + 1}"
        token = ASRToken(
            start=self._pending_start,
            end=self._pending_end,
            text=text,
            speaker=-1,
            probability=1.0,
            detected_language="mock",
        )
        self.committed.append(token)
        return [token], self._pending_end

    def get_buffer(self):
        return None


def run_transcription(args) -> None:
    if args.backend != "simulstreaming" and not args.mock_transcription:
        raise ValueError(
            "This utility is designed for the SimulStreaming backend. "
            "Please run with --backend simulstreaming."
        )

    if not args.transcription:
        raise ValueError("Transcription must be enabled for this script.")

    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    logging.basicConfig(level=log_level, format="%(asctime)s [%(levelname)s] %(message)s")

    input_path = Path(args.input_file).expanduser().resolve()
    transcript_path = Path(args.transcript_output).expanduser().resolve()
    log_path = Path(args.json_log_output).expanduser().resolve()
    latency_path = Path(args.latency_output).expanduser().resolve()

    logging.info("Loading audio from %s", input_path)
    audio, sr = _load_audio(input_path)
    if sr != SAMPLING_RATE:
        raise RuntimeError(f"Expected sampling rate {SAMPLING_RATE}, got {sr}")

    total_samples = audio.shape[0]
    total_duration = total_samples / sr

    logging.info("Audio duration: %.2f seconds", total_duration)

    chunk_duration = max(args.chunk_duration, 0.001)
    chunk_samples = int(round(chunk_duration * sr))
    if chunk_samples <= 0:
        raise ValueError("Chunk duration must be positive")

    engine_kwargs = vars(args).copy()
    for field in [
        "input_file",
        "transcript_output",
        "json_log_output",
        "latency_output",
        "chunk_duration",
        "no_sleep",
        "mock_transcription",
    ]:
        engine_kwargs.pop(field, None)

    if args.mock_transcription:
        engine = None
        online_processor = MockOnlineProcessor()
    else:
        engine = TranscriptionEngine(**engine_kwargs)
        online_processor = online_factory(engine.args, engine.asr)

    _ensure_directory(transcript_path)
    _ensure_directory(log_path)
    _ensure_directory(latency_path)

    chunk_processing_times: List[float] = []
    token_latencies: List[float] = []
    schedule_delays: List[float] = []

    overall_start = time.perf_counter()
    next_scheduled_time = overall_start + chunk_duration

    with log_path.open("w", encoding="utf-8") as log_file:
        audio_end_pointer = 0
        chunk_index = 0
        while audio_end_pointer < total_samples:
            chunk_start_sample = audio_end_pointer
            chunk_end_sample = min(chunk_start_sample + chunk_samples, total_samples)
            chunk = audio[chunk_start_sample:chunk_end_sample]

            chunk_start_time = chunk_start_sample / sr
            chunk_end_time = chunk_end_sample / sr
            actual_chunk_duration = chunk_end_time - chunk_start_time

            scheduled_time_rel = next_scheduled_time - overall_start

            if not args.no_sleep:
                now = time.perf_counter()
                if now < next_scheduled_time:
                    time.sleep(next_scheduled_time - now)

            if hasattr(online_processor, "prepare_chunk"):
                online_processor.prepare_chunk(chunk_start_time, chunk_end_time)

            send_time = time.perf_counter()
            schedule_delay = send_time - next_scheduled_time
            schedule_delays.append(schedule_delay)

            online_processor.insert_audio_chunk(np.asarray(chunk, dtype=np.float32), chunk_end_time)
            tokens, processed_until = online_processor.process_iter(is_last=False)
            process_end_time = time.perf_counter()

            send_time_rel = send_time - overall_start
            process_end_time_rel = process_end_time - overall_start

            chunk_processing = process_end_time - send_time
            chunk_processing_times.append(chunk_processing)

            chunk_latency = process_end_time_rel - chunk_end_time

            token_payload = _tokens_to_dict(tokens)

            for token in tokens:
                reference_end = token.end if token.end is not None else processed_until
                token_latencies.append(process_end_time_rel - reference_end)

            payload = {
                "event": "chunk_processed",
                "chunk_index": chunk_index,
                "chunk_start": chunk_start_time,
                "chunk_end": chunk_end_time,
                "scheduled_time": scheduled_time_rel,
                "send_time": send_time_rel,
                "process_end_time": process_end_time_rel,
                "chunk_processing_time": chunk_processing,
                "chunk_latency": chunk_latency,
                "schedule_delay": schedule_delay,
                "audio_processed_until": processed_until,
                "tokens": token_payload,
                "buffer_transcription": _serialize_dataclass(online_processor.get_buffer()),
                "committed_tokens": _tokens_to_dict(online_processor.committed),
            }
            _write_json_line(log_file, payload)

            audio_end_pointer = chunk_end_sample
            chunk_index += 1
            next_scheduled_time += actual_chunk_duration

        tokens, processed_until = online_processor.process_iter(is_last=True)
        final_process_end = time.perf_counter()
        final_process_end_rel = final_process_end - overall_start
        final_payload = {
            "event": "finalize",
            "process_end_time": final_process_end_rel,
            "audio_processed_until": processed_until,
            "tokens": _tokens_to_dict(tokens),
            "buffer_transcription": _serialize_dataclass(online_processor.get_buffer()),
            "committed_tokens": _tokens_to_dict(online_processor.committed),
        }
        _write_json_line(log_file, final_payload)

        for token in tokens:
            reference_end = token.end if token.end is not None else processed_until
            token_latencies.append(final_process_end_rel - reference_end)

    if hasattr(online_processor, "model"):
        try:
            online_processor.model.refresh_segment(complete=True)
        except Exception:  # pragma: no cover - best effort cleanup
            logging.debug("Failed to refresh model segment during cleanup", exc_info=True)

    overall_end = time.perf_counter()

    committed_text = "".join(token.text for token in online_processor.committed)
    if committed_text and not committed_text.endswith("\n"):
        committed_text = committed_text + "\n"
    transcript_path.write_text(committed_text, encoding="utf-8")

    _write_latency_summary(
        latency_path=latency_path,
        total_audio_duration=total_duration,
        wall_time=overall_end - overall_start,
        chunk_processing_times=chunk_processing_times,
        token_latencies=token_latencies,
        schedule_delays=schedule_delays,
        total_chunks=len(chunk_processing_times),
        total_tokens=len(online_processor.committed),
        chunk_duration=args.chunk_duration,
    )


def _write_latency_summary(
    *,
    latency_path: Path,
    total_audio_duration: float,
    wall_time: float,
    chunk_processing_times: List[float],
    token_latencies: List[float],
    schedule_delays: List[float],
    total_chunks: int,
    total_tokens: int,
    chunk_duration: float,
) -> None:
    def _format_stats(values: List[float]) -> str:
        if not values:
            return "n/a"
        return (
            f"min={min(values):.4f}s, max={max(values):.4f}s, "
            f"avg={sum(values)/len(values):.4f}s"
        )

    real_time_factor = wall_time / total_audio_duration if total_audio_duration > 0 else float("inf")

    summary_lines = [
        "Latency Metrics Summary",
        "========================",
        f"Total audio duration: {total_audio_duration:.3f}s",
        "  Explanation: Length of the processed audio after resampling to 16 kHz.",
        f"Total wall-clock time: {wall_time:.3f}s",
        "  Explanation: Actual elapsed time spent streaming and decoding the audio.",
        f"Real-time factor: {real_time_factor:.3f}",
        "  Explanation: Wall-clock time divided by audio duration (<= 1 indicates faster than real time).",
        f"Configured chunk duration: {chunk_duration:.3f}s",
        "  Explanation: Target duration of each audio slice sent to the model.",
        f"Chunk processing time stats: {_format_stats(chunk_processing_times)}",
        "  Explanation: Time between dispatching a chunk and obtaining model output.",
        f"Token latency stats: {_format_stats(token_latencies)}",
        "  Explanation: Delay between the end timestamp of each token and the moment it was available.",
        f"Scheduling delay stats: {_format_stats(schedule_delays)}",
        "  Explanation: Difference between the planned dispatch time of each chunk and when it was actually sent.",
        f"Chunks processed: {total_chunks}",
        "  Explanation: Number of streaming iterations executed.",
        f"Tokens committed: {total_tokens}",
        "  Explanation: Count of tokens returned by the model across the entire session.",
    ]

    latency_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")


def main():
    parser = create_parser()
    args = parser.parse_args()
    args = postprocess_parsed_args(args)
    run_transcription(args)


if __name__ == "__main__":
    main()
