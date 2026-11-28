#!/usr/bin/env python3
"""Measure TensorRT SegFormer stages on the Jetson in three separate sessions.

The evaluator records network-only and end-to-end latency, full-path energy,
ADE20K mIoU, memory use, session spread, and TensorRT layer precision.
Run it inside the course lpcv environment with the course PowerWindow helper.
"""

from __future__ import annotations

import argparse
import csv
import gc
import os
import re
import socket
import sys
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "model"
BASELINE_DIR = ROOT / "baseline"
COURSE_HELPER_DIR = Path("/course-data/lab1")
sys.path.insert(0, str(COURSE_HELPER_DIR))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(BASELINE_DIR))

import baseline  # noqa: E402

SESSION_LOG_COLUMNS = (
    "session", "model_name", "engine_path", "engine_sha256", "board_hostname", "started_utc", "finished_utc",
    "idle_confirmed_before", "idle_confirmed_after", "power_mode", "torch_version",
    "tensorrt_version", "engine_precision_audit_status",
)
EXTRA_RESULT_COLUMNS = (
    "measurement_session", "board_hostname", "power_mode", "torch_version", "tensorrt_version",
    "engine_expected_precision", "engine_total_layer_count", "engine_expected_precision_layer_count",
    "engine_expected_precision_layer_fraction", "engine_int8_layer_count", "engine_int8_layer_fraction",
    "engine_precision_report_path", "engine_precision_report_status",
    "session_idle_confirmed_before", "session_idle_confirmed_after",
    "latency_e2e_mean_ms", "latency_e2e_p50_ms", "latency_e2e_p99_ms", "latency_e2e_fps",
    "latency_e2e_warmup_seconds", "latency_e2e_warmup_iterations", "latency_e2e_scope",
    "energy_measurement_scope",
)
RESULT_COLUMNS = baseline.RESULT_COLUMNS + EXTRA_RESULT_COLUMNS


def _torch_dtype(torch: Any, trt_dtype: Any) -> Any:
    """Map the TensorRT I/O type to the matching PyTorch tensor type."""
    mapping = {
        "FLOAT": torch.float32,
        "HALF": torch.float16,
        "INT8": torch.int8,
        "INT32": torch.int32,
        "BOOL": torch.bool,
    }
    name = str(trt_dtype).split(".")[-1].upper()
    if name not in mapping:
        raise TypeError(f"Unsupported TensorRT tensor type: {trt_dtype}")
    return mapping[name]


def load_engine(model_path: str, precision: str, device: str):
    """Deserialize a TensorRT engine file and create its execution context."""
    import tensorrt as trt

    engine_path = Path(model_path)
    if not engine_path.is_file():
        raise FileNotFoundError(f"TensorRT engine not found: {engine_path}")
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
    if engine is None:
        raise RuntimeError(f"TensorRT could not deserialize engine: {engine_path}")
    context = engine.create_execution_context()
    if context is None:
        raise RuntimeError(f"Could not create TensorRT execution context: {engine_path}")
    input_names = []
    output_names = []
    for index in range(engine.num_io_tensors):
        name = engine.get_tensor_name(index)
        if engine.get_tensor_mode(name).name == "INPUT":
            input_names.append(name)
        else:
            output_names.append(name)
    if len(input_names) != 1 or len(output_names) != 1:
        raise ValueError(f"Expected one engine input and output; found {len(input_names)} and {len(output_names)}")
    return {"engine": engine, "context": context, "runtime": runtime,
            "input_name": input_names[0], "output_name": output_names[0],
            "output": None, "input_shape": None}


def run_engine(model, inputs, device):
    """Run the loaded engine using the baseline's normalized pixel tensor."""
    import torch

    engine = model["engine"]
    context = model["context"]
    # Resolve the engine bindings for this fixed-shape input before enqueueing.
    input_names = []
    output_names = []
    for index in range(engine.num_io_tensors):
        name = engine.get_tensor_name(index)
        if engine.get_tensor_mode(name).name == "INPUT":
            input_names.append(name)
        else:
            output_names.append(name)
    if len(input_names) != 1 or len(output_names) != 1:
        raise ValueError(
            f"Expected one engine input and output; found {len(input_names)} "
            f"input(s), {len(output_names)} output(s)."
        )

    input_name, output_name = input_names[0], output_names[0]
    pixel_values = inputs["pixel_values"]
    input_shape = tuple(int(dim) for dim in pixel_values.shape)
    if not context.set_input_shape(input_name, input_shape):
        raise ValueError(f"Engine rejected input shape {input_shape}")
    pixel_values = pixel_values.to(
        device=device,
        dtype=_torch_dtype(torch, engine.get_tensor_dtype(input_name)),
    ).contiguous()
    output_shape = tuple(int(dim) for dim in context.get_tensor_shape(output_name))
    if any(dim < 1 for dim in output_shape):
        raise ValueError(f"Engine has unresolved output shape: {output_shape}")
    logits = torch.empty(
        output_shape, device=device,
        dtype=_torch_dtype(torch, engine.get_tensor_dtype(output_name)),
    )
    context.set_tensor_address(input_name, pixel_values.data_ptr())
    context.set_tensor_address(output_name, logits.data_ptr())
    stream = torch.cuda.current_stream(device)
    if not context.execute_async_v3(stream.cuda_stream):
        raise RuntimeError("TensorRT engine execution failed")
    return logits


def _precision_from_path(path: Path) -> str:
    name = path.stem.lower()
    if "int8" in name:
        return "int8"
    if "fp16" in name or "half" in name:
        return "fp16"
    return "unknown"


def _configuration_name(path: Path) -> str:
    match = re.search(r"(?i)S[0-4](?=_|\b)", path.stem)
    return match.group(0).upper() if match else path.stem


def inspect_engine_precision(engine_path: Path, report_dir: Path) -> dict[str, Any]:
    """Use the course inspector's per-layer report to count executed precisions."""
    report_path = report_dir / f"{engine_path.stem}.layers.csv"
    try:
        report_dir.mkdir(parents=True, exist_ok=True)
        helper_path = COURSE_HELPER_DIR / "lab1_trt.py"
        if not helper_path.is_file():
            raise FileNotFoundError(f"Course layer inspector not found: {helper_path}")
        subprocess.run(
            [sys.executable, str(helper_path), "layers", "--engine", str(engine_path),
             "--csv", str(report_path)],
            check=True,
        )
        with report_path.open(newline="", encoding="utf-8") as stream:
            all_layers = list(csv.DictReader(stream))
        layers = [row for row in all_layers if row.get("is_reformat", "").lower() != "true"]
        if not layers:
            raise RuntimeError("Course inspector returned no non-reformatting layers")
        unknown = [row for row in layers if row.get("output_precision") == "UNKNOWN"]
        if unknown:
            raise RuntimeError(f"Course inspector could not determine precision for {len(unknown)} layers")

        precisions = [row.get("output_precision", "").upper() for row in layers]
        int8_count = sum(precision == "INT8" for precision in precisions)
        expected = "int8" if _precision_from_path(engine_path) == "int8" else "fp16"
        expected_count = int8_count if expected == "int8" else sum(
            precision == "FP16" for precision in precisions
        )
        return {
            "engine_expected_precision": expected,
            "engine_total_layer_count": len(layers),
            "engine_expected_precision_layer_count": expected_count,
            "engine_expected_precision_layer_fraction": expected_count / len(layers),
            "engine_int8_layer_count": int8_count,
            "engine_int8_layer_fraction": int8_count / len(layers),
            "engine_precision_report_path": str(report_path),
            "engine_precision_report_status": "measured",
        }
    except Exception as exc:
        return {"engine_precision_report_path": str(report_path),
                "engine_precision_report_status": f"unavailable: {type(exc).__name__}: {exc}"}


def _command_text(command: list[str]) -> str:
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=10, check=False)
        return (result.stdout + result.stderr).strip()
    except (OSError, subprocess.TimeoutExpired):
        return ""


def _check_measurement_environment(board_hostname: str) -> tuple[str, str, str]:
    """Require the course container, CUDA, and an explicitly identified 15 W board."""
    import torch
    import tensorrt as trt

    if not Path("/.dockerenv").exists() and not Path("/run/.containerenv").exists():
        raise RuntimeError("Run this evaluator inside the course lpcv container.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; run this evaluator on the Jetson Orin Nano.")
    device_name = torch.cuda.get_device_name(torch.cuda.current_device())
    if "orin" not in device_name.lower():
        raise RuntimeError(f"Expected a Jetson Orin GPU, found {device_name!r}.")

    configured_mode = os.environ.get("LPCV_POWER_MODE", "").strip()
    if configured_mode and configured_mode.upper().replace(" ", "") not in {"15", "15W"}:
        raise RuntimeError(f"LPCV_POWER_MODE must be 15 W, found {configured_mode!r}.")
    query = _command_text(["nvpmodel", "-q"])
    if "15W" not in query.upper().replace(" ", ""):
        raise RuntimeError(
            "The actual nvpmodel setting could not be verified as 15 W. "
            f"nvpmodel -q returned: {query or '(no output)'}"
        )
    power_mode = "15 W (verified with nvpmodel -q)"
    if not board_hostname.strip():
        raise RuntimeError("Pass --board-hostname with the course board hostname for the report.")
    if board_hostname not in {"jetson-lpcv-01", "jetson-lpcv-02", "jetson-lpcv-03", "jetson-lpcv-04"}:
        raise RuntimeError(f"Unknown course board hostname: {board_hostname!r}.")
    actual_hostname = socket.gethostname().split(".")[0]
    known_course_boards = {"jetson-lpcv-01", "jetson-lpcv-02", "jetson-lpcv-03", "jetson-lpcv-04"}
    if actual_hostname in known_course_boards and actual_hostname != board_hostname:
        raise RuntimeError(f"--board-hostname is {board_hostname}, but this host reports {actual_hostname}.")
    return device_name, power_mode, str(trt.__version__)


def _confirm_idle(message: str) -> str:
    answer = input(message).strip()
    if answer != "YES":
        raise RuntimeError("Measurement stopped: board-idle confirmation was not YES.")
    return "yes"


def _session_already_recorded(output_csv: Path, session: int) -> bool:
    if not output_csv.is_file() or output_csv.stat().st_size == 0:
        return False
    with output_csv.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        return any(row.get("measurement_session") == str(session) for row in reader)


def _sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_session_plan(session_log: Path, session: int, engines: list[Path],
                           environment: dict[str, str]) -> None:
    """Require ordered sessions with the same engines, board, and toolchain."""
    if not session_log.is_file() or session_log.stat().st_size == 0:
        if session != 1:
            raise RuntimeError("Start with --session 1; measurement sessions must run in order.")
        return
    with session_log.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    prior_sessions = sorted({int(row["session"]) for row in rows})
    if prior_sessions != list(range(1, session)):
        raise RuntimeError(f"Expected completed sessions {list(range(1, session))}, found {prior_sessions}.")
    if not prior_sessions:
        return
    previous_rows = [row for row in rows if int(row["session"]) == session - 1]
    previous_engines = {row["engine_path"]: row["engine_sha256"] for row in previous_rows}
    current_engines = {str(path.resolve()): _sha256_file(path) for path in engines}
    if previous_engines != current_engines:
        raise RuntimeError("Use exactly the same engine files in all sessions. "
                           f"Previous session: {sorted(previous_engines)}; current: {sorted(current_engines)}")
    for field, current_value in environment.items():
        previous_values = {row.get(field, "") for row in previous_rows}
        if previous_values != {current_value}:
            raise RuntimeError(
                f"Keep {field} unchanged across sessions; previous values were "
                f"{sorted(previous_values)}, current value is {current_value!r}."
            )


def _append_session_log(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.is_file() and path.stat().st_size > 0
    if exists:
        with path.open(newline="", encoding="utf-8") as stream:
            if tuple(next(csv.reader(stream), [])) != SESSION_LOG_COLUMNS:
                raise ValueError(f"Session log schema does not match: {path}")
    with path.open("a" if exists else "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=SESSION_LOG_COLUMNS)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def _check_csv_header(path: Path, expected: tuple[str, ...]) -> None:
    if path.is_file() and path.stat().st_size:
        with path.open(newline="", encoding="utf-8") as stream:
            actual = tuple(next(csv.reader(stream), []))
        if actual != expected:
            raise ValueError(f"CSV schema mismatch at {path}; choose a new output path rather than mixing results.")


def _save_result(row: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.is_file() and path.stat().st_size > 0
    _check_csv_header(path, RESULT_COLUMNS)
    with path.open("a" if exists else "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=RESULT_COLUMNS)
        if not exists:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in RESULT_COLUMNS})


def print_session_summary(results_csv: Path) -> None:
    """Show session ranges and whether changes exceed the observed spread."""
    with results_csv.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        if row.get("measurement_session") in {"1", "2", "3"}:
            grouped.setdefault(row["model_name"], []).append(row)

    def values(items: list[dict[str, str]], field: str) -> list[float]:
        return [float(row[field]) for row in items if row.get(field) not in (None, "")]

    print("\nThree-session measurement summary (range is max minus min):")
    summary: dict[str, dict[str, tuple[float, float, float]]] = {}
    for name, items in sorted(grouped.items()):
        summary[name] = {}
        observed_sessions = sorted({row["measurement_session"] for row in items})
        print(f"{name}: sessions={observed_sessions}")
        for field, label in (("mean_iou_percent", "mIoU (%)"), ("latency_e2e_fps", "end-to-end FPS"),
                             ("latency_e2e_p50_ms", "end-to-end P50 (ms)"),
                             ("latency_e2e_p99_ms", "end-to-end P99 (ms)"),
                             ("latency_p50_ms", "preprocessed-input P50 (ms)"),
                             ("latency_p99_ms", "preprocessed-input P99 (ms)"),
                             ("energy_per_inference_mj", "energy (mJ/inference)")):
            data = values(items, field)
            if data:
                mean_value = sum(data) / len(data)
                spread = max(data) - min(data)
                summary[name][field] = (mean_value, spread, len(data))
                print(f"  {label}: mean={mean_value:.3f}, session spread={spread:.3f}, n={len(data)}")
        accuracy = summary[name].get("mean_iou_percent")
        p99 = summary[name].get("latency_e2e_p99_ms")
        if accuracy and p99:
            accuracy_values = values(items, "mean_iou_percent")
            p99_values = values(items, "latency_e2e_p99_ms")
            all_sessions_pass = len(accuracy_values) == 3 and len(p99_values) == 3 and \
                min(accuracy_values) >= 43.81 and max(p99_values) <= 50
            print(f"  decision rule (P99 <= 50 ms and mIoU >= 43.81%): "
                  f"{'PASS in all three sessions' if all_sessions_pass else 'not met in all three sessions'}")

    baseline_name = next((name for name in summary if re.search(r"(?i)\bS0\b", name)), None)
    if baseline_name is None:
        print("No S0 row found; comparison-to-baseline significance was not calculated.")
        return
    print(f"\nDifferences from {baseline_name}; count a difference only when |change| exceeds the larger session range:")
    for name, metrics in sorted(summary.items()):
        if name == baseline_name:
            continue
        print(f"{name}:")
        for field, label in (("latency_e2e_p99_ms", "end-to-end P99"),
                             ("latency_e2e_fps", "end-to-end FPS"),
                             ("energy_per_inference_mj", "energy")):
            if field not in metrics or field not in summary[baseline_name]:
                continue
            candidate, candidate_spread, _ = metrics[field]
            base, base_spread, _ = summary[baseline_name][field]
            if metrics[field][2] < 3 or summary[baseline_name][field][2] < 3:
                print(f"  {label}: not comparable yet; need three separate sessions for both configurations")
                continue
            threshold = max(candidate_spread, base_spread)
            difference = candidate - base if field.endswith("fps") else base - candidate
            change_label = "model - S0" if field.endswith("fps") else "S0 - model"
            print(f"  {label}: {change_label}={difference:.3f}; spread threshold={threshold:.3f}; "
                  f"larger than spread={'yes' if abs(difference) > threshold else 'no'}")


def _process_full_frame(model: Any, image: Any, processor: Any, device: Any) -> None:
    inputs = baseline.preprocess_image(image, processor, height=512, width=512)
    outputs = baseline.run_inference(model, inputs, device, inference_fn=run_engine)
    logits = baseline._extract_logits(outputs)
    baseline._to_class_maps(logits, 150, target_size=(512, 512))


def measure_end_to_end_latency(model: Any, image: Any, processor: Any, device: Any,
                               timed_iterations: int = 1000, warmup_seconds: float = 2.0) -> dict[str, Any]:
    """Time preprocessing, TensorRT inference, and full-resolution mask conversion."""
    import numpy as np
    import torch

    started = time.perf_counter()
    warmup_count = 0
    while time.perf_counter() - started < warmup_seconds or warmup_count < 2:
        _process_full_frame(model, image, processor, device)
        torch.cuda.synchronize(device)
        warmup_count += 1
    warmup_actual = time.perf_counter() - started

    samples_ms = []
    for _ in range(timed_iterations):
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        _process_full_frame(model, image, processor, device)
        torch.cuda.synchronize(device)
        samples_ms.append((time.perf_counter() - started) * 1000.0)
    values = np.asarray(samples_ms)
    mean_ms = float(values.mean())
    return {
        "latency_e2e_mean_ms": mean_ms,
        "latency_e2e_p50_ms": float(np.percentile(values, 50)),
        "latency_e2e_p99_ms": float(np.percentile(values, 99)),
        "latency_e2e_fps": 1000.0 / mean_ms if mean_ms else None,
        "latency_e2e_warmup_seconds": warmup_actual,
        "latency_e2e_warmup_iterations": warmup_count,
        "latency_e2e_scope": "image preprocessing + inference + full-resolution class-mask conversion",
    }


def measure_end_to_end_energy(model: Any, image: Any, processor: Any, device: Any) -> dict[str, Any]:
    """Measure per-frame energy for the full path with the course PowerWindow helper."""
    import torch

    power_window = baseline._power_window_class()
    idle_window = power_window().start()
    time.sleep(10.0)
    idle_window.stop()

    warmup_started = time.perf_counter()
    warmup_count = 0
    while time.perf_counter() - warmup_started < 2.0:
        _process_full_frame(model, image, processor, device)
        torch.cuda.synchronize(device)
        warmup_count += 1
    warmup_actual = time.perf_counter() - warmup_started

    busy_window = power_window().start()
    busy_started = time.perf_counter()
    inference_count = 0
    try:
        while time.perf_counter() - busy_started < 20.0:
            _process_full_frame(model, image, processor, device)
            torch.cuda.synchronize(device)
            inference_count += 1
    except BaseException:
        try:
            busy_window.stop()
        except Exception:
            pass
        raise
    busy_window.stop()

    duration = busy_window.duration_s
    idle_mw = idle_window.mean_mw
    busy_mw = busy_window.mean_mw
    energy = busy_mw * duration / inference_count
    incremental = (busy_mw - idle_mw) * duration / inference_count
    return {
        "energy_per_inference_mj": energy,
        "energy_incremental_per_inference_mj": incremental,
        "power_idle_mean_mw": idle_mw,
        "power_busy_mean_mw": busy_mw,
        "energy_idle_sample_count": idle_window.n_samples,
        "energy_busy_sample_count": busy_window.n_samples,
        "energy_busy_duration_s": duration,
        "energy_inference_count": inference_count,
        "energy_warmup_seconds": warmup_actual,
        "energy_warmup_inferences": warmup_count,
        "energy_status": "measured with course PowerWindow; full-path and idle-adjusted values reported",
        "energy_measurement_scope": "image preprocessing + inference + full-resolution class-mask conversion",
    }


def evaluate_engine(engine_path: Path, limit: int) -> dict[str, Any]:
    precision = _precision_from_path(engine_path)
    if precision == "unknown":
        raise ValueError(f"Engine name must identify fp16 or int8 precision: {engine_path.name}")
    label = _configuration_name(engine_path)
    device = baseline._resolve_device("cuda")
    idle_system_ram = baseline._read_tegrastats_ram()
    processor = baseline._load_processor(str(engine_path))
    model = load_engine(str(engine_path), precision, str(device))
    model_loaded_system_ram = baseline._read_tegrastats_ram()
    model_loaded_cuda_memory = baseline._cuda_memory_snapshot(device)
    dataset = baseline.load_evaluation_dataset(
        baseline.DATASET_ID, split="validation", image_processor=processor,
        dataset_config=baseline.DATASET_CONFIG, num_images=limit,
    )
    if len(dataset) != limit:
        raise ValueError(f"Expected {limit} validation images, found {len(dataset)}")
    sample = dataset[0]["image"]

    energy = measure_end_to_end_energy(model, sample, processor, device)
    memory = baseline.measure_memory(
        model, sample, processor, device, height=512, width=512,
        idle_system_ram=idle_system_ram,
        model_loaded_system_ram=model_loaded_system_ram,
        model_loaded_cuda_memory=model_loaded_cuda_memory,
        inference_fn=run_engine,
    )
    latency = baseline.measure_latency(
        model, sample, processor, device, warmup_iterations=2, timed_iterations=1000,
        warmup_seconds=2.0, height=512, width=512, inference_fn=run_engine,
    )
    end_to_end_latency = measure_end_to_end_latency(model, sample, processor, device)
    accuracy = baseline.evaluate_accuracy(
        model, dataset, processor, device, num_classes=150, height=512, width=512,
        inference_fn=run_engine,
    )
    unavailable = []
    if memory.get("system_ram_idle_used_mb") is None or memory.get("system_ram_model_loaded_used_mb") is None:
        unavailable.append("system RAM snapshots")
    if memory.get("system_ram_peak_status", "").startswith("unavailable"):
        unavailable.append("system RAM peak")
    if energy.get("energy_status", "").startswith("unavailable"):
        unavailable.append("energy")
    return {
        "model_name": f"{label} {engine_path.stem} TensorRT",
        "model": str(engine_path), "checkpoint": str(engine_path),
        "architecture": "SegFormer-B2 encoder + semantic segmentation head",
        "runtime": "TensorRT", "device": str(device),
        "dataset": baseline.DATASET_ID, "dataset_config": baseline.DATASET_CONFIG,
        "split": "validation", "evaluation_set_id": f"{baseline.DATASET_ID}:validation:first-{limit}",
        "num_images": limit, "evaluation_images_requested": limit, "num_classes": 150,
        "input_height": 512, "input_width": 512, "precision": precision, "batch_size": 1,
        "warmup_seconds_configured": 2.0, "warmup_iterations_configured": 2,
        "warmup_seconds_actual": latency["warmup_seconds"], "warmup_iterations": latency["warmup_iterations"],
        "timed_iterations_configured": 1000,
        **baseline.count_parameters(model),
        **{key: value for key, value in accuracy.items()
           if key not in ("confusion_matrix", "class_iou_percent")},
        **latency, **end_to_end_latency, **memory, **energy,
        "status": "partial" if unavailable else "measured",
        "status_details": "; ".join(f"{item} unavailable" for item in unavailable)
        if unavailable else "all configured measurements completed",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", type=Path, action="append",
                        help="engine file to evaluate; repeat for several (default: all model/*.engine)")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "jetson_measurements.csv",
                        help="per-session results CSV path")
    parser.add_argument("--session", type=int, required=True, choices=(1, 2, 3),
                        help="measurement session number; run 1, 2, and 3 in separate board sessions")
    parser.add_argument("--board-hostname", required=True,
                        help="course board hostname shown by the board host, e.g. jetson-lpcv-03")
    parser.add_argument("--session-log", type=Path, default=ROOT / "results" / "measurement_sessions.csv",
                        help="audit log for board-idle confirmations and session metadata")
    parser.add_argument("--precision-report-dir", type=Path,
                        default=ROOT / "results" / "engine_precision",
                        help="directory for TensorRT per-layer precision JSON reports")
    args = parser.parse_args()
    limit = baseline.DEFAULT_EVALUATION_IMAGES

    engines = args.engine or sorted(MODEL_DIR.glob("*.engine"))
    engines = [path.expanduser().resolve() for path in engines]
    if not engines:
        parser.error(f"No .engine files found in {MODEL_DIR}; pass --engine PATH")
    missing = [str(path) for path in engines if not path.is_file()]
    if missing:
        parser.error("Engine file(s) not found: " + ", ".join(missing))
    output = args.output.expanduser()
    if not output.is_absolute():
        output = Path.cwd() / output
    output = output.resolve()
    session_log = args.session_log.expanduser().resolve()
    report_dir = args.precision_report_dir.expanduser().resolve()
    try:
        _check_csv_header(output, RESULT_COLUMNS)
        _check_csv_header(session_log, SESSION_LOG_COLUMNS)
    except ValueError as exc:
        parser.error(str(exc))
    if not any(_configuration_name(path) == "S0" for path in engines):
        parser.error("Include the S0 FP16 TensorRT baseline in every session.")
    expected_precisions = {"S0": "fp16", "S1": "int8", "S2": "fp16", "S3": "int8", "S4": "fp16"}
    stage_names = [_configuration_name(path) for path in engines]
    if any(stage not in expected_precisions for stage in stage_names):
        parser.error("Engine filenames must identify one of the planned stages S0-S4.")
    if len(stage_names) != len(set(stage_names)):
        parser.error("Pass only one engine file for each S0-S4 stage.")
    for engine_path in engines:
        stage = _configuration_name(engine_path)
        if stage in expected_precisions and _precision_from_path(engine_path) != expected_precisions[stage]:
            parser.error(f"{stage} must be an {expected_precisions[stage].upper()} engine: {engine_path.name}")
    missing_stages = sorted(set(expected_precisions) - set(stage_names))
    if missing_stages:
        parser.error(
            "Provide all S0-S4 TensorRT engines before starting measurements; missing: "
            + ", ".join(missing_stages)
        )
    if _session_already_recorded(output, args.session):
        parser.error(f"Session {args.session} already has measurements in {output}; do not repeat or overwrite it.")

    import torch
    device_name, power_mode, tensorrt_version = _check_measurement_environment(args.board_hostname)
    board_hostname = args.board_hostname.strip()
    try:
        _validate_session_plan(session_log, args.session, engines, {
            "board_hostname": board_hostname,
            "power_mode": power_mode,
            "torch_version": str(torch.__version__),
            "tensorrt_version": tensorrt_version,
        })
    except RuntimeError as exc:
        parser.error(str(exc))
    if output.is_file() and output.stat().st_size:
        with output.open(newline="", encoding="utf-8") as stream:
            old_rows = list(csv.DictReader(stream))
        other_boards = {row.get("board_hostname") for row in old_rows if row.get("board_hostname")}
        if other_boards and other_boards != {board_hostname}:
            parser.error(f"Results already contain board(s) {sorted(other_boards)}; all sessions must use {board_hostname}.")

    validation_set = baseline.load_evaluation_dataset(
        baseline.DATASET_ID, split="validation",
        dataset_config=baseline.DATASET_CONFIG, num_images=limit,
    )
    if len(validation_set) != limit:
        parser.error(f"Expected the fixed {limit}-image ADE20K validation set; found {len(validation_set)} images.")
    del validation_set
    baseline._power_window_class()

    print(f"Measurement session {args.session}; course board {board_hostname}; device {device_name}")
    print(f"Runtime: PyTorch {torch.__version__}; TensorRT {tensorrt_version}; power mode: {power_mode}")
    print(f"Protocol: batch 1, 2 s warm-up, 1,000 synchronized inferences, {limit} ADE20K validation images, "
          "10 s idle power window, at least 20 s busy power window.")
    print("Primary latency/FPS and energy include image preprocessing, TensorRT inference, and full-resolution "
          "class-mask conversion. Preprocessed-input inference latency is retained as a diagnostic.")
    if input("Confirm this is the course Orin Nano in lpcv at 15 W and only you are measuring on it. Type YES: ").strip() != "YES":
        parser.error("Measurement session cancelled; no measurements were run.")

    session_results: list[dict[str, Any]] = []
    session_records: list[dict[str, Any]] = []
    for index, engine_path in enumerate(engines, start=1):
        precision = _precision_from_path(engine_path)
        if precision == "unknown":
            parser.error(f"Engine name must include fp16 or int8: {engine_path.name}")
        label = _configuration_name(engine_path)
        precision_report = inspect_engine_precision(engine_path, report_dir)
        if precision_report.get("engine_precision_report_status") != "measured":
            raise RuntimeError(
                f"Cannot measure {engine_path.name}: TensorRT layer precision could not be verified. "
                "Rebuild with detailed profiling verbosity, then rerun this session. "
                f"Inspector status: {precision_report.get('engine_precision_report_status')}"
            )

        idle_before = _confirm_idle(
            f"\n[{index}/{len(engines)}] {label} ({precision.upper()}): confirm board idle before measurement; type YES: "
        )
        started_utc = datetime.now(timezone.utc).isoformat()
        print(f"[{index}/{len(engines)}] Measuring {engine_path.name} ({precision.upper()})", flush=True)
        result = evaluate_engine(engine_path, limit)
        idle_after = _confirm_idle(f"{label}: confirm board idle after measurement; type YES: ")
        finished_utc = datetime.now(timezone.utc).isoformat()
        result.update(precision_report)
        result.update({
            "measurement_session": args.session,
            "board_hostname": board_hostname,
            "power_mode": power_mode,
            "torch_version": str(torch.__version__),
            "tensorrt_version": tensorrt_version,
            "session_idle_confirmed_before": idle_before,
            "session_idle_confirmed_after": idle_after,
        })
        session_results.append(result)
        session_records.append({
            "session": args.session,
            "model_name": result["model_name"],
            "engine_path": str(engine_path),
            "engine_sha256": _sha256_file(engine_path),
            "board_hostname": board_hostname,
            "started_utc": started_utc,
            "finished_utc": finished_utc,
            "idle_confirmed_before": idle_before,
            "idle_confirmed_after": idle_after,
            "power_mode": power_mode,
            "torch_version": str(torch.__version__),
            "tensorrt_version": tensorrt_version,
            "engine_precision_audit_status": precision_report["engine_precision_report_status"],
        })
        print(
            f"{label}: mIoU {result['mean_iou_percent']:.2f}%; "
            f"end-to-end P50 {result['latency_e2e_p50_ms']:.3f} ms; "
            f"end-to-end P99 {result['latency_e2e_p99_ms']:.3f} ms; "
            f"end-to-end FPS {result['latency_e2e_fps']:.2f}; "
            f"energy {result['energy_per_inference_mj']:.3f} mJ/inference; "
            f"{precision.upper()} layers "
            f"{result['engine_expected_precision_layer_count']}/{result['engine_total_layer_count']} "
            f"({100 * result['engine_expected_precision_layer_fraction']:.1f}%); "
            f"INT8 layers {result['engine_int8_layer_count']}/{result['engine_total_layer_count']}",
            flush=True,
        )
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    for result in session_results:
        _save_result(result, output)
    for record in session_records:
        _append_session_log(session_log, record)
    print(f"\nSession {args.session} saved to {output}")
    if args.session == 3:
        print_session_summary(output)


if __name__ == "__main__":
    main()
