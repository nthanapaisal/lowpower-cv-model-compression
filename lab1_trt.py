#!/usr/bin/env python3
"""CS 6301 Lab 1 TensorRT helper.

Target environment: course lpcv container, TensorRT 10.11.0.33.
The build path uses TensorRT's implicit post-training calibration API as
specified by the Lab 1 handout. TensorRT 10.x deprecates this API, but it is
still available in the course's pinned version.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import tensorrt as trt
import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode

DATA_ROOT = Path("/course-data/imagewoof2-320")
SPLIT_ROOT = Path("/course-data/lab1/splits")
INPUT_SHAPE = (1, 3, 224, 224)
OUTPUT_SHAPE = (1, 10)
LOGGER = trt.Logger(trt.Logger.WARNING)


def eval_transform():
    """Shared ImageNet inference preprocessing for evaluation/calibration."""
    return transforms.Compose([
        transforms.Resize(256, interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225)),
    ])


def _class_names() -> list[str]:
    root = DATA_ROOT / "train"
    if not root.is_dir():
        raise FileNotFoundError(f"Imagewoof training directory not found: {root}")
    return sorted(p.name for p in root.iterdir() if p.is_dir())


def read_split(name: str) -> list[tuple[str, int]]:
    """Return ordered (absolute image path, integer label) pairs.

    fit/tune/calib read the instructor-supplied relative-path lists. val is
    enumerated by sorted class-folder name and then sorted file name.
    """
    allowed = {"fit", "tune", "calib", "val"}
    if name not in allowed:
        raise ValueError(f"Unknown split {name!r}; expected one of {sorted(allowed)}")
    classes = _class_names()
    class_to_label = {folder: i for i, folder in enumerate(classes)}
    pairs: list[tuple[str, int]] = []
    if name == "val":
        root = DATA_ROOT / "val"
        if not root.is_dir():
            raise FileNotFoundError(f"Imagewoof validation directory not found: {root}")
        for class_name in classes:
            class_dir = root / class_name
            if not class_dir.is_dir():
                raise FileNotFoundError(f"Validation class directory not found: {class_dir}")
            for path in sorted(p for p in class_dir.iterdir()
                               if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}):
                pairs.append((str(path), class_to_label[class_name]))
        return pairs

    list_path = SPLIT_ROOT / f"{name}.txt"
    if not list_path.is_file():
        raise FileNotFoundError(f"Instructor split list not found: {list_path}")
    with list_path.open("r", encoding="utf-8") as f:
        for line_number, raw in enumerate(f, start=1):
            rel = raw.strip()
            if not rel or rel.startswith("#"):
                continue
            rel_path = Path(rel)
            # Split paths are relative to DATA_ROOT; reject paths that escape it.
            if rel_path.is_absolute() or ".." in rel_path.parts:
                raise ValueError(f"Invalid relative path in {list_path}:{line_number}: {rel!r}")
            if len(rel_path.parts) < 3 or rel_path.parts[0] != "train":
                raise ValueError(f"Unexpected split path in {list_path}:{line_number}: {rel!r}")
            class_name = rel_path.parts[1]
            if class_name not in class_to_label:
                raise ValueError(f"Unknown class folder in {list_path}:{line_number}: {class_name!r}")
            absolute = DATA_ROOT / rel_path
            if not absolute.is_file():
                raise FileNotFoundError(f"Image listed in {list_path}:{line_number} not found: {absolute}")
            pairs.append((str(absolute), class_to_label[class_name]))
    return pairs


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _require_tensorrt() -> None:
    version = getattr(trt, "__version__", "unknown")
    print(f"TensorRT version: {version}")
    if not version.startswith("10.11."):
        print("WARNING: this helper was prepared for TensorRT 10.11.x; validate API behavior before using it.", file=sys.stderr)


def _check_onnx(path: Path, network: Any) -> None:
    if network.num_inputs != 1:
        raise ValueError(f"Expected exactly one ONNX input, found {network.num_inputs}")
    if network.num_outputs != 1:
        raise ValueError(f"Expected exactly one ONNX output, found {network.num_outputs}")
    input_tensor = network.get_input(0)
    output_tensor = network.get_output(0)
    in_shape = tuple(int(x) for x in input_tensor.shape)
    out_shape = tuple(int(x) for x in output_tensor.shape)
    if in_shape != INPUT_SHAPE:
        raise ValueError(f"ONNX input must be fixed shape {INPUT_SHAPE}; got {in_shape}. Dynamic dimensions are not supported.")
    if out_shape != OUTPUT_SHAPE:
        raise ValueError(f"ONNX output must be fixed shape {OUTPUT_SHAPE}; got {out_shape}. Expected 10 class logits.")


class _ImagewoofCalibrator(trt.IInt8EntropyCalibrator2):
    """Batch-1 entropy calibrator using the handout's shared transform."""
    def __init__(self, image_pairs: list[tuple[str, int]], cache_path: Path,
                 recalibrate: bool = False):
        super().__init__()
        self.image_pairs = image_pairs
        self.cache_path = cache_path
        self.recalibrate = recalibrate
        self.index = 0
        self._device_batch = None
        self.cache_was_read = False
        self.cache_was_written = False
        self.transform = eval_transform()

    def get_batch_size(self) -> int:
        return 1

    def get_batch(self, names):
        if self.index >= len(self.image_pairs):
            return None
        image_path, _label = self.image_pairs[self.index]
        self.index += 1
        with Image.open(image_path) as image:
            tensor = self.transform(image.convert("RGB")).unsqueeze(0).contiguous()
        self._device_batch = tensor.to(device="cuda", dtype=torch.float32)
        # Ensure the pointer's storage is populated before TensorRT consumes it.
        torch.cuda.synchronize()
        return [int(self._device_batch.data_ptr())]

    def read_calibration_cache(self):
        if self.recalibrate or not self.cache_path.is_file():
            return None
        data = self.cache_path.read_bytes()
        if not data:
            return None
        self.cache_was_read = True
        return data

    def write_calibration_cache(self, cache):
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.cache_path.with_suffix(self.cache_path.suffix + ".tmp")
        tmp.write_bytes(cache)
        tmp.replace(self.cache_path)
        self.cache_was_written = True


def _build(args) -> None:
    onnx_path = Path(args.onnx).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()
    if not onnx_path.is_file():
        raise FileNotFoundError(f"ONNX file not found: {onnx_path}")
    if args.precision not in {"fp16", "int8"}:
        raise ValueError("--precision must be fp16 or int8")
    _require_tensorrt()
    onnx_hash = _sha256(onnx_path)
    print(f"ONNX: {onnx_path}")
    print(f"ONNX SHA-256: {onnx_hash}")

    builder = trt.Builder(LOGGER)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, LOGGER)
    if not parser.parse(onnx_path.read_bytes()):
        errors = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise RuntimeError(f"TensorRT failed to parse ONNX:\n{errors}")
    _check_onnx(onnx_path, network)

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
    # Detailed profiling data enables engine-inspector precision reporting.
    try:
        config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    except Exception as exc:
        print(f"WARNING: could not enable detailed layer inspection: {exc}", file=sys.stderr)

    calibrator = None
    if args.precision == "fp16":
        config.set_flag(trt.BuilderFlag.FP16)
    else:
        config.set_flag(trt.BuilderFlag.INT8)
        config.set_flag(trt.BuilderFlag.FP16)  # permitted fallback for non-INT8 layers
        # Full ONNX content hash prevents stale calibration ranges across models.
        cache_path = onnx_path.with_name(f"{onnx_path.name}.{onnx_hash}.calib.cache")
        calib_pairs = read_split("calib")
        if len(calib_pairs) != 500:
            raise RuntimeError(f"Expected exactly 500 calibration images, found {len(calib_pairs)}")
        calibrator = _ImagewoofCalibrator(calib_pairs, cache_path, args.recalibrate)
        config.int8_calibrator = calibrator
        print("Calibrator: IInt8EntropyCalibrator2")
        print(f"Calibration images: {len(calib_pairs)} (batch size 1)")
        print(f"Calibration cache: {cache_path}")
        print(f"Calibration cache mode: {'force fresh calibration' if args.recalibrate else ('existing cache offered for reuse' if cache_path.is_file() else 'no cache; calibration required')}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    print("Building TensorRT engine; use an idle measurement board.", flush=True)
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT build_serialized_network returned None")
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp_path.write_bytes(bytes(serialized))
    tmp_path.replace(out_path)
    size_mib = out_path.stat().st_size / (1024 * 1024)
    print(f"Engine: {out_path}")
    print(f"Engine size: {size_mib:.3f} MiB ({out_path.stat().st_size} bytes)")
    print(f"Precision requested: {args.precision.upper()}")
    if calibrator is not None:
        if calibrator.cache_was_read:
            status = "reused existing calibration cache"
        elif calibrator.cache_was_written:
            status = "ran calibration and wrote a new cache"
        else:
            status = "TensorRT did not report cache use/write; inspect build log and engine"
        print(f"Calibration result: {status}")
        print(f"Calibration cache exists: {cache_path.is_file()}")


def _load_engine(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"TensorRT engine not found: {path}")
    runtime = trt.Runtime(LOGGER)
    engine = runtime.deserialize_cuda_engine(path.read_bytes())
    if engine is None:
        raise RuntimeError(f"TensorRT could not deserialize engine: {path}")
    return runtime, engine


def _get_io_names(engine) -> tuple[str, str]:
    inputs, outputs = [], []
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        mode = engine.get_tensor_mode(name)
        if mode == trt.TensorIOMode.INPUT:
            inputs.append(name)
        elif mode == trt.TensorIOMode.OUTPUT:
            outputs.append(name)
    if len(inputs) != 1 or len(outputs) != 1:
        raise RuntimeError(f"Expected one input and one output tensor; found inputs={inputs}, outputs={outputs}")
    return inputs[0], outputs[0]


class Engine:
    """Run a fixed-shape TensorRT engine on a CUDA FP32 PyTorch tensor."""
    def __init__(self, path):
        self.path = str(path)
        self._runtime, self.engine = _load_engine(Path(path).expanduser())
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError(f"Could not create execution context for {path}")
        self.input_name, self.output_name = _get_io_names(self.engine)
        in_shape = tuple(int(x) for x in self.engine.get_tensor_shape(self.input_name))
        out_shape = tuple(int(x) for x in self.engine.get_tensor_shape(self.output_name))
        if in_shape != INPUT_SHAPE or out_shape != OUTPUT_SHAPE:
            raise ValueError(f"Engine I/O shape mismatch: input {in_shape}, output {out_shape}; expected {INPUT_SHAPE} -> {OUTPUT_SHAPE}")
        if self.engine.get_tensor_dtype(self.input_name) != trt.float32:
            raise TypeError(f"Engine input must be FP32, got {self.engine.get_tensor_dtype(self.input_name)}")
        if self.engine.get_tensor_dtype(self.output_name) != trt.float32:
            raise TypeError(f"Engine output must be FP32, got {self.engine.get_tensor_dtype(self.output_name)}")
        self._stream = torch.cuda.Stream()
        self._input_ready = torch.cuda.Event()

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if not isinstance(x, torch.Tensor):
            raise TypeError("Engine input must be a torch.Tensor")
        if tuple(x.shape) != INPUT_SHAPE:
            raise ValueError(f"Expected input shape {INPUT_SHAPE}, got {tuple(x.shape)}")
        if x.device.type != "cuda":
            raise ValueError("Engine input must be a CUDA tensor")
        if x.dtype != torch.float32:
            raise TypeError(f"Engine input must be torch.float32, got {x.dtype}")
        if not x.is_contiguous():
            x = x.contiguous()

        current = torch.cuda.current_stream(device=x.device)
        output = torch.empty(OUTPUT_SHAPE, dtype=torch.float32, device=x.device)
        input_ready = torch.cuda.Event()
        output_ready = torch.cuda.Event()
        input_ready.record(current)
        self._stream.wait_event(input_ready)
        # Keep allocator-managed input/output storage alive until TensorRT has
        # finished using it on its own stream.
        x.record_stream(self._stream)
        output.record_stream(self._stream)
        self.context.set_tensor_address(self.input_name, int(x.data_ptr()))
        self.context.set_tensor_address(self.output_name, int(output.data_ptr()))
        ok = self.context.execute_async_v3(stream_handle=self._stream.cuda_stream)
        if not ok:
            raise RuntimeError("TensorRT execute_async_v3 failed")
        output_ready.record(self._stream)
        current.wait_event(output_ready)
        return output


def _precision_from_formats(value: Any) -> str:
    """Classify output datatype without confusing names/layouts with precision."""
    if isinstance(value, dict):
        # Only inspect datatype/format fields and nested output records; never
        # infer a precision from an output tensor's name or arbitrary metadata.
        for key in ("Format/Datatype", "DataType", "Datatype", "dtype", "Type", "Format"):
            if key in value:
                found = _precision_from_formats(value[key])
                if found != "UNKNOWN":
                    return found
        for child in value.values():
            if isinstance(child, (dict, list)):
                found = _precision_from_formats(child)
                if found != "UNKNOWN":
                    return found
        return "UNKNOWN"
    if isinstance(value, list):
        for child in value:
            found = _precision_from_formats(child)
            if found != "UNKNOWN":
                return found
        return "UNKNOWN"
    if isinstance(value, str):
        s = value.upper()
        # TensorRT may append a layout after a colon, e.g. "Int8:CHW32".
        if re.search(r"\bINT\s*8\b", s):
            return "INT8"
        if re.search(r"\b(?:HALF|FP16|FLOAT16)\b", s):
            return "FP16"
        if re.search(r"\b(?:FLOAT|FLOAT32|FP32)\b", s):
            return "FP32"
        if re.search(r"\bINT\s*32\b", s):
            return "INT32"
        if re.search(r"\bINT\s*64\b", s):
            return "INT64"
        if re.search(r"\bBOOL\b", s):
            return "BOOL"
        if re.search(r"\bUINT\s*8\b", s):
            return "UINT8"
        if re.search(r"\bUINT\s*16\b", s):
            return "UINT16"
        if re.search(r"\bUINT\s*32\b", s):
            return "UINT32"
        if re.search(r"\bUINT\s*64\b", s):
            return "UINT64"
    return "UNKNOWN"


def _layers(args) -> None:
    engine_path = Path(args.engine).expanduser().resolve()
    runtime, engine = _load_engine(engine_path)
    inspector = engine.create_engine_inspector()
    # TensorRT's inspector can return empty records if this context is only a
    # temporary expression; retain it for the full inspection loop.
    context = engine.create_execution_context()
    if context is None:
        raise RuntimeError("TensorRT could not create an inspection execution context")
    inspector.execution_context = context
    rows = []
    for index in range(engine.num_layers):
        raw = inspector.get_layer_information(index, trt.LayerInformationFormat.JSON)
        try:
            info = json.loads(raw)
        except Exception:
            info = {"raw": raw}
        name = str(info.get("Name", info.get("name", f"layer_{index}")))
        layer_type = str(info.get("LayerType", info.get("Type", info.get("type", "UNKNOWN"))))
        outputs = info.get("Outputs", info.get("outputs", []))
        inputs = info.get("Inputs", info.get("inputs", []))
        output_precision = _precision_from_formats(outputs)
        is_reformat = ("REFORMAT" in name.upper() or "REFORMAT" in layer_type.upper()
                       or "SHUFFLE" in layer_type.upper() and "REFORMAT" in name.upper())
        rows.append({
            "layer_index": index,
            "name": name,
            "type": layer_type,
            "is_reformat": is_reformat,
            "output_precision": output_precision,
            "inputs_json": json.dumps(inputs, sort_keys=True),
            "outputs_json": json.dumps(outputs, sort_keys=True),
            "inspector_json": json.dumps(info, sort_keys=True),
        })

    csv_path = Path(args.csv).expanduser().resolve() if args.csv else engine_path.with_suffix(".layers.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    columns = ["layer_index", "name", "type", "is_reformat", "output_precision", "inputs_json", "outputs_json", "inspector_json"]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    reformat_count = sum(r["is_reformat"] for r in rows)
    non_reformat = [r for r in rows if not r["is_reformat"]]
    # Keep known non-floating output types distinct from parser failures. The
    # INT8 fraction denominator remains every non-reformatting layer, as the
    # handout specifies, including any layer with an auxiliary integer output.
    precision_categories = ("INT8", "FP16", "FP32", "INT32", "INT64", "BOOL",
                            "UINT8", "UINT16", "UINT32", "UINT64", "UNKNOWN")
    counts = {p: sum(r["output_precision"] == p for r in non_reformat)
              for p in precision_categories}
    int8_fraction = (100.0 * counts["INT8"] / len(non_reformat)) if non_reformat else 0.0
    print(f"Engine: {engine_path}")
    print(f"Engine size: {engine_path.stat().st_size / (1024 * 1024):.3f} MiB ({engine_path.stat().st_size} bytes)")
    print(f"Layer CSV: {csv_path}")
    print(f"Total layers: {len(rows)}")
    print(f"Reformatting layers: {reformat_count}")
    print(f"Non-reformatting layers with INT8 output: {counts['INT8']}")
    print(f"Non-reformatting layers with FP16 output: {counts['FP16']}")
    print(f"Non-reformatting layers with FP32 output: {counts['FP32']}")
    other_precision_counts = {p: counts[p] for p in precision_categories
                              if p not in {"INT8", "FP16", "FP32", "UNKNOWN"} and counts[p]}
    if other_precision_counts:
        details = ", ".join(f"{p}={n}" for p, n in sorted(other_precision_counts.items()))
        print(f"Non-reformatting layers with other known output precision: {details}")
    print(f"Non-reformatting layers with unknown output precision: {counts['UNKNOWN']}")
    print(f"Fraction of non-reformatting layers in INT8: {int8_fraction:.2f}%")
    if counts["UNKNOWN"]:
        print("WARNING: some layer output datatypes could not be parsed from TensorRT inspector output; inspect the CSV before reporting these counts.", file=sys.stderr)
    non_int8_types: dict[str, dict[str, int]] = {}
    for row in non_reformat:
        precision = row["output_precision"]
        if precision != "INT8":
            layer_type = row["type"]
            bucket = non_int8_types.setdefault(layer_type, {})
            bucket[precision] = bucket.get(precision, 0) + 1
    print("Non-INT8 non-reformatting layers grouped by layer type:")
    if non_int8_types:
        for layer_type, precision_counts in sorted(non_int8_types.items()):
            count = sum(precision_counts.values())
            detail = ", ".join(f"{p}={n}" for p, n in sorted(precision_counts.items()))
            print(f"  {layer_type}: {count} ({detail})")
    else:
        print("  (none)")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build", help="build a fixed-batch TensorRT engine from ONNX")
    build.add_argument("--onnx", required=True, help="input fixed-shape ONNX model")
    build.add_argument("--precision", required=True, choices=("fp16", "int8"))
    build.add_argument("--out", required=True, help="output engine path")
    build.add_argument("--recalibrate", action="store_true", help="ignore any existing INT8 calibration cache and recalibrate")
    build.set_defaults(func=_build)

    layers = sub.add_parser("layers", help="write layer-precision CSV and summary")
    layers.add_argument("--engine", required=True, help="serialized TensorRT engine")
    layers.add_argument("--csv", default=None, help="optional output CSV path")
    layers.set_defaults(func=_layers)

    args = parser.parse_args(argv)
    try:
        args.func(args)
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
