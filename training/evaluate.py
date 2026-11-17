#!/usr/bin/env python3
"""Evaluate TensorRT segmentation engines on 2,000 ADE20K validation images.

By default this evaluates every ``model/*.engine`` file and appends results to
``baseline_results.csv`` using the baseline evaluator's shared result schema.
Run on the Jetson LPCV environment where TensorRT and the course power helper
are available.
"""

from __future__ import annotations

import argparse
import gc
import sys
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
    return engine, context, runtime


def run_engine(model, inputs, device):
    """Run the loaded engine using the baseline's normalized pixel tensor."""
    import torch

    engine, context, _runtime = model
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


def evaluate_engine(engine_path: Path, limit: int, output_csv: Path) -> dict[str, Any]:
    precision = _precision_from_path(engine_path)
    config = baseline.EvaluationConfig(
        model_name=f"{engine_path.stem} TensorRT",
        model_path=str(engine_path),
        checkpoint=str(engine_path),
        architecture="SegFormer-B2 encoder + semantic segmentation head",
        precision=precision,
        runtime="TensorRT",
        device="cuda",
        batch_size=1,
        input_height=512,
        input_width=512,
        warmup_iterations=2,
        warmup_seconds=2.0,
        timed_iterations=1000,
        evaluation_images=limit,
        num_classes=150,
        output_csv=str(output_csv),
    )
    return baseline.run_evaluation(
        config,
        model_loader=load_engine,
        inference_fn=run_engine,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", type=Path, action="append",
                        help="engine file to evaluate; repeat for several (default: all model/*.engine)")
    parser.add_argument("--limit", type=int, default=2000,
                        help="number of ordered ADE20K validation images (default: 2000)")
    parser.add_argument("--output", type=Path, default=ROOT / "baseline_results.csv",
                        help="CSV results path (default: baseline_results.csv)")
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit must be positive")

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

    for index, engine_path in enumerate(engines, start=1):
        print(f"\n[{index}/{len(engines)}] Evaluating {engine_path}", flush=True)
        try:
            result = evaluate_engine(engine_path, args.limit, output)
            print(
                f"{engine_path.name}: {result['status']}; "
                f"{result['num_images']} images; "
                f"mIoU {result['mean_iou_percent']:.2f}%; "
                f"pixel accuracy {result['pixel_accuracy_percent']:.2f}%",
                flush=True,
            )
        finally:
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass


if __name__ == "__main__":
    main()
