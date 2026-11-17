#!/usr/bin/env python3
"""Evaluate SegFormer models on the Jetson with the baseline protocol (S0-S4).

Run from the repository root, inside lpcv, on a measurement board:

    # accuracy only: top-1 pixel accuracy, correct/total pixels, mIoU (2000 val images)
    python3 training/evaluate.py training/model/segformer_b2_pruned40_ft.onnx --accuracy-only

    # everything the baseline reported: energy (power_window.py), memory, P50/P99, FPS, accuracy
    python3 training/evaluate.py training/model/segformer_b2_pruned40_ft.onnx

Models:
    *.onnx            built into a TensorRT engine (--precision fp16 by default, or fp32);
                      the engine is cached next to the ONNX file as <name>.<precision>.engine
    *.torchscript.pt  run in PyTorch FP32, the runtime of the Milestone 1 baseline

Everything is measured by baseline/baseline.py (jetson/baseline.py on older
branches): the same preprocessing, the first 2000 validation images in sorted
order from ./ade20k-scene-parse-150, the same mIoU code, and energy from the
course-supplied power_window.py (/course-data/lab1). INT8 engines need a
calibrator and are built by quantization.py, not here.

"Top-1" for segmentation is per pixel: the share of labeled pixels whose
highest-scoring class is the true class, i.e. correct pixels / labeled pixels.
Pixels labeled 0 ("other") are not counted, as in the official ADE20K metric.
"""

from __future__ import annotations

import argparse
import csv
import gc
import os
import sys
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent.parent
for folder in (REPO_DIR / "baseline", REPO_DIR / "jetson", Path("/course-data/lab1")):
    if folder.is_dir():
        sys.path.append(str(folder))
import baseline as bl  # noqa: E402


# --------------------------------------------------------------------------
# TensorRT (input and output types are read from the engine, so FP32- and
# FP16-input ONNX files both work)
# --------------------------------------------------------------------------

def load_tensorrt(model_path: str, precision: str, device) -> tuple:
    import tensorrt as trt
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    onnx_path = Path(model_path)
    engine_path = onnx_path.with_suffix(f".{precision}.engine")
    if engine_path.is_file() and engine_path.stat().st_mtime >= onnx_path.stat().st_mtime:
        print(f"Using cached engine {engine_path}", flush=True)
        engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
    else:
        print(f"Building a TensorRT {precision.upper()} engine from {onnx_path} (this takes a few minutes)", flush=True)
        builder = trt.Builder(logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, logger)
        if not parser.parse(onnx_path.read_bytes()):
            errors = "; ".join(str(parser.get_error(i)) for i in range(parser.num_errors))
            raise RuntimeError(f"Could not read ONNX model {onnx_path}: {errors}")
        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
        if precision == "fp16":
            config.set_flag(trt.BuilderFlag.FP16)
        serialized = builder.build_serialized_network(network, config)
        if serialized is None:
            raise RuntimeError("TensorRT could not build the engine")
        engine_path.write_bytes(bytes(serialized))
        engine = runtime.deserialize_cuda_engine(serialized)
    return engine, engine.create_execution_context(), runtime


def run_tensorrt(model, inputs, device):
    import tensorrt as trt
    import torch
    engine, context, _ = model
    names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
    input_name = next(n for n in names if engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT)
    output_name = next(n for n in names if engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT)
    types = {trt.DataType.FLOAT: torch.float32, trt.DataType.HALF: torch.float16}
    pixels = inputs["pixel_values"].to(device=device, dtype=types[engine.get_tensor_dtype(input_name)]).contiguous()
    context.set_input_shape(input_name, tuple(pixels.shape))
    logits = torch.empty(tuple(context.get_tensor_shape(output_name)), device=device,
                         dtype=types[engine.get_tensor_dtype(output_name)])
    context.set_tensor_address(input_name, pixels.data_ptr())
    context.set_tensor_address(output_name, logits.data_ptr())
    context.execute_async_v3(torch.cuda.current_stream(device).cuda_stream)
    return logits


# --------------------------------------------------------------------------

def _runtime(path: Path, precision: str):
    """(runtime label, precision label, model_loader, inference_fn)"""
    if path.suffix == ".onnx":
        return "TensorRT", precision, load_tensorrt, run_tensorrt
    if path.name.endswith(".torchscript.pt"):
        return "PyTorch", "fp32", None, None
    raise SystemExit(f"Unsupported model file: {path} (use .onnx or .torchscript.pt)")


def _processor_loader(preprocessing: Path | None):
    if preprocessing is None:
        return None
    return lambda _model_path: bl._load_processor(str(preprocessing.parent / "model.placeholder"))


def accuracy(path: Path, precision: str, limit: int, preprocessing: Path | None) -> dict:
    runtime, precision, loader, inference_fn = _runtime(path, precision)
    device = bl._resolve_device("cuda")
    processor = (_processor_loader(preprocessing) or bl._load_processor)(str(path))
    model = bl.load_model(str(path), precision, str(device), loader=loader)
    dataset = bl.load_evaluation_dataset(bl.DATASET_ID, num_images=limit)
    result = bl.evaluate_accuracy(model, dataset, processor, device, inference_fn=inference_fn)
    confusion = result["confusion_matrix"]
    correct, total = int(confusion.trace()), int(confusion.sum())
    del model
    gc.collect()
    return {"model": path.name, "runtime": runtime, "precision": precision, "num_images": result["num_images"],
            "mean_iou_percent": round(result["mean_iou_percent"], 3),
            "top1_pixel_accuracy_percent": round(100.0 * correct / total, 3),
            "correct_pixels": correct, "labeled_pixels": total}


def full(path: Path, precision: str, limit: int, preprocessing: Path | None, output: Path) -> dict:
    runtime, precision, loader, inference_fn = _runtime(path, precision)
    config = bl.EvaluationConfig(
        model_name=path.name, model_path=str(path), checkpoint=path.name,
        architecture="MiT-B2 encoder + MLP decoder" + (" (pruned)" if "pruned" in path.name else ""),
        precision=precision, runtime=runtime, device="cuda", batch_size=1, input_height=512, input_width=512,
        warmup_seconds=2.0, timed_iterations=1000, evaluation_images=limit, output_csv=str(output))
    results = bl.run_evaluation(config, model_loader=loader, processor_loader=_processor_loader(preprocessing),
                                inference_fn=inference_fn)
    total = int(results["valid_pixels"])
    correct = int(round(results["pixel_accuracy_percent"] / 100.0 * total))
    return {"model": path.name, "runtime": runtime, "precision": precision, "num_images": results["num_images"],
            "mean_iou_percent": round(results["mean_iou_percent"], 3),
            "top1_pixel_accuracy_percent": round(results["pixel_accuracy_percent"], 3),
            "correct_pixels": correct, "labeled_pixels": total,
            "latency_p50_ms": round(results["latency_p50_ms"], 3), "latency_p99_ms": round(results["latency_p99_ms"], 3),
            "fps": round(results["latency_fps"], 3), "energy_mj": results.get("energy_per_inference_mj"),
            "status": results.get("status")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("models", nargs="+", type=Path, help=".onnx or .torchscript.pt files")
    parser.add_argument("--precision", choices=("fp16", "fp32"), default="fp16",
                        help="TensorRT engine precision for .onnx files (default fp16)")
    parser.add_argument("--accuracy-only", action="store_true", help="skip energy, memory and latency")
    parser.add_argument("--limit", type=int, default=bl.DEFAULT_EVALUATION_IMAGES, help="validation images")
    parser.add_argument("--preprocessing", type=Path, default=None,
                        help="preprocessing.json, if it is not next to the model file")
    parser.add_argument("--output", type=Path, default=REPO_DIR / "evaluation_results.csv",
                        help="full results, same columns as baseline_results.csv")
    parser.add_argument("--summary", type=Path, default=REPO_DIR / "evaluation_summary.csv")
    args = parser.parse_args()
    if os.environ.get("LPCV_POWER_MODE", "").upper().replace(" ", "") not in ("15W", "15"):
        print("Warning: LPCV_POWER_MODE is not 15W. Run inside lpcv on a measurement board.", flush=True)

    rows = []
    for path in args.models:
        if not path.is_file():
            raise SystemExit(f"{path} not found")
        row = (accuracy(path, args.precision, args.limit, args.preprocessing) if args.accuracy_only
               else full(path, args.precision, args.limit, args.preprocessing, args.output))
        rows.append(row)
        print(f"\n{row['model']} ({row['runtime']} {row['precision'].upper()}, {row['num_images']} validation images)")
        print(f"  top-1 pixel accuracy: {row['top1_pixel_accuracy_percent']:.2f}%  "
              f"({row['correct_pixels']:,} correct / {row['labeled_pixels']:,} labeled pixels)")
        print(f"  mIoU: {row['mean_iou_percent']:.2f}%")
        if "latency_p99_ms" in row:
            print(f"  P50 {row['latency_p50_ms']:.2f} ms, P99 {row['latency_p99_ms']:.2f} ms, {row['fps']:.2f} FPS, "
                  f"energy {row['energy_mj']} mJ per inference ({row['status']})", flush=True)

    fields = ["model", "runtime", "precision", "num_images", "mean_iou_percent", "top1_pixel_accuracy_percent",
              "correct_pixels", "labeled_pixels", "latency_p50_ms", "latency_p99_ms", "fps", "energy_mj", "status"]
    exists = args.summary.is_file() and args.summary.stat().st_size > 0
    with args.summary.open("a" if exists else "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerows(rows)
    print(f"\nSummary appended to {args.summary}" + ("" if args.accuracy_only else f"; full results in {args.output}"))


if __name__ == "__main__":
    main()
