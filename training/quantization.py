#!/usr/bin/env python3
"""Build TensorRT engines for the SegFormer-B2 S0-S4 comparison.

Run inside the course ``lpcv`` container. S0/S1 share the FP32 baseline ONNX;
S2/S3 share the pruned and fine-tuned ONNX; S4 uses the distilled ONNX.
The course helper is used for per-layer inspection after each engine build.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

from PIL import Image
import tensorrt as trt
import torch

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "model"
COURSE_HELPER_DIR = Path("/course-data/lab1")
INPUT_SHAPE = (1, 3, 512, 512)
OUTPUT_SHAPE = (1, 150, 128, 128)
STAGES = {
    "S0": ("s0", "fp16"),
    "S1": ("s1", "int8"),
    "S2": ("s2", "fp16"),
    "S3": ("s3", "int8"),
    "S4": ("s4", "fp16"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def calibration_images(data_root: Path, count: int, seed: int) -> list[Path]:
    image_root = data_root / "images" / "training"
    if not image_root.is_dir():
        raise FileNotFoundError(f"ADE20K training images not found: {image_root}")
    images = sorted(path for path in image_root.rglob("*")
                    if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if count < 1 or count > len(images):
        raise ValueError(f"Requested {count} calibration images; found {len(images)}")
    random.Random(seed).shuffle(images)
    return images[:count]


def validate_onnx(onnx_path: Path):
    if not onnx_path.is_file():
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(onnx_path.read_bytes()):
        errors = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise RuntimeError(f"TensorRT could not parse {onnx_path}:\n{errors}")
    if network.num_inputs != 1 or network.num_outputs != 1:
        raise ValueError(f"Expected one ONNX input and output; found {network.num_inputs} and {network.num_outputs}")
    input_tensor, output_tensor = network.get_input(0), network.get_output(0)
    if tuple(input_tensor.shape) != INPUT_SHAPE or tuple(output_tensor.shape) != OUTPUT_SHAPE:
        raise ValueError(
            f"Expected ONNX shapes {INPUT_SHAPE} -> {OUTPUT_SHAPE}; "
            f"found {tuple(input_tensor.shape)} -> {tuple(output_tensor.shape)}"
        )
    if input_tensor.dtype not in (trt.float32, trt.float16):
        raise TypeError(f"Expected FP32 or FP16 ONNX input; found {input_tensor.dtype}")
    return builder, network


class ADECalibrator(trt.IInt8EntropyCalibrator2):
    """Batch-one TensorRT calibrator using the saved SegFormer preprocessing."""

    def __init__(self, images: list[Path], cache_path: Path, preprocessing_path: Path,
                 recalibrate: bool):
        super().__init__()
        if preprocessing_path.name != "preprocessing.json":
            raise ValueError("Preprocessing file must be named preprocessing.json")
        sys.path.insert(0, str(ROOT / "baseline"))
        import baseline

        self.images = images
        self.cache_path = cache_path
        self.recalibrate = recalibrate
        self.index = 0
        self.device_batch = None
        # TensorRT's legacy INT8 calibrators require FP32 calibration batches,
        # even when the ONNX model uses FP16 I/O.
        self.input_dtype = torch.float32
        self.processor = baseline._load_processor(str(preprocessing_path))
        self.preprocess_image = baseline.preprocess_image
        self.cache_reused = False
        self.cache_written = False

    def get_batch_size(self) -> int:
        return 1

    def get_batch(self, names):
        if self.index >= len(self.images):
            return None
        if len(names) != 1:
            raise ValueError(f"Expected one TensorRT input name, got {names}")
        with Image.open(self.images[self.index]) as image_file:
            inputs = self.preprocess_image(image_file.convert("RGB"), self.processor, height=512, width=512)
        self.index += 1
        self.device_batch = inputs["pixel_values"].to(device="cuda", dtype=self.input_dtype).contiguous()
        if tuple(self.device_batch.shape) != INPUT_SHAPE:
            raise ValueError(f"Calibration input has shape {tuple(self.device_batch.shape)}, expected {INPUT_SHAPE}")
        torch.cuda.synchronize()
        return [int(self.device_batch.data_ptr())]

    def read_calibration_cache(self):
        if self.recalibrate or not self.cache_path.is_file():
            return None
        cache = self.cache_path.read_bytes()
        self.cache_reused = bool(cache)
        return cache or None

    def write_calibration_cache(self, cache):
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.cache_path.with_suffix(self.cache_path.suffix + ".tmp")
        temporary.write_bytes(cache)
        temporary.replace(self.cache_path)
        self.cache_written = True


def build_engine(stage: str, onnx_path: Path, precision: str, engine_path: Path,
                 data_root: Path, preprocessing_path: Path, count: int,
                 seed: int, recalibrate: bool) -> dict[str, Any]:
    builder, network = validate_onnx(onnx_path)
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
    config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED

    images: list[Path] = []
    calibrator = None
    onnx_hash = sha256(onnx_path)
    preprocessing_hash = None
    cache_path = None
    if precision == "fp16":
        config.set_flag(trt.BuilderFlag.FP16)
    elif precision == "int8":
        images = calibration_images(data_root, count, seed)
        if not preprocessing_path.is_file():
            raise FileNotFoundError(f"Preprocessing metadata not found: {preprocessing_path}")
        preprocessing_hash = sha256(preprocessing_path)
        image_list_hash = hashlib.sha256(
            "\n".join(path.relative_to(data_root).as_posix() for path in images).encode("utf-8")
        ).hexdigest()
        cache_path = onnx_path.with_name(
            f"{onnx_path.name}.{onnx_hash[:12]}.{preprocessing_hash[:12]}.{image_list_hash[:12]}.calib.cache"
        )
        calibrator = ADECalibrator(images, cache_path, preprocessing_path, recalibrate)
        config.set_flag(trt.BuilderFlag.INT8)
        config.set_flag(trt.BuilderFlag.FP16)
        config.int8_calibrator = calibrator
    else:
        raise ValueError(f"Unsupported precision: {precision}")

    engine_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[{stage}] building {precision.upper()} TensorRT engine from {onnx_path}", flush=True)
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError(f"TensorRT failed to build {stage} engine")
    temporary = engine_path.with_suffix(engine_path.suffix + ".tmp")
    temporary.write_bytes(bytes(serialized))
    temporary.replace(engine_path)
    build_info = {
        "stage": stage,
        "source_onnx": str(onnx_path),
        "source_onnx_sha256": onnx_hash,
        "engine": str(engine_path),
        "precision_requested": precision,
        "input_shape": INPUT_SHAPE,
        "output_shape": OUTPUT_SHAPE,
        "calibration_count": len(images),
        "calibration_images": [str(path) for path in images],
        "calibration_seed": seed if images else None,
        "calibration_cache": str(cache_path) if cache_path else None,
        "calibration_cache_reused": calibrator.cache_reused if calibrator else False,
        "calibration_cache_written": calibrator.cache_written if calibrator else False,
        "preprocessing_sha256": preprocessing_hash,
    }
    info_path = engine_path.with_suffix(".build.json")
    info_path.write_text(json.dumps(build_info, indent=2) + "\n", encoding="utf-8")
    print(f"[{stage}] saved {engine_path} ({engine_path.stat().st_size / 1048576:.2f} MiB)", flush=True)
    return build_info


def inspect_engine(engine_path: Path) -> Path:
    helper = COURSE_HELPER_DIR / "lab1_trt.py"
    if not helper.is_file():
        raise FileNotFoundError(f"Course TensorRT layer inspector not found: {helper}")
    report_path = engine_path.with_suffix(".layers.csv")
    subprocess.run(
        [sys.executable, str(helper), "layers", "--engine", str(engine_path), "--csv", str(report_path)],
        check=True,
    )
    return report_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--s0-onnx", type=Path, default=MODEL_DIR / "segformer_b2.onnx")
    parser.add_argument("--s1-onnx", type=Path, default=MODEL_DIR / "segformer_b2.onnx")
    parser.add_argument("--s2-onnx", type=Path, default=MODEL_DIR / "segformer_b2_pruned68_ft.onnx",
                        help="pruned/fine-tuned ONNX used for both S2 FP16 and S3 INT8")
    parser.add_argument("--s4-onnx", type=Path, default=MODEL_DIR / "segformer_b2_pruned68_ft_distilled.onnx")
    parser.add_argument("--preprocessing", type=Path, default=MODEL_DIR / "preprocessing.json")
    parser.add_argument("--data-root", type=Path, default=ROOT / "ade20k-scene-parse-150")
    parser.add_argument("--calibration-images", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--only", action="append", choices=tuple(STAGES),
                        help="build only this stage; may be repeated")
    parser.add_argument("--recalibrate", action="store_true")
    args = parser.parse_args()

    if args.calibration_images < 1:
        parser.error("--calibration-images must be positive")
    sources = {
        "s0": args.s0_onnx.expanduser().resolve(),
        "s1": args.s1_onnx.expanduser().resolve(),
        "s2": args.s2_onnx.expanduser().resolve(),
        "s3": args.s2_onnx.expanduser().resolve(),
        "s4": args.s4_onnx.expanduser().resolve(),
    }
    stages = args.only or list(STAGES)
    missing = sorted({str(sources[STAGES[stage][0]]) for stage in stages
                      if not sources[STAGES[stage][0]].is_file()})
    if missing:
        parser.error("Required ONNX file(s) not found: " + ", ".join(missing))

    data_root = args.data_root.expanduser().resolve()
    preprocessing_path = args.preprocessing.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    manifest_path = output_dir / "quantization_manifest.json"
    manifest = {"stages": {}}
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest.setdefault("stages", {})

    for stage in stages:
        source_key, precision = STAGES[stage]
        if precision == "int8" and not preprocessing_path.is_file():
            parser.error(f"Preprocessing metadata not found: {preprocessing_path}")
        engine_path = output_dir / f"{stage.lower()}_{precision}.engine"
        try:
            info = build_engine(
                stage, sources[source_key], precision, engine_path, data_root,
                preprocessing_path, args.calibration_images, args.seed, args.recalibrate,
            )
            layer_report = inspect_engine(engine_path)
        except Exception as exc:
            print(f"{stage} failed: {type(exc).__name__}: {exc}", file=sys.stderr)
            return 1
        info_path = engine_path.with_suffix(".build.json")
        manifest["stages"][stage] = {
            "source_onnx": str(sources[source_key]),
            "engine": str(engine_path),
            "precision": precision,
            "build_info": str(info_path),
            "layer_report_csv": str(layer_report),
            "calibration_images": info["calibration_count"],
        }
        print(f"[{stage}] layer report: {layer_report}", flush=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    temporary.replace(manifest_path)
    print(f"Build manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
