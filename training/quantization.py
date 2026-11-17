#!/usr/bin/env python3
"""Build FP16 and/or INT8 TensorRT engines with the professor's helper.

Both input routes produce ``.engine`` files:

    python3 training/quantization.py torchscript --model model/segformer_b2.torchscript.pt
    python3 training/quantization.py onnx --model model/pruned.onnx

TorchScript is first exported to a temporary ONNX file. Existing ONNX input is
passed directly to ``lab1_trt.py``. Engines are saved under ``model/`` by
default. Use ``--precision fp16``, ``int8``, or ``both`` (the default).

The provided helper currently validates inputs as (1, 3, 224, 224) and outputs
as (1, 10). SegFormer B2 uses different dimensions, so the helper will reject
that model unless its ONNX graph meets the helper's required shapes.
"""

import argparse
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
COURSE_HELPER_DIR = Path("/course-data/lab1")
PRECISIONS = ("fp16", "int8")


def _build_with_helper(onnx_path: Path, output_dir: Path, stem: str,
                       precisions: tuple[str, ...], recalibrate: bool) -> None:
    if not COURSE_HELPER_DIR.is_dir():
        raise FileNotFoundError(
            f"Professor helper directory not found: {COURSE_HELPER_DIR}. "
            "Run this on the course LPCV environment."
        )
    sys.path.insert(0, str(COURSE_HELPER_DIR))
    import lab1_trt

    output_dir.mkdir(parents=True, exist_ok=True)
    for precision in precisions:
        engine_path = output_dir / f"{stem}_{precision}.engine"
        command = [
            "build", "--onnx", str(onnx_path), "--precision", precision,
            "--out", str(engine_path),
        ]
        if recalibrate and precision == "int8":
            command.append("--recalibrate")
        print(f"Building {precision.upper()} engine: {engine_path}", flush=True)
        status = lab1_trt.main(command)
        if status:
            raise RuntimeError(
                f"Professor helper failed to build {precision.upper()} engine "
                f"(status {status})."
            )


def _torchscript_to_onnx(path: Path, height: int, width: int, output_dir: Path,
                         stem: str, precisions: tuple[str, ...],
                         recalibrate: bool) -> None:
    import torch

    model = torch.jit.load(str(path), map_location="cpu").eval()
    sample = torch.zeros((1, 3, height, width), dtype=torch.float32)
    with tempfile.TemporaryDirectory(prefix="quantization-onnx-") as temp:
        onnx_path = Path(temp) / f"{stem}.onnx"
        torch.onnx.export(
            model, sample, str(onnx_path), input_names=["input"],
            output_names=["output"], opset_version=17, dynamo=False,
        )
        _build_with_helper(
            onnx_path, output_dir, stem, precisions, recalibrate
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="source_type", required=True)
    for source_type in ("torchscript", "onnx"):
        sub = subparsers.add_parser(source_type)
        sub.add_argument("--model", type=Path, required=True,
                         help="input TorchScript .pt or ONNX model")
        sub.add_argument("--output-dir", type=Path, default=ROOT / "model",
                         help="directory for the generated .engine files")
        sub.add_argument("--precision", choices=("fp16", "int8", "both"),
                         default="both", help="engine precision(s), default: both")
        sub.add_argument("--recalibrate", action="store_true",
                         help="force fresh calibration when building INT8")
        if source_type == "torchscript":
            sub.add_argument("--height", type=int, default=512,
                             help="TorchScript input height (default: 512)")
            sub.add_argument("--width", type=int, default=512,
                             help="TorchScript input width (default: 512)")

    args = parser.parse_args()
    source = args.model.expanduser().resolve()
    if not source.is_file():
        parser.error(f"model file not found: {source}")
    output_dir = args.output_dir.expanduser()
    if not output_dir.is_absolute():
        output_dir = Path.cwd() / output_dir
    precisions = PRECISIONS if args.precision == "both" else (args.precision,)
    stem = source.stem.replace(".torchscript", "")

    try:
        if args.source_type == "torchscript":
            _torchscript_to_onnx(
                source, args.height, args.width, output_dir, stem,
                precisions, args.recalibrate,
            )
        else:
            if source.suffix.lower() != ".onnx":
                parser.error("onnx mode requires an input file ending in .onnx")
            _build_with_helper(
                source, output_dir, stem, precisions, args.recalibrate
            )
    except Exception as exc:
        parser.exit(1, f"ERROR: {type(exc).__name__}: {exc}\n")


if __name__ == "__main__":
    main()
