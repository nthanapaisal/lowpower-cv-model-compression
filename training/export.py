#!/usr/bin/env python3
"""Export SegFormer models to fixed-shape ONNX for TensorRT (S0-S4).

    python3 training/export.py training/model/segformer_b2_pruned40_ft.torchscript.pt
    python3 training/export.py training/model/segformer_b2.torchscript.pt --fp16

Input: a TorchScript file (from load_model.py, prune.py or finetune.py; needs
PyTorch only) or a .pth checkpoint from prune.py/finetune.py (needs
Transformers, rebuilt with prune.load_checkpoint).

Output: <name>.onnx (or <name>_fp16.onnx with --fp16), opset 17, input "input"
[1, 3, 512, 512] and output "logits" [1, 150, 128, 128], fixed shapes as in
Lab 1. With --fp16 the weights, input and output are FP16, like
segformer_b2_fp16.onnx in baseline/inference.py. Without it they are FP32, and
TensorRT can still build an FP16 engine from it with the FP16 builder flag.

If onnxruntime is installed, the ONNX model is run once on a random image and
compared with PyTorch.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

SIZE = 512


def load_any(path: Path):
    """A module that maps pixel_values [1,3,512,512] to logits [1,150,128,128]."""
    if path.name.endswith(".torchscript.pt"):
        return torch.jit.load(str(path), map_location="cpu").eval()
    from prune import load_checkpoint, wrap   # needs Transformers
    return wrap(load_checkpoint(path)).eval()


def export_onnx(source: Path, output: Path | None = None, fp16: bool = False, check: bool = True) -> Path:
    model = load_any(Path(source))
    if output is None:
        base = Path(source).name.replace(".torchscript.pt", "").replace(".pth", "")
        output = Path(source).parent / f"{base}{'_fp16' if fp16 else ''}.onnx"
    output.parent.mkdir(parents=True, exist_ok=True)
    example = torch.randn(1, 3, SIZE, SIZE)
    with torch.inference_mode():
        expected = model(example).float()
    if fp16:
        model = model.half()
        example = example.half()
    kwargs = {"input_names": ["input"], "output_names": ["logits"], "opset_version": 17, "dynamic_axes": None}
    try:
        torch.onnx.export(model, example, str(output), dynamo=False, **kwargs)
    except TypeError:  # older PyTorch without the dynamo argument
        torch.onnx.export(model, example, str(output), **kwargs)
    if check:
        _check(output, example, expected, tolerance=2e-2 if fp16 else 1e-3)
    return output


def _check(path: Path, example, expected, tolerance: float) -> None:
    try:
        import numpy as np
        import onnxruntime
    except ImportError:
        print("onnxruntime not installed: ONNX output not compared with PyTorch.")
        return
    session = onnxruntime.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    got = session.run(None, {"input": example.numpy()})[0].astype(np.float32)
    want = expected.numpy()
    error = float(np.abs(got - want).max() / (np.abs(want).max() or 1.0))
    agree = float((got.argmax(1) == want.argmax(1)).mean())
    print(f"ONNX check: max relative error {error:.2e}, same class at {agree:.2%} of output pixels")
    if error > tolerance:
        raise RuntimeError(f"ONNX output differs from PyTorch (relative error {error:.2e}): {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("models", nargs="+", type=Path, help=".torchscript.pt or .pth files")
    parser.add_argument("--fp16", action="store_true", help="FP16 weights, input and output")
    parser.add_argument("--output-dir", type=Path, default=None, help="default: next to each input")
    parser.add_argument("--no-check", action="store_true")
    args = parser.parse_args()
    for source in args.models:
        output = None
        if args.output_dir is not None:
            base = source.name.replace(".torchscript.pt", "").replace(".pth", "")
            output = args.output_dir / f"{base}{'_fp16' if args.fp16 else ''}.onnx"
        path = export_onnx(source, output, fp16=args.fp16, check=not args.no_check)
        print(f"Saved fixed-shape ONNX model: {path}", flush=True)


if __name__ == "__main__":
    main()
