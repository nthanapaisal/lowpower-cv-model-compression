#!/usr/bin/env python3
"""CPU smoke test for exported SegFormer TorchScript model files.

Usage:
    python3 test_torchscript.py          # test every model found in ./model
    python3 test_torchscript.py b0       # test just B0

This uses a synthetic 512x512 FP32 tensor. It verifies that each artifact can
be loaded and produces finite 150-class segmentation logits; it does not score
accuracy or require Transformers, CUDA, or the ADE20K dataset.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parent / "model"
MODEL_FILES = {
    "b0": "segformer_b0.torchscript.pt",
    "b1": "segformer_b1.torchscript.pt",
    "b2": "segformer_b2.torchscript.pt",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", nargs="?", choices=("b0", "b1", "b2", "all"), default="all")
    args = parser.parse_args()

    try:
        import torch
    except ImportError as exc:
        raise SystemExit("PyTorch is required to load and test the TorchScript file.") from exc

    metadata_path = MODEL_DIR / "preprocessing.json"
    if not metadata_path.is_file():
        raise SystemExit(f"Missing preprocessing metadata: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    height, width = int(metadata["height"]), int(metadata["width"])
    if (height, width) != (512, 512):
        raise SystemExit(f"Expected exported 512x512 input, got {height}x{width}")

    selected = MODEL_FILES if args.model == "all" else {args.model: MODEL_FILES[args.model]}
    for name, filename in selected.items():
        path = MODEL_DIR / filename
        if not path.is_file():
            raise SystemExit(f"Missing {name.upper()} TorchScript file: {path}")
        print(f"Testing {name.upper()}: {path} ({path.stat().st_size / 1024**2:.1f} MiB)", flush=True)
        model = torch.jit.load(str(path), map_location="cpu").eval()
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        sample = torch.zeros((1, 3, height, width), dtype=torch.float32)
        with torch.inference_mode():
            logits = model(sample)
        if not isinstance(logits, torch.Tensor):
            raise RuntimeError(f"Expected tensor logits, received {type(logits).__name__}")
        if logits.ndim != 4 or logits.shape[0] != 1 or logits.shape[1] != 150:
            raise RuntimeError(f"Expected logits [1,150,H,W], received {tuple(logits.shape)}")
        if not torch.isfinite(logits).all().item():
            raise RuntimeError("Model output contains NaN or infinity")
        print(f"PASS {name.upper()}: logits={tuple(logits.shape)}, dtype={logits.dtype}, "
              f"parameters={parameter_count:,}", flush=True)
        del logits, sample, model
        gc.collect()

    print("TorchScript smoke test passed.", flush=True)


if __name__ == "__main__":
    main()
