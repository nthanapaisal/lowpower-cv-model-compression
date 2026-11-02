#!/usr/bin/env python3
"""Download SegFormer B0/B1/B2 and export standalone TorchScript models.

Run on a machine where PyTorch and Transformers are installed:
    python3 load_model.py

Creates ./model/segformer_b{0,1,2}.torchscript.pt plus preprocessing.json.
The .pt files contain the model graph and weights and can be loaded with
torch.jit.load() on the Jetson without Transformers.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "model"
HF_CACHE = PROJECT_DIR / ".cache" / "huggingface"
os.environ["HF_HOME"] = str(HF_CACHE)
os.environ["HF_HUB_CACHE"] = str(HF_CACHE / "hub")

CHECKPOINTS = (
    ("b0", "nvidia/segformer-b0-finetuned-ade-512-512"),
    ("b1", "nvidia/segformer-b1-finetuned-ade-512-512"),
    ("b2", "nvidia/segformer-b2-finetuned-ade-512-512"),
)


def main() -> None:
    try:
        import torch
        from torch import nn
        from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation
    except ImportError as exc:
        raise SystemExit(
            "Export requires PyTorch and Transformers on this computer. "
            "Run this script in an environment where both are available."
        ) from exc

    class SegFormerLogits(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, pixel_values):
            return self.model(pixel_values=pixel_values).logits

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    preprocessor_saved = False
    for variant, checkpoint in CHECKPOINTS:
        print(f"Loading {checkpoint}", flush=True)
        processor = AutoImageProcessor.from_pretrained(checkpoint)
        if not preprocessor_saved:
            metadata = {
                "height": 512,
                "width": 512,
                "image_mean": [float(value) for value in processor.image_mean],
                "image_std": [float(value) for value in processor.image_std],
                "do_rescale": bool(processor.do_rescale),
                "rescale_factor": float(processor.rescale_factor),
                "do_normalize": bool(processor.do_normalize),
                "resample": int(processor.resample),
                "source_checkpoint": checkpoint,
            }
            (OUTPUT_DIR / "preprocessing.json").write_text(
                json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
            )
            preprocessor_saved = True

        model = AutoModelForSemanticSegmentation.from_pretrained(checkpoint)
        model.eval().to(device="cpu", dtype=torch.float32)
        wrapper = SegFormerLogits(model).eval()
        example = torch.zeros((1, 3, 512, 512), dtype=torch.float32)
        with torch.inference_mode():
            eager_output = wrapper(example)
            traced = torch.jit.trace(wrapper, example, strict=False, check_trace=False)
            traced_output = traced(example)
        torch.testing.assert_close(traced_output, eager_output, rtol=1e-4, atol=1e-4)

        output_path = OUTPUT_DIR / f"segformer_{variant}.torchscript.pt"
        torch.jit.save(traced, str(output_path))
        print(f"Saved {output_path} ({output_path.stat().st_size / 1024**2:.1f} MiB)", flush=True)
        del traced, wrapper, model, processor

    print(f"Export complete: {OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
