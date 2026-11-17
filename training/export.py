"""Export the four Lab 1 MobileNetV2 checkpoints as fixed-shape ONNX models."""

import argparse
from pathlib import Path

import timm
import torch


MODEL_NAMES = ("S0", "S30", "S50", "S70")


def load_model(checkpoint_path: Path, architecture: str, num_classes: int):
    """Load either a regular state dict or a physically pruned full model."""
    try:
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
    except TypeError:  # Older course PyTorch versions lack weights_only.
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if isinstance(checkpoint, torch.nn.Module):
        model = checkpoint
    elif isinstance(checkpoint, dict) and isinstance(
        checkpoint.get("model"), torch.nn.Module
    ):
        model = checkpoint["model"]
    else:
        state_dict = checkpoint.get("model", checkpoint) if isinstance(
            checkpoint, dict
        ) else checkpoint
        model = timm.create_model(
            architecture, pretrained=False, num_classes=num_classes
        )
        model.load_state_dict(state_dict)
    return model.cpu().eval()


def export_onnx(checkpoint_path: Path, output_path: Path, architecture: str,
                num_classes: int):
    model = load_model(checkpoint_path, architecture, num_classes)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    example = torch.randn(1, 3, 224, 224)
    torch.onnx.export(
        model,
        example,
        output_path,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes=None,
        opset_version=17,
        dynamo=False,
    )
    print(f"saved fixed-shape ONNX model: {output_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoints", nargs=4, type=Path,
        metavar=("S0", "S30", "S50", "S70"),
        default=[Path("results") / f"{name}.pth" for name in MODEL_NAMES],
        help="checkpoints in S0, S30, S50, S70 order "
             "(default: results/<model>.pth)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("."),
        help="directory for S0.onnx, S30.onnx, S50.onnx, and S70.onnx",
    )
    parser.add_argument("--architecture", default="mobilenetv2_100")
    parser.add_argument("--num-classes", type=int, default=10)
    args = parser.parse_args()

    missing = [path for path in args.checkpoints if not path.is_file()]
    if missing:
        paths = ", ".join(str(path) for path in missing)
        parser.error(f"missing checkpoint(s): {paths}")

    for name, checkpoint in zip(MODEL_NAMES, args.checkpoints):
        export_onnx(
            checkpoint, args.output_dir / f"{name}.onnx",
            args.architecture, args.num_classes,
        )


if __name__ == "__main__":
    main()
