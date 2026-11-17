#!/usr/bin/env python3
"""Export a saved PyTorch ``.pth`` model/checkpoint to ONNX.

Examples (from the repository root)::

    python3 training/export.py S0
    python3 training/export.py S0 --checkpoint results/S0.pth

The default lookup checks ``results/``, ``training/results/``, ``model/`` and
``training/model/`` for ``<name>.pth``. A checkpoint containing a serialized
``torch.nn.Module`` can be exported directly. For a weights-only state dict,
provide ``--architecture`` and ``--num-classes`` to construct the model with
timm first.
"""

import argparse
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]


def find_checkpoint(name: str, explicit_path: Path | None) -> Path:
    if explicit_path is not None:
        path = explicit_path.expanduser()
        if not path.is_absolute():
            path = Path.cwd() / path
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        return path

    filename = name if name.lower().endswith(".pth") else f"{name}.pth"
    candidates = [ROOT / folder / filename for folder in (
        "results", "training/results", "model", "training/model"
    )]
    for path in candidates:
        if path.is_file():
            return path
    searched = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        f"No checkpoint for {name!r}. Looked in: {searched}. "
        "Use --checkpoint to specify its location."
    )


def load_model(checkpoint_path: Path, architecture: str | None,
               num_classes: int | None) -> torch.nn.Module:
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch versions before weights_only was added.
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if isinstance(checkpoint, torch.nn.Module):
        model = checkpoint
    elif isinstance(checkpoint, dict) and isinstance(checkpoint.get("model"), torch.nn.Module):
        model = checkpoint["model"]
    else:
        if not architecture:
            raise ValueError(
                f"{checkpoint_path} contains weights only. Pass --architecture "
                "and (if needed) --num-classes so the model structure can be created."
            )
        try:
            import timm
        except ImportError as exc:
            raise SystemExit("Loading a weights-only checkpoint requires timm.") from exc
        state_dict = checkpoint
        if isinstance(checkpoint, dict):
            for key in ("state_dict", "model_state_dict", "model"):
                if key in checkpoint and isinstance(checkpoint[key], dict):
                    state_dict = checkpoint[key]
                    break
        if not isinstance(state_dict, dict):
            raise ValueError(f"Unsupported checkpoint format: {checkpoint_path}")
        model = timm.create_model(
            architecture, pretrained=False,
            num_classes=num_classes if num_classes is not None else 1000,
        )
        # Handle checkpoints saved from DistributedDataParallel.
        state_dict = {key.removeprefix("module."): value
                      for key, value in state_dict.items()}
        model.load_state_dict(state_dict)
    return model.cpu().eval()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="model/checkpoint name, for example S0")
    parser.add_argument("--checkpoint", type=Path,
                        help="explicit .pth path (overrides automatic lookup)")
    parser.add_argument("--output", type=Path,
                        help="ONNX output path (default: onnx/<model>.onnx)")
    parser.add_argument("--architecture",
                        help="timm architecture for a weights-only checkpoint")
    parser.add_argument("--num-classes", type=int,
                        help="classifier output count for --architecture")
    parser.add_argument("--height", type=int, default=224,
                        help="input image height (default: 224)")
    parser.add_argument("--width", type=int, default=224,
                        help="input image width (default: 224)")
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    checkpoint_path = find_checkpoint(args.model, args.checkpoint)
    output_path = args.output or (ROOT / "onnx" / f"{args.model}.onnx")
    if not output_path.is_absolute():
        output_path = Path.cwd() / output_path
    model = load_model(checkpoint_path, args.architecture, args.num_classes)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sample = torch.zeros(1, 3, args.height, args.width, dtype=torch.float32)
    torch.onnx.export(
        model, sample, str(output_path),
        input_names=["input"], output_names=["output"],
        opset_version=args.opset, dynamo=False,
    )
    print(f"Exported {checkpoint_path} -> {output_path}")


if __name__ == "__main__":
    main()
