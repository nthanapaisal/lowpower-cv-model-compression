#!/usr/bin/env python3
"""Fine-tune a pruned SegFormer-B2 on the stratified 1000-image ADE20K subset.

Pruning lowers the mIoU; this retrains the pruned model so the surviving
channels take over the lost work. It trains the TorchScript file that
prune.py writes, so it needs PyTorch only (no Transformers): it runs on a
GPU machine (Colab) or on a Jetson training board.

    python3 training/finetune.py training/model/segformer_b2_pruned40.torchscript.pt

Defaults follow the plan: the 1000 images listed in training/subset_train_1000.txt,
10 epochs, batch size 1 with gradients accumulated over 8 images (the Jetson's
memory allows only batch 1-2 at 512x512), AdamW at 6e-5 with linear decay.
For a short round between pruning steps use --epochs 1.

Writes, next to the input model:
    <name>_ft.torchscript.pt   for latency and evaluate.py
    <name>_ft.pth              state dict, for prune.py --start (next pruning step)
    <name>_ft.onnx             unless --no-onnx (export.py does the same later)
    <name>_ft_log.csv          loss and wall-clock time per epoch

Preprocessing matches the evaluation: images are scaled to [0, 1] and normalized
with the mean and std in preprocessing.json; labels 1-150 become classes 0-149
and 0 ("other") is ignored. Training adds random scaling (0.5-2.0x), a random
512x512 crop and horizontal flips. BatchNorm and dropout stay as they were
traced (inference mode), which keeps BatchNorm statistics stable at batch 1.

Time 20 steps before a full run, to give the wall-clock estimate:
    python3 training/finetune.py <model> --max-steps 20 --no-onnx
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

PROJECT_DIR = Path(__file__).resolve().parent
REPO_DIR = PROJECT_DIR.parent
DEFAULT_DATA = REPO_DIR / "ade20k-scene-parse-150"
DEFAULT_LIST = PROJECT_DIR / "subset_train_1000.txt"
NUM_CLASSES = 150
IGNORE = 255
SIZE = 512


def find_data_root(path: Path) -> Path:
    for candidate in (path, path / "ADEChallengeData2016"):
        if (candidate / "images" / "training").is_dir() and (candidate / "annotations" / "training").is_dir():
            return candidate
    raise SystemExit(f"No images/training and annotations/training under {path}. "
                     "Run `python3 training/prune.py subset` or pass --data.")


def training_pairs(root: Path, list_path: Path | None) -> list[tuple[Path, Path]]:
    if list_path is None or not list_path.is_file():
        raise SystemExit(f"Training list {list_path} not found. Fine-tuning uses only the stratified subset "
                         "from `python3 training/prune.py subset`.")
    names = [line.strip() for line in list_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    pairs = [(root / "images" / "training" / n, root / "annotations" / "training" / (Path(n).stem + ".png"))
             for n in names]
    missing = [str(p) for pair in pairs for p in pair if not p.is_file()]
    if missing:
        raise SystemExit(f"{len(missing)} listed files are missing; first: {missing[0]}")
    return pairs


def load_preprocessing(model_path: Path) -> dict:
    for candidate in (model_path.parent / "preprocessing.json", PROJECT_DIR / "model" / "preprocessing.json",
                      REPO_DIR / "model" / "preprocessing.json"):
        if candidate.is_file():
            return json.loads(candidate.read_text(encoding="utf-8"))
    raise SystemExit("preprocessing.json not found next to the model. Run load_model.py or prune.py first.")


def to_class_ids(mask: np.ndarray) -> np.ndarray:
    """SceneParse150: 0 = other (ignored), 1..150 = classes; the model predicts 0..149."""
    labels = mask.astype(np.int64)
    if labels.ndim == 3:
        labels = labels[..., 0]
    labels = np.where(labels == 0, IGNORE, labels - 1)
    labels[(labels < 0) | (labels >= NUM_CLASSES)] = IGNORE
    return labels


class SubsetDataset(Dataset):
    def __init__(self, pairs, preprocessing: dict):
        self.pairs = pairs
        self.mean = np.asarray(preprocessing["image_mean"], dtype=np.float32)
        self.std = np.asarray(preprocessing["image_std"], dtype=np.float32)
        self.scale = float(preprocessing.get("rescale_factor", 1 / 255))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        from PIL import Image
        image_path, mask_path = self.pairs[index]
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path)
        side = int(SIZE * random.uniform(0.5, 2.0))
        image = image.resize((side, side), Image.BILINEAR)
        mask = mask.resize((side, side), Image.NEAREST)
        pixels = (np.asarray(image, dtype=np.float32) * self.scale - self.mean) / self.std
        pixels = torch.from_numpy(np.ascontiguousarray(pixels.transpose(2, 0, 1)))
        labels = torch.from_numpy(to_class_ids(np.asarray(mask)))
        if side < SIZE:
            pad = SIZE - side
            pixels = nn.functional.pad(pixels, (0, pad, 0, pad), value=0.0)
            labels = nn.functional.pad(labels, (0, pad, 0, pad), value=IGNORE)
        top, left = random.randint(0, pixels.shape[1] - SIZE), random.randint(0, pixels.shape[2] - SIZE)
        pixels, labels = pixels[:, top:top + SIZE, left:left + SIZE], labels[top:top + SIZE, left:left + SIZE]
        if random.random() < 0.5:
            pixels, labels = pixels.flip(-1), labels.flip(-1)
        return pixels.contiguous(), labels.contiguous()


def save_outputs(model, stem: Path, onnx_path: Path | None) -> list[Path]:
    """TorchScript, state dict, and (unless onnx_path is None) ONNX for the fine-tuned model."""
    model.eval()
    written = [Path(f"{stem}.torchscript.pt"), Path(f"{stem}.pth")]
    cpu = model.to("cpu")
    torch.jit.save(cpu, str(written[0]))
    torch.save({"state_dict": cpu.state_dict(), "meta": {"fine_tuned": True}}, str(written[1]))
    if onnx_path is not None:
        from export import export_onnx
        written.append(export_onnx(written[0], onnx_path))
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model", type=Path, help="pruned .torchscript.pt from prune.py")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="folder with images/ and annotations/")
    parser.add_argument("--list", type=Path, default=DEFAULT_LIST, help="training image names (the 1000 subset)")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--accumulate", type=int, default=8, help="images per optimizer step = batch x this")
    parser.add_argument("--lr", type=float, default=6e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=0, help="stop after this many images (timing runs)")
    parser.add_argument("--device", default=None, help="default: cuda if available, else cpu")
    parser.add_argument("--no-onnx", action="store_true", help="skip the ONNX export at the end")
    parser.add_argument("--onnx", type=Path, default=None, help="ONNX output path (default: <name>_ft.onnx)")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    root = find_data_root(args.data)
    pairs = training_pairs(root, args.list)
    print(f"Fine-tuning on {len(pairs)} images listed in {args.list} (stratified subset).", flush=True)
    preprocessing = load_preprocessing(args.model)
    if (int(preprocessing["height"]), int(preprocessing["width"])) != (SIZE, SIZE):
        raise SystemExit("Expected a 512x512 model.")

    model = torch.jit.load(str(args.model), map_location=device)
    model.train()
    parameters = [p for p in model.parameters() if p.requires_grad]
    print(f"Model {args.model.name}: {sum(p.numel() for p in parameters):,} trainable parameters on {device}. "
          f"Training images: {len(pairs)} from {root}.", flush=True)

    loader = DataLoader(SubsetDataset(pairs, preprocessing), batch_size=args.batch_size, shuffle=True,
                        drop_last=True, num_workers=args.workers, pin_memory=device.type == "cuda",
                        persistent_workers=args.workers > 0)
    images_per_epoch = len(loader) * args.batch_size
    total_images = args.max_steps or args.epochs * images_per_epoch
    total_updates = max(1, math.ceil(total_images / (args.batch_size * args.accumulate)))
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=args.weight_decay)
    schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: max(0.0, 1 - step / total_updates))
    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE)

    stem = args.model.parent / (args.model.name.replace(".torchscript.pt", "") + "_ft")
    log, seen, started = [], 0, time.time()
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(1, args.epochs + 1):
        running, batches, epoch_start = 0.0, 0, time.time()
        for pixels, labels in loader:
            pixels, labels = pixels.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            logits = model(pixels)
            logits = nn.functional.interpolate(logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
            loss = criterion(logits, labels)
            (loss / args.accumulate).backward()
            batches += 1
            seen += pixels.shape[0]
            if batches % args.accumulate == 0:
                optimizer.step()
                schedule.step()
                optimizer.zero_grad(set_to_none=True)
            running += loss.item()
            if seen in (args.batch_size * 5, args.batch_size * 20) or seen % 100 == 0:
                per_image = (time.time() - started) / seen
                print(f"  epoch {epoch} image {seen}/{total_images}  loss {running / batches:.3f}  "
                      f"{per_image:.2f} s/image  (about {per_image * images_per_epoch / 60:.0f} min per epoch, "
                      f"{per_image * total_images / 3600:.1f} h in total)", flush=True)
            if seen >= total_images:
                break
        if batches % args.accumulate:
            optimizer.step()
            schedule.step()
            optimizer.zero_grad(set_to_none=True)
        log.append({"epoch": epoch, "images": seen, "train_loss": round(running / max(batches, 1), 4),
                    "epoch_minutes": round((time.time() - epoch_start) / 60, 2),
                    "total_minutes": round((time.time() - started) / 60, 2)})
        print(f"Epoch {epoch}: loss {log[-1]['train_loss']}, {log[-1]['epoch_minutes']} min", flush=True)
        if seen >= total_images:
            break

    with Path(f"{stem}_log.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(log[0]))
        writer.writeheader()
        writer.writerows(log)
    written = save_outputs(model, stem, None if args.no_onnx else (args.onnx or Path(f"{stem}.onnx")))
    print(f"Wall-clock training time: {(time.time() - started) / 60:.1f} min for {seen} images "
          f"({(time.time() - started) / max(seen, 1):.2f} s/image on {device}).")
    for path in written + [Path(f"{stem}_log.csv")]:
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
