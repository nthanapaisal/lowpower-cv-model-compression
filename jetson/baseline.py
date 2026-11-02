#!/usr/bin/env python3
"""Reusable semantic segmentation evaluator for the Jetson baseline/compressed models.

Run from the project root with ``python3 jetson/baseline.py`` after exporting
the TorchScript models into ``./model`` and placing ADE20K under
``./ade20k-scene-parse-150`` (or ``./data``). Transformers is not needed here.
"""

from __future__ import annotations

import csv
import gc
import json
import shutil
import subprocess
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional

# Keep Hugging Face model downloads in the directory from which this script is
# run, instead of the user's home cache. ADE20K itself is read from ./data.
try:
    import numpy as np
except ImportError:  # The --check-data path only needs Pillow.
    np = None  # type: ignore[assignment]


MODEL_ID = "nvidia/segformer-b0-finetuned-ade-512-512"
DATASET_ID = "zhoubolei/scene_parse_150"
DATASET_CONFIG = "scene_parsing"
ARCHITECTURE = "MiT-B0 encoder + MLP decoder"
MODEL_VARIANTS = (
    ("SegFormer-B0", "nvidia/segformer-b0-finetuned-ade-512-512", "MiT-B0 encoder + MLP decoder"),
    ("SegFormer-B1", "nvidia/segformer-b1-finetuned-ade-512-512", "MiT-B1 encoder + MLP decoder"),
    ("SegFormer-B2", "nvidia/segformer-b2-finetuned-ade-512-512", "MiT-B2 encoder + MLP decoder"),
)
NUM_CLASSES = 150
SOURCE_IGNORE_LABEL = 0
MODEL_IGNORE_LABEL = 255
DEFAULT_EVALUATION_IMAGES = 2000
INTER_MODEL_COOLDOWN_SECONDS = 60
HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent if HERE.name == "jetson" else HERE
MODEL_DIR = PROJECT_ROOT / "model"
MODEL_FILES = {
    "SegFormer-B0": "segformer_b0.torchscript.pt",
    "SegFormer-B1": "segformer_b1.torchscript.pt",
    "SegFormer-B2": "segformer_b2.torchscript.pt",
}


@dataclass
class EvaluationConfig:
    model_name: str = "SegFormer-B0"
    model_path: str = MODEL_ID
    checkpoint: Optional[str] = None
    architecture: str = ARCHITECTURE
    dataset_name: str = DATASET_ID
    dataset_config: str = DATASET_CONFIG
    split: str = "validation"
    input_height: int = 512
    input_width: int = 512
    precision: str = "fp32"
    batch_size: int = 1
    warmup_iterations: int = 2
    warmup_seconds: float = 2.0
    timed_iterations: int = 1000
    evaluation_images: Optional[int] = DEFAULT_EVALUATION_IMAGES
    num_classes: int = NUM_CLASSES
    device: str = "cuda"
    runtime: str = "PyTorch"
    output_csv: str = "baseline_results.csv"


RESULT_COLUMNS = (
    "model_name", "model", "checkpoint", "architecture", "runtime", "device",
    "dataset", "dataset_config", "split", "evaluation_set_id",
    "num_images", "evaluation_images_requested", "num_classes", "input_height", "input_width", "precision", "batch_size",
    "warmup_seconds_configured", "warmup_iterations_configured", "warmup_seconds_actual",
    "warmup_iterations", "timed_iterations_configured", "latency_mean_ms", "latency_p50_ms",
    "latency_p99_ms", "latency_fps", "mean_iou_percent", "pixel_accuracy_percent", "valid_pixels",
    "total_parameters", "trainable_parameters", "system_ram_idle_used_mb", "system_ram_idle_total_mb",
    "system_ram_model_loaded_used_mb", "system_ram_peak_during_inference_used_mb",
    "system_ram_inference_samples", "system_ram_peak_status", "cuda_loaded_allocated_mb",
    "cuda_loaded_reserved_mb", "cuda_peak_allocated_mb", "cuda_peak_reserved_mb",
    "energy_per_inference_mj", "energy_incremental_per_inference_mj", "power_idle_mean_mw",
    "power_busy_mean_mw", "energy_idle_sample_count", "energy_busy_sample_count",
    "energy_busy_duration_s", "energy_inference_count", "energy_warmup_seconds",
    "energy_warmup_inferences", "energy_status", "status", "status_details",
)


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required to load this baseline. Activate the lpcv environment.") from exc
    return torch


def _resolve_device(device: str):
    torch = _torch()
    if device != "cuda":
        raise ValueError("This reference-model comparison requires CUDA on the Jetson Orin Nano.")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable. Run this script on the Jetson CUDA environment.")
    return torch.device(device)


def _cuda_memory_snapshot(device: Any) -> tuple[Optional[float], Optional[float]]:
    torch = _torch()
    if getattr(device, "type", None) != "cuda" or not torch.cuda.is_available():
        return None, None
    return (torch.cuda.memory_allocated(device) / 1024**2,
            torch.cuda.memory_reserved(device) / 1024**2)


def load_model(model_path: str, precision: str = "fp32", device: str = "cuda", *,
               loader: Optional[Callable[..., Any]] = None) -> Any:
    """Load a model. Pass ``loader`` to use another architecture/deployment runtime.

    A custom loader receives ``model_path``, ``precision`` and ``device``. The
    returned object must be consumable by the supplied inference_fn adapter.
    """
    if loader is not None:
        return loader(model_path=model_path, precision=precision, device=device)
    torch = _torch()
    resolved = _resolve_device(device)
    if precision != "fp32":
        raise ValueError("The exported reference models are FP32; other precision modes are not enabled.")
    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(f"TorchScript model not found: {path}. Run load_model.py and copy ./model to this directory.")
    model = torch.jit.load(str(path), map_location=resolved)
    model.to(device=resolved, dtype=torch.float32)
    model.eval()
    return model


def load_evaluation_dataset(dataset_name: str, split: str = "validation", image_processor: Any = None,
                            *, dataset_config: str = DATASET_CONFIG,
                            num_images: Optional[int] = None) -> Any:
    """Load sorted validation images and annotations from the local ADE20K tree."""
    if split != "validation":
        raise ValueError("Baseline accuracy uses the labeled local validation split only.")
    # Accept the project's current extracted folder, the prior documented
    # ./data location, and the extra directory created by unzipping the
    # official ADEChallengeData2016.zip into either location.
    roots = (
        PROJECT_ROOT / "ade20k-scene-parse-150",
        PROJECT_ROOT / "data" / "ade20k-scene-parse-150",
        PROJECT_ROOT / "data",
        Path.cwd() / "ade20k-scene-parse-150",
        Path.cwd() / "data" / "ade20k-scene-parse-150",
        Path.cwd() / "data",
    )
    seen: set[Path] = set()
    locations: list[tuple[Path, Path, Path]] = []
    for root in roots:
        for candidate in (root, root / "ADEChallengeData2016"):
            candidate = candidate.resolve()
            if candidate in seen:
                continue
            seen.add(candidate)
            image_dir = candidate / "images" / "validation"
            annotation_dir = candidate / "annotations" / "validation"
            locations.append((candidate, image_dir, annotation_dir))
    resolved = next(((root, images, masks) for root, images, masks in locations
                     if images.is_dir() and masks.is_dir()), None)
    if resolved is None:
        expected = PROJECT_ROOT / "ade20k-scene-parse-150"
        raise FileNotFoundError(
            "Local ADE20K validation data was not found. Expected folders like "
            f"{expected}/images/validation and {expected}/annotations/validation "
            "(or the same structure under ./data). Extract ADEChallengeData2016.zip there."
        )
    root, image_dir, annotation_dir = resolved
    image_paths = sorted(image_dir.glob("*.jpg"), key=lambda p: p.name)
    pairs = [(image, annotation_dir / f"{image.stem}.png") for image in image_paths]
    missing = [str(image) for image, annotation in pairs if not annotation.is_file()]
    if not pairs:
        raise FileNotFoundError(f"No validation JPG images found in {image_dir}")
    if missing:
        raise FileNotFoundError(f"Missing validation annotations for {len(missing)} images; first: {missing[0]}")
    if num_images is not None:
        pairs = pairs[:max(0, num_images)]
    print(f"Local validation data: {root}", flush=True)
    print(f"Local validation images available: {len(image_paths)}", flush=True)
    print(f"Images selected for evaluation: {len(pairs)}", flush=True)

    class LocalADE20KValidation:
        def __len__(self) -> int:
            return len(pairs)

        def __getitem__(self, index: int) -> dict[str, Any]:
            try:
                from PIL import Image
            except ImportError as exc:
                raise RuntimeError("Pillow is required to read local ADE20K images.") from exc
            image_path, annotation_path = pairs[index]
            with Image.open(image_path) as image_file:
                image = image_file.convert("RGB")
            with Image.open(annotation_path) as annotation_file:
                # Preserve indexed mask values if Pillow opens the PNG as P.
                annotation = annotation_file.copy()
            return {"image": image, "annotation": annotation}

    # Keep the same interface as the Hugging Face Dataset object without
    # loading image pixels into memory until each item is evaluated.
    _ = dataset_name, dataset_config, image_processor
    return LocalADE20KValidation()


def preprocess_image(image: Any, image_processor: Any, *, height: int = 512, width: int = 512) -> Any:
    """Apply the checkpoint processor with the evaluator's fixed input size."""
    if hasattr(image, "convert"):
        image = image.convert("RGB")
    if image_processor is None:
        return image
    # HF processors differ slightly by Transformers version. Update their size
    # in place so baseline and compressed checkpoints use this explicit config.
    if hasattr(image_processor, "size"):
        current = image_processor.size
        if isinstance(current, Mapping):
            image_processor.size = {"height": height, "width": width}
        else:
            image_processor.size = {"height": height, "width": width}
    return image_processor(images=image, return_tensors="pt")


def run_inference(model: Any, inputs: Any, device: Any, *,
                  inference_fn: Optional[Callable[[Any, Any, Any], Any]] = None) -> Any:
    """Run one inference; override inference_fn to adapt TensorRT/other runtimes."""
    if inference_fn is not None:
        return inference_fn(model, inputs, device)
    torch = _torch()
    if isinstance(inputs, Mapping):
        inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}
        with torch.inference_mode():
            return model(**inputs)
    with torch.inference_mode():
        return model(inputs)


def _extract_logits(outputs: Any) -> Any:
    if hasattr(outputs, "logits"):
        return outputs.logits
    if isinstance(outputs, Mapping):
        for key in ("logits", "out", "predictions"):
            if key in outputs:
                return outputs[key]
    if isinstance(outputs, (tuple, list)):
        return outputs[0]
    return outputs


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _to_class_maps(predictions: Any, num_classes: int, target_size: Optional[tuple[int, int]] = None) -> np.ndarray:
    """Convert logits [B,C,h,w] or class IDs [B,H,W] to class-ID maps."""
    torch = _torch()
    if hasattr(predictions, "shape") and len(predictions.shape) == 4:
        if predictions.shape[1] != num_classes:
            raise ValueError(f"Expected {num_classes} output classes, got {predictions.shape[1]} channels")
        if target_size is not None and tuple(predictions.shape[-2:]) != tuple(target_size):
            if isinstance(predictions, np.ndarray):
                tensor = torch.from_numpy(predictions)
            else:
                tensor = predictions
            tensor = torch.nn.functional.interpolate(tensor.float(), size=target_size,
                                                      mode="bilinear", align_corners=False)
            predictions = tensor.argmax(dim=1)
        else:
            predictions = predictions.argmax(dim=1)
    result = _to_numpy(predictions).astype(np.int64, copy=False)
    if result.ndim == 2:
        result = result[None, ...]
    if result.ndim != 3:
        raise ValueError(f"Predictions must resolve to [batch,height,width], got shape {result.shape}")
    return result


def calculate_segmentation_metrics(predictions: Any, labels: Any, num_classes: int,
                                   ignore_index: int = MODEL_IGNORE_LABEL) -> dict[str, Any]:
    """Calculate global confusion matrix, class IoU, mIoU and pixel accuracy."""
    predictions = _to_class_maps(predictions, num_classes)
    labels = _to_numpy(labels).astype(np.int64, copy=False)
    if labels.ndim == 2:
        labels = labels[None, ...]
    if predictions.shape != labels.shape:
        raise ValueError(f"Prediction/label shape mismatch: {predictions.shape} vs {labels.shape}")
    valid = (labels != ignore_index) & (labels >= 0) & (labels < num_classes)
    valid &= (predictions >= 0) & (predictions < num_classes)
    encoded = labels[valid] * num_classes + predictions[valid]
    confusion = np.bincount(encoded, minlength=num_classes**2).reshape(num_classes, num_classes)
    return _metrics_from_confusion(confusion)


def _metrics_from_confusion(confusion: np.ndarray) -> dict[str, Any]:
    diagonal = np.diag(confusion).astype(np.float64)
    union = confusion.sum(axis=0) + confusion.sum(axis=1) - diagonal
    class_iou = np.divide(diagonal, union, out=np.full(len(diagonal), np.nan), where=union > 0)
    total = int(confusion.sum())
    return {"confusion_matrix": confusion, "mean_iou_percent": float(np.nanmean(class_iou) * 100),
            "pixel_accuracy_percent": float(diagonal.sum() / total * 100) if total else float("nan"),
            "class_iou_percent": [None if np.isnan(x) else float(x * 100) for x in class_iou],
            "valid_pixels": total}


def _source_mask_to_model_ids(mask: Any) -> np.ndarray:
    """SceneParse150 uses 0=other/void and 1..150=classes; model uses 0..149."""
    labels = _to_numpy(mask).astype(np.int64, copy=True)
    if labels.ndim == 3 and labels.shape[-1] in (1, 3, 4):
        labels = labels[..., 0]
    labels[labels == SOURCE_IGNORE_LABEL] = MODEL_IGNORE_LABEL
    valid = labels != MODEL_IGNORE_LABEL
    labels[valid] -= 1
    labels[(labels < 0) | (labels >= NUM_CLASSES)] = MODEL_IGNORE_LABEL
    return labels


def _load_processor(model_path: str) -> Any:
    """Load the exported, Transformers-free preprocessing configuration."""
    if np is None:
        raise RuntimeError("NumPy is required for FP32 image preprocessing.")
    config_path = Path(model_path).parent / "preprocessing.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Preprocessing config not found: {config_path}. Copy the full ./model directory.")
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Pillow is required for local ADE20K preprocessing.") from exc
    with config_path.open(encoding="utf-8") as stream:
        config = json.load(stream)

    class LocalSegFormerProcessor:
        def __init__(self, values: Mapping[str, Any]):
            self.size = {"height": int(values["height"]), "width": int(values["width"])}
            self.image_mean = np.asarray(values["image_mean"], dtype=np.float32)
            self.image_std = np.asarray(values["image_std"], dtype=np.float32)
            self.do_rescale = bool(values["do_rescale"])
            self.rescale_factor = float(values["rescale_factor"])
            self.do_normalize = bool(values["do_normalize"])
            self.resample = int(values["resample"])

        def __call__(self, images: Any, return_tensors: str = "pt") -> dict[str, Any]:
            if return_tensors != "pt":
                raise ValueError("The local processor supports PyTorch tensors only.")
            torch = _torch()
            image = images.convert("RGB") if hasattr(images, "convert") else Image.fromarray(images).convert("RGB")
            image = image.resize((self.size["width"], self.size["height"]), resample=self.resample)
            pixels = np.asarray(image, dtype=np.float32)
            if self.do_rescale:
                pixels *= self.rescale_factor
            if self.do_normalize:
                pixels = (pixels - self.image_mean) / self.image_std
            chw = np.ascontiguousarray(pixels.transpose(2, 0, 1))
            return {"pixel_values": torch.from_numpy(chw).unsqueeze(0)}

    return LocalSegFormerProcessor(config)


def _sync(device: Any) -> None:
    try:
        torch = _torch()
    except RuntimeError:
        return  # Non-PyTorch runtimes should synchronize in their inference adapter.
    if (getattr(device, "type", None) == "cuda" or device == "cuda") and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def _prepare_one(example: Mapping[str, Any], image_processor: Any, device: Any,
                 height: int, width: int) -> tuple[Any, np.ndarray]:
    encoded = preprocess_image(example["image"], image_processor, height=height, width=width)
    labels = _source_mask_to_model_ids(example["annotation"])
    if isinstance(encoded, Mapping):
        encoded = {k: v.to(device) if hasattr(v, "to") else v for k, v in encoded.items()}
    return encoded, labels


def evaluate_accuracy(model: Any, dataset: Any, image_processor: Any, device: Any,
                      *, num_classes: int = NUM_CLASSES, height: int = 512, width: int = 512,
                      inference_fn: Optional[Callable[[Any, Any, Any], Any]] = None,
                      progress: bool = True) -> dict[str, Any]:
    """Evaluate an already-loaded model and dataset; no model or data is loaded here."""
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    count = len(dataset)
    for index in range(count):
        inputs, labels = _prepare_one(dataset[index], image_processor, device, height, width)
        outputs = run_inference(model, inputs, device, inference_fn=inference_fn)
        logits = _extract_logits(outputs)
        predictions = _to_class_maps(logits, num_classes, target_size=labels.shape)
        # Confusion accumulates over the complete evaluation set (global mIoU).
        one = calculate_segmentation_metrics(predictions, labels[None, ...], num_classes)
        confusion += one["confusion_matrix"]
        if progress and ((index + 1) % 100 == 0 or index + 1 == count):
            print(f"Accuracy evaluation: {index + 1}/{count} images", flush=True)
    return _metrics_from_confusion(confusion) | {"num_images": count}


def measure_latency(model: Any, image: Any, image_processor: Any, device: Any,
                    warmup_iterations: int = 2, timed_iterations: int = 1000, *,
                    warmup_seconds: Optional[float] = None, height: int = 512, width: int = 512,
                    inference_fn: Optional[Callable[[Any, Any, Any], Any]] = None) -> dict[str, Any]:
    """Measure repeat inference latency with synchronized device timing."""
    if timed_iterations < 1:
        raise ValueError("timed_iterations must be positive")
    inputs = preprocess_image(image, image_processor, height=height, width=width)
    if isinstance(inputs, Mapping):
        inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
    if warmup_seconds is None:
        for _ in range(max(warmup_iterations, 0)):
            run_inference(model, inputs, device, inference_fn=inference_fn)
        actual_warmup_seconds = None
    else:
        start_warmup = time.perf_counter()
        iterations = 0
        while time.perf_counter() - start_warmup < warmup_seconds or iterations < warmup_iterations:
            run_inference(model, inputs, device, inference_fn=inference_fn)
            _sync(device)
            iterations += 1
        actual_warmup_seconds = time.perf_counter() - start_warmup
    _sync(device)
    timings_ms = []
    for _ in range(timed_iterations):
        _sync(device)
        start = time.perf_counter()
        run_inference(model, inputs, device, inference_fn=inference_fn)
        _sync(device)
        timings_ms.append((time.perf_counter() - start) * 1000.0)
    values = np.asarray(timings_ms)
    mean_ms = float(values.mean())
    return {"warmup_iterations": warmup_iterations, "warmup_seconds": actual_warmup_seconds,
            "timed_iterations": timed_iterations, "latency_mean_ms": mean_ms,
            "latency_p50_ms": float(np.percentile(values, 50)),
            "latency_p99_ms": float(np.percentile(values, 99)),
            "latency_fps": float(1000.0 / mean_ms) if mean_ms else None}


def _read_tegrastats_ram() -> Optional[dict[str, int]]:
    """Read total system RAM telemetry (not CUDA allocation) when tegrastats exists."""
    executable = shutil.which("tegrastats")
    if executable is None:
        return None
    try:
        completed = subprocess.run([executable, "--interval", "100", "--count", "1"],
                                   capture_output=True, text=True, timeout=3, check=False)
        import re
        match = re.search(r"RAM\s+(\d+)/(\d+)MB", completed.stdout + completed.stderr)
        if match:
            return {"used_mb": int(match.group(1)), "total_mb": int(match.group(2))}
    except (OSError, subprocess.TimeoutExpired):
        pass
    return None


def _ram_from_power_samples(samples: Iterable[str]) -> list[int]:
    """Extract total-system RAM use (MB) from raw course tegrastats lines."""
    values = []
    for line in samples:
        match = re.search(r"\bRAM\s+(\d+)/(\d+)MB", line)
        if match:
            values.append(int(match.group(1)))
    return values


def _power_window_class():
    """Import the course helper both as a script and as a jetson package module."""
    try:
        from .power_window import PowerWindow
    except ImportError:
        from power_window import PowerWindow
    return PowerWindow


def measure_memory(model: Any, image: Any, image_processor: Any, device: Any, *,
                   height: int = 512, width: int = 512,
                   idle_system_ram: Optional[Mapping[str, int]] = None,
                   model_loaded_system_ram: Optional[Mapping[str, int]] = None,
                   model_loaded_cuda_memory: Optional[tuple[Optional[float], Optional[float]]] = None,
                   monitor_seconds: float = 2.0,
                   inference_fn: Optional[Callable[[Any, Any, Any], Any]] = None) -> dict[str, Any]:
    """Capture peak host RAM during sustained inference and CUDA allocator usage.

    Host RAM samples come from the course tegrastats log in /course-power.log.
    CUDA allocated/reserved memory is reported separately as PyTorch allocator
    telemetry; it is not substituted for total system RAM.
    """
    try:
        torch = _torch()
    except RuntimeError:
        torch = None
    idle = dict(idle_system_ram) if idle_system_ram is not None else _read_tegrastats_ram()
    is_torch_cuda = torch is not None and (getattr(device, "type", None) == "cuda" or device == "cuda") and torch.cuda.is_available()
    if model_loaded_system_ram is None:
        model_loaded_system_ram = _read_tegrastats_ram()
    if is_torch_cuda:
        torch.cuda.reset_peak_memory_stats(device)
        if model_loaded_cuda_memory is None:
            model_loaded_cuda_memory = _cuda_memory_snapshot(device)
        loaded_allocated_mb, loaded_reserved_mb = model_loaded_cuda_memory
    else:
        loaded_allocated_mb = loaded_reserved_mb = None
    inputs = preprocess_image(image, image_processor, height=height, width=width)
    if isinstance(inputs, Mapping):
        inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
    peak_ram_samples: list[int] = []
    monitor_error = None
    window = None
    try:
        PowerWindow = _power_window_class()
        window = PowerWindow().start()
    except (ImportError, OSError, RuntimeError) as exc:
        monitor_error = str(exc)

    # Keep inference outside telemetry exception handlers: model errors, in
    # particular CUDA OOM, must propagate and must never trigger a retry.
    try:
        _sync(device)
        end_at = time.perf_counter() + max(monitor_seconds, 0.1)
        while time.perf_counter() < end_at:
            run_inference(model, inputs, device, inference_fn=inference_fn)
        _sync(device)
    except BaseException:
        if window is not None:
            try:
                window.stop()
            except Exception:
                pass
        raise
    if window is not None:
        try:
            window.stop()
            peak_ram_samples = _ram_from_power_samples(window.samples)
        except (OSError, RuntimeError, ValueError) as exc:
            monitor_error = str(exc)
    if is_torch_cuda:
        peak_allocated_mb = torch.cuda.max_memory_allocated(device) / 1024**2
        peak_reserved_mb = torch.cuda.max_memory_reserved(device) / 1024**2
    else:
        peak_allocated_mb = peak_reserved_mb = None
    return {"system_ram_idle_used_mb": idle["used_mb"] if idle else None,
            "system_ram_idle_total_mb": idle["total_mb"] if idle else None,
            "system_ram_model_loaded_used_mb": model_loaded_system_ram["used_mb"] if model_loaded_system_ram else None,
            "system_ram_peak_during_inference_used_mb": max(peak_ram_samples) if peak_ram_samples else None,
            "system_ram_inference_samples": len(peak_ram_samples),
            "system_ram_peak_status": (f"unavailable: {monitor_error}" if monitor_error else
                                       ("measured" if peak_ram_samples else "unavailable: no RAM samples")),
            "cuda_loaded_allocated_mb": loaded_allocated_mb,
            "cuda_loaded_reserved_mb": loaded_reserved_mb,
            "cuda_peak_allocated_mb": peak_allocated_mb,
            "cuda_peak_reserved_mb": peak_reserved_mb,
            "system_ram_source": "tegrastats" if idle else "unavailable",
            "cuda_memory_source": "torch.cuda" if peak_allocated_mb is not None else "unavailable"}


def measure_energy(model: Any, image: Any, image_processor: Any, device: Any, *,
                   height: int = 512, width: int = 512, idle_seconds: float = 10.0,
                   busy_seconds: float = 20.0, warmup_seconds: float = 2.0,
                   inference_fn: Optional[Callable[[Any, Any, Any], Any]] = None) -> dict[str, Any]:
    """Measure VDD_IN with the supplied PowerWindow idle/busy window protocol."""
    try:
        PowerWindow = _power_window_class()
    except ImportError as exc:
        return {"energy_per_inference_mj": None, "energy_incremental_per_inference_mj": None,
                "energy_status": f"unavailable: cannot import power_window.py ({exc})"}
    inputs = preprocess_image(image, image_processor, height=height, width=width)
    if isinstance(inputs, Mapping):
        inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
    try:
        print(f"Measuring course idle power window ({idle_seconds:g} s)", flush=True)
        idle_window = PowerWindow().start()
    except (OSError, RuntimeError, ValueError) as exc:
        return {"energy_per_inference_mj": None, "energy_incremental_per_inference_mj": None,
                "energy_status": f"unavailable: course idle PowerWindow failed ({exc})"}
    time.sleep(idle_seconds)
    try:
        idle_window.stop()
    except (OSError, RuntimeError, ValueError) as exc:
        return {"energy_per_inference_mj": None, "energy_incremental_per_inference_mj": None,
                "energy_status": f"unavailable: course idle PowerWindow failed ({exc})"}

    # Explicit model warm-up is outside the busy power window and its inference
    # count. Inference exceptions are intentionally allowed to propagate.
    warmup_started = time.perf_counter()
    warmup_count = 0
    while time.perf_counter() - warmup_started < warmup_seconds:
        run_inference(model, inputs, device, inference_fn=inference_fn)
        _sync(device)
        warmup_count += 1
    actual_warmup_seconds = time.perf_counter() - warmup_started

    busy_seconds = max(20.0, busy_seconds)
    print(f"Measuring course busy power window (at least {busy_seconds:g} s)", flush=True)
    try:
        busy_window = PowerWindow().start()
    except (OSError, RuntimeError, ValueError) as exc:
        return {"energy_per_inference_mj": None, "energy_incremental_per_inference_mj": None,
                "energy_status": f"unavailable: course busy PowerWindow failed ({exc})"}
    started = time.perf_counter()
    inference_count = 0
    try:
        _sync(device)
        while time.perf_counter() - started < busy_seconds:
            run_inference(model, inputs, device, inference_fn=inference_fn)
            _sync(device)
            inference_count += 1
    except BaseException:
        try:
            busy_window.stop()
        except Exception:
            pass
        raise
    try:
        busy_window.stop()
    except (OSError, RuntimeError, ValueError) as exc:
        return {"energy_per_inference_mj": None, "energy_incremental_per_inference_mj": None,
                "energy_status": f"unavailable: course busy PowerWindow failed ({exc})"}
    duration = busy_window.duration_s
    try:
        idle_mean_mw = idle_window.mean_mw
        busy_mean_mw = busy_window.mean_mw
    except RuntimeError as exc:
        return {"energy_per_inference_mj": None, "energy_incremental_per_inference_mj": None,
                "energy_status": f"unavailable: course PowerWindow samples invalid ({exc})"}
    total_per_inference = busy_mean_mw * duration / inference_count if inference_count else None
    idle_adjusted = ((busy_mean_mw - idle_mean_mw) * duration / inference_count
                     if inference_count else None)
    return {"energy_per_inference_mj": total_per_inference,
            "energy_incremental_per_inference_mj": idle_adjusted,
            "power_idle_mean_mw": idle_mean_mw,
            "power_busy_mean_mw": busy_mean_mw,
            "energy_idle_sample_count": idle_window.n_samples,
            "energy_busy_sample_count": busy_window.n_samples,
            "energy_busy_duration_s": duration,
            "energy_inference_count": inference_count,
            "energy_warmup_seconds": actual_warmup_seconds,
            "energy_warmup_inferences": warmup_count,
            "energy_status": "measured with course PowerWindow; total and idle-adjusted values reported"}


def count_parameters(model: Any) -> dict[str, int]:
    """Count parameters for PyTorch models; report unavailable for other runtimes."""
    if not hasattr(model, "parameters"):
        return {"total_parameters": -1, "trainable_parameters": -1}
    params = list(model.parameters())
    return {"total_parameters": sum(p.numel() for p in params),
            "trainable_parameters": sum(p.numel() for p in params if getattr(p, "requires_grad", False))}


def _json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def save_results(results: Mapping[str, Any], output_path: str | Path) -> None:
    """Append to the stable comparison CSV; reject incompatible headers."""
    row = {key: _json_value(results.get(key)) for key in RESULT_COLUMNS}
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    if exists:
        with path.open("r", newline="", encoding="utf-8") as stream:
            existing_columns = next(csv.reader(stream), [])
        if tuple(existing_columns) != RESULT_COLUMNS:
            raise ValueError(
                f"Existing results CSV has a different schema: {path}. Choose a new --output path "
                "or migrate the file explicitly; existing results were not changed."
            )
    with path.open("a" if exists else "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=RESULT_COLUMNS)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def run_evaluation(config: EvaluationConfig, *, model_loader: Optional[Callable[..., Any]] = None,
                   dataset_loader: Optional[Callable[..., Any]] = None,
                   processor_loader: Optional[Callable[[str], Any]] = None,
                   inference_fn: Optional[Callable[[Any, Any, Any], Any]] = None) -> dict[str, Any]:
    """Load configured resources and collect accuracy, latency, memory, energy metrics."""
    idle_system_ram = _read_tegrastats_ram()
    device = _resolve_device(config.device)
    model = processor = dataset = None
    try:
        print(f"Loading {config.model_name} TorchScript model {config.model_path} "
              f"from {config.checkpoint or config.model_path} ({config.precision}) on {device}", flush=True)
        processor = processor_loader(config.model_path) if processor_loader else _load_processor(config.model_path)
        model = load_model(config.model_path, config.precision, str(device), loader=model_loader)
        # Capture loaded memory before the first inference or dataset evaluation.
        model_loaded_system_ram = _read_tegrastats_ram()
        model_loaded_cuda_memory = _cuda_memory_snapshot(device)
        loader = dataset_loader or load_evaluation_dataset
        dataset = loader(config.dataset_name, config.split, processor,
                         dataset_config=config.dataset_config, num_images=config.evaluation_images)
        if not len(dataset):
            raise ValueError("Evaluation dataset is empty")
        sample = dataset[0]["image"]

        # The PowerWindow idle baseline precedes this model's first inference.
        energy = measure_energy(model, sample, processor, device,
                                height=config.input_height, width=config.input_width,
                                warmup_seconds=config.warmup_seconds, inference_fn=inference_fn)
        print("Measuring memory before accuracy evaluation", flush=True)
        memory = measure_memory(model, sample, processor, device, height=config.input_height,
                                width=config.input_width, idle_system_ram=idle_system_ram,
                                model_loaded_system_ram=model_loaded_system_ram,
                                model_loaded_cuda_memory=model_loaded_cuda_memory,
                                inference_fn=inference_fn)
        print(f"Measuring latency ({config.timed_iterations} timed inferences)", flush=True)
        latency = measure_latency(model, sample, processor, device,
                                  warmup_iterations=config.warmup_iterations,
                                  timed_iterations=config.timed_iterations, warmup_seconds=config.warmup_seconds,
                                  height=config.input_height, width=config.input_width, inference_fn=inference_fn)
        print(f"Evaluating accuracy on {len(dataset)} images", flush=True)
        accuracy = evaluate_accuracy(model, dataset, processor, device, num_classes=config.num_classes,
                                     height=config.input_height, width=config.input_width,
                                     inference_fn=inference_fn)
        unavailable = []
        if memory.get("system_ram_idle_used_mb") is None or memory.get("system_ram_model_loaded_used_mb") is None:
            unavailable.append("system RAM snapshots")
        if memory.get("system_ram_peak_status", "").startswith("unavailable"):
            unavailable.append("system RAM peak")
        if memory.get("cuda_peak_allocated_mb") is None:
            unavailable.append("CUDA allocator memory")
        if energy.get("energy_status", "").startswith("unavailable"):
            unavailable.append("energy")
        source_checkpoint = config.checkpoint or config.model_path
        results = {"model_name": config.model_name, "model": source_checkpoint,
                   "checkpoint": source_checkpoint, "architecture": config.architecture,
                   "runtime": config.runtime, "device": str(device), "dataset": config.dataset_name,
                   "dataset_config": config.dataset_config, "split": config.split,
                   "evaluation_set_id": f"{config.dataset_name}:{config.split}:first-{len(dataset)}",
                   "num_classes": config.num_classes, "input_height": config.input_height,
                   "input_width": config.input_width, "precision": config.precision,
                   "batch_size": config.batch_size, "evaluation_images_requested": config.evaluation_images,
                   "warmup_iterations_configured": config.warmup_iterations,
                   "warmup_seconds_configured": config.warmup_seconds,
                   "timed_iterations_configured": config.timed_iterations, **count_parameters(model),
                   **{k: v for k, v in accuracy.items() if k not in ("confusion_matrix", "class_iou_percent")},
                   **latency, **memory, **energy,
                   "warmup_seconds_actual": latency["warmup_seconds"],
                   "status": "partial" if unavailable else "measured",
                   "status_details": "; ".join(f"{item} unavailable" for item in unavailable) or "all configured measurements completed"}
        save_results(results, config.output_csv)
        print(f"{config.model_name}: {results['status']} - mIoU {results['mean_iou_percent']:.2f}%", flush=True)
        return results
    finally:
        # Drop this model before the next variant is loaded.
        model = processor = dataset = None
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            # Cleanup is best-effort; it must not mask an evaluation/OOM error.
            pass


def _failure_result(config: EvaluationConfig, exc: BaseException) -> dict[str, Any]:
    """Create a schema-complete failure row without fabricating measurements."""
    return {
        "model_name": config.model_name, "model": config.checkpoint or config.model_path,
        "checkpoint": config.checkpoint or config.model_path,
        "architecture": config.architecture, "runtime": config.runtime, "device": config.device,
        "dataset": config.dataset_name, "dataset_config": config.dataset_config, "split": config.split,
        "evaluation_set_id": f"{config.dataset_name}:{config.split}:first-{config.evaluation_images or 'all'}",
        "evaluation_images_requested": config.evaluation_images, "num_classes": config.num_classes,
        "input_height": config.input_height, "input_width": config.input_width,
        "precision": config.precision, "batch_size": config.batch_size,
        "warmup_seconds_configured": config.warmup_seconds,
        "warmup_iterations_configured": config.warmup_iterations,
        "timed_iterations_configured": config.timed_iterations,
        "energy_status": "not measured due to model evaluation failure",
        "system_ram_peak_status": "not measured due to model evaluation failure",
        "status": "failed", "status_details": f"{type(exc).__name__}: {exc}",
    }


def _is_cuda_oom(exc: BaseException) -> bool:
    try:
        torch = _torch()
        oom_type = getattr(torch.cuda, "OutOfMemoryError", ())
        if oom_type and isinstance(exc, oom_type):
            return True
    except RuntimeError:
        pass
    message = str(exc).lower()
    return "out of memory" in message and ("cuda" in message or "cudnn" in message)


def _recover_cuda() -> bool:
    """Release cached allocations and check that CUDA is still responsive."""
    gc.collect()
    try:
        torch = _torch()
        if not torch.cuda.is_available():
            return False
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.mem_get_info()
        return True
    except Exception:
        return False


def _print_comparison(results: list[dict[str, Any]]) -> None:
    print("\nSegFormer reference comparison")
    headers = ("Model / checkpoint", "Input", "Precision / runtime", "Images", "mIoU (%)",
               "Pixel acc. (%)", "P50 ms", "P99 ms", "Energy mJ/inf", "Peak RAM MB",
               "CUDA peak alloc/res MB", "Status")
    print(" | ".join(headers))
    print("-" * 190)
    by_name = {row["model_name"]: row for row in results}
    for name, _, _ in MODEL_VARIANTS:
        row = by_name.get(name, {})
        checkpoint = row.get("checkpoint", "")
        model_and_weights = f"{name} ({checkpoint})" if checkpoint else name
        resolution = (f"{row['input_width']}x{row['input_height']}"
                      if row.get("input_width") and row.get("input_height") else "")
        precision_runtime = (f"{row.get('precision', '')} / {row.get('runtime', '')}"
                             if row.get("precision") else "")
        cuda_peak = ""
        if row.get("cuda_peak_allocated_mb") is not None or row.get("cuda_peak_reserved_mb") is not None:
            cuda_peak = f"{row.get('cuda_peak_allocated_mb', '')}/{row.get('cuda_peak_reserved_mb', '')}"
        vals = (model_and_weights, resolution, precision_runtime, row.get("num_images", ""),
                row.get("mean_iou_percent", ""), row.get("pixel_accuracy_percent", ""),
                row.get("latency_p50_ms", ""), row.get("latency_p99_ms", ""),
                row.get("energy_per_inference_mj", ""),
                row.get("system_ram_peak_during_inference_used_mb", ""), cuda_peak,
                row.get("status", "not run"))
        print(" | ".join(str(v) for v in vals))


def check_local_models() -> None:
    """Run one real validation image through every local TorchScript model on CUDA."""
    torch = _torch()
    device = _resolve_device("cuda")
    first_model_path = MODEL_DIR / MODEL_FILES[MODEL_VARIANTS[0][0]]
    processor = _load_processor(str(first_model_path))
    dataset = load_evaluation_dataset(DATASET_ID, split="validation", num_images=1)
    if not len(dataset):
        raise ValueError("The local validation dataset has no images.")
    image = dataset[0]["image"]
    inputs = preprocess_image(image, processor, height=512, width=512)
    inputs = {key: value.to(device) if hasattr(value, "to") else value
              for key, value in inputs.items()}

    for model_name, _, _ in MODEL_VARIANTS:
        model_path = MODEL_DIR / MODEL_FILES[model_name]
        print(f"Smoke testing {model_name}: {model_path}", flush=True)
        model = load_model(str(model_path), precision="fp32", device=str(device))
        try:
            with torch.inference_mode():
                logits = _extract_logits(run_inference(model, inputs, device))
            _sync(device)
            if logits.ndim != 4 or logits.shape[0] != 1 or logits.shape[1] != NUM_CLASSES:
                raise RuntimeError(f"Unexpected model output shape: {tuple(logits.shape)}")
            if logits.dtype != torch.float32:
                raise RuntimeError(f"Expected FP32 output, got {logits.dtype}")
            if not torch.isfinite(logits).all().item():
                raise RuntimeError("Model output contains NaN or infinity")
            print(f"PASS {model_name}: CUDA {tuple(logits.shape)}, {logits.dtype}", flush=True)
        finally:
            del model
            gc.collect()
            torch.cuda.empty_cache()


def main() -> None:
    """Run B0, B1, and B2 sequentially using one shared evaluation pipeline."""
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=DEFAULT_EVALUATION_IMAGES,
                        help=f"Number of validation images from the fixed dataset order (default: {DEFAULT_EVALUATION_IMAGES})")
    parser.add_argument("--check-data", action="store_true",
                        help="Load and verify the selected local validation images, then exit before model inference")
    parser.add_argument("--check-models", action="store_true",
                        help="Run one validation image through each local TorchScript model on CUDA, then exit")
    parser.add_argument("--output", default="baseline_results.csv", help="Comparison CSV path")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.check_data:
        dataset = load_evaluation_dataset(DATASET_ID, num_images=args.limit)
        print(f"Validation file pairs ready: {len(dataset)}", flush=True)
        print("loaded successfully", flush=True)
        return
    missing_models = [str(MODEL_DIR / filename) for filename in MODEL_FILES.values()
                      if not (MODEL_DIR / filename).is_file()]
    if missing_models or not (MODEL_DIR / "preprocessing.json").is_file():
        parser.error(
            "TorchScript model files are missing. Run python3 load_model.py where Transformers is available, "
            "then copy the complete ./model directory to this project. Missing: "
            + (", ".join(missing_models) if missing_models else str(MODEL_DIR / "preprocessing.json"))
        )
    if np is None:
        raise RuntimeError("NumPy is required for model inference and accuracy metrics. Install the project requirements first.")
    if args.check_models:
        check_local_models()
        return
    results = []
    device_recovered = True
    for model_name, checkpoint, architecture in MODEL_VARIANTS:
        local_model = MODEL_DIR / MODEL_FILES[model_name]
        config = EvaluationConfig(model_name=model_name, model_path=str(local_model), checkpoint=checkpoint,
                                  architecture=architecture,
                                  precision="fp32", device="cuda", batch_size=1,
                                  input_height=512, input_width=512, evaluation_images=args.limit,
                                  output_csv=args.output)
        if not device_recovered:
            failure = RuntimeError("Skipped because CUDA did not recover after an earlier model failure")
            row = _failure_result(config, failure)
            save_results(row, config.output_csv)
            results.append(row)
            continue
        try:
            results.append(run_evaluation(config))
        except Exception as exc:
            oom = _is_cuda_oom(exc)
            details = f"CUDA OOM: {exc}" if oom else f"{type(exc).__name__}: {exc}"
            row = _failure_result(config, RuntimeError(details))
            save_results(row, config.output_csv)
            results.append(row)
            print(f"{model_name} failed; recording status and checking CUDA recovery: {details}", flush=True)
            device_recovered = _recover_cuda()
            if not device_recovered:
                print("CUDA did not recover; remaining models will be recorded as skipped.", flush=True)
        if model_name != MODEL_VARIANTS[-1][0] and device_recovered:
            print(f"Cooling down for {INTER_MODEL_COOLDOWN_SECONDS} seconds before the next model.", flush=True)
            time.sleep(INTER_MODEL_COOLDOWN_SECONDS)
    _print_comparison(results)


if __name__ == "__main__":
    main()
