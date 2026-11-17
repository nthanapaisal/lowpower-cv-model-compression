#!/usr/bin/env python3
"""Structured pruning of SegFormer-B2. Everything pruning-related is in this file.

Commands (run from the repository root):

  python3 training/prune.py prune
      Laptop or GPU machine. Loads SegFormer-B2 FP32 through load_model.py's
      checkpoint and cache and builds a ladder of candidates that remove 10%,
      20%, ... (--step, up to --max-ratio) of the channels in each pruned layer.
      The ladder is cheap; the Jetson decides where to stop (see latency).
      For each candidate, in training/model/ (git-ignored):
        segformer_b2_prunedNN.pth             (for finetune.py, and --start below)
        segformer_b2_prunedNN.torchscript.pt  (for timing in PyTorch on the Jetson)
        segformer_b2_prunedNN.onnx            (for TensorRT engines, e.g. with lab1_trt.py)
      and the metrics, in training/ (tracked by git):
        pruning_metrics.txt   (before/after report: params, MACs, MiB, channels
                               kept, sparsity, changed layers per block)
        pruning_metrics.csv   (one row per model)
      Needs: torch, transformers, torch-pruning.

  python3 training/prune.py prune --start segformer_b2_pruned10_ft.pth --ratios 0.2
      Iterative pruning: continue from a fine-tuned checkpoint. A ratio is always
      a fraction of the ORIGINAL channels, so 0.2 here removes another 10%.

  python3 training/prune.py onnx segformer_b2_pruned40_ft.pth
      Export any checkpoint (for example the fine-tuned model) to ONNX, checked
      against PyTorch with onnxruntime if it is installed.

  python3 training/prune.py latency
      Jetson only. Walks the candidates from least to most pruned and measures
      each with the course protocol (batch 1, 512x512, 2 s warm-up, 1000 timed
      runs, GPU synchronization), in --sessions separate sessions. It stops at
      the first candidate whose median P99 meets --target-ms, or earlier if two
      consecutive candidates differ in P99 by less than the spread between
      sessions: the unpruned layers then bound the latency, and more pruning
      cannot reach the target.
      Times the TorchScript files in PyTorch FP32. Needs: torch only. Energy is
      NOT measured here: every reported energy number must come from
      power_window.py (jetson/baseline.py).

  python3 training/prune.py subset --zip ADEChallengeData2016.zip
      Picks 1000 of the 20,210 training images, stratified by which of the 150
      classes each image contains, and copies them plus the 2000 validation
      images into ade20k-scene-parse-150/ (the zip's layout), ready to copy to
      the Jetson or upload to Colab. The image names go to
      training/subset_train_1000.txt. Needs: numpy, Pillow.

What is pruned (Section III of the proposal)
  * Whole channels (structured pruning): the result is a smaller dense model.
  * The hidden width of the 16 Mix-FFN (MLP) blocks, and the decoder: the four
    projections and the fuse layer. Attention, patch embeddings, LayerNorms
    and the 150 class outputs are not pruned.
  * Inside each layer, the channels with the smallest L2 norm go first.
"""

from __future__ import annotations

import argparse
import copy
import csv
import gc
import io
import re
import sys
import time
import zipfile
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
REPO_DIR = PROJECT_DIR.parent
OUTPUT_DIR = PROJECT_DIR / "model"
sys.path.insert(0, str(PROJECT_DIR))
from load_model import CHECKPOINTS  # noqa: E402  (also points the Hugging Face cache at training/.cache)

B2_CHECKPOINT = dict(CHECKPOINTS)["b2"]
DEFAULT_STEP = 0.1        # ladder step, as a fraction of the original channels
DEFAULT_MAX_RATIO = 0.9   # the ladder is cheap; the Jetson decides where to stop
INPUT_SIZE = 512
GRAPH_SIZE = 128   # input size used to trace the dependency graph (structure only)
CHECK_SIZE = 256   # input size used by the self-check
NUM_CLASSES = 150


# ==========================================================================
# Loading and saving
# ==========================================================================

def load_original(checkpoint: str = B2_CHECKPOINT):
    """SegFormer-B2 FP32, the same checkpoint and cache that load_model.py uses."""
    import torch
    from transformers import AutoModelForSemanticSegmentation
    model = AutoModelForSemanticSegmentation.from_pretrained(checkpoint)
    return model.eval().to(dtype=torch.float32)


def wrap(model):
    """pixel_values in, logits out: the same interface as load_model.py's TorchScript files."""
    from torch import nn

    class SegFormerLogits(nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.model = inner

        def forward(self, pixel_values):
            return self.model(pixel_values=pixel_values).logits

    return SegFormerLogits(model).eval()


def save_checkpoint(model, path, meta: dict | None = None) -> None:
    """Write a .pth that load_checkpoint() can rebuild without Torch-Pruning.

    finetune.py should save its result with this function too, so that the
    next pruning round can start from it (prune --start <file>).
    """
    import torch
    torch.save({"state_dict": model.state_dict(), "config": model.config.to_dict(),
                "meta": meta or {}}, str(path))


def export_torchscript(model, path) -> None:
    """Traced FP32 model for the Jetson, loadable with torch.jit.load (no Transformers)."""
    import torch
    wrapper = wrap(model).cpu().eval()
    example = torch.zeros(1, 3, INPUT_SIZE, INPUT_SIZE)
    with torch.inference_mode():
        eager = wrapper(example)
        traced = torch.jit.trace(wrapper, example, strict=False, check_trace=False)
        torch.testing.assert_close(traced(example), eager, rtol=1e-4, atol=1e-4)
    torch.jit.save(traced, str(path))


def export_onnx(model, path, check: bool = True) -> None:
    """ONNX (opset 17) with input "input" [1,3,512,512] and output "logits", like export.py.

    TensorRT reads this file with its built-in ONNX parser. If onnxruntime is
    installed, the exported model is run once and compared with PyTorch.
    """
    import torch
    wrapper = wrap(model).cpu().eval()
    example = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE)
    export_kwargs = {"input_names": ["input"], "output_names": ["logits"], "opset_version": 17}
    try:
        torch.onnx.export(wrapper, example, str(path), dynamo=False, **export_kwargs)
    except TypeError:  # older PyTorch without the dynamo argument
        torch.onnx.export(wrapper, example, str(path), **export_kwargs)
    if not check:
        return
    try:
        import numpy as np
        import onnxruntime
    except ImportError:
        return
    session = onnxruntime.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    with torch.inference_mode():
        expected = wrapper(example).numpy()
    got = session.run(None, {"input": example.numpy()})[0]
    error = float(np.abs(got - expected).max() / (np.abs(expected).max() or 1.0))
    if error > 1e-3:
        raise RuntimeError(f"ONNX output differs from PyTorch (relative error {error:.2e}): {path}")


def save_preprocessing(checkpoint: str, output_dir: Path) -> None:
    """Write output_dir/preprocessing.json (same fields as load_model.py) if it is missing.

    finetune.py, evaluate.py and jetson/baseline.py read the input size, mean and
    std from this file, next to the model files.
    """
    import json
    from transformers import AutoImageProcessor
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / "preprocessing.json"
    if target.is_file():
        return
    processor = AutoImageProcessor.from_pretrained(checkpoint)
    target.write_text(json.dumps({
        "height": INPUT_SIZE, "width": INPUT_SIZE,
        "image_mean": [float(v) for v in processor.image_mean],
        "image_std": [float(v) for v in processor.image_std],
        "do_rescale": bool(processor.do_rescale), "rescale_factor": float(processor.rescale_factor),
        "do_normalize": bool(processor.do_normalize), "resample": int(processor.resample),
        "source_checkpoint": checkpoint}, indent=2) + "\n", encoding="utf-8")


def _is_depthwise(module) -> bool:
    from torch import nn
    return (isinstance(module, nn.Conv2d) and module.groups > 1
            and module.groups == module.in_channels == module.out_channels)


def _resize_to(model, state_dict: dict) -> None:
    """Shrink the layers of a full-size model to the shapes in a pruned state dict."""
    import torch
    from torch import nn
    for name, module in model.named_modules():
        weight = state_dict.get(f"{name}.weight")
        if weight is None or getattr(module, "weight", None) is None or module.weight.shape == weight.shape:
            continue
        if isinstance(module, nn.Linear):
            module.out_features, module.in_features = weight.shape
        elif isinstance(module, nn.Conv2d):
            depthwise = _is_depthwise(module)
            module.out_channels = weight.shape[0]
            module.groups = weight.shape[0] if depthwise else module.groups
            module.in_channels = weight.shape[1] * module.groups
        elif isinstance(module, nn.BatchNorm2d):
            module.num_features = weight.shape[0]
            module.running_mean = torch.zeros(weight.shape[0])
            module.running_var = torch.ones(weight.shape[0])
        else:
            raise RuntimeError(f"Unexpected pruned layer at {name}: {type(module).__name__}")
        module.weight = nn.Parameter(torch.empty_like(weight))
        if getattr(module, "bias", None) is not None:
            module.bias = nn.Parameter(torch.empty(weight.shape[0]))


def load_checkpoint(path, checkpoint: str = B2_CHECKPOINT):
    """Rebuild a (pruned, possibly fine-tuned) SegFormer from a .pth file.

    Accepts a file written by save_checkpoint(), or a plain state_dict (keys may
    start with "model." or "module."), in which case the architecture settings
    come from the original checkpoint's config.
    """
    import torch
    from transformers import AutoConfig, SegformerConfig, SegformerForSemanticSegmentation
    saved = torch.load(str(path), map_location="cpu", weights_only=False)
    if isinstance(saved, dict) and "state_dict" in saved:
        state_dict = saved["state_dict"]
        config = SegformerConfig.from_dict(saved["config"]) if "config" in saved \
            else AutoConfig.from_pretrained(checkpoint)
    else:
        state_dict, config = saved, AutoConfig.from_pretrained(checkpoint)
    for prefix in ("module.", "model."):
        if all(k.startswith(prefix) for k in state_dict):
            state_dict = {k[len(prefix):]: v for k, v in state_dict.items()}
    model = SegformerForSemanticSegmentation(config)
    _resize_to(model, state_dict)
    model.load_state_dict(state_dict, strict=True)
    return model.eval()


# ==========================================================================
# What to prune
# ==========================================================================

def _out_width(layer) -> int:
    from torch import nn
    return layer.out_features if isinstance(layer, nn.Linear) else layer.out_channels


def _stage_of(name: str) -> int | None:
    match = re.search(r"(?:stages|block)\.(\d+)\.", name)
    return int(match.group(1)) if match else None


def find_targets(model) -> list[dict]:
    """The 21 layers whose output channels are removed.

    Found by structure, not by name (names differ between Transformers versions):
      "mlp":     first Linear of each Mix-FFN (two Linears around one depthwise conv)
      "decoder": the four decoder projections and the fuse layer
    Each target also records its ORIGINAL width, from the model config, so that a
    ratio always means "fraction of the original channels removed".
    """
    from torch import nn
    config = model.config
    targets: list[dict] = []
    for name, module in model.named_modules():
        linears = [(n, c) for n, c in module.named_children() if isinstance(c, nn.Linear)]
        depthwise = [m for m in module.modules() if _is_depthwise(m)]
        if len(linears) == 2 and len(depthwise) == 1:
            first, second = linears[0][1], linears[1][1]
            if first.out_features == depthwise[0].out_channels == second.in_features:
                stage = _stage_of(name + ".")
                original = config.hidden_sizes[stage] * config.mlp_ratios[stage]
                targets.append({"name": f"{name}.{linears[0][0]}", "layer": first, "kind": "mlp",
                                "stage": stage, "original": original})
    head_name, head = next(((n, m) for n, m in model.named_modules() if hasattr(m, "linear_fuse")), (None, None))
    if head is None:
        raise RuntimeError("Could not find the SegFormer decode head (no linear_fuse layer).")
    for name, module in head.named_modules():
        if isinstance(module, nn.Linear):
            targets.append({"name": f"{head_name}.{name}", "layer": module, "kind": "decoder",
                            "stage": None, "original": config.decoder_hidden_size})
    targets.append({"name": f"{head_name}.linear_fuse", "layer": head.linear_fuse, "kind": "decoder",
                    "stage": None, "original": config.decoder_hidden_size})
    if sum(t["kind"] == "mlp" for t in targets) != sum(config.depths):
        raise RuntimeError("Did not find one Mix-FFN per encoder block.")
    return targets


# ==========================================================================
# Pruning
# ==========================================================================

def _importance():
    import torch_pruning as tp
    cls = getattr(tp.importance, "GroupMagnitudeImportance", None) or tp.importance.GroupNormImportance
    return cls(p=2)  # L2 norm of each channel across all the layers it lives in


def _mask_like_pruning(reference, names: dict, group) -> None:
    """Zero, in an unpruned copy, exactly the weights the group is about to remove."""
    import torch
    modules = dict(reference.named_modules())
    for dep, idxs in group:
        module = modules.get(names.get(id(dep.target.module), ""))
        if module is None or getattr(module, "weight", None) is None:
            continue
        with torch.no_grad():
            if "out_channels" in dep.handler.__name__:
                module.weight[idxs] = 0
                if getattr(module, "bias", None) is not None:
                    module.bias[idxs] = 0
            elif not _is_depthwise(module):
                module.weight[:, idxs] = 0


def _prune_batch(wrapper, batch: list[dict], ratios: dict, example, verify: bool) -> None:
    import torch
    import torch_pruning as tp
    from torch import nn
    graph = tp.DependencyGraph().build_dependency(wrapper, example_inputs=example[..., :GRAPH_SIZE, :GRAPH_SIZE])
    importance = _importance()
    names = {id(m): n for n, m in wrapper.named_modules()}
    reference = copy.deepcopy(wrapper) if verify else None
    changed = False
    for target in batch:
        layer = target["layer"]
        keep = max(8, int(round(target["original"] * (1.0 - ratios[target["kind"]]))))
        remove = _out_width(layer) - keep
        if remove <= 0:
            continue
        prune_fn = tp.prune_linear_out_channels if isinstance(layer, nn.Linear) else tp.prune_conv_out_channels
        scores = importance(graph.get_pruning_group(layer, prune_fn, idxs=list(range(_out_width(layer)))))
        idxs = sorted(torch.argsort(scores)[:remove].tolist())
        group = graph.get_pruning_group(layer, prune_fn, idxs=idxs)
        if not graph.check_pruning_group(group):
            raise RuntimeError(f"Torch-Pruning refused to prune {target['name']}")
        if reference is not None:
            _mask_like_pruning(reference, names, group)
        group.prune()
        changed = True
    del graph
    gc.collect()
    if reference is not None and changed:
        check = example[..., :CHECK_SIZE, :CHECK_SIZE]
        with torch.inference_mode():
            wanted, got = reference(check), wrapper(check)
        error = (wanted - got).abs().max().item() / (wanted.abs().max().item() or 1.0)
        if error > 1e-3:
            raise RuntimeError(f"Self-check failed (relative error {error:.2e}): the pruned model does not "
                               "match the original with the same channels zeroed. Do not use this output.")


def prune_model(model, ratio: float, mlp_scale: float = 1.0, decoder_scale: float = 1.0,
                verify: bool = True):
    """Prune `model` in place so each target keeps (1 - ratio) of its ORIGINAL channels.

    MLP targets use ratio * mlp_scale, decoder targets ratio * decoder_scale.
    Self-check: each batch is compared with a copy in which the same channels are
    zeroed instead of removed; the outputs must match. The four decoder
    projections feed one shared layer, so each is pruned in its own batch.
    """
    import torch
    from torch import nn
    wrapper = wrap(model)
    torch.manual_seed(0)
    example = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE)
    targets = find_targets(model)
    ratios = {"mlp": min(ratio * mlp_scale, 0.95), "decoder": min(ratio * decoder_scale, 0.95)}
    projections = [t for t in targets if t["kind"] == "decoder" and isinstance(t["layer"], nn.Linear)]
    others = [t for t in targets if not any(t is p for p in projections)]
    for batch in [others] + [[p] for p in projections]:
        _prune_batch(wrapper, batch, ratios, example, verify)
    return model


# ==========================================================================
# Metrics
# ==========================================================================

def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters())


def _part(name: str) -> str:
    if "decode_head" in name:
        if "linear_fuse" in name:
            return "decoder fuse"
        if "classifier" in name:
            return "decoder classifier"
        return "decoder projections"
    if ".mlp." in name:
        return "encoder MLP (pruned)"
    if ".attention." in name:
        return "attention layers (not pruned)"
    if "patch_embeddings" in name:
        return "patch embeddings (not pruned)"
    return "other"


def count_macs_by_part(model) -> dict:
    """Multiply-accumulates (G) for one 512x512 image, by part of the model.

    Linear and convolution layers are counted with hooks; the two attention
    matrix products (Q*K^T and attention*V) with the formula from the config
    (attention is never pruned). The total is the quantity the SegFormer paper
    reports as "FLOPs" (62.4 G for B2).
    """
    import torch
    from torch import nn
    parts: dict = {}
    names = {id(m): n for n, m in model.named_modules()}

    def hook(module, inputs, output):
        if isinstance(module, nn.Linear):
            macs = output.numel() // output.shape[-1] * module.in_features * module.out_features
        else:
            macs = output.numel() * (module.in_channels // module.groups) * \
                module.kernel_size[0] * module.kernel_size[1]
        part = _part(names[id(module)])
        parts[part] = parts.get(part, 0) + macs

    handles = [m.register_forward_hook(hook) for m in model.modules() if isinstance(m, (nn.Linear, nn.Conv2d))]
    try:
        with torch.inference_mode():
            wrap(model)(torch.zeros(1, 3, INPUT_SIZE, INPUT_SIZE))
    finally:
        for handle in handles:
            handle.remove()
    config, resolution, products = model.config, INPUT_SIZE, 0
    for stage in range(len(config.depths)):
        resolution //= config.strides[stage]
        tokens = resolution * resolution
        keys = (resolution // config.sr_ratios[stage]) ** 2
        products += config.depths[stage] * 2 * tokens * keys * config.hidden_sizes[stage]
    parts["attention products QK^T, AV (not pruned)"] = products
    return {part: value / 1e9 for part, value in parts.items()}


def count_macs(model) -> float:
    """Total multiply-accumulates (G) for one 512x512 image."""
    return sum(count_macs_by_part(model).values())


def size_mib(model) -> float:
    """Size of the FP32 state dict as saved with torch.save, in MiB."""
    import torch
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return buffer.tell() / 2**20


def layer_shapes(model) -> dict:
    return {name: tuple(module.weight.shape) for name, module in model.named_modules()
            if getattr(module, "weight", None) is not None}


def channels(model) -> list[dict]:
    """Current and original width of every pruned target, in model order."""
    rows = []
    counters: dict = {}
    for target in find_targets(model):
        if target["kind"] == "mlp":
            counters["block"] = counters.get("block", 0) + 1
            stage = target["stage"]
            counters[stage] = counters.get(stage, 0) + 1
            label = f"block {counters['block']} (stage {stage + 1}, block {counters[stage]})"
        elif target["name"].endswith("linear_fuse"):
            label = "decoder fuse"
        else:
            counters["proj"] = counters.get("proj", 0) + 1
            label = f"decoder projection {counters['proj']} (from stage {counters['proj']})"
        rows.append({"label": label, "kind": target["kind"], "name": target["name"],
                     "original": target["original"], "kept": _out_width(target["layer"])})
    return rows


def measure(model, reference_shapes: dict | None = None) -> dict:
    """All metrics for one model. Pass the unpruned model's layer_shapes() to list changed layers."""
    widths = channels(model)
    shapes = layer_shapes(model)
    changed = []
    if reference_shapes is not None:
        changed = [(name, reference_shapes[name], shape) for name, shape in shapes.items()
                   if reference_shapes.get(name) != shape]
    macs_by_part = count_macs_by_part(model)
    return {"params": count_params(model), "macs_g": sum(macs_by_part.values()), "macs_by_part": macs_by_part,
            "size_mib": size_mib(model),
            "channels_kept": sum(w["kept"] for w in widths),
            "channels_original": sum(w["original"] for w in widths),
            "widths": widths, "changed_layers": changed}


def _block_lines(result: dict) -> list[str]:
    """One line per pruned block, in the requested format, plus the layers that changed in it."""
    changed = result["changed_layers"]
    lines = []
    for w in result["widths"]:
        unit = "expanded channels" if w["kind"] == "mlp" else "channels"
        if w["kind"] == "mlp":
            prefix = w["name"].rsplit(".", 1)[0] + "."
            touched = [n[len(prefix):] for n, _, _ in changed if n.startswith(prefix)]
            detail = f"  [{', '.join(touched)}]" if touched else ""
        else:
            detail = ""
        lines.append(f"{w['label']}: {w['original']} -> {w['kept']} {unit}{detail}")
    decoder = [(n, old, new) for n, old, new in changed if n.startswith("decode_head")]
    if decoder:
        lines.append("decoder layers that changed shape (weight shape before -> after):")
        lines += [f"  {n}: {old} -> {new}" for n, old, new in decoder]
    return lines


def write_metrics(results: list[dict], txt_path: Path, csv_path: Path, note: str = "") -> None:
    """Write the before/after report (text) and one CSV row per model."""
    before = results[0]
    lines = ["SegFormer-B2 structured pruning: metrics", "=" * 60]
    if note:
        lines += [note, ""]
    lines += ["Definitions",
              "  params           all weights and biases",
              "  MACs             multiply-accumulates for one 512x512 image (the SegFormer paper calls these FLOPs)",
              "  file size        FP32 state dict saved with torch.save",
              "  channels kept    output channels left in the 21 pruned layers (16 MLP + 5 decoder)",
              "  sparsity         share removed; structured pruning leaves no zeros, the layers are smaller",
              "",
              f"Unpruned MACs by part (total {before['macs_g']:.2f} G):"]
    for part, value in sorted(before["macs_by_part"].items(), key=lambda kv: -kv[1]):
        lines.append(f"  {part:<42} {value:6.2f} G  {value / before['macs_g']:6.1%}")
    lines += [""]
    for result in results[1:]:
        params_sparsity = 1 - result["params"] / before["params"]
        channel_sparsity = 1 - result["channels_kept"] / result["channels_original"]
        lines += [f"{result['name']}  (ratio {result['ratio']:.0%}"
                  + (f", MLP {result['mlp_ratio']:.0%}, decoder {result['decoder_ratio']:.0%}"
                     if result["mlp_ratio"] != result["decoder_ratio"] else "")
                  + (f", continued from {result['start']}" if result.get("start") else "") + ")",
                  "-" * 60,
                  f"1. params:         {before['params']:,} -> {result['params']:,}  "
                  f"({before['params'] / 1e6:.2f} M -> {result['params'] / 1e6:.2f} M)",
                  f"2. MACs:           {before['macs_g']:.2f} G -> {result['macs_g']:.2f} G  "
                  f"({before['macs_g'] / result['macs_g']:.2f}x fewer)",
                  f"3. file size:      {before['size_mib']:.1f} MiB -> {result['size_mib']:.1f} MiB",
                  f"4. channels kept:  {before['channels_kept']:,} -> {result['channels_kept']:,} "
                  f"of {result['channels_original']:,}",
                  f"5. sparsity:       {params_sparsity:.1%} of parameters removed, "
                  f"{channel_sparsity:.1%} of channels removed in the pruned layers, "
                  f"{1 - result['macs_g'] / before['macs_g']:.1%} of MACs removed",
                  "6. layers changed, by block:"]
        lines += [f"   {line}" for line in _block_lines(result)]
        lines += [""]
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    fields = ["model", "ratio", "mlp_ratio", "decoder_ratio", "params", "params_m", "macs_g", "size_mib",
              "channels_kept", "channels_original", "params_removed_percent", "channels_removed_percent",
              "macs_removed_percent", "pth_file", "torchscript_file", "onnx_file"]
    fields += [w["label"] for w in before["widths"]]
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for result in results:
            row = {"model": result["name"], "ratio": result["ratio"], "mlp_ratio": result["mlp_ratio"],
                   "decoder_ratio": result["decoder_ratio"], "params": result["params"],
                   "params_m": round(result["params"] / 1e6, 3), "macs_g": round(result["macs_g"], 3),
                   "size_mib": round(result["size_mib"], 2), "channels_kept": result["channels_kept"],
                   "channels_original": result["channels_original"],
                   "params_removed_percent": round(100 * (1 - result["params"] / before["params"]), 2),
                   "channels_removed_percent": round(100 * (1 - result["channels_kept"] / result["channels_original"]), 2),
                   "macs_removed_percent": round(100 * (1 - result["macs_g"] / before["macs_g"]), 2),
                   "pth_file": result.get("pth", ""), "torchscript_file": result.get("torchscript", ""),
                   "onnx_file": result.get("onnx", "")}
            row.update({w["label"]: w["kept"] for w in result["widths"]})
            writer.writerow(row)


def run_prune(model, ratios, mlp_scale: float, decoder_scale: float, output_dir: Path,
              prefix: str = "segformer_b2", start: str | None = None, verify: bool = True,
              note: str = "", metrics_dir: Path | None = None) -> list[dict]:
    """Prune step by step, save every candidate, and write the metrics files."""
    from transformers import SegformerForSemanticSegmentation
    output_dir.mkdir(parents=True, exist_ok=True)
    # "Before" is always the unpruned B2. Parameter, MAC, size and channel
    # counts depend only on the shapes, so a fresh copy of the architecture
    # gives the same numbers even when continuing from a pruned checkpoint.
    original = SegformerForSemanticSegmentation(copy.deepcopy(model.config)).eval()
    reference_shapes = layer_shapes(original)
    results = [dict(measure(original), name=f"{prefix} (unpruned)", ratio=0.0, mlp_ratio=0.0, decoder_ratio=0.0)]
    del original
    print(f"Unpruned: {results[0]['params'] / 1e6:.2f} M params, {results[0]['macs_g']:.2f} G MACs, "
          f"{results[0]['size_mib']:.1f} MiB", flush=True)
    for ratio in sorted(ratios):
        prune_model(model, ratio, mlp_scale, decoder_scale, verify=verify)
        name = f"{prefix}_pruned{int(round(ratio * 100)):02d}"
        meta = {"source_checkpoint": B2_CHECKPOINT, "ratio": ratio, "mlp_ratio": ratio * mlp_scale,
                "decoder_ratio": ratio * decoder_scale, "criterion": "L2 norm per channel, per layer",
                "started_from": start, "fine_tuned": False}
        save_checkpoint(model, output_dir / f"{name}.pth", meta)
        export_torchscript(model, output_dir / f"{name}.torchscript.pt")
        export_onnx(model, output_dir / f"{name}.onnx")
        result = dict(measure(model, reference_shapes), name=name, ratio=ratio, mlp_ratio=min(ratio * mlp_scale, 0.95),
                      decoder_ratio=min(ratio * decoder_scale, 0.95), start=start,
                      pth=f"{name}.pth", torchscript=f"{name}.torchscript.pt", onnx=f"{name}.onnx")
        results.append(result)
        print(f"{name}: {result['params'] / 1e6:.2f} M params, {result['macs_g']:.2f} G MACs, "
              f"{result['size_mib']:.1f} MiB, self-check {'passed' if verify else 'skipped'}", flush=True)
    metrics_dir = metrics_dir or PROJECT_DIR
    stem = "pruning_metrics" + (f"_from_{Path(start).stem}" if start else "")
    write_metrics(results, metrics_dir / f"{stem}.txt", metrics_dir / f"{stem}.csv", note)
    print(f"Metrics: {metrics_dir / (stem + '.txt')} and {stem}.csv", flush=True)
    return results


# ==========================================================================
# Latency on the Jetson
# ==========================================================================

def measure_latency(path, device: str = "cuda", warmup_seconds: float = 2.0, iterations: int = 1000) -> dict:
    """Course protocol: batch 1, 512x512, warm-up excluded, GPU-synchronized timing."""
    import numpy as np
    import torch
    if device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is not available. Run the latency sweep on the Jetson.")
    model = torch.jit.load(str(path), map_location=device).eval()
    image = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE, device=device)

    def sync():
        if device == "cuda":
            torch.cuda.synchronize()

    with torch.inference_mode():
        start = time.perf_counter()
        while time.perf_counter() - start < warmup_seconds:
            model(image)
            sync()
        timings = []
        for _ in range(iterations):
            sync()
            began = time.perf_counter()
            model(image)
            sync()
            timings.append((time.perf_counter() - began) * 1000.0)
    values = np.asarray(timings)
    result = {"params": sum(p.numel() for p in model.parameters()), "mean_ms": float(values.mean()),
              "p50_ms": float(np.percentile(values, 50)), "p99_ms": float(np.percentile(values, 99)),
              "fps": float(1000.0 / values.mean())}
    del model
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    return result


def _ratio_of(path) -> float:
    """0.0 for the unpruned model, NN/100 for segformer_b2_prunedNN."""
    match = re.search(r"pruned(\d+)", Path(path).name)
    return int(match.group(1)) / 100 if match else 0.0


def refine_suggestion(rows: list[dict], target_ms: float) -> float | None:
    """Ratio to build next, halving the interval around the target, or None when done.

    Refines only while the two candidates around the target are more than one
    percentage point apart and their P99 differs by more than the session
    spread (a finer step could not be told apart from noise otherwise).
    """
    if len(rows) < 2 or rows[-1]["p99_ms"] > target_ms or rows[-2]["p99_ms"] <= target_ms:
        return None
    low, high = _ratio_of(rows[-2]["model"]), _ratio_of(rows[-1]["model"])
    noise = max(rows[-2]["spread_ms"], rows[-1]["spread_ms"])
    if high - low <= 0.01 or rows[-2]["p99_ms"] - rows[-1]["p99_ms"] <= noise:
        return None
    return round((low + high) / 2, 2)


def stopping_decision(rows: list[dict], target_ms: float) -> str | None:
    """The proposal's stopping rule, applied after each new candidate.

    rows: measured so far, least pruned first, each with p99_ms (median over
    sessions) and spread_ms (max - min P99 over sessions).
    Returns "target" if the last candidate meets the target, "floor" if the
    last two pruned candidates differ in P99 by less than their session spread
    (the unpruned layers bound the latency), or None to keep going.
    """
    last = rows[-1]
    if "pruned" in last["model"] and last["p99_ms"] <= target_ms:
        return "target"
    pruned = [row for row in rows if "pruned" in row["model"]]
    if len(pruned) >= 2:
        previous, current = pruned[-2], pruned[-1]
        if previous["p99_ms"] - current["p99_ms"] < max(previous["spread_ms"], current["spread_ms"]):
            return "floor"
    return None


def run_latency(files: list[Path], target_ms: float, device: str, iterations: int, sessions: int,
                cooldown: float, output: Path, stop: bool = True) -> list[dict]:
    """Measure candidates in order (least pruned first) and apply the stopping rule."""
    import numpy as np
    rows, decision = [], None
    for index, path in enumerate(files):
        p99s, p50s, fps = [], [], []
        for session in range(sessions):
            print(f"Measuring {path.name}, session {session + 1}/{sessions}", flush=True)
            result = measure_latency(path, device=device, iterations=iterations)
            p99s.append(result["p99_ms"]), p50s.append(result["p50_ms"]), fps.append(result["fps"])
            if cooldown > 0:
                time.sleep(cooldown)
        rows.append({"model": path.name, "params": result["params"],
                     "p50_ms": round(float(np.median(p50s)), 3), "p99_ms": round(float(np.median(p99s)), 3),
                     "spread_ms": round(max(p99s) - min(p99s), 3), "fps": round(float(np.median(fps)), 3),
                     "p99_sessions_ms": " ".join(f"{v:.3f}" for v in p99s), "target_ms": target_ms,
                     "meets_target": float(np.median(p99s)) <= target_ms})
        decision = stopping_decision(rows, target_ms)
        if stop and decision:
            break
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n{'model':<44}{'params (M)':>11}{'P50 ms':>9}{'P99 ms':>9}{'spread':>8}{'FPS':>8}  <= {target_ms:g} ms")
    for row in rows:
        print(f"{row['model']:<44}{row['params'] / 1e6:>11.2f}{row['p50_ms']:>9.2f}{row['p99_ms']:>9.2f}"
              f"{row['spread_ms']:>8.2f}{row['fps']:>8.2f}  {'yes' if row['meets_target'] else 'no'}")
    if decision == "target":
        print(f"\nPruning target: {rows[-1]['model']} is the least pruned candidate under {target_ms:g} ms.")
        nxt = refine_suggestion(rows, target_ms)
        if nxt is not None:
            print(f"Refine: python3 training/prune.py prune --ratios {nxt}   (then run latency again)")
        else:
            print("No finer step is worth measuring: this is the pruning target.")
    elif decision == "floor":
        print(f"\nLatency floor: {rows[-2]['model']} and {rows[-1]['model']} differ by less than the session "
              "spread. Further pruning cannot reach the target.")
    else:
        print("\nNo stopping point yet: build more candidates (prune --max-ratio) and measure again.")
    print(f"Saved {output}", flush=True)
    return rows


# ==========================================================================
# Stratified training subset (for fine-tuning the pruned models)
# ==========================================================================

def _zip_pairs(archive: zipfile.ZipFile, split: str) -> list[tuple[str, str]]:
    names = set(archive.namelist())
    images = sorted(n for n in names if re.search(rf"images/{split}/[^/]+\.jpg$", n))
    pairs = []
    for image in images:
        annotation = image.replace(f"images/{split}/", f"annotations/{split}/")[:-4] + ".png"
        if annotation not in names:
            raise SystemExit(f"Missing annotation for {image}")
        pairs.append((image, annotation))
    return pairs


def class_presence(zip_path: Path, split: str = "training", cache: Path | None = None,
                   time_budget: float | None = None):
    """Boolean matrix [images x 150]: which classes appear in each annotation.

    A class counts as present if at least one pixel has it (0 = "other" is ignored).
    Resumable: progress is saved to `cache`, so a stopped run continues where it left off.
    """
    import numpy as np
    from PIL import Image
    started = time.time()
    with zipfile.ZipFile(zip_path) as archive:
        pairs = _zip_pairs(archive, split)
        presence = np.zeros((len(pairs), NUM_CLASSES), dtype=bool)
        done = 0
        if cache is not None and cache.is_file():
            saved = np.load(cache)
            if saved["presence"].shape == presence.shape:
                presence, done = saved["presence"], int(saved["done"])
        for index in range(done, len(pairs)):
            with archive.open(pairs[index][1]) as stream:
                labels = np.asarray(Image.open(stream))
            present = np.unique(labels)
            present = present[(present >= 1) & (present <= NUM_CLASSES)] - 1
            presence[index, present] = True
            done = index + 1
            if done % 2000 == 0:
                print(f"  read {done}/{len(pairs)} {split} annotations", flush=True)
                if cache is not None:
                    np.savez(cache, presence=presence, done=done)
            if time_budget is not None and time.time() - started > time_budget:
                break
    if cache is not None:
        np.savez(cache, presence=presence, done=done)
    return [p[0] for p in pairs], presence, done == len(pairs)


def stratified_subset(presence, size: int, seed: int = 0):
    """Pick `size` images whose class frequencies match the full set (multi-label).

    Iterative stratification (Sechidis et al., 2011): repeatedly take the class
    with the fewest images left and hand each of its images to the subset (or to
    the rest) that still needs that class most. Rare classes are placed first,
    so they are not crowded out by common ones. Then the subset is trimmed or
    filled to exactly `size`, moving images that hurt the class balance least.
    """
    import numpy as np
    rng = np.random.default_rng(seed)
    count, classes = presence.shape
    fraction = size / count
    need = np.array([fraction * presence.sum(0), (1 - fraction) * presence.sum(0)], dtype=float)
    need_total = np.array([size, count - size], dtype=float)
    chosen = np.zeros(count, dtype=bool)
    left = np.ones(count, dtype=bool)
    while left.any():
        remaining = presence[left].sum(0)
        if remaining.max() == 0:   # images without any of the 150 classes
            for index in rng.permutation(np.flatnonzero(left)):
                side = 0 if need_total[0] >= need_total[1] else 1
                chosen[index] = side == 0
                need_total[side] -= 1
                left[index] = False
            break
        label = int(np.argmin(np.where(remaining > 0, remaining, np.inf)))
        for index in rng.permutation(np.flatnonzero(left & presence[:, label])):
            if need[0, label] != need[1, label]:
                side = int(np.argmax(need[:, label]))
            elif need_total[0] != need_total[1]:
                side = int(np.argmax(need_total))
            else:
                side = int(rng.integers(2))
            chosen[index] = side == 0
            need[side] -= presence[index]
            need_total[side] -= 1
            left[index] = False
    target = fraction * presence.sum(0)
    while chosen.sum() != size:
        counts = presence[chosen].sum(0)
        surplus = counts - target
        if chosen.sum() > size:   # drop the image whose classes are most over-represented
            pool = np.flatnonzero(chosen)
            score = (presence[pool] * surplus).sum(1) / np.maximum(presence[pool].sum(1), 1)
            chosen[pool[np.argmax(score + rng.random(len(pool)) * 1e-9)]] = False
        else:                     # add the image whose classes are most under-represented
            pool = np.flatnonzero(~chosen)
            score = (presence[pool] * -surplus).sum(1) / np.maximum(presence[pool].sum(1), 1)
            chosen[pool[np.argmax(score + rng.random(len(pool)) * 1e-9)]] = True
    return np.flatnonzero(chosen)


def subset_balance(presence, chosen) -> dict:
    import numpy as np
    full = presence.sum(0)
    picked = presence[chosen].sum(0)
    expected = full * len(chosen) / len(presence)
    present = full > 0
    return {"classes_in_full_set": int(present.sum()), "classes_in_subset": int((picked > 0).sum()),
            "min_images_per_class": int(picked[present].min()),
            "max_abs_deviation_images": float(np.abs(picked - expected)[present].max()),
            "full": full, "picked": picked, "expected": expected}


def write_subset(zip_path: Path, chosen_images: list[str], out_dir: Path, include_validation: bool = True) -> None:
    """Copy the chosen training pairs (and all validation pairs) out of the zip, keeping its layout."""
    with zipfile.ZipFile(zip_path) as archive:
        pairs = [(image, image.replace("images/training/", "annotations/training/")[:-4] + ".png")
                 for image in chosen_images]
        if include_validation:
            pairs += _zip_pairs(archive, "validation")
        for image, annotation in pairs:
            for member in (image, annotation):
                relative = member.split("ADEChallengeData2016/", 1)[-1]
                target = out_dir / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(member) as source:
                    target.write_bytes(source.read())


def run_subset(zip_path: Path, size: int, seed: int, out_dir: Path, list_path: Path,
               time_budget: float | None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = out_dir / ".presence_training.npz"   # inside the git-ignored data folder
    images, presence, complete = class_presence(zip_path, "training", cache=cache, time_budget=time_budget)
    if not complete:
        print(f"Paused after the time budget; run the same command again to continue ({cache}).")
        return
    chosen = stratified_subset(presence, size, seed)
    balance = subset_balance(presence, chosen)
    chosen_images = [images[i] for i in chosen]
    list_path.write_text("".join(Path(name).name + "\n" for name in chosen_images), encoding="utf-8")
    with (list_path.with_suffix(".classes.csv")).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["class_id", "images_in_full_set", "images_in_subset", "expected_in_subset"])
        for c in range(NUM_CLASSES):
            writer.writerow([c + 1, int(balance["full"][c]), int(balance["picked"][c]),
                             round(float(balance["expected"][c]), 2)])
    print(f"Chosen {len(chosen)} of {len(images)} training images (seed {seed}).")
    print(f"Classes present: {balance['classes_in_subset']} of {balance['classes_in_full_set']}; "
          f"fewest images for any class: {balance['min_images_per_class']}; "
          f"largest gap from the proportional count: {balance['max_abs_deviation_images']:.1f} images.")
    print(f"Image list: {list_path}")
    write_subset(zip_path, chosen_images, out_dir)
    print(f"Copied {len(chosen)} training and all validation image/annotation pairs to {out_dir}")


# ==========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)

    p = commands.add_parser("prune", help="build pruned candidates and the metrics files")
    p.add_argument("--step", type=float, default=DEFAULT_STEP, help="ladder step (fraction of original channels)")
    p.add_argument("--max-ratio", type=float, default=DEFAULT_MAX_RATIO, help="last rung of the ladder")
    p.add_argument("--ratios", type=float, nargs="+", default=None,
                   help="explicit ratios instead of the ladder, e.g. to refine between two rungs")
    p.add_argument("--mlp-scale", type=float, default=1.0, help="multiplier on the ratio for the MLP layers")
    p.add_argument("--decoder-scale", type=float, default=1.0, help="multiplier on the ratio for the decoder")
    p.add_argument("--start", type=Path, default=None, help="continue from a (fine-tuned) .pth checkpoint")
    p.add_argument("--checkpoint", default=B2_CHECKPOINT, help="Hugging Face id or local folder (default: B2)")
    p.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    p.add_argument("--prefix", default="segformer_b2")
    p.add_argument("--metrics-dir", type=Path, default=PROJECT_DIR,
                   help="where pruning_metrics.txt/.csv go (default: training/, tracked by git)")
    p.add_argument("--no-verify", action="store_true", help="skip the self-check")

    l = commands.add_parser("latency", help="Jetson: P99 latency of the TorchScript candidates")
    l.add_argument("models", nargs="*", type=Path)
    l.add_argument("--target-ms", type=float, default=50.0)
    l.add_argument("--iterations", type=int, default=1000)
    l.add_argument("--sessions", type=int, default=3, help="separate timing sessions per model")
    l.add_argument("--cooldown", type=float, default=60.0, help="seconds between sessions")
    l.add_argument("--no-stop", action="store_true", help="measure every file instead of stopping early")
    l.add_argument("--device", default="cuda")
    l.add_argument("--output", type=Path, default=Path("pruned_latency.csv"))

    o = commands.add_parser("onnx", help="export a .pth checkpoint (e.g. the fine-tuned model) to ONNX")
    o.add_argument("checkpoint_file", type=Path)
    o.add_argument("--output", type=Path, default=None, help="default: same name with .onnx")
    o.add_argument("--checkpoint", default=B2_CHECKPOINT, help="config source for a plain state dict")

    s = commands.add_parser("subset", help="stratified training subset + validation set from the ADE20K zip")
    s.add_argument("--zip", type=Path, default=REPO_DIR / "ADEChallengeData2016.zip")
    s.add_argument("--size", type=int, default=1000)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--out", type=Path, default=REPO_DIR / "ade20k-scene-parse-150")
    s.add_argument("--list", type=Path, default=PROJECT_DIR / "subset_train_1000.txt")
    s.add_argument("--time-budget", type=float, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.command == "prune":
        if args.ratios is None:
            steps = int(round(args.max_ratio / args.step))
            args.ratios = [round(args.step * k, 4) for k in range(1, steps + 1)]
        if any(not 0.0 < r <= 0.95 for r in args.ratios):
            parser.error("ratios must be above 0 and at most 0.95")
        if args.start:
            print(f"Continuing from {args.start}", flush=True)
            model = load_checkpoint(args.start, args.checkpoint)
        else:
            print(f"Loading {args.checkpoint}", flush=True)
            model = load_original(args.checkpoint)
        save_preprocessing(args.checkpoint, args.output_dir)
        run_prune(model, args.ratios, args.mlp_scale, args.decoder_scale, args.output_dir,
                  prefix=args.prefix, start=str(args.start) if args.start else None, verify=not args.no_verify,
                  metrics_dir=args.metrics_dir)
    elif args.command == "latency":
        pattern = re.compile(r"segformer_b2(_pruned\d+)?\.torchscript\.pt$")   # not *_ft files
        found = {p for d in (REPO_DIR / "model", OUTPUT_DIR) for p in d.glob("segformer_b2*.torchscript.pt")
                 if pattern.fullmatch(p.name)}
        files = args.models or sorted(found, key=_ratio_of)
        if not files:
            parser.error("No segformer_b2*.torchscript.pt files found in model/ or training/model/.")
        run_latency(files, args.target_ms, args.device, args.iterations, args.sessions, args.cooldown,
                    args.output, stop=not args.no_stop)
    elif args.command == "onnx":
        model = load_checkpoint(args.checkpoint_file, args.checkpoint)
        output = args.output or args.checkpoint_file.with_suffix(".onnx")
        export_onnx(model, output)
        print(f"Saved {output}", flush=True)
    else:
        if not args.zip.is_file():
            parser.error(f"{args.zip} not found")
        run_subset(args.zip, args.size, args.seed, args.out, args.list, args.time_budget)


if __name__ == "__main__":
    main()
