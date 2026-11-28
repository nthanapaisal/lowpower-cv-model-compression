#!/usr/bin/env python3
"""Validate and summarize a fixed-shape SegFormer ONNX graph.

Run after export and before building a TensorRT engine::

    python3 training/inspect_onnx.py onnx/segformer_b2.onnx

The summary reports the graph inputs and outputs, opset, operator counts,
non-standard operator domains, and every node. Review the node list for
unexpected operators or graph expansion before compiling the model.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path


EXPECTED_INPUT_SHAPE = (1, 3, 512, 512)
EXPECTED_OUTPUT_SHAPE = (1, 150, 128, 128)
EXPECTED_OPSET = 17


def tensor_shape(value_info) -> tuple[int | str, ...]:
    """Return static dimensions as integers and dynamic dimensions as names."""
    result = []
    for dim in value_info.type.tensor_type.shape.dim:
        if dim.HasField("dim_value"):
            result.append(int(dim.dim_value))
        else:
            result.append(dim.dim_param or "?")
    return tuple(result)


def format_shape(shape: tuple[int | str, ...]) -> str:
    return "[" + ", ".join(str(dim) for dim in shape) + "]"


def inspect_model(path: Path) -> int:
    try:
        import onnx
    except ImportError as exc:
        raise SystemExit("The onnx package is required. Install project requirements first.") from exc

    model = onnx.load(str(path), load_external_data=True)
    onnx.checker.check_model(model)
    try:
        model = onnx.shape_inference.infer_shapes(model)
    except Exception as exc:
        print(f"WARNING: ONNX shape inference could not complete: {exc}")

    graph = model.graph
    initializer_names = {item.name for item in graph.initializer}
    inputs = [item for item in graph.input if item.name not in initializer_names]
    outputs = list(graph.output)
    opsets = {item.domain or "ai.onnx": item.version for item in model.opset_import}
    counts = Counter((node.domain or "ai.onnx", node.op_type) for node in graph.node)
    standard_domain = {"", "ai.onnx"}
    custom_domains = sorted({node.domain for node in graph.node if node.domain not in standard_domain})

    print(f"ONNX file: {path.resolve()}")
    print(f"IR version: {model.ir_version}")
    print(f"Opset imports: {opsets}")
    print(f"Graph name: {graph.name or '(unnamed)'}")
    print(f"Graph nodes: {len(graph.node)}")
    print(f"Initializers: {len(graph.initializer)}")

    print("\nInputs:")
    for item in inputs:
        print(f"  {item.name}: {format_shape(tensor_shape(item))}")
    print("Outputs:")
    for item in outputs:
        print(f"  {item.name}: {format_shape(tensor_shape(item))}")

    print("\nOperator counts:")
    for (domain, op_type), count in sorted(counts.items()):
        label = op_type if domain in standard_domain else f"{domain}::{op_type}"
        print(f"  {label}: {count}")
    if custom_domains:
        print(f"\nWARNING: non-standard operator domains found: {', '.join(custom_domains)}")

    print("\nNodes (review this list for unexpected operators):")
    for index, node in enumerate(graph.node):
        domain = f"{node.domain}::" if node.domain not in standard_domain else ""
        name = node.name or "(unnamed)"
        print(f"  {index:04d} {domain}{node.op_type}  name={name}")

    errors = []
    if len(inputs) != 1:
        errors.append(f"expected one model input, found {len(inputs)}")
    elif tensor_shape(inputs[0]) != EXPECTED_INPUT_SHAPE:
        errors.append(
            f"expected input shape {format_shape(EXPECTED_INPUT_SHAPE)}, "
            f"found {format_shape(tensor_shape(inputs[0]))}"
        )
    if len(outputs) != 1:
        errors.append(f"expected one model output, found {len(outputs)}")
    elif tensor_shape(outputs[0]) != EXPECTED_OUTPUT_SHAPE:
        errors.append(
            f"expected output shape {format_shape(EXPECTED_OUTPUT_SHAPE)}, "
            f"found {format_shape(tensor_shape(outputs[0]))}"
        )
    if opsets.get("ai.onnx") != EXPECTED_OPSET:
        errors.append(f"expected ONNX opset {EXPECTED_OPSET}, found {opsets.get('ai.onnx')}")
    if custom_domains:
        errors.append("graph contains non-standard operator domains")

    if errors:
        print("\nFAILED validation:")
        for error in errors:
            print(f"  - {error}")
        return 1

    print("\nPASS: one fixed-shape SegFormer input/output and opset 17; ONNX checker passed.")
    print("Review the node list above and open the graph in Netron for visual inspection before TensorRT build.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("onnx_file", type=Path, help="exported ONNX model")
    args = parser.parse_args()
    path = args.onnx_file.expanduser()
    if not path.is_file():
        parser.error(f"ONNX file not found: {path}")
    return inspect_model(path)


if __name__ == "__main__":
    raise SystemExit(main())
