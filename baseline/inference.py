"""Legacy one-off FP16 TensorRT runner retained for earlier experiments.

Use ``training/quantization.py`` and ``training/evaluate.py`` for comparable
Milestone 2 S0-S4 results, including the fixed dataset and repeat sessions.
"""

from pathlib import Path

import torch

try:
    from . import baseline
except ImportError:
    import baseline


ROOT = Path(__file__).resolve().parent.parent
FP16_MODEL = ROOT / "model" / "segformer_b2_fp16.onnx"


def load_tensorrt_model(model_path, precision, device):
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    with open(model_path, "rb") as model_file:
        if not parser.parse(model_file.read()):
            raise RuntimeError("Could not read ONNX model: " + str(model_path))
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
    if precision == "fp16":
        config.set_flag(trt.BuilderFlag.FP16)
    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        raise RuntimeError("TensorRT could not build the engine")
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(serialized_engine)
    context = engine.create_execution_context()
    return engine, context, runtime


def run_tensorrt(model, inputs, device):
    engine, context, _ = model
    pixel_values = inputs["pixel_values"].to(device=device, dtype=torch.float16).contiguous()
    input_name = engine.get_tensor_name(0)
    output_name = engine.get_tensor_name(1)
    context.set_input_shape(input_name, tuple(pixel_values.shape))
    output_shape = tuple(context.get_tensor_shape(output_name))
    logits = torch.empty(output_shape, device=device, dtype=torch.float16)
    context.set_tensor_address(input_name, pixel_values.data_ptr())
    context.set_tensor_address(output_name, logits.data_ptr())
    context.execute_async_v3(torch.cuda.current_stream(device).cuda_stream)
    return logits


def inference_model(model):
    model_path = Path(model)
    print("Model being inferred:", model_path, flush=True)
    config = baseline.EvaluationConfig(
        model_name="SegFormer-B2 FP16 TensorRT",
        model_path=str(model_path),
        checkpoint="nvidia/segformer-b2-finetuned-ade-512-512",
        architecture="MiT-B2 encoder + MLP decoder",
        precision="fp16",
        runtime="TensorRT",
        batch_size=1,
        warmup_seconds=2,
        timed_iterations=1000,
        evaluation_images=baseline.DEFAULT_EVALUATION_IMAGES,
        output_csv=str(ROOT / "baseline_results.csv"),
    )
    return baseline.run_evaluation(
        config,
        model_loader=load_tensorrt_model,
        inference_fn=run_tensorrt,
    )


def main():
    inference_model(FP16_MODEL)


if __name__ == "__main__":
    main()
