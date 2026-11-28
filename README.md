Sources:
# Compression (Pruning and Quantization) on SegFormer-B2
- Model Checkpoint: https://huggingface.co/nvidia/segformer-b2-finetuned-ade-512-512
- Dataset HugginFace(ADE20K SceneParse150): https://huggingface.co/datasets/zhoubolei/scene_parse_150
- Dataset Source (ADE20K SceneParse150): http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip
- Original SegFormer Paper: https://arxiv.org/abs/2105.15203
---------------------------------------------------------------
Description:

---------------------------------------------------------------
Reference Model:
Segformer-B2 FP32

Models: 
S0: Quantized FP16 (Baseline)
S1: Quantized INT8 
S2: Pruned fine-tuning + FP16 
S3: Pruned fine-tuning + INT8 
S4: Pruned fine-tuning + FP16 + distillation 
---------------------------------------------------------------
Files: 

./baseline:
- All files for running baseline models in FP32 Pytorch for model selection from Segform Family in the proposal (Milestone 1)

./training 
- ./training/load_model.py to load models locally into ./model
- ./training/export.py exports the B2 TorchScript baseline to fixed-shape ONNX
- ./training/quantization.py builds S0-S4 TensorRT engines and records layer precision reports
- ./training/evaluate.py measures the engines on the Jetson over three separate sessions
- ./training/pruning.py, ./training/finetune.py, and ./training/distill.py are currently empty; teammate code and the S4 distilled ONNX are still needed

Locally Existings Directories:
- ./model
- ./ade20k-scene-parse-150
---------------------------------------------------------------
Pre-req:
Data:
- Dataset Source (ADE20K SceneParse150): http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip

Before measuring on Jetson, run `scripts/board_status.sh`, enter `lpcv`, then run `python3 scripts/env_check.py --role measurement`. Save the outputs in `output_txt/`.

---------------------------------------------------------------
Flow:

Teammate: pruning.py -> finetune.py -> export.py, then hand off the pruned/fine-tuned S2 ONNX and, if completed, the distilled S4 ONNX. S3 reuses the S2 ONNX with INT8 precision.

Owner: export the B2 FP32 baseline ONNX; build S0 FP16 and S1 INT8 from it, plus S2 FP16/S3 INT8 from the S2 ONNX and S4 FP16 from the S4 ONNX; then evaluate all five TensorRT engines with the same evaluation script and protocol.

Run after the teammate handoff:

```bash
python3 training/export.py model/segformer_b2.torchscript.pt
python3 training/quantization.py
python3 training/evaluate.py --session 1 --board-hostname jetson-lpcv-03
python3 training/evaluate.py --session 2 --board-hostname jetson-lpcv-03
python3 training/evaluate.py --session 3 --board-hostname jetson-lpcv-03
```

Run sessions 1, 2, and 3 on the same idle board in separate sessions. Keep all five engine files in `model/` for every session.


