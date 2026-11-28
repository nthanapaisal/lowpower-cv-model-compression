Sources:
# Compression (Pruning and Quantization) on Segformer-B0
- Model Checkpoint: https://huggingface.co/nvidia/segformer-b0-finetuned-ade-512-512
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
- ./training/prune.py prune Segformer-B2 FP32
- ./training/finetune.py finetune using subset training data (1000) from ADE20K -> onnx
- ./training/export.py pruned + finetune -> onnx
- ./training/evaluate.py evaluate models engines (S0-S4)
- ./training/quantization.py quantize S0 - S4
- ./training/distill.py

Locally Existings Directories:
- ./model
- ./ade20k-scene-parse-150
---------------------------------------------------------------
Pre-req:
Data:
- Dataset Source (ADE20K SceneParse150): http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip

Before running on Jetson run these scripts below and paste them /output_txt:
- ./scripts/board_status.sh
- lpcv
- ./scripts/python3 env_check.py

---------------------------------------------------------------
Flow:

Luke: load_model.py -> (1) prune.py -> (2) finetune.py -> (3) export.py into onnx

Nikki: load_model.py -> (0) export.py into onnx -> (1) quantization.py engines -> (2) evaluate.py -> (4) distill.py -> evaluate.py


