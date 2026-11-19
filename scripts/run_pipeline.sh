#!/usr/bin/env bash
# One pruning ratio, start to finish. Run on Colab or a GPU machine; this only
# prepares the model for Nikki, who deploys and measures it on the Jetson.
#
#   prune -> finetune -> export to ONNX
#
#   scripts/run_pipeline.sh --ratio 0.40
#   scripts/run_pipeline.sh --ratio auto                    # the ratio `prune.py latency` chose (results/pruning_target.txt)
#   scripts/run_pipeline.sh --ratio 0.40 --epochs 10
#   scripts/run_pipeline.sh --ratio 0.40 --max-steps 20     # quick smoke/timing run
#   scripts/run_pipeline.sh --ratio 0.40 --dry-run          # print the commands only
#
# --ratio is the pruning target (fraction of channels removed), e.g. the one
# `prune.py latency` stopped at on the Jetson. Needs model/segformer_b2.torchscript.pt
# (load_model.py) and the 1000-image subset in ade20k-scene-parse-150/
# (scripts/sync_subset.sh --val: finetune.py evaluates before/after on the 2000 validation images,
# evaluate-only, printing Top-1 and correct/total). evaluate.py is not run here: it, and all energy,
# latency and engine measurements, belong to the Jetson.
# Everything printed is also appended to output_txt/std_out_prune_finetune.txt.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

RATIO=""; EPOCHS=10; MAX_STEPS=""; DRY=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --ratio)      RATIO="${2:?--ratio needs a value}"; shift 2 ;;
    --epochs)     EPOCHS="${2:?--epochs needs a value}"; shift 2 ;;
    --max-steps)  MAX_STEPS="${2:?--max-steps needs a value}"; shift 2 ;;
    --dry-run)    DRY=1; shift ;;
    -h|--help)    sed -n '2,22p' "$0"; exit 0 ;;
    *) echo "Unknown option: $1 (see --help)" >&2; exit 1 ;;
  esac
done
[[ -n "$RATIO" ]] || { echo "--ratio is required, e.g. --ratio 0.40 or --ratio auto" >&2; exit 1; }
if [[ "$RATIO" == "auto" ]]; then   # the ratio chosen by the Jetson latency sweep
  [[ -s results/pruning_target.txt ]] || { echo "results/pruning_target.txt not found: run prune.py latency on the Jetson (and copy results/ here) first" >&2; exit 1; }
  RATIO="$(head -n 1 results/pruning_target.txt | tr -d '[:space:]')"
  echo "Pruning ratio from results/pruning_target.txt: $RATIO"
fi

# prune.py names files segformer_b2_prunedNN, NN = round(ratio * 100), two digits
NN=$(python3 -c "import sys; print(f'{int(round(float(sys.argv[1]) * 100)):02d}')" "$RATIO")
MODEL_DIR="model"
PRUNED="$MODEL_DIR/segformer_b2_pruned$NN.torchscript.pt"
TUNED="$MODEL_DIR/segformer_b2_pruned${NN}_ft.torchscript.pt"
LOG="output_txt/std_out_prune_finetune.txt"

FT_EXTRA=(--epochs "$EPOCHS")       # finetune.py does not export; ONNX is its own step below
[[ -n "$MAX_STEPS" ]] && FT_EXTRA+=(--max-steps "$MAX_STEPS" --no-eval)   # timing run: no val pass

step() {  # step "<title>" cmd...
  local title="$1"; shift
  echo; echo "=== $title"; echo "+ $*"
  [[ "$DRY" -eq 1 ]] && return 0
  "$@"
}

run() {
  step "1/3 prune to $RATIO"                          python3 training/prune.py prune --ratios "$RATIO"
  step "2/3 fine-tune (1000 train images) + val before/after"  python3 training/finetune.py "$PRUNED" "${FT_EXTRA[@]}"
  step "3/3 export fine-tuned model to ONNX"           python3 training/export.py "$TUNED"
  echo; echo "Done. Hand to Nikki (Jetson):  $MODEL_DIR/segformer_b2_pruned${NN}_ft.onnx"
}

if [[ "$DRY" -eq 1 ]]; then
  run
else
  mkdir -p output_txt
  { echo; echo "##### run_pipeline.sh --ratio $RATIO --epochs $EPOCHS $(date -u +%Y-%m-%dT%H:%MZ)"; } >> "$LOG"
  run 2>&1 | tee -a "$LOG"
fi
