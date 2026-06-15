#!/bin/bash
# PRISM+ Stage 3 — LoRA-SPA B5 data-efficiency sweep on DREDS
# Single-GPU sequential: 5 sample sizes × 1 sensor = 5 short runs (~30 min each)
set -u
REPO_DIR=/inspire/ssd/project/robot-dna/liangxiujian-253308390319/0-XIUJIANLIANG/prism_plus
CACHE_ROOT=/inspire/hdd/global_user/liangxiujian-253308390319
LOG_DIR=$CACHE_ROOT/0-XIUJIANLIANG/checkpoints/prism_plus/stage3_lora/logs
mkdir -p "$LOG_DIR"
LOGFILE=$LOG_DIR/train_$(date +%Y%m%d_%H%M%S).log
exec > >(tee -a "$LOGFILE") 2>&1

echo "========== PRISM+ Stage 3 — LoRA-SPA B5 sweep =========="
echo "Date: $(date)"
echo "Host: $(hostname)"

export HF_HOME="$CACHE_ROOT/huggingface"
export HF_DATASETS_CACHE="$CACHE_ROOT/huggingface/datasets"
export TRANSFORMERS_CACHE="$CACHE_ROOT/huggingface/transformers"
export TORCH_HOME="$CACHE_ROOT/torch"
export TORCH_EXTENSIONS_DIR="$CACHE_ROOT/torch_extensions"
export PIP_CACHE_DIR="$CACHE_ROOT/pip"
export WANDB_CACHE_DIR="$CACHE_ROOT/wandb"
export WANDB_MODE=disabled

source "$CACHE_ROOT/miniconda3/etc/profile.d/conda.sh"
conda activate prism

cd "$REPO_DIR"
SENSOR=${SENSOR:-dreds_d415}
SEED=${SEED:-42}

for N in 10 50 100 200 500; do
    echo ""
    echo "========== n_train=$N sensor=$SENSOR seed=$SEED =========="
    python tools/train_bnd_v6_lora.py \
        --config configs/stage3_lora.yaml \
        --sensor "$SENSOR" --n_train $N --seed $SEED
done

echo
echo "========== B5 sweep done. Aggregating =========="
for N in 10 50 100 200 500; do
    SUMMARY=$CACHE_ROOT/0-XIUJIANLIANG/checkpoints/prism_plus/stage3_lora/${SENSOR}_n${N}_s${SEED}/summary.json
    if [ -f "$SUMMARY" ]; then
        IOU=$(python -c "import json; d=json.load(open('$SUMMARY')); print(d['test'].get('inv_iou', 0))")
        echo "n=$N -> test inv_iou = $IOU"
    fi
done
