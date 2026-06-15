#!/bin/bash
# PRISM+ Stage 4 — TNSM temporal training on DREDS pseudo-video
set -u
REPO_DIR=/inspire/ssd/project/robot-dna/liangxiujian-253308390319/0-XIUJIANLIANG/prism_plus
CACHE_ROOT=/inspire/hdd/global_user/liangxiujian-253308390319
LOG_DIR=$CACHE_ROOT/0-XIUJIANLIANG/checkpoints/prism_plus/stage4_tnsm/logs
mkdir -p "$LOG_DIR"
LOGFILE=$LOG_DIR/train_$(date +%Y%m%d_%H%M%S).log
exec > >(tee -a "$LOGFILE") 2>&1

echo "========== PRISM+ Stage 4 — TNSM (DREDS pseudo-video) =========="
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
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_DEBUG=INFO

source "$CACHE_ROOT/miniconda3/etc/profile.d/conda.sh"
conda activate prism

cd "$REPO_DIR"
GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
echo "GPU Count: $GPU_COUNT"

torchrun --nproc_per_node="$GPU_COUNT" --master_port=29502 \
    tools/train_bnd_v7_tnsm.py \
    --config configs/stage4_tnsm.yaml \
    --seed 42

echo "Training exited: $?"
