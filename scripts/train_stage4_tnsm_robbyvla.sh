#!/bin/bash
set -u
REPO_DIR=/inspire/ssd/project/robot-dna/liangxiujian-253308390319/0-XIUJIANLIANG/prism_plus
CACHE_ROOT=/inspire/hdd/global_user/liangxiujian-253308390319
LOG_DIR=$CACHE_ROOT/0-XIUJIANLIANG/checkpoints/prism_plus/stage4_tnsm_robbyvla/logs
mkdir -p "$LOG_DIR"
LOGFILE=$LOG_DIR/train_$(date +%Y%m%d_%H%M%S).log
exec > >(tee -a "$LOGFILE") 2>&1
echo '========== PRISM+ Stage 4 — TNSM on RobbyVla real robot video =========='
echo "Date: $(date)  Host: $(hostname)"
export HF_HOME="$CACHE_ROOT/huggingface"
export TORCH_HOME="$CACHE_ROOT/torch"
export PIP_CACHE_DIR="$CACHE_ROOT/pip"
export WANDB_MODE=disabled
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source "$CACHE_ROOT/miniconda3/etc/profile.d/conda.sh"
conda activate prism
cd "$REPO_DIR"
GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
torchrun --nproc_per_node="$GPU_COUNT" --master_port=29503 \
    tools/train_bnd_v7_tnsm.py \
    --config configs/stage4_tnsm_robbyvla.yaml --seed 42
echo "Training exited: $?"
