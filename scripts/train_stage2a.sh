#!/bin/bash
# PRISM+ Stage 2a — Mask-conditioned NRG (C2, standalone)
set -u

REPO_DIR="/inspire/ssd/project/robot-dna/liangxiujian-253308390319/0-XIUJIANLIANG/prism_plus"
CACHE_ROOT="/inspire/hdd/global_user/liangxiujian-253308390319"
LOG_DIR="$CACHE_ROOT/0-XIUJIANLIANG/checkpoints/prism_plus/stage2a_nrg/logs"
mkdir -p "$LOG_DIR"
LOGFILE="$LOG_DIR/train_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOGFILE") 2>&1

echo "========== PRISM+ Stage 2a — Mask-Conditioned NRG (C2) =========="
echo "Date:    $(date)"
echo "Host:    $(hostname)"
echo "Log:     $LOGFILE"

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

python -c "import torch; print('torch:', torch.__version__, '| cuda:', torch.cuda.is_available(), '| devices:', torch.cuda.device_count())"

cd "$REPO_DIR"
GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
echo "GPU Count: $GPU_COUNT"

torchrun --nproc_per_node="$GPU_COUNT" --master_port=29501 \
    tools/train_nrg_v2.py \
    --config configs/stage2a_nrg.yaml \
    --seed 42

echo "Training exited: $?"
