#!/bin/bash
# Stage 1: Spatial-SPR BND Training  
PRISM_PLUS="/inspire/ssd/project/robot-dna/liangxiujian-253308390319/0-XIUJIANLIANG/prism_plus"
CACHE_ROOT="/inspire/hdd/global_user/liangxiujian-253308390319"

export HF_HOME="$CACHE_ROOT/huggingface"
export TRANSFORMERS_CACHE="$CACHE_ROOT/huggingface/transformers"
export TORCH_HOME="$CACHE_ROOT/torch"
export TORCH_EXTENSIONS_DIR="$CACHE_ROOT/torch_extensions"
export PIP_CACHE_DIR="$CACHE_ROOT/pip"
export WANDB_CACHE_DIR="$CACHE_ROOT/wandb"
export HDD_PYPATH="$CACHE_ROOT/python-packages"
export PYTHONPATH="$HDD_PYPATH${PYTHONPATH:+:$PYTHONPATH}"

source "$CACHE_ROOT/miniconda3/etc/profile.d/conda.sh"
conda activate latpixdepth

GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
echo "Using $GPU_COUNT GPUs"

cd "$PRISM_PLUS"
torchrun --nproc_per_node=$GPU_COUNT --master_port=29500 \
    train_stage1_bnd.py \
    --config configs/stage1_spatial_spr.yaml \
    --seed 42
