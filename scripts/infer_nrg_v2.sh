#!/bin/bash
set -u
REPO_DIR=/inspire/ssd/project/robot-dna/liangxiujian-253308390319/0-XIUJIANLIANG/prism_plus
CACHE_ROOT=/inspire/hdd/global_user/liangxiujian-253308390319
export HF_HOME=$CACHE_ROOT/huggingface
export TORCH_HOME=$CACHE_ROOT/torch
export TORCH_EXTENSIONS_DIR=$CACHE_ROOT/torch_extensions
export PIP_CACHE_DIR=$CACHE_ROOT/pip
export WANDB_MODE=disabled
source $CACHE_ROOT/miniconda3/etc/profile.d/conda.sh && conda activate prism
cd $REPO_DIR
python tools/infer_nrg_v2.py \
    --bnd_ckpt $CACHE_ROOT/0-XIUJIANLIANG/checkpoints/prism_plus/stage1_v6/best.pt \
    --nrg_ckpt $CACHE_ROOT/0-XIUJIANLIANG/checkpoints/prism_plus/stage2a_nrg/best.pt \
    --data_root /inspire/hdd/project/robot-dna/liangxiujian-253308390319/ByteCameraDepth \
    --n_samples 100 --num_steps 50 \
    --output_dir $CACHE_ROOT/0-XIUJIANLIANG/checkpoints/prism_plus/stage2a_nrg/vis_100
