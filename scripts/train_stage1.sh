#!/bin/bash
PRISM_PLUS="/inspire/ssd/project/robot-dna/liangxiujian-253308390319/0-XIUJIANLIANG/prism_plus"
CACHE_ROOT="/inspire/hdd/global_user/liangxiujian-253308390319"

# === 诊断阶段 ===
echo "========== PRISM+ Stage 1 Training =========="
echo "Date: $(date)"
echo "Hostname: $(hostname)"
echo "PWD: $(pwd)"

echo ""
echo "--- Storage Check ---"
ls /inspire/hdd/global_user/liangxiujian-253308390319/ 2>/dev/null && echo "HDD: OK" || echo "HDD: MISSING!"
ls /inspire/ssd/project/robot-dna/liangxiujian-253308390319/0-XIUJIANLIANG/ 2>/dev/null && echo "SSD: OK" || echo "SSD: MISSING!"

echo ""
echo "--- GPU Check ---"
nvidia-smi --list-gpus
GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
echo "GPU Count: $GPU_COUNT"

echo ""
echo "--- Conda Setup ---"
CONDA_SH="$CACHE_ROOT/miniconda3/etc/profile.d/conda.sh"
if [ -f "$CONDA_SH" ]; then
    source "$CONDA_SH"
    echo "Conda sourced from HDD: OK"
    conda activate latpixdepth && echo "latpixdepth env: OK" || echo "latpixdepth env: FAILED"
else
    echo "HDD conda not found, trying system Python..."
    which python3 && python3 --version
fi

# 缓存路径
export HF_HOME="$CACHE_ROOT/huggingface"
export HF_DATASETS_CACHE="$CACHE_ROOT/huggingface/datasets"
export TRANSFORMERS_CACHE="$CACHE_ROOT/huggingface/transformers"
export TORCH_HOME="$CACHE_ROOT/torch"
export TORCH_EXTENSIONS_DIR="$CACHE_ROOT/torch_extensions"
export PIP_CACHE_DIR="$CACHE_ROOT/pip"
export WANDB_CACHE_DIR="$CACHE_ROOT/wandb"
export HDD_PYPATH="$CACHE_ROOT/python-packages"
export PYTHONPATH="$HDD_PYPATH${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo ""
echo "--- Python/Torch Check ---"
python -c "import torch; print('torch:', torch.__version__, '| cuda:', torch.cuda.is_available(), '| devices:', torch.cuda.device_count())" 2>/dev/null || echo "torch import failed"

echo ""
echo "--- Starting Training ---"
cd "$PRISM_PLUS" || { echo "ERROR: Cannot cd to $PRISM_PLUS"; exit 1; }

torchrun \
    --nproc_per_node=$GPU_COUNT \
    --master_port=29500 \
    train_stage1_bnd.py \
    --config configs/stage1_spatial_spr.yaml \
    --seed 42

echo "========== Training Complete =========="
