#!/bin/bash
PRISM_PLUS="/inspire/ssd/project/robot-dna/liangxiujian-253308390319/0-XIUJIANLIANG/prism_plus"
CACHE_ROOT="/inspire/hdd/global_user/liangxiujian-253308390319"
LOG_DIR="$CACHE_ROOT/0-XIUJIANLIANG/checkpoints/prism_plus/stage1/logs"
mkdir -p "$LOG_DIR"
LOGFILE="$LOG_DIR/train_$(date +%Y%m%d_%H%M%S).log"

# 日志同时输出到终端和 HDD 文件
exec > >(tee -a "$LOGFILE") 2>&1

echo "========== PRISM+ Stage 1 Training =========="
echo "Date: $(date)"
echo "Hostname: $(hostname)"
echo "Logfile: $LOGFILE"

# 缓存路径
export HF_HOME="$CACHE_ROOT/huggingface"
export HF_DATASETS_CACHE="$CACHE_ROOT/huggingface/datasets"
export TRANSFORMERS_CACHE="$CACHE_ROOT/huggingface/transformers"
export TORCH_HOME="$CACHE_ROOT/torch"
export TORCH_EXTENSIONS_DIR="$CACHE_ROOT/torch_extensions"
export PIP_CACHE_DIR="$CACHE_ROOT/pip"
export WANDB_CACHE_DIR="$CACHE_ROOT/wandb"
export WANDB_MODE=disabled
export HDD_PYPATH="$CACHE_ROOT/python-packages"
export PYTHONPATH="$HDD_PYPATH${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_DEBUG=INFO

echo ""
echo "--- Storage Check ---"
ls "$CACHE_ROOT/" > /dev/null 2>&1 && echo "HDD global_user: OK" || echo "HDD global_user: MISSING!"
ls "$PRISM_PLUS/" > /dev/null 2>&1 && echo "SSD prism_plus:  OK" || echo "SSD prism_plus:  MISSING!"
ls "$CACHE_ROOT/../../../project/robot-dna/liangxiujian-253308390319/ByteCameraDepth/" > /dev/null 2>&1 \
    && echo "ByteCameraDepth: OK" || echo "ByteCameraDepth: MISSING (check /inspire/hdd/project/...)"

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
    echo "Conda: sourced from HDD"
    conda activate latpixdepth
    echo "Env: $CONDA_DEFAULT_ENV"
    python -c "import torch; print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available(), 'devices:', torch.cuda.device_count())"
else
    echo "ERROR: HDD conda not found at $CONDA_SH"
    exit 1
fi

echo ""
echo "--- Starting torchrun ---"
cd "$PRISM_PLUS" || { echo "ERROR: cd $PRISM_PLUS failed"; exit 1; }

torchrun \
    --nproc_per_node=$GPU_COUNT \
    --master_port=29500 \
    train_stage1_bnd.py \
    --config configs/stage1_spatial_spr.yaml \
    --seed 42

EXIT_CODE=$?
echo ""
echo "Training exited with code: $EXIT_CODE"
echo "Log saved to: $LOGFILE"
