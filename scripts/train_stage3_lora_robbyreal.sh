#!/bin/bash
# Sweep across 5 RobbyReal sensors × 5 sample sizes = 25 runs (~7h on 1xH100)
set -u
REPO_DIR=/inspire/ssd/project/robot-dna/liangxiujian-253308390319/0-XIUJIANLIANG/prism_plus
CACHE_ROOT=/inspire/hdd/global_user/liangxiujian-253308390319
LOG_DIR=$CACHE_ROOT/0-XIUJIANLIANG/checkpoints/prism_plus/stage3_lora_robbyreal/logs
mkdir -p "$LOG_DIR"
LOGFILE=$LOG_DIR/sweep_$(date +%Y%m%d_%H%M%S).log
exec > >(tee -a "$LOGFILE") 2>&1
echo "========== RobbyReal 5-sensor B5 sweep =========="
echo "Date: $(date)  Host: $(hostname)"

export HF_HOME=$CACHE_ROOT/huggingface
export TORCH_HOME=$CACHE_ROOT/torch
export TRANSFORMERS_CACHE=$CACHE_ROOT/huggingface/transformers
export PIP_CACHE_DIR=$CACHE_ROOT/pip
export WANDB_MODE=disabled
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source $CACHE_ROOT/miniconda3/etc/profile.d/conda.sh && conda activate prism
cd $REPO_DIR

SEED=${SEED:-42}
for SENSOR in orbbec_335 orbbec_335L realsense_D415 realsense_D435 realsense_D455; do
  for N in 10 50 100 200 500; do
    echo ""
    echo "========== sensor=$SENSOR n_train=$N seed=$SEED =========="
    python tools/train_bnd_v6_lora.py \
      --config configs/stage3_lora_robbyreal.yaml \
      --sensor "$SENSOR" --n_train $N --seed $SEED
  done
done
echo
echo "=== Aggregating ==="
for SENSOR in orbbec_335 orbbec_335L realsense_D415 realsense_D435 realsense_D455; do
  for N in 10 50 100 200 500; do
    S=$CACHE_ROOT/0-XIUJIANLIANG/checkpoints/prism_plus/stage3_lora_robbyreal/${SENSOR}_n${N}_s${SEED}/summary.json
    [ -f "$S" ] && python -c "import json; d=json.load(open('$S')); print(f'$SENSOR n=$N test_iou={d[chr(34)+'test'+chr(34)].get(chr(34)+'inv_iou'+chr(34),0):.4f}')"
  done
done
