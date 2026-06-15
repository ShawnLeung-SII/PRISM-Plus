#!/bin/bash
# B5 data-efficiency sweep: 10/50/100/200/500 samples × {sensor} × seeds
set -u
REPO_DIR=/inspire/ssd/project/robot-dna/liangxiujian-253308390319/0-XIUJIANLIANG/prism_plus
CACHE_ROOT=/inspire/hdd/global_user/liangxiujian-253308390319

export HF_HOME="$CACHE_ROOT/huggingface"
export TORCH_HOME="$CACHE_ROOT/torch"
export WANDB_MODE=disabled
source "$CACHE_ROOT/miniconda3/etc/profile.d/conda.sh"
conda activate prism

SENSOR=${1:-dreds_d415}
SEED=${2:-42}

for N in 10 50 100 200 500; do
    echo "==================== n_train=$N sensor=$SENSOR seed=$SEED ===================="
    cd "$REPO_DIR"
    python tools/train_bnd_v6_lora.py \
        --config configs/stage3_lora.yaml \
        --sensor "$SENSOR" \
        --n_train $N --seed $SEED
done

echo
echo 'B5 sweep done. Aggregate via:'
echo "  jq -s '.[].test.inv_iou' $CACHE_ROOT/0-XIUJIANLIANG/checkpoints/prism_plus/stage3_lora/${SENSOR}_n*/summary.json"
