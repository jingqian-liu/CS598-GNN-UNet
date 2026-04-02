#!/bin/bash
# Train UnetHeteroGVP on the ATOM3D LBA task.
#
# Usage:
#   bash scripts/run_lba.sh                          # defaults
#   SPLIT=60 LR=1e-3 bash scripts/run_lba.sh        # override via env vars
#   bash scripts/run_lba.sh --wandb                  # enable W&B logging
#   bash scripts/run_lba.sh --num_layers 7 --s_dim 64

source scripts/common.sh
cd "$(dirname "$0")"/..

# ------------------------------------------------------------------ #
# Configurable defaults (override with env vars or extra CLI args)     #
# ------------------------------------------------------------------ #
SPLIT=${SPLIT:-30}
LR=${LR:-0.0005}
WD=${WD:-0.001}
BATCH=${BATCH:-8}
EPOCHS=${EPOCHS:-100}
LAYERS=${LAYERS:-5}
S_DIM=${S_DIM:-128}
POOL=${POOL:-mean}
WORKERS=${WORKERS:-4}
DEVICES=${DEVICES:-1}
SEED=${SEED:-$RANDOM}

NAME="unet_hetero_gvp/split${SPLIT}/L${LAYERS}_s${S_DIM}_${POOL}/lr${LR}_wd${WD}"

$exec scripts/train_lba.py \
    --split       $SPLIT      \
    --lr          $LR         \
    --wd          $WD         \
    --batch_size  $BATCH      \
    --epochs      $EPOCHS     \
    --num_layers  $LAYERS     \
    --s_dim       $S_DIM      \
    --pool        $POOL       \
    --num_workers $WORKERS    \
    --devices     $DEVICES    \
    --seed        $SEED       \
    --name        "$NAME"     \
    "$@"
