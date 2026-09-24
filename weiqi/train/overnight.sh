#!/bin/bash
# Overnight chain: full OGS pull -> dataset build -> supervised training.
# Each stage is resumable; rerunning this script continues where it left off
# (ogs_pull skips downloaded SGFs, dataset rebuild is idempotent, training
# resumes from runs/imit_full/latest.pt if present).
set -u
cd "$(dirname "$0")"
V="$HOME/workspace/venvs/torch-cpu/bin/python"

echo "=== [1/3] OGS pull ==="
$V -m gotrain.ogs_pull --out data/ogs_full --max-games 100000 \
    --seed-ladder-size 1830 --max-players 1500 --delay 0.5 \
    >> data/ogs_full_pull.log 2>&1

echo "=== [2/3] dataset build ==="
$V -m gotrain.dataset --sgf data/ogs_full/sgf --out data/ds_full \
    >> data/ds_full_build.log 2>&1

echo "=== [3/3] training ==="
RESUME=""
if [ -f runs/imit_full/latest.pt ]; then
    RESUME="--resume runs/imit_full/latest.pt"
fi
$V -m gotrain.train_cloning --data data/ds_full --out runs/imit_full \
    --max-steps 200000 --val-every 1000 --ckpt-every 500 $RESUME \
    >> runs/imit_full_train.log 2>&1

echo "=== done ==="
