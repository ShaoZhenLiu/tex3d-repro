#!/bin/bash
# 41_tex3d_tasks268.sh — Tex3D 原生逐任务纹理：task2(GPU0)∥task6(GPU1)，先完者接 task8
# 上游原生粒度=每任务一张纹理（reset_texture per task）；与 task0 通用部署版并行。
set -u
D=/data/shaozhen.liu
TRD=$D/code/tex3d/openvla-oft/experiments/robot/libero
PYT=$D/envs/tex3d-oft-mj330/bin/python
OUT=$D/code/tex3d/experiments/tex3d_native
mkdir -p "$OUT/logs"
log() { echo "[t3d268 $(date +%H:%M:%S)] $*"; }
ENVV="MUJOCO_GL=egl PYOPENGL_PLATFORM=egl HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled TMPDIR=$D/tmp PYTHONPATH=$D/code/tex3d:$D/code/tex3d/openvla-oft"
train_task() { # $1=gpu $2=task_id
  local g=$1 tid=$2
  nvidia-smi --id=$g --query-gpu=memory.used --format=csv,noheader
  for attempt in 1 2 3; do
    (cd $D/code/tex3d-ws/g7 && env CUDA_VISIBLE_DEVICES=$g MUJOCO_EGL_DEVICE_ID=$g $ENVV \
      "$PYT" "$TRD/attack_oft_tex3d_native.py" --iters 5000 --seed 7 --task-id $tid --tag task${tid}_r1 \
      > "$OUT/logs/train_t${tid}.log" 2>&1) && return 0
    log "task$tid 第$attempt次失败，120s 后重试"; sleep 120
  done
  return 1
}
lane_a() { train_task 0 2 && log "task2 完成" && train_task 0 8 && log "task8 完成（lane A）"; }
lane_b() { train_task 1 6 && log "task6 完成"; }
lane_a & lane_b & wait
log "DONE-TEX3D-268（task2/6/8 训练层收口）"
