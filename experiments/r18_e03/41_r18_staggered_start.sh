#!/bin/bash
# 41_r18_staggered_start.sh — 错峰启动：逐个发射，等前一个进入迭代再启动下一个
# 根因：多进程同时加载模型(~10-14GB/个)打满共享存储→页失效风暴→全机冻结
set -u
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
D=/data/shaozhen.liu
EX=$D/code/tex3d/experiments/r18_e03
LOGD=$EX/train_logs
WD=$D/code/tex3d/experiments/pi0_cloth/watchdog_events.log
OFTC=$D/code/tex3d/experiments/pi0_cloth/oft_cloth
PY=$D/envs/tex3d-oft-mj330/bin/python
RUNNER=$D/code/tex3d/openvla-oft/experiments/robot/libero/attack_oft_cloth_ablate.py
MR8LOG=$OFTC/train_20261021_224922_tvfix-mr/sampling_log.txt
ENVV="MUJOCO_GL=egl PYOPENGL_PLATFORM=egl HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled PYTHONPATH=$D/code/tex3d:$D/code/tex3d/openvla-oft"
mkdir -p "$LOGD"
log() { echo "$(date '+%F %T') [stag] $*" | tee -a "$WD"; }
gpu_free() { local u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$1" 2>/dev/null | tr -d ' '); [ -n "$u" ] && [ $((49140 - u)) -ge 30000 ] && return 0 || return 1; }
find_gpu() { for g in 5 1 2 0; do gpu_free $g && { echo $g; return; }; done; return 1; }
job_done() { local t=$1 li=$2 d; for d in $(ls -dt $OFTC/train_*_"$t" 2>/dev/null); do [ -f "$d/Universal_Cloth_OV.png" ] && grep -q "\[ckpt-final\] it${li}" "$d/gradient_log.txt" 2>/dev/null && return 0; done; return 1; }
launch_one() {  # $1=tag $2=lastiter $3..=args; 等进入迭代后返回
  local tag=$1 li=$2; shift 2; local g mylog=$LOGD/$tag.log od rargs=()
  if job_done $tag $li; then log "$tag 已完成跳过"; return 0; fi
  g=$(find_gpu) || { log "$tag 无空闲卡"; return 1; }
  od=$(ls -dt $OFTC/train_*_$tag 2>/dev/null | head -1)
  [ -n "$od" ] && ls "$od"/Ep0_Cloth_Delta_OV_it*.pt >/dev/null 2>&1 && ! grep -q "ckpt-final" "$od/gradient_log.txt" 2>/dev/null && rargs=(--resume-dir "$od")
  log "$tag 启动 GPU$g ${rargs[@]:+续训}"
  local pre=$(grep -ac "^\[train\] it" $mylog 2>/dev/null || echo 0)
  ( cd $D/code/tex3d-ws/g7 && env CUDA_VISIBLE_DEVICES=$g MUJOCO_EGL_DEVICE_ID=$g $ENVV \
    $PY $RUNNER train --tag $tag "$@" ${rargs[@]+"${rargs[@]}"} ) >> $mylog 2>&1 &
  local pid=$!
  # 等首个新 it 行（模型加载完成的标志，3-8分钟）
  local t=0
  while [ $t -lt 60 ]; do
    sleep 15; t=$((t+1))
    local cur=$(grep -ac "^\[train\] it" $mylog 2>/dev/null || echo 0)
    [ "$cur" -gt "$pre" ] && { log "$tag 已进入迭代(等60s缓冲)"; sleep 60; return 0; }
    ! kill -0 $pid 2>/dev/null && { log "$tag 进程退出(可能已完成或失败)"; return 1; }
  done
  log "$tag 15分钟未进入迭代，继续等（后台）"
  return 0
}
log "=== 错峰启动开始 ==="
launch_one e07_625_native 00624 --ref native --stream replay --stream-file "$MR8LOG" --uv on --iters 625 --batch 8 --seed 7 --checkpoint-every 625 --ckpt-step-suffix
launch_one e07_625_mr     00624 --ref mr     --stream replay --stream-file "$MR8LOG" --uv on --iters 625 --batch 8 --seed 7 --checkpoint-every 625 --ckpt-step-suffix
launch_one e05_s8_mr      04999 --ref mr     --stream fresh --uv on --iters 5000 --batch 8 --seed 8 --ckpt-step-suffix
launch_one e05_s9_mr      04999 --ref mr     --stream fresh --uv on --iters 5000 --batch 8 --seed 9 --ckpt-step-suffix
log "短程与配对臂已发射；bs1_native(长程)在下一批错峰启动"
