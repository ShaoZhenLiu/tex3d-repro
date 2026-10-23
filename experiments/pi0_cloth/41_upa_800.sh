#!/bin/bash
# 41_upa_800.sh — UPA-RFAS 原生全量 800（用户 2026-10-23 晚授权：跳过开发格门槛，直接 800）
#   训练仍由原续跑 python（孤儿子 41_upa_resume.sh 包装）完成；本链取代其评测阶段——
#   41_upa_resume.sh 的 bash 包装已按 PID 移除（python 不受影响，继续训到 iter2000）。
#   8 格 = {π0,π0.5} × {libero_spatial, libero_object, libero_goal, libero_10} × 100 回合，
#   同一张 patch 主/腕各挂、位置独立采样（与 native2d 波同款 eval_patch 通道）。
#   泳道：GPU2 = 等训练完成后 3 格 pi05（obj/sp/go）；
#         GPU1 = 等 DONE-T3DDEEP-PI0 后 3 格 pi0（sp/obj/go）；
#         GPU3 = 等 DONE-T3DDEEP-PI05 后 2 格 l10（pi0/pi05）。
#   GPU 纪律：仅 0-3；启动前现查。完成标记：DONE-UPA-800。
set -u
D=/data/shaozhen.liu
PI=$D/code/tex3d/pi
CKPT0=$D/model/RLinf-Pi0-LIBERO-130-fullshot-SFT
CKPT5=$D/model/RLinf-Pi05-LIBERO-130-fullshot-SFT
UPAOUT=$D/code/tex3d/experiments/upa_native
LOGT=$UPAOUT/logs/train_upa_resume.log
DEEPLOG=$D/code/tex3d/experiments/tex3d_native/logs/chain_deepen.log
mkdir -p "$UPAOUT/logs"
log() { echo "[upa800 $(date +%m%d_%H:%M:%S)] $*"; }

wait_log() { # $1=file $2=pattern $3=最大次数(×300s)
  local i=0
  while [ $i -lt "${3:-240}" ]; do
    tr -d '\r' < "$1" 2>/dev/null | grep -aq "$2" && { log "已见 $2"; return 0; }
    sleep 300; i=$((i+1))
  done
  log "FAIL-WAIT-TIMEOUT $2"; return 1
}

ensure_patch() {
  [ -f "$UPAOUT/upa_native_patch.pt" ] && { log "patch 已存在"; return 0; }
  wait_log "$LOGT" "Attack done" 240 || return 1
  sleep 30
  local RUN_DIR
  RUN_DIR=$(ls -dt /data/shaozhen.liu/code/UPA-RFAS/run/our/upa_resume_r1_* 2>/dev/null | head -1)
  [ -n "$RUN_DIR" ] || { log "FAIL-NO-RUNDIR"; return 1; }
  cp "$RUN_DIR/last/patch.pt" "$UPAOUT/upa_native_patch.pt" 2>/dev/null \
    || cp "$RUN_DIR/final/patch.pt" "$UPAOUT/upa_native_patch.pt"
  [ -f "$UPAOUT/upa_native_patch.pt" ] || { log "FAIL-NO-PATCH"; return 1; }
  log "训练完成 $RUN_DIR"
  sha256sum "$UPAOUT/upa_native_patch.pt"
}

cell_done() {
  local res n
  res=$(grep -a '\[eval\] 结果' "$UPAOUT/logs/$1.log" 2>/dev/null | tail -1 | sed 's/.*结果: //' | tr -d '\r')
  n=$(grep -ac '^Task:' "$res" 2>/dev/null)
  [ -n "$res" ] && [ -f "$res" ] && [ "${n:-0}" -ge 100 ]
}
run_cell() {  # $1=gpu $2=name $3=suite $4=ckpt $5=modelcfg
  local g=$1 name=$2 suite=$3 ckpt=$4 mcfg=${5:-} res n extra=()
  [ -n "$mcfg" ] && extra+=(--model-config "$mcfg")
  cell_done "$name" && { log "$name 已完成"; return 0; }
  log "评测 $name GPU$g ($suite)"
  for attempt in 1 2 3 4; do
    env CLOTH_QUERY_SEED=1 bash "$PI/run_patch_eval.sh" "$g" eval \
      --condition patch --tasks 0,1,2,3,4,5,6,7,8,9 --trials 10 --num-denoise 10 \
      --init-offset 10 --suite "$suite" --ckpt "$ckpt" \
      --patch-primary "$UPAOUT/upa_native_patch.pt" --patch-wrist "$UPAOUT/upa_native_patch.pt" \
      "${extra[@]}" >> "$UPAOUT/logs/${name}.log" 2>&1
    res=$(grep -a '\[eval\] 结果' "$UPAOUT/logs/$name.log" 2>/dev/null | tail -1 | sed 's/.*结果: //' | tr -d '\r')
    n=$(grep -ac '^Task:' "$res" 2>/dev/null)
    if [ -n "$res" ] && [ -f "$res" ] && [ "${n:-0}" -ge 100 ]; then
      log "$name 完成 -> $(grep -a 'overall robot success' "$res" | tail -1)"; return 0
    fi
    sleep 45
  done
  log "$name 四次失败"; return 1
}

lane_gpu2() {
  local g=2
  nvidia-smi --id=$g --query-gpu=memory.used --format=csv,noheader
  ensure_patch || exit 1
  run_cell $g upanat_pi05_obj libero_object "$CKPT5" pi05_libero
  run_cell $g upanat_pi05_sp  libero_spatial "$CKPT5" pi05_libero
  run_cell $g upanat_pi05_go  libero_goal "$CKPT5" pi05_libero
  log "DONE-UPA800-GPU2"
}
lane_gpu1() {
  local g=1
  wait_log "$DEEPLOG" "DONE-T3DDEEP-PI0" 240 || exit 1
  sleep 120
  nvidia-smi --id=$g --query-gpu=memory.used --format=csv,noheader
  run_cell $g upanat_pi0_sp  libero_spatial "$CKPT0"
  run_cell $g upanat_pi0_obj libero_object  "$CKPT0"
  run_cell $g upanat_pi0_go  libero_goal    "$CKPT0"
  log "DONE-UPA800-GPU1"
}
lane_gpu3() {
  local g=3
  wait_log "$DEEPLOG" "DONE-T3DDEEP-PI05" 240 || exit 1
  sleep 120
  nvidia-smi --id=$g --query-gpu=memory.used --format=csv,noheader
  run_cell $g upanat_pi0_l10  libero_10 "$CKPT0"
  run_cell $g upanat_pi05_l10 libero_10 "$CKPT5" pi05_libero
  log "DONE-UPA800-GPU3"
}
lane_gpu2 & lane_gpu1 & lane_gpu3 & wait
log "DONE-UPA-800"
