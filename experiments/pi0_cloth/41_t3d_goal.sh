#!/bin/bash
# 41_t3d_goal.sh — Tex3D 原生逐任务纹理 × libero_goal 套件（用户选项C，2026-10-23 授权）
#   碗在 goal 全部 10 任务场景在场（BDDL 核查 2026-10-23）。
#   每任务一张纹理：--suite libero_goal（采帧/任务文本），源模型仍 OFT-spatial
#   （cfg.task_suite_name 恒 spatial，unnorm 口径不变——披露登记）。
#   泳道：lane1 GPU1(即刻) g0,g4,g8 | lane2 GPU3(即刻) g1,g5,g9
#         lane3 GPU0(等 DONE-T3D-TASK8) g2,g6 + 通用部署两格(g0纹理×10任务×2目标)
#         lane4 GPU2(等 DONE-UPA-RESUME) g3,g7 + θ0 两格(same θ0 PNG×10任务×2目标)
#   每任务：训练 5000×批10 → t3dg_t{tid}_{pi0,pi05} 两格（各 10 回合）。
#   GPU 纪律：仅 0-3；启动前现查。
set -u
D=/data/shaozhen.liu
TRD=$D/code/tex3d/openvla-oft/experiments/robot/libero
PYT=$D/envs/tex3d-oft-mj330/bin/python
PI=$D/code/tex3d/pi
OUT=$D/code/tex3d/experiments/tex3d_native
CKPT0=$D/model/RLinf-Pi0-LIBERO-130-fullshot-SFT
CKPT5=$D/model/RLinf-Pi05-LIBERO-130-fullshot-SFT
TH0=$OUT/train_20261023_072153_r1/theta0_baked.png
mkdir -p "$OUT/logs"
log() { echo "[t3dg $(date +%m%d_%H:%M:%S)] $*"; }
ENVV="MUJOCO_GL=egl PYOPENGL_PLATFORM=egl HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled TMPDIR=$D/tmp PYTHONPATH=$D/code/tex3d:$D/code/tex3d/openvla-oft"

wait_marker() { # $1=file $2=marker $3=最大等待次数(×300s)
  local i=0
  while [ $i -lt "${3:-240}" ]; do
    grep -aq "$2" "$1" 2>/dev/null && { log "已见标记 $2"; return 0; }
    sleep 300; i=$((i+1))
  done
  log "FAIL-WAIT-TIMEOUT $2"; return 1
}

train_goal() { # $1=gpu $2=tid
  local g=$1 tid=$2
  nvidia-smi --id=$g --query-gpu=memory.used --format=csv,noheader
  for attempt in 1 2 3; do
    (cd $D/code/tex3d-ws/g7 && env CUDA_VISIBLE_DEVICES=$g MUJOCO_EGL_DEVICE_ID=$g $ENVV \
      "$PYT" "$TRD/attack_oft_tex3d_native.py" --iters 5000 --seed 7 --task-id $tid \
      --suite libero_goal --tag goal${tid}_r1 \
      > "$OUT/logs/train_goal_t${tid}.log" 2>&1) && break
    log "goal-t$tid 第$attempt次失败，120s 后重试"; sleep 120
  done
  TDIR=$(ls -dt "$OUT"/train_*_goal${tid}_r1 2>/dev/null | head -1)
  [ -n "$TDIR" ] && [ -f "$TDIR/akita_bowl_adv_baked.png" ] || { log "FAIL-GOAL-T$tid-TRAIN"; return 1; }
  log "goal-t$tid 训练完成 $TDIR"
  sha256sum "$TDIR/akita_bowl_adv_baked.png"
  echo "$TDIR" > "$OUT/logs/tdir_goal_t${tid}.txt"
}

cell_done() {
  local res n
  res=$(grep -a '\[eval\] 结果' "$OUT/logs/$1.log" 2>/dev/null | tail -1 | sed 's/.*结果: //' | tr -d '\r')
  n=$(grep -ac '^Task:' "$res" 2>/dev/null)
  [ -n "$res" ] && [ -f "$res" ] && [ "${n:-0}" -ge "$2" ]
}
run_cell() {  # $1=gpu $2=name $3=tasks $4=texture $5=ckpt $6=modelcfg $7=min_lines
  local g=$1 name=$2 tasks=$3 tex=$4 ckpt=$5 mcfg=${6:-} need=${7:-10} res n extra=()
  [ -n "$mcfg" ] && extra+=(--model-config "$mcfg")
  cell_done "$name" "$need" && { log "$name 已完成"; return 0; }
  log "评测 $name GPU$g (tasks=$tasks)"
  for attempt in 1 2 3 4; do
    env CLOTH_QUERY_SEED=1 bash "$PI/run_objtex_eval.sh" "$g" eval \
      --condition objtex --tasks "$tasks" --trials 10 --num-denoise 10 \
      --init-offset 10 --suite libero_goal --texture "$tex" --ckpt "$ckpt" \
      "${extra[@]}" >> "$OUT/logs/${name}.log" 2>&1
    res=$(grep -a '\[eval\] 结果' "$OUT/logs/$name.log" 2>/dev/null | tail -1 | sed 's/.*结果: //' | tr -d '\r')
    n=$(grep -ac '^Task:' "$res" 2>/dev/null)
    if [ -n "$res" ] && [ -f "$res" ] && [ "${n:-0}" -ge "$need" ]; then
      log "$name 完成 -> $(grep -a 'overall robot success' "$res" | tail -1)"; return 0
    fi
    sleep 45
  done
  log "$name 四次失败"; return 1
}

eval_task_pair() { # $1=gpu $2=tid
  local g=$1 tid=$2 tdir
  tdir=$(cat "$OUT/logs/tdir_goal_t${tid}.txt" 2>/dev/null)
  [ -n "$tdir" ] && [ -f "$tdir/akita_bowl_adv_baked.png" ] || { log "SKIP-T$tid-NO-TEXTURE"; return 1; }
  run_cell "$g" "t3dg_t${tid}_pi0"  "$tid" "$tdir/akita_bowl_adv_baked.png" "$CKPT0" "" 10
  run_cell "$g" "t3dg_t${tid}_pi05" "$tid" "$tdir/akita_bowl_adv_baked.png" "$CKPT5" pi05_libero 10
}

lane1() { local g=1; for tid in 0 4 8; do train_goal $g $tid && eval_task_pair $g $tid; done; log "DONE-T3D-GOAL-LANE1"; }
lane2() { local g=3; sleep 60; for tid in 1 5 9; do train_goal $g $tid && eval_task_pair $g $tid; done; log "DONE-T3D-GOAL-LANE2"; }
lane3() {
  local g=0
  wait_marker "$OUT/logs/chain_t8.log" "DONE-T3D-TASK8" 240 || exit 1
  sleep 120
  for tid in 2 6; do train_goal $g $tid && eval_task_pair $g $tid; done
  # 通用部署：g0 纹理挂 goal 全 10 任务（等 lane1 的 g0 产物）
  local i=0 tdir0=""
  while [ $i -lt 240 ]; do
    [ -f "$OUT/logs/tdir_goal_t0.txt" ] && { tdir0=$(cat "$OUT/logs/tdir_goal_t0.txt"); \
      [ -f "$tdir0/akita_bowl_adv_baked.png" ] && break; }
    sleep 300; i=$((i+1))
  done
  [ -n "$tdir0" ] && [ -f "$tdir0/akita_bowl_adv_baked.png" ] || { log "FAIL-NO-G0-TEXTURE"; log "DONE-T3D-GOAL-LANE3"; return; }
  run_cell "$g" "t3dg_uni_pi0"  "0,1,2,3,4,5,6,7,8,9" "$tdir0/akita_bowl_adv_baked.png" "$CKPT0" "" 100
  run_cell "$g" "t3dg_uni_pi05" "0,1,2,3,4,5,6,7,8,9" "$tdir0/akita_bowl_adv_baked.png" "$CKPT5" pi05_libero 100
  log "DONE-T3D-GOAL-LANE3"
}
lane4() {
  local g=2
  wait_marker "$D/code/tex3d/experiments/upa_native/logs/chain_resume.log" "DONE-UPA-RESUME" 240 || exit 1
  sleep 120
  for tid in 3 7; do train_goal $g $tid && eval_task_pair $g $tid; done
  run_cell "$g" "t3dg_th0_pi0"  "0,1,2,3,4,5,6,7,8,9" "$TH0" "$CKPT0" "" 100
  run_cell "$g" "t3dg_th0_pi05" "0,1,2,3,4,5,6,7,8,9" "$TH0" "$CKPT5" pi05_libero 100
  log "DONE-T3D-GOAL-LANE4"
}
"$1"
