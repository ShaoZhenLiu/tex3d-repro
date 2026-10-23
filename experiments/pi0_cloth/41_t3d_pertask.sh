#!/bin/bash
# 41_t3d_pertask.sh — Tex3D 逐任务粒度评测（变体b：每任务用自己的纹理）
#   task0/2/6 纹理已就绪（task8 两格由 41_t3d_task8.sh 训完自动补）。
#   泳道1 GPU1：pi0 三格；泳道2 GPU3：pi0.5 三格（60s 错峰，每格 10 回合）。
#   GPU 纪律：仅 0-3；启动前现查。
set -u
D=/data/shaozhen.liu
PI=$D/code/tex3d/pi
OUT=$D/code/tex3d/experiments/tex3d_native
CKPT0=$D/model/RLinf-Pi0-LIBERO-130-fullshot-SFT
CKPT5=$D/model/RLinf-Pi05-LIBERO-130-fullshot-SFT
mkdir -p "$OUT/logs"
log() { echo "[t3dp $(date +%m%d_%H:%M:%S)] $*"; }

TEX0=$OUT/train_20261023_072153_r1/akita_bowl_adv_baked.png
TEX2=$OUT/train_20261023_093631_task2_r1/akita_bowl_adv_baked.png
TEX6=$OUT/train_20261023_093630_task6_r1/akita_bowl_adv_baked.png
for t in "$TEX0" "$TEX2" "$TEX6"; do
  [ -f "$t" ] || { log "FAIL-TEXTURE-MISSING $t"; exit 1; }
done
sha256sum "$TEX0" "$TEX2" "$TEX6"

cell_done() {
  local res n
  res=$(grep -a '\[eval\] 结果' "$OUT/logs/$1.log" 2>/dev/null | tail -1 | sed 's/.*结果: //' | tr -d '\r')
  n=$(grep -ac '^Task:' "$res" 2>/dev/null)
  [ -n "$res" ] && [ -f "$res" ] && [ "${n:-0}" -ge 10 ]
}
run_cell() {  # $1=gpu $2=name $3=task $4=texture $5=ckpt $6=modelcfg
  local g=$1 name=$2 tid=$3 tex=$4 ckpt=$5 mcfg=${6:-} res n extra=()
  [ -n "$mcfg" ] && extra+=(--model-config "$mcfg")
  cell_done "$name" && { log "$name 已完成"; return 0; }
  log "评测 $name GPU$g (task$tid)"
  for attempt in 1 2 3 4; do
    env CLOTH_QUERY_SEED=1 bash "$PI/run_objtex_eval.sh" "$g" eval \
      --condition objtex --tasks "$tid" --trials 10 --num-denoise 10 \
      --init-offset 10 --suite libero_spatial --texture "$tex" --ckpt "$ckpt" \
      "${extra[@]}" >> "$OUT/logs/${name}.log" 2>&1
    res=$(grep -a '\[eval\] 结果' "$OUT/logs/$name.log" 2>/dev/null | tail -1 | sed 's/.*结果: //' | tr -d '\r')
    n=$(grep -ac '^Task:' "$res" 2>/dev/null)
    if [ -n "$res" ] && [ -f "$res" ] && [ "${n:-0}" -ge 10 ]; then
      log "$name 完成 -> $(grep -a 'overall robot success' "$res" | tail -1)"; return 0
    fi
    sleep 45
  done
  log "$name 四次失败"; return 1
}

lane_pi0() {
  local g=1
  nvidia-smi --id=$g --query-gpu=memory.used --format=csv,noheader
  run_cell $g t3dp_t0_pi0  0 "$TEX0" "$CKPT0"
  run_cell $g t3dp_t2_pi0  2 "$TEX2" "$CKPT0"
  run_cell $g t3dp_t6_pi0  6 "$TEX6" "$CKPT0"
  log "LANE-PI0-DONE"
}
lane_pi05() {
  local g=3
  sleep 60
  nvidia-smi --id=$g --query-gpu=memory.used --format=csv,noheader
  run_cell $g t3dp_t0_pi05 0 "$TEX0" "$CKPT5" pi05_libero
  run_cell $g t3dp_t2_pi05 2 "$TEX2" "$CKPT5" pi05_libero
  run_cell $g t3dp_t6_pi05 6 "$TEX6" "$CKPT5" pi05_libero
  log "LANE-PI05-DONE"
}
lane_pi0 & lane_pi05 & wait
log "DONE-T3D-PERTASK-026"
