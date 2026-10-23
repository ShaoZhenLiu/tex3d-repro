#!/bin/bash
# 41_t3d_deepen.sh — Tex3D 通用部署评测加深（用户选项A，2026-10-23 指示"加深到50"）
#   约束：每任务仅 50 个初态（0-49），T3D 训练用 0-9（train/eval 分离口径），
#   评测上限 = inits 10-49 = 40/任务（"50"不可达，除非混入训练初态——已向用户说明）。
#   执行：{Plain, Adv, θ0} × {π0, π0.5} × 任务{0,2,6,8} × inits10-49 = 6 格 × 160 = 960 回合。
#   Plain 实现 = objtex 条件注入原始资产纹理（两份拷贝 sha 一致 8a646d98…，机制中性无扰动）。
#   泳道：GPU1=π0 三格，GPU3=π0.5 三格（60s 错峰）。GPU 纪律：仅 0-3。
set -u
D=/data/shaozhen.liu
PI=$D/code/tex3d/pi
OUT=$D/code/tex3d/experiments/tex3d_native
CKPT0=$D/model/RLinf-Pi0-LIBERO-130-fullshot-SFT
CKPT5=$D/model/RLinf-Pi05-LIBERO-130-fullshot-SFT
ADV=$OUT/train_20261023_072153_r1/akita_bowl_adv_baked.png
TH0=$OUT/train_20261023_072153_r1/theta0_baked.png
PLAIN=$D/code/tex3d-ws/g7/LIBERO-master/libero/libero/assets/stable_scanned_objects/akita_black_bowl/texture.png
mkdir -p "$OUT/logs"
log() { echo "[t3dd $(date +%m%d_%H:%M:%S)] $*"; }
for t in "$ADV" "$TH0" "$PLAIN"; do [ -f "$t" ] || { log "FAIL-TEXTURE $t"; exit 1; }; done
sha256sum "$ADV" "$TH0" "$PLAIN"

cell_done() {
  local res n
  res=$(grep -a '\[eval\] 结果' "$OUT/logs/$1.log" 2>/dev/null | tail -1 | sed 's/.*结果: //' | tr -d '\r')
  n=$(grep -ac '^Task:' "$res" 2>/dev/null)
  [ -n "$res" ] && [ -f "$res" ] && [ "${n:-0}" -ge 160 ]
}
run_cell() {  # $1=gpu $2=name $3=texture $4=ckpt $5=modelcfg
  local g=$1 name=$2 tex=$3 ckpt=$4 mcfg=${5:-} res n extra=()
  [ -n "$mcfg" ] && extra+=(--model-config "$mcfg")
  cell_done "$name" && { log "$name 已完成"; return 0; }
  log "评测 $name GPU$g"
  for attempt in 1 2 3 4; do
    env CLOTH_QUERY_SEED=1 bash "$PI/run_objtex_eval.sh" "$g" eval \
      --condition objtex --tasks 0,2,6,8 --trials 40 --num-denoise 10 \
      --init-offset 10 --suite libero_spatial --texture "$tex" --ckpt "$ckpt" \
      "${extra[@]}" >> "$OUT/logs/${name}.log" 2>&1
    res=$(grep -a '\[eval\] 结果' "$OUT/logs/$name.log" 2>/dev/null | tail -1 | sed 's/.*结果: //' | tr -d '\r')
    n=$(grep -ac '^Task:' "$res" 2>/dev/null)
    if [ -n "$res" ] && [ -f "$res" ] && [ "${n:-0}" -ge 160 ]; then
      log "$name 完成 -> $(grep -a 'overall robot success' "$res" | tail -1)"; return 0
    fi
    sleep 45
  done
  log "$name 四次失败"; return 1
}

lane_pi0() {
  local g=1
  nvidia-smi --id=$g --query-gpu=memory.used --format=csv,noheader
  run_cell $g t3ddeep_plain_pi0  "$PLAIN" "$CKPT0"
  run_cell $g t3ddeep_adv_pi0    "$ADV"   "$CKPT0"
  run_cell $g t3ddeep_th0_pi0    "$TH0"   "$CKPT0"
  log "DONE-T3DDEEP-PI0"
}
lane_pi05() {
  local g=3
  sleep 60
  nvidia-smi --id=$g --query-gpu=memory.used --format=csv,noheader
  run_cell $g t3ddeep_plain_pi05 "$PLAIN" "$CKPT5" pi05_libero
  run_cell $g t3ddeep_adv_pi05   "$ADV"   "$CKPT5" pi05_libero
  run_cell $g t3ddeep_th0_pi05   "$TH0"   "$CKPT5" pi05_libero
  log "DONE-T3DDEEP-PI05"
}
lane_pi0 & lane_pi05 & wait
log "DONE-T3D-DEEPEN"
