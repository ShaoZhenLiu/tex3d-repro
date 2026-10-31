#!/bin/bash
# 09_r21_eval_wave.sh — 阶段③混合臂评测波（09，4臂×π0.5-Sp/Obj×100=800回合）
# GPU1/GPU2 各一泳道，每泳道串行跑分配到的臂
set -u
D=/data/shaozhen.liu
PI=$D/code/tex3d/pi
EX=$D/code/tex3d/experiments/r21_eval_09
LOGD=$EX/logs; DONED=$LOGD/done
CKPT5=$D/model/RLinf-Pi05-LIBERO-130-fullshot-SFT
mkdir -p "$LOGD" "$DONED"
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" >> "$LOGD/master.log"; }
mem_free() { nvidia-smi --id="$1" --query-gpu=memory.free --format=csv,noheader,nounits | head -1; }

# 纹理路径（09 上已有 4 臂训练目录）
TEXDIR=$D/code/tex3d/experiments/pi0_cloth/oft_cloth
get_tex() { ls -dt $TEXDIR/train_*_r21_$1 2>/dev/null | head -1; }

run_cell() {  # $1=name $2=gpu $3=texture $4=suite
  local name=$1 g=$2 tex=$3 suite=$4 res n attempt log_f="$LOGD/${name}.log"
  [ -f "$DONED/$name.done" ] && { log "$name 跳过"; return 0; }
  for attempt in 1 2 3 4; do
    local w=0
    while [ "$(mem_free "$g")" -lt 22000 ]; do sleep 120; w=$((w+1)); [ $w -gt 60 ] && return 1; done
    log "$name 尝试$attempt GPU$g tex=$(sha256sum "$tex" | cut -c1-8) free=$(mem_free "$g")"
    bash "$PI/run_cloth330.sh" "$g" eval --condition adv --suite "$suite" \
      --tasks 0,1,2,3,4,5,6,7,8,9 --trials 10 --num-denoise 10 --init-offset 10 \
      --max-steps 400 --texture "$tex" --ckpt "$CKPT5" --model-config pi05_libero \
      >> "$log_f" 2>&1
    res=$(grep -a '\[eval\] 结果' "$log_f" 2>/dev/null | tail -1 | sed 's/.*结果: //' | tr -d '\r')
    n=$(grep -ac '^Task:' "$res" 2>/dev/null)
    if [ -n "$res" ] && [ -f "$res" ] && [ "${n:-0}" -ge 100 ]; then
      echo "$res" > "$DONED/$name.done"
      log "$name 完成 -> $(grep -a 'overall robot success' "$res" | tail -1)"
      return 0
    fi
    sleep 120
  done
  log "$name FAILED"
}

ARMS_L1="n2m_s7 m2n_s8"   # GPU1 泳道
ARMS_L2="m2n_s8 m2n_s9"   # GPU2 泳道

# GPU1 泳道
( for arm in $ARMS_L1; do
    tex="$(get_tex $arm)/Universal_Cloth_OV.png"
    [ -f "$tex" ] || { log "臂 $arm 纹理缺失"; continue; }
    log "臂 $arm 纹理 sha=$(sha256sum "$tex" | cut -c1-12)"
    run_cell r21_${arm}_sp 1 "$tex" libero_spatial
    run_cell r21_${arm}_obj 1 "$tex" libero_object
  done ) &
sleep 120
# GPU2 泳道
( for arm in $ARMS_L2; do
    tex="$(get_tex $arm)/Universal_Cloth_OV.png"
    [ -f "$tex" ] || { log "臂 $arm 纹理缺失"; continue; }
    log "臂 $arm 纹理 sha=$(sha256sum "$tex" | cut -c1-12)"
    run_cell r21_${arm}_sp 2 "$tex" libero_spatial
    run_cell r21_${arm}_obj 2 "$tex" libero_object
  done ) &

log "r21 评测波拉起（GPU1×2臂 + GPU2×2臂，120s 错峰）"
wait
log "r21 评测收口 done=$(ls "$DONED" | wc -l)/8"
