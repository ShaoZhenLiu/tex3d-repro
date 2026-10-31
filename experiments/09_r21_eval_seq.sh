#!/bin/bash
# 09_r21_eval_seq.sh — 阶段③评测·简化顺序版（不用子壳）
set -u
D=/data/shaozhen.liu
PI=$D/code/tex3d/pi
EX=$D/code/tex3d/experiments/r21_eval_09
LOGD=$EX/logs; DONED=$LOGD/done
CKPT5=$D/model/RLinf-Pi05-LIBERO-130-fullshot-SFT
TEXDIR=$D/code/tex3d/experiments/pi0_cloth/oft_cloth
mkdir -p "$LOGD" "$DONED"
echo "[$(date '+%m-%d %H:%M:%S')] seq eval 启动 GPU=$1 pid=$$" >> "$LOGD/master.log"
GPU=$1

mem_free() { nvidia-smi --id="$GPU" --query-gpu=memory.free --format=csv,noheader,nounits | head -1; }

run_cell() {
  local name=$1 tex=$2 suite=$3
  local log_f="$LOGD/${name}.log"
  [ -f "$DONED/$name.done" ] && { echo "[$(date '+%m-%d %H:%M:%S')] $name skip" >> "$LOGD/master.log"; return 0; }
  local attempt res n
  for attempt in 1 2 3 4; do
    local w=0
    while [ "$(mem_free)" -lt 22000 ]; do
      sleep 120; w=$((w+1)); [ $w -gt 120 ] && return 1
    done
    echo "[$(date '+%m-%d %H:%M:%S')] $name att$attempt GPU$GPU free=$(mem_free)" >> "$LOGD/master.log"
    bash "$PI/run_cloth330.sh" "$GPU" eval --condition adv --suite "$suite" \
      --tasks 0,1,2,3,4,5,6,7,8,9 --trials 10 --num-denoise 10 --init-offset 10 \
      --max-steps 400 --texture "$tex" --ckpt "$CKPT5" --model-config pi05_libero \
      >> "$log_f" 2>&1
    res=$(grep -a '\[eval\] 结果' "$log_f" 2>/dev/null | tail -1 | sed 's/.*结果: //' | tr -d '\r')
    n=$(grep -ac '^Task:' "$res" 2>/dev/null)
    if [ -n "$res" ] && [ -f "$res" ] && [ "${n:-0}" -ge 100 ]; then
      echo "$res" > "$DONED/$name.done"
      echo "[$(date '+%m-%d %H:%M:%S')] $name DONE $(grep -a 'overall' "$res" | tail -1)" >> "$LOGD/master.log"
      return 0
    fi
    sleep 300
  done
  echo "[$(date '+%m-%d %H:%M:%S')] $name FAILED" >> "$LOGD/master.log"
}

for arm in n2m_s7 m2n_s8 n2m_s8 m2n_s9; do
  tex="$(ls -dt $TEXDIR/train_*_r21_$arm 2>/dev/null | head -1)/Universal_Cloth_OV.png"
  [ -f "$tex" ] || continue
  run_cell "r21_${arm}_sp" "$tex" libero_spatial
  run_cell "r21_${arm}_obj" "$tex" libero_object
done
echo "[$(date '+%m-%d %H:%M:%S')] seq eval 收口 done=$(ls "$DONED" | wc -l)/8" >> "$LOGD/master.log"
