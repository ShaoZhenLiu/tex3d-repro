#!/bin/bash
# 41_c1_eval.sh — 第12轮 C1 开发格：π0.5-Object native / π0.5-Spatial native（查退化）
#   可选第 4 格 obj-rms（衰减到 F0 RMS 后，纹理就绪后由 41_c1_eval_rms.sh 单独跑）
#   全部开 CLOTH_QUERY_SEED 记录；判读：与 EDPA-native(B2v2) 同格直接配对
set -u
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
D=/data/shaozhen.liu
PI=$D/code/tex3d/pi
EX=$D/code/tex3d/experiments/pi0_cloth
LOGD=$EX/main_logs
WD=$EX/watchdog_events.log
CKPT5=$D/model/RLinf-Pi05-LIBERO-130-fullshot-SFT
VDIR=$(ls -dt $EX/oft_cloth/train_*_c1* 2>/dev/null | grep -v c1diag | head -1)
TEX="$VDIR/Universal_Cloth_OV.png"
[ -f "$TEX" ] || { echo "$(date '+%F %T') [c1eval] 纹理缺失 $TEX" >> "$WD"; exit 1; }

cell_done() {
  local res n
  res=$(grep -a '\[eval\] 结果' "$LOGD/$1.log" 2>/dev/null | tail -1 | sed 's/.*结果: //' | tr -d '\r')
  n=$(grep -c '^Task:' "$res" 2>/dev/null)
  [ -n "$res" ] && [ -f "$res" ] && [ "${n:-0}" -ge 100 ]
}

run_cell() {  # $1=gpu $2=name $3=suite $4=texture
  local g=$1 name=$2 suite=$3 tex=$4
  if cell_done "$name"; then
    echo "$(date '+%F %T') [c1eval] $name 已完成，跳过" >> "$WD"; return 0
  fi
  local log="$LOGD/${name}.log" res n
  echo "$(date '+%F %T') [c1eval] $name GPU$g" >> "$WD"
  for attempt in 1 2 3 4; do
    CLOTH_QUERY_SEED=1 CLOTH_QUERY_SEED_LOG=$EX/edpa_logs/qseed_${name}.csv \
    bash "$PI/run_cloth330.sh" "$g" eval --condition adv --suite "$suite" \
      --tasks 0,1,2,3,4,5,6,7,8,9 --trials 10 --num-denoise 10 --init-offset 10 \
      --texture "$tex" --ckpt "$CKPT5" --model-config pi05_libero >> "$log" 2>&1
    res=$(grep -a '\[eval\] 结果' "$log" 2>/dev/null | tail -1 | sed 's/.*结果: //' | tr -d '\r')
    n=$(grep -c '^Task:' "$res" 2>/dev/null)
    if [ -n "$res" ] && [ -f "$res" ] && [ "${n:-0}" -ge 100 ]; then
      echo "$(date '+%F %T') [c1eval] $name 完成 -> $(grep -a 'overall robot success' "$res" | tail -1)" >> "$WD"
      return 0
    fi
    sleep 45
  done
  echo "$(date '+%F %T') [c1eval] $name 四次失败 FAILED" >> "$WD"
  return 1
}

echo "$(date '+%F %T') [c1eval] C1 开发格启动（obj-native GPU3 / sp-native GPU1 / obj-rms GPU3）" >> "$WD"
rc=0
pids=()
( sleep 0;   run_cell 3 mtf_pi05_obj_c1 libero_object "$TEX" ) & pids+=($!)
( sleep 90;  run_cell 1 mtf_pi05_sp_c1  libero_spatial "$TEX" ) & pids+=($!)
( sleep 180; run_cell 3 mtf_pi05_obj_c1rms libero_object "$EX/texpush/c1_f0rms.png" ) & pids+=($!)
for p in "${pids[@]}"; do wait "$p" || rc=1; done

ok=0; sum=""
for name in mtf_pi05_obj_c1 mtf_pi05_sp_c1 mtf_pi05_obj_c1rms; do
  cell_done "$name" && ok=$((ok+1))
  res=$(grep -a '\[eval\] 结果' "$LOGD/${name}.log" 2>/dev/null | tail -1 | sed 's/.*结果: //' | tr -d '\r')
  sum="$sum$name=$(grep -a 'overall robot success' "$res" 2>/dev/null | tail -1 | grep -o '[0-9]*/100') "
done
if [ "$rc" -eq 0 ] && [ "$ok" -eq 3 ]; then
  echo "$(date '+%F %T') DONE-C1EVAL 3/3：$sum" >> "$WD"
else
  echo "$(date '+%F %T') DONE-C1EVAL-WITH-FAILURES rc=$rc 终检 $ok/3：$sum" >> "$WD"
fi
