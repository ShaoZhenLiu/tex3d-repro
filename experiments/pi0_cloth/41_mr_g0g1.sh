#!/bin/bash
# GPU4 损坏后 π0-obj/π0-10 两格改道 GPU0/GPU1（append 同名 log，供主编排终检）
set -u
D=/data/shaozhen.liu
PI=$D/code/tex3d/pi
EX=$D/code/tex3d/experiments/pi0_cloth
LOGD=$EX/main_logs
WD=$EX/watchdog_events.log
CKPT0=$D/model/RLinf-Pi0-LIBERO-130-fullshot-SFT
MRTEX="$EX/oft_cloth/train_20261021_224922_tvfix-mr/Universal_Cloth_OV.png"
cell_done() {
  local res n
  res=$(grep -a "\[eval\] 结果" "$LOGD/$1.log" 2>/dev/null | tail -1 | sed "s/.*结果: //" | tr -d "\r")
  n=$(grep -c "^Task:" "$res" 2>/dev/null)
  [ -n "$res" ] && [ -f "$res" ] && [ "${n:-0}" -ge 100 ]
}
run_cell() {
  local g=$1 name=$2 suite=$3
  if cell_done "$name"; then echo "$(date "+%F %T") [g0g1] $name 已完成" >> "$WD"; return 0; fi
  local log="$LOGD/${name}.log" res n
  echo "$(date "+%F %T") [g0g1] $name GPU$g" >> "$WD"
  for attempt in 1 2 3 4; do
    CLOTH_QUERY_SEED=1 CLOTH_QUERY_SEED_LOG=$EX/edpa_logs/qseed_${name}.csv \
    bash "$PI/run_cloth330.sh" "$g" eval --condition adv --suite "$suite" \
      --tasks 0,1,2,3,4,5,6,7,8,9 --trials 10 --num-denoise 10 --init-offset 10 \
      --texture "$MRTEX" --ckpt "$CKPT0" >> "$log" 2>&1
    res=$(grep -a "\[eval\] 结果" "$log" 2>/dev/null | tail -1 | sed "s/.*结果: //" | tr -d "\r")
    n=$(grep -c "^Task:" "$res" 2>/dev/null)
    if [ -n "$res" ] && [ -f "$res" ] && [ "${n:-0}" -ge 100 ]; then
      echo "$(date "+%F %T") [g0g1] $name 完成 -> $(grep -a "overall robot success" "$res" | tail -1)" >> "$WD"; return 0
    fi
    sleep 45
  done
  echo "$(date "+%F %T") [g0g1] $name 四次失败 FAILED" >> "$WD"; return 1
}
pids=()
( sleep 0;  run_cell 0 mtf_pi0_obj_mr libero_object ) & pids+=($!)
( sleep 120; run_cell 1 mtf_pi0_10_mr libero_10 ) & pids+=($!)
for p in "${pids[@]}"; do wait "$p"; done
