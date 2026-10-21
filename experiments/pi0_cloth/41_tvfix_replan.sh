#!/bin/bash
# replan: sp-bs8@GPU2 + bs1rms-obj@GPU3 + bs8rms-obj@GPU0 并行；全 4 格完成写 DONE
set -u
D=/data/shaozhen.liu
PI=$D/code/tex3d/pi
EX=$D/code/tex3d/experiments/pi0_cloth
LOGD=$EX/main_logs
WD=$EX/watchdog_events.log
CKPT5=$D/model/RLinf-Pi05-LIBERO-130-fullshot-SFT
cell_done() {
  local res n
  res=$(grep -a "\[eval\] 结果" "$LOGD/$1.log" 2>/dev/null | tail -1 | sed "s/.*结果: //" | tr -d "\r")
  n=$(grep -c "^Task:" "$res" 2>/dev/null)
  [ -n "$res" ] && [ -f "$res" ] && [ "${n:-0}" -ge 100 ]
}
run_cell() {
  local g=$1 name=$2 suite=$3 tex=$4
  if cell_done "$name"; then echo "$(date "+%F %T") [replan] $name 已完成" >> "$WD"; return 0; fi
  local log="$LOGD/${name}.log" res n
  echo "$(date "+%F %T") [replan] $name GPU$g" >> "$WD"
  for attempt in 1 2 3 4; do
    CLOTH_QUERY_SEED=1 CLOTH_QUERY_SEED_LOG=$EX/edpa_logs/qseed_${name}.csv \
    bash "$PI/run_cloth330.sh" "$g" eval --condition adv --suite "$suite" \
      --tasks 0,1,2,3,4,5,6,7,8,9 --trials 10 --num-denoise 10 --init-offset 10 \
      --texture "$tex" --ckpt "$CKPT5" --model-config pi05_libero >> "$log" 2>&1
    res=$(grep -a "\[eval\] 结果" "$log" 2>/dev/null | tail -1 | sed "s/.*结果: //" | tr -d "\r")
    n=$(grep -c "^Task:" "$res" 2>/dev/null)
    if [ -n "$res" ] && [ -f "$res" ] && [ "${n:-0}" -ge 100 ]; then
      echo "$(date "+%F %T") [replan] $name 完成 -> $(grep -a "overall robot success" "$res" | tail -1)" >> "$WD"; return 0
    fi
    sleep 45
  done
  echo "$(date "+%F %T") [replan] $name 四次失败 FAILED" >> "$WD"; return 1
}
BS8TEX="$EX/oft_cloth/train_20261021_153634_tvfix-bs8/Universal_Cloth_OV.png"
pids=()
( run_cell 2 mtf_pi05_sp_tvfixbs8   libero_spatial "$BS8TEX" ) & pids+=($!)
( run_cell 3 mtf_pi05_obj_bs1fixrms libero_object  "$EX/texpush/bs1fix_f0rms.png" ) & pids+=($!)
( run_cell 0 mtf_pi05_obj_bs8fixrms libero_object  "$EX/texpush/bs8fix_f0rms.png" ) & pids+=($!)
for p in "${pids[@]}"; do wait "$p"; done
# 终检含 GPU5 上已在跑的 obj-bs8
ok=0; sum=""
for name in mtf_pi05_obj_tvfixbs8 mtf_pi05_sp_tvfixbs8 mtf_pi05_obj_bs1fixrms mtf_pi05_obj_bs8fixrms; do
  cell_done "$name" && ok=$((ok+1))
  res=$(grep -a "\[eval\] 结果" "$LOGD/${name}.log" 2>/dev/null | tail -1 | sed "s/.*结果: //" | tr -d "\r")
  sum="$sum$name=$(grep -a "overall robot success" "$res" 2>/dev/null | tail -1 | grep -o "[0-9]*/100") "
done
echo "$(date "+%F %T") DONE-TVFIX-ROUND2 $ok/4：$sum" >> "$WD"
