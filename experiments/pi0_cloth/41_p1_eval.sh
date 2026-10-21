#!/bin/bash
# 41_p1_eval.sh — 第12轮 P1 幅度校准：π0.5-Object 两格（EDPA@F0-RMS / UPA@F0-RMS）
#   各 100 回合（init 10-19），GPU0 两泳道 90s 错峰；开启按查询种子记录（老师 §4 P1）
#   只对新增条件设 CLOTH_QUERY_SEED=1（评测器补丁默认关闭，历史行为不变）
set -u
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
D=/data/shaozhen.liu
PI=$D/code/tex3d/pi
EX=$D/code/tex3d/experiments/pi0_cloth
LOGD=$EX/main_logs
WD=$EX/watchdog_events.log
CKPT5=$D/model/RLinf-Pi05-LIBERO-130-fullshot-SFT

cell_done() {
  local res n
  res=$(grep -a '\[eval\] 结果' "$LOGD/$1.log" 2>/dev/null | tail -1 | sed 's/.*结果: //' | tr -d '\r')
  n=$(grep -c '^Task:' "$res" 2>/dev/null)
  [ -n "$res" ] && [ -f "$res" ] && [ "${n:-0}" -ge 100 ]
}

run_cell() {  # $1=gpu $2=name $3=texture
  local g=$1 name=$2 tex=$3
  if cell_done "$name"; then
    echo "$(date '+%F %T') [p1] $name 已完成，跳过" >> "$WD"; return 0
  fi
  local log="$LOGD/${name}.log" res n
  echo "$(date '+%F %T') [p1] $name GPU$g tex=$tex" >> "$WD"
  for attempt in 1 2 3 4; do
    CLOTH_QUERY_SEED=1 CLOTH_QUERY_SEED_LOG=$EX/edpa_logs/qseed_${name}.csv \
    bash "$PI/run_cloth330.sh" "$g" eval --condition adv --suite libero_object \
      --tasks 0,1,2,3,4,5,6,7,8,9 --trials 10 --num-denoise 10 --init-offset 10 \
      --texture "$tex" --ckpt "$CKPT5" --model-config pi05_libero >> "$log" 2>&1
    res=$(grep -a '\[eval\] 结果' "$log" 2>/dev/null | tail -1 | sed 's/.*结果: //' | tr -d '\r')
    n=$(grep -c '^Task:' "$res" 2>/dev/null)
    if [ -n "$res" ] && [ -f "$res" ] && [ "${n:-0}" -ge 100 ]; then
      echo "$(date '+%F %T') [p1] $name 完成 -> $(grep -a 'overall robot success' "$res" | tail -1)" >> "$WD"
      return 0
    fi
    sleep 45
  done
  echo "$(date '+%F %T') [p1] $name 四次失败 FAILED" >> "$WD"
  return 1
}

echo "$(date '+%F %T') [p1] 幅度校准启动：2 格 × 100 回合（GPU0 两泳道）" >> "$WD"
rc=0
pids=()
( sleep 0;  run_cell 0 mtf_pi05_obj_edpav2rms "$EX/texpush/edpav2_f0rms.png" ) & pids+=($!)
( sleep 90; run_cell 0 mtf_pi05_obj_uparms    "$EX/texpush/upa_f0rms.png" ) & pids+=($!)
for p in "${pids[@]}"; do wait "$p" || rc=1; done

ok=0; sum=""
for name in mtf_pi05_obj_edpav2rms mtf_pi05_obj_uparms; do
  cell_done "$name" && ok=$((ok+1))
  res=$(grep -a '\[eval\] 结果' "$LOGD/${name}.log" 2>/dev/null | tail -1 | sed 's/.*结果: //' | tr -d '\r')
  sum="$sum$name=$(grep -a 'overall robot success' "$res" 2>/dev/null | tail -1 | grep -o '[0-9]*/100') "
done
if [ "$rc" -eq 0 ] && [ "$ok" -eq 2 ]; then
  echo "$(date '+%F %T') DONE-P1EVAL 2/2 全部完成：$sum" >> "$WD"
else
  echo "$(date '+%F %T') DONE-P1EVAL-WITH-FAILURES rc=$rc 终检 $ok/2：$sum" >> "$WD"
fi
