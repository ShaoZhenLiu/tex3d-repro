#!/bin/bash
# 41_tvfix_eval.sh — 第13轮 阶段B/C 首轮评测：π0.5-Object + Spatial 各 100 回合
#   用法: bash 41_tvfix_eval.sh <bs1|bs8> <GPU> [<GPU2>]
#   条件名: mtf_pi05_{obj,sp}_tvfix<bs>；开查询种子记录
set -u
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
D=/data/shaozhen.liu
PI=$D/code/tex3d/pi
EX=$D/code/tex3d/experiments/pi0_cloth
LOGD=$EX/main_logs
WD=$EX/watchdog_events.log
CKPT5=$D/model/RLinf-Pi05-LIBERO-130-fullshot-SFT
BS=${1:?用法: 41_tvfix_eval.sh <bs1|bs8> <GPU> [<GPU2>]}
G1=${2:?GPU}
G2=${3:-$G1}
VDIR=$(ls -dt $EX/oft_cloth/train_*_tvfix-$BS 2>/dev/null | head -1)
TEX="$VDIR/Universal_Cloth_OV.png"
[ -f "$TEX" ] || { echo "$(date '+%F %T') [tvfix-eval] 纹理缺失 $TEX" >> "$WD"; exit 1; }

cell_done() {
  local res n
  res=$(grep -a '\[eval\] 结果' "$LOGD/$1.log" 2>/dev/null | tail -1 | sed 's/.*结果: //' | tr -d '\r')
  n=$(grep -c '^Task:' "$res" 2>/dev/null)
  [ -n "$res" ] && [ -f "$res" ] && [ "${n:-0}" -ge 100 ]
}

run_cell() {  # $1=gpu $2=name $3=suite
  local g=$1 name=$2 suite=$3
  if cell_done "$name"; then
    echo "$(date '+%F %T') [tvfix-eval] $name 已完成，跳过" >> "$WD"; return 0
  fi
  local log="$LOGD/${name}.log" res n
  echo "$(date '+%F %T') [tvfix-eval] $name GPU$g" >> "$WD"
  for attempt in 1 2 3 4; do
    CLOTH_QUERY_SEED=1 CLOTH_QUERY_SEED_LOG=$EX/edpa_logs/qseed_${name}.csv \
    bash "$PI/run_cloth330.sh" "$g" eval --condition adv --suite "$suite" \
      --tasks 0,1,2,3,4,5,6,7,8,9 --trials 10 --num-denoise 10 --init-offset 10 \
      --texture "$TEX" --ckpt "$CKPT5" --model-config pi05_libero >> "$log" 2>&1
    res=$(grep -a '\[eval\] 结果' "$log" 2>/dev/null | tail -1 | sed 's/.*结果: //' | tr -d '\r')
    n=$(grep -c '^Task:' "$res" 2>/dev/null)
    if [ -n "$res" ] && [ -f "$res" ] && [ "${n:-0}" -ge 100 ]; then
      echo "$(date '+%F %T') [tvfix-eval] $name 完成 -> $(grep -a 'overall robot success' "$res" | tail -1)" >> "$WD"
      return 0
    fi
    sleep 45
  done
  echo "$(date '+%F %T') [tvfix-eval] $name 四次失败 FAILED" >> "$WD"
  return 1
}

echo "$(date '+%F %T') [tvfix-eval] F0-TVfix-$BS 首轮 2 格（obj GPU$G1 / sp GPU$G2）tex=$TEX" >> "$WD"
rc=0
pids=()
( sleep 0;   run_cell "$G1" "mtf_pi05_obj_tvfix$BS" libero_object ) & pids+=($!)
( sleep 90;  run_cell "$G2" "mtf_pi05_sp_tvfix$BS"  libero_spatial ) & pids+=($!)
for p in "${pids[@]}"; do wait "$p" || rc=1; done

ok=0; sum=""
for name in "mtf_pi05_obj_tvfix$BS" "mtf_pi05_sp_tvfix$BS"; do
  cell_done "$name" && ok=$((ok+1))
  res=$(grep -a '\[eval\] 结果' "$LOGD/${name}.log" 2>/dev/null | tail -1 | sed 's/.*结果: //' | tr -d '\r')
  sum="$sum$name=$(grep -a 'overall robot success' "$res" 2>/dev/null | tail -1 | grep -o '[0-9]*/100') "
done
if [ "$rc" -eq 0 ] && [ "$ok" -eq 2 ]; then
  echo "$(date '+%F %T') DONE-TVFIX-${BS}-EVAL 2/2：$sum" >> "$WD"
else
  echo "$(date '+%F %T') DONE-TVFIX-${BS}-EVAL-WITH-FAILURES rc=$rc 终检 $ok/2：$sum" >> "$WD"
fi
