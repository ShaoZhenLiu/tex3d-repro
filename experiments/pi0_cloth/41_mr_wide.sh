#!/bin/bash
# 41_mr_wide.sh — MR 候选扩展评测（第13轮, 用户指示）：π0 四套件 + π0.5 Goal/Long
#   （补齐 800 的其余 6 格；π0.5 obj/sp 已有计入）+ MR@RMS 衰减格 + 金丝雀
#   8 泳道 / 4 卡（每卡 2），cell_done 幂等，查询种子记录开启
set -u
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
D=/data/shaozhen.liu
PI=$D/code/tex3d/pi
EX=$D/code/tex3d/experiments/pi0_cloth
LOGD=$EX/main_logs
WD=$EX/watchdog_events.log
CKPT5=$D/model/RLinf-Pi05-LIBERO-130-fullshot-SFT
CKPT0=$D/model/RLinf-Pi0-LIBERO-130-fullshot-SFT
MRTEX="$EX/oft_cloth/train_20261021_224922_tvfix-mr/Universal_Cloth_OV.png"
F0TEX=$EX/texpush/f0_seed7.png

cell_done() {
  local res n
  res=$(grep -a "\[eval\] 结果" "$LOGD/$1.log" 2>/dev/null | tail -1 | sed "s/.*结果: //" | tr -d "\r")
  n=$(grep -c "^Task:" "$res" 2>/dev/null)
  [ -n "$res" ] && [ -f "$res" ] && [ "${n:-0}" -ge 100 ]
}

run_cell() {  # $1=gpu $2=name $3=suite $4=texture $5=ckpt $6=modelcfg(optional)
  local g=$1 name=$2 suite=$3 tex=$4 ckpt=$5 mcfg=${6:-}
  if cell_done "$name"; then
    echo "$(date "+%F %T") [mrwide] $name 已完成，跳过" >> "$WD"; return 0
  fi
  local log="$LOGD/${name}.log" res n extra=()
  [ -n "$mcfg" ] && extra+=(--model-config "$mcfg")
  echo "$(date "+%F %T") [mrwide] $name GPU$g" >> "$WD"
  for attempt in 1 2 3 4; do
    CLOTH_QUERY_SEED=1 CLOTH_QUERY_SEED_LOG=$EX/edpa_logs/qseed_${name}.csv \
    bash "$PI/run_cloth330.sh" "$g" eval --condition adv --suite "$suite" \
      --tasks 0,1,2,3,4,5,6,7,8,9 --trials 10 --num-denoise 10 --init-offset 10 \
      --texture "$tex" --ckpt "$ckpt" "${extra[@]}" >> "$log" 2>&1
    res=$(grep -a "\[eval\] 结果" "$log" 2>/dev/null | tail -1 | sed "s/.*结果: //" | tr -d "\r")
    n=$(grep -c "^Task:" "$res" 2>/dev/null)
    if [ -n "$res" ] && [ -f "$res" ] && [ "${n:-0}" -ge 100 ]; then
      echo "$(date "+%F %T") [mrwide] $name 完成 -> $(grep -a "overall robot success" "$res" | tail -1)" >> "$WD"
      return 0
    fi
    sleep 45
  done
  echo "$(date "+%F %T") [mrwide] $name 四次失败 FAILED" >> "$WD"
  return 1
}

echo "$(date "+%F %T") [mrwide] MR 扩展 8 泳道启动（GPU2/4/5/6 各 2）" >> "$WD"
rc=0
pids=()
( sleep 0;   run_cell 2 mtf_pi0_sp_mr    libero_spatial "$MRTEX" "$CKPT0" ) & pids+=($!)
( sleep 360; run_cell 2 mtf_pi0_go_mr    libero_goal   "$MRTEX" "$CKPT0" ) & pids+=($!)
( sleep 180; run_cell 4 mtf_pi0_obj_mr   libero_object "$MRTEX" "$CKPT0" ) & pids+=($!)
( sleep 540; run_cell 4 mtf_pi0_10_mr    libero_10     "$MRTEX" "$CKPT0" ) & pids+=($!)
( sleep 90;  run_cell 5 mtf_pi05_go_mr   libero_goal   "$MRTEX" "$CKPT5" pi05_libero ) & pids+=($!)
( sleep 450; run_cell 5 mtf_pi05_10_mr   libero_10     "$MRTEX" "$CKPT5" pi05_libero ) & pids+=($!)
( sleep 270; run_cell 6 mtf_pi05_obj_mrrms libero_object "$EX/texpush/mrfix_f0rms.png" "$CKPT5" pi05_libero ) & pids+=($!)
( sleep 630; run_cell 6 mtf_canary3_pi0_sp_f0 libero_spatial "$F0TEX" "$CKPT0" ) & pids+=($!)
for p in "${pids[@]}"; do wait "$p" || rc=1; done

ok=0; sum=""
for name in mtf_pi0_sp_mr mtf_pi0_obj_mr mtf_pi0_go_mr mtf_pi0_10_mr mtf_pi05_go_mr mtf_pi05_10_mr mtf_pi05_obj_mrrms mtf_canary3_pi0_sp_f0; do
  cell_done "$name" && ok=$((ok+1))
  res=$(grep -a "\[eval\] 结果" "$LOGD/${name}.log" 2>/dev/null | tail -1 | sed "s/.*结果: //" | tr -d "\r")
  sum="$sum$name=$(grep -a "overall robot success" "$res" 2>/dev/null | tail -1 | grep -o "[0-9]*/100") "
done
CRES=$(grep -a "\[eval\] 结果" "$LOGD/mtf_canary3_pi0_sp_f0.log" 2>/dev/null | tail -1 | sed "s/.*结果: //" | tr -d "\r")
CS=$(grep -a "overall robot success" "$CRES" 2>/dev/null | tail -1 | grep -o "[0-9]*/100" | cut -d/ -f1)
echo "$(date "+%F %T") [mrwide] 金丝雀 succ=${CS:-未出}（历史 42±15）" >> "$WD"
if [ "$rc" -eq 0 ] && [ "$ok" -eq 8 ]; then
  echo "$(date "+%F %T") DONE-MR-WIDE 8/8：$sum" >> "$WD"
else
  echo "$(date "+%F %T") DONE-MR-WIDE-WITH-FAILURES rc=$rc 终检 $ok/8：$sum" >> "$WD"
fi
