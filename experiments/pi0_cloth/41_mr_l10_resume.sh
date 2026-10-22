#!/bin/bash
# 续跑方案（用户指示）：pi05-10 部分文件已含 task0-8 共90回合 → 只跑 task9 的10回合 → 合并为100
set -u
D=/data/shaozhen.liu
PI=$D/code/tex3d/pi
EX=$D/code/tex3d/experiments/pi0_cloth
LOGD=$EX/main_logs
WD=$EX/watchdog_events.log
CKPT5=$D/model/RLinf-Pi05-LIBERO-130-fullshot-SFT
MRTEX="$EX/oft_cloth/train_20261021_224922_tvfix-mr/Universal_Cloth_OV.png"
name=mtf_pi05_10_mr
log=$LOGD/${name}.log
for attempt in 1 2 3; do
  CLOTH_QUERY_SEED=1 CLOTH_QUERY_SEED_LOG=$EX/edpa_logs/qseed_${name}_t9.csv \
  bash "$PI/run_cloth330.sh" 0 eval --condition adv --suite libero_10 \
    --tasks 9 --trials 10 --num-denoise 10 --init-offset 10 \
    --texture "$MRTEX" --ckpt "$CKPT5" --model-config pi05_libero >> "$log.t9" 2>&1
  res=$(grep -a "\[eval\] 结果" "$log.t9" 2>/dev/null | tail -1 | sed "s/.*结果: //" | tr -d "\r")
  n=$(grep -c "^Task:" "$res" 2>/dev/null)
  if [ -n "$res" ] && [ -f "$res" ] && [ "${n:-0}" -ge 10 ]; then
    merged="$EX/eval_merged_${name}_$(date +%H%M%S)/evaluation_result.txt"
    mkdir -p "$(dirname "$merged")"
    { echo "condition=adv (merged: tasks0-8 from gpu5partial + task9 fresh run; reason: GPU5 died at 90/100)"; echo; cat "$res"; } > "$merged"
    grep -a "^Task:" "$LOGD/${name}.log.gpu5partial" >> "$merged"
    mn=$(grep -c "^Task:" "$merged")
    succ=$(grep -ac "Success: True" "$merged")
    echo "overall robot success: $succ/$mn" >> "$merged"
    echo "[eval] 结果: $merged" >> "$log"
    echo "$(date "+%F %T") [gpu0-l10] 续跑合并完成: task9=$res + 90回合部分 → $merged ($succ/$mn)" >> "$WD"
    exit 0
  fi
  sleep 45
done
echo "$(date "+%F %T") [gpu0-l10] task9 补跑三次失败 FAILED" >> "$WD"
