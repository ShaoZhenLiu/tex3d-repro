#!/bin/bash
# 41_r18bc_eval_master.sh — 评测总波自愈循环（与训练主控对称）
#   每 10 分钟：40/40 收口则退出；否则若评测波不在跑则幂等重启
#   （cell_done 跳过已完成格 → 重复启动安全）。
set -u
EX=/data/shaozhen.liu/code/tex3d/experiments/r18_e03
WD=/data/shaozhen.liu/code/tex3d/experiments/pi0_cloth/watchdog_events.log
while true; do
  if grep -q "DONE-R18BC-EVAL 40/40" "$EX/evalbc_logs/summary.txt" 2>/dev/null; then
    echo "$(date '+%F %T') [evalm] 40/40 收口，退出" >> "$WD"
    break
  fi
  if ! ps -u shaozhen.liu -o cmd | grep "41_r18bc_eval_wav[e]" >/dev/null 2>&1; then
    echo "$(date '+%F %T') [evalm] 评测波不在跑，幂等重启" >> "$WD"
    ( cd "$EX" && setsid nohup bash 41_r18bc_eval_wave.sh \
        >> eval_wave_master.log 2>&1 < /dev/null & )
  fi
  sleep 600
done
