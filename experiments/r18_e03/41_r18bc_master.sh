#!/bin/bash
# 41_r18bc_master.sh — 训练队列主控自愈循环
#   每 10 分钟：summary 出现 DONE-R18BC-TRAIN 10/10 则退出；
#   否则若没有存活队列实例则以幂等方式重启（job_done 跳过已完成 +
#   run_job 防双跑门等待在跑同标签作业 → 重复启动安全）。
set -u
EX=/data/shaozhen.liu/code/tex3d/experiments/r18_e03
WD=/data/shaozhen.liu/code/tex3d/experiments/pi0_cloth/watchdog_events.log
while true; do
  if grep -q "DONE-R18BC-TRAIN 10/10" "$EX/train_logs/summary.txt" 2>/dev/null; then
    echo "$(date '+%F %T') [master] 10/10 收口，退出" >> "$WD"
    break
  fi
  if ! ps -u shaozhen.liu -o cmd | grep "41_r18bc_train_queu[e]" >/dev/null 2>&1; then
    echo "$(date '+%F %T') [master] 队列不在跑，幂等重启" >> "$WD"
    ( cd "$EX" && setsid nohup bash 41_r18bc_train_queue.sh \
        >> train_queue_master.log 2>&1 < /dev/null & )
  fi
  sleep 600
done
