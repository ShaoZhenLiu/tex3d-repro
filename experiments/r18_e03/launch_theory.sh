#!/bin/bash
# launch_theory.sh — 启动理论队列并自动写 GPU5 保留标记（真实队列 pid，防死认领被清）
cd /data/shaozhen.liu/code/tex3d/experiments/r18_e03
setsid nohup bash 41_theory_diag_queue.sh \
  >> /data/shaozhen.liu/code/tex3d/experiments/pi0_cloth/theory_queue_logs/nohup.out 2>&1 < /dev/null &
sleep 3
QP=$(pgrep -f 'bash 41_theory_diag_queue.sh' | head -1)
if [ -n "$QP" ]; then
  echo "$QP 31000" > /data/shaozhen.liu/code/tex3d/experiments/r18_e01e02/gpu_claims/theoryhold.5
fi
echo "LAUNCHED pid=$QP hold=$(cat /data/shaozhen.liu/code/tex3d/experiments/r18_e01e02/gpu_claims/theoryhold.5 2>/dev/null)"
