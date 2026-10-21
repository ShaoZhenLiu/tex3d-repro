#!/bin/bash
# bs1/bs8 训完自动触发首轮评测（π0.5 obj+sp 各100）
set -u
EX=/data/shaozhen.liu/code/tex3d/experiments/pi0_cloth
WD=$EX/watchdog_events.log
while ! grep -aq "完成，产物" $EX/edpa_logs/train_tvfix_bs1.log 2>/dev/null; do sleep 60; done
echo "$(date "+%F %T") [tvfix-chain] bs1 训完 → 启动首轮评测" >> "$WD"
bash $EX/41_tvfix_eval.sh bs1 1 1
while ! grep -aq "完成，产物" $EX/edpa_logs/train_tvfix_bs8.log 2>/dev/null; do sleep 120; done
echo "$(date "+%F %T") [tvfix-chain] bs8 训完 → 启动首轮评测" >> "$WD"
bash $EX/41_tvfix_eval.sh bs8 3 3
