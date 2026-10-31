#!/bin/bash
# 41_watch_redo.sh — 等 GPU0 主master(1713147)退出后重发 r6 0 spatial,goal
# （补 pre-patch 失败的 plain_sp；done 幂等跳过已完格）
set -u
EX=/data/shaozhen.liu/code/tex3d/experiments/r24_groot_eval
W=$EX/logs/watch_redo_plain.log
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" >> "$W"; }
log "等待 GPU0 主master 1713147 退出后重发（补 plain_sp）"
while kill -0 1713147 2>/dev/null; do sleep 300; done
log "master 已退出；重发 r6 0 spatial,goal（done 幂等，只补失败格）"
cd /data/shaozhen.liu/code/tex3d/experiments && setsid nohup bash 41_gr00t_eval_wave_r6.sh 0 spatial,goal >> $EX/logs/master_redo_gpu0.log 2>&1 < /dev/null &
log "重发完成（pid $!）"
