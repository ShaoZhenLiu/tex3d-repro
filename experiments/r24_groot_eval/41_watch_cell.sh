#!/bin/bash
# 41_watch_cell.sh — 等 GR00T 单格结果满100行 → 封存done → 发该套件余下条件泳道
# 用法: 41_watch_cell.sh <cell名> <GPU> <suite>
set -u
EX=/data/shaozhen.liu/code/tex3d/experiments/r24_groot_eval
CELL=$1; GPU=$2; SUITE=$3
RES=$EX/results/${CELL}.txt; DONED=$EX/logs/done
W=$EX/logs/watch_${CELL}.log
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" >> "$W"; }
log "watcher 启动：等 $RES 达 100 Task 行"
while true; do
  n=$(grep -ac '^Task:' "$RES" 2>/dev/null || echo 0)
  [ "${n:-0}" -ge 100 ] && break
  if ! pgrep -f "eval_gr00t_final_r5.py .*--out $RES" >/dev/null 2>&1; then
    sleep 60
    n=$(grep -ac '^Task:' "$RES" 2>/dev/null || echo 0)
    if [ "${n:-0}" -ge 100 ]; then break; fi
    log "子进程已退出且 n=$n<100（异常终止），不自动发后续泳道，待人工处置"
    exit 1
  fi
  sleep 90
done
echo "$RES" > "$DONED/${CELL}.done"
log "n=100 封存 done；发余条件泳道 GPU$GPU suite=$SUITE"
cd "$EX" && setsid nohup bash /data/shaozhen.liu/code/tex3d/experiments/41_gr00t_eval_wave_r6.sh "$GPU" "$SUITE" >> "$EX/logs/master_lane2_${SUITE}.log" 2>&1 < /dev/null &
log "泳道已发（$!）"
