#!/bin/bash
# 桌布训练链 + 内存看门狗
# 用法（建议 setsid 启动以免 ssh 断开误杀）:
#   setsid bash cloth_guarded.sh [collect_train|collect|train] [GPU] [ITERS] &
# 行为:
#   collect_train: 帧池缓存缺失则先 --collect-only 落盘退出，再用 --load-pool 训练
#   train: 有缓存走 --load-pool（零环境、最小内存），无缓存进程内现场采集
# 看门狗: 可用内存连续 2 次 <8GB → TERM 整组训练进程，防止把机器压垮（load-460 事故的自动保险丝）
set -u
MODE="${1:-collect_train}"
GPU="${2:-0}"
ITERS="${3:-5000}"
DIR=/data/shaozhen.liu/code/tex3d/experiments/pi0_cloth
PYDIR=/data/shaozhen.liu/code/tex3d/pi
mkdir -p "$DIR"
RUNLOG="$DIR/guarded_run_$(date +%Y%m%d_%H%M%S).log"

export GUARD_MODE="$MODE" GUARD_GPU="$GPU" GUARD_ITERS="$ITERS" \
       GUARD_PY="$PYDIR" GUARD_DIR="$DIR"

bash -c '
  if [ "$GUARD_MODE" = collect ] || [ "$GUARD_MODE" = collect_train ]; then
    if [ -s "$GUARD_DIR/cloth_pool.pt" ]; then
      echo "[guard] 帧池缓存已存在，跳过采集"
    else
      echo "[guard] $(date +%F\ %T) 开始帧池采集（GPU$GUARD_GPU）"
      bash "$GUARD_PY/run_cloth.sh" "$GUARD_GPU" train --collect-only \
        --pool-file "$GUARD_DIR/cloth_pool.pt"
      rc=$?
      echo "[guard] $(date +%F\ %T) collect-only rc=$rc"
      [ $rc -ne 0 ] && exit $rc
    fi
  fi
  if [ "$GUARD_MODE" = train ] || [ "$GUARD_MODE" = collect_train ]; then
    if [ -s "$GUARD_DIR/cloth_pool.pt" ]; then
      bash "$GUARD_PY/run_cloth.sh" "$GUARD_GPU" train --load-pool \
        --iters "$GUARD_ITERS" --pool-file "$GUARD_DIR/cloth_pool.pt"
    else
      echo "[guard] 缺帧池缓存，回退进程内现场采集+训练"
      bash "$GUARD_PY/run_cloth.sh" "$GUARD_GPU" train --iters "$GUARD_ITERS"
    fi
    echo "[guard] $(date +%F\ %T) train rc=$?"
  fi
' >"$RUNLOG" 2>&1 &
CHAIN=$!
echo "[guard] 训练链 PID $CHAIN，日志 $RUNLOG"

strikes=0
while kill -0 "$CHAIN" 2>/dev/null; do
  avail=$(free -m | awk '/^Mem:/{print $7}')
  if [ "${avail:-999999}" -lt 8000 ]; then
    strikes=$((strikes+1))
    echo "[guard] $(date +%F\ %T) available=${avail}MB 偏低（$strikes/2）" >> "$RUNLOG"
    if [ "$strikes" -ge 2 ]; then
      echo "[guard] $(date +%F\ %T) 可用内存 <8GB，终止训练链保护机器" >> "$RUNLOG"
      pkill -TERM -P "$CHAIN" 2>/dev/null
      kill -TERM "$CHAIN" 2>/dev/null
      sleep 10
      pkill -KILL -P "$CHAIN" 2>/dev/null
      kill -KILL "$CHAIN" 2>/dev/null
      exit 42
    fi
  else
    strikes=0
  fi
  sleep 60
done
wait "$CHAIN" 2>/dev/null
echo "[guard] $(date +%F\ %T) 训练链结束，日志 $RUNLOG"
