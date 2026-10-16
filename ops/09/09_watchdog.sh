#!/bin/bash
# 外部看门狗 v4（09，每 5 分钟轮询，覆盖 p0_logs + p1_logs）
#   DONE 判定 v4 修复：块日志最后一行 "[eval] 结果" 所指 evaluation_result.txt 含
#   "overall robot success" 即已完成（overall 只写结果文件不进 stdout，v3 判据永远为假）。
#   楔死判定：日志 >35 分钟无新行且未完成且有存活评测进程 → 杀之（链内重试自动重启）。
EX=/data/shaozhen.liu/code/tex3d/experiments/pi0_cloth
WD=$EX/watchdog_events.log
STATE=/tmp/p0_watchdog_state
mkdir -p "$STATE"

event() {  # $1=类型 $2=详情（30 分钟去重）
  local sig now prev
  sig=$(printf '%s|%s' "$1" "$2" | md5sum | cut -c1-12)
  now=$(date +%s)
  prev=$(cat "$STATE/$sig" 2>/dev/null || echo 0)
  [ $((now - prev)) -lt 1800 ] && return
  echo "$now" > "$STATE/$sig"
  echo "$(date '+%F %T') [watchdog] $1: $2" >> "$WD"
}

block_done() {  # $1=块日志路径
  local res
  res=$(grep "\[eval\] 结果" "$1" 2>/dev/null | tail -1 | sed "s/.*结果: //" | tr -d "\r")
  [ -n "$res" ] && [ -f "$res" ] && grep -q "overall robot success" "$res" 2>/dev/null
}

while true; do
  if ! pgrep -f "09_p0_matrix.sh" > /dev/null && ! pgrep -f "09_p1_carrier.sh" > /dev/null; then
    grep -q "载体筛选全部完成 DONE-P1" "$WD" 2>/dev/null || \
      event CRITICAL "P0/P1 主链均死亡且未见 DONE-P1"
  fi

  for f in "$EX"/p0_logs/p0_*.log "$EX"/p1_logs/p1_*.log; do
    [ -f "$f" ] || continue
    block_done "$f" && continue
    eps=$(grep -c '^Task:' "$f")
    [ "${eps:-0}" -eq 0 ] && continue          # 排队/加载中，不判
    age=$(( $(date +%s) - $(stat -c %Y "$f") ))
    if [ $age -gt 2100 ]; then
      pids=$(ps -u shaozhen.liu -o pid=,cmd= | grep "attack_pi0_cloth.py eval" | grep -v grep | awk '{print $1}')
      if [ -n "$pids" ]; then
        echo "$(date '+%F %T') [watchdog] STALL-KILL: $(basename "$f") eps=$eps 日志 ${age}s 无新行，杀 $pids 触发链内重试" >> "$WD"
        kill $pids 2>/dev/null
      else
        event STALL_NOPROC "$(basename "$f") eps=$eps 停滞且无存活评测进程（链应已重试）"
      fi
    fi
    tail -30 "$f" 2>/dev/null | grep -qE "Traceback|CUDA (error|EHDM)|illegal memory" && \
      event ERROR_PATTERN "$(basename "$f") 日志尾错误栈（链内自动重试中）"
  done

  sleep 300
done
