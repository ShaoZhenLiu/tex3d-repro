#!/bin/bash
# jobq.sh — 动态 GPU 作业队列：作业行 "name|deps|cmd"，cmd 中 __GPU__ 会被替换为实时选中卡
# 用法: jobq.sh <jobs_file> <state_dir> <n_workers=3>
# 选卡规则: 0/1/3/2 优先级内选空闲显存最大者；准入 = 剩余 >= 22000MiB（单任务足迹 ~18G，天然支持单卡并行 2-3 任务）
set -u
JOBS=$1; STATE=$2; NW=${3:-3}
EX=/data/shaozhen.liu/code/tex3d/experiments/pi0_cloth
WD=$EX/watchdog_events.log
mkdir -p "$STATE"/claimed "$STATE"/done "$STATE"/failed

pick_gpu() {
  nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits | awk -F', ' '
    { u[$1]=$2; t[$1]=$3 }
    END {
      best=-1; bf=-1; split("0 1 3 2", o, " ")
      for (i=1; i<=4; i++) { g=o[i]; if (g in t) { f=t[g]-u[g]; if (f>=22000 && f>bf) { bf=f; best=g } } }
      print best
    }'
}

worker() {
  local wid=$1 name deps cmd picked PickedCmd ok rc attempt G total term d
  while true; do
    picked=""
    while IFS='|' read -r name deps cmd; do
      [ -z "$name" ] && continue
      case "$name" in \#*) continue ;; esac
      [ -e "$STATE/done/$name" ] && continue
      [ -e "$STATE/failed/$name" ] && continue
      [ -e "$STATE/claimed/$name" ] && continue
      ok=1
      IFS=',' read -ra DA <<< "$deps"
      for d in ${DA[@]+"${DA[@]}"}; do
        [ -n "$d" ] && [ ! -e "$STATE/done/$d" ] && { ok=0; break; }
      done
      [ "$ok" -eq 1 ] || continue
      mkdir "$STATE/claimed/$name" 2>/dev/null || continue
      picked=$name; PickedCmd=$cmd
      break
    done < "$JOBS"

    if [ -z "$picked" ]; then
      total=0; term=0
      while IFS='|' read -r name deps cmd; do
        [ -z "$name" ] && continue; case "$name" in \#*) continue ;; esac
        total=$((total+1))
        { [ -e "$STATE/done/$name" ] || [ -e "$STATE/failed/$name" ]; } && term=$((term+1))
      done < "$JOBS"
      [ "$term" -ge "$total" ] && break
      sleep 20; continue
    fi

    rc=1
    for attempt in 1 2 3; do
      G=$(pick_gpu)
      if [ "${G:- -1}" -lt 0 ]; then
        echo "$(date '+%F %T') [jobq] w$wid $picked 无可用 GPU（全卡 <22G 空闲），等 120s" >> "$WD"
        sleep 120; G=$(pick_gpu); [ "${G:- -1}" -lt 0 ] && continue
      fi
      echo "$(date '+%F %T') [jobq] w$wid $picked 第${attempt}次 GPU$G 启动" >> "$WD"
      bash -c "${PickedCmd//__GPU__/$G}"
      rc=$?
      [ "$rc" -eq 0 ] && break
      echo "$(date '+%F %T') [jobq] w$wid $picked 第${attempt}次失败 rc=$rc" >> "$WD"
      sleep 20
    done
    if [ "$rc" -eq 0 ]; then
      mv "$STATE/claimed/$picked" "$STATE/done/$picked"
      echo "$(date '+%F %T') [jobq] w$wid $picked DONE" >> "$WD"
    else
      mv "$STATE/claimed/$picked" "$STATE/failed/$picked"
      echo "$(date '+%F %T') [jobq] w$wid $picked FAILED（3 次）" >> "$WD"
    fi
  done
}

echo "$(date '+%F %T') [jobq] 启动 jobs=$JOBS workers=$NW" >> "$WD"
for w in $(seq 1 "$NW"); do worker "$w" & done
wait
echo "$(date '+%F %T') [jobq] 全部作业终态" >> "$WD"
