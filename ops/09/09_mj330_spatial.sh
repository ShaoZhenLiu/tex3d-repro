#!/bin/bash
# MJ330: full spatial original baseline under mujoco 3.3.0 (clone venv)
#   vs ND10's 58% under 3.8.1 — same weights, same init states, 10-step denoising
#   2-GPU split: GPU0 t0-4 / GPU1 t5-9. Tree unpatched; restored at end unless FS130 running.
D=/data/shaozhen.liu
PI=$D/code/tex3d/pi
EX=$D/code/tex3d/experiments/pi0_cloth
LOGD=$EX/mj330_logs
WD=$EX/watchdog_events.log
mkdir -p "$LOGD"
echo "$(date '+%F %T') [mj330] chain start master=$$ (spatial original, mujoco 3.3.0)" >> "$WD"

wait_gpu_free() {
  while true; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$1" | tr -d ' ')
    [ "$used" -lt 2000 ] && break
    echo "$(date '+%F %T') [mj330] GPU$1 busy (${used}MiB), waiting" >> "$WD"
    sleep 300
  done
}

run_split() {  # $1=GPU $2=tasks
  local g=$1 tasks=$2
  local name="mj330_orig_sp_t${2//,/_}"
  local log="$LOGD/${name}.log"
  local attempt res
  for attempt in 1 2 3; do
    echo "$(date '+%F %T') [mj330] $name attempt $attempt start GPU$g" >> "$WD"
    wait_gpu_free "$g"
    bash "$PI/run_cloth330.sh" "$g" eval --condition original --suite libero_spatial \
      --tasks "$tasks" --trials 10 --num-denoise 10 >> "$log" 2>&1
    res=$(grep -a "\[eval\] 结果" "$log" | tail -1 | sed "s/.*结果: //" | tr -d "\r")
    if [ -n "$res" ] && [ -f "$res" ] && grep -q "overall robot success" "$res"; then
      echo "$(date '+%F %T') [mj330] $name attempt $attempt OK -> $res" >> "$WD"
      return 0
    fi
    echo "$(date '+%F %T') [mj330] $name attempt $attempt incomplete, retry in 30s" >> "$WD"
    sleep 30
  done
  echo "$(date '+%F %T') [mj330] $name FAILED after 3 attempts" >> "$WD"
  return 1
}

cd "$PI" && bash run_cloth.sh 0 patch --off >> "$WD" 2>&1
run_split 0 0,1,2,3,4 & p1=$!
run_split 1 5,6,7,8,9 & p2=$!
rc=0
wait $p1 || rc=1; wait $p2 || rc=1

# race-safe restore: leave tree unpatched if FS130 evals already started (they restore at end)
if grep -q "fs130.*eval chain start" "$WD"; then
  echo "$(date '+%F %T') [mj330] FS130 active, leaving tree unpatched to it" >> "$WD"
else
  cd "$PI" && bash run_cloth.sh 0 patch --on >> "$WD" 2>&1
fi
if [ "$rc" -eq 0 ]; then
  echo "$(date '+%F %T') [mj330] DONE-MJ330 all splits complete" >> "$WD"
else
  echo "$(date '+%F %T') [mj330] DONE-MJ330-WITH-FAILURES rc=$rc" >> "$WD"
fi
