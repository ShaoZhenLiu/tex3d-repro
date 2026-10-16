#!/bin/bash
# ND10: original condition with 10-step denoising (vs historical 1-step default)
#   libero_spatial, tasks 0-9 x 10 trials (=100 eps, same init states as P0 records)
#   3-GPU split: GPU0 t0-3 / GPU1 t4-6 / GPU3 t7-9 (GPU2 = other user, never use)
#   Tree must be UNPATCHED (original condition). Restores patched state at end.
# Completion check: result dir newer than attempt start containing "overall robot success".
D=/data/shaozhen.liu
PI=$D/code/tex3d/pi
EX=$D/code/tex3d/experiments/pi0_cloth
LOGD=$EX/nd10_logs
WD=$EX/watchdog_events.log
mkdir -p "$LOGD"
echo "$(date '+%F %T') [nd10] chain start master=$$ (num_denoise=10, original, spatial)" >> "$WD"

wait_gpu_free() {
  while true; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$1" | tr -d ' ')
    [ "$used" -lt 2000 ] && break
    echo "$(date '+%F %T') [nd10] GPU$1 busy (${used}MiB), waiting" >> "$WD"
    sleep 300
  done
}

run_split() {  # $1=GPU $2=tasks(comma list)
  local g=$1 tasks=$2
  local name="nd10_orig_sp_t${2//,/_}"
  local log="$LOGD/${name}.log"
  local attempt t0 res
  for attempt in 1 2 3; do
    t0=$(date +%s)
    echo "$(date '+%F %T') [nd10] $name attempt $attempt start GPU$g" >> "$WD"
    wait_gpu_free "$g"
    bash "$PI/run_cloth.sh" "$g" eval --condition original --suite libero_spatial \
      --tasks "$tasks" --trials 10 --num-denoise 10 >> "$log" 2>&1
    res=$(grep -a "\[eval\] 结果" "$log" | tail -1 | sed "s/.*结果: //" | tr -d "\r")
    if [ -n "$res" ] && [ -f "$res" ] && grep -q "overall robot success" "$res"; then
      echo "$(date '+%F %T') [nd10] $name attempt $attempt OK -> $res" >> "$WD"
      return 0
    fi
    echo "$(date '+%F %T') [nd10] $name attempt $attempt incomplete, retry in 30s" >> "$WD"
    sleep 30
  done
  echo "$(date '+%F %T') [nd10] $name FAILED after 3 attempts" >> "$WD"
  return 1
}

cd "$PI" && bash run_cloth.sh 0 patch --off >> "$WD" 2>&1
run_split 0 0,1,2,3 & p1=$!
run_split 1 4,5,6 &     p2=$!
run_split 3 7,8,9 &     p3=$!
rc=0
wait $p1 || rc=1
wait $p2 || rc=1
wait $p3 || rc=1
cd "$PI" && bash run_cloth.sh 0 patch --on >> "$WD" 2>&1
if [ "$rc" -eq 0 ]; then
  echo "$(date '+%F %T') [nd10] DONE-ND10 all splits complete" >> "$WD"
else
  echo "$(date '+%F %T') [nd10] DONE-ND10-WITH-FAILURES rc=$rc (see nd10 logs)" >> "$WD"
fi
