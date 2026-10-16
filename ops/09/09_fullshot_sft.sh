#!/bin/bash
# FS130: RLinf-Pi0-LIBERO-130-fullshot-SFT baseline verification
#   Phase DL: download via hf-mirror.com (resume+retry, skip optimizer.pt), sha256 verify
#   Phase E1: spatial original, 10-step (GPU0) vs 1-step (GPU1), 10 tasks x 10 eps each
#   Phase E2: object + goal original 10-step (GPU0/GPU1)
#   Phase E3: libero_10 original 10-step (GPU0)
#   Tree unpatched for original; patched state restored at end. GPU2/3 = other users, never use.
D=/data/shaozhen.liu
CKPT=$D/model/RLinf-Pi0-LIBERO-130-fullshot-SFT
PI=$D/code/tex3d/pi
EX=$D/code/tex3d/experiments/pi0_cloth
LOGD=$EX/fs130_logs
WD=$EX/watchdog_events.log
EXPECT=28f1e20c0a81a091084b76ab8cdac76869408a4569b1ba347dbaca5c34688313
B=https://hf-mirror.com/RLinf/RLinf-Pi0-LIBERO-130-fullshot-SFT/resolve/main
mkdir -p "$LOGD" "$CKPT/physical-intelligence/libero"
echo "$(date '+%F %T') [fs130] chain start master=$$" >> "$WD"

wait_gpu_free() {
  while true; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$1" | tr -d ' ')
    [ "$used" -lt 2000 ] && break
    echo "$(date '+%F %T') [fs130] GPU$1 busy (${used}MiB), waiting" >> "$WD"
    sleep 300
  done
}

run_block() {  # $1=GPU $2=suite $3=num_denoise
  local g=$1 suite=$2 nd=$3
  local name="fs130_orig_${suite}_nd${nd}"
  local log="$LOGD/${name}.log"
  local attempt res
  for attempt in 1 2 3; do
    echo "$(date '+%F %T') [fs130] $name attempt $attempt start GPU$g" >> "$WD"
    wait_gpu_free "$g"
    bash "$PI/run_cloth.sh" "$g" eval --condition original --suite "$suite" \
      --tasks 0,1,2,3,4,5,6,7,8,9 --trials 10 --num-denoise "$nd" \
      --ckpt "$CKPT" >> "$log" 2>&1
    res=$(grep -a "\[eval\] 结果" "$log" | tail -1 | sed "s/.*结果: //" | tr -d "\r")
    if [ -n "$res" ] && [ -f "$res" ] && grep -q "overall robot success" "$res"; then
      echo "$(date '+%F %T') [fs130] $name attempt $attempt OK -> $res" >> "$WD"
      return 0
    fi
    echo "$(date '+%F %T') [fs130] $name attempt $attempt incomplete, retry in 30s" >> "$WD"
    sleep 30
  done
  echo "$(date '+%F %T') [fs130] $name FAILED after 3 attempts" >> "$WD"
  return 1
}

# ---------- Phase DL ----------
for f in metadata.pt physical-intelligence/libero/norm_stats.json; do
  curl -sL --retry 3 -o "$CKPT/$f" "$B/$f" \
    && echo "$(date '+%F %T') [fs130] dl ok $f" >> "$WD" \
    || { echo "$(date '+%F %T') [fs130] dl FAIL $f" >> "$WD"; }
done
for i in 1 2 3 4 5 6 7 8; do
  curl -sL --retry 3 -C - -o "$CKPT/model.safetensors" "$B/model.safetensors" \
    && { echo "$(date '+%F %T') [fs130] dl ok model.safetensors (attempt $i)" >> "$WD"; break; }
  echo "$(date '+%F %T') [fs130] model.safetensors attempt $i interrupted, resume in 15s" >> "$WD"
  sleep 15
done
GOT=$(sha256sum "$CKPT/model.safetensors" 2>/dev/null | awk '{print $1}')
if [ "$GOT" != "$EXPECT" ]; then
  echo "$(date '+%F %T') [fs130] FS-FAILED-HASH got=$GOT" >> "$WD"
  exit 1
fi
mkdir -p "$CKPT/assets/physical-intelligence/libero"
cp "$CKPT/physical-intelligence/libero/norm_stats.json" "$CKPT/assets/physical-intelligence/libero/norm_stats.json"
echo "$(date '+%F %T') [fs130] hash OK, assets laid out" >> "$WD"

# ---------- Evals (tree unpatched) ----------
cd "$PI" && bash run_cloth.sh 0 patch --off >> "$WD" 2>&1
rc=0
run_block 0 libero_spatial 10 & p1=$!
run_block 1 libero_spatial 1  & p2=$!
wait $p1 || rc=1; wait $p2 || rc=1
run_block 0 libero_object 10 & p1=$!
run_block 1 libero_goal 10   & p2=$!
wait $p1 || rc=1; wait $p2 || rc=1
run_block 0 libero_10 10 || rc=1
cd "$PI" && bash run_cloth.sh 0 patch --on >> "$WD" 2>&1
if [ "$rc" -eq 0 ]; then
  echo "$(date '+%F %T') [fs130] DONE-FS130 all evals complete" >> "$WD"
else
  echo "$(date '+%F %T') [fs130] DONE-FS130-WITH-FAILURES rc=$rc" >> "$WD"
fi
