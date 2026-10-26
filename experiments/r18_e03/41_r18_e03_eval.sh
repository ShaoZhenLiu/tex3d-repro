#!/bin/bash
# 41_r18_e03_eval.sh — 第18轮 E03：UV-off 纹理八格目标评测（800 回合，零训练）
#   八格 = {π0,π0.5} × {spatial,object,goal,10}；每格 100 回合（tasks 0-9 显式 ×
#   trials 10、初态 10-19、num_denoise 10、max_steps 400、MuJoCo3.3.0）
#   纹理 = 最新 train_*_e03-uvoff/Universal_Cloth_OV.png（启动时发现 + SHA 登记，
#   不使用中间 checkpoint；固定终点）。
#   GPU 纪律：与批次A共用 flock 准入/claim 目录（全局每卡≤2 我方进程）、
#   VRAM 守卫 need≈17000MiB（π 实测稳态≈15-16.4G）、60s 错峰、重试×12、GPU4 永不使用。
set -u
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
D=/data/shaozhen.liu
PI=$D/code/tex3d/pi
EX=$D/code/tex3d/experiments/r18_e03
LOGD=$EX/e03_logs
CLAIMD=$D/code/tex3d/experiments/r18_e01e02/gpu_claims
LOCK=$D/code/tex3d/experiments/r18_e01e02/gpu_admit.lock
WD=$D/code/tex3d/experiments/pi0_cloth/watchdog_events.log
OFTC=$D/code/tex3d/experiments/pi0_cloth/oft_cloth
CKPT0=$D/model/RLinf-Pi0-LIBERO-130-fullshot-SFT
CKPT5=$D/model/RLinf-Pi05-LIBERO-130-fullshot-SFT
mkdir -p "$LOGD" "$CLAIMD"
VDIR=$(ls -dt $OFTC/train_*_e03-uvoff 2>/dev/null | grep -v smoke | head -1)
TEX="$VDIR/Universal_Cloth_OV.png"
[ -f "$TEX" ] || { echo "$(date '+%F %T') [e03] 纹理缺失 $TEX" >> "$WD"; exit 1; }
grep -q "it04999" "$VDIR/gradient_log.txt" 2>/dev/null || \
  { echo "$(date '+%F %T') [e03] 终点不完整(无 it04999) $VDIR" >> "$WD"; exit 1; }
echo "$(date '+%F %T') [e03] 冻结纹理 $TEX sha=$(sha256sum "$TEX" | cut -c1-16)" >> "$WD"
log() { echo "$(date '+%F %T') [e03] $*" | tee -a "$WD"; }

gpu_used_mi() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$1" 2>/dev/null | tr -d ' '; }
release_claim() { rm -f "$CLAIMD/$1".* 2>/dev/null; }
pick_gpu() {
  # 准入：Σ(已有claim需求)+自身需求 ≤ 当前空闲，且 claim 数<2；claim 记泳道壳 pid
  local name=$1 need=$2 lane_pid=$3 tries=0 g used free cnt f pid ned sum_need out
  while [ $tries -lt 2160 ]; do
    out=$(
      (
        flock 9
        for g in 5 7 6 2 0 1 3; do
          sum_need=0; cnt=0
          for f in "$CLAIMD"/*."$g"; do
            [ -e "$f" ] || continue
            read -r pid ned < "$f" 2>/dev/null || { rm -f "$f"; continue; }
            if [ -z "$pid" ] || ! kill -0 "$pid" 2>/dev/null; then rm -f "$f"; continue; fi
            sum_need=$((sum_need + ${ned:-17000})); cnt=$((cnt + 1))
          done
          [ "$cnt" -ge 2 ] && continue
          used=$(gpu_used_mi "$g")
          [ -z "$used" ] && continue
          free=$((49140 - used))
          if [ $((sum_need + need)) -le "$free" ]; then
            echo "$lane_pid $need" > "$CLAIMD/${name}.$g"
            echo "$g"
            exit 0
          fi
        done
        exit 1
      ) 9>"$LOCK"
    ) && { echo "$out"; return 0; }
    sleep 15; tries=$((tries + 1))
  done
  return 1
}
cell_done() {
  local res n
  res=$(grep -a '\[eval\] 结果' "$LOGD/$1.log" 2>/dev/null | tail -1 | sed 's/.*结果: //' | tr -d '\r')
  n=$(grep -ac '^Task:' "$res" 2>/dev/null)
  [ -n "$res" ] && [ -f "$res" ] && [ "${n:-0}" -ge 100 ]
}
run_cell() {  # $1=name $2=suite $3=ckpt $4=modelcfg(可空)
  local name=$1 suite=$2 ckpt=$3 mcfg=${4:-} res n g extra=() mypid=$BASHPID
  if cell_done "$name"; then log "评测 $name 已完成，跳过"; return 0; fi
  [ -n "$mcfg" ] && extra=(--model-config "$mcfg")
  local log_f="$LOGD/${name}.log"
  for attempt in 1 2 3 4 5 6 7 8 9 10 11 12; do
    release_claim "$name"
    g=$(pick_gpu "$name" 17000 "$mypid") || { log "评测 $name 等待 GPU 空闲超时 FAILED"; return 1; }
    log "评测 $name 尝试 $attempt GPU$g"
    bash "$PI/run_cloth330.sh" "$g" eval --condition adv --suite "$suite" \
      --tasks 0,1,2,3,4,5,6,7,8,9 --trials 10 --num-denoise 10 --init-offset 10 \
      --max-steps 400 --texture "$TEX" --ckpt "$ckpt" \
      ${extra[@]+"${extra[@]}"} >> "$log_f" 2>&1
    res=$(grep -a '\[eval\] 结果' "$log_f" 2>/dev/null | tail -1 | sed 's/.*结果: //' | tr -d '\r')
    n=$(grep -ac '^Task:' "$res" 2>/dev/null)
    if [ -n "$res" ] && [ -f "$res" ] && [ "${n:-0}" -ge 100 ]; then
      release_claim "$name"
      log "评测 $name 完成 -> $(grep -a 'overall robot success' "$res" | tail -1)"
      return 0
    fi
    sleep 120
  done
  release_claim "$name"
  log "评测 $name 十二次失败 FAILED"; return 1
}

pids=()
( sleep 0;   run_cell e03_uv0_pi0_spatial  libero_spatial "$CKPT0" ) & pids+=($!)
( sleep 60;  run_cell e03_uv0_pi0_object   libero_object  "$CKPT0" ) & pids+=($!)
( sleep 120; run_cell e03_uv0_pi0_goal     libero_goal    "$CKPT0" ) & pids+=($!)
( sleep 180; run_cell e03_uv0_pi0_long     libero_10      "$CKPT0" ) & pids+=($!)
( sleep 240; run_cell e03_uv0_pi05_spatial libero_spatial "$CKPT5" pi05_libero ) & pids+=($!)
( sleep 300; run_cell e03_uv0_pi05_object  libero_object  "$CKPT5" pi05_libero ) & pids+=($!)
( sleep 360; run_cell e03_uv0_pi05_goal    libero_goal    "$CKPT5" pi05_libero ) & pids+=($!)
( sleep 420; run_cell e03_uv0_pi05_long    libero_10      "$CKPT5" pi05_libero ) & pids+=($!)
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
ok=0; sum=""
for name in e03_uv0_pi0_spatial e03_uv0_pi0_object e03_uv0_pi0_goal e03_uv0_pi0_long e03_uv0_pi05_spatial e03_uv0_pi05_object e03_uv0_pi05_goal e03_uv0_pi05_long; do
  cell_done "$name" && ok=$((ok + 1))
  res=$(grep -a '\[eval\] 结果' "$LOGD/$name.log" 2>/dev/null | tail -1 | sed 's/.*结果: //' | tr -d '\r')
  sum="$sum$name=$(grep -a 'overall robot success' "$res" 2>/dev/null | tail -1 | grep -o '[0-9]*/100') "
done
if [ "$fail" = 0 ] && [ "$ok" = 8 ]; then
  echo "DONE-E03-EVAL 8/8 $(date)：$sum" >> "$LOGD/summary.txt"; log "DONE-E03-EVAL 8/8：$sum"
else
  echo "E03-EVAL-WITH-FAILURES ok=$ok/8 $(date)：$sum" >> "$LOGD/summary.txt"; log "E03-EVAL-WITH-FAILURES ok=$ok/8：$sum"
fi
