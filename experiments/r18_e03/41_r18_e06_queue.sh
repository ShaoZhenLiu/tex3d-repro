#!/bin/bash
# 41_r18_e06_queue.sh — r18 E06：π0 源双臂训练队列（R-N fresh → R-M replay 同流）
#   入口 = attack_pi0_cloth_e06.py（五修复已冒烟：真TV/批8平均/采集10步denoise/
#   无qseed/主视图披露；check 断言全部通过）
#   启动前置：e06_pool.pt 存在（80 帧已采集登记）。
#   顺序按用户指令：E04/E05/E07 之后启动本队列。
set -u
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
D=/data/shaozhen.liu
EX=$D/code/tex3d/experiments/r18_e03
LOGD=$EX/e06_train_logs
CLAIMD=$D/code/tex3d/experiments/r18_e01e02/gpu_claims
LOCK=$D/code/tex3d/experiments/r18_e01e02/gpu_admit.lock
WD=$D/code/tex3d/experiments/pi0_cloth/watchdog_events.log
PI=$D/code/tex3d/pi
POOL=$D/code/tex3d/experiments/pi0_cloth/e06_pi0src/e06_pool.pt
E06ROOT=$D/code/tex3d/experiments/pi0_cloth/e06_pi0src
mkdir -p "$LOGD" "$CLAIMD"
log() { echo "$(date '+%F %T') [e06q] $*" | tee -a "$WD"; }

gpu_used_mi() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$1" 2>/dev/null | tr -d ' '; }
release_claim() { rm -f "$CLAIMD/$1".* 2>/dev/null; }
pick_gpu() {
  local name=$1 lane_pid=$2 need=$3 tries=0 g used free cnt f pid ned sum_need out
  while [ $tries -lt 2880 ]; do
    out=$(
      (
        flock 9
        for g in 6 5 1 2 0; do
          sum_need=0; cnt=0
          for f in "$CLAIMD"/*."$g"; do
            [ -e "$f" ] || continue
            read -r pid ned < "$f" 2>/dev/null || { rm -f "$f"; continue; }
            if [ -z "$pid" ] || ! kill -0 "$pid" 2>/dev/null; then rm -f "$f"; continue; fi
            sum_need=$((sum_need + ${ned:-23000})); cnt=$((cnt + 1))
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
job_done() {
  local tag=$1 d
  d=$(ls -dt $E06ROOT/train_*_"$tag" 2>/dev/null | head -1)
  [ -n "$d" ] && [ -f "$d/Universal_Cloth.png" ] && \
    grep -q "\[ckpt-final\] it04999" "$d/gradient_log.txt" 2>/dev/null
}
find_dir() { ls -dt $E06ROOT/train_*_"$1" 2>/dev/null | head -1; }

run_job() {  # $1=name $2=need 其余为 runner 参数
  local name=$1 need=$2; shift 2
  local g mypid=$BASHPID attempt d
  if job_done "$name"; then log "训练 $name 已完成，跳过"; return 0; fi
  for attempt in 1 2 3 4 5 6 7 8; do
    release_claim "$name"
    g=$(pick_gpu "$name" "$mypid" "$need") || { log "训练 $name 等待 GPU 超时 FAILED"; return 1; }
    log "训练 $name 尝试 $attempt GPU$g"
    d=$(ls -dt $E06ROOT/train_*_"$name" 2>/dev/null | head -1)
    ( cd $PI && env CUDA_VISIBLE_DEVICES=$g MUJOCO_EGL_DEVICE_ID=$g MUJOCO_GL=egl \
        PYOPENGL_PLATFORM=egl OPENPI_DATA_HOME=$D/.cache/openpi \
        MPLCONFIGDIR=$D/tmp/attack_pi_mpl HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
        WANDB_MODE=disabled PYTHONUNBUFFERED=1 \
        PYTHONPATH=$D/code/tex3d-deps/taming-transformers \
        TORCH_EXTENSIONS_DIR=/data6/shaozhen.liu/tex3d-mig/torch_extensions \
        $D/envs/tex3d-pi0-mj330/bin/python $PI/attack_pi0_cloth_e06.py train \
        --pool "$POOL" "$@" ) >> "$LOGD/${name}.log" 2>&1
    if job_done "$name"; then
      release_claim "$name"
      d=$(find_dir "$name")
      log "训练 $name 完成 -> $d sha=$(sha256sum "$d/Universal_Cloth.png" | cut -c1-16)"
      return 0
    fi
    sleep 120
  done
  release_claim "$name"
  log "训练 $name 八次失败 FAILED"; return 1
}

pids=()
( run_job e06_rn 23000 --tag e06_rn --ref native --stream fresh --uv on \
    --iters 5000 --batch 8 --seed 7 --ckpt-step-suffix
  run_job e06_rm 23000 --tag e06_rm --ref matched --stream replay \
    --stream-file "$(find_dir e06_rn)/sampling_log.txt" --uv on \
    --iters 5000 --batch 8 --seed 7 --ckpt-step-suffix ) & pids+=($!)

fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
if [ "$fail" = 0 ] && job_done e06_rn && job_done e06_rm; then
  echo "DONE-E06-TRAIN 2/2 $(date)" >> "$LOGD/summary.txt"; log "DONE-E06-TRAIN 2/2"
else
  echo "E06-TRAIN-WITH-FAILURES $(date)" >> "$LOGD/summary.txt"; log "E06-TRAIN-WITH-FAILURES"
fi
