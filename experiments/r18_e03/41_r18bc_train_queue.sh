#!/bin/bash
# 41_r18bc_train_queue.sh — r18批次B/C 训练队列（E04×2 + E05×4 + E07×4，共10个训练）
#   入口 = attack_oft_cloth_ablate.py（循环=生产配方逐行同数学；sha 部署时登记）
#   成对共享流：先跑 fresh 臂（写 sampling_log + stream_full.pt 全精度UV），
#   另一臂 replay 该日志 —— 两臂帧/UV 流逐条相同，只差参考（mr/native）或 UV(on/off)。
#   E04：task0 子池（80帧，MR缓存按全局ID子集对齐），pregen 流 streams/e04_task0_seed7
#   E05：seed8/9 各一对（native fresh → mr replay），全池，UV on
#   E07：625双臂=重放 MR8 日志前625条（UV用文本6位小数，浮点位级不主张）；
#        bs1双臂=40000×1 fresh → replay，ckpt 步数后缀
#   GPU 纪律：与评测共享 flock 准入/claim（每卡≤2、need=20000、GPU4 永不使用）、重试×4
set -u
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
D=/data/shaozhen.liu
EX=$D/code/tex3d/experiments/r18_e03
LOGD=$EX/train_logs
CLAIMD=$D/code/tex3d/experiments/r18_e01e02/gpu_claims
LOCK=$D/code/tex3d/experiments/r18_e01e02/gpu_admit.lock
WD=$D/code/tex3d/experiments/pi0_cloth/watchdog_events.log
PY=$D/envs/tex3d-oft-mj330/bin/python
RUNNER=$D/code/tex3d/openvla-oft/experiments/robot/libero/attack_oft_cloth_ablate.py
OFTC=$D/code/tex3d/experiments/pi0_cloth/oft_cloth
MR8LOG=$OFTC/train_20261021_224922_tvfix-mr/sampling_log.txt
PREGEN=$EX/streams/e04_task0_seed7.txt
ENVV="MUJOCO_GL=egl PYOPENGL_PLATFORM=egl HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled PYTHONPATH=$D/code/tex3d:$D/code/tex3d/openvla-oft"
mkdir -p "$LOGD" "$CLAIMD"
log() { echo "$(date '+%F %T') [trainq] $*" | tee -a "$WD"; }

# 错峰发射门（IO风暴教训：多进程同时加载10-14GB模型会打满共享存储）：
# 全局 flock 串行，距上次同类发射≥$2秒 且 IO 红线内（bi≤500000 且 wa≤10）才放行
io_busy() {
  local s bi wa
  s=$(vmstat 1 2 2>/dev/null | tail -1)
  [ -z "$s" ] && return 1
  bi=$(echo "$s" | awk '{print $9}'); wa=$(echo "$s" | awk '{print $16}')
  [ -n "$bi" ] && [ -n "$wa" ] && { [ "$bi" -gt 500000 ] || [ "$wa" -gt 10 ]; }
}
gate_launch() {  # $1=train/eval $2=最小间隔秒
  local ts_f="$EX/.gate_$1.ts" lk_f="$EX/.gate_$1.lock" now last
  while :; do
    if (
      flock 9
      now=$(date +%s); last=$(cat "$ts_f" 2>/dev/null || echo 0)
      if [ $((now - last)) -ge $2 ] && ! io_busy; then echo "$now" > "$ts_f"; exit 0; fi
      exit 1
    ) 9>"$lk_f"; then return 0; fi
    sleep 30
  done
}

gpu_used_mi() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$1" 2>/dev/null | tr -d ' '; }
release_claim() { rm -f "$CLAIMD/$1".* 2>/dev/null; }
pick_gpu() {
  local name=$1 lane_pid=$2 tries=0 g used free cnt f pid ned sum_need out
  while [ $tries -lt 2880 ]; do
    out=$(
      (
        flock 9
        for g in 5 1 2 0; do
          sum_need=0; cnt=0
          for f in "$CLAIMD"/*."$g"; do
            [ -e "$f" ] || continue
            read -r pid ned < "$f" 2>/dev/null || { rm -f "$f"; continue; }
            if [ -z "$pid" ] || ! kill -0 "$pid" 2>/dev/null; then rm -f "$f"; continue; fi
            sum_need=$((sum_need + ${ned:-26000})); cnt=$((cnt + 1))
          done
          [ "$cnt" -ge 2 ] && continue
          used=$(gpu_used_mi "$g")
          [ -z "$used" ] && continue
          free=$((49140 - used))
          if [ $((sum_need + 26000)) -le "$free" ]; then
            echo "$lane_pid 26000" > "$CLAIMD/${name}.$g"
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
# 训练完成判据：gradient_log 末次 [ckpt-final] itXXXXX（$2=末iter）+ 规范名 PNG 存在
job_done() {
  # 扫描全部同名目录（防未收口新目录遮蔽已完成旧目录导致无限重跑）
  local tag=$1 last_it=$2 d
  for d in $(ls -dt $OFTC/train_*_"$tag" 2>/dev/null); do
    if [ -f "$d/Universal_Cloth_OV.png" ] && \
       grep -q "\[ckpt-final\] it${last_it}" "$d/gradient_log.txt" 2>/dev/null; then
      return 0
    fi
  done
  return 1
}
find_dir() {
  # 优先返回已收口目录（stream 依赖用），否则最新目录
  local d
  for d in $(ls -dt $OFTC/train_*_"$1" 2>/dev/null); do
    if [ -f "$d/Universal_Cloth_OV.png" ] && \
       grep -q "ckpt-final" "$d/gradient_log.txt" 2>/dev/null; then
      echo "$d"; return 0
    fi
  done
  ls -dt $OFTC/train_*_"$1" 2>/dev/null | head -1
}

wait_dep() {  # $1=依赖标签 $2=末iter：等其训练终点标记（依赖臂先收口）
  while ! job_done "$1" "$2"; do
    sleep 300
  done
}
run_job() {  # $1=name $2=末iter 其后为 runner 参数
  local name=$1 last_it=$2; shift 2
  local g mypid=$BASHPID attempt d
  # 防双跑门：同标签作业已在跑（旧实例/孤儿）则等待其退出，再按 job_done 判定
  while ps -u shaozhen.liu -o cmd | grep -- "--tag ${name}[- ]" >/dev/null 2>&1; do
    sleep 60
  done
  if job_done "$name" "$last_it"; then log "训练 $name 已完成，跳过"; return 0; fi
  for attempt in 1 2 3 4 5 6 7 8; do
    # 防双跑门（循环内复查）：同标签作业已在跑则回等待（手动发射/孤儿竞态窗口）
    while ps -u shaozhen.liu -o cmd | grep -- "--tag ${name}[- ]" >/dev/null 2>&1; do
      sleep 60
    done
    if job_done "$name" "$last_it"; then log "训练 $name 已完成（循环内复查），跳过"; return 0; fi
    release_claim "$name"
    g=$(pick_gpu "$name" "$mypid") || { log "训练 $name 等待 GPU 超时 FAILED"; return 1; }
    # 断点续训：旧目录有步数 checkpoint 且未收口 → 自动带 --resume-dir（≤625步损失）
    local od rargs=()
    od=$(find_dir "$name")
    if [ -n "$od" ] && ls "$od"/Ep0_Cloth_Delta_OV_it*.pt >/dev/null 2>&1 \
       && ! grep -q "ckpt-final" "$od/gradient_log.txt" 2>/dev/null; then
      rargs=(--resume-dir "$od")
      log "训练 $name 尝试 $attempt GPU$g（续训自 $od）"
    else
      log "训练 $name 尝试 $attempt GPU$g"
    fi
    gate_launch train 600
    ( cd $D/code/tex3d-ws/g7 && env CUDA_VISIBLE_DEVICES=$g MUJOCO_EGL_DEVICE_ID=$g \
      $ENVV $PY $RUNNER train --tag "$name" "$@" ${rargs[@]+"${rargs[@]}"} ) \
      >> "$LOGD/${name}.log" 2>&1
    if job_done "$name" "$last_it"; then
      release_claim "$name"
      d=$(find_dir "$name")
      log "训练 $name 完成 -> $d sha=$(sha256sum "$d/Universal_Cloth_OV.png" | cut -c1-16)"
      return 0
    fi
    sleep 120
  done
  release_claim "$name"
  log "训练 $name 四次失败 FAILED"; return 1
}

pids=()
# L1：E04-a（UV-on） → E07-bs1 native → E07-bs1 mr（重放 bs1-native 流）
( sleep 0
  run_job e04a 04999 --ref mr --stream replay --stream-file "$PREGEN" --uv on \
    --filter-task 0 --iters 5000 --batch 8 --seed 7 --ckpt-step-suffix
  run_job e07_bs1_native 39999 --ref native --stream fresh --uv on \
    --iters 40000 --batch 1 --seed 7 --checkpoint-every 5000 --ckpt-step-suffix
  wait_dep e07_bs1_native 39999
  run_job e07_bs1_mr 39999 --ref mr --stream fresh --uv on \
    --iters 40000 --batch 1 --seed 7 --checkpoint-every 5000 --ckpt-step-suffix
) & pids+=($!)
# L2：E04-b（UV-off，同流） → E07-625 双臂
( sleep 60
  run_job e04b 04999 --ref mr --stream replay --stream-file "$PREGEN" --uv off \
    --filter-task 0 --iters 5000 --batch 8 --seed 7 --ckpt-step-suffix
  run_job e07_625_native 00624 --ref native --stream replay --stream-file "$MR8LOG" \
    --uv on --iters 625 --batch 8 --seed 7 --checkpoint-every 625 --ckpt-step-suffix
  run_job e07_625_mr 00624 --ref mr --stream replay --stream-file "$MR8LOG" \
    --uv on --iters 625 --batch 8 --seed 7 --checkpoint-every 625 --ckpt-step-suffix
) & pids+=($!)
# L3：E05 seed9 对 —— native 臂已由 train_20261027_145929_e05_s9_native 完成
# （[ckpt-final] it04999），此处移除其 run_job 防止任何形式的重复执行；直接等依赖→s9_mr
( sleep 120
  while ps -u shaozhen.liu -o cmd | grep "tag e05_s9_nativ[e]" >/dev/null 2>&1; do
    sleep 60
  done
  wait_dep e05_s9_native 04999
  run_job e05_s9_mr 04999 --ref mr --stream fresh --uv on \
    --iters 5000 --batch 8 --seed 9 --ckpt-step-suffix
) & pids+=($!)
# L4：E05 seed8 对
( sleep 180
  run_job e05_s8_native 04999 --ref native --stream fresh --uv on \
    --iters 5000 --batch 8 --seed 8 --ckpt-step-suffix
  wait_dep e05_s8_native 04999
  run_job e05_s8_mr 04999 --ref mr --stream fresh --uv on \
    --iters 5000 --batch 8 --seed 8 --ckpt-step-suffix
) & pids+=($!)

fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
ok=0
for t in e04a e04b e05_s8_native e05_s8_mr e05_s9_native e05_s9_mr e07_625_native e07_625_mr e07_bs1_native e07_bs1_mr; do
  case $t in
    e07_625_*) li=00624;; e07_bs1_*) li=39999;; *) li=04999;;
  esac
  job_done "$t" "$li" && ok=$((ok + 1))
done
if [ "$fail" = 0 ] && [ "$ok" = 10 ]; then
  echo "DONE-R18BC-TRAIN 10/10 $(date)" >> "$LOGD/summary.txt"; log "DONE-R18BC-TRAIN 10/10"
else
  echo "R18BC-TRAIN-WITH-FAILURES ok=$ok/10 $(date)" >> "$LOGD/summary.txt"; log "R18BC-TRAIN-WITH-FAILURES ok=$ok/10"
fi
