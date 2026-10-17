#!/bin/bash
# FS-ATK: fullshot-SFT 骨干攻击链（09，GPU0/1，mujoco 3.3.0 全程）
#   1) 切 30x20_solid 健康载体 + patch
#   2) 采集共享池：spatial t0,2,6,8，满轨迹 stride10 cap40 TAAO，摆放抖动，fullshot 权重（GPU1 先起）
#   3) 双臂同池训练：A=fsA_eot --uv-eot（GPU0）/ B=fsB_noeot（GPU1），ε64 seed7 5000iter
#   4) 快评：未见任务 {1,3,4,5,7,9} plain + advA + advB（10 集）
#   5) 矩阵：四套件 × {plain, advA} × 100 集（original 复用 §十五 fullshot 行）
#   6) 恢复 60x40 grid 默认资产
D=/data/shaozhen.liu
PI=$D/code/tex3d/pi
EX=$D/code/tex3d/experiments/pi0_cloth
LIVE=$D/code/tex3d-deps/libero-pi/libero/libero/assets/tablecloth
STAGE=$D/code/tex3d-deps/cloth_variants
CKPT=$D/model/RLinf-Pi0-LIBERO-130-fullshot-SFT
POOL=$EX/cloth_pool_fs130.pt
LOGD=$EX/fsatk_logs
WD=$EX/watchdog_events.log
CARRIER=30x20_solid
EPS=0.2510
UT=1,3,4,5,7,9
mkdir -p "$LOGD"
echo "$(date '+%F %T') [fsatk] 链启动 master=$$（fullshot 骨干，mujoco 3.3.0，GPU0/1）" >> "$WD"

wait_gpu_free() {
  while true; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$1" | tr -d ' ')
    [ "$used" -lt 60000 ] && break
    echo "$(date '+%F %T') [fsatk] GPU$1 被占(${used}MiB)，等待" >> "$WD"
    sleep 300
  done
}

run_block() {  # $1=GPU $2=name 其余为 run_cloth330 eval 参数（--condition 起）
  local g=$1 name=$2; shift 2
  local log="$LOGD/${name}.log"
  local attempt res
  for attempt in 1 2 3; do
    echo "$(date '+%F %T') [fsatk] $name 第${attempt}次 GPU$g" >> "$WD"
    wait_gpu_free "$g"
    bash "$PI/run_cloth330.sh" "$g" eval --ckpt "$CKPT" "$@" >> "$log" 2>&1
    res=$(grep -a "\[eval\] 结果" "$log" | tail -1 | sed "s/.*结果: //" | tr -d "\r")
    if [ -n "$res" ] && [ -f "$res" ] && grep -q "overall robot success" "$res"; then
      echo "$(date '+%F %T') [fsatk] $name OK -> $res" >> "$WD"
      return 0
    fi
    echo "$(date '+%F %T') [fsatk] $name 第${attempt}次未完成，30s 后重试" >> "$WD"
    sleep 30
  done
  echo "$(date '+%F %T') [fsatk] $name FAILED（3 次）" >> "$WD"
  return 1
}

# ---- 1) 载体切换 + 补丁 ----
cp "$STAGE/$CARRIER/tablecloth.obj" "$LIVE/tablecloth.obj"
cp "$STAGE/$CARRIER/cloth_base.png" "$LIVE/cloth_base.png"
cd "$PI" && bash run_cloth330.sh 0 patch --on >> "$WD" 2>&1
sync
echo "$(date '+%F %T') [fsatk] 载体已切 $CARRIER" >> "$WD"

# ---- 2) 共享池采集（GPU1 先起，GPU0 可能被他人占用）----
if [ ! -f "$POOL" ]; then
  wait_gpu_free 1
  echo "$(date '+%F %T') [fsatk] 采集共享池 GPU1" >> "$WD"
  bash "$PI/run_cloth330.sh" 1 train --suite libero_spatial --pool-tasks 0,2,6,8 \
    --pool-mode full --stride 10 --max-frames-ep 40 --frame-weighting taao \
    --place-jitter --collect-only --pool-file "$POOL" --ckpt "$CKPT" \
    > "$LOGD/fs_collect.log" 2>&1 \
    || { echo "$(date '+%F %T') [fsatk] 采集失败，退出" >> "$WD"; exit 1; }
  echo "$(date '+%F %T') [fsatk] 池采集完成" >> "$WD"
fi

# ---- 3) 双臂并行训练 ----
echo "$(date '+%F %T') [fsatk] 双臂训练启动（GPU0=A uv-eot / GPU1=B no-eot）" >> "$WD"
( wait_gpu_free 0
  bash "$PI/run_cloth330.sh" 0 train --suite libero_spatial --pool-tasks 0,2,6,8 \
    --pool-mode full --frame-weighting taao --epsilon $EPS --seed 7 --iters 5000 \
    --load-pool --pool-file "$POOL" --uv-eot --tag fsA_eot --ckpt "$CKPT" \
    > "$LOGD/fs_trainA.log" 2>&1 ) & pidA=$!
( wait_gpu_free 1
  bash "$PI/run_cloth330.sh" 1 train --suite libero_spatial --pool-tasks 0,2,6,8 \
    --pool-mode full --frame-weighting taao --epsilon $EPS --seed 7 --iters 5000 \
    --load-pool --pool-file "$POOL" --tag fsB_noeot --ckpt "$CKPT" \
    > "$LOGD/fs_trainB.log" 2>&1 ) & pidB=$!
wait $pidA $pidB
TEX_A=$(ls -t $EX/train_*fsA_eot/Universal_Cloth.png 2>/dev/null | head -1)
TEX_B=$(ls -t $EX/train_*fsB_noeot/Universal_Cloth.png 2>/dev/null | head -1)
echo "$(date '+%F %T') [fsatk] 双臂训练完成 A=$TEX_A B=$TEX_B" >> "$WD"

# ---- 4) 快评：未见任务 plain + 两臂 adv ----
rc=0
run_block 0 fs_qk_plain  --condition plain --suite libero_spatial --tasks $UT --trials 10 & p1=$!
run_block 1 fs_qk_advA   --condition adv   --suite libero_spatial --tasks $UT --trials 10 --texture "$TEX_A" & p2=$!
wait $p1 || rc=1; wait $p2 || rc=1
run_block 0 fs_qk_advB   --condition adv   --suite libero_spatial --tasks $UT --trials 10 --texture "$TEX_B" || rc=1

# ---- 5) 矩阵：四套件 × {plain, advA}（A=uv-eot 为默认主臂，如快评 B 显著更优再补 B 矩阵）----
for suite in libero_spatial libero_object libero_goal libero_10; do
  run_block 0 fs_mx_plain_${suite} --condition plain --suite $suite --tasks 0,1,2,3,4,5,6,7,8,9 --trials 10 & p1=$!
  run_block 1 fs_mx_advA_${suite}  --condition adv   --suite $suite --tasks 0,1,2,3,4,5,6,7,8,9 --trials 10 --texture "$TEX_A" & p2=$!
  wait $p1 || rc=1; wait $p2 || rc=1
done

# ---- 6) 恢复默认资产 ----
cp "$STAGE/_live_backup.obj" "$LIVE/tablecloth.obj" 2>/dev/null || cp "$STAGE/60x40_grid/tablecloth.obj" "$LIVE/tablecloth.obj"
cp "$STAGE/_live_backup.png" "$LIVE/cloth_base.png" 2>/dev/null || cp "$STAGE/60x40_grid/cloth_base.png" "$LIVE/cloth_base.png"
if [ "$rc" -eq 0 ]; then
  echo "$(date '+%F %T') [fsatk] DONE-FSATK 全部完成" >> "$WD"
else
  echo "$(date '+%F %T') [fsatk] DONE-FSATK-WITH-FAILURES rc=$rc" >> "$WD"
fi
