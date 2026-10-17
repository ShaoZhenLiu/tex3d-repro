#!/bin/bash
# F4: four-arm loss comparison at 60x40 native (advisor step 3)
#   Arms: A1=MSE | A2=MSE+con | A3=MSE+con+task | A4=con+task   (all uv-eot, eps64, seed7, 5000it)
#   λ calibration from gradient smokes: mse g~1.0e-3, con(λ1)~0.34x, task(λ1)~4.4x
#     -> --lambda-con 3.0 --lambda-task 0.25 (each aux ~1x mse contribution)
#   Evals: GPU0 = OFT-Spatial adv x100 per arm (decision metric);
#          GPU1 = pi0 spatial x100 + unseen {1,3,4,5,7,9} x60 per arm.
#   Trees: pi0 tree -> 60x40_solid + patched for the whole run; OFT tree switched at eval phase.
D=/data/shaozhen.liu
PI=$D/code/tex3d/pi
OFT=$D/code/tex3d/openvla-oft/experiments/robot/libero
EX=$D/code/tex3d/experiments/pi0_cloth
STAGE=$D/code/tex3d-deps/cloth_variants
LIVE_PI=$D/code/tex3d-deps/libero-pi/libero/libero/assets/tablecloth
LIVE_OFT=$D/code/tex3d-deps/libero/libero/libero/assets/tablecloth
CKPT=$D/model/RLinf-Pi0-LIBERO-130-fullshot-SFT
POOL=$EX/cloth_pool_fs6040.pt
LOGD=$EX/f4_logs
WD=$EX/watchdog_events.log
UT=1,3,4,5,7,9
mkdir -p "$LOGD"
echo "$(date '+%F %T') [f4] 四臂链启动 master=$$（60x40 原生）" >> "$WD"

bash /data/shaozhen.liu/code/tex3d/ops/09/ckpt_guard.sh "$CKPT" >> "$WD" 2>&1 || exit 1
cp "$STAGE/60x40_solid/tablecloth.obj" "$LIVE_PI/tablecloth.obj"
cp "$STAGE/60x40_solid/cloth_base.png" "$LIVE_PI/cloth_base.png"
cd "$PI" && bash run_cloth330.sh 0 patch --on >> "$WD" 2>&1
sync

wait_gpu_free() {
  while true; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$1" | tr -d ' ')
    [ "$used" -lt 60000 ] && break
    echo "$(date '+%F %T') [f4] GPU$1 被占(${used}MiB)，等待" >> "$WD"
    sleep 300
  done
}

# ---- 1) 60x40 原生帧池 ----
if [ ! -f "$POOL" ]; then
  wait_gpu_free 0
  echo "$(date '+%F %T') [f4] 采集 60x40 原生池 GPU0" >> "$WD"
  bash "$PI/run_cloth330.sh" 0 train --suite libero_spatial --pool-tasks 0,2,6,8 \
    --pool-mode full --stride 10 --max-frames-ep 40 --frame-weighting taao \
    --place-jitter --collect-only --pool-file "$POOL" --ckpt "$CKPT" \
    > "$LOGD/f4_collect.log" 2>&1 \
    || { echo "$(date '+%F %T') [f4] 采集失败，退出" >> "$WD"; exit 1; }
  echo "$(date '+%F %T') [f4] 池采集完成" >> "$WD"
fi

# ---- 2) 四臂两轮训练 ----
train_arm() {  # $1=GPU $2=tag $3=lambda_mse $4=lambda_con $5=lambda_task
  wait_gpu_free "$1"
  bash "$PI/run_cloth330.sh" "$1" train --suite libero_spatial --pool-tasks 0,2,6,8 \
    --pool-mode full --frame-weighting taao --epsilon 0.2510 --seed 7 --iters 5000 \
    --load-pool --pool-file "$POOL" --uv-eot --ckpt "$CKPT" \
    --lambda-mse "$3" --lambda-con "$4" --lambda-task "$5" --tag "$2" \
    > "$LOGD/f4_train_$2.log" 2>&1
}
echo "$(date '+%F %T') [f4] Round1: A1(GPU0) + A2(GPU1)" >> "$WD"
train_arm 0 f4a1 1.0 0.0 0.0 & p1=$!
train_arm 1 f4a2 1.0 3.0 0.0 & p2=$!
wait $p1 || echo "$(date '+%F %T') [f4] A1 训练异常" >> "$WD"
wait $p2 || echo "$(date '+%F %T') [f4] A2 训练异常" >> "$WD"
echo "$(date '+%F %T') [f4] Round2: A3(GPU0) + A4(GPU1)" >> "$WD"
train_arm 0 f4a3 1.0 3.0 0.25 & p1=$!
train_arm 1 f4a4 0.0 3.0 0.25 & p2=$!
wait $p1 || echo "$(date '+%F %T') [f4] A3 训练异常" >> "$WD"
wait $p2 || echo "$(date '+%F %T') [f4] A4 训练异常" >> "$WD"
T1=$(ls -t $EX/train_*f4a1/Universal_Cloth.png 2>/dev/null | head -1)
T2=$(ls -t $EX/train_*f4a2/Universal_Cloth.png 2>/dev/null | head -1)
T3=$(ls -t $EX/train_*f4a3/Universal_Cloth.png 2>/dev/null | head -1)
T4=$(ls -t $EX/train_*f4a4/Universal_Cloth.png 2>/dev/null | head -1)
echo "$(date '+%F %T') [f4] 四臂训练完成 A1=$T1 A2=$T2 A3=$T3 A4=$T4" >> "$WD"

# ---- 3) 评测：GPU1=π0 队列，GPU0=OFT 队列 ----
pi_eval() {  # $1=name 其余为 eval 参数
  local name=$1; shift
  local log="$LOGD/${name}.log"
  for attempt in 1 2 3; do
    wait_gpu_free 1
    bash "$PI/run_cloth330.sh" 1 eval --ckpt "$CKPT" "$@" >> "$log" 2>&1
    res=$(grep -a "\[eval\] 结果" "$log" | tail -1 | sed "s/.*结果: //" | tr -d "\r")
    if [ -n "$res" ] && [ -f "$res" ] && grep -q "overall robot success" "$res"; then
      echo "$(date '+%F %T') [f4] $name OK -> $res" >> "$WD"; return 0
    fi
    echo "$(date '+%F %T') [f4] $name 第${attempt}次未完成，重试" >> "$WD"; sleep 30
  done
  echo "$(date '+%F %T') [f4] $name FAILED" >> "$WD"; return 1
}

ENVV="CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled PYTHONPATH=$D/code/tex3d:$D/code/tex3d/openvla-oft"
PYOFT=$D/envs/tex3d-oft-mj330/bin/python
cd "$D/code/tex3d-ws/g7" || exit 1
cp "$STAGE/60x40_solid/tablecloth.obj" "$LIVE_OFT/tablecloth.obj"
cp "$STAGE/60x40_solid/cloth_base.png" "$LIVE_OFT/cloth_base.png"
sync

oft_eval() {  # $1=name $2=texture
  local name=$1 tex=$2
  local log="$LOGD/${name}.log"
  for attempt in 1 2 3; do
    wait_gpu_free 0
    env $ENVV "$PYOFT" "$OFT/transfer_pi2oft.py" eval --condition adv \
      --tasks 0,1,2,3,4,5,6,7,8,9 --trials 10 --texture "$tex" >> "$log" 2>&1
    res=$(grep -a "\[eval\] 结果" "$log" | tail -1 | sed "s/.*结果: //" | tr -d "\r")
    if [ -n "$res" ] && [ -f "$res" ] && grep -q "overall robot success" "$res"; then
      echo "$(date '+%F %T') [f4] $name OK -> $res" >> "$WD"; return 0
    fi
    echo "$(date '+%F %T') [f4] $name 第${attempt}次未完成，重试" >> "$WD"; sleep 30
  done
  echo "$(date '+%F %T') [f4] $name FAILED" >> "$WD"; return 1
}

( # GPU1: π0 队列
  cd "$PI"
  pi_eval f4_pi_a1_sp --condition adv --suite libero_spatial --tasks 0,1,2,3,4,5,6,7,8,9 --trials 10 --texture "$T1"
  pi_eval f4_pi_a1_ut --condition adv --suite libero_spatial --tasks $UT --trials 10 --texture "$T1"
  pi_eval f4_pi_a2_sp --condition adv --suite libero_spatial --tasks 0,1,2,3,4,5,6,7,8,9 --trials 10 --texture "$T2"
  pi_eval f4_pi_a2_ut --condition adv --suite libero_spatial --tasks $UT --trials 10 --texture "$T2"
  pi_eval f4_pi_a3_sp --condition adv --suite libero_spatial --tasks 0,1,2,3,4,5,6,7,8,9 --trials 10 --texture "$T3"
  pi_eval f4_pi_a3_ut --condition adv --suite libero_spatial --tasks $UT --trials 10 --texture "$T3"
  pi_eval f4_pi_a4_sp --condition adv --suite libero_spatial --tasks 0,1,2,3,4,5,6,7,8,9 --trials 10 --texture "$T4"
  pi_eval f4_pi_a4_ut --condition adv --suite libero_spatial --tasks $UT --trials 10 --texture "$T4"
) & pidpi=$!

( # GPU0: OFT 队列（决策指标）
  cd "$D/code/tex3d-ws/g7"
  oft_eval f4_oft_a1 "$T1"
  oft_eval f4_oft_a2 "$T2"
  oft_eval f4_oft_a3 "$T3"
  oft_eval f4_oft_a4 "$T4"
  env $ENVV "$PYOFT" "$OFT/transfer_pi2oft.py" patch --off >> "$WD" 2>&1
) & pidoft=$!

wait $pidpi; wait $pidoft

# ---- 4) 恢复常态 ----
cp "$STAGE/60x40_grid/tablecloth.obj" "$LIVE_PI/tablecloth.obj"
cp "$STAGE/60x40_grid/cloth_base.png" "$LIVE_PI/cloth_base.png"
cp "$STAGE/60x40_grid/tablecloth.obj" "$LIVE_OFT/tablecloth.obj"
cp "$STAGE/60x40_grid/cloth_base.png" "$LIVE_OFT/cloth_base.png"
echo "$(date '+%F %T') [f4] DONE-F4 四臂+双侧评测完成" >> "$WD"
