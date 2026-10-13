#!/bin/bash
# cloth_eps32.sh — ε=32/255 降火力对照链
#   GPU0: 4 路并行重训（复用已有帧池，不重采）
#   GPU1: 训练完成后 4 张纹理 × t0 × 25 集快评（与 ε=64 的 0/25 天花板对照）
# 启动：setsid bash cloth_eps32.sh </dev/null >/dev/null 2>&1 &
set -u
PI=/data/shaozhen.liu/code/tex3d/pi
EX=/data/shaozhen.liu/code/tex3d/experiments/pi0_cloth
POOL1=$EX/cloth_pool.pt
POOL2=$EX/cloth_pool_v2.pt
LOG=$EX/eps32_$(date +%Y%m%d_%H%M%S).log
exec >> "$LOG" 2>&1
echo "=== eps32 链启动 $(date) ==="
EPS=0.12549

train() { # pool weighting seed tag
  bash $PI/run_cloth.sh 0 train --load-pool --pool-file "$1" \
    --iters 5000 --epsilon $EPS --frame-weighting "$2" --seed "$3" --tag "$4" \
    > "$EX/train_$4.log" 2>&1 &
}

[ -s "$POOL1" ] && train "$POOL1" uniform 7 eps32_v1cfg
[ -s "$POOL2" ] && train "$POOL2" uniform 7 eps32_v2u_s7
sleep 60
[ -s "$POOL2" ] && train "$POOL2" taao 7 eps32_v2t_s7
sleep 60
[ -s "$POOL2" ] && train "$POOL2" taao 8 eps32_v2t_s8
wait
echo "=== eps32 四路训练结束 $(date) ==="

for tag in eps32_v1cfg eps32_v2u_s7 eps32_v2t_s7 eps32_v2t_s8; do
  TEX=$(ls -td $EX/train_*_"$tag"/Universal_Cloth.png 2>/dev/null | head -1)
  if [ -z "$TEX" ]; then echo "[eps32] 缺纹理 $tag，跳过"; continue; fi
  echo "[eps32 $(date +%T)] 快评 $tag：adv × t0 × 25 集"
  bash $PI/run_cloth.sh 1 eval --condition adv --tasks 0 --trials 25 \
    --texture "$TEX" >> "$EX/spot_$tag.log" 2>&1
  echo "[eps32 $(date +%T)] 快评 $tag 结束 rc=$?"
done
bash $PI/run_cloth.sh 1 patch --off
echo "=== eps32 链全部完成 $(date) ==="
