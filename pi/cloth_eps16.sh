#!/bin/bash
# cloth_eps16.sh — ε=16/255 剂量延伸（GPU0 训练+快评）对照链
#   GPU0: 4 路并行重训（复用已有帧池，不重采）
#   GPU1: 训练完成后 4 张纹理 × t0 × 25 集快评（与 ε=64 的 0/25 天花板对照）
# 启动：setsid bash cloth_eps16.sh </dev/null >/dev/null 2>&1 &
set -u
PI=/data/shaozhen.liu/code/tex3d/pi
EX=/data/shaozhen.liu/code/tex3d/experiments/pi0_cloth
POOL1=$EX/cloth_pool.pt
POOL2=$EX/cloth_pool_v2.pt
LOG=$EX/eps16_$(date +%Y%m%d_%H%M%S).log
exec >> "$LOG" 2>&1
echo "=== eps16 链启动 $(date) ==="
EPS=0.062745

train() { # pool weighting seed tag
  bash $PI/run_cloth.sh 0 train --load-pool --pool-file "$1" \
    --iters 5000 --epsilon $EPS --frame-weighting "$2" --seed "$3" --tag "$4" \
    > "$EX/train_$4.log" 2>&1 &
}

[ -s "$POOL1" ] && train "$POOL1" uniform 7 eps16_v1cfg
[ -s "$POOL2" ] && train "$POOL2" uniform 7 eps16_v2u_s7
sleep 60
[ -s "$POOL2" ] && train "$POOL2" taao 7 eps16_v2t_s7
sleep 60
[ -s "$POOL2" ] && train "$POOL2" taao 8 eps16_v2t_s8
wait
echo "=== eps16 四路训练结束 $(date) ==="

for tag in eps16_v1cfg eps16_v2u_s7 eps16_v2t_s7 eps16_v2t_s8; do
  TEX=$(ls -td $EX/train_*_"$tag"/Universal_Cloth.png 2>/dev/null | head -1)
  if [ -z "$TEX" ]; then echo "[eps16] 缺纹理 $tag，跳过"; continue; fi
  echo "[eps16 $(date +%T)] 快评 $tag：adv × t0 × 25 集"
  bash $PI/run_cloth.sh 0 eval --condition adv --tasks 0 --trials 25 \
    --texture "$TEX" >> "$EX/spot_$tag.log" 2>&1
  echo "[eps16 $(date +%T)] 快评 $tag 结束 rc=$?"
done
bash $PI/run_cloth.sh 0 patch --off
echo "=== eps16 链全部完成 $(date) ==="
