#!/bin/bash
# cloth_v2.sh — GPU3 上的 v2 链：满轨迹采样 + TAAO 权重
#   1) 满轨迹帧池采集（stride=10, cap=40/集, TAAO 权重内嵌）
#   2) 4 路并行训练（2×2：{taao,uniform}×{seed7,seed8}，显存 4×~17G < 80G）
#   3) 等通宵编排(cloth_night)完全退出后（XML 相位门控），4 张纹理 × t0 × 25 集快评
#   4) 收尾：XML 还原 clean
# 启动：setsid bash cloth_v2.sh </dev/null >/dev/null 2>&1 &
set -u
PI=/data/shaozhen.liu/code/tex3d/pi
EX=/data/shaozhen.liu/code/tex3d/experiments/pi0_cloth
POOL=$EX/cloth_pool_v2.pt
LOG=$EX/v2_$(date +%Y%m%d_%H%M%S).log
exec >> "$LOG" 2>&1
echo "=== v2 链启动 $(date) ==="

if [ -s "$POOL" ]; then
  echo "[v2] 满轨迹帧池缓存命中，跳过采集"
else
  echo "[v2 $(date +%T)] 满轨迹采集开始（需 XML=patched；当前 night 处于 ON 阶段，兼容）"
  bash $PI/run_cloth.sh 3 train --collect-only --pool-file "$POOL" \
    --pool-mode full --stride 10 --max-frames-ep 40 --frame-weighting taao \
    || { echo "[v2] 采集失败，链退出"; exit 1; }
fi

for spec in "taao 7" "uniform 7" "taao 8" "uniform 8"; do
  set -- $spec; W=$1; S=$2
  echo "[v2 $(date +%T)] 训练 v2${W}_s${S} 启动"
  bash $PI/run_cloth.sh 3 train --load-pool --pool-file "$POOL" \
    --iters 5000 --frame-weighting "$W" --seed "$S" --tag "v2${W}_s${S}" \
    > "$EX/train_v2${W}_s${S}.log" 2>&1 &
  sleep 75
done
wait
echo "=== v2 四路训练全部结束 $(date) ==="

# 相位门控：等通宵编排主进程退出（其 adv/plain 臂要求 XML=patched、original 臂要求
# clean；快评为 adv 也要求 patched，必须错开 original 阶段）
while pgrep -f 'cloth_nigh[t]' >/dev/null 2>&1; do
  echo "[v2 $(date +%T)] 等待 night 编排结束…"; sleep 180
done
sleep 60
echo "[v2 $(date +%T)] night 已结束，开始快评"

for tag in v2taao_s7 v2uniform_s7 v2taao_s8 v2uniform_s8; do
  TEX=$(ls -td $EX/train_*_"$tag"/Universal_Cloth.png 2>/dev/null | head -1)
  if [ -z "$TEX" ]; then echo "[v2] 缺纹理 $tag，跳过"; continue; fi
  echo "[v2 $(date +%T)] 快评 $tag：adv × t0 × 25 集（纹理 $TEX）"
  bash $PI/run_cloth.sh 3 eval --condition adv --tasks 0 --trials 25 \
    --texture "$TEX" >> "$EX/spot_$tag.log" 2>&1
  echo "[v2 $(date +%T)] 快评 $tag 结束 rc=$?"
done

bash $PI/run_cloth.sh 3 patch --off
echo "=== v2 链全部完成 $(date) ==="
