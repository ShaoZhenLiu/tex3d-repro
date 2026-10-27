#!/bin/bash
# 41_r18_e06_master.sh — E06 自动串联（用户顺序：E04/E05/E07 全部完成后）
#   等 DONE-R18BC-TRAIN 10/10 → 幂等驱动 41_r18_e06_queue.sh（2/2）
#   → 双臂 OFT-Spatial 闭环 200 回合（adv×2；O/P 复用 E02 已核验结果）
#   → E06R18-DONE 标记
set -u
D=/data/shaozhen.liu
EX=$D/code/tex3d/experiments/r18_e03
E06ROOT=$D/code/tex3d/experiments/pi0_cloth/e06_pi0src
WD=$D/code/tex3d/experiments/pi0_cloth/watchdog_events.log
LOGD=$EX/e06_ofs_logs
mkdir -p "$LOGD"
log() { echo "$(date '+%F %T') [e06m] $*" | tee -a "$WD"; }

# 1) 等主训练队列 10/10
while ! grep -q "DONE-R18BC-TRAIN 10/10" "$EX/train_logs/summary.txt" 2>/dev/null; do
  sleep 600
done
log "主队列 10/10 达成，启动 E06 训练队列"

# 2) 幂等驱动 E06 训练队列直至 2/2
while ! grep -q "DONE-E06-TRAIN 2/2" "$EX/e06_train_logs/summary.txt" 2>/dev/null; do
  if ! ps -u shaozhen.liu -o cmd | grep "41_r18_e06_queu[e]" >/dev/null 2>&1; then
    ( cd "$EX" && setsid nohup bash 41_r18_e06_queue.sh \
        >> e06_queue_master.log 2>&1 < /dev/null & )
  fi
  sleep 600
done
log "E06 双臂训练 2/2 完成"

# 3) OFT-Spatial 两臂 200 回合（O/P 复用 E02；条件=adv，初态10-19，400+10步）
gpu_used_mi() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$1" 2>/dev/null | tr -d ' '; }
pick1() {  # 找一块空闲≥20000 的卡（GPU4 永不）
  local g used
  for g in 5 7 6 2 0 1 3; do
    used=$(gpu_used_mi "$g")
    [ -n "$used" ] && [ $((49140 - used)) -ge 20000 ] && { echo "$g"; return 0; }
  done
  return 1
}
run_e06_eval() {  # $1=tag $2=gpu
  local tag=$1 g=$2 d res n
  d=$(ls -dt $E06ROOT/train_*_"$tag" 2>/dev/null | head -1)
  local tex="$d/Universal_Cloth.png"
  [ -f "$tex" ] || { log "E06 eval $tag 缺纹理"; return 1; }
  for attempt in 1 2 3 4 5 6 7 8; do
    log "E06 eval $tag 尝试 $attempt GPU$g"
    ( cd $D/code/tex3d-ws/g7 && env CUDA_VISIBLE_DEVICES=$g MUJOCO_EGL_DEVICE_ID=$g \
      MUJOCO_GL=egl PYOPENGL_PLATFORM=egl HF_HUB_OFFLINE=1 \
      TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled \
      PYTHONPATH=$D/code/tex3d:$D/code/tex3d/openvla-oft \
      $D/envs/tex3d-oft-mj330/bin/python \
      $D/code/tex3d/openvla-oft/experiments/robot/libero/transfer_pi2oft.py eval \
      --condition adv --suite libero_spatial \
      --tasks 0,1,2,3,4,5,6,7,8,9 --trials 10 --init-offset 10 --max-steps 400 \
      --texture "$tex" ) >> "$LOGD/${tag}.log" 2>&1
    res=$(grep -a '\[eval\] 结果' "$LOGD/${tag}.log" 2>/dev/null | tail -1 | sed 's/.*结果: //' | tr -d '\r')
    n=$(grep -ac '^Task:' "$res" 2>/dev/null)
    if [ -n "$res" ] && [ -f "$res" ] && [ "${n:-0}" -ge 100 ]; then
      log "E06 eval $tag 完成 -> $(grep -a 'overall robot success' "$res" | tail -1)"
      return 0
    fi
    sleep 120
    g=$(pick1) || true
    [ -z "$g" ] && g=5
  done
  log "E06 eval $tag 八次失败 FAILED"; return 1
}
# 两个 adv 都为 patch-on，可并行；每 5 分钟探测一次双卡可用
gA=""; gB=""
while [ -z "$gA" ] || [ -z "$gB" ] || [ "$gA" = "$gB" ]; do
  gA=$(pick1); sleep 5; gB=$(pick1)
  [ -n "$gA" ] && [ -z "$gB" ] && gB=""
  [ -z "$gA" ] && [ -n "$gB" ] && gA=""
  [ -n "$gA" ] && [ -n "$gB" ] && [ "$gA" = "$gB" ] && sleep 300
  [ -z "$gA" ] && sleep 300
done
log "E06 OFT 评测双臂并行 GPU$gA/GPU$gB"
run_e06_eval e06_rn "$gA" &
PA=$!
run_e06_eval e06_rm "$gB" &
PB=$!
wait $PA $PB
echo "E06R18-DONE $(date)" >> "$EX/train_logs/summary.txt"
log "E06R18-DONE（OFT 200 回合收口；π0 源端 80 诊断回合另行手动可选）"
