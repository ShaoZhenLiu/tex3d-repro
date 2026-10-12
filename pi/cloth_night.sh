#!/bin/bash
# cloth_night.sh — 通宵编排（09 服务器）
#   GPU0: 帧池采集 → 通用训练 5000 iter + plain/adv 评测臂
#   GPU1: plain/adv 评测臂
#   GPU2: px 队列自行跑完后 → original 评测臂
#   GPU3: 留给其他人，本脚本不碰
# 关键约束：场景 XML 的 on/off 切换只发生在主线程屏障处（单一写者），
#           plain/adv 阶段全局 patched，original 阶段全局 clean，绝不交错。
# 启动：setsid bash cloth_night.sh </dev/null >/dev/null 2>&1 &
# 日志：experiments/pi0_cloth/night_*.log
set -u
PI=/data/shaozhen.liu/code/tex3d/pi
EX=/data/shaozhen.liu/code/tex3d/experiments/pi0_cloth
POOL=$EX/cloth_pool.pt
FLAG=$EX/train_done.flag
TPATH=$EX/texture_path.txt
LOG=$EX/night_$(date +%Y%m%d_%H%M%S).log
mkdir -p "$EX"
exec >>"$LOG" 2>&1
echo "=== night 编排启动 $(date) ==="

rm -f "$FLAG" "$TPATH"

# 内存预检：可用 <40G 先等（保护机器优先于进度）
while [ "$(free -m | awk '/^Mem:/{print $7}')" -lt 40960 ]; do
  echo "[main] 可用内存不足 40G，等待 60s"; sleep 60
done

# 单一写者：确保 patched（已是则 no-op）
bash $PI/run_cloth.sh 0 patch --on

export NIGHT_A_PID=""
export POOL FLAG TPATH PI EX

worker() { # GPU TAG spec...  spec=cond:tasks
  local GPU=$1 TAG=$2; shift 2
  for spec in "$@"; do
    local cond=${spec%%:*} tasks=${spec#*:} extra=""
    if [ "$cond" = adv ]; then
      while [ ! -f "$FLAG" ] && kill -0 "$NIGHT_A_PID" 2>/dev/null; do
        sleep 30
      done
      if [ ! -f "$FLAG" ]; then
        echo "[$TAG] 训练链已退出且无纹理，跳过 adv 臂"; return
      fi
      extra="--texture $(cat "$TPATH")"
    fi
    echo "[$TAG $(date +%T)] >>> $cond tasks=$tasks 开始"
    bash $PI/run_cloth.sh "$GPU" eval --condition "$cond" --tasks "$tasks" \
         --trials 50 $extra
    echo "[$TAG $(date +%T)] <<< $cond tasks=$tasks 结束 rc=$?"
  done
}

# ---------- ON 阶段（patched：plain 8 臂 + adv 8 臂）----------
(
  if [ -s "$POOL" ]; then
    echo "[A] 帧池缓存命中，跳过采集"
  else
    echo "[A $(date +%T)] 帧池采集开始（4 任务 × 2 回合 × 10 帧）"
    bash $PI/run_cloth.sh 0 train --collect-only --pool-file "$POOL" \
      || { echo "[A] 采集失败，A 链退出"; exit 1; }
  fi
  echo "[A $(date +%T)] 通用训练开始（5000 iter）"
  bash $PI/run_cloth.sh 0 train --load-pool --pool-file "$POOL" --iters 5000 \
    || { echo "[A] 训练失败，A 链退出"; exit 1; }
  ls -t $EX/train_*/Universal_Cloth.png 2>/dev/null | head -1 > "$TPATH"
  touch "$FLAG"
  echo "[A $(date +%T)] 纹理出炉：$(cat "$TPATH")"
) & A=$!
export NIGHT_A_PID=$A

worker 1 W1 plain:0,2 adv:0,2 & W1=$!
worker 1 W2 plain:1,8 adv:1,8 & W2=$!
worker 0 W3 plain:5,7 plain:6,9 & W3=$!
worker 0 W4 adv:5,7 adv:6,9 & W4=$!

wait $A $W1 $W2 $W3 $W4
echo "=== ON 阶段（plain+adv 16 臂 + 训练）全部结束 $(date) ==="

# ---------- OFF 阶段（clean：original 8 臂）----------
bash $PI/run_cloth.sh 0 patch --off
worker 0 O1 original:0,2 & O1=$!
worker 1 O2 original:1,8 & O2=$!
( # GPU2：等 px 队列（attack_openvla_px）彻底结束再上，避免挤占
  while pgrep -f 'attack_openvla_p[x]' >/dev/null 2>&1; do sleep 60; done
  worker 2 O3 original:5,7
) & O3=$!
(
  while pgrep -f 'attack_openvla_p[x]' >/dev/null 2>&1; do sleep 60; done
  worker 2 O4 original:6,9
) & O4=$!
wait $O1 $O2 $O3 $O4
echo "=== OFF 阶段（original 8 臂）结束 $(date) ==="

# ---------- 汇总 ----------
echo "=== 结果汇总 $(date) ==="
for f in $EX/eval_*/evaluation_result.txt; do
  echo "----- $f"
  tail -15 "$f" 2>/dev/null
done
echo "=== night 编排全部完成 $(date) ==="
