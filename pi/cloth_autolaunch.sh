#!/bin/bash
# 桌布训练自动哨兵（在 09 上常驻，setsid 启动）
#   setsid bash cloth_autolaunch.sh &
# 逻辑: 每 10 分钟查一次可用内存；连续两次 >=57GB（=220G wandb 泄漏被清理，
#       目前仅 ~14G）且当前无桌布训练在跑 → 自动在 GPU0 启动采集+训练（带看门狗）。
# 取消: pkill -f cloth_autolaunch.sh
# 手动强启（不等清理，风险自担）: setsid bash cloth_guarded.sh collect_train 0 5000 &
LOG=/data/shaozhen.liu/code/tex3d/experiments/pi0_cloth/autolaunch.log
mkdir -p "$(dirname "$LOG")"
exec >>"$LOG" 2>&1
echo "=== $(date '+%F %T') 哨兵启动（阈值 available>=57GB 连续两次） ==="
strikes=0
while true; do
  if pgrep -f "attack_pi0_cloth.py train" >/dev/null 2>&1; then
    echo "$(date '+%F %T') 检测到桌布训练已在运行，哨兵退出"
    exit 0
  fi
  avail_kb=$(free -k | awk '/^Mem:/{print $7}')
  if [ "${avail_kb:-0}" -ge 60000000 ]; then
    strikes=$((strikes+1))
    echo "$(date '+%F %T') available=$((avail_kb/1024))MB 达标（$strikes/2）"
    if [ "$strikes" -ge 2 ]; then break; fi
  else
    strikes=0
  fi
  sleep 600
done
echo "$(date '+%F %T') 内存已释放，自动启动：GPU0 采集+训练 5000 iters"
bash /data/shaozhen.liu/code/tex3d/pi/cloth_guarded.sh collect_train 0 5000
echo "=== $(date '+%F %T') 哨兵结束 ==="
