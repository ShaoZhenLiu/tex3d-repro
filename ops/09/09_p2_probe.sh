#!/bin/bash
# P2 探针链（09，GPU0/1/3）：选定载体 30x20_solid 上：
#   A) long2 纹理原幅度（α=1.0）   —— 旧扰动搬到小布，信号是否还在
#   B) α=0.5 / α=0.25 折减         —— 非饱和区间粗定位
#   C) 匹配随机对照 ×2 seed         —— 背景脆弱性 vs 优化特异性
# 每臂 = dev 任务（spatial t0,6 / goal t1 / object t1）× 10 集，3 卡按套件并行。
# 前置：等 p0r 三块全部完成（各自日志出现 overall 行）。
D=/data/shaozhen.liu
PI=$D/code/tex3d/pi
EX=$D/code/tex3d/experiments/pi0_cloth
LIVE=$D/code/tex3d-deps/libero-pi/libero/libero/assets/tablecloth
STAGE=$D/code/tex3d-deps/cloth_variants
TEXD=$EX/p2_textures
LOGD=$EX/p2_logs
WD=$EX/watchdog_events.log
CARRIER=30x20_solid
mkdir -p "$LOGD"
echo "$(date '+%F %T') [chain-P2] 启动（等待 p0r 三块完成）" >> "$WD"

while true; do
  ok=0
  for f in "$EX"/p0_logs/p0r_*.log; do grep -q "\[eval\] 结果" "$f" 2>/dev/null && ok=$((ok+1)); done
  [ "$ok" -ge 3 ] && break
  sleep 300
done
echo "$(date '+%F %T') [chain-P2] p0r 完成，切换载体 $CARRIER 开始探针" >> "$WD"

cp "$STAGE/$CARRIER/tablecloth.obj" "$LIVE/tablecloth.obj"
cp "$STAGE/$CARRIER/cloth_base.png" "$LIVE/cloth_base.png"
cd "$PI" && bash run_cloth.sh 0 patch --on >> "$WD" 2>&1
sync

run_block() {  # $1=GPU $2=name $3=cond $4=suite $5=tasks 其余 extra
  local g=$1 name=$2 cond=$3 suite=$4 tasks=$5; shift 5
  local log="$LOGD/p2_${name}_${suite}.log"
  for attempt in 1 2 3; do
    echo "$(date '+%F %T') [chain-P2] ${name} ${suite} 第${attempt}次 GPU$g" >> "$WD"
    bash "$PI/run_cloth.sh" "$g" eval --condition "$cond" --suite "$suite" \
      --tasks "$tasks" --trials 10 "$@" >> "$log" 2>&1
    rc=$?
    if [ $rc -eq 0 ] && grep -q "\[eval\] 结果" "$log"; then
      echo "$(date '+%F %T') [chain-P2] ${name} ${suite} 完成" >> "$WD"
      return 0
    fi
    echo "$(date '+%F %T') [watchdog] P2 ${name} ${suite} rc=$rc 重试" >> "$WD"
    sleep 30
  done
  echo "$(date '+%F %T') [watchdog] P2 ${name} ${suite} FAILED" >> "$WD"
}

# 各臂：adv 条件 + 对应纹理 PNG；plain 基线直接复用 P1 的 30x20_solid 结果（同资产同初态）
ARMS=(
  "long2a100|$TEXD/long2_a100_on_$CARRIER.png"
  "long2a50|$TEXD/long2_a50_on_$CARRIER.png"
  "long2a25|$TEXD/long2_a25_on_$CARRIER.png"
  "rand_s7|$TEXD/rand_s7_on_$CARRIER.png"
  "rand_s8|$TEXD/rand_s8_on_$CARRIER.png"
)
for arm in "${ARMS[@]}"; do
  name="${arm%%|*}"; tex="${arm#*|}"
  [ -f "$tex" ] || { echo "$(date '+%F %T') [chain-P2] 缺纹理 $tex，跳过 $name" >> "$WD"; continue; }
  pids=()
  run_block 0 "$name" adv libero_spatial "0,6" --texture "$tex" & pids+=($!)
  run_block 1 "$name" adv libero_goal    "1"    --texture "$tex" & pids+=($!)
  run_block 3 "$name" adv libero_object  "1"    --texture "$tex" & pids+=($!)
  for p in "${pids[@]}"; do wait "$p"; done
  echo "$(date '+%F %T') [chain-P2] === 臂 $name 三块完成 ===" >> "$WD"
done

# 收尾：恢复 60x40 grid 原版资产（保持后续训练管线默认一致）
cp "$STAGE/_live_backup.obj" "$LIVE/tablecloth.obj" 2>/dev/null || cp "$STAGE/60x40_grid/tablecloth.obj" "$LIVE/tablecloth.obj"
cp "$STAGE/_live_backup.png" "$LIVE/cloth_base.png" 2>/dev/null || cp "$STAGE/60x40_grid/cloth_base.png" "$LIVE/cloth_base.png"
echo "$(date '+%F %T') [chain-P2] 探针全部完成 DONE-P2" >> "$WD"
