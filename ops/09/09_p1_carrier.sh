#!/bin/bash
# P1 载体筛选链 v2（09，GPU0/1/3）：3 尺寸 × 3 底图 = 9 配置 × 4 开发任务 × 10 集 plain + original 对照。
# v2 修复：完成判据同 P0 v2（结果文件含 overall），已有完整结果的块跳过。
# 前置：等待 P0.2 完成（watchdog_events.log 出现 "矩阵链全部完成 DONE"）。
# 资产切换安全：live 资产位三卡共享——只在 wait 栅栏（无评测进程）时 cp 切换。
D=/data/shaozhen.liu
PI=$D/code/tex3d/pi
EX=$D/code/tex3d/experiments/pi0_cloth
LIVE=$D/code/tex3d-deps/libero-pi/libero/libero/assets/tablecloth
STAGE=$D/code/tex3d-deps/cloth_variants
LOGD=$EX/p1_logs
WD=$EX/watchdog_events.log
mkdir -p "$LOGD"

echo "$(date '+%F %T') [chain-P1] v2 启动（等待 P0.2 DONE）" >> "$WD"
while ! grep -q "矩阵链全部完成 DONE" "$WD" 2>/dev/null; do sleep 300; done
echo "$(date '+%F %T') [chain-P1] P0.2 已完成，开始载体筛选" >> "$WD"

cp "$LIVE/tablecloth.obj" "$STAGE/_live_backup.obj" 2>/dev/null
cp "$LIVE/cloth_base.png" "$STAGE/_live_backup.png" 2>/dev/null

wait_gpu_free() {
  while true; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$1" | tr -d ' ')
    [ "$used" -lt 2000 ] && break
    echo "$(date '+%F %T') [chain-P1] GPU$1 被占(${used}MiB)，等待" >> "$WD"
    sleep 300
  done
}

have_result() {  # $1=tag $2=cond $3=suite $4=期望集数——按"tag 时间窗内的结果目录"判重
  # P1 的块复用 eval_<cond>_<suite>_<ts> 目录名，tag 靠启动时刻区分；这里用块日志的
  # 最后一行结果目录 + 该目录的 variant 标记文件（启动器写入）双重确认。
  local log="$LOGD/p1_${1}_${2}_${3}.log" res marker
  res=$(grep "\[eval\] 结果" "$log" 2>/dev/null | tail -1 | sed "s/.*结果: //" | tr -d "\r")
  if [ -n "$res" ] && [ -f "$res" ] && grep -q "overall robot success" "$res" \
     && [ "$(grep -c '^Task:' "$res")" -ge "$4" ]; then
    marker="$(dirname "$res")/variant.txt"
    [ -f "$marker" ] && [ "$(cat "$marker")" = "$1" ] && return 0
  fi
  return 1
}

run_block() {  # $1=GPU $2=tag $3=cond $4=suite $5=tasks $6=期望集数 其余 extra
  local g=$1 tag=$2 cond=$3 suite=$4 tasks=$5 want=$6; shift 6
  local log="$LOGD/p1_${tag}_${cond}_${suite}.log"
  local before after res newest
  have_result "$tag" "$cond" "$suite" "$want" && return 0
  for attempt in 1 2 3; do
    before=$(grep -c "\[eval\] 结果" "$log" 2>/dev/null)
    wait_gpu_free "$g"
    echo "$(date '+%F %T') [chain-P1] ${tag} ${cond} ${suite} 第${attempt}次 GPU$g" >> "$WD"
    bash "$PI/run_cloth.sh" "$g" eval --condition "$cond" --suite "$suite" \
      --tasks "$tasks" --trials 10 "$@" >> "$log" 2>&1
    after=$(grep -c "\[eval\] 结果" "$log" 2>/dev/null)
    if [ "${after:-0}" -gt "${before:-0}" ]; then
      newest=$(grep "\[eval\] 结果" "$log" | tail -1 | sed "s/.*结果: //" | tr -d "\r")
      # 本次产出的结果目录打上 variant 标记，供判重与后续归因
      [ -n "$newest" ] && [ -d "$(dirname "$newest")" ] && echo "$tag" > "$(dirname "$newest")/variant.txt"
      if [ -n "$newest" ] && [ -f "$newest" ] && grep -q "overall robot success" "$newest" \
         && [ "$(grep -c '^Task:' "$newest")" -ge "$want" ]; then
        echo "$(date '+%F %T') [chain-P1] ${tag} ${cond} ${suite} 完成 → $newest" >> "$WD"
        return 0
      fi
    fi
    echo "$(date '+%F %T') [watchdog] P1 ${tag} ${cond} ${suite} 第${attempt}次未完成，重试" >> "$WD"
    sleep 30
  done
  echo "$(date '+%F %T') [watchdog] P1 ${tag} ${cond} ${suite} 三次失败 FAILED" >> "$WD"
  return 1
}

DEV_SPATIAL="0,6"; DEV_GOAL="1"; DEV_OBJECT="1"

# ---- Phase A：9 配置 plain ----
cd "$PI" && bash run_cloth.sh 0 patch --on >> "$WD" 2>&1
for size in 60x40 45x30 30x20; do
  for style in grid solid wood; do
    tag="${size}_${style}"
    cp "$STAGE/$tag/tablecloth.obj" "$LIVE/tablecloth.obj"
    cp "$STAGE/$tag/cloth_base.png" "$LIVE/cloth_base.png"
    sync
    echo "$(date '+%F %T') [chain-P1] === 配置 ${tag} 资产已切换 ===" >> "$WD"
    pids=()
    run_block 0 "$tag" plain libero_spatial "$DEV_SPATIAL" 20 & pids+=($!)
    run_block 1 "$tag" plain libero_goal    "$DEV_GOAL"    10 & pids+=($!)
    run_block 3 "$tag" plain libero_object  "$DEV_OBJECT"  10 & pids+=($!)
    for p in "${pids[@]}"; do wait "$p"; done
    echo "$(date '+%F %T') [chain-P1] === 配置 ${tag} 三块完成 ===" >> "$WD"
  done
done

# ---- Phase B：original 对照（clean 树）----
cd "$PI" && bash run_cloth.sh 0 patch --off >> "$WD" 2>&1
pids=()
run_block 0 orig60x40 original libero_spatial "$DEV_SPATIAL" 20 & pids+=($!)
run_block 1 orig60x40 original libero_goal    "$DEV_GOAL"    10 & pids+=($!)
run_block 3 orig60x40 original libero_object  "$DEV_OBJECT"  10 & pids+=($!)
for p in "${pids[@]}"; do wait "$p"; done

# ---- 收尾 ----
cp "$STAGE/_live_backup.obj" "$LIVE/tablecloth.obj"
cp "$STAGE/_live_backup.png" "$LIVE/cloth_base.png"
cd "$PI" && bash run_cloth.sh 0 patch --on >> "$WD" 2>&1
echo "$(date '+%F %T') [chain-P1] 载体筛选全部完成 DONE-P1" >> "$WD"
