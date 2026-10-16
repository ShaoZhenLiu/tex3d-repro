#!/bin/bash
# P3b 跨套件快评（09，GPU0/1/3）：P3 arm-A（uv-eot）布 × SOG {goal,object,10} × {plain,adv}
#   30x20_solid 载体已 live（P3r 结束态）。plain 需按新载体重测；original 复用 P0（无布与载体无关）。
#   每块 10 任务 × 10 集。完成后恢复 60x40 grid 默认资产。
D=/data/shaozhen.liu
PI=$D/code/tex3d/pi
EX=$D/code/tex3d/experiments/pi0_cloth
LIVE=$D/code/tex3d-deps/libero-pi/libero/libero/assets/tablecloth
STAGE=$D/code/tex3d-deps/cloth_variants
TA=$EX/train_20261016_051845_p3e_eot/Universal_Cloth.png
LOGD=$EX/p3b_logs
WD=$EX/watchdog_events.log
mkdir -p "$LOGD"
echo "$(date '+%F %T') [chain-P3b] 跨套件快评启动（arm-A uv-eot 布，30x20 solid）" >> "$WD"

run_block() {  # $1=GPU $2=cond $3=suite
  local g=$1 cond=$2 suite=$3
  local log="$LOGD/p3b_${cond}_${suite}.log"
  for attempt in 1 2 3; do
    echo "$(date '+%F %T') [chain-P3b] ${cond} ${suite} 第${attempt}次 GPU$g" >> "$WD"
    local extra=""
    [ "$cond" = adv ] && extra="--texture $TA"
    bash "$PI/run_cloth.sh" "$g" eval --condition "$cond" --suite "$suite" \
      --tasks 0,1,2,3,4,5,6,7,8,9 --trials 10 $extra >> "$log" 2>&1
    rc=$?
    if [ $rc -eq 0 ] && grep -q "\[eval\] 结果" "$log"; then
      echo "$(date '+%F %T') [chain-P3b] ${cond} ${suite} 完成" >> "$WD"
      return 0
    fi
    echo "$(date '+%F %T') [watchdog] P3b ${cond} ${suite} rc=$rc 重试" >> "$WD"
    sleep 30
  done
  echo "$(date '+%F %T') [watchdog] P3b ${cond} ${suite} FAILED" >> "$WD"
}

run_gpu_queue() {  # $1=GPU 其余 "cond:suite" 列表
  local g=$1; shift
  for spec in "$@"; do
    run_block "$g" "${spec%%:*}" "${spec#*:}"
  done
}

pids=()
run_gpu_queue 0 "plain:libero_goal" "adv:libero_goal" & pids+=($!)
run_gpu_queue 1 "plain:libero_object" "adv:libero_object" & pids+=($!)
run_gpu_queue 3 "plain:libero_10" "adv:libero_10" & pids+=($!)
for p in "${pids[@]}"; do wait "$p"; done

cp "$STAGE/_live_backup.obj" "$LIVE/tablecloth.obj" 2>/dev/null || cp "$STAGE/60x40_grid/tablecloth.obj" "$LIVE/tablecloth.obj"
cp "$STAGE/_live_backup.png" "$LIVE/cloth_base.png" 2>/dev/null || cp "$STAGE/60x40_grid/cloth_base.png" "$LIVE/cloth_base.png"
echo "$(date '+%F %T') [chain-P3b] 跨套件快评全部完成 DONE-P3b" >> "$WD"
