#!/bin/bash
# P3 重训链（09，GPU0/1/3）：冻结载体 30x20_solid，源 = SOG-spatial，快评协议。
#   1) 采集共享池：spatial t0,t2,t6,t8 × 2 集，满轨迹 stride10 cap40，TAAO，采集期摆放抖动
#   2) 双臂同池训练（隔离 UV 域 EoT 的贡献）：
#        A=p3e_eot   --uv-eot（外观域 EoT 开）
#        B=p3e_noeot （UV EoT 关；池共享，采集抖动两臂相同）
#   3) 快评：源套件未见任务 {1,3,4,5,7,9} × plain 基线 + 两臂 adv（10 集）
#      original 复用 P0 的 spatial original（无布条件与载体无关）
# 前置：等待 P2 完成（watchdog_events.log 出现 DONE-P2）。
D=/data/shaozhen.liu
PI=$D/code/tex3d/pi
EX=/data/shaozhen.liu/code/tex3d/experiments/pi0_cloth
LIVE=$D/code/tex3d-deps/libero-pi/libero/libero/assets/tablecloth
STAGE=$D/code/tex3d-deps/cloth_variants
POOL=$EX/cloth_pool_p3_sogsp.pt
LOGD=$EX/p3_logs
WD=$EX/watchdog_events.log
CARRIER=30x20_solid
EPS=0.2510   # 64/255
mkdir -p "$LOGD"
echo "$(date '+%F %T') [chain-P3] 启动（等待 DONE-P2）" >> "$WD"
while ! grep -q "探针全部完成 DONE-P2" "$WD" 2>/dev/null; do sleep 300; done
echo "$(date '+%F %T') [chain-P3] P2 完成，切换载体开始重训" >> "$WD"

cp "$STAGE/$CARRIER/tablecloth.obj" "$LIVE/tablecloth.obj"
cp "$STAGE/$CARRIER/cloth_base.png" "$LIVE/cloth_base.png"
cd "$PI" && bash run_cloth.sh 0 patch --on >> "$WD" 2>&1
sync

# 1) 共享池采集（放置抖动开 → 池内布位姿多样，两臂共享）
if [ ! -f "$POOL" ]; then
  echo "$(date '+%F %T') [chain-P3] 采集共享池 GPU0" >> "$WD"
  bash "$PI/run_cloth.sh" 0 train --suite libero_spatial --pool-tasks 0,2,6,8 \
    --pool-mode full --stride 10 --max-frames-ep 40 --frame-weighting taao \
    --place-jitter --collect-only --pool-file "$POOL" \
    > "$LOGD/p3_collect.log" 2>&1 \
    || { echo "$(date '+%F %T') [watchdog] P3 采集失败，退出" >> "$WD"; exit 1; }
  echo "$(date '+%F %T') [chain-P3] 池采集完成" >> "$WD"
fi

# 2) 双臂并行训练（同池同 seed，仅 UV EoT 不同）
TEX_A=$EX/train_p3e_eot*/Universal_Cloth.png
TEX_B=$EX/train_p3e_noeot*/Universal_Cloth.png
echo "$(date '+%F %T') [chain-P3] 双臂训练启动（GPU1=A uv-eot / GPU3=B no-uv-eot）" >> "$WD"
bash "$PI/run_cloth.sh" 1 train --suite libero_spatial --pool-tasks 0,2,6,8 \
  --pool-mode full --frame-weighting taao --epsilon $EPS --seed 7 --iters 5000 \
  --load-pool --pool-file "$POOL" --uv-eot --tag p3e_eot \
  > "$LOGD/p3_trainA.log" 2>&1 &
pidA=$!
bash "$PI/run_cloth.sh" 3 train --suite libero_spatial --pool-tasks 0,2,6,8 \
  --pool-mode full --frame-weighting taao --epsilon $EPS --seed 7 --iters 5000 \
  --load-pool --pool-file "$POOL" --tag p3e_noeot \
  > "$LOGD/p3_trainB.log" 2>&1 &
pidB=$!
wait $pidA $pidB
echo "$(date '+%F %T') [chain-P3] 双臂训练完成" >> "$WD"

# 3) 快评：未见任务 plain + 两臂 adv
UT=1,3,4,5,7,9
run_block() {  # $1=GPU $2=name $3=cond $4=suite $5=tasks 其余 extra
  local g=$1 name=$2 cond=$3 suite=$4 tasks=$5; shift 5
  local log="$LOGD/p3_${name}_${suite}.log"
  for attempt in 1 2 3; do
    echo "$(date '+%F %T') [chain-P3] ${name} ${suite} 第${attempt}次 GPU$g" >> "$WD"
    bash "$PI/run_cloth.sh" "$g" eval --condition "$cond" --suite "$suite" \
      --tasks "$tasks" --trials 10 "$@" >> "$log" 2>&1
    rc=$?
    if [ $rc -eq 0 ] && grep -q "\[eval\] 结果" "$log"; then
      echo "$(date '+%F %T') [chain-P3] ${name} ${suite} 完成" >> "$WD"
      return 0
    fi
    echo "$(date '+%F %T') [watchdog] P3 ${name} ${suite} rc=$rc 重试" >> "$WD"
    sleep 30
  done
  echo "$(date '+%F %T') [watchdog] P3 ${name} ${suite} FAILED" >> "$WD"
}
TA=$(ls -t $TEX_A 2>/dev/null | head -1); TB=$(ls -t $TEX_B 2>/dev/null | head -1)
pids=()
run_block 0 plain30x20 plain libero_spatial "$UT" & pids+=($!)
run_block 1 advA adv libero_spatial "$UT" --texture "$TA" & pids+=($!)
run_block 3 advB adv libero_spatial "$UT" --texture "$TB" & pids+=($!)
for p in "${pids[@]}"; do wait "$p"; done

# 收尾：恢复 60x40 grid 默认资产
cp "$STAGE/_live_backup.obj" "$LIVE/tablecloth.obj" 2>/dev/null || cp "$STAGE/60x40_grid/tablecloth.obj" "$LIVE/tablecloth.obj"
cp "$STAGE/_live_backup.png" "$LIVE/cloth_base.png" 2>/dev/null || cp "$STAGE/60x40_grid/cloth_base.png" "$LIVE/cloth_base.png"
echo "$(date '+%F %T') [chain-P3] 重训+快评全部完成 DONE-P3" >> "$WD"
