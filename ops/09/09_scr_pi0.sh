#!/bin/bash
# SCR: pi0-side carrier size screening on fullshot backbone (advisor step 2)
#   GPU1, mujoco 3.3.0. For each size in {45x30, 60x40}:
#     swap pi0-tree assets -> plain eval (spatial 100 eps) -> fsA stretch probe (adv 100 eps)
#   30x20 plain already known (92%). Restores 60x40 grid at end.
D=/data/shaozhen.liu
PI=$D/code/tex3d/pi
EX=$D/code/tex3d/experiments/pi0_cloth
STAGE=$D/code/tex3d-deps/cloth_variants
LIVE=$D/code/tex3d-deps/libero-pi/libero/libero/assets/tablecloth
CKPT=$D/model/RLinf-Pi0-LIBERO-130-fullshot-SFT
TEX=$EX/train_20261016_202932_fsA_eot/Universal_Cloth.png
LOGD=$EX/scr_logs
WD=$EX/watchdog_events.log
mkdir -p "$LOGD"
echo "$(date '+%F %T') [scr] 链启动 master=$$（pi0 尺寸粗筛 GPU1）" >> "$WD"

wait_gpu_free() {
  while true; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1 | tr -d ' ')
    [ "$used" -lt 60000 ] && break
    echo "$(date '+%F %T') [scr] GPU1 被占(${used}MiB)，等待" >> "$WD"
    sleep 300
  done
}

run_block() {  # $1=name 其余为 eval 参数
  local name=$1; shift
  local log="$LOGD/${name}.log"
  local attempt res
  for attempt in 1 2 3; do
    echo "$(date '+%F %T') [scr] $name 第${attempt}次 GPU1" >> "$WD"
    wait_gpu_free
    bash "$PI/run_cloth330.sh" 1 eval --ckpt "$CKPT" "$@" >> "$log" 2>&1
    res=$(grep -a "\[eval\] 结果" "$log" | tail -1 | sed "s/.*结果: //" | tr -d "\r")
    if [ -n "$res" ] && [ -f "$res" ] && grep -q "overall robot success" "$res"; then
      echo "$(date '+%F %T') [scr] $name OK -> $res" >> "$WD"
      return 0
    fi
    echo "$(date '+%F %T') [scr] $name 第${attempt}次未完成，30s 后重试" >> "$WD"
    sleep 30
  done
  echo "$(date '+%F %T') [scr] $name FAILED（3 次）" >> "$WD"
  return 1
}

rc=0
for SIZE in 45x30 60x40; do
  cp "$STAGE/${SIZE}_solid/tablecloth.obj" "$LIVE/tablecloth.obj"
  cp "$STAGE/${SIZE}_solid/cloth_base.png" "$LIVE/cloth_base.png"
  cd "$PI" && bash run_cloth330.sh 1 patch --on >> "$WD" 2>&1
  sync
  echo "$(date '+%F %T') [scr] 资产已切 ${SIZE}_solid" >> "$WD"
  run_block scr_plain_${SIZE} --condition plain --suite libero_spatial \
    --tasks 0,1,2,3,4,5,6,7,8,9 --trials 10 || rc=1
  run_block scr_probe_${SIZE} --condition adv --suite libero_spatial \
    --tasks 0,1,2,3,4,5,6,7,8,9 --trials 10 --texture "$TEX" || rc=1
done

cp "$STAGE/60x40_grid/tablecloth.obj" "$LIVE/tablecloth.obj"
cp "$STAGE/60x40_grid/cloth_base.png" "$LIVE/cloth_base.png"
if [ "$rc" -eq 0 ]; then
  echo "$(date '+%F %T') [scr] DONE-SCR pi0 尺寸粗筛完成" >> "$WD"
else
  echo "$(date '+%F %T') [scr] DONE-SCR-WITH-FAILURES rc=$rc" >> "$WD"
fi
