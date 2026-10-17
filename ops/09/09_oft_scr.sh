#!/bin/bash
# OFT-SCR: OFT-side plain screening at 45x30 / 60x40 (GPU0, mujoco 3.3.0 clone)
D=/data/shaozhen.liu
OFT=$D/code/tex3d/openvla-oft/experiments/robot/libero
EX=$D/code/tex3d/experiments/pi0_cloth
STAGE=$D/code/tex3d-deps/cloth_variants
CLOTH_OFT=$D/code/tex3d-deps/libero/libero/libero/assets/tablecloth
LOGD=$EX/xoft_logs
WD=$EX/watchdog_events.log
mkdir -p "$LOGD"
echo "$(date '+%F %T') [oftscr] 链启动 master=$$" >> "$WD"

ENVV="CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled PYTHONPATH=$D/code/tex3d:$D/code/tex3d/openvla-oft"
PY=$D/envs/tex3d-oft-mj330/bin/python
cd "$D/code/tex3d-ws/g7" || exit 1

wait_gpu_free() {
  while true; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 | tr -d ' ')
    [ "$used" -lt 60000 ] && break
    echo "$(date '+%F %T') [oftscr] GPU0 被占(${used}MiB)，等待" >> "$WD"
    sleep 300
  done
}

oft_run() {
  local name=$1; shift
  local log="$LOGD/${name}.log"
  local attempt res
  for attempt in 1 2 3; do
    echo "$(date '+%F %T') [oftscr] $name 第${attempt}次" >> "$WD"
    wait_gpu_free
    env $ENVV "$PY" "$OFT/transfer_pi2oft.py" eval --condition plain \
      --tasks 0,1,2,3,4,5,6,7,8,9 --trials 10 "$@" >> "$log" 2>&1
    res=$(grep -a "\[eval\] 结果" "$log" | tail -1 | sed "s/.*结果: //" | tr -d "\r")
    if [ -n "$res" ] && [ -f "$res" ] && grep -q "overall robot success" "$res"; then
      echo "$(date '+%F %T') [oftscr] $name OK -> $res" >> "$WD"
      return 0
    fi
    echo "$(date '+%F %T') [oftscr] $name 第${attempt}次未完成，重试" >> "$WD"
    sleep 30
  done
  echo "$(date '+%F %T') [oftscr] $name FAILED" >> "$WD"
  return 1
}

rc=0
for SIZE in 45x30 60x40; do
  cp "$STAGE/${SIZE}_solid/tablecloth.obj" "$CLOTH_OFT/tablecloth.obj"
  cp "$STAGE/${SIZE}_solid/cloth_base.png" "$CLOTH_OFT/cloth_base.png"
  sync
  echo "$(date '+%F %T') [oftscr] OFT 树资产已切 ${SIZE}_solid" >> "$WD"
  oft_run oftscr_plain_${SIZE} || rc=1
done

cp "$STAGE/60x40_grid/tablecloth.obj" "$CLOTH_OFT/tablecloth.obj"
cp "$STAGE/60x40_grid/cloth_base.png" "$CLOTH_OFT/cloth_base.png"
if [ "$rc" -eq 0 ]; then
  echo "$(date '+%F %T') [oftscr] DONE-OFTSCR 完成" >> "$WD"
else
  echo "$(date '+%F %T') [oftscr] DONE-OFTSCR-WITH-FAILURES rc=$rc" >> "$WD"
fi
