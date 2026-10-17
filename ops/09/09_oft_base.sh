#!/bin/bash
# XOFT: OFT-Spatial transfer baseline with existing fsA MSE texture (advisor step 1)
#   GPU0. Waits for tex3d-oft-mj330 clone, installs mujoco 3.3.0, then:
#   original / plain / adv(fsA) x spatial 10 tasks x 10 eps, each 3-retry.
#   transfer_pi2oft eval auto-manages XML patch per condition; we only swap assets.
D=/data/shaozhen.liu
SRC=$D/envs/tex3d-oft
CLONE=$D/envs/tex3d-oft-mj330
OFT=$D/code/tex3d/openvla-oft/experiments/robot/libero
EX=$D/code/tex3d/experiments/pi0_cloth
STAGE=$D/code/tex3d-deps/cloth_variants
CLOTH_OFT=$D/code/tex3d-deps/libero/libero/libero/assets/tablecloth
TEX=$EX/train_20261016_202932_fsA_eot/Universal_Cloth.png
LOGD=$EX/xoft_logs
WD=$EX/watchdog_events.log
mkdir -p "$LOGD"
echo "$(date '+%F %T') [xoft] 链启动 master=$$（等 venv 克隆完成）" >> "$WD"

SRCN=$(find "$SRC" -name '*.so' 2>/dev/null | wc -l)
while [ "$(find "$CLONE" -name '*.so' 2>/dev/null | wc -l)" -lt "$SRCN" ]; do sleep 20; done
"$CLONE/bin/python" -m pip install -q mujoco==3.3.0 2>&1 | tail -1
MJ=$("$CLONE/bin/python" -c 'import mujoco; print(mujoco.__version__)')
[ "$MJ" = "3.3.0" ] || { echo "$(date '+%F %T') [xoft] FS venv 异常 mujoco=$MJ，退出" >> "$WD"; exit 1; }
echo "$(date '+%F %T') [xoft] venv 就绪 mujoco=$MJ" >> "$WD"

ENVV="CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled PYTHONPATH=$D/code/tex3d:$D/code/tex3d/openvla-oft"
PY=$CLONE/bin/python
cd "$D/code/tex3d-ws/g7" || exit 1

wait_gpu_free() {
  while true; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 | tr -d ' ')
    [ "$used" -lt 60000 ] && break
    echo "$(date '+%F %T') [xoft] GPU0 被占(${used}MiB)，等待" >> "$WD"
    sleep 300
  done
}

oft_run() {  # $1=name $2=condition 其余为额外参数
  local name=$1 cond=$2; shift 2
  local log="$LOGD/${name}.log"
  local attempt res
  for attempt in 1 2 3; do
    echo "$(date '+%F %T') [xoft] $name 第${attempt}次" >> "$WD"
    wait_gpu_free
    env $ENVV "$PY" "$OFT/transfer_pi2oft.py" eval --condition "$cond" \
      --tasks 0,1,2,3,4,5,6,7,8,9 --trials 10 "$@" >> "$log" 2>&1
    res=$(grep -a "\[eval\] 结果" "$log" | tail -1 | sed "s/.*结果: //" | tr -d "\r")
    if [ -n "$res" ] && [ -f "$res" ] && grep -q "overall robot success" "$res"; then
      echo "$(date '+%F %T') [xoft] $name OK -> $res" >> "$WD"
      return 0
    fi
    echo "$(date '+%F %T') [xoft] $name 第${attempt}次未完成，30s 后重试" >> "$WD"
    sleep 30
  done
  echo "$(date '+%F %T') [xoft] $name FAILED（3 次）" >> "$WD"
  return 1
}

# 资产切 30x20_solid（OFT 树独立于 pi0 树）
cp "$STAGE/30x20_solid/tablecloth.obj" "$CLOTH_OFT/tablecloth.obj"
cp "$STAGE/30x20_solid/cloth_base.png" "$CLOTH_OFT/cloth_base.png"
sync
echo "$(date '+%F %T') [xoft] OFT 树资产已切 30x20_solid" >> "$WD"

rc=0
oft_run xoft_original original || rc=1
oft_run xoft_plain plain || rc=1
oft_run xoft_adv adv --texture "$TEX" || rc=1

# 恢复 60x40 grid 常态（与 pi0 树同步的休眠态）
cp "$STAGE/60x40_grid/tablecloth.obj" "$CLOTH_OFT/tablecloth.obj"
cp "$STAGE/60x40_grid/cloth_base.png" "$CLOTH_OFT/cloth_base.png"
env $ENVV "$PY" "$OFT/transfer_pi2oft.py" patch --off >> "$WD" 2>&1
if [ "$rc" -eq 0 ]; then
  echo "$(date '+%F %T') [xoft] DONE-XOFT 基线三条件完成" >> "$WD"
else
  echo "$(date '+%F %T') [xoft] DONE-XOFT-WITH-FAILURES rc=$rc" >> "$WD"
fi
