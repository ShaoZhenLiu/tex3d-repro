#!/bin/bash
# 41_t3d_task8.sh — task8 重训 + 逐任务粒度评测（2026-10-23 维修后）
#   背景：task8 首跑 14:37 在 GPU5 上 CUDA 崩溃（3677/5000，无 checkpoint）；
#   服务器 18:14 维修重启；用户指示此后仅用 GPU0-3。
#   GPU0 现查：qwen-tts 4.5G + dl 6.5G ≈ 11.1G 已用，T3D 训练需 ~35.6G → 可容纳（余 ~2.3G）。
#   训完 GPU0 自动接 t3dp_t8_{pi0,pi05} 两格（各 10 回合）。
set -u
D=/data/shaozhen.liu
TRD=$D/code/tex3d/openvla-oft/experiments/robot/libero
PYT=$D/envs/tex3d-oft-mj330/bin/python
PYP=$D/envs/tex3d-pi0-mj330/bin/python
PI=$D/code/tex3d/pi
OUT=$D/code/tex3d/experiments/tex3d_native
CKPT0=$D/model/RLinf-Pi0-LIBERO-130-fullshot-SFT
CKPT5=$D/model/RLinf-Pi05-LIBERO-130-fullshot-SFT
mkdir -p "$OUT/logs"
log() { echo "[t3dt8 $(date +%m%d_%H:%M:%S)] $*"; }
ENVV="MUJOCO_GL=egl PYOPENGL_PLATFORM=egl HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled TMPDIR=$D/tmp PYTHONPATH=$D/code/tex3d:$D/code/tex3d/openvla-oft"

# —— 阶段1：task8 重训（GPU0，5000×批10，~6.5h 无争用）——
GPU=0
log "现查 GPU$GPU"
nvidia-smi --id=$GPU --query-gpu=memory.used,utilization.gpu --format=csv,noheader
cd $D/code/tex3d-ws/g7
for attempt in 1 2 3; do
  env CUDA_VISIBLE_DEVICES=$GPU MUJOCO_EGL_DEVICE_ID=$GPU $ENVV \
    "$PYT" "$TRD/attack_oft_tex3d_native.py" --iters 5000 --seed 7 --task-id 8 --tag task8_r2 \
    > "$OUT/logs/train_t8_r2.log" 2>&1 && break
  log "task8 第$attempt次失败，120s 后重试"; sleep 120
done
TDIR=$(ls -dt "$OUT"/train_*_task8_r2 2>/dev/null | head -1)
[ -z "$TDIR" ] && { log "FAIL-T8-TRAIN"; exit 1; }
ADVPNG="$TDIR/akita_bowl_adv_baked.png"
[ -f "$ADVPNG" ] || { log "FAIL-T8-NO-BAKED-PNG ($TDIR)"; exit 1; }
log "task8 训练完成 $TDIR"
sha256sum "$ADVPNG" "$TDIR/adv_noise.pt"

# —— 阶段2：逐任务粒度 task8 两格（GPU0 已腾空）——
cell_done() {
  local res n
  res=$(grep -a '\[eval\] 结果' "$OUT/logs/$1.log" 2>/dev/null | tail -1 | sed 's/.*结果: //' | tr -d '\r')
  n=$(grep -ac '^Task:' "$res" 2>/dev/null)
  [ -n "$res" ] && [ -f "$res" ] && [ "${n:-0}" -ge 10 ]
}
run_cell() {  # $1=name $2=ckpt $3=modelcfg
  local name=$1 ckpt=$2 mcfg=${3:-} res n extra=()
  [ -n "$mcfg" ] && extra+=(--model-config "$mcfg")
  cell_done "$name" && { log "$name 已完成"; return 0; }
  log "评测 $name GPU$GPU"
  for attempt in 1 2 3 4; do
    env CLOTH_QUERY_SEED=1 bash "$PI/run_objtex_eval.sh" "$GPU" eval \
      --condition objtex --tasks 8 --trials 10 --num-denoise 10 \
      --init-offset 10 --suite libero_spatial --texture "$ADVPNG" --ckpt "$ckpt" \
      "${extra[@]}" >> "$OUT/logs/${name}.log" 2>&1
    res=$(grep -a '\[eval\] 结果' "$OUT/logs/$name.log" 2>/dev/null | tail -1 | sed 's/.*结果: //' | tr -d '\r')
    n=$(grep -ac '^Task:' "$res" 2>/dev/null)
    if [ -n "$res" ] && [ -f "$res" ] && [ "${n:-0}" -ge 10 ]; then
      log "$name 完成 -> $(grep -a 'overall robot success' "$res" | tail -1)"; return 0
    fi
    sleep 45
  done
  log "$name 四次失败"; return 1
}
run_cell t3dp_t8_pi0  "$CKPT0"
run_cell t3dp_t8_pi05 "$CKPT5" pi05_libero
log "DONE-T3D-TASK8"
