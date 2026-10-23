#!/bin/bash
# 41_tex3d_native.sh — Tex3D 原生对象 3D 纹理链（第14轮P2，排队：等 native2d 波结束）
#   零扰动核验+训练（GPU1，上游默认 5000×批10，~4h）→ 
#   共同子集评测：{objtex(Adv), objtex0(θ0对照)} × {π0,π0.5} × Spatial{0,2,6,8}×10初态 = 160 回合
#   GPU 纪律：仅 0-3（用户规定）；训练等波结束；启动前现查 nvidia-smi。
set -u
D=/data/shaozhen.liu
TRD=$D/code/tex3d/openvla-oft/experiments/robot/libero
PYT=$D/envs/tex3d-oft-mj330/bin/python
PYP=$D/envs/tex3d-pi0-mj330/bin/python
PI=$D/code/tex3d/pi
OUT=$D/code/tex3d/experiments/tex3d_native
LOGD=$D/code/tex3d/experiments/edpa_native2d/eval_logs
CKPT0=$D/model/RLinf-Pi0-LIBERO-130-fullshot-SFT
CKPT5=$D/model/RLinf-Pi05-LIBERO-130-fullshot-SFT
mkdir -p "$OUT/logs"
log() { echo "[t3d $(date +%H:%M:%S)] $*"; }
ENVV="MUJOCO_GL=egl PYOPENGL_PLATFORM=egl HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled TMPDIR=$D/tmp PYTHONPATH=$D/code/tex3d:$D/code/tex3d/openvla-oft"

# 阶段0：等波
for i in $(seq 1 100); do
  grep -q "DONE-NATIVE2D-EVAL" "$LOGD/summary.txt" 2>/dev/null && { log "波已收口"; break; }
  [ "$i" = 100 ] && log "等待超时，继续（GPU 现查保护）"
  sleep 180
done

# 阶段1：零扰动核验 + 训练（GPU1）
GPU=3
nvidia-smi --id=$GPU --query-gpu=memory.used,utilization.gpu --format=csv,noheader
cd $D/code/tex3d-ws/g7
for attempt in 1 2 3; do
  env CUDA_VISIBLE_DEVICES=$GPU MUJOCO_EGL_DEVICE_ID=$GPU $ENVV \
    "$PYT" "$TRD/attack_oft_tex3d_native.py" --iters 5000 --seed 7 --tag r1 \
    > "$OUT/logs/train_tex3d_native.log" 2>&1 && break
  log "训练第$attempt次失败，120s 后重试"; sleep 120
done
TDIR=$(ls -dt "$OUT"/train_*_r1 2>/dev/null | head -1)
[ -z "$TDIR" ] && { log "FAIL-T3D-TRAIN"; exit 1; }
ADVPNG="$TDIR/akita_bowl_adv_baked.png"
TH0PNG="$TDIR/theta0_baked.png"
log "训练完成 $TDIR"; grep -E "sha256|零扰动|delta" "$TDIR/run_config.txt" || true

# 阶段2：共同子集 160 回合（4 格，GPU1）
cell_done() {
  local res n
  res=$(grep -a '\[eval\] 结果' "$OUT/logs/$1.log" 2>/dev/null | tail -1 | sed 's/.*结果: //' | tr -d '\r')
  n=$(grep -ac '^Task:' "$res" 2>/dev/null)
  [ -n "$res" ] && [ -f "$res" ] && [ "${n:-0}" -ge 40 ]
}
run_cell() {  # $1=name $2=cond $3=texture $4=ckpt $5=modelcfg
  local name=$1 cond=$2 tex=$3 ckpt=$4 mcfg=${5:-} res n extra=()
  [ -n "$mcfg" ] && extra+=(--model-config "$mcfg")
  cell_done "$name" && { log "$name 已完成"; return 0; }
  log "评测 $name"
  for attempt in 1 2 3 4; do
    env CLOTH_QUERY_SEED=1 bash "$PI/run_objtex_eval.sh" "$GPU" eval \
      --condition "$cond" --tasks 0,2,6,8 --trials 10 --num-denoise 10 \
      --init-offset 10 --suite libero_spatial --texture "$tex" --ckpt "$ckpt" \
      "${extra[@]}" >> "$OUT/logs/${name}.log" 2>&1
    res=$(grep -a '\[eval\] 结果' "$OUT/logs/${name}.log" 2>/dev/null | tail -1 | sed 's/.*结果: //' | tr -d '\r')
    n=$(grep -ac '^Task:' "$res" 2>/dev/null)
    if [ -n "$res" ] && [ -f "$res" ] && [ "${n:-0}" -ge 40 ]; then
      log "$name 完成 -> $(grep -a 'overall robot success' "$res" | tail -1)"; return 0
    fi
    sleep 45
  done
  log "$name 四次失败"; return 1
}
run_cell t3d_adv_pi0  objtex  "$ADVPNG" "$CKPT0"
run_cell t3d_th0_pi0  objtex0 "$TH0PNG" "$CKPT0"
run_cell t3d_adv_pi05 objtex  "$ADVPNG" "$CKPT5" pi05_libero
run_cell t3d_th0_pi05 objtex0 "$TH0PNG" "$CKPT5" pi05_libero
log "DONE-TEX3D-NATIVE"
