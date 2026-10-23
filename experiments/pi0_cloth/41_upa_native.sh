#!/bin/bash
# 41_upa_native.sh — UPA-RFAS 原生基线链（第14轮P1-UPA，排队执行）
#   等待 native2d 评测波结束 → 官方默认组件训练（九月现场 trans.py，仅换数据集
#   为 libero_spatial：源=openvla-7b-finetuned-libero-spatial + RLDS，官方
#   λ_con/attn/text/featL1=1.0/1.0/0.5/0.1，patch 3×50×50，iter2000×bs2×inner50，
#   seed42（trans.py 内置），几何增广 on，扩展项全默认关）→
#   π0.5-Object/Spatial 两开发格 200（同一张 patch 主/腕各挂、位置独立采样）。
#   GPU 纪律：仅 0-3（用户规定）；训练等波结束再启动；启动前现查 nvidia-smi。
set -u
D=/data/shaozhen.liu
UPA=$D/code/UPA-RFAS
PYT=/data6/shaozhen.liu/miniforge3/envs/upa-rfas/bin/python
PI=$D/code/tex3d/pi
CKPT5=$D/model/RLinf-Pi05-LIBERO-130-fullshot-SFT
LOGD=$D/code/tex3d/experiments/edpa_native2d/eval_logs   # 波结束标记在此
UPAOUT=$D/code/tex3d/experiments/upa_native
mkdir -p "$UPAOUT" "$UPAOUT/logs"
log() { echo "[upa $(date +%H:%M:%S)] $*"; }

# —— 阶段0：等待 native2d 评测波结束（最长 5h）——
log "等待 native2d 评测波（DONE-NATIVE2D-EVAL / 16 格 cell_done）"
for i in $(seq 1 100); do
  if grep -q "DONE-NATIVE2D-EVAL" "$LOGD/summary.txt" 2>/dev/null; then log "波已收口"; break; fi
  if [ "$i" = 100 ]; then log "等待超时（5h），仍继续尝试训练（GPU 现查保护）"; fi
  sleep 180
done

# —— 阶段1：UPA 官方默认训练（GPU0，现查后启动）——
GPU=2
log "现查 GPU$GPU"
nvidia-smi --id=$GPU --query-gpu=memory.used,utilization.gpu --format=csv,noheader
cd "$UPA"
ENVV="WANDB_MODE=offline TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 PYTHONPATH=$UPA/openvla-main"
for attempt in 1 2 3; do
  env CUDA_VISIBLE_DEVICES=$GPU MUJOCO_GL=egl PYOPENGL_PLATFORM=egl $ENVV \
    "$PYT" trans.py --maskidx 0,1,2 --lr 1e-3 --step 8 --iter 2000 --accumulate 1 \
    --bs 2 --warmup 20 --patch_size 3,50,50 --geometry true --wandb_project false \
    --innerLoop 50 --dataset libero_spatial --vla_path /data/shaozhen.liu/model/openvla-7b-finetuned-libero-spatial --tag upa_native_r1 \
    > "$UPAOUT/logs/train_upa_native.log" 2>&1 && break
  log "训练第$attempt次失败，120s 后重试"; sleep 120
done
RUN_DIR=$(ls -dt "$UPA"/run/our/upa_native_r1_* 2>/dev/null | head -1)
[ -z "$RUN_DIR" ] && { log "FAIL-UPA-TRAIN（无输出目录）"; exit 1; }
log "训练完成 $RUN_DIR"
cp "$RUN_DIR/last/patch.pt" "$UPAOUT/upa_native_patch.pt" 2>/dev/null \
  || cp "$RUN_DIR/final/patch.pt" "$UPAOUT/upa_native_patch.pt"
sha256sum "$UPAOUT/upa_native_patch.pt"

# —— 阶段2：开发格 200（π0.5-Object/Spatial；同一 patch 主/腕各挂）——
cell_done() {
  local res n
  res=$(grep -a '\[eval\] 结果' "$UPAOUT/logs/$1.log" 2>/dev/null | tail -1 | sed 's/.*结果: //' | tr -d '\r')
  n=$(grep -ac '^Task:' "$res" 2>/dev/null)
  [ -n "$res" ] && [ -f "$res" ] && [ "${n:-0}" -ge 100 ]
}
run_cell() {  # $1=gpu $2=name $3=suite
  local g=$1 name=$2 suite=$3 res n
  cell_done "$name" && { log "$name 已完成"; return 0; }
  log "评测 $name GPU$g"
  for attempt in 1 2 3 4; do
    env CLOTH_QUERY_SEED=1 bash "$PI/run_patch_eval.sh" "$g" eval \
      --condition patch --tasks 0,1,2,3,4,5,6,7,8,9 --trials 10 --num-denoise 10 \
      --init-offset 10 --suite "$suite" --ckpt "$CKPT5" --model-config pi05_libero \
      --patch-primary "$UPAOUT/upa_native_patch.pt" --patch-wrist "$UPAOUT/upa_native_patch.pt" \
      >> "$UPAOUT/logs/${name}.log" 2>&1
    res=$(grep -a '\[eval\] 结果' "$UPAOUT/logs/${name}.log" 2>/dev/null | tail -1 | sed 's/.*结果: //' | tr -d '\r')
    n=$(grep -ac '^Task:' "$res" 2>/dev/null)
    if [ -n "$res" ] && [ -f "$res" ] && [ "${n:-0}" -ge 100 ]; then
      log "$name 完成 -> $(grep -a 'overall robot success' "$res" | tail -1)"; return 0
    fi
    sleep 45
  done
  log "$name 四次失败"; return 1
}
run_cell 0 upanat_pi05_obj libero_object
run_cell 0 upanat_pi05_sp libero_spatial
log "DONE-UPA-NATIVE"
