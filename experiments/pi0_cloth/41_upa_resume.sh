#!/bin/bash
# 41_upa_resume.sh — UPA-RFAS 官方断点续跑链（2026-10-23 服务器维修后）
#   背景：upa_native_r1_a8d5aee8 于 iter1637/2000 被 GPU 故障杀死（16:32，早于 18:14 重启），
#   最新快照 1400/patch.pt（15:17）。官方 feature_space.py 自带 resume：
#   --patch_path(PNG, Image.open) + --start_iter（循环 range(start_iter, num_iter)，调度器按剩余步数构建）。
#   决策：PNG 量化续跑（patch∈[0,1]，量化误差≤1/510，600 iters 重优化），省 ~8h vs 干净重训。
#   披露：8-bit 量化重启、优化器状态重置、数据顺序从 seed42 头部重放、LR 按剩余 600 步重排（官方机制行为）。
#   GPU 纪律：仅 0-3；训练 GPU2；启动前现查 nvidia-smi。
set -u
D=/data/shaozhen.liu
UPA=$D/code/UPA-RFAS
PYT=/data6/shaozhen.liu/miniforge3/envs/upa-rfas/bin/python
PI=$D/code/tex3d/pi
CKPT5=$D/model/RLinf-Pi05-LIBERO-130-fullshot-SFT
UPAOUT=$D/code/tex3d/experiments/upa_native
SNAP=$UPA/run/our/upa_native_r1_a8d5aee8-43bf-49b7-aba2-998e1f5cbdad/1400/patch.pt
mkdir -p "$UPAOUT" "$UPAOUT/logs"
log() { echo "[upares $(date +%m%d_%H:%M:%S)] $*"; }

# —— 阶段1：快照转 PNG（官方 loader 仅吃 PNG）——
"$PYT" - <<'EOF'
import torch, numpy as np
from PIL import Image
src = "/data/shaozhen.liu/code/UPA-RFAS/run/our/upa_native_r1_a8d5aee8-43bf-49b7-aba2-998e1f5cbdad/1400/patch.pt"
dst = "/data/shaozhen.liu/code/tex3d/experiments/upa_native/resume_iter1400.png"
p = torch.load(src, map_location="cpu")
assert p.min() >= 0.0 and p.max() <= 1.0, (p.min().item(), p.max().item())
arr = (p.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
Image.fromarray(arr).save(dst)
q = torch.from_numpy(np.array(Image.open(dst)).astype(np.float32) / 255).permute(2, 0, 1)
print(f"[png] quant_max_err={(q - p).abs().max().item():.6f} shape={tuple(p.shape)}")
EOF
[ -f "$UPAOUT/resume_iter1400.png" ] || { log "FAIL-PNG-CONVERT"; exit 1; }

# —— 阶段2：官方 resume 训练（GPU2，剩余 1400→2000 共 600 iters）——
GPU=2
log "现查 GPU$GPU"
nvidia-smi --id=$GPU --query-gpu=memory.used,utilization.gpu --format=csv,noheader
cd "$UPA"
ENVV="WANDB_MODE=offline TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 PYTHONPATH=$UPA/openvla-main"
for attempt in 1 2 3; do
  env CUDA_VISIBLE_DEVICES=$GPU MUJOCO_GL=egl PYOPENGL_PLATFORM=egl $ENVV \
    "$PYT" trans.py --maskidx 0,1,2 --lr 1e-3 --step 8 --iter 2000 --accumulate 1 \
    --bs 2 --warmup 20 --patch_size 3,50,50 --geometry true --wandb_project false \
    --innerLoop 50 --dataset libero_spatial --vla_path $D/model/openvla-7b-finetuned-libero-spatial \
    --tag upa_resume_r1 --patch_path "$UPAOUT/resume_iter1400.png" --start_iter 1400 \
    > "$UPAOUT/logs/train_upa_resume.log" 2>&1 && break
  log "续跑第$attempt次失败，120s 后重试"; sleep 120
done
grep -a "\[resume\] loaded patch" "$UPAOUT/logs/train_upa_resume.log" | head -1 || { log "FAIL-RESUME-NOT-ENGAGED"; exit 1; }
tr -d '\r' < "$UPAOUT/logs/train_upa_resume.log" | grep -aq "Attack done" || { log "FAIL-UPA-RESUME-INCOMPLETE"; exit 1; }
RUN_DIR=$(ls -dt "$UPA"/run/our/upa_resume_r1_* 2>/dev/null | head -1)
[ -z "$RUN_DIR" ] && { log "FAIL-UPA-RESUME（无输出目录）"; exit 1; }
log "续跑完成 $RUN_DIR"
cp "$RUN_DIR/last/patch.pt" "$UPAOUT/upa_native_patch.pt" 2>/dev/null \
  || cp "$RUN_DIR/final/patch.pt" "$UPAOUT/upa_native_patch.pt"
[ -f "$UPAOUT/upa_native_patch.pt" ] || { log "FAIL-NO-FINAL-PATCH"; exit 1; }
sha256sum "$UPAOUT/upa_native_patch.pt"

# —— 阶段3：开发格 200（π0.5-Object/Spatial；同一 patch 主/腕各挂）——
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
    res=$(grep -a '\[eval\] 结果' "$UPAOUT/logs/$name.log" 2>/dev/null | tail -1 | sed 's/.*结果: //' | tr -d '\r')
    n=$(grep -ac '^Task:' "$res" 2>/dev/null)
    if [ -n "$res" ] && [ -f "$res" ] && [ "${n:-0}" -ge 100 ]; then
      log "$name 完成 -> $(grep -a 'overall robot success' "$res" | tail -1)"; return 0
    fi
    sleep 45
  done
  log "$name 四次失败"; return 1
}
run_cell 2 upanat_pi05_obj libero_object
run_cell 2 upanat_pi05_sp libero_spatial
log "DONE-UPA-RESUME"
