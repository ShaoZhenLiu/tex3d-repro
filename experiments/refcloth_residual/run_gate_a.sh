#!/bin/bash
# run_gate_a.sh — residual_lab 阶段一/二执行链（41）
#   纹理表 → capture fit/validation/locked_test → 基线回归（matched 30it + ablate 参照）
# GPU 认领：r18 flock 协议同款；本队规则：GPU4 永不、GPU6 不启用、
#   GPU0 仅 free≥22000；每卡我方进程≤2。
set -u
D=/data/shaozhen.liu
REPO=$D/code/tex3d
RL=$REPO/openvla-oft/experiments/robot/libero/residual_lab
CFG=$REPO/configs/refcloth_residual/experiment.yaml
CLAIMD=$REPO/experiments/r18_e01e02/gpu_claims
LOCK=$REPO/experiments/r18_e01e02/gpu_admit.lock
PY=$D/envs/tex3d-oft-mj330/bin/python
NEED=${NEED_MI:-28000}
NAME=resid

mkdir -p "$CLAIMD"
ENVB="OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PYTHONPATH=$REPO:$REPO/openvla-oft:$REPO/openvla-oft/experiments/robot/libero"

gpu_used(){ nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$1" 2>/dev/null | tr -d ' '; }

try_claim(){
  local g used sum cnt f pid ned
  (
    flock 9
    for g in 1 3 5 7 0; do          # 4/6 永不（管理员/用户规则）
      [ "$g" = "0" ] && { u=$(gpu_used 0); [ "$u" -le 27140 ] || continue; }
      sum=0; cnt=0
      for f in "$CLAIMD"/*."$g"; do
        [ -e "$f" ] || continue
        read -r pid ned < "$f" 2>/dev/null || { rm -f "$f"; continue; }
        if [ -z "$pid" ] || ! kill -0 "$pid" 2>/dev/null; then rm -f "$f"; continue; fi
        sum=$((sum + ${ned:-17000})); cnt=$((cnt + 1))
      done
      [ "$cnt" -ge 2 ] && continue
      used=$(gpu_used "$g"); [ -z "$used" ] && continue
      if [ $((used + sum + NEED)) -le 49140 ]; then
        echo "$$ $NEED" > "$CLAIMD/${NAME}.$g"; echo "$g"; exit 0
      fi
    done
    exit 1
  ) 9>"$LOCK"
}

wait_claim(){
  local tries=0 g
  while true; do
    if g=$(try_claim); then echo "$g"; return 0; fi
    tries=$((tries+1)); echo "[claim] 等GPU（第${tries}次）" >&2
    [ "$tries" -ge 240 ] && { echo ""; return 1; }   # 8h 放弃
    sleep 120
  done
}

G=$(wait_claim)
[ -z "$G" ] && { echo "无可用GPU，退出"; exit 1; }
echo "[run] 认领 GPU$G (need=$NEED)"
trap 'rm -f "$CLAIMD/${NAME}.$G"' EXIT

cd $D/code/tex3d-ws/g7
run(){ echo "[run] $*"; env $ENVB CUDA_VISIBLE_DEVICES=$G MUJOCO_EGL_DEVICE_ID=$G "$@"; }

RC=0
run $PY -c "from residual_lab.adapters import ResidualCtx; ctx=ResidualCtx(); print('[smoke] ctx ok, adv_noise', tuple(ctx.renderer.adv_noise.shape), 'm=229376 expected'); ctx.cleanup()" || { echo "[smoke] FAILED"; exit 1; }
run $PY -m residual_lab.cli textures  --config "$CFG" || RC=1
run $PY -m residual_lab.cli capture   --config "$CFG" --split fit          || RC=1
run $PY -m residual_lab.cli capture   --config "$CFG" --split validation  || RC=1
run $PY -m residual_lab.cli capture   --config "$CFG" --split locked_test || RC=1
if [ "$RC" = "0" ]; then
  run $PY -m residual_lab.cli regress --config "$CFG" --iters 30 || RC=1
fi
echo "[run] 完成 RC=$RC（GPU$G 释放）"
exit $RC
