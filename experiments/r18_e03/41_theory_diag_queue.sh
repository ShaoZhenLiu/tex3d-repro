#!/bin/bash
# 41_theory_diag_queue.sh — RefCloth 理论验证实验队列（老师理论稿 §六 五项）
#   v2：接入 r18 共享 GPU 认领协议（flock + CLAIMD/<name>.<gpu>，与评测波同规则，
#   消除与动态认领方的竞态 OOM）。GPU4 永不使用。
#   任务序：t2 → t5 → t4 → traj(重放 MR8 流 1500 iters) → t3 → t1
set -u
D=/data/shaozhen.liu
LIB=$D/code/tex3d/openvla-oft/experiments/robot/libero
LOGD=$D/code/tex3d/experiments/pi0_cloth/theory_queue_logs
MR8=$D/code/tex3d/experiments/pi0_cloth/oft_cloth/train_20261021_224922_tvfix-mr
CLAIMD=$D/code/tex3d/experiments/r18_e01e02/gpu_claims
LOCK=$D/code/tex3d/experiments/r18_e01e02/gpu_admit.lock
PY=$D/envs/tex3d-oft-mj330/bin/python
mkdir -p "$LOGD" "$CLAIMD"
ENVBASE="OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled
PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PYTHONPATH=$D/code/tex3d:$D/code/tex3d/openvla-oft"

log() { echo "$(date '+%F %T') [tq] $*" >> "$LOGD/queue.log"; }
gpu_used() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$1" 2>/dev/null | tr -d ' '; }
gpu_uuid() { nvidia-smi --query-gpu=uuid --format=csv,noheader -i "$1" 2>/dev/null | tr -d ' '; }
loaded_on() {  # $1=gpu index → 该卡上已加载(≥8000MiB)的进程数（其占用已计入 used）
  nvidia-smi --query-compute-apps=gpu_uuid,used_memory --format=csv,noheader,nounits 2>/dev/null \
    | awk -F', *' -v u="$1" '$1==u && $2>=8000 {n++} END{print n+0}'
}
release_claim() { rm -f "$CLAIMD/$1".* 2>/dev/null; }

# 单次尝试认领：满足（used + 未加载认领需求 + 我的 need ≤ 49140 且该卡认领数<2）
# 则写 claim 文件并输出 GPU 号；失败输出空。与评测波同锁同格式。
# 已加载进程（该卡 ≥8G 的进程）按每个抵扣 16000 认领需求——其显存已在 used 里，
# 不重复计费（否则永远进不了有评测道在跑的卡）。
try_claim() {  # $1=name $2=need_mi
  local name=$1 need=$2 g used sum cnt eff nl f pid ned
  (
    flock 9
    for g in 1 2 5 6 0 3 7; do
      sum=0; cnt=0
      for f in "$CLAIMD"/*."$g"; do
        [ -e "$f" ] || continue
        read -r pid ned < "$f" 2>/dev/null || { rm -f "$f"; continue; }
        if [ -z "$pid" ] || ! kill -0 "$pid" 2>/dev/null; then rm -f "$f"; continue; fi
        [ "$pid" = "$$" ] && continue          # 自己的保留标记不计（防自锁）
        sum=$((sum + ${ned:-17000})); cnt=$((cnt + 1))
      done
      [ "$cnt" -ge 2 ] && continue
      used=$(gpu_used "$g")
      [ -z "$used" ] && continue
      nl=$(loaded_on "$(gpu_uuid "$g")")
      eff=$((sum - nl * 16000)); [ "$eff" -lt 0 ] && eff=0
      if [ $((used + eff + need)) -le 49140 ]; then
        echo "$$ $need" > "$CLAIMD/${name}.$g"
        echo "$g"
        exit 0
      fi
    done
    exit 1
  ) 9>"$LOCK"
}

wait_run() {  # $1=name $2=need_mi $3=logfile；其余为命令
  local name=$1 need=$2 lf=$3; shift 3
  local g rc tries=0
  while true; do
    if g=$(try_claim "$name" "$need"); then
      log "$name 认领 GPU$g（need=${need}）"
      ( cd $D/code/tex3d-ws/g7 && env $ENVBASE CUDA_VISIBLE_DEVICES=$g \
          MUJOCO_EGL_DEVICE_ID=$g "$@" ) >> "$lf" 2>&1
      rc=$?
      release_claim "$name"
      log "$name 结束 rc=$rc"
      [ $rc -eq 0 ] && return 0
      tries=$((tries + 1))
      log "$name 失败（第 $tries 次）"
      [ $tries -ge 6 ] && { log "$name 六次失败放弃"; return 1; }
      sleep 300
    else
      log "$name 等 GPU（need=${need}）"
      sleep 120
    fi
  done
}

# ---- 1. T2 梯度对齐（含 ρ_flip） ----
wait_run t2 31000 "$LOGD/t2.log" $PY $LIB/refcloth_theory_diag.py \
  t2_gradalign --batches 20 --batch 8 --seed 1234

# ---- 2. T5 任务梯度对齐 ----
wait_run t5 31000 "$LOGD/t5.log" $PY $LIB/refcloth_theory_diag.py \
  t5_task --frames-per-task 48 --uv-draws 3

# ---- 3. T4 batch 方差分解 ----
wait_run t4 31000 "$LOGD/t4.log" $PY $LIB/refcloth_theory_diag.py \
  t4_var --reps 24 --batch 8 --decomp-uvs 6 --decomp-reps 8 --seed 1234

# ---- 4. T3 轨迹生成：重放 MR8 流前 1500 iters（生产循环 + 步数 ckpt） ----
wait_run traj 36000 "$LOGD/traj.log" $PY $LIB/attack_oft_cloth_ablate.py train \
  --tag theory-traj --ref mr --stream replay \
  --stream-file "$MR8/sampling_log.txt" --uv on \
  --iters 1500 --batch 8 --seed 7 --ckpt-step-suffix --checkpoint-every 125
TRAJ=$(ls -dt $D/code/tex3d/experiments/pi0_cloth/oft_cloth/train_*_theory-traj 2>/dev/null | head -1)
log "traj 目录：$TRAJ"

# ---- 5. T3 单步收益 ----
if [ -n "$TRAJ" ]; then
  wait_run t3 31000 "$LOGD/t3.log" $PY $LIB/refcloth_theory_diag.py \
    t3_onestep --traj-dir "$TRAJ" --diag-seed 777
else
  log "t3 跳过：无轨迹目录"
fi

# ---- 6. T1 渲染偏差抵消 ----
wait_run t1 31000 "$LOGD/t1.log" $PY $LIB/refcloth_theory_diag.py \
  t1_bias --states-per-task 8 --t1-stride 10

log "THEORY-DIAG-ALL-DONE"
