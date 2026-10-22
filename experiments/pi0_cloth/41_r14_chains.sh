#!/bin/bash
# 41_r14_chains.sh — 老师第13轮评阅(20261022) §7 执行链：P1 EDPA-TVfix + P2 MR-bs1
#   P1: EDPA-TVfix 冒烟30(GPU1) → 全训5000 seed7 batch8(GPU1) → 800 回合(8格)
#   P2: F0-TVfix-bs1-MR 冒烟30(GPU7) → 全训5000 seed7 batch1(GPU7) → π0.5-Obj/Sp 200
#   评测 11 格 = EDPA-TVfix 8 + MR-bs1 2 + 金丝雀(π0-sp-F0, 期望42±15, 不混主样本)
#   GPU 纪律：仅用 0/1/2/7（4/5/6 今日重启后 CUDA 已过但保守不用）；每卡≤2我方进程；
#   评测每隔 60s 加载一条（用户规定）；训练期间 GPU1/GPU7 各独占。
#   新评测格开启 CLOTH_QUERY_SEED=1（逐查询种子记录，默认关闭的历史行为不变）。
set -u
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
D=/data/shaozhen.liu
PI=$D/code/tex3d/pi
EX=$D/code/tex3d/experiments/pi0_cloth
LOGD=$EX/main_logs
WD=$EX/watchdog_events.log
POOL=$EX/oft_cloth/r0f_pool.pt
MRC=$EX/oft_cloth/mr_plain_cache.pt
TRD=$D/code/tex3d/openvla-oft/experiments/robot/libero
F0TEX=$EX/texpush/f0_seed7.png
CKPT5=$D/model/RLinf-Pi05-LIBERO-130-fullshot-SFT
CKPT0=$D/model/RLinf-Pi0-LIBERO-130-fullshot-SFT
PYT=$D/envs/tex3d-oft-mj330/bin/python
mkdir -p "$LOGD"
ENVV="MUJOCO_GL=egl PYOPENGL_PLATFORM=egl HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled TMPDIR=$D/tmp PYTHONPATH=$D/code/tex3d:$D/code/tex3d/openvla-oft"

log() { echo "$(date '+%F %T') [r14] $*" >> "$WD"; }

train_edpatvfix() {  # $1=gpu $2=iters $3=tag $4=log
  (cd "$D/code/tex3d-ws/g7" && env CUDA_VISIBLE_DEVICES=$1 MUJOCO_EGL_DEVICE_ID=$1 $ENVV \
    $PYT $TRD/attack_oft_cloth_edpa_tvfix.py train \
    --pool-tasks 0,2,6,8 --pool-file "$POOL" --load-pool \
    --iters "$2" --seed 7 --uv-eot --edpa-alpha 0.2 --edpa-batch 8 \
    --tag "$3") >> "$4" 2>&1
}
train_mrbs1() {  # $1=gpu $2=iters $3=tag $4=log
  (cd "$D/code/tex3d-ws/g7" && env CUDA_VISIBLE_DEVICES=$1 MUJOCO_EGL_DEVICE_ID=$1 $ENVV \
    $PYT $TRD/attack_oft_cloth_tvfix_mr.py train \
    --pool-tasks 0,2,6,8 --pool-file "$POOL" --load-pool \
    --iters "$2" --seed 7 --uv-eot --batch 1 --mr-cache "$MRC" \
    --tag "$3") >> "$4" 2>&1
}

cell_done() {
  local res n
  res=$(grep -a '\[eval\] 结果' "$LOGD/$1.log" 2>/dev/null | tail -1 | sed 's/.*结果: //' | tr -d '\r')
  n=$(grep -c '^Task:' "$res" 2>/dev/null)
  [ -n "$res" ] && [ -f "$res" ] && [ "${n:-0}" -ge 100 ]
}

run_cell() {  # $1=gpu $2=name $3=suite $4=ckpt $5=texture $6=modelcfg(optional)
  local g=$1 name=$2 suite=$3 ckpt=$4 tex=$5 mcfg=${6:-}
  if cell_done "$name"; then
    log "评测 $name 已完成，跳过"; return 0
  fi
  local log="$LOGD/${name}.log" res n extra=()
  [ -n "$mcfg" ] && extra+=(--model-config "$mcfg")
  log "评测 $name GPU$g"
  for attempt in 1 2 3 4; do
    env CLOTH_QUERY_SEED=1 bash "$PI/run_cloth330.sh" "$g" eval --condition adv --suite "$suite" \
      --tasks 0,1,2,3,4,5,6,7,8,9 --trials 10 --num-denoise 10 --init-offset 10 \
      --texture "$tex" --ckpt "$ckpt" "${extra[@]}" >> "$log" 2>&1
    res=$(grep -a '\[eval\] 结果' "$log" 2>/dev/null | tail -1 | sed 's/.*结果: //' | tr -d '\r')
    n=$(grep -c '^Task:' "$res" 2>/dev/null)
    if [ -n "$res" ] && [ -f "$res" ] && [ "${n:-0}" -ge 100 ]; then
      log "评测 $name 完成 -> $(grep -a 'overall robot success' "$res" | tail -1)"
      return 0
    fi
    sleep 45
  done
  log "评测 $name 四次失败 FAILED"; return 1
}

log "链启动：P1 EDPA-TVfix + P2 MR-bs1（服务器今日重启过，GPU4/5/6 CUDA 已验证但不用）"

# ---------- 阶段 1：冒烟 30 iter ----------
log "冒烟开始：EDPA-TVfix GPU1 / MR-bs1 GPU7（各 30 iter）"
train_edpatvfix 1 30 edpatvfix-smoke "$LOGD/train_edpatvfix_smoke.log" &
P1S=$!
train_mrbs1 7 30 tvfix-mr-bs1-smoke "$LOGD/train_mrbs1_smoke.log" &
P2S=$!
wait $P1S; wait $P2S

ESDIR=$(ls -dt $EX/oft_cloth/train_*_edpatvfix-smoke 2>/dev/null | head -1)
BSDIR=$(ls -dt $EX/oft_cloth/train_*_tvfix-mr-bs1-smoke 2>/dev/null | head -1)
if ! grep -q "it00029" "$ESDIR/gradient_log.txt" 2>/dev/null || \
   grep -qi "nan" "$ESDIR/gradient_log.txt" 2>/dev/null; then
  log "EDPA-TVfix 冒烟失败 dir=$ESDIR（见 train_edpatvfix_smoke.log）"; exit 1
fi
if ! grep -q "it00029" "$BSDIR/gradient_log.txt" 2>/dev/null || \
   grep -qi "nan" "$BSDIR/gradient_log.txt" 2>/dev/null; then
  log "MR-bs1 冒烟失败 dir=$BSDIR（见 train_mrbs1_smoke.log）"; exit 1
fi
log "冒烟通过：EDPA-TVfix $(tail -1 "$ESDIR/gradient_log.txt") | MR-bs1 $(tail -1 "$BSDIR/gradient_log.txt")"

# ---------- 阶段 2：全训 5000（并行，GPU1 / GPU7 独占） ----------
log "全训开始：EDPA-TVfix GPU1 / MR-bs1 GPU7（5000 更新 seed7）"
T0=$(date +%s)
train_edpatvfix 1 5000 edpatvfix "$LOGD/train_edpatvfix.log" &
P1=$!
train_mrbs1 7 5000 tvfix-mr-bs1 "$LOGD/train_mrbs1.log" &
P2=$!
wait $P1; RC1=$?
wait $P2; RC2=$?
log "全训结束 rc_edpatvfix=$RC1 rc_mrbs1=$RC2 用时 $(( ($(date +%s) - T0) / 60 ))min"

ETEXDIR=$(ls -dt $EX/oft_cloth/train_*_edpatvfix 2>/dev/null | grep -v smoke | head -1)
BTEXDIR=$(ls -dt $EX/oft_cloth/train_*_tvfix-mr-bs1 2>/dev/null | grep -v smoke | head -1)
ETEX="$ETEXDIR/Universal_Cloth_OV.png"
BTEX="$BTEXDIR/Universal_Cloth_OV.png"
if [ ! -f "$ETEX" ] || ! grep -q "it04999" "$ETEXDIR/gradient_log.txt" 2>/dev/null; then
  log "EDPA-TVfix 全训不完整 dir=$ETEXDIR FAILED"; exit 1
fi
if [ ! -f "$BTEX" ] || ! grep -q "it04999" "$BTEXDIR/gradient_log.txt" 2>/dev/null; then
  log "MR-bs1 全训不完整 dir=$BTEXDIR FAILED"; exit 1
fi
log "EDPA-TVfix 纹理 $ETEX sha=$(sha256sum "$ETEX" | cut -c1-16) $(grep -a '\[ckpt\] it04999' "$ETEXDIR/gradient_log.txt" | tail -1)"
log "MR-bs1 纹理 $BTEX sha=$(sha256sum "$BTEX" | cut -c1-16) $(grep -a '\[ckpt\] it04999' "$BTEXDIR/gradient_log.txt" | tail -1)"

# ---------- 阶段 3：评测 11 格（8 lane × GPU0/1/2/7，每隔 60s 加载一条） ----------
log "评测波启动：EDPA-TVfix 800(8格) + MR-bs1 200(2格) + 金丝雀，8 lane 错峰 60s"
rc=0
pids=()
# 先装开发两格 + 金丝雀（老师 §7-P1：先检查输入/训练/结果格式），其余格子无条件补齐 800
( sleep 0;   run_cell 0 mtf_edpatvfix_pi05_obj libero_object "$CKPT5" "$ETEX" pi05_libero; run_cell 0 mtf_edpatvfix_pi05_10 libero_10 "$CKPT5" "$ETEX" pi05_libero ) & pids+=($!)
( sleep 60;  run_cell 0 mtf_edpatvfix_pi05_sp libero_spatial "$CKPT5" "$ETEX" pi05_libero; run_cell 0 mtf_mrbs1_pi05_obj libero_object "$CKPT5" "$BTEX" pi05_libero ) & pids+=($!)
( sleep 120; run_cell 1 mtf_canary3_pi0_sp_f0 libero_spatial "$CKPT0" "$F0TEX"; run_cell 1 mtf_mrbs1_pi05_sp libero_spatial "$CKPT5" "$BTEX" pi05_libero ) & pids+=($!)
( sleep 180; run_cell 1 mtf_edpatvfix_pi0_sp libero_spatial "$CKPT0" "$ETEX" ) & pids+=($!)
( sleep 240; run_cell 2 mtf_edpatvfix_pi0_obj libero_object "$CKPT0" "$ETEX" ) & pids+=($!)
( sleep 300; run_cell 2 mtf_edpatvfix_pi0_go libero_goal "$CKPT0" "$ETEX" ) & pids+=($!)
( sleep 360; run_cell 7 mtf_edpatvfix_pi0_10 libero_10 "$CKPT0" "$ETEX" ) & pids+=($!)
( sleep 420; run_cell 7 mtf_edpatvfix_pi05_go libero_goal "$CKPT5" "$ETEX" pi05_libero ) & pids+=($!)
for p in "${pids[@]}"; do wait "$p" || rc=1; done

# ---------- 收尾 ----------
ok=0; sum=""
for name in mtf_edpatvfix_pi05_obj mtf_edpatvfix_pi05_sp mtf_edpatvfix_pi0_sp \
            mtf_edpatvfix_pi0_obj mtf_edpatvfix_pi0_go mtf_edpatvfix_pi0_10 \
            mtf_edpatvfix_pi05_go mtf_edpatvfix_pi05_10 \
            mtf_mrbs1_pi05_obj mtf_mrbs1_pi05_sp; do
  cell_done "$name" && ok=$((ok+1))
  res=$(grep -a '\[eval\] 结果' "$LOGD/${name}.log" 2>/dev/null | tail -1 | sed 's/.*结果: //' | tr -d '\r')
  sum="$sum$name=$(grep -a 'overall robot success' "$res" 2>/dev/null | tail -1 | grep -o '[0-9]*/100') "
done
CRES=$(grep -a '\[eval\] 结果' "$LOGD/mtf_canary3_pi0_sp_f0.log" 2>/dev/null | tail -1 | sed 's/.*结果: //' | tr -d '\r')
CS=$(grep -a 'overall robot success' "$CRES" 2>/dev/null | tail -1 | grep -o '[0-9]*/100' | cut -d/ -f1)
log "金丝雀 succ=${CS:-未出}（历史 42±15；第4次）"
if [ "$rc" -eq 0 ] && [ "$ok" -eq 10 ]; then
  log "DONE-R14-P1P2 10/10 全部完成：$sum"
else
  log "DONE-R14-P1P2-WITH-FAILURES rc=$rc 终检 $ok/10：$sum"
fi
