#!/bin/bash
# 41_r18bc_eval_wave.sh — r18批次B/C 评测总波（E03 8格 + E04 2×8格 + E05 4×2格 + E07 4×2格 = 40格 4000回合）
#   每格 = 100 回合（tasks 0-9 显式 × trials 10、初态 10-19、num_denoise 10、
#   max_steps 400、MuJoCo3.3.0、独立查询噪声）。
#   每格先等对应纹理训练终点标记（gradient_log 含 it<last> + 规范名 PNG），
#   再经共享 flock 准入取卡（need=17000，π 实测稳态≈15-16.4G；每卡≤2 我方进程，
#   GPU4 永不使用），重试×12。
set -u
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
D=/data/shaozhen.liu
PI=$D/code/tex3d/pi
EX=$D/code/tex3d/experiments/r18_e03
LOGD=$EX/evalbc_logs
CLAIMD=$D/code/tex3d/experiments/r18_e01e02/gpu_claims
LOCK=$D/code/tex3d/experiments/r18_e01e02/gpu_admit.lock
WD=$D/code/tex3d/experiments/pi0_cloth/watchdog_events.log
OFTC=$D/code/tex3d/experiments/pi0_cloth/oft_cloth
CKPT0=$D/model/RLinf-Pi0-LIBERO-130-fullshot-SFT
CKPT5=$D/model/RLinf-Pi05-LIBERO-130-fullshot-SFT
mkdir -p "$LOGD" "$CLAIMD"
log() { echo "$(date '+%F %T') [evalbc] $*" | tee -a "$WD"; }

gpu_used_mi() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$1" 2>/dev/null | tr -d ' '; }
release_claim() { rm -f "$CLAIMD/$1".* 2>/dev/null; }
pick_gpu() {
  local name=$1 lane_pid=$2 tries=0 g used free cnt f pid ned sum_need out
  while [ $tries -lt 2880 ]; do
    out=$(
      (
        flock 9
        for g in 6 5 0; do
          sum_need=0; cnt=0
          for f in "$CLAIMD"/*."$g"; do
            [ -e "$f" ] || continue
            read -r pid ned < "$f" 2>/dev/null || { rm -f "$f"; continue; }
            if [ -z "$pid" ] || ! kill -0 "$pid" 2>/dev/null; then rm -f "$f"; continue; fi
            sum_need=$((sum_need + ${ned:-17000})); cnt=$((cnt + 1))
          done
          [ "$cnt" -ge 2 ] && continue
          used=$(gpu_used_mi "$g")
          [ -z "$used" ] && continue
          free=$((49140 - used))
          if [ $((sum_need + 17000)) -le "$free" ]; then
            echo "$lane_pid 17000" > "$CLAIMD/${name}.$g"
            echo "$g"
            exit 0
          fi
        done
        exit 1
      ) 9>"$LOCK"
    ) && { echo "$out"; return 0; }
    sleep 15; tries=$((tries + 1))
  done
  return 1
}
cell_done() {
  local res n
  res=$(grep -a '\[eval\] 结果' "$LOGD/$1.log" 2>/dev/null | tail -1 | sed 's/.*结果: //' | tr -d '\r')
  n=$(grep -ac '^Task:' "$res" 2>/dev/null)
  [ -n "$res" ] && [ -f "$res" ] && [ "${n:-0}" -ge 100 ]
}
tex_ready() {  # $1=train tag $2=last marker (it04999/it00624/it39999)
  local d
  d=$(ls -dt $OFTC/train_*_"$1" 2>/dev/null | grep -v smoke | head -1)
  [ -n "$d" ] && [ -f "$d/Universal_Cloth_OV.png" ] && \
    grep -q "$2" "$d/gradient_log.txt" 2>/dev/null && echo "$d"
}
run_cell() {  # $1=name $2=tag $3=lastmark $4=suite $5=ckpt $6=modelcfg
  local name=$1 tag=$2 lastmark=$3 suite=$4 ckpt=$5 mcfg=${6:-}
  local res n g extra=() mypid=$BASHPID d t=0
  if cell_done "$name"; then log "评测 $name 已完成，跳过"; return 0; fi
  [ -n "$mcfg" ] && extra=(--model-config "$mcfg")
  d=""
  while [ $t -lt 720 ]; do
    d=$(tex_ready "$tag" "$lastmark")
    [ -n "$d" ] && break
    sleep 60; t=$((t + 1))
  done
  if [ -z "$d" ]; then log "评测 $name 等待纹理 $tag 超时 FAILED"; return 1; fi
  local TEX="$d/Universal_Cloth_OV.png" log_f="$LOGD/${name}.log"
  for attempt in 1 2 3 4 5 6 7 8 9 10 11 12; do
    release_claim "$name"
    g=$(pick_gpu "$name" "$mypid") || { log "评测 $name 等待 GPU 超时 FAILED"; return 1; }
    log "评测 $name 尝试 $attempt GPU$g tex=$(sha256sum "$TEX" | cut -c1-8)"
    bash "$PI/run_cloth330.sh" "$g" eval --condition adv --suite "$suite" \
      --tasks 0,1,2,3,4,5,6,7,8,9 --trials 10 --num-denoise 10 --init-offset 10 \
      --max-steps 400 --texture "$TEX" --ckpt "$ckpt" \
      ${extra[@]+"${extra[@]}"} >> "$log_f" 2>&1
    res=$(grep -a '\[eval\] 结果' "$log_f" 2>/dev/null | tail -1 | sed 's/.*结果: //' | tr -d '\r')
    n=$(grep -ac '^Task:' "$res" 2>/dev/null)
    if [ -n "$res" ] && [ -f "$res" ] && [ "${n:-0}" -ge 100 ]; then
      release_claim "$name"
      log "评测 $name 完成 -> $(grep -a 'overall robot success' "$res" | tail -1)"
      return 0
    fi
    sleep 120
  done
  release_claim "$name"
  log "评测 $name 十二次失败 FAILED"; return 1
}

# 8 泳道，每泳道 5 格（E03 先行——纹理最先就绪）
pids=()
( run_cell e03_uv0_pi0_spatial  e03-uvoff it04999 libero_spatial "$CKPT0"
  run_cell e04a_pi0_spatial     e04a      it04999 libero_spatial "$CKPT0"
  run_cell e04b_pi0_spatial     e04b      it04999 libero_spatial "$CKPT0"
  run_cell e05_s8n_pi05_spatial e05_s8_native it04999 libero_spatial "$CKPT5" pi05_libero
  run_cell e05_s9n_pi05_spatial e05_s9_native it04999 libero_spatial "$CKPT5" pi05_libero ) & pids+=($!)
( run_cell e03_uv0_pi0_object   e03-uvoff it04999 libero_object  "$CKPT0"
  run_cell e04a_pi0_object      e04a      it04999 libero_object  "$CKPT0"
  run_cell e04b_pi0_object      e04b      it04999 libero_object  "$CKPT0"
  run_cell e05_s8n_pi05_object  e05_s8_native it04999 libero_object  "$CKPT5" pi05_libero
  run_cell e05_s9n_pi05_object  e05_s9_native it04999 libero_object  "$CKPT5" pi05_libero ) & pids+=($!)
( run_cell e03_uv0_pi0_goal     e03-uvoff it04999 libero_goal    "$CKPT0"
  run_cell e04a_pi0_goal        e04a      it04999 libero_goal    "$CKPT0"
  run_cell e04b_pi0_goal        e04b      it04999 libero_goal    "$CKPT0"
  run_cell e05_s8m_pi05_spatial e05_s8_mr it04999 libero_spatial "$CKPT5" pi05_libero
  run_cell e05_s9m_pi05_spatial e05_s9_mr it04999 libero_spatial "$CKPT5" pi05_libero ) & pids+=($!)
( run_cell e03_uv0_pi0_long     e03-uvoff it04999 libero_10      "$CKPT0"
  run_cell e04a_pi0_long        e04a      it04999 libero_10      "$CKPT0"
  run_cell e04b_pi0_long        e04b      it04999 libero_10      "$CKPT0"
  run_cell e05_s8m_pi05_object  e05_s8_mr it04999 libero_object  "$CKPT5" pi05_libero
  run_cell e05_s9m_pi05_object  e05_s9_mr it04999 libero_object  "$CKPT5" pi05_libero ) & pids+=($!)
( run_cell e03_uv0_pi05_spatial e03-uvoff it04999 libero_spatial "$CKPT5" pi05_libero
  run_cell e04a_pi05_spatial    e04a      it04999 libero_spatial "$CKPT5" pi05_libero
  run_cell e04b_pi05_spatial    e04b      it04999 libero_spatial "$CKPT5" pi05_libero
  run_cell e07_625n_pi05_spatial e07_625_native it00624 libero_spatial "$CKPT5" pi05_libero
  run_cell e07_bs1n_pi05_spatial e07_bs1_native it39999 libero_spatial "$CKPT5" pi05_libero ) & pids+=($!)
( run_cell e03_uv0_pi05_object  e03-uvoff it04999 libero_object  "$CKPT5" pi05_libero
  run_cell e04a_pi05_object     e04a      it04999 libero_object  "$CKPT5" pi05_libero
  run_cell e04b_pi05_object     e04b      it04999 libero_object  "$CKPT5" pi05_libero
  run_cell e07_625n_pi05_object e07_625_native it00624 libero_object  "$CKPT5" pi05_libero
  run_cell e07_bs1n_pi05_object e07_bs1_native it39999 libero_object  "$CKPT5" pi05_libero ) & pids+=($!)
( run_cell e03_uv0_pi05_goal    e03-uvoff it04999 libero_goal    "$CKPT5" pi05_libero
  run_cell e04a_pi05_goal       e04a      it04999 libero_goal    "$CKPT5" pi05_libero
  run_cell e04b_pi05_goal       e04b      it04999 libero_goal    "$CKPT5" pi05_libero
  run_cell e07_625m_pi05_spatial e07_625_mr it00624 libero_spatial "$CKPT5" pi05_libero
  run_cell e07_bs1m_pi05_spatial e07_bs1_mr it39999 libero_spatial "$CKPT5" pi05_libero ) & pids+=($!)
( run_cell e03_uv0_pi05_long    e03-uvoff it04999 libero_10      "$CKPT5" pi05_libero
  run_cell e04a_pi05_long       e04a      it04999 libero_10      "$CKPT5" pi05_libero
  run_cell e04b_pi05_long       e04b      it04999 libero_10      "$CKPT5" pi05_libero
  run_cell e07_625m_pi05_object e07_625_mr it00624 libero_object  "$CKPT5" pi05_libero
  run_cell e07_bs1m_pi05_object e07_bs1_mr it39999 libero_object  "$CKPT5" pi05_libero ) & pids+=($!)

fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
names="e03_uv0_pi0_spatial e03_uv0_pi0_object e03_uv0_pi0_goal e03_uv0_pi0_long e03_uv0_pi05_spatial e03_uv0_pi05_object e03_uv0_pi05_goal e03_uv0_pi05_long"
for t in e04a e04b; do for m in pi0_spatial pi0_object pi0_goal pi0_long pi05_spatial pi05_object pi05_goal pi05_long; do names="$names ${t}_${m}"; done; done
for t in e05_s8n e05_s8m e05_s9n e05_s9m e07_625n e07_625m e07_bs1n e07_bs1m; do
  names="$names ${t}_pi05_spatial ${t}_pi05_object"; done
ok=0; sum=""
for name in $names; do
  cell_done "$name" && ok=$((ok + 1))
done
if [ "$fail" = 0 ] && [ "$ok" = 40 ]; then
  echo "DONE-R18BC-EVAL 40/40 $(date)" >> "$LOGD/summary.txt"; log "DONE-R18BC-EVAL 40/40"
else
  echo "R18BC-EVAL-WITH-FAILURES ok=$ok/40 $(date)" >> "$LOGD/summary.txt"; log "R18BC-EVAL-WITH-FAILURES ok=$ok/40"
fi
