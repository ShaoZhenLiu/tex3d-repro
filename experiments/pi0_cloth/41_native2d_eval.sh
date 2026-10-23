#!/bin/bash
# 41_native2d_eval.sh — EDPA-native-2D 全量评测（第14轮P1续，用户指令：仅 GPU0-3）
#   16 格 = {patch(Adv), patch_random(P_method)} × {π0,π0.5} × 四套件，每格 100 回合
#   Original 复用 F0 O 行（无布同场景），不重跑；DONE-NATIVE2D-EVAL 收口。
#   GPU 纪律：仅 0/1/2/3，每卡 2 泳道，60s 错峰（用户规定）。
set -u
D=/data/shaozhen.liu
PI=$D/code/tex3d/pi
LOGD=$D/code/tex3d/experiments/edpa_native2d/eval_logs
mkdir -p "$LOGD"
CKPT0=$D/model/RLinf-Pi0-LIBERO-130-fullshot-SFT
CKPT5=$D/model/RLinf-Pi05-LIBERO-130-fullshot-SFT
PPRI=$D/code/tex3d/experiments/edpa_native2d/train_20261022_194828_cam-primary_r1/perturbation.pt
PWRI=$D/code/tex3d/experiments/edpa_native2d/train_20261022_195505_cam-wrist_r1/perturbation.pt
log() { echo "[n2eval $(date +%H:%M:%S)] $*"; }
cell_done() {
  local res n
  res=$(grep -a '\[eval\] 结果' "$LOGD/$1.log" 2>/dev/null | tail -1 | sed 's/.*结果: //' | tr -d '\r')
  n=$(grep -ac '^Task:' "$res" 2>/dev/null)
  [ -n "$res" ] && [ -f "$res" ] && [ "${n:-0}" -ge 100 ]
}
run_cell() {  # $1=gpu $2=name $3=suite $4=cond $5=ckpt $6=modelcfg
  local g=$1 name=$2 suite=$3 cond=$4 ckpt=$5 mcfg=${6:-}
  if cell_done "$name"; then log "评测 $name 已完成，跳过"; return 0; fi
  local log_f="$LOGD/${name}.log" res n extra=() pargs=()
  [ -n "$mcfg" ] && extra+=(--model-config "$mcfg")
  if [ "$cond" = adv ]; then pargs=(--condition patch --patch-primary "$PPRI" --patch-wrist "$PWRI")
  else pargs=(--condition patch_random); fi
  log "评测 $name GPU$g cond=$cond"
  for attempt in 1 2 3 4; do
    env CLOTH_QUERY_SEED=1 bash "$PI/run_patch_eval.sh" "$g" eval \
      --tasks 0,1,2,3,4,5,6,7,8,9 --trials 10 --num-denoise 10 --init-offset 10 \
      --suite "$suite" --ckpt "$ckpt" "${pargs[@]}" "${extra[@]}" >> "$log_f" 2>&1
    res=$(grep -a '\[eval\] 结果' "$log_f" 2>/dev/null | tail -1 | sed 's/.*结果: //' | tr -d '\r')
    n=$(grep -ac '^Task:' "$res" 2>/dev/null)
    if [ -n "$res" ] && [ -f "$res" ] && [ "${n:-0}" -ge 100 ]; then
      log "评测 $name 完成 -> $(grep -a 'overall robot success' "$res" | tail -1)"; return 0
    fi
    sleep 45
  done
  log "评测 $name 四次失败 FAILED"; return 1
}
# 8 泳道（GPU0-3 × 2），每泳道 2 格，60s 错峰
( sleep 0;   run_cell 0 nat2d_adv_pi0_sp  libero_spatial adv "$CKPT0"; run_cell 0 nat2d_adv_pi0_go  libero_goal adv "$CKPT0" ) & pids+=($!)
( sleep 60;  run_cell 0 nat2d_rnd_pi0_sp  libero_spatial rnd "$CKPT0"; run_cell 0 nat2d_rnd_pi0_go  libero_goal rnd "$CKPT0" ) & pids+=($!)
( sleep 120; run_cell 1 nat2d_adv_pi0_obj libero_object  adv "$CKPT0"; run_cell 1 nat2d_adv_pi0_10  libero_10  adv "$CKPT0" ) & pids+=($!)
( sleep 180; run_cell 1 nat2d_rnd_pi0_obj libero_object  rnd "$CKPT0"; run_cell 1 nat2d_rnd_pi0_10  libero_10  rnd "$CKPT0" ) & pids+=($!)
( sleep 240; run_cell 2 nat2d_adv_pi05_sp libero_spatial adv "$CKPT5" pi05_libero; run_cell 2 nat2d_adv_pi05_go libero_goal adv "$CKPT5" pi05_libero ) & pids+=($!)
( sleep 300; run_cell 2 nat2d_rnd_pi05_sp libero_spatial rnd "$CKPT5" pi05_libero; run_cell 2 nat2d_rnd_pi05_go libero_goal rnd "$CKPT5" pi05_libero ) & pids+=($!)
( sleep 360; run_cell 3 nat2d_adv_pi05_obj libero_object adv "$CKPT5" pi05_libero; run_cell 3 nat2d_adv_pi05_10 libero_10 adv "$CKPT5" pi05_libero ) & pids+=($!)
( sleep 420; run_cell 3 nat2d_rnd_pi05_obj libero_object rnd "$CKPT5" pi05_libero; run_cell 3 nat2d_rnd_pi05_10 libero_10 rnd "$CKPT5" pi05_libero ) & pids+=($!)
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
if [ "$fail" = 0 ]; then
  echo "DONE-NATIVE2D-EVAL 16/16 $(date)" >> "$LOGD/summary.txt"
  for f in "$LOGD"/*.log; do
    r=$(grep -a '\[eval\] 结果' "$f" | tail -1 | sed 's/.*结果: //')
    [ -f "$r" ] && echo "$(basename "$f" .log): $(grep -a 'overall robot success' "$r" | tail -1)"
  done | tee -a "$LOGD/summary.txt"
else
  echo "NATIVE2D-EVAL-HAD-FAILURES $(date)" >> "$LOGD/summary.txt"
fi
