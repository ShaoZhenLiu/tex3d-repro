#!/bin/bash
# 41_r18_e01_n8.sh — 第18轮 E01：Native8(tvfix_bs8_native) 补六个模型×套件格（零训练）
#   六格 = π0×{spatial,object,goal,10} + π0.5×{goal,10}；每格 100 回合（tasks 0-9 显式 ×
#   trials 10、初态 10-19、num_denoise 10、max_steps 400、replan5、MuJoCo3.3.0）
#   协议对齐主表 MR/TVbs8 既有行：独立查询噪声（不设 CLOTH_QUERY_SEED）。
#   纹理冻结核验：sha256 必须等于 ee241502…（第18轮附录B Native8）。
#   GPU 纪律（2026-10-26 用户授权更新：除 GPU4 外全部可用）：候选 0/1/2/3/5/6/7、
#   每卡≤2 我方进程（flock 原子准入 + claim 需求求和）、VRAM 守卫 need≈17000MiB
#   （π 实测≈8.6G 加载 + EGL 渲染/推理稳态≈15-16.4G）、60s 错峰、重试×4、动态选卡抗租户波动。GPU4 永不使用。
set -u
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
D=/data/shaozhen.liu
PI=$D/code/tex3d/pi
EX=$D/code/tex3d/experiments/r18_e01e02
LOGD=$EX/e01_logs
CLAIMD=$EX/gpu_claims
LOCK=$EX/gpu_admit.lock
WD=$D/code/tex3d/experiments/pi0_cloth/watchdog_events.log
TEX=$D/code/tex3d/experiments/pi0_cloth/oft_cloth/train_20261021_153634_tvfix-bs8/Universal_Cloth_OV.png
CKPT0=$D/model/RLinf-Pi0-LIBERO-130-fullshot-SFT
CKPT5=$D/model/RLinf-Pi05-LIBERO-130-fullshot-SFT
mkdir -p "$LOGD" "$CLAIMD"
expected=ee241502c8d1523ae3fcb3b038304ff77c22772f5c0b4e56f7aa0a0bf71509ec
actual=$(sha256sum "$TEX" | cut -d' ' -f1)
[ "$actual" = "$expected" ] || { echo "E01 纹理 SHA 不符: $actual" >> "$WD"; exit 1; }
log() { echo "$(date '+%F %T') [e01] $*" | tee -a "$WD"; }

gpu_used_mi() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$1" 2>/dev/null | tr -d ' '; }
release_claim() { rm -f "$CLAIMD/$1".* 2>/dev/null; }
pick_gpu() {  # $1=泳道名 $2=需要空闲MiB → 成功输出 GPU id（claim 已登记）
  # 准入规则：该卡已有 claim 的需求之和 + 自身需求 ≤ 当前空闲，且 claim 数<2
  # claim 记泳道壳 pid（函数入口捕获 BASHPID；命令替换子壳的 BASHPID 会即刻退出）
  local name=$1 need=$2 lane_pid=$BASHPID tries=0 g used free cnt f pid ned sum_need out
  while [ $tries -lt 2160 ]; do
    out=$(
      (
        flock 9
        for g in 5 7 6 2 0 1 3; do
          sum_need=0; cnt=0
          for f in "$CLAIMD"/*."$g"; do
            [ -e "$f" ] || continue
            read -r pid ned < "$f" 2>/dev/null || { rm -f "$f"; continue; }
            if [ -z "$pid" ] || ! kill -0 "$pid" 2>/dev/null; then rm -f "$f"; continue; fi
            sum_need=$((sum_need + ${ned:-10000})); cnt=$((cnt + 1))
          done
          [ "$cnt" -ge 2 ] && continue
          used=$(gpu_used_mi "$g")
          [ -z "$used" ] && continue
          free=$((49140 - used))
          if [ $((sum_need + need)) -le "$free" ]; then
            echo "$lane_pid $need" > "$CLAIMD/${name}.$g"
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
run_cell() {  # $1=name $2=suite $3=ckpt $4=modelcfg(可空)
  local name=$1 suite=$2 ckpt=$3 mcfg=${4:-} res n g extra=()
  if cell_done "$name"; then log "评测 $name 已完成，跳过"; return 0; fi
  [ -n "$mcfg" ] && extra=(--model-config "$mcfg")
  local log_f="$LOGD/${name}.log"
  for attempt in 1 2 3 4 5 6 7 8 9 10 11 12; do
    release_claim "$name"
    g=$(pick_gpu "$name" 17000) || { log "评测 $name 等待 GPU 空闲超时 FAILED"; return 1; }
    log "评测 $name 尝试 $attempt GPU$g"
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
  log "评测 $name 四次失败 FAILED"; return 1
}

# 六泳道，60s 错峰；GPU 由动态准入分配
pids=()
( sleep 0;   run_cell e01_n8_pi0_spatial libero_spatial "$CKPT0" ) & pids+=($!)
( sleep 60;  run_cell e01_n8_pi0_goal    libero_goal    "$CKPT0" ) & pids+=($!)
( sleep 120; run_cell e01_n8_pi0_object  libero_object  "$CKPT0" ) & pids+=($!)
( sleep 180; run_cell e01_n8_pi0_long    libero_10      "$CKPT0" ) & pids+=($!)
( sleep 240; run_cell e01_n8_pi05_goal   libero_goal    "$CKPT5" pi05_libero ) & pids+=($!)
( sleep 300; run_cell e01_n8_pi05_long   libero_10      "$CKPT5" pi05_libero ) & pids+=($!)
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
ok=0; sum=""
for name in e01_n8_pi0_spatial e01_n8_pi0_goal e01_n8_pi0_object e01_n8_pi0_long e01_n8_pi05_goal e01_n8_pi05_long; do
  cell_done "$name" && ok=$((ok + 1))
  res=$(grep -a '\[eval\] 结果' "$LOGD/$name.log" 2>/dev/null | tail -1 | sed 's/.*结果: //' | tr -d '\r')
  sum="$sum$name=$(grep -a 'overall robot success' "$res" 2>/dev/null | tail -1 | grep -o '[0-9]*/100') "
done
if [ "$fail" = 0 ] && [ "$ok" = 6 ]; then
  echo "DONE-E01-N8 6/6 $(date)：$sum" >> "$LOGD/summary.txt"; log "DONE-E01-N8 6/6：$sum"
else
  echo "E01-N8-WITH-FAILURES ok=$ok/6 $(date)：$sum" >> "$LOGD/summary.txt"; log "E01-N8-WITH-FAILURES ok=$ok/6：$sum"
fi
